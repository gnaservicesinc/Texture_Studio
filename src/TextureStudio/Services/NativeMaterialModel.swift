import Accelerate
import CryptoKit
import Foundation
import Metal
import MetalPerformanceShadersGraph

struct NativeMaterialPrediction: Sendable {
    let width: Int, height: Int, channels: Int
    /// Planar NCHW, the direct Float32 network output. No clipping/stretching.
    let values: [Float]
}

/// The pinned PBRnxt SCUNetV2 + RRDB mapping, expressed directly as an Apple
/// MPSGraph. The graph retains every generator decoder and the chosen complete
/// output branch; only the previously declared final 4x enlargement is omitted.
final class NativeMaterialModel: @unchecked Sendable {
    static let pinnedSHA256 = "3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d"
    static let pinnedFilename = "pbrnxt_402236.pth"
    struct AdapterLayer: Codable, Sendable {
        let weightShape: [Int], rank: Int
        let alpha: Float
        enum CodingKeys: String, CodingKey { case weightShape = "weight_shape"; case rank, alpha }
    }
    struct Architecture: Sendable {
        let dim: Int, heads: Int, encoderBlocks: Int, decoderBlocks: Int, fusionBlocks: Int, rrdbBlocks: Int, rrdbWidth: Int, growth: Int
        static let pinned = Architecture(dim: 96, heads: 1, encoderBlocks: 2, decoderBlocks: 2, fusionBlocks: 4, rrdbBlocks: 12, rrdbWidth: 32, growth: 32)
        /// Used by numeric tests of the complete operation sequence.
        static let test = Architecture(dim: 4, heads: 1, encoderBlocks: 2, decoderBlocks: 2, fusionBlocks: 2, rrdbBlocks: 1, rrdbWidth: 2, growth: 2)
    }
    let baseWeights: [String: NativeTensor]
    private(set) var adapterWeights: [String: NativeTensor]
    let layers: [String: AdapterLayer]
    var configuration: [String: Any]
    let baseSHA256: String
    let architecture: Architecture
    private var cached: Program?

    init(baseWeights: [String: NativeTensor], adapterWeights: [String: NativeTensor] = [:],
         layers: [String: AdapterLayer] = [:], configuration: [String: Any] = [:], baseSHA256: String,
         architecture: Architecture = .pinned) throws {
        guard !baseWeights.isEmpty else { throw StudioError("The material model has no learned tensors.") }
        for (name, tensor) in baseWeights {
            var byteCount = tensor.dtype == "I64" ? 8 : 4
            for dimension in tensor.shape {
                let product = byteCount.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !product.overflow else { throw StudioError("Material tensor dimensions overflow: \(name)") }
                byteCount = product.partialValue
            }
            guard ["F32", "I64", "I32"].contains(tensor.dtype), !tensor.shape.isEmpty,
                  tensor.bytes.count == byteCount else {
                throw StudioError("Material tensor precision or dimensions are invalid: \(name)")
            }
            if name.hasSuffix(".relative_position_index") {
                guard tensor.dtype == "I64", tensor.shape == [64, 64] else { throw StudioError("The material attention position index has invalid storage.") }
                try tensor.bytes.withUnsafeBytes { bytes in
                    for offset in stride(from: 0, to: bytes.count, by: 8) {
                        let index = bytes.loadUnaligned(fromByteOffset: offset, as: Int64.self)
                        guard (0..<225).contains(index) else { throw StudioError("The material attention position index exceeds its learned table.") }
                    }
                }
            }
        }
        let expectedFactors = Set(layers.keys.flatMap { [$0 + ".lora_A", $0 + ".lora_B"] })
        guard Set(adapterWeights.keys) == expectedFactors else { throw StudioError("Material adapter factors differ from their recorded layers.") }
        for (name, layer) in layers {
            guard let weight = baseWeights[name + ".weight"], weight.dtype == "F32",
                  [2, 4].contains(weight.shape.count), weight.shape == layer.weightShape,
                  !name.contains(".dwconv."), layer.rank > 0, layer.rank <= 4096,
                  layer.alpha.isFinite, layer.alpha > 0 else {
                throw StudioError("Material adapter layer differs from its exact base weight: \(name)")
            }
            let incoming = weight.shape.dropFirst().reduce(1, *)
            let aCount = layer.rank.multipliedReportingOverflow(by: incoming)
            let bCount = layer.rank.multipliedReportingOverflow(by: weight.shape[0])
            let aBytes = aCount.partialValue.multipliedReportingOverflow(by: 4)
            let bBytes = bCount.partialValue.multipliedReportingOverflow(by: 4)
            guard !aCount.overflow, !bCount.overflow, !aBytes.overflow, !bBytes.overflow,
                  let a = adapterWeights[name + ".lora_A"], let b = adapterWeights[name + ".lora_B"],
                  a.dtype == "F32", b.dtype == "F32", a.shape == [layer.rank, incoming],
                  b.shape == [weight.shape[0], layer.rank], a.bytes.count == aBytes.partialValue,
                  b.bytes.count == bBytes.partialValue else {
                throw StudioError("Material adapter factors have invalid native dimensions: \(name)")
            }
        }
        self.baseWeights = baseWeights; self.adapterWeights = adapterWeights
        self.layers = layers; self.configuration = configuration; self.baseSHA256 = baseSHA256
        self.architecture = architecture
    }

    static func load(checkpointURL: URL?, expectedSHA256: String? = nil, baseURL: URL,
                     target: String, scope: String = "final-map", rank: Int = 8, alpha: Float = 8,
                     training: Bool = false, seed: UInt64 = 17) throws -> NativeMaterialModel {
        var configuration: [String: Any] = [:], adapters: [String: NativeTensor] = [:]
        var specs: [String: AdapterLayer] = [:], base: [String: NativeTensor], digest: String
        if let checkpointURL {
            _ = try NativeMaterialCheckpoint.inspect(at: checkpointURL, expectedSHA256: expectedSHA256)
            let checkpoint = try NativeSafetensors(contentsOf: resolvedCheckpoint(checkpointURL), expectedSHA256: expectedSHA256)
            configuration = try JSONSerialization.jsonObject(with: Data(checkpoint.metadata["configuration"]!.utf8)) as! [String: Any]
            guard configuration["target"] as? String == target,
                  !training || configuration["schema"] as? String != "texture-studio-material-lora-v1" || configuration["scope"] as? String == scope else {
                throw StudioError("The selected checkpoint target or trained layer scope differs from this operation.")
            }
            if configuration["schema"] as? String == "texture-studio-material-checkpoint-v1" {
                base = try checkpoint.nativeTensors(); digest = checkpoint.sha256
                configuration["base"] = ["sha256": digest, "architecture": "pbrnxt-native-v1", "name": "Texture Studio custom material base"]
            } else {
                adapters = try checkpoint.nativeTensors()
                let specData = try JSONSerialization.data(withJSONObject: configuration["layers"]!)
                specs = try JSONDecoder().decode([String: AdapterLayer].self, from: specData)
                let loaded = try readWeights(baseURL); base = loaded.0; digest = loaded.1
                guard (configuration["base"] as? [String: Any])?["sha256"] as? String == digest else {
                    throw StudioError("The material adapter requires its exact recorded base weights.")
                }
            }
        } else {
            let loaded = try readWeights(baseURL); base = loaded.0; digest = loaded.1
            configuration = ["base": ["sha256": digest, "architecture": "pbrnxt-native-v1", "name": "PBRnxt material mapping"], "step": 0]
        }
        if training && specs.isEmpty {
            guard rank > 0, rank <= 4096, alpha.isFinite, alpha > 0, ["final-map", "map-decoder"].contains(scope),
                  let branch = ["normal": 1, "roughness": 2, "height": 3][target] else { throw StudioError("Invalid material adapter parameters.") }
            let prefixes = ["ups.\(branch)."] + (scope == "map-decoder" ? ["gen.m_dec_\(branch).", "gen.m_tail_\(branch)."] : [])
            var random = NativeMaterialRandom(seed: seed)
            for key in base.keys.sorted() where key.hasSuffix(".weight") && prefixes.contains(where: { key.hasPrefix($0) }) && !key.contains(".dwconv.") {
                let weight = base[key]!
                guard [2, 4].contains(weight.shape.count), weight.dtype == "F32" else { continue }
                let layer = String(key.dropLast(7))
                var incoming = 1
                for dimension in weight.shape.dropFirst() {
                    let count = incoming.multipliedReportingOverflow(by: dimension)
                    guard dimension > 0, !count.overflow else { throw StudioError("Material adapter dimensions overflow: \(layer)") }
                    incoming = count.partialValue
                }
                let aCount = rank.multipliedReportingOverflow(by: incoming)
                let bCount = rank.multipliedReportingOverflow(by: weight.shape[0])
                guard !aCount.overflow, !bCount.overflow,
                      aCount.partialValue <= Int.max / 4, bCount.partialValue <= Int.max / 4 else {
                    throw StudioError("Material adapter allocation dimensions overflow: \(layer)")
                }
                let bound = 1 / sqrt(Float(incoming))
                let a = (0..<aCount.partialValue).map { _ in (random.unit() * 2 - 1) * bound }
                specs[layer] = AdapterLayer(weightShape: weight.shape, rank: rank, alpha: alpha)
                adapters[layer + ".lora_A"] = .floats(a, shape: [rank, incoming])
                adapters[layer + ".lora_B"] = .floats([Float](repeating: 0, count: bCount.partialValue), shape: [weight.shape[0], rank])
            }
            guard !specs.isEmpty else { throw StudioError("The recorded model has no trained layers for this target.") }
        }
        configuration["architecture"] = "pbrnxt-native-v1"
        configuration["target"] = target; configuration["scope"] = scope
        configuration["schema"] = specs.isEmpty ? "texture-studio-material-checkpoint-v1" : "texture-studio-material-lora-v1"
        return try NativeMaterialModel(baseWeights: base, adapterWeights: adapters, layers: specs, configuration: configuration, baseSHA256: digest)
    }

    static func resolvedCheckpoint(_ url: URL) -> URL {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
            let full = url.appendingPathComponent("model.safetensors")
            return FileManager.default.fileExists(atPath: full.path) ? full : url.appendingPathComponent("adapter.safetensors")
        }
        return url
    }
    private static func readWeights(_ url: URL) throws -> ([String: NativeTensor], String) {
        if url.pathExtension == "safetensors" {
            let snapshot = try NativeSafetensors(contentsOf: url)
            try snapshot.validateFiniteFloatingPoint()
            return (try snapshot.nativeTensors(), snapshot.sha256)
        }
        let loaded = try NativeTorchCheckpoint.load(url: url, expectedSHA256: pinnedSHA256)
        return (loaded.tensors, loaded.sha256)
    }
    func predict(rgb: [Float], width: Int, height: Int, target: String) throws -> NativeMaterialPrediction {
        try validateInput(rgb, width: width, height: height)
        let program = try program(width: width, height: height, target: target)
        let result = try program.execute(rgb: rgb, adapters: adapterWeights)
        return NativeMaterialPrediction(width: width, height: height, channels: target == "normal" ? 3 : 1, values: result.output)
    }
    func program(width: Int, height: Int, target: String) throws -> Program {
        if let cached, cached.width == width, cached.height == height, cached.target == target { return cached }
        let next = try Program(model: self, width: width, height: height, target: target)
        cached = next; return next
    }
    func updateAdapters(_ adapters: [String: NativeTensor]) { self.adapterWeights = adapters }
    func validateInput(_ rgb: [Float], width: Int, height: Int) throws {
        guard width >= 64, height >= 64, width % 64 == 0, height % 64 == 0,
              rgb.count == width * height * 3, rgb.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw StudioError("Material inference needs finite native RGB on dimensions divisible by 64, without resizing or padding.")
        }
        try Task.checkCancellation()
    }
    func checkpointConfiguration(size: Int, step: Int, validation: [String: Any]? = nil) throws -> [String: Any] {
        var record = configuration
        record["step"] = step; record["training_size"] = size
        record["image_padding"] = false; record["image_resizing"] = false
        record["input_transfer"] = "sRGB diffuse codes / code maximum"
        record["target_transfer"] = "linear numeric source codes / code maximum"
        record["native_runtime"] = "Apple MPSGraph Float32, reduced precision fast math disabled"
        record["layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(layers))
        record["trained_utc"] = ISO8601DateFormatter().string(from: Date())
        if let validation { record["validation"] = validation }
        return record
    }
    func fusedWeights() throws -> [String: NativeTensor] {
        var fused = baseWeights
        for (name, specification) in layers {
            try Task.checkCancellation()
            guard let base = baseWeights[name + ".weight"], let a = adapterWeights[name + ".lora_A"],
                  let b = adapterWeights[name + ".lora_B"] else { throw StudioError("Incomplete material adapter state.") }
            let graph = MPSGraph()
            let wa = graph.constant(a.bytes, shape: a.shape.ns, dataType: .float32)
            let wb = graph.constant(b.bytes, shape: b.shape.ns, dataType: .float32)
            let delta = graph.multiplication(graph.matrixMultiplication(primary: wb, secondary: wa, name: nil),
                graph.constant(Double(specification.alpha / Float(specification.rank)), dataType: .float32), name: nil)
            let weight = graph.constant(base.bytes, shape: base.shape.ns, dataType: .float32)
            let sum = graph.addition(weight, graph.reshape(delta, shape: base.shape.ns, name: nil), name: nil)
            let values = try NativeGraphExecution.run(graph, feeds: [:], targets: [sum])[0]
            fused[name + ".weight"] = .floats(values, shape: base.shape)
        }
        return fused
    }

    final class Program {
        let graph = MPSGraph()
        let width: Int, height: Int, target: String
        let input: MPSGraphTensor
        private(set) var output: MPSGraphTensor!
        private(set) var parameterFeeds: [String: MPSGraphTensor] = [:]
        private(set) var parameters: [String: NativeTensor] = [:]
        private let baseWeights: [String: NativeTensor]
        private let layers: [String: AdapterLayer]
        private let architecture: Architecture
        private let freezesForTraining: Bool
        private var optimizationProgram: Program?
        private var constants: [String: MPSGraphTensor] = [:]
        private(set) var targetInput: MPSGraphTensor?
        private(set) var loss: MPSGraphTensor?
        private(set) var valueLoss: MPSGraphTensor?
        private(set) var gradientLoss: MPSGraphTensor?
        private var gradients: [String: MPSGraphTensor] = [:]
        private var optimizerFeeds: [String: MPSGraphTensor] = [:]
        private var optimizerOutputs: [String: MPSGraphTensor] = [:]
        private var learningRate: MPSGraphTensor?
        private var optimizerStep: MPSGraphTensor?
        private var executableCache: [String: MPSGraphExecutable] = [:]
        private var frozenStages: [[(source: MPSGraphTensor, feed: MPSGraphTensor)]] = []
        var frozenStageCount: Int { frozenStages.count + (optimizationProgram?.frozenStageCount ?? 0) }
        var optimizerPrepared: Bool { learningRate != nil || optimizationProgram?.optimizerPrepared == true }
        convenience init(model: NativeMaterialModel, width: Int, height: Int, target: String) throws {
            try self.init(baseWeights: model.baseWeights, layers: model.layers, architecture: model.architecture,
                adapters: model.adapterWeights, width: width, height: height, target: target, freezesForTraining: false)
        }
        private init(baseWeights: [String: NativeTensor], layers: [String: AdapterLayer], architecture: Architecture,
                     adapters: [String: NativeTensor], width: Int, height: Int, target: String, freezesForTraining: Bool) throws {
            guard ["height", "roughness", "normal"].contains(target), width >= 64, height >= 64,
                  width % 64 == 0, height % 64 == 0 else { throw StudioError("Unsupported native material grid or target.") }
            // Swift dictionaries/Data share immutable storage. The lazy
            // optimizer graph neither copies the base bytes nor retains the
            // owning model, which would form a cycle with its program cache.
            self.baseWeights = baseWeights; self.layers = layers; self.architecture = architecture
            self.freezesForTraining = freezesForTraining
            self.width = width; self.height = height; self.target = target
            input = graph.placeholder(shape: [1, 3, height, width].ns, dataType: .float32, name: "native_rgb")
            for (name, tensor) in adapters {
                parameters[name] = tensor
                parameterFeeds[name] = graph.placeholder(shape: tensor.shape.ns, dataType: .float32, name: name)
            }
            output = try build(input)
            guard output.shape?.map(\.intValue) == [1, target == "normal" ? 3 : 1, height, width] else {
                throw StudioError("The material graph changed native output dimensions.")
            }
        }
        private func shape(_ tensor: MPSGraphTensor) -> [Int] { tensor.shape!.map(\.intValue) }
        private func c(_ value: Double) -> MPSGraphTensor { graph.constant(value, dataType: .float32) }
        private func add(_ lhs: MPSGraphTensor, _ rhs: MPSGraphTensor) -> MPSGraphTensor { graph.addition(lhs, rhs, name: nil) }
        private func mul(_ lhs: MPSGraphTensor, _ rhs: MPSGraphTensor) -> MPSGraphTensor { graph.multiplication(lhs, rhs, name: nil) }
        private func sub(_ lhs: MPSGraphTensor, _ rhs: MPSGraphTensor) -> MPSGraphTensor { graph.subtraction(lhs, rhs, name: nil) }
        private func div(_ lhs: MPSGraphTensor, _ rhs: MPSGraphTensor) -> MPSGraphTensor { graph.division(lhs, rhs, name: nil) }
        private func reshape(_ tensor: MPSGraphTensor, _ shape: [Int]) -> MPSGraphTensor { graph.reshape(tensor, shape: shape.ns, name: nil) }
        private func permute(_ tensor: MPSGraphTensor, _ axes: [Int]) -> MPSGraphTensor { graph.transpose(tensor, permutation: axes.ns, name: nil) }
        private func slice(_ tensor: MPSGraphTensor, _ axis: Int, _ start: Int, _ length: Int) -> MPSGraphTensor { graph.sliceTensor(tensor, dimension: axis, start: start, length: length, name: nil) }
        private func cat(_ tensors: [MPSGraphTensor], _ axis: Int) -> MPSGraphTensor { graph.concatTensors(tensors, dimension: axis, name: nil) }
        private func tensor(_ name: String, shape required: [Int]? = nil) throws -> MPSGraphTensor {
            if let value = constants[name] { return value }
            guard let weight = baseWeights[name], required == nil || weight.shape == required else { throw StudioError("Material model tensor missing or shape differs: \(name)") }
            let dtype: MPSDataType = weight.dtype == "I64" ? .int64 : weight.dtype == "I32" ? .int32 : .float32
            var value = graph.constant(weight.bytes, shape: weight.shape.ns, dataType: dtype)
            if name.hasSuffix(".weight"), let spec = layers[String(name.dropLast(7))] {
                let layer = String(name.dropLast(7))
                guard let a = parameterFeeds[layer + ".lora_A"], let b = parameterFeeds[layer + ".lora_B"] else { throw StudioError("Missing material LoRA factors.") }
                value = add(value, reshape(mul(graph.matrixMultiplication(primary: b, secondary: a, name: nil), c(Double(spec.alpha / Float(spec.rank)))), weight.shape))
            }
            constants[name] = value; return value
        }
        private func conv(_ value: MPSGraphTensor, _ name: String, stride: Int = 1, padding: Int? = nil, groups: Int = 1) throws -> MPSGraphTensor {
            guard let weight = baseWeights[name + ".weight"], weight.dtype == "F32", weight.shape.count == 4,
                  shape(value)[1] == weight.shape[1] * groups else { throw StudioError("Invalid learned material convolution: \(name)") }
            let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: stride, strideInY: stride, dilationRateInX: 1, dilationRateInY: 1,
                groups: groups, paddingLeft: padding ?? weight.shape[3] / 2, paddingRight: padding ?? weight.shape[3] / 2,
                paddingTop: padding ?? weight.shape[2] / 2, paddingBottom: padding ?? weight.shape[2] / 2,
                paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
            var result = graph.convolution2D(value, weights: try tensor(name + ".weight"), descriptor: descriptor, name: name)
            if let bias = baseWeights[name + ".bias"] {
                guard bias.shape == [weight.shape[0]], bias.dtype == "F32" else { throw StudioError("Invalid material convolution bias.") }
                result = add(result, reshape(try tensor(name + ".bias"), [1, weight.shape[0], 1, 1]))
            }
            return result
        }
        private func linear(_ value: MPSGraphTensor, _ name: String, explicitBias: MPSGraphTensor? = nil) throws -> MPSGraphTensor {
            guard let weight = baseWeights[name + ".weight"], weight.dtype == "F32", weight.shape.count == 2,
                  shape(value).last == weight.shape[1] else { throw StudioError("Invalid learned material linear layer: \(name)") }
            var result = graph.matrixMultiplication(primary: value, secondary: permute(try tensor(name + ".weight"), [1, 0]), name: name)
            if let explicitBias { result = add(result, explicitBias) }
            else if baseWeights[name + ".bias"] != nil { result = add(result, try tensor(name + ".bias", shape: [weight.shape[0]])) }
            return result
        }
        private func norm(_ value: MPSGraphTensor, _ name: String, epsilon: Double) throws -> MPSGraphTensor {
            let axis = shape(value).count - 1, channels = shape(value).last!
            let mean = graph.mean(of: value, axes: [NSNumber(value: axis)], name: nil)
            let centered = sub(value, mean)
            let variance = graph.mean(of: mul(centered, centered), axes: [NSNumber(value: axis)], name: nil)
            return add(mul(div(centered, graph.squareRoot(with: add(variance, c(epsilon)), name: nil)), try tensor(name + ".weight", shape: [channels])), try tensor(name + ".bias", shape: [channels]))
        }
        private func gelu(_ value: MPSGraphTensor) -> MPSGraphTensor {
            mul(mul(value, c(0.5)), add(c(1), graph.erf(with: mul(value, c(1 / sqrt(2))), name: nil)))
        }
        private func leaky(_ value: MPSGraphTensor) -> MPSGraphTensor { graph.leakyReLU(with: value, alpha: 0.2, name: nil) }
        private func convNeXt(_ value: MPSGraphTensor, _ prefix: String) throws -> MPSGraphTensor {
            let channels = shape(value)[1]
            var x = permute(try conv(value, prefix + ".dwconv", groups: channels), [0, 2, 3, 1])
            x = try norm(x, prefix + ".norm", epsilon: 1e-6)
            x = gelu(try linear(x, prefix + ".pwconv1"))
            let energy = graph.squareRoot(with: graph.reductionSum(with: mul(x, x), axes: [1, 2], name: nil), name: nil)
            let normalized = div(energy, add(graph.mean(of: energy, axes: [3], name: nil), c(1e-6)))
            x = add(add(mul(try tensor(prefix + ".grn.gamma", shape: [1, 1, 1, channels * 4]), mul(x, normalized)), try tensor(prefix + ".grn.beta", shape: [1, 1, 1, channels * 4])), x)
            x = try linear(x, prefix + ".pwconv2")
            x = mul(x, try tensor(prefix + ".gamma", shape: [channels]))
            return add(value, permute(x, [0, 3, 1, 2]))
        }
        private func roll(_ value: MPSGraphTensor, axis: Int, shift: Int) -> MPSGraphTensor {
            let length = shape(value)[axis], offset = ((shift % length) + length) % length
            if offset == 0 { return value }
            return cat([slice(value, axis, length - offset, offset), slice(value, axis, 0, length - offset)], axis)
        }
        private func windows(_ value: MPSGraphTensor) -> MPSGraphTensor {
            let s = shape(value)
            return reshape(permute(reshape(value, [s[0], s[1] / 8, 8, s[2] / 8, 8, s[3]]), [0, 1, 3, 2, 4, 5]), [-1, 64, s[3]])
        }
        private func reverseWindows(_ value: MPSGraphTensor, height: Int, width: Int) -> MPSGraphTensor {
            let channels = shape(value).last!
            return reshape(permute(reshape(value, [1, height / 8, width / 8, 8, 8, channels]), [0, 1, 3, 2, 4, 5]), [1, height, width, channels])
        }
        private func swin(_ value: MPSGraphTensor, _ prefix: String, shifted: Bool) throws -> MPSGraphTensor {
            let s = shape(value), h = s[1], w = s[2], channels = s[3], heads = architecture.heads
            var x = shifted ? roll(roll(value, axis: 1, shift: -4), axis: 2, shift: -4) : value
            x = windows(x)
            let bias = cat([try tensor(prefix + ".msa.q_bias", shape: [channels]), graph.constant(0, shape: [channels].ns, dataType: .float32), try tensor(prefix + ".msa.v_bias", shape: [channels])], 0)
            let qkv = permute(reshape(try linear(x, prefix + ".msa.embedding_layer", explicitBias: bias), [-1, 64, 3, heads, channels / heads]), [2, 0, 3, 1, 4])
            let q = reshape(slice(qkv, 0, 0, 1), [-1, heads, 64, channels / heads])
            let k = reshape(slice(qkv, 0, 1, 1), [-1, heads, 64, channels / heads])
            let v = reshape(slice(qkv, 0, 2, 1), [-1, heads, 64, channels / heads])
            func unit(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
                let magnitude = graph.squareRoot(with: graph.reductionSum(with: mul(tensor, tensor), axes: [3], name: nil), name: nil)
                return div(tensor, graph.maximum(magnitude, c(1e-12), name: nil))
            }
            var attention = graph.matrixMultiplication(primary: unit(q), secondary: permute(unit(k), [0, 1, 3, 2]), name: nil)
            let scale = graph.exponent(with: graph.minimum(try tensor(prefix + ".msa.logit_scale", shape: [heads, 1, 1]), c(4.605170185988092), name: nil), name: nil)
            attention = mul(attention, scale)
            let coordinates = try tensor(prefix + ".msa.relative_coords_table", shape: [1, 15, 15, 2])
            let table = try linear(graph.reLU(with: try linear(coordinates, prefix + ".msa.cpb_mlp.0"), name: nil), prefix + ".msa.cpb_mlp.2")
            let indices = reshape(try tensor(prefix + ".msa.relative_position_index", shape: [64, 64]), [4096])
            let biasTable = graph.gather(withUpdatesTensor: reshape(table, [225, heads]), indicesTensor: indices, axis: 0, batchDimensions: 0, name: nil)
            let relative = mul(c(16), graph.sigmoid(with: permute(reshape(biasTable, [64, 64, heads]), [2, 0, 1]), name: nil))
            attention = add(attention, relative)
            if shifted {
                // Region IDs are 1 scalar per token, avoiding a large CPU mask.
                var labels = [Float](repeating: 0, count: h * w)
                for y in 0..<h { for x in 0..<w {
                    let ry = y < h - 8 ? 0 : y < h - 4 ? 1 : 2
                    let rx = x < w - 8 ? 0 : x < w - 4 ? 1 : 2
                    labels[y * w + x] = Float(ry * 3 + rx)
                } }
                let maskTokens = reshape(windows(graph.constant(labels.withUnsafeBytes { Data($0) }, shape: [1, h, w, 1].ns, dataType: .float32)), [-1, 64])
                let difference = sub(reshape(maskTokens, [-1, 64, 1]), reshape(maskTokens, [-1, 1, 64]))
                let mask = graph.select(predicate: graph.notEqual(difference, c(0), name: nil), trueTensor: c(-100), falseTensor: c(0), name: nil)
                attention = add(attention, reshape(mask, [-1, 1, 64, 64]))
            }
            attention = graph.softMax(with: attention, axis: -1, name: nil)
            x = reshape(permute(graph.matrixMultiplication(primary: attention, secondary: v, name: nil), [0, 2, 1, 3]), [-1, 64, channels])
            x = try linear(x, prefix + ".msa.linear")
            x = reverseWindows(x, height: h, width: w)
            if shifted { x = roll(roll(x, axis: 1, shift: 4), axis: 2, shift: 4) }
            x = add(value, try norm(x, prefix + ".ln1", epsilon: 1e-5))
            let mlp = try linear(gelu(try linear(x, prefix + ".mlp.0")), prefix + ".mlp.2")
            return add(x, try norm(mlp, prefix + ".ln2", epsilon: 1e-5))
        }
        private func block(_ value: MPSGraphTensor, _ prefix: String, shifted: Bool) throws -> MPSGraphTensor {
            try Task.checkCancellation()
            let mixed = try conv(value, prefix + ".conv1_1", padding: 0), half = shape(value)[1] / 2
            let convolution = try convNeXt(slice(mixed, 1, 0, half), prefix + ".conv_block")
            // Preserve upstream transpose(1,3), including the exchanged H/W.
            let transformer = try swin(permute(slice(mixed, 1, half, half), [0, 3, 2, 1]), prefix + ".trans_block", shifted: shifted)
            return add(value, try conv(cat([convolution, permute(transformer, [0, 3, 2, 1])], 1), prefix + ".conv1_2", padding: 0))
        }
        private func up2(_ value: MPSGraphTensor) -> MPSGraphTensor {
            NativeGraphExecution.nearestNeighbor2(value, graph: graph)
        }
        private func decode(_ body: MPSGraphTensor, _ skips: [MPSGraphTensor], _ branch: Int) throws -> MPSGraphTensor {
            var x = body
            for level in (1...3).reversed() {
                let prefix = "gen.m_dec_\(branch).m_up\(level)"
                x = add(x, skips[level - 1])
                x = leaky(try conv(up2(x), prefix + ".0.up.1"))
                x = leaky(try conv(x, prefix + ".0.up.3"))
                for i in 0..<architecture.decoderBlocks {
                    x = try block(x, prefix + ".\(i + 1)", shifted: i % 2 == 1)
                }
            }
            return x
        }
        private func rdb(_ value: MPSGraphTensor, _ prefix: String) throws -> MPSGraphTensor {
            let x1 = leaky(try conv(value, prefix + ".conv1.0"))
            let x2 = add(leaky(try conv(cat([value, x1], 1), prefix + ".conv2.0")), try conv(value, prefix + ".conv1x1", padding: 0))
            let x3 = leaky(try conv(cat([value, x1, x2], 1), prefix + ".conv3.0"))
            let x4 = add(leaky(try conv(cat([value, x1, x2, x3], 1), prefix + ".conv4.0")), x2)
            let x5 = try conv(cat([value, x1, x2, x3, x4], 1), prefix + ".conv5.0")
            return add(value, mul(x5, c(0.2)))
        }
        private func build(_ value: MPSGraphTensor) throws -> MPSGraphTensor {
            let specification = architecture
            let branch = ["normal": 1, "roughness": 2, "height": 3][target]!
            let adaptingGenerator = layers.keys.contains { $0.hasPrefix("gen.") }
            var initial = try conv(value, "gen.m_head")
            var x = initial, skips: [MPSGraphTensor] = []
            for level in 1...3 {
                let prefix = "gen.m_enc.m_down\(level)"
                for i in 0..<specification.encoderBlocks { x = try block(x, prefix + ".\(i)", shifted: i % 2 == 1) }
                x = try conv(x, prefix + ".\(specification.encoderBlocks)", stride: 2, padding: 0)
                skips.append(x)
            }
            for i in 0..<specification.encoderBlocks { x = try block(x, "gen.m_body.\(i)", shifted: i % 2 == 1) }
            // Frozen features cross a GPU tensor boundary. MPSGraph has no
            // public stopGradient API; a new placeholder disconnects only the
            // weights known to contain no adapters, without CPU readback.
            let encoderFrozen = freezesForTraining && adaptingGenerator && !layers.keys.contains {
                $0.hasPrefix("gen.m_head") || $0.hasPrefix("gen.m_enc.") || $0.hasPrefix("gen.m_body.")
            }
            if encoderFrozen {
                let features = freeze([initial] + skips + [x], name: "frozen_encoder")
                initial = features[0]; skips = Array(features[1...3]); x = features[4]
            }
            var decoded = try (0..<4).map { try decode(x, skips, $0) }
            if encoderFrozen {
                let frozen = (0..<4).filter { index in
                    !layers.keys.contains { $0.hasPrefix("gen.m_dec_\(index).") }
                }
                if !frozen.isEmpty {
                    let features = freeze(frozen.map { decoded[$0] }, name: "frozen_decoders")
                    for (offset, index) in frozen.enumerated() { decoded[index] = features[offset] }
                }
            }
            let concatenated = cat(decoded, 1)
            x = try conv(concatenated, "gen.m_fuse.0")
            for i in 0..<specification.fusionBlocks { x = try block(x, "gen.m_fuse.\(i + 1)", shifted: i % 2 == 1) }
            x = add(try conv(x, "gen.m_fuse.\(specification.fusionBlocks + 1)"), concatenated)
            let generator = try (0..<4).map { branch in
                try conv(add(slice(x, 1, branch * specification.dim, specification.dim), initial), "gen.m_tail_\(branch).0")
            }
            var generatorFeatures = cat(generator, 1)
            if freezesForTraining && !adaptingGenerator && !layers.isEmpty {
                generatorFeatures = freeze([generatorFeatures], name: "frozen_generator")[0]
            }
            x = cat([value, generatorFeatures], 1)
            let prefix = "ups.\(branch).model"
            x = try conv(x, prefix + ".0")
            let residual = x
            for i in 0..<specification.rrdbBlocks {
                let input = x
                for r in 1...3 { x = try rdb(x, prefix + ".1.sub.\(i).RDB\(r)") }
                x = add(input, mul(x, c(0.2)))
            }
            x = add(residual, try conv(x, prefix + ".1.sub.\(specification.rrdbBlocks)"))
            // .2 and .5 are the two original nearest2 upsamplers, omitted.
            for index in [3, 6, 8] { x = leaky(try conv(x, prefix + ".\(index)")) }
            return try conv(x, prefix + ".10")
        }
        private func freeze(_ sources: [MPSGraphTensor], name: String) -> [MPSGraphTensor] {
            let stage = sources.enumerated().map { index, source in
                (source: source, feed: graph.placeholder(shape: source.shape, dataType: .float32, name: "\(name)_\(index)"))
            }
            frozenStages.append(stage)
            return stage.map(\.feed)
        }
        struct Execution { let output: [Float]; let loss: Float?, valueLoss: Float?, gradientLoss: Float?; let updated: [String: NativeTensor]; let optimizerState: [String: NativeTensor] }
        func execute(rgb: [Float], adapters: [String: NativeTensor], reference: [Float]? = nil,
                     learningRate: Float? = nil, step: Int = 1, optimizerState: [String: NativeTensor] = [:]) throws -> Execution {
            if reference != nil, learningRate != nil, !freezesForTraining {
                if optimizationProgram == nil {
                    optimizationProgram = try Program(baseWeights: baseWeights, layers: layers, architecture: architecture,
                        adapters: parameters, width: width, height: height, target: target, freezesForTraining: true)
                }
                return try optimizationProgram!.execute(rgb: rgb, adapters: adapters, reference: reference,
                    learningRate: learningRate, step: step, optimizerState: optimizerState)
            }
            var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
            feeds[input] = try NativeGraphExecution.tensorData(.floats(rgb, shape: [1, 3, height, width]))
            for (name, placeholder) in parameterFeeds { feeds[placeholder] = try NativeGraphExecution.tensorData(adapters[name]!) }
            var targets = [output!]
            if let reference {
                try prepareLoss()
                guard reference.count == width * height * (target == "normal" ? 3 : 1), reference.allSatisfy(\.isFinite) else { throw StudioError("Native training target grid or values are invalid.") }
                feeds[targetInput!] = try NativeGraphExecution.tensorData(.floats(reference, shape: shape(output)))
                targets += [loss!, valueLoss!, gradientLoss!]
            }
            let optimize = reference != nil && learningRate != nil
            var ordered: [String] = []
            if optimize {
                try prepareOptimizer()
                guard let learningRate, learningRate.isFinite, learningRate > 0 else { throw StudioError("Invalid learning rate.") }
                feeds[self.learningRate!] = try NativeGraphExecution.tensorData(.floats([learningRate], shape: []))
                feeds[optimizerStep!] = try NativeGraphExecution.tensorData(.floats([Float(step)], shape: []))
                for (key, placeholder) in optimizerFeeds {
                    let parameterName = String(key.dropLast(2))
                    let value = optimizerState[key] ?? .floats([Float](repeating: 0, count: adapters[parameterName]!.shape.reduce(1, *)), shape: adapters[parameterName]!.shape)
                    feeds[placeholder] = try NativeGraphExecution.tensorData(value)
                }
                ordered = optimizerOutputs.keys.sorted()
                targets += ordered.map { optimizerOutputs[$0]! }
            }
            for stage in frozenStages {
                let data = try NativeGraphExecution.runData(graph, feeds: feeds, targets: stage.map(\.source), cache: &executableCache)
                for (index, feature) in stage.enumerated() { feeds[feature.feed] = data[index] }
            }
            let result = try NativeGraphExecution.run(graph, feeds: feeds, targets: targets, cache: &executableCache)
            guard result[0].allSatisfy(\.isFinite), reference == nil || result[1][0].isFinite else { throw StudioError("Material model produced nonfinite values; weights remain untouched.") }
            var updated: [String: NativeTensor] = [:], state: [String: NativeTensor] = [:]
            if optimize {
                for (i, key) in ordered.enumerated() {
                    let values = result[i + 4]
                    guard values.allSatisfy(\.isFinite) else { throw StudioError("Native material gradients or optimizer state are nonfinite; stopped before applying weights.") }
                    if key.hasSuffix(".m") || key.hasSuffix(".v") { state[key] = .floats(values, shape: adapters[String(key.dropLast(2))]!.shape) }
                    else { updated[key] = .floats(values, shape: adapters[key]!.shape) }
                }
            }
            return Execution(output: result[0], loss: reference == nil ? nil : result[1][0], valueLoss: reference == nil ? nil : result[2][0], gradientLoss: reference == nil ? nil : result[3][0], updated: updated, optimizerState: state)
        }
        private func prepareLoss() throws {
            if targetInput != nil { return }
            let reference = graph.placeholder(shape: output.shape, dataType: .float32, name: "native_linear_target")
            targetInput = reference
            let axes = [0, 1, 2, 3].ns
            let absolute = graph.mean(of: graph.absolute(with: sub(output, reference), name: nil), axes: axes, name: nil)
            var detail = c(0)
            for step in [1, 2, 4, 8] { for axis in [2, 3] {
                let length = shape(output)[axis] - step
                let pd = sub(slice(output, axis, step, length), slice(output, axis, 0, length))
                let td = sub(slice(reference, axis, step, length), slice(reference, axis, 0, length))
                let term = graph.mean(of: graph.absolute(with: sub(pd, td), name: nil), axes: axes, name: nil)
                detail = add(detail, div(term, c(Double(step))))
            } }
            valueLoss = absolute; gradientLoss = detail; loss = add(absolute, mul(detail, c(4)))
        }
        private func prepareOptimizer() throws {
            if optimizerPrepared { return }
            guard !parameterFeeds.isEmpty else { throw StudioError("Native training needs recorded material LoRA layers.") }
            try prepareLoss()
            let allParameters = parameterFeeds.keys.sorted()
            let derivative = graph.gradients(of: loss!, with: allParameters.map { parameterFeeds[$0]! }, name: "material_lora_gradients")
            var normSquared = c(0)
            for key in allParameters {
                guard let gradient = derivative[parameterFeeds[key]!] else { throw StudioError("Material graph could not differentiate adapter \(key).") }
                gradients[key] = gradient
                normSquared = add(normSquared, graph.reductionSum(with: mul(gradient, gradient), axes: Array(0..<shape(gradient).count).ns, name: nil))
            }
            let clip = graph.minimum(c(1), div(c(1), add(graph.squareRoot(with: normSquared, name: nil), c(1e-6))), name: nil)
            learningRate = graph.placeholder(shape: [], dataType: .float32, name: "learning_rate")
            optimizerStep = graph.placeholder(shape: [], dataType: .float32, name: "optimizer_step")
            let b1 = c(0.9), b2 = c(0.999)
            let correction1 = sub(c(1), graph.power(b1, optimizerStep!, name: nil))
            let correction2 = sub(c(1), graph.power(b2, optimizerStep!, name: nil))
            for key in allParameters {
                let parameter = parameterFeeds[key]!, parameterShape = parameter.shape!
                let m = graph.placeholder(shape: parameterShape, dataType: .float32, name: key + ".m")
                let v = graph.placeholder(shape: parameterShape, dataType: .float32, name: key + ".v")
                optimizerFeeds[key + ".m"] = m; optimizerFeeds[key + ".v"] = v
                let gradient = mul(gradients[key]!, clip)
                let nextM = add(mul(m, b1), mul(gradient, c(0.1)))
                let nextV = add(mul(v, b2), mul(mul(gradient, gradient), c(0.001)))
                let numerator = div(nextM, correction1)
                let denominator = add(div(graph.squareRoot(with: nextV, name: nil), graph.squareRoot(with: correction2, name: nil)), c(1e-8))
                optimizerOutputs[key] = sub(parameter, mul(learningRate!, div(numerator, denominator)))
                optimizerOutputs[key + ".m"] = nextM; optimizerOutputs[key + ".v"] = nextV
            }
        }
    }
}

enum NativeGraphExecution {
    static func nearestNeighbor2(_ value: MPSGraphTensor, graph: MPSGraph) -> MPSGraphTensor {
        let shape = value.shape!.map(\.intValue)
        let expanded = graph.reshape(value, shape: [shape[0], shape[1], shape[2], 1, shape[3], 1].ns, name: nil)
        // MPSGraphTileOp has no autodiff implementation. Concatenating the
        // singleton axes repeats the exact same nearest-neighbor samples,
        // while its derivative sums both copies back into the source.
        let rows = graph.concatTensors([expanded, expanded], dimension: 3, name: nil)
        let samples = graph.concatTensors([rows, rows], dimension: 5, name: nil)
        return graph.reshape(samples, shape: [shape[0], shape[1], shape[2] * 2, shape[3] * 2].ns, name: nil)
    }
    static func tensorData(_ value: NativeTensor) throws -> MPSGraphTensorData {
        guard let device = MTLCreateSystemDefaultDevice() else { throw StudioError("Metal is unavailable for native material inference.") }
        return MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: value.bytes, shape: value.shape.ns, dataType: .float32)
    }
    static func run(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor]) throws -> [[Float]] {
        var cache: [String: MPSGraphExecutable] = [:]
        return try run(graph, feeds: feeds, targets: targets, cache: &cache)
    }
    static func run(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor], cache: inout [String: MPSGraphExecutable]) throws -> [[Float]] {
        let result = try runData(graph, feeds: feeds, targets: targets, cache: &cache)
        return result.map { value in
            let count = value.shape.reduce(1) { $0 * $1.intValue }
            var values = [Float](repeating: 0, count: count)
            values.withUnsafeMutableBytes { value.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
            return values
        }
    }
    /// Retains frozen intermediate features on the GPU between graph stages.
    static func runData(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor], cache: inout [String: MPSGraphExecutable]) throws -> [MPSGraphTensorData] {
        try Task.checkCancellation()
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw StudioError("Metal is unavailable for native material computation.") }
        let key = targets.map { String(describing: ObjectIdentifier($0)) }.joined(separator: "|")
        let executable: MPSGraphExecutable
        if let existing = cache[key] { executable = existing }
        else {
            let descriptor = MPSGraphCompilationDescriptor()
            descriptor.reducedPrecisionFastMath = .none
            let shaped = Dictionary(uniqueKeysWithValues: feeds.map { ($0.key, MPSGraphShapedType(shape: $0.value.shape, dataType: $0.value.dataType)) })
            executable = graph.compile(with: MPSGraphDevice(mtlDevice: device), feeds: shaped, targetTensors: targets, targetOperations: nil, compilationDescriptor: descriptor)
            cache[key] = executable
        }
        let executionInputs = executable.feedTensors!.map { feeds[$0]! }
        let result = executable.run(with: queue, inputs: executionInputs, results: nil, executionDescriptor: nil)
        try Task.checkCancellation()
        guard let compiledTargets = executable.targetTensors, compiledTargets.count == result.count else {
            throw StudioError("Native material graph returned incomplete result identities.")
        }
        var byTensor: [MPSGraphTensor: MPSGraphTensorData] = [:]
        for (index, tensor) in compiledTargets.enumerated() { byTensor[tensor] = result[index] }
        return try targets.map { tensor in
            guard let value = byTensor[tensor] else { throw StudioError("Native material graph omitted a requested result.") }
            return value
        }
    }
}

struct NativeMaterialRandom {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state; value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
    mutating func shuffle<T>(_ values: inout [T]) {
        if values.count < 2 { return }
        for index in stride(from: values.count - 1, through: 1, by: -1) { values.swapAt(index, Int(next() % UInt64(index + 1))) }
    }
}

extension NativeTensor {
    static func floats(_ values: [Float], shape: [Int]) -> NativeTensor {
        NativeTensor(dtype: "F32", shape: shape, bytes: values.withUnsafeBytes { Data($0) })
    }
    func floatValues() throws -> [Float] {
        guard dtype == "F32", bytes.count % 4 == 0 else { throw StudioError("Material computation requires exact Float32 samples.") }
        return bytes.withUnsafeBytes { raw in (0..<bytes.count / 4).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) } }
    }
}
private extension Array where Element == Int { var ns: [NSNumber] { map { NSNumber(value: $0) } } }
