import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeMaterialModelTests: XCTestCase {
    func testPinnedLayoutMatchesPublishedCheckpointTensorDescriptors() {
        // Digest independently recorded from the pinned archive's data.pkl
        // metadata, fetched with bounded range requests without weight data.
        // All 2892 consumed tensors match; the four auxiliary heads are unused.
        let fixture = NativeMaterialModelFixture(architecture: .pinned, shapesOnly: true)
        let record = fixture.weights.keys.sorted().map { name in
            let tensor = fixture.weights[name]!
            return name + "|" + tensor.dtype + "|" + tensor.shape.map(String.init).joined(separator: ",")
        }.joined(separator: "\n")
        XCTAssertEqual(fixture.weights.count, 2892)
        XCTAssertEqual(NativeMaterialTrainer.checksum(Data(record.utf8)), "b0761102648c4bde2debd3b6ae7828a18f6e5e2605289af5e31c51866bdb20d9")
    }
    func testCompleteNativeGraphPreservesRectangleAndPredictsAllMapShapes() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
        XCTAssertFalse(Thread.isMainThread)
        let fixture = NativeMaterialModelFixture()
        let model = try fixture.model()
        let rgb = (0..<3 * 64 * 128).map { Float($0 % 100) / 100 }
        for target in ["height", "normal", "roughness"] {
            let result = try model.predict(rgb: rgb, width: 128, height: 64, target: target)
            XCTAssertEqual(result.width, 128); XCTAssertEqual(result.height, 64)
            XCTAssertEqual(result.values.count, 128 * 64 * (target == "normal" ? 3 : 1))
            XCTAssertTrue(result.values.allSatisfy(\.isFinite))
        }
        }.value
    }
    func testLoRAGradientAdamUpdateAndLossAreRealOnFullOperationSequence() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
        XCTAssertFalse(Thread.isMainThread)
        let fixture = NativeMaterialModelFixture()
        let model = try fixture.model(adapter: true)
        let rgb = (0..<3 * 64 * 64).map { Float($0 % 100) / 100 }
        let program = try model.program(width: 64, height: 64, target: "height")
        let before = try program.execute(rgb: rgb, adapters: model.adapterWeights)
        let reference = before.output.map { $0 + 0.01 }
        let update = try program.execute(rgb: rgb, adapters: model.adapterWeights, reference: reference, learningRate: 1e-3)
        XCTAssertEqual(update.valueLoss!, 0.01, accuracy: 2e-6)
        XCTAssertGreaterThan(update.loss!, 0)
        XCTAssertFalse(update.updated.isEmpty)
        XCTAssertEqual(update.optimizerState.count, update.updated.count * 2)
        let name = "ups.3.model.10.lora_B"
        XCTAssertNotEqual(update.updated[name]!.bytes, model.adapterWeights[name]!.bytes)
        let changed = try program.execute(rgb: rgb, adapters: update.updated)
        XCTAssertLessThan(zip(changed.output, reference).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / Float(reference.count), update.valueLoss!)
        XCTAssertEqual(try model.fusedWeights()["ups.3.model.10.weight"]!.bytes, model.baseWeights["ups.3.model.10.weight"]!.bytes)
        model.updateAdapters(update.updated)
        let fused = try NativeMaterialModel(baseWeights: model.fusedWeights(), baseSHA256: "fixture", architecture: .test)
        let merged = try fused.predict(rgb: rgb, width: 64, height: 64, target: "height")
        for (first, second) in zip(changed.output, merged.values) { XCTAssertEqual(first, second, accuracy: 2e-5) }
        }.value
    }
    func testRejectsAlteredPixelGridAndMissingLearnedLayer() throws {
        let fixture = NativeMaterialModelFixture()
        let model = try fixture.model()
        XCTAssertThrowsError(try model.predict(rgb: [Float](repeating: 0, count: 65 * 64 * 3), width: 65, height: 64, target: "height"))
        var weights = fixture.weights
        weights.removeValue(forKey: "gen.m_head.weight")
        let incomplete = try NativeMaterialModel(baseWeights: weights, baseSHA256: "fixture", architecture: .test)
        XCTAssertThrowsError(try incomplete.predict(rgb: [Float](repeating: 0, count: 64 * 64 * 3), width: 64, height: 64, target: "height"))
        XCTAssertThrowsError(try model.program(width: 0, height: 64, target: "height"))
        var invalid = fixture.weights
        let name = "gen.m_body.0.trans_block.msa.relative_position_index"
        let indices = [Int64](repeating: 225, count: 4096)
        invalid[name] = .init(dtype: "I64", shape: [64, 64], bytes: indices.withUnsafeBytes { Data($0) })
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: invalid, baseSHA256: "fixture", architecture: .test))
    }
    func testMalformedAdapterDimensionsFailBeforeGraphCompilation() throws {
        let name = "ups.3.model.10", base = [name + ".weight": NativeTensor.floats([1, 2], shape: [1, 2])]
        let layers = [name: NativeMaterialModel.AdapterLayer(weightShape: [1, 2], rank: 1, alpha: 1)]
        let factors = [name + ".lora_A": NativeTensor.floats([1, 2], shape: [1, 2]), name + ".lora_B": NativeTensor.floats([0], shape: [1, 1])]
        XCTAssertNoThrow(try NativeMaterialModel(baseWeights: base, adapterWeights: factors, layers: layers, baseSHA256: "fixture"))
        // Equal element counts with a different matrix layout are incompatible.
        let transposed = [name: NativeMaterialModel.AdapterLayer(weightShape: [2, 1], rank: 1, alpha: 1)]
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: factors, layers: transposed, baseSHA256: "fixture"))
        var missing = factors; missing.removeValue(forKey: name + ".lora_B")
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: missing, layers: layers, baseSHA256: "fixture"))
        var extra = factors; extra["unexpected.lora_A"] = .floats([1], shape: [1])
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: extra, layers: layers, baseSHA256: "fixture"))
        var wrongType = factors; wrongType[name + ".lora_B"] = .init(dtype: "I32", shape: [1, 1], bytes: Data(repeating: 0, count: 4))
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: wrongType, layers: layers, baseSHA256: "fixture"))
        for rank in [0, 4097, Int.max] {
            let invalid = [name: NativeMaterialModel.AdapterLayer(weightShape: [1, 2], rank: rank, alpha: 1)]
            XCTAssertThrowsError(try NativeMaterialModel(baseWeights: base, adapterWeights: factors, layers: invalid, baseSHA256: "fixture"))
        }
        let overflow = [name + ".weight": NativeTensor(dtype: "F32", shape: [Int.max, 2], bytes: Data())]
        XCTAssertThrowsError(try NativeMaterialModel(baseWeights: overflow, baseSHA256: "fixture"))
    }
}

/// Every PBRnxt operation is present with smaller channel widths; fixture
/// tensors are synthesized directly in Swift, without a Python runtime.
struct NativeMaterialModelFixture {
    var weights: [String: NativeTensor] = [:]
    init(architecture a: NativeMaterialModel.Architecture = .test, shapesOnly: Bool = false) {
        let d = a.dim
        func add(_ name: String, _ shape: [Int], _ value: Float) {
            weights[name] = shapesOnly ? NativeTensor(dtype: "F32", shape: shape, bytes: Data()) : .floats([Float](repeating: value, count: shape.reduce(1, *)), shape: shape)
        }
        func conv(_ name: String, _ incoming: Int, _ outgoing: Int, kernel: Int = 3, bias: Bool = true) {
            add(name + ".weight", [outgoing, incoming, kernel, kernel], 0.05 / Float(incoming * kernel * kernel))
            if bias { add(name + ".bias", [outgoing], 0.01) }
        }
        func linear(_ name: String, _ incoming: Int, _ outgoing: Int, bias: Bool = true) {
            add(name + ".weight", [outgoing, incoming], 0.05 / Float(incoming))
            if bias { add(name + ".bias", [outgoing], 0.01) }
        }
        func norm(_ name: String, _ dim: Int) { add(name + ".weight", [dim], 1); add(name + ".bias", [dim], 0) }
        func block(_ prefix: String, _ channels: Int) {
            let half = channels / 2
            conv(prefix + ".conv1_1", channels, channels, kernel: 1)
            conv(prefix + ".conv1_2", channels, channels, kernel: 1)
            conv(prefix + ".conv_block.dwconv", 1, half, kernel: 7)
            norm(prefix + ".conv_block.norm", half)
            linear(prefix + ".conv_block.pwconv1", half, half * 4)
            add(prefix + ".conv_block.grn.gamma", [1, 1, 1, half * 4], 0)
            add(prefix + ".conv_block.grn.beta", [1, 1, 1, half * 4], 0)
            linear(prefix + ".conv_block.pwconv2", half * 4, half)
            add(prefix + ".conv_block.gamma", [half], 1e-6)
            let p = prefix + ".trans_block"
            norm(p + ".ln1", half); norm(p + ".ln2", half)
            linear(p + ".mlp.0", half, half * 4); linear(p + ".mlp.2", half * 4, half)
            linear(p + ".msa.embedding_layer", half, half * 3, bias: false)
            add(p + ".msa.q_bias", [half], 0); add(p + ".msa.v_bias", [half], 0)
            add(p + ".msa.logit_scale", [a.heads, 1, 1], log(10))
            add(p + ".msa.relative_coords_table", [1, 15, 15, 2], 0)
            weights[p + ".msa.relative_position_index"] = NativeTensor(dtype: "I64", shape: [64, 64], bytes: [Int64](repeating: 0, count: 4096).withUnsafeBytes { Data($0) })
            linear(p + ".msa.cpb_mlp.0", 2, 512); linear(p + ".msa.cpb_mlp.2", 512, a.heads, bias: false)
            linear(p + ".msa.linear", half, half)
        }
        conv("gen.m_head", 3, d, bias: false)
        for level in 1...3 {
            let channels = d << (level - 1), prefix = "gen.m_enc.m_down\(level)"
            for i in 0..<a.encoderBlocks { block(prefix + ".\(i)", channels) }
            conv(prefix + ".\(a.encoderBlocks)", channels, channels * 2, kernel: 2, bias: false)
        }
        for i in 0..<a.encoderBlocks { block("gen.m_body.\(i)", d * 8) }
        for branch in 0..<4 {
            for level in (1...3).reversed() {
                let channels = d << level, prefix = "gen.m_dec_\(branch).m_up\(level)"
                conv(prefix + ".0.up.1", channels, channels, bias: false)
                conv(prefix + ".0.up.3", channels, channels / 2, bias: false)
                for i in 0..<a.decoderBlocks { block(prefix + ".\(i + 1)", channels / 2) }
            }
            conv("gen.m_tail_\(branch).0", d, [3, 3, 1, 1][branch], bias: false)
        }
        conv("gen.m_fuse.0", d * 4, d)
        for i in 0..<a.fusionBlocks { block("gen.m_fuse.\(i + 1)", d) }
        conv("gen.m_fuse.\(a.fusionBlocks + 1)", d, d * 4)
        for branch in 0..<4 {
            let p = "ups.\(branch).model", width = a.rrdbWidth, growth = a.growth
            conv(p + ".0", 11, width)
            for i in 0..<a.rrdbBlocks { for r in 1...3 {
                let q = p + ".1.sub.\(i).RDB\(r)"
                conv(q + ".conv1x1", width, growth, kernel: 1, bias: false)
                for index in 1...4 { conv(q + ".conv\(index).0", width + (index - 1) * growth, growth) }
                conv(q + ".conv5.0", width + 4 * growth, width)
            } }
            conv(p + ".1.sub.\(a.rrdbBlocks)", width, width)
            for index in [3, 6, 8] { conv(p + ".\(index)", width, width) }
            conv(p + ".10", width, [3, 3, 1, 1][branch])
        }
    }
    func model(adapter: Bool = false) throws -> NativeMaterialModel {
        let name = "ups.3.model.10", shape = weights[name + ".weight"]!.shape
        let specs: [String: NativeMaterialModel.AdapterLayer] = adapter ? [name: .init(weightShape: shape, rank: 1, alpha: 1)] : [:]
        let adapters: [String: NativeTensor] = adapter ? [name + ".lora_A": .floats([Float](repeating: 0.1, count: shape.dropFirst().reduce(1, *)), shape: [1, shape.dropFirst().reduce(1, *)]), name + ".lora_B": .floats([0], shape: [1, 1])] : [:]
        return try NativeMaterialModel(baseWeights: weights, adapterWeights: adapters, layers: specs, baseSHA256: "fixture", architecture: .test)
    }
}
