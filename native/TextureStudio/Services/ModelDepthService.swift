import CoreImage
import CoreML
import CoreVideo
import Foundation

protocol ModelValidating: Sendable {
    func inspect(at url: URL) async throws -> ModelInterface
}

/// Serializes Core ML prediction and bounds ML input independently of export resolution.
/// The returned scalar values retain model output precision; no preview normalization occurs.
actor ModelDepthService: ModelValidating {
    private var predicting = false
    func inspect(at url: URL) async throws -> ModelInterface {
        let loaded = try load(at: url, computeUnits: .cpuOnly)
        defer { loaded.cleanup() }
        return try Self.describe(loaded.model)
    }

    func predict(image: CGImage, modelURL: URL, outputName: String? = nil) async throws -> ModelDepthResult {
        guard !predicting else { throw LocalModelError.busy }
        predicting = true
        defer { predicting = false }
        try Task.checkCancellation()
        let loaded = try load(at: modelURL, computeUnits: .all)
        defer { loaded.cleanup() }
        let interface = try Self.describe(loaded.model)
        guard let selected = outputName ?? interface.unambiguousOutput else {
            throw LocalModelError.chooseOutput(interface.outputs.map(\.name))
        }
        guard interface.outputs.contains(where: { $0.name == selected }) else {
            throw LocalModelError.unsupported("the selected output no longer exists")
        }
        let description = loaded.model.modelDescription.inputDescriptionsByName[interface.inputName]!
        let constraint = description.imageConstraint!
        let buffer = try Self.inputBuffer(image: image, constraint: constraint)
        let provider = try MLDictionaryFeatureProvider(dictionary: [interface.inputName: MLFeatureValue(pixelBuffer: buffer)])
        try Task.checkCancellation()
        let prediction = try Self.synchronousPrediction(model: loaded.model, provider: provider)
        try Task.checkCancellation()
        guard let value = prediction.featureValue(for: selected) else {
            throw LocalModelError.unsupported("the selected prediction is missing")
        }
        let decoded: (Int, Int, [Float])
        if let pixelBuffer = value.imageBufferValue {
            decoded = try Self.decode(pixelBuffer)
        } else if let array = value.multiArrayValue {
            decoded = try Self.decode(array)
        } else {
            throw LocalModelError.unsupported("depth must be a floating-point image or scalar tensor")
        }
        guard decoded.2.allSatisfy(\.isFinite) else {
            throw LocalModelError.unsupported("prediction contains non-finite values")
        }
        return ModelDepthResult(width: decoded.0, height: decoded.1, values: decoded.2,
            outputName: selected, interpretation: interface.interpretation)
    }

    private struct LoadedModel {
        let model: MLModel
        let temporaryURL: URL?
        func cleanup() { if let temporaryURL { try? FileManager.default.removeItem(at: temporaryURL) } }
    }

    // Keep the synchronous overload in a synchronous helper. The actor owns the
    // model for this operation; this also avoids sending MLModel to a concurrent
    // async overload and keeps inference strictly sequential.
    private static func synchronousPrediction(model: MLModel, provider: any MLFeatureProvider) throws -> any MLFeatureProvider {
        try model.prediction(from: provider)
    }

    private func load(at url: URL, computeUnits: MLComputeUnits) throws -> LoadedModel {
        guard FileManager.default.fileExists(atPath: url.path) else { throw LocalModelError.missingModel }
        let ext = url.pathExtension.lowercased()
        guard ["mlmodel", "mlpackage", "mlmodelc"].contains(ext) else {
            throw LocalModelError.unsupported("choose an .mlpackage, .mlmodel, or .mlmodelc")
        }
        let compiled = ext == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        do {
            return LoadedModel(model: try MLModel(contentsOf: compiled, configuration: config),
                temporaryURL: ext == "mlmodelc" ? nil : compiled)
        } catch {
            if ext != "mlmodelc" { try? FileManager.default.removeItem(at: compiled) }
            throw error
        }
    }

    private static func checkDimensions(_ width: Int, _ height: Int) throws {
        guard width > 0, height > 0, width <= 4096, height <= 4096, width * height <= 4_194_304 else {
            throw LocalModelError.unsupported("model image dimensions exceed the 4-megapixel inference budget")
        }
    }

    private static func describe(_ model: MLModel) throws -> ModelInterface {
        let required = model.modelDescription.inputDescriptionsByName.values.filter { !$0.isOptional }
        guard required.count == 1, let input = required.first, let image = input.imageConstraint else {
            throw LocalModelError.unsupported("it must have one required image input and no other required inputs")
        }
        guard image.pixelFormatType == kCVPixelFormatType_32BGRA else {
            throw LocalModelError.unsupported("the image input must accept standard RGB/BGRA pixels")
        }
        try checkDimensions(image.pixelsWide, image.pixelsHigh)
        let outputs: [ModelOutputDescriptor] = try model.modelDescription.outputDescriptionsByName.values
            .sorted { $0.name < $1.name }.compactMap { output in
                if let image = output.imageConstraint,
                   [kCVPixelFormatType_OneComponent16Half, kCVPixelFormatType_OneComponent32Float].contains(image.pixelFormatType) {
                    try checkDimensions(image.pixelsWide, image.pixelsHigh)
                    return ModelOutputDescriptor(name: output.name, width: image.pixelsWide, height: image.pixelsHigh,
                        storage: image.pixelFormatType == kCVPixelFormatType_OneComponent16Half ? "Float16 image" : "Float32 image")
                }
                if let array = output.multiArrayConstraint,
                   [MLMultiArrayDataType.float16, .float32, .double].contains(array.dataType) {
                    let shape = array.shape.map(\.intValue)
                    guard shape.count >= 2, shape.dropLast(2).allSatisfy({ $0 == 1 }) else { return nil }
                    let width = shape[shape.count - 1], height = shape[shape.count - 2]
                    try checkDimensions(width, height)
                    return ModelOutputDescriptor(name: output.name, width: width, height: height,
                        storage: "\(array.dataType == .float16 ? "Float16" : array.dataType == .float32 ? "Float32" : "Float64") tensor")
                }
                return nil
            }
        guard !outputs.isEmpty else {
            throw LocalModelError.unsupported("no floating-point grayscale depth output is available")
        }
        let metadata = model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]
        let knownName = metadata?["com.apple.developer.machine-learning.models.name"] ?? ""
        let interpretation = knownName.hasPrefix("DepthAnything")
            ? "Relative inverse depth: larger values are nearer. Native FP16 predictions; full source field of view resized to the model input. No metric scale."
            : "Raw scalar model output. Direction and units depend on your model; verify them before using displacement. Full source field of view resized to the model input."
        return ModelInterface(inputName: input.name, inputWidth: image.pixelsWide, inputHeight: image.pixelsHigh,
            outputs: outputs, interpretation: interpretation)
    }

    private static func inputBuffer(image: CGImage, constraint: MLImageConstraint) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true]
        guard CVPixelBufferCreate(kCFAllocatorDefault, constraint.pixelsWide, constraint.pixelsHigh,
            constraint.pixelFormatType, attributes as CFDictionary, &buffer) == kCVReturnSuccess, let buffer else {
            throw LocalModelError.unsupported("could not allocate the bounded model input")
        }
        let ciImage = CIImage(cgImage: image)
        let resized = ciImage.transformed(by: CGAffineTransform(scaleX: CGFloat(constraint.pixelsWide) / CGFloat(image.width),
            y: CGFloat(constraint.pixelsHigh) / CGFloat(image.height)))
        // The image model embeds its own normalization. Supply display RGB in sRGB,
        // with no external normalization or HDR tone mapping.
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(options: [.workingColorSpace: colorSpace, .outputColorSpace: colorSpace])
        context.render(resized, to: buffer, bounds: CGRect(x: 0, y: 0,
            width: constraint.pixelsWide, height: constraint.pixelsHigh), colorSpace: colorSpace)
        return buffer
    }

    static func decode(_ buffer: CVPixelBuffer) throws -> (Int, Int, [Float]) {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        try checkDimensions(width, height)
        let format = CVPixelBufferGetPixelFormatType(buffer)
        guard [kCVPixelFormatType_OneComponent16Half, kCVPixelFormatType_OneComponent32Float].contains(format),
              !CVPixelBufferIsPlanar(buffer) else {
            throw LocalModelError.unsupported("depth image output is not a scalar float pixel buffer")
        }
        guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
            throw LocalModelError.unsupported("could not read the model output buffer")
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw LocalModelError.unsupported("model output buffer is empty")
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var values = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = base.advanced(by: y * stride)
            for x in 0..<width {
                values[y * width + x] = format == kCVPixelFormatType_OneComponent16Half
                    ? Float(row.load(fromByteOffset: x * 2, as: Float16.self))
                    : row.load(fromByteOffset: x * 4, as: Float.self)
            }
        }
        return (width, height, values)
    }

    static func decode(_ array: MLMultiArray) throws -> (Int, Int, [Float]) {
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        guard shape.count >= 2, shape.dropLast(2).allSatisfy({ $0 == 1 }) else {
            throw LocalModelError.unsupported("depth tensor must have only one scalar channel")
        }
        let width = shape[shape.count - 1], height = shape[shape.count - 2]
        try checkDimensions(width, height)
        var values = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let index = y * strides[strides.count - 2] + x * strides[strides.count - 1]
                switch array.dataType {
                case .float16: values[y * width + x] = Float(array.dataPointer.load(fromByteOffset: index * 2, as: Float16.self))
                case .float32: values[y * width + x] = array.dataPointer.load(fromByteOffset: index * 4, as: Float.self)
                case .double: values[y * width + x] = Float(array.dataPointer.load(fromByteOffset: index * 8, as: Double.self))
                default: throw LocalModelError.unsupported("depth tensor is not floating-point")
                }
            }
        }
        return (width, height, values)
    }
}
