import CoreImage
import CoreML
import CoreVideo
import CryptoKit
import Foundation

/// Dedicated native adapter for the two-input DepthPro Core ML conversion.
/// One service owns one compiled model; cached training inputs reuse it.
final class NativeDepthProService: @unchecked Sendable {
    static let inputSize = 1536
    static let upstreamRepository = "coreml-projects/DepthPro-coreml"
    static let upstreamRevision = "e3b374e875fa7c89ca4c80203e1fb12e5e02bceb"
    static let pinnedPackageSHA256 = "e62c68496d25c76e099fbbcfb7a39814a201a1f2d9a6a9411e37feec2d5d94ec"
    let modelSHA256: String
    private let model: MLModel
    private let widthDataType: MLMultiArrayDataType
    private let temporaryCompiledURL: URL?
    private let lock = NSLock()
    private let context: CIContext

    init(modelURL: URL) throws {
        guard FileManager.default.fileExists(atPath: modelURL.path) else { throw LocalModelError.missingModel }
        guard ["mlmodel", "mlpackage", "mlmodelc"].contains(modelURL.pathExtension.lowercased()) else {
            throw LocalModelError.unsupported("choose the DepthPro .mlpackage, .mlmodel, or .mlmodelc")
        }
        modelSHA256 = try Self.modelIdentity(at: modelURL)
        let compiled = modelURL.pathExtension.lowercased() == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
        let temporary = compiled == modelURL ? nil : compiled
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            let loaded = try MLModel(contentsOf: compiled, configuration: configuration)
            widthDataType = try Self.validateInterface(loaded.modelDescription)
            model = loaded
            temporaryCompiledURL = temporary
            let space = CGColorSpace(name: CGColorSpace.sRGB)!
            context = CIContext(options: [.workingColorSpace: space, .outputColorSpace: space])
        } catch {
            if let temporary { try? FileManager.default.removeItem(at: temporary) }
            throw error
        }
    }

    deinit {
        if let temporaryCompiledURL { try? FileManager.default.removeItem(at: temporaryCompiledURL) }
    }

    func predict(image: CGImage) throws -> ModelDepthResult {
        lock.lock()
        defer { lock.unlock() }
        try Task.checkCancellation()
        let started = Date()
        let width = try Self.originalWidth(image.width, dataType: widthDataType)
        let pixels = try inputBuffer(image: image)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixels),
            "originalWidth": MLFeatureValue(multiArray: width)
        ])
        let result = try model.prediction(from: input)
        try Task.checkCancellation()
        guard let values = result.featureValue(for: "depthMeters")?.multiArrayValue else {
            throw LocalModelError.unsupported("DepthPro did not return its depthMeters tensor")
        }
        let decoded = try ModelDepthService.decode(values)
        guard decoded.0 == Self.inputSize, decoded.1 == Self.inputSize,
              decoded.2.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw LocalModelError.unsupported("DepthPro returned invalid or non-positive camera depth in meters")
        }
        let pinned = modelSHA256 == Self.pinnedPackageSHA256
        return ModelDepthResult(width: decoded.0, height: decoded.1, values: decoded.2,
            outputName: "depthMeters",
            interpretation: "Camera depth in meters: larger values are farther. Full source field of view resized to 1536×1536; raw floating-point values with no display normalization.",
            provenance: ModelDepthProvenance(modelID: pinned ? Self.upstreamRepository : "developer-depthpro-coreml",
                revision: pinned ? Self.upstreamRevision : modelSHA256,
                checkpointSHA256: modelSHA256, upstreamRevision: pinned ? Self.upstreamRevision : "", backend: "Core ML",
                device: "Core ML all compute units", precision: values.dataType == .float16 ? "Float16" : values.dataType == .double ? "Float64" : "Float32",
                inputWidth: image.width, inputHeight: image.height, processResolution: Self.inputSize,
                fullSourceFieldOfView: true, rowOrder: "top-to-bottom", elapsedSeconds: Date().timeIntervalSince(started)))
    }

    /// The width describes the source before the model's 1536-square resize.
    /// The published conversion requires Float16; respect its declared precision.
    static func originalWidth(_ width: Int, dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        guard width > 0, [.float16, .float32, .double].contains(dataType) else {
            throw LocalModelError.unsupported("DepthPro originalWidth must be a positive floating-point scalar")
        }
        if dataType == .float16, !Float16(width).isFinite {
            throw LocalModelError.unsupported("source width exceeds DepthPro originalWidth precision")
        }
        let array = try MLMultiArray(shape: [1, 1, 1, 1], dataType: dataType)
        array[0] = NSNumber(value: width)
        return array
    }

    static func validateInterface(_ description: MLModelDescription) throws -> MLMultiArrayDataType {
        let required = description.inputDescriptionsByName.filter { !$0.value.isOptional }
        guard Set(required.keys) == ["image", "originalWidth"],
              let image = required["image"]?.imageConstraint,
              image.pixelsWide == inputSize, image.pixelsHigh == inputSize,
              image.pixelFormatType == kCVPixelFormatType_32BGRA,
              let width = required["originalWidth"]?.multiArrayConstraint,
              width.shape.map(\.intValue) == [1, 1, 1, 1],
              [.float16, .float32, .double].contains(width.dataType),
              let depth = description.outputDescriptionsByName["depthMeters"]?.multiArrayConstraint,
              depth.shape.map(\.intValue) == [1, 1, inputSize, inputSize],
              [.float16, .float32, .double].contains(depth.dataType) else {
            throw LocalModelError.unsupported("DepthPro requires an RGB image at 1536×1536, originalWidth [1,1,1,1], and scalar depthMeters [1,1,1536,1536]")
        }
        return width.dataType
    }

    private func inputBuffer(image: CGImage) throws -> CVPixelBuffer {
        var optional: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true]
        guard CVPixelBufferCreate(kCFAllocatorDefault, Self.inputSize, Self.inputSize,
            kCVPixelFormatType_32BGRA, attributes as CFDictionary, &optional) == kCVReturnSuccess,
              let buffer = optional else {
            throw LocalModelError.unsupported("could not allocate the DepthPro RGB input")
        }
        let resized = CIImage(cgImage: image).transformed(by: CGAffineTransform(
            scaleX: CGFloat(Self.inputSize) / CGFloat(image.width),
            y: CGFloat(Self.inputSize) / CGFloat(image.height)))
        // Core ML embeds normalization. Preserve the full image without cropping,
        // padding, external normalization, or HDR tone mapping.
        context.render(resized, to: buffer, bounds: CGRect(x: 0, y: 0, width: Self.inputSize, height: Self.inputSize),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        return buffer
    }

    /// Relocation-independent identity of the specification and every weight.
    /// Read large weights in bounded chunks rather than loading a 1.9 GB package.
    static func modelIdentity(at url: URL) throws -> String {
        let keys: Set<URLResourceKey> = [.isDirectoryKey, .isRegularFileKey, .fileSizeKey]
        let root = url.standardizedFileURL
        let directory = try root.resourceValues(forKeys: keys).isDirectory == true
        let files: [URL]
        if directory {
            guard let enumerator = FileManager.default.enumerator(at: root,
                includingPropertiesForKeys: Array(keys)) else { throw LocalModelError.missingModel }
            var found: [URL] = []
            for case let file as URL in enumerator {
                if try file.resourceValues(forKeys: keys).isRegularFile == true { found.append(file) }
            }
            guard !found.isEmpty else { throw LocalModelError.unsupported("DepthPro package is empty") }
            files = found.sorted { $0.path < $1.path }
        } else { files = [root] }
        var identity = SHA256()
        identity.update(data: Data("texture-studio-depthpro-package-v1\n".utf8))
        for file in files {
            try Task.checkCancellation()
            let name = directory ? String(file.path.dropFirst(root.path.count + 1)) : "model.mlmodel"
            let count = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            let reader = try FileHandle(forReadingFrom: file)
            var digest = SHA256()
            do {
                while let chunk = try reader.read(upToCount: 1_048_576), !chunk.isEmpty {
                    try Task.checkCancellation()
                    digest.update(data: chunk)
                }
                try reader.close()
            } catch { try? reader.close(); throw error }
            let hex = digest.finalize().map { String(format: "%02x", $0) }.joined()
            identity.update(data: Data("\(name.utf8.count):\(name):\(count):\(hex)\n".utf8))
        }
        return identity.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
