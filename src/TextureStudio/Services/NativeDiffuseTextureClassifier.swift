import CoreML
import CoreImage
import CryptoKit
import Foundation
import ImageIO
import Vision

/// The classifier's image adapter is independent of material training crops.
/// Vision prepares an in-memory image for the model's declared image input;
/// source pixels and reconstruction training files are never rewritten.
final class NativeDiffuseTextureClassifier: @unchecked Sendable {
    static let bundledModelSHA256 = "e2936fcef0781fb4f85eb7e879cc2f7d865528415180d894afd74eddc1e048be"
    static let bundledResourceName = "DiffuseTextureTypeClassifier"

    struct Prediction: Codable, Sendable {
        let label: String
        let confidence: Double
        let probabilities: [String: Double]
        let modelSHA256: String
    }

    let modelURL: URL
    let modelSHA256: String
    let labels: [String]
    let inputName: String
    let inputWidth: Int
    let inputHeight: Int
    let predictedFeatureName: String
    let predictedProbabilitiesName: String
    let preprocessing: String
    private let visionModel: VNCoreMLModel
    private let cropAndScale: VNImageCropAndScaleOption
    private let temporaryCompiledURL: URL?
    private let lock = NSLock()
    private let colorContext = CIContext(options: [.workingFormat: CIFormat.RGBAf])

    init(modelURL requested: URL? = nil) throws {
        let source: URL
        if let requested { source = requested.standardizedFileURL }
        else if let bundled = Bundle.main.url(forResource: Self.bundledResourceName, withExtension: "mlmodelc") {
            source = bundled
        } else {
            throw StudioError("The bundled diffuse classifier is missing. Choose an exported Core ML image classifier.")
        }
        guard FileManager.default.fileExists(atPath: source.path),
              ["mlmodel", "mlmodelc", "mlpackage"].contains(source.pathExtension.lowercased()) else {
            throw StudioError("Choose an existing Core ML .mlmodel, .mlpackage or compiled .mlmodelc image classifier.")
        }
        modelURL = source
        modelSHA256 = try requested == nil ? Self.bundledModelSHA256 : Self.modelIdentity(at: source)
        let compiled = source.pathExtension.lowercased() == "mlmodelc" ? source : try MLModel.compileModel(at: source)
        let temporary = compiled == source ? nil : compiled
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            let model = try MLModel(contentsOf: compiled, configuration: configuration)
            let description = model.modelDescription
            let inputs = description.inputDescriptionsByName.filter { !$0.value.isOptional }
            guard inputs.count == 1, let input = inputs.first, let image = input.value.imageConstraint,
                  image.pixelsWide > 0, image.pixelsHigh > 0,
                  let classes = description.classLabels as? [String], !classes.isEmpty,
                  Set(classes).count == classes.count, classes.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
                  let predicted = description.predictedFeatureName,
                  description.outputDescriptionsByName[predicted]?.type == .string,
                  let probabilities = description.predictedProbabilitiesName,
                  description.outputDescriptionsByName[probabilities]?.type == .dictionary else {
                throw StudioError("The selected Core ML model needs one image input, string class labels and class probabilities.")
            }
            labels = classes
            inputName = input.key
            inputWidth = image.pixelsWide; inputHeight = image.pixelsHigh
            predictedFeatureName = predicted; predictedProbabilitiesName = probabilities
            // Create ML's exported Image Feature Print model declares its image
            // grid and includes its extractor/normalization. The feature-print
            // image adapter uses scaleFill, preserving the complete field of view.
            // A custom export may record an explicit Vision adapter preference.
            let metadata = description.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
            switch metadata["texture_studio.image_crop_and_scale"] ?? "scaleFill" {
            case "scaleFill": cropAndScale = .scaleFill; preprocessing = "Vision scaleFill to declared \(image.pixelsWide) × \(image.pixelsHigh) image input"
            case "centerCrop": cropAndScale = .centerCrop; preprocessing = "Vision centerCrop to declared \(image.pixelsWide) × \(image.pixelsHigh) image input"
            case "scaleFit": cropAndScale = .scaleFit; preprocessing = "Vision scaleFit to declared \(image.pixelsWide) × \(image.pixelsHigh) image input"
            default: throw StudioError("The classifier records an unsupported Vision image crop-and-scale option.")
            }
            visionModel = try VNCoreMLModel(for: model)
            temporaryCompiledURL = temporary
        } catch {
            if let temporary { try? FileManager.default.removeItem(at: temporary) }
            throw error
        }
    }

    deinit {
        if let temporaryCompiledURL { try? FileManager.default.removeItem(at: temporaryCompiledURL) }
    }

    func predict(url: URL) throws -> Prediction {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw StudioError("The classifier cannot read this diffuse image.")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any]
        let orientation = (properties?[kCGImagePropertyOrientation as String] as? NSNumber)
            .flatMap { CGImagePropertyOrientation(rawValue: $0.uint32Value) } ?? .up
        return try predict(handler: VNImageRequestHandler(url: url, orientation: orientation, options: [:]))
    }

    /// Prepared diffuse frames are already oriented. Render their tagged linear
    /// color to sRGB for the classifier's BGRA image input, independently of the
    /// compact reconstruction model's linear diffuse tensor contract.
    func predict(image: CIImage) throws -> Prediction {
        try Task.checkCancellation()
        guard image.extent.width.isFinite, image.extent.height.isFinite,
              image.extent.width > 0, image.extent.height > 0,
              let frame = colorContext.createCGImage(image, from: image.extent, format: .RGBA8,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!) else {
            throw StudioError("The classifier cannot prepare this diffuse color frame.")
        }
        return try predict(handler: VNImageRequestHandler(cgImage: frame, orientation: .up, options: [:]))
    }

    private func predict(handler: VNImageRequestHandler) throws -> Prediction {
        lock.lock()
        defer { lock.unlock() }
        try Task.checkCancellation()
        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = cropAndScale
        try handler.perform([request])
        try Task.checkCancellation()
        guard let observations = request.results as? [VNClassificationObservation], !observations.isEmpty else {
            throw StudioError("The diffuse classifier returned no image classes.")
        }
        var probabilities = Dictionary(uniqueKeysWithValues: labels.map { ($0, 0.0) })
        var seen = Set<String>()
        for observation in observations {
            let value = Double(observation.confidence)
            guard probabilities[observation.identifier] != nil, seen.insert(observation.identifier).inserted,
                  value.isFinite, value >= 0 else {
                throw StudioError("The diffuse classifier returned invalid class probabilities.")
            }
            // Vision passes Core ML confidence through unchanged. Retain the
            // model's scores rather than renormalizing or inventing confidence.
            probabilities[observation.identifier] = value
        }
        guard let best = observations.max(by: { $0.confidence < $1.confidence }) else {
            throw StudioError("The diffuse classifier returned no prediction.")
        }
        return Prediction(label: best.identifier, confidence: Double(best.confidence),
            probabilities: probabilities, modelSHA256: modelSHA256)
    }

    /// Original .mlmodel files use their ordinary SHA-256. Directory models use
    /// a relocation-independent inventory of all files and their exact hashes.
    static func modelIdentity(at url: URL) throws -> String {
        let root = url.standardizedFileURL
        if try root.resourceValues(forKeys: [.isDirectoryKey]).isDirectory != true {
            return try fileSHA256(root)
        }
        guard let enumeration = FileManager.default.enumerator(at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else {
            throw StudioError("The classifier package cannot be inspected.")
        }
        var files: [URL] = []
        for case let file as URL in enumeration {
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard values.isSymbolicLink != true else { throw StudioError("Classifier packages cannot contain symbolic links.") }
            if values.isRegularFile == true { files.append(file) }
        }
        guard !files.isEmpty else { throw StudioError("The classifier package is empty.") }
        var digest = SHA256()
        digest.update(data: Data("texture-studio-diffuse-classifier-package-v1\n".utf8))
        for file in files.sorted(by: { $0.path < $1.path }) {
            try Task.checkCancellation()
            let relative = String(file.path.dropFirst(root.path.count + 1))
            digest.update(data: Data("\(relative.utf8.count):\(relative):\(try fileSHA256(file))\n".utf8))
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func fileSHA256(_ url: URL) throws -> String {
        let reader = try FileHandle(forReadingFrom: url)
        defer { try? reader.close() }
        var digest = SHA256()
        while let bytes = try reader.read(upToCount: 1_048_576), !bytes.isEmpty {
            try Task.checkCancellation()
            digest.update(data: bytes)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
