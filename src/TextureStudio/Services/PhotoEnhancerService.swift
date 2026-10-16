import CoreImage
import CoreML
import CryptoKit
import Foundation

struct PhotoEnhancerDescriptor: Codable, Sendable {
    let schema: String
    let name: String
    let product: String
    let architecture: String
    let width: Int
    let input: String
    let output: String
    let pixelContract: String
    let modelFile: String
    let modelSHA256: String
    let weights: String
    let weightsSHA256: String
    let runID: String
    let updates: Int
    let qualified: Bool

    static let architectureID = "loom-factory-photo-compact-naf-v1"
    static let contract = "opaque sRGB numeric Float32 NCHW; input [0,1]; raw residual RGB output; no vector normalization"
    static func root(_ product: String, bundle: Bundle = .main) -> URL? {
        bundle.resourceURL?.appendingPathComponent("PhotoEnhancers/" + product)
    }
    static func load(_ root: URL, product: String) throws -> Self {
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("enhancer.json")))
        guard value.schema == "loom.photo-enhancer.v1", value.architecture == architectureID,
              value.product == product, ["photo-repair-v1", "photo-lighting-v1"].contains(product),
              value.input == "rgb", value.output == "enhanced_rgb", value.pixelContract == contract,
              value.modelFile == "Enhancer.mlmodel", value.weights == "weights.safetensors",
              [16,22,32,46].contains(value.width), value.updates > 0 else {
            throw StudioError("Unsupported or untrained photo enhancer contract.")
        }
        for (file, expected) in [(value.modelFile, value.modelSHA256), (value.weights, value.weightsSHA256)] {
            let data = try Data(contentsOf: root.appendingPathComponent(file), options: .mappedIfSafe)
            let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard actual == expected else { throw StudioError("Photo enhancer checksum changed: \(file).") }
        }
        return value
    }
    static func identities(_ settings: TextureSettings) throws -> [String: String] {
        var result: [String:String] = [:]
        for (product, strength) in [("photo-repair-v1", settings.photoRepairStrength), ("photo-lighting-v1", settings.photoLightingStrength)] where strength > 0 {
            guard let root = root(product) else { throw StudioError("Missing photo enhancer in this build.") }
            result[product] = try load(root, product: product).modelSHA256
        }
        return result
    }
    static func available(_ product: String) -> Bool {
        guard let root = root(product) else { return false }
        return (try? load(root, product: product)) != nil
    }
}

/// Optional build-time RGB products; no material map or normal-vector decoding.
actor PhotoEnhancerService {
    private var loaded: [String: MLModel] = [:]
    private let context: CIContext
    private let resourceRoot: URL?
    private let computeUnits: MLComputeUnits
    init(resourceRoot: URL? = Bundle.main.resourceURL, computeUnits: MLComputeUnits = .all) {
        self.resourceRoot = resourceRoot; self.computeUnits = computeUnits
        context = CIContext(options: [.workingFormat: CIFormat.RGBAf, .cacheIntermediates: false, .useSoftwareRenderer: computeUnits == .cpuOnly])
    }
    func apply(_ image: CIImage, repair: Float, lighting: Float) throws -> CIImage {
        var result = image
        for (product, strength) in [("photo-repair-v1", repair), ("photo-lighting-v1", lighting)] where strength > 0 {
            try Task.checkCancellation()
            guard strength.isFinite, strength <= 1, let root = resourceRoot?.appendingPathComponent("PhotoEnhancers/" + product) else { throw StudioError("Invalid photo enhancer strength.") }
            let descriptor = try PhotoEnhancerDescriptor.load(root, product: product)
            // First release is explicitly 1K. Whole-field attention cannot be tiled silently.
            guard result.extent.width == 1024, result.extent.height == 1024 else { throw StudioError("Photo enhancers currently support 1024 × 1024 opaque photos. Choose 1K output or disable the enhancers.") }
            let model: MLModel
            if let cached = loaded[product] { model = cached }
            else {
                let compiled = root.appendingPathComponent("Enhancer.mlmodelc")
                let configuration = MLModelConfiguration(); configuration.computeUnits = computeUnits
                model = try MLModel(contentsOf: compiled, configuration: configuration)
                let attributes = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
                guard attributes?["loom.product"] == product, attributes?["loom.architecture"] == descriptor.architecture,
                      attributes?["loom.width"] == String(descriptor.width),
                      model.modelDescription.inputDescriptionsByName["rgb"]?.multiArrayConstraint?.dataType == .float32,
                      model.modelDescription.outputDescriptionsByName["enhanced_rgb"]?.multiArrayConstraint?.dataType == .float32 else {
                    throw StudioError("The photo enhancer's compiled RGB interface does not match its descriptor.")
                }
                loaded[product] = model
            }
            let count = 1024*1024, space = CGColorSpace(name: CGColorSpace.sRGB)!
            var rgba = [Float](repeating: 0, count: count*4)
            rgba.withUnsafeMutableBytes { context.render(result, toBitmap: $0.baseAddress!, rowBytes: 1024*16, bounds: result.extent, format: .RGBAf, colorSpace: space) }
            let input = try MLMultiArray(shape: [1,3,1024,1024], dataType: .float32)
            let pointer = input.dataPointer.assumingMemoryBound(to: Float.self)
            for pixel in 0..<count {
                guard rgba[pixel*4+3] >= 0.99999 else { throw StudioError("Photo enhancers do not yet support alpha materials such as nets. Disable them for this photo.") }
                for channel in 0..<3 {
                    let value = rgba[pixel*4+channel]
                    guard value.isFinite, value >= 0, value <= 1 else { throw StudioError("Photo enhancers currently require SDR RGB in [0,1]. Disable HDR gain-map processing or the enhancers.") }
                    pointer[channel*count + pixel] = value
                }
            }
            let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: input)]))
            guard let output = prediction.featureValue(for: "enhanced_rgb")?.multiArrayValue,
                  output.dataType == .float32, output.shape.map(\.intValue) == [1,3,1024,1024] else { throw StudioError("Photo enhancer returned an incompatible RGB tensor.") }
            let values = output.dataPointer.assumingMemoryBound(to: Float.self), strides = output.strides.map(\.intValue)
            for pixel in 0..<count {
                for channel in 0..<3 {
                    let value = values[channel*strides[1] + (pixel/1024)*strides[2] + (pixel%1024)*strides[3]]
                    guard value.isFinite else { throw StudioError("Photo enhancer produced non-finite RGB.") }
                    let index = pixel*4+channel
                    rgba[index] = min(1, max(0, rgba[index] + strength*(value-rgba[index])))
                }
                rgba[pixel*4+3] = 1
            }
            result = rgba.withUnsafeBytes { CIImage(bitmapData: Data($0), bytesPerRow: 1024*16, size: CGSize(width:1024,height:1024), format:.RGBAf, colorSpace:space) }
                .transformed(by: CGAffineTransform(translationX: image.extent.minX, y: image.extent.minY))
        }
        return result
    }
}
