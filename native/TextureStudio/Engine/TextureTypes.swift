import Foundation
import CoreImage

enum EXRPrecision: String, Codable, CaseIterable, Sendable, Identifiable {
    case float16, float32
    var id: String { rawValue }
    var title: String { self == .float16 ? "16-bit float EXR" : "32-bit float EXR" }
}

struct TextureSettings: Codable, Sendable, Equatable {
    var rotationX: Double = 0
    var rotationY: Double = 0
    var rotationZ: Double = 0
    var focalLengthPixels: Double? = nil
    var cropScale: Double = 1
    var cropOffsetX: Double = 0
    var cropOffsetY: Double = 0
    var outputSize: Int = 1024
    var useSupportingViews: Bool = true
    var useHDRGainMap: Bool = true
    var lensDistortion: Double = 0
    var exrPrecision: EXRPrecision = .float32
    var lightingStrength: Float = 0.65
    var lightingRadius: Float = 0.12
    var noiseReduction: Float = 0.015
    var heightStrength: Float = 1
    var heightDetail: Float = 0
    var surfacePlaneRemoval: Float = 1
    var depthCleanup: Float = 0.25
    var heightInvert: Bool = false
    var adaptiveRelief: Bool = true
    var attachedMapIsHeight: Bool = true
    var modelProcessResolution: Int = 1036
    var roughnessBase: Float = 0.65
    var roughnessDetail: Float = 0.2
    var materialWidthMeters: Double = 0.25
    var displacementScaleMeters: Double = 0.005
    static let outputSizes = [1024, 2048, 4098, 8192]
    static let modelProcessResolutions = [1036, 1540, 2044]
    init() {}

    private enum CodingKeys: String, CodingKey {
        case rotationX, rotationY, rotationZ, focalLengthPixels, cropScale, cropOffsetX, cropOffsetY,
             outputSize, useSupportingViews, useHDRGainMap, lensDistortion,
             exrPrecision, lightingStrength, lightingRadius, noiseReduction, heightStrength,
             heightDetail, surfacePlaneRemoval, depthCleanup, heightInvert, adaptiveRelief, attachedMapIsHeight, modelProcessResolution,
             roughnessBase, roughnessDetail, materialWidthMeters, displacementScaleMeters
    }
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy:CodingKeys.self)
        rotationX = try c.decodeIfPresent(Double.self,forKey:.rotationX) ?? rotationX
        rotationY = try c.decodeIfPresent(Double.self,forKey:.rotationY) ?? rotationY
        rotationZ = try c.decodeIfPresent(Double.self,forKey:.rotationZ) ?? rotationZ
        focalLengthPixels = try c.decodeIfPresent(Double.self,forKey:.focalLengthPixels)
        cropScale = try c.decodeIfPresent(Double.self,forKey:.cropScale) ?? cropScale
        cropOffsetX = try c.decodeIfPresent(Double.self,forKey:.cropOffsetX) ?? cropOffsetX
        cropOffsetY = try c.decodeIfPresent(Double.self,forKey:.cropOffsetY) ?? cropOffsetY
        outputSize = try c.decodeIfPresent(Int.self,forKey:.outputSize) ?? outputSize
        // Unknown legacy keys, including useEmbeddedDepth, are deliberately
        // ignored. There is no portrait-depth source in the material workflow.
        useSupportingViews = try c.decodeIfPresent(Bool.self,forKey:.useSupportingViews) ?? useSupportingViews
        useHDRGainMap = try c.decodeIfPresent(Bool.self,forKey:.useHDRGainMap) ?? useHDRGainMap
        lensDistortion = try c.decodeIfPresent(Double.self,forKey:.lensDistortion) ?? lensDistortion
        exrPrecision = try c.decodeIfPresent(EXRPrecision.self,forKey:.exrPrecision) ?? exrPrecision
        lightingStrength = try c.decodeIfPresent(Float.self,forKey:.lightingStrength) ?? lightingStrength
        lightingRadius = try c.decodeIfPresent(Float.self,forKey:.lightingRadius) ?? lightingRadius
        noiseReduction = try c.decodeIfPresent(Float.self,forKey:.noiseReduction) ?? noiseReduction
        heightStrength = try c.decodeIfPresent(Float.self,forKey:.heightStrength) ?? heightStrength
        heightDetail = try c.decodeIfPresent(Float.self,forKey:.heightDetail) ?? heightDetail
        surfacePlaneRemoval = try c.decodeIfPresent(Float.self,forKey:.surfacePlaneRemoval) ?? surfacePlaneRemoval
        depthCleanup = try c.decodeIfPresent(Float.self,forKey:.depthCleanup) ?? depthCleanup
        heightInvert = try c.decodeIfPresent(Bool.self,forKey:.heightInvert) ?? heightInvert
        adaptiveRelief = try c.decodeIfPresent(Bool.self,forKey:.adaptiveRelief) ?? adaptiveRelief
        attachedMapIsHeight = try c.decodeIfPresent(Bool.self,forKey:.attachedMapIsHeight) ?? attachedMapIsHeight
        modelProcessResolution = try c.decodeIfPresent(Int.self,forKey:.modelProcessResolution) ?? modelProcessResolution
        roughnessBase = try c.decodeIfPresent(Float.self,forKey:.roughnessBase) ?? roughnessBase
        roughnessDetail = try c.decodeIfPresent(Float.self,forKey:.roughnessDetail) ?? roughnessDetail
        materialWidthMeters = try c.decodeIfPresent(Double.self,forKey:.materialWidthMeters) ?? materialWidthMeters
        displacementScaleMeters = try c.decodeIfPresent(Double.self,forKey:.displacementScaleMeters) ?? displacementScaleMeters
    }
}

struct CameraMetadata: Codable, Sendable {
    var make: String?
    var model: String?
    var lensModel: String?
    var focalLengthMillimeters: Double?
    var focalLength35mm: Double?
    var aperture: Double?
    var exposureSeconds: Double?
    var iso: Double?
    var orientation: UInt32 = 1
    var sourceBitDepth: Int?
    var colorProfile: String?
    var auxiliaryTypes: [String] = []
    var summary: String {
        let camera = [make, model, lensModel].compactMap { $0 }.joined(separator: " · ")
        let focal = focalLengthMillimeters.map { String(format: "%.1f mm", $0) }
        return [camera.isEmpty ? "Camera metadata unavailable" : camera, focal].compactMap { $0 }.joined(separator: " · ")
    }
}

struct TextureSource: @unchecked Sendable {
    let url: URL
    let orientedImage: CIImage
    let camera: CameraMetadata
    let pixelWidth: Int
    let pixelHeight: Int
    var supportingViews: [CIImage] = []
    var primaryImageIndex: Int = 0
    var supportingImageIndices: [Int] = []
    var hdrImage: CIImage? = nil
    var metadata: CameraMetadata { camera }
}

enum DepthInterpretation: String, Codable, Sendable {
    case distance, inverseDepth, surfaceHeight
}

struct TextureDepth: @unchecked Sendable {
    let image: CIImage
    let sourceLabel: String
    let interpretation: DepthInterpretation
    init(image: CIImage, sourceLabel: String, interpretation: DepthInterpretation = .distance) {
        self.image = image
        self.sourceLabel = sourceLabel
        self.interpretation = interpretation
    }
    init(width: Int, height: Int, values: [Float], sourceLabel: String,
         interpretation: DepthInterpretation = .distance) throws {
        guard width > 0, height > 0, width <= 16384, height <= 16384,
              values.count == width * height else { throw TextureError.invalidDepth("Invalid depth dimensions or sample count.") }
        guard values.allSatisfy({ $0.isFinite }) else {
            throw TextureError.invalidDepth("Depth contains nonfinite samples. Repair or mask them before attaching it.")
        }
        // CIImage bitmap scanlines are top-down, matching Core ML tensors and CVPixelBuffer rows.
        self.init(image: CIImage(bitmapData: values.withUnsafeBytes { Data($0) }, bytesPerRow: width * 4,
                                size: CGSize(width: width, height: height), format: .Rf, colorSpace: nil),
                  sourceLabel: sourceLabel, interpretation: interpretation)
    }
}

struct MaterialResult: @unchecked Sendable {
    let diffuse: CIImage
    let roughness: CIImage
    let normal: CIImage
    let height: CIImage
    let crop: CGRect
    let warnings: [String]
    let outputSize: Int
    let depthOrigin: String
    let settings: TextureSettings
    let sourceURL: URL
    let camera: CameraMetadata
    var depth: CIImage { height }
}

enum TextureError: LocalizedError {
    case invalidImage(String)
    case invalidDepth(String)
    case invalidSettings(String)
    case insufficientResources(String)
    case missingMetal
    case missingKernels
    case processing(String)
    var errorDescription: String? {
        switch self {
        case .invalidImage(let reason), .invalidDepth(let reason), .invalidSettings(let reason),
             .insufficientResources(let reason), .processing(let reason): return reason
        case .missingMetal: return "Texture Studio needs a Mac with Metal support."
        case .missingKernels: return "Texture Studio's Metal processing library is missing. Rebuild the app with Xcode."
        }
    }
}
