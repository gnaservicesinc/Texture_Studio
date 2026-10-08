import AppKit
import CoreImage
import CryptoKit
import Foundation

struct ReviewLoadedImage: @unchecked Sendable {
    let image: CGImage
    let sourceURL: URL
    let sourceSHA256: String
    let pixelWidth: Int
    let pixelHeight: Int
}

enum ReviewImageError: LocalizedError {
    case invalidImage, sourceChanged, destinationExists
    var errorDescription: String? {
        switch self {
        case .invalidImage: "The source could not be decoded at its full resolution."
        case .sourceChanged: "The original map changed after inspection. Reopen it before exporting."
        case .destinationExists: "Choose a new filename. Original-map export does not overwrite files."
        }
    }
}

actor ReviewImageLoader {
    static let shared = ReviewImageLoader()
    private let numericContext = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: true,
                                                     .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
    private let photoContext = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: true])
    func load(_ url: URL, numeric: Bool, contrast: Double = 1, midpoint: Double = 0.5,
              expectedSHA256: String? = nil) throws -> ReviewLoadedImage {
        let bytes = try Data(contentsOf: url, options: .mappedIfSafe)
        let hash = Self.hash(bytes)
        if let expectedSHA256, hash != expectedSHA256 { throw ReviewImageError.sourceChanged }
        let options: [CIImageOption: Any] = numeric ? [.colorSpace: NSNull(), .applyOrientationProperty: false] : [.applyOrientationProperty: true]
        guard let source = CIImage(data: bytes, options: options),
              [source.extent.origin.x, source.extent.origin.y, source.extent.width, source.extent.height].allSatisfy(\.isFinite),
              source.extent.width > 0, source.extent.height > 0,
              source.extent.width <= 16384, source.extent.height <= 16384,
              source.extent.width * source.extent.height <= 150_000_000 else {
            throw ReviewImageError.invalidImage
        }
        let display: CIImage
        if numeric {
            let gain = min(32, max(1, contrast)), offset = min(1, max(0, midpoint)) * (1 - gain)
            display = source.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: gain, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: gain, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: gain, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBiasVector": CIVector(x: offset, y: offset, z: offset, w: 0)])
        } else { display = source }
        let context = numeric ? numericContext : photoContext
        guard let image = context.createCGImage(display, from: source.extent, format: .RGBA8,
                                                 colorSpace: numeric ? nil : CGColorSpace(name: CGColorSpace.sRGB)) else {
            throw ReviewImageError.invalidImage
        }
        return ReviewLoadedImage(image: image, sourceURL: url, sourceSHA256: hash,
                                 pixelWidth: image.width, pixelHeight: image.height)
    }
    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func exportOriginal(_ source: URL, expectedSHA256: String, to destination: URL) throws {
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw ReviewImageError.destinationExists }
        let bytes = try Data(contentsOf: source, options: .mappedIfSafe)
        guard hash(bytes) == expectedSHA256 else { throw ReviewImageError.sourceChanged }
        try bytes.write(to: destination, options: .withoutOverwriting)
        guard hash(try Data(contentsOf: destination, options: .mappedIfSafe)) == expectedSHA256 else {
            throw ReviewImageError.sourceChanged
        }
    }
}
