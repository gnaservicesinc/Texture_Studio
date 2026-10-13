import AppKit
import CoreImage
import CryptoKit
import Darwin
import Foundation
import ImageIO

struct ReviewLoadedImage: @unchecked Sendable {
    let image: CGImage
    let sourceURL: URL
    let sourceSHA256: String
    let pixelWidth: Int
    let pixelHeight: Int
    let storageDescription: String
}

enum ReviewImageError: LocalizedError {
    case missingSource, invalidImage, sourceChanged
    var errorDescription: String? {
        switch self {
        case .missingSource: "The source file is no longer present."
        case .invalidImage: "A display preview is unavailable for this file. Its original data remains available for export."
        case .sourceChanged: "The source data changed after inspection. Reopen the map to inspect its current revision."
        }
    }
}

actor ReviewImageLoader {
    static let shared = ReviewImageLoader()
    // Display conversion can use Metal. The original bytes remain untouched;
    // numeric maps retain their linear values until conversion to the display.
    private let numericContext = CIContext(options: [.cacheIntermediates: false, .workingFormat: CIFormat.RGBAf,
                                                     .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
    private let photoContext = CIContext(options: [.cacheIntermediates: false])
    func load(_ url: URL, numeric: Bool, contrast: Double = 1, midpoint: Double = 0.5,
              expectedSHA256: String? = nil) async throws -> ReviewLoadedImage {
        let bytes = try Self.readSource(url)
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
            guard contrast.isFinite, contrast > 0, midpoint.isFinite, (0...1).contains(midpoint) else {
                throw StudioError("Display contrast must be positive and finite, and midpoint must be from 0 to 1.")
            }
            let gain = contrast, offset = midpoint * (1 - gain)
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
                                 pixelWidth: image.width, pixelHeight: image.height,
                                 storageDescription: Self.storageDescription(bytes))
    }
    /// Report the stored channel precision, not the 8-bit display conversion.
    /// These header reads leave the original samples completely untouched.
    static func storageDescription(_ bytes: Data) -> String {
        let pngMagic = Data([137, 80, 78, 71, 13, 10, 26, 10])
        if bytes.count >= 33, bytes.prefix(8) == pngMagic,
           bytes.subdata(in: 12..<16) == Data("IHDR".utf8) {
            return "\(bytes[24])-bit integer PNG"
        }
        if bytes.count >= 4, bytes.prefix(4) == Data([0x76, 0x2f, 0x31, 0x01]),
           let types = try? FloatEXRWriter.channelTypes(bytes), !types.isEmpty {
            let formats = Set(types.map { $0 == 1 ? "16-bit float" : $0 == 2 ? "32-bit float" : "32-bit integer" }).sorted()
            return formats.joined(separator: " / ") + " EXR"
        }
        if let source = CGImageSourceCreateWithData(bytes as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
           let properties = CGImageSourceCopyPropertiesAtIndex(source, CGImageSourceGetPrimaryImageIndex(source), nil) as? [String: Any],
           let bits = (properties[kCGImagePropertyDepth as String] as? NSNumber)?.intValue {
            return "\(bits)-bit per channel"
        }
        return "Stored precision unavailable"
    }
    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    /// Only ENOENT is a missing source. Permissions and unsupported decoding
    /// must never cause a valid dataset entry to be removed.
    static func readSource(_ url: URL) throws -> Data {
        do { return try Data(contentsOf: url, options: .mappedIfSafe) }
        catch {
            let issue = error as NSError
            if (issue.domain == NSCocoaErrorDomain && issue.code == NSFileReadNoSuchFileError)
                || (issue.domain == NSPOSIXErrorDomain && issue.code == ENOENT) {
                throw ReviewImageError.missingSource
            }
            throw error
        }
    }
}
