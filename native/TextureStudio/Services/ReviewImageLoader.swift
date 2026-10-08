import AppKit
import CoreImage
import CryptoKit
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

struct ReviewOriginalExport: Sendable {
    let sourceURL: URL
    let filename: String
    let expectedSHA256: String?
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
    // Display conversion can use Metal. The original bytes remain untouched;
    // numeric maps retain their linear values until conversion to the display.
    private let numericContext = CIContext(options: [.cacheIntermediates: false, .workingFormat: CIFormat.RGBAf,
                                                     .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
    private let photoContext = CIContext(options: [.cacheIntermediates: false])
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
    static func exportOriginal(_ source: URL, expectedSHA256: String? = nil, to destination: URL) throws {
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw ReviewImageError.destinationExists }
        let bytes = try Data(contentsOf: source, options: .mappedIfSafe)
        let originalHash = hash(bytes)
        if let expectedSHA256, originalHash != expectedSHA256 { throw ReviewImageError.sourceChanged }
        try bytes.write(to: destination, options: .withoutOverwriting)
        guard hash(try Data(contentsOf: destination, options: .mappedIfSafe)) == originalHash else {
            throw ReviewImageError.sourceChanged
        }
    }
    static func exportOriginals(_ exports: [ReviewOriginalExport], to directory: URL) throws {
        guard !FileManager.default.fileExists(atPath: directory.path) else { throw ReviewImageError.destinationExists }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        do {
            var filenames = Set<String>()
            for item in exports {
                let requested = URL(fileURLWithPath: item.filename).lastPathComponent
                let stem = (requested as NSString).deletingPathExtension
                let suffix = (requested as NSString).pathExtension
                var filename = requested
                var number = 2
                while filenames.contains(filename.lowercased()) {
                    filename = "\(stem)-\(number)" + (suffix.isEmpty ? "" : ".\(suffix)")
                    number += 1
                }
                filenames.insert(filename.lowercased())
                try exportOriginal(item.sourceURL, expectedSHA256: item.expectedSHA256,
                                   to: directory.appendingPathComponent(filename))
            }
        } catch {
            // This directory was created exclusively for this export. Never
            // remove or overwrite an existing user folder on a failed copy.
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }
}
