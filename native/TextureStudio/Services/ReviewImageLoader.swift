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

struct ReviewOriginalExport: Sendable {
    let sourceURL: URL
    let filename: String
    let expectedSHA256: String?
}

enum ReviewImageError: LocalizedError {
    case missingSource, invalidImage, sourceChanged, destinationExists
    var errorDescription: String? {
        switch self {
        case .missingSource: "The source file is no longer present."
        case .invalidImage: "A display preview is unavailable for this file. Its original data remains available for export."
        case .sourceChanged: "The source data changed after inspection. Refresh the dataset or reopen the comparison to inspect its current revision."
        case .destinationExists: "Choose a new filename. Original-map export does not overwrite files."
        }
    }
}

actor ReviewImageLoader {
    static let shared = ReviewImageLoader()
    typealias Reconstruct = @Sendable (URL, MapReviewDisplayTransform, URL) async throws -> Void
    private let reconstruct: Reconstruct
    init(reconstruct: @escaping Reconstruct = ReviewImageLoader.reconstructSource) { self.reconstruct = reconstruct }
    // Display conversion can use Metal. The original bytes remain untouched;
    // numeric maps retain their linear values until conversion to the display.
    private let numericContext = CIContext(options: [.cacheIntermediates: false, .workingFormat: CIFormat.RGBAf,
                                                     .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
    private let photoContext = CIContext(options: [.cacheIntermediates: false])
    func load(_ url: URL, numeric: Bool, contrast: Double = 1, midpoint: Double = 0.5,
              expectedSHA256: String? = nil, displayTransform: MapReviewDisplayTransform? = nil) async throws -> ReviewLoadedImage {
        let bytes = try Self.readSource(url)
        let hash = Self.hash(bytes)
        if let expectedSHA256, hash != expectedSHA256 { throw ReviewImageError.sourceChanged }
        if let transform = displayTransform {
            guard transform.sourceSHA256 == hash else { throw ReviewImageError.sourceChanged }
            guard transform.algorithm == MapReviewDisplayTransform.exactCrop, (1...16384).contains(transform.size),
                  ["input", "height", "roughness", "normal"].contains(transform.mapType),
                  ["opengl", "directx"].contains(transform.normalConvention) else {
                throw StudioError("The recorded training crop is unsupported. Recreate the review with the current material trainer.")
            }
            let dimensions = Self.pixelDimensions(bytes)
            if let rectangle = transform.cropRectangle {
                guard dimensions.count == 2, rectangle.count == 4, rectangle[0] >= 0, rectangle[1] >= 0,
                      rectangle[2] == transform.size, rectangle[3] == transform.size,
                      rectangle[0] <= dimensions[0] - rectangle[2], rectangle[1] <= dimensions[1] - rectangle[3] else {
                    throw StudioError("The recorded training crop is outside the original source map.")
                }
            }
            if dimensions != [transform.size, transform.size] || (transform.mapType == "normal" && transform.normalConvention == "directx") {
                let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("material-review-\(UUID().uuidString)")
                try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
                defer { try? FileManager.default.removeItem(at: temporary) }
                let reconstructed = temporary.appendingPathComponent("training-source.png")
                try await reconstruct(url, transform, reconstructed)
                try Task.checkCancellation()
                let displayed: ReviewLoadedImage
                do { displayed = try await load(reconstructed, numeric: numeric, contrast: contrast, midpoint: midpoint) }
                catch ReviewImageError.missingSource {
                    // A failed reconstruction is not a missing original and
                    // must never remove its candidate from the review list.
                    throw ReviewImageError.invalidImage
                }
                guard displayed.pixelWidth == transform.size, displayed.pixelHeight == transform.size else {
                    throw StudioError("The review source does not match its recorded training grid.")
                }
                // Fully realize the image before removing all temporary bytes.
                return ReviewLoadedImage(image: displayed.image, sourceURL: url, sourceSHA256: hash,
                    pixelWidth: displayed.pixelWidth, pixelHeight: displayed.pixelHeight,
                    storageDescription: "Training grid reconstructed · original \(dimensions.map(String.init).joined(separator: " × ")) · \(Self.storageDescription(bytes))")
            }
        }
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
    private static func pixelDimensions(_ bytes: Data) -> [Int] {
        guard let source = CGImageSourceCreateWithData(bytes as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
              let width = (properties[kCGImagePropertyPixelWidth as String] as? NSNumber)?.intValue,
              let height = (properties[kCGImagePropertyPixelHeight as String] as? NSNumber)?.intValue else { return [] }
        return [width, height]
    }
    private static func reconstructSource(_ source: URL, _ transform: MapReviewDisplayTransform, _ output: URL) async throws {
        let defaults = UserDefaults(suiteName: "org.ipde.material-tools") ?? .standard
        let workspace = defaults.string(forKey: "workspace") ?? FileManager.default.currentDirectoryPath
        let python = defaults.string(forKey: "python") ?? MaterialWorkbenchRuntime.defaultPython(workspace: URL(fileURLWithPath: workspace))
        let backend = Bundle.main.resourceURL!.appendingPathComponent("MaterialBackend/material_model_workbench.py")
        guard FileManager.default.isExecutableFile(atPath: python), FileManager.default.fileExists(atPath: backend.path) else {
            throw StudioError("Locate the material Python runtime in Settings to reconstruct this training grid.")
        }
        let process = WorkbenchProcess()
        await WorkbenchLifecycle.shared.add(process)
        do {
            var arguments = ["-B", backend.path, "review-source",
                "--image", source.path, "--expected-sha256", transform.sourceSHA256, "--size", String(transform.size), "--output", output.path,
                "--map-type", transform.mapType, "--normal-convention", transform.normalConvention]
            if let rectangle = transform.cropRectangle { arguments += ["--source-rectangle"] + rectangle.map(String.init) }
            _ = try await process.run(executable: URL(fileURLWithPath: python), arguments: arguments,
                directory: FileManager.default.fileExists(atPath: workspace) ? URL(fileURLWithPath: workspace) : nil,
                log: output.deletingLastPathComponent().appendingPathComponent("worker.log"), onLog: { _ in })
            await WorkbenchLifecycle.shared.remove(process)
        } catch { await WorkbenchLifecycle.shared.remove(process); throw error }
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
    static func exportOriginal(_ source: URL, expectedSHA256: String? = nil, to destination: URL) throws {
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw ReviewImageError.destinationExists }
        let bytes = try readSource(source)
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
