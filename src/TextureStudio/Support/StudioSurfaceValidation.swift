import CoreImage
import Foundation
import ImageIO

/// Explicit developer smoke using the same store, selected model, saved runtime,
/// preview, and export operations as the native app. It never installs a model.
@MainActor
enum StudioSurfaceValidation {
    static func importDirectory(_ path: String) async throws {
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let files = try FileManager.default.contentsOfDirectory(at: directory,
            includingPropertiesForKeys: [.isRegularFileKey], options: .skipsHiddenFiles)
            .filter { ["heic", "dng"].contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw StudioError("No HEIC or DNG files in the import validation directory.") }
        let engine = TextureEngine()
        var report: [ImportSummary] = []
        for file in files {
            let imported = try await engine.importPhoto(file)
            let preview = try await engine.preview(imported.orientedImage, maxDimension: 256)
            guard let source = CGImageSourceCreateWithURL(file as CFURL,
                [kCGImageSourceShouldCache: false] as CFDictionary),
                let properties = CGImageSourceCopyPropertiesAtIndex(source, imported.primaryImageIndex, nil) as? [String: Any],
                let width = (properties[kCGImagePropertyPixelWidth as String] as? NSNumber)?.intValue,
                let height = (properties[kCGImagePropertyPixelHeight as String] as? NSNumber)?.intValue else {
                throw StudioError("Cannot compare source dimensions for \(file.lastPathComponent).")
            }
            let swapsAxes = [5, 6, 7, 8].contains(imported.camera.orientation)
            guard imported.pixelWidth == (swapsAxes ? height : width),
                  imported.pixelHeight == (swapsAxes ? width : height),
                  preview.width > 0, preview.height > 0,
                  max(preview.width, preview.height) <= 256 else {
                throw StudioError("Native orientation or bounded preview dimensions differ for \(file.lastPathComponent).")
            }
            let type = CGImageSourceGetType(source).map { $0 as String } ?? "unknown"
            report.append(ImportSummary(photo: file.path, type: type,
                sourceWidth: width, sourceHeight: height, orientation: imported.camera.orientation,
                orientedWidth: imported.pixelWidth, orientedHeight: imported.pixelHeight,
                previewWidth: preview.width, previewHeight: preview.height,
                sourceBitDepth: imported.camera.sourceBitDepth,
                auxiliaryTypes: imported.camera.auxiliaryTypes))
            print("Native import passed: \(file.lastPathComponent) · \(type) · EXIF \(imported.camera.orientation) · \(imported.pixelWidth) × \(imported.pixelHeight)")
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        if let output = ProcessInfo.processInfo.environment["TEXTURE_STUDIO_SMOKE_IMPORT_REPORT"] {
            let url = URL(fileURLWithPath: output)
            guard !FileManager.default.fileExists(atPath: url.path) else {
                throw StudioError("Choose a new import validation report path.")
            }
            try data.write(to: url, options: .atomic)
        } else { print(String(decoding: data, as: UTF8.self)) }
        print("Texture Studio native import validation passed: \(report.count) photos")
    }

    private struct ImportSummary: Encodable {
        let photo: String
        let type: String
        let sourceWidth: Int
        let sourceHeight: Int
        let orientation: UInt32
        let orientedWidth: Int
        let orientedHeight: Int
        let previewWidth: Int
        let previewHeight: Int
        let sourceBitDepth: Int?
        let auxiliaryTypes: [String]
    }
}
