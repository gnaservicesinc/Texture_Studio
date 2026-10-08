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

    static func run() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let photoPath = environment["TEXTURE_STUDIO_SMOKE_PHOTO"] else {
            throw StudioError("DA3 surface validation requires TEXTURE_STUDIO_SMOKE_PHOTO.")
        }
        let started = Date()
        let models = ModelManager()
        let runtime = PythonDepthService()
        let workspace = TextureWorkspace(pythonDepthService: runtime)
        guard models.availableURL(for: LocalModelDescriptor.da3GiantID) != nil else {
            throw StudioError("Install or locate the exact DA3 GIANT 1.1 model before running this smoke.")
        }
        // Give the saved-runtime probe scheduled by init a turn to begin.
        await Task.yield()
        try await waitUntil(timeout: 130, label: "saved Python runtime probe") { !runtime.isBusy }
        guard runtime.runtimeStatus.isReady else { throw StudioError(runtime.runtimeStatus.message) }
        workspace.importPhoto(URL(fileURLWithPath: photoPath))
        try await waitForWorkspace(workspace, timeout: 120)
        guard workspace.source != nil else { throw StudioError("Native photo import did not publish a source.") }
        guard workspace.depthChoice == .model, workspace.modelID == LocalModelDescriptor.da3GiantID else {
            throw StudioError("The native default did not select DA3 GIANT 1.1.")
        }
        if let edge = environment["TEXTURE_STUDIO_SMOKE_INFERENCE_EDGE"] {
            guard let value = Int(edge) else { throw StudioError("Invalid inference edge.") }
            try PythonDepthService.validateResolution(value)
            workspace.settings.modelProcessResolution = value
        }
        workspace.settings.outputSize = 1024
        workspace.settings.exrPrecision = .float32
        workspace.selectedPreview = .height
        workspace.updatePreview(models: models)
        try await waitForWorkspace(workspace, timeout: 1900)
        guard let rawDepth = workspace.generatedDepth, let provenance = workspace.generatedProvenance,
              let material = workspace.result, let preview = workspace.preview,
              provenance.modelID == "depth-anything/DA3-GIANT-1.1",
              provenance.checkpointSHA256 == LocalModelDescriptor.da3WeightsSHA256,
              provenance.device == "mps", provenance.precision == "Float32",
              preview.width == 1024, preview.height == 1024,
              workspace.renderedPreview == .height else {
            throw StudioError("Native preview did not use the exact DA3 prediction and final surface height.")
        }
        let output: URL
        let persistent = environment["TEXTURE_STUDIO_SMOKE_EXPORT_PATH"]
        if let persistent { output = URL(fileURLWithPath: persistent, isDirectory: true) }
        else { output = FileManager.default.temporaryDirectory.appendingPathComponent("texture-surface-\(UUID().uuidString)", isDirectory: true) }
        // Only the newly created temporary output is removed; explicit export
        // paths and existing folders are never deleted by this validation.
        defer { if persistent == nil { try? FileManager.default.removeItem(at: output) } }
        workspace.export(to: output, models: models, reveal: false)
        try await waitForWorkspace(workspace, timeout: 300)
        guard workspace.exportURL == output else { throw StudioError("Native store export did not finish.") }
        for name in ["diffuse.png", "roughness.exr", "normal.exr", "displacement.exr"] {
            let url = output.appendingPathComponent(name)
            guard let image = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let properties = CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [String: Any],
                  (properties[kCGImagePropertyPixelWidth as String] as? NSNumber)?.intValue == 1024,
                  (properties[kCGImagePropertyPixelHeight as String] as? NSNumber)?.intValue == 1024 else {
                throw StudioError("Native exported map failed 1024-pixel ImageIO validation: \(name)")
            }
            if name.hasSuffix(".exr") { try FloatEXRWriter.verifyChannelPrecision(at: url, expected: .float32) }
        }
        for name in ["material.json", "depth-source.json", "blender_material.py", "BLENDER.txt"] {
            guard FileManager.default.fileExists(atPath: output.appendingPathComponent(name).path) else {
                throw StudioError("Missing native export metadata or Blender setup: \(name)")
            }
        }
        let exportedProvenance = try JSONDecoder().decode(ModelDepthProvenance.self,
            from: Data(contentsOf: output.appendingPathComponent("depth-source.json")))
        guard exportedProvenance == provenance else { throw StudioError("Export changed the selected depth provenance.") }
        let height = try heightRange(material.height)
        let rawRange = try heightRange(rawDepth.image)
        let summary = ValidationSummary(photo: photoPath, model: provenance, outputSize: 1024,
            rawDepthWidth: Int(rawDepth.image.extent.width), rawDepthHeight: Int(rawDepth.image.extent.height),
            rawDepthMinimum: rawRange.0, rawDepthMaximum: rawRange.1,
            finalHeightMinimum: height.0,
            finalHeightMaximum: height.1,
            elapsedSeconds: Date().timeIntervalSince(started))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(summary).write(to: output.appendingPathComponent("native-validation.json"), options: .atomic)
        print("Texture Studio store DA3 validation passed: \(photoPath) → \(output.path)")
    }

    private static func waitForWorkspace(_ workspace: TextureWorkspace, timeout: Double) async throws {
        do {
            try await waitUntil(timeout: timeout, label: workspace.activity) { !workspace.isBusy }
        } catch {
            workspace.cancel()
            throw error
        }
        if let notice = workspace.notice { throw StudioError(notice.message) }
    }

    private static func waitUntil(timeout: Double, label: String, finished: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !finished() {
            try Task.checkCancellation()
            guard Date() < deadline else { throw StudioError("Native validation timed out: \(label)") }
            try await Task.sleep(for: .milliseconds(100))
        }
    }

    private static func heightRange(_ image: CIImage) throws -> (Float, Float) {
        let width = Int(image.extent.width), height = Int(image.extent.height)
        guard width > 0, height > 0, width <= 2044, height <= 2044 else {
            throw StudioError("Unexpected validation map size.")
        }
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        var samples = [Float](repeating: 0, count: width * height)
        samples.withUnsafeMutableBytes {
            context.render(image, toBitmap: $0.baseAddress!, rowBytes: width * 4,
                bounds: image.extent, format: .Rf, colorSpace: nil)
        }
        guard samples.allSatisfy(\.isFinite) else { throw StudioError("The native height/depth map is nonfinite.") }
        return (samples.min()!, samples.max()!)
    }

    private struct ValidationSummary: Encodable {
        let photo: String
        let model: ModelDepthProvenance
        let outputSize: Int
        let rawDepthWidth: Int
        let rawDepthHeight: Int
        let rawDepthMinimum: Float
        let rawDepthMaximum: Float
        let finalHeightMinimum: Float
        let finalHeightMaximum: Float
        let elapsedSeconds: Double
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
