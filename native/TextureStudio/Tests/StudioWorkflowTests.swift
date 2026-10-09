import AppKit
import CoreImage
import XCTest
@testable import TextureStudio

@MainActor
final class StudioWorkflowTests: XCTestCase {
    func testNativeMaterialIsReusedByInspectionExportAndPrecisionChanges() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        var predictions = 0
        var processedSizes: [Int] = []
        let workspace = TextureWorkspace(checkpointRegistryURL: registry, checkpointPredictor: { _, _, size in
            predictions += 1
            return try TextureDepth(width: 2, height: 2, values: [0.45, 0.5, 0.55, 0.6], sourceLabel: "Fixture \(size)", interpretation: .surfaceHeight)
        }, materialProcessor: { source, settings, _ in
            processedSizes.append(settings.outputSize)
            return Self.material(source, settings: settings)
        })
        workspace.source = source(root.appendingPathComponent("surface.png"))
        workspace.attachedDepth = try TextureDepth(width: 2, height: 2, values: [0.45, 0.5, 0.55, 0.6],
            sourceLabel: "Attached height", interpretation: .surfaceHeight)
        workspace.depthChoice = .attached
        workspace.settings.outputSize = 2048
        workspace.selectedPreview = .height
        let models = ModelManager()
        workspace.updatePreview(models: models)
        try await finish(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(predictions, 0)
        XCTAssertEqual(processedSizes, [2048])
        XCTAssertFalse(workspace.materialNeedsUpdate)
        XCTAssertTrue(workspace.fullQualityAvailable)
        let map = try XCTUnwrap(workspace.cachedMaterialMapURL(.height))
        let native = try XCTUnwrap(CIImage(contentsOf: map, options: [.colorSpace: NSNull()]))
        XCTAssertEqual(native.extent.width, 2048)
        XCTAssertEqual(native.extent.height, 2048)
        let bytes = try Data(contentsOf: map)

        let existingWindows = Set(NSApp.windows.map(ObjectIdentifier.init))
        workspace.inspectFullQuality(models: models)
        XCTAssertFalse(workspace.isBusy, "Inspection only opens the already generated map")
        for window in NSApp.windows where !existingWindows.contains(ObjectIdentifier(window)) { window.close() }
        workspace.export(to: root.appendingPathComponent("float32"), models: models, reveal: false)
        try await finish(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("float32/displacement.exr")), bytes)

        workspace.settings.exrPrecision = .float16
        XCTAssertFalse(workspace.materialNeedsUpdate, "Storage precision does not change the material")
        XCTAssertTrue(workspace.fullQualityAvailable)
        workspace.export(to: root.appendingPathComponent("float16"), models: models, reveal: false)
        try await finish(workspace)
        XCTAssertNil(workspace.notice)
        try FloatEXRWriter.verifyChannelPrecision(at: root.appendingPathComponent("float16/displacement.exr"), expected: .float16)
        XCTAssertEqual(predictions, 0)
        XCTAssertEqual(processedSizes, [2048])

        workspace.settings.lightingStrength = 0.3
        XCTAssertTrue(workspace.materialNeedsUpdate)
        XCTAssertFalse(workspace.fullQualityAvailable)
        workspace.updatePreview(models: models)
        try await finish(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(predictions, 0, "An attached map bypasses model prediction")
        XCTAssertEqual(processedSizes, [2048, 2048])

        workspace.settings.outputSize = 1024
        workspace.updatePreview(models: models)
        try await finish(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(predictions, 0, "A different output size resamples the attached map without legacy inference")
        XCTAssertEqual(processedSizes, [2048, 2048, 1024])
        workspace.attachedDepth = try TextureDepth(width: 2, height: 2, values: [0.5, 0.6, 0.7, 0.8],
            sourceLabel: "Replacement attached height", interpretation: .surfaceHeight)
        XCTAssertTrue(workspace.materialNeedsUpdate)
        workspace.updatePreview(models: models)
        try await finish(workspace)
        XCTAssertEqual(predictions, 0)
        XCTAssertEqual(processedSizes, [2048, 2048, 1024, 1024])
    }

    func testStudioChoicesSurviveImportAndRelaunchWithoutSavingRecipe() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "studio-workflow-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let registry = root.appendingPathComponent("absent.json")
        let workspace = TextureWorkspace(checkpointRegistryURL: registry, preferences: defaults)
        workspace.depthChoice = .photoDetail
        workspace.modelID = "custom-depth"
        workspace.customInverseDepth = false
        workspace.showInspector = false
        workspace.selectedPreview = .normal
        workspace.settings.outputSize = 2048
        workspace.settings.rotationX = 4.2
        workspace.settings.exrPrecision = .float16
        workspace.settings.lightingStrength = 0.27
        let expected = workspace.settings
        let photo = root.appendingPathComponent("new-photo.png")
        try CIContext().writePNGRepresentation(of: source(photo).orientedImage, to: photo, format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        workspace.importPhoto(photo)
        try await finish(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(workspace.settings, expected)
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.selectedPreview, .normal)
        XCTAssertEqual(workspace.renderedPreview, .source, "A new photo must not display the previous photo's map")
        let reopened = TextureWorkspace(checkpointRegistryURL: registry, preferences: defaults)
        XCTAssertEqual(reopened.settings, expected)
        XCTAssertEqual(reopened.depthChoice, .photoDetail)
        XCTAssertEqual(reopened.modelID, "custom-depth")
        XCTAssertFalse(reopened.customInverseDepth)
        XCTAssertFalse(reopened.showInspector)
        XCTAssertEqual(reopened.selectedPreview, .normal)
    }

    private static func material(_ source: TextureSource, settings: TextureSettings) -> MaterialResult {
        let size = CGFloat(settings.outputSize)
        let extent = CGRect(x: 0, y: 0, width: size, height: size)
        let gray = CIImage(color: CIColor(red: 0.51, green: 0.51, blue: 0.51)).cropped(to: extent)
        return MaterialResult(diffuse: gray, roughness: gray,
            normal: CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 1)).cropped(to: extent), height: gray,
            crop: source.orientedImage.extent, warnings: [], outputSize: settings.outputSize,
            depthOrigin: "Fixture attached height", settings: settings, sourceURL: source.url, camera: source.camera)
    }
    private func source(_ url: URL) -> TextureSource {
        TextureSource(url: url, orientedImage: CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32)),
            camera: CameraMetadata(), pixelWidth: 32, pixelHeight: 32)
    }
    private func checkpoint(_ url: URL, sha: String) -> SelectedMaterialCheckpoint {
        SelectedMaterialCheckpoint(checkpointPath: url.path, sha256: sha, target: "height", pythonPath: "/python",
            workspacePath: url.deletingLastPathComponent().path, modelDirectory: "/encoder", codeDirectory: "/code")
    }
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("studio-workflow-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    private func finish(_ workspace: TextureWorkspace) async throws {
        for _ in 0..<6000 {
            if !workspace.isBusy { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("Studio operation did not finish")
    }
}
