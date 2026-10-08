import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialSelectionTests: XCTestCase {
    func testRetiredCheckpointCannotBeSelectedForProductionOrOverwriteRegistry() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let existing = selection(path: root.appendingPathComponent("original.pt").path, hash: "original")
        try Data("original checkpoint bytes".utf8).write(to: URL(fileURLWithPath: existing.checkpointPath))
        try existing.save(to: registry)
        let originalRegistry = try Data(contentsOf: registry)
        let name = "material-selection-test-\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: name)!
        defer { preferences.removePersistentDomain(forName: name) }
        let store = WorkbenchStore(preferences: preferences, managedWorkspaceURL: root,
            selectedCheckpointRegistryURL: registry)
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let checkpoint = try decoder.decode(WorkbenchCheckpoint.self, from: JSONSerialization.data(withJSONObject: [
            "checkpoint_path": "/models/brick/checkpoint.selected.pt", "sha256": "brick-sha",
            "schema": "texture-studio-material-training-cycle-v1", "target": "height", "step": 7600,
            "compatible": true, "variant": "frozen"
        ]))
        store.checkpoints = [checkpoint]; store.selectedCheckpointId = checkpoint.id
        store.useSelectedInStudio()
        XCTAssertEqual(store.error, MaterialTrainingPolicy.retirementNotice)
        XCTAssertFalse(checkpoint.supportsStudioInference)
        XCTAssertFalse(checkpoint.supportsTrainingWarmStart)
        XCTAssertEqual(try Data(contentsOf: registry), originalRegistry)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: existing.checkpointPath)), Data("original checkpoint bytes".utf8))
        XCTAssertEqual(store.checkpoints.first?.sha256, "brick-sha", "Legacy checkpoint stays available for review/export")
    }

    func testRestoringRetiredSelectionUsesFlatReliefWithoutSilentlySelectingDA3() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let selected = selection(path: "/trained/soil/model.pt", hash: "soil-checkpoint")
        try selected.save(to: registry)
        let workspace = TextureWorkspace(checkpointRegistryURL: registry)
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.activeHeightSourceLabel, "Flat surface")
        XCTAssertEqual(workspace.selectedMaterialCheckpoint?.sha256, selected.sha256, "Retirement keeps historical identity")
        XCTAssertEqual(workspace.heightSourceNotice, MaterialTrainingPolicy.retirementNotice)
        workspace.depthChoice = .materialCheckpoint
        XCTAssertEqual(workspace.depthChoice, .photoDetail, "Programmatic activation also follows production policy")
        XCTAssertEqual(try SelectedMaterialCheckpoint.read(from: registry).sha256, selected.sha256)
    }

    func testRetiredSelectionNotificationPreservesExplicitSourceAndAttachedMap() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let workspace = TextureWorkspace(checkpointRegistryURL: registry)
        workspace.depthChoice = .photoDetail
        let selected = selection(path: "/trained/brick.pt", hash: "brick")
        try selected.save(to: registry)
        NotificationCenter.default.post(name: SelectedMaterialCheckpoint.changeNotification, object: nil)
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.heightSourceNotice, MaterialTrainingPolicy.retirementNotice)
        workspace.attachedDepth = try TextureDepth(width: 2, height: 2, values: [0.2, 0.4, 0.6, 0.8],
            sourceLabel: "User height map", interpretation: .surfaceHeight)
        workspace.depthChoice = .attached
        NotificationCenter.default.post(name: SelectedMaterialCheckpoint.changeNotification, object: nil)
        XCTAssertEqual(workspace.depthChoice, .attached)
        workspace.depthChoice = .materialCheckpoint
        XCTAssertEqual(workspace.depthChoice, .attached, "The user's existing height map is retained")
        XCTAssertEqual(workspace.attachedDepth?.sourceLabel, "User height map")
        workspace.depthChoice = .model
        workspace.reloadSelectedCheckpoint(activate: true)
        XCTAssertEqual(workspace.depthChoice, .model, "An explicit alternative remains the user's choice")
    }

    func testSavedDINOChoiceMigratesToFlatAndRemembersTheSafeSelection() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "retired-studio-settings-\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        var settings = TextureSettings(); settings.outputSize = 2048; settings.lightingStrength = 0.31
        StudioPreferences(settings: settings, depthChoice: .materialCheckpoint).save(to: preferences)
        let workspace = TextureWorkspace(checkpointRegistryURL: root.appendingPathComponent("missing.json"), preferences: preferences)
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.settings, settings)
        XCTAssertEqual(StudioPreferences.load(from: preferences).depthChoice, .photoDetail)
        XCTAssertEqual(workspace.heightSourceNotice, MaterialTrainingPolicy.retirementNotice)
        XCTAssertFalse(DepthChoice.studioChoices.contains(.materialCheckpoint))
    }

    func testRetiredRecipeOpensWithoutRuntimeAndPreservesOriginalDocument() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let photo = root.appendingPathComponent("photo.png")
        try CIContext().writePNGRepresentation(of: CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32)),
            to: photo, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let model = selection(path: "/trained/model-a.pt", hash: "model-a")
        var recipe = TextureRecipe(photoPath: photo.path, depthChoice: .materialCheckpoint,
            modelID: LocalModelDescriptor.da3GiantID, customInverseDepth: true, settings: TextureSettings())
        recipe.materialCheckpoint = MaterialCheckpointIdentity(model)
        let encoded = try JSONEncoder().encode(recipe)
        XCTAssertFalse(String(decoding: encoded, as: UTF8.self).contains("pythonPath"))
        let recipeURL = root.appendingPathComponent("recipe.json")
        try encoded.write(to: recipeURL)
        let workspace = TextureWorkspace(checkpointRegistryURL: root.appendingPathComponent("missing.json"))
        workspace.openRecipe(recipeURL)
        for _ in 0..<1000 {
            if !workspace.isBusy { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertFalse(workspace.isBusy)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(workspace.source?.url, photo)
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.heightSourceNotice, MaterialTrainingPolicy.retirementNotice)
        XCTAssertEqual(try Data(contentsOf: recipeURL), encoded, "Opening an old recipe does not rewrite or delete historical metadata")
        XCTAssertNil(try workspace.makeRecipe().materialCheckpoint, "New recipes cannot reactivate the retired architecture")
        XCTAssertThrowsError(try workspace.materialCheckpointProvenance())
    }

    func testProductionCheckpointServiceRejectsRetiredModelBeforeLaunchingProcess() async throws {
        let source = TextureSource(url: URL(fileURLWithPath: "/fixture.png"),
            orientedImage: CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32)),
            camera: CameraMetadata(), pixelWidth: 32, pixelHeight: 32)
        do {
            _ = try await MaterialCheckpointService().predict(source: source,
                checkpoint: selection(path: "/nonexistent/model.pt", hash: "legacy"), size: 1024)
            XCTFail("Retired production inference should be rejected")
        } catch {
            XCTAssertEqual(error.localizedDescription, MaterialTrainingPolicy.retirementNotice,
                "Policy must reject before checking files or running the legacy Python backend")
        }
    }

    func testAttachedHeightPreservesItsNumericAmplitudeThroughStudio() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let context = CIContext(options: [.workingFormat: CIFormat.RGBAf,
            .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        for amplitude: Float in [0.25, 1.125] {
            let workspace = TextureWorkspace(checkpointRegistryURL: root.appendingPathComponent("missing.json"))
            workspace.source = TextureSource(url: root.appendingPathComponent("photo.png"),
                orientedImage: CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32)),
                camera: CameraMetadata(), pixelWidth: 32, pixelHeight: 32)
            workspace.attachedDepth = try TextureDepth(width: 32, height: 32,
                values: [Float](repeating: amplitude, count: 32 * 32),
                sourceLabel: "User numeric height", interpretation: .surfaceHeight)
            workspace.settings.attachedMapIsHeight = true
            workspace.depthChoice = .attached
            workspace.updatePreview(models: ModelManager())
            for _ in 0..<3000 {
                if !workspace.isBusy { break }
                try await Task.sleep(for: .milliseconds(5))
            }
            XCTAssertFalse(workspace.isBusy)
            XCTAssertNil(workspace.notice)
            let material = try XCTUnwrap(workspace.result)
            var actual: Float = 0
            withUnsafeMutableBytes(of: &actual) {
                context.render(material.height, toBitmap: $0.baseAddress!, rowBytes: 4,
                    bounds: CGRect(x: 512, y: 512, width: 1, height: 1), format: .Rf, colorSpace: nil)
            }
            XCTAssertEqual(actual, amplitude, accuracy: 0.00001,
                "User height data must retain its range; camera-depth normalization would replace a constant map with 0.5")
        }
    }

    private func selection(path: String, hash: String) -> SelectedMaterialCheckpoint {
        SelectedMaterialCheckpoint(checkpointPath: path, sha256: hash, target: "height", pythonPath: "/python",
            workspacePath: "/workspace", modelDirectory: "/encoder", codeDirectory: "/code")
    }
    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("selection-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
