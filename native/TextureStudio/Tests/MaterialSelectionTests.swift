import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialSelectionTests: XCTestCase {
    func testUsingCheckpointSavesItsIdentityAndArchitectureForStudio() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
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
        XCTAssertNil(store.error)
        let selected = try SelectedMaterialCheckpoint.read(from: registry)
        XCTAssertEqual(selected.sha256, checkpoint.sha256)
        XCTAssertEqual(selected.title, checkpoint.title)
        XCTAssertEqual(selected.modelSummary, checkpoint.modelSummary)
        XCTAssertTrue(selected.modelSummary!.contains("DINOv2"))
        XCTAssertFalse(selected.modelSummary!.contains("DA3"))
        XCTAssertFalse(store.activity.contains("Choose Material checkpoint"))
    }

    func testSelectionNotificationSwitchesAnOpenStudioAndClearsOldDepth() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let workspace = TextureWorkspace(checkpointRegistryURL: registry)
        workspace.generatedDepth = try TextureDepth(width: 2, height: 2, values: [0, 1, 2, 3], sourceLabel: "DA3")
        XCTAssertEqual(workspace.depthChoice, .model)
        let selected = selection(path: "/trained/soil/model.pt", hash: "soil-checkpoint")
        try selected.save(to: registry)
        NotificationCenter.default.post(name: SelectedMaterialCheckpoint.changeNotification, object: nil)
        XCTAssertEqual(workspace.depthChoice, .materialCheckpoint)
        XCTAssertEqual(workspace.selectedMaterialCheckpoint?.sha256, selected.sha256)
        XCTAssertEqual(workspace.activeHeightSourceLabel, selected.title)
        XCTAssertNil(workspace.generatedDepth)
        XCTAssertTrue(workspace.hasEdits)
        // An explicit repeat selection must also override a manually chosen DA3.
        workspace.depthChoice = .model
        NotificationCenter.default.post(name: SelectedMaterialCheckpoint.changeNotification, object: nil)
        XCTAssertEqual(workspace.depthChoice, .materialCheckpoint)
    }

    func testReactivationNoticesDifferentModelButKeepsExplicitAlternativeForSameModel() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        try selection(path: "/trained/old.pt", hash: "old").save(to: registry)
        let workspace = TextureWorkspace(checkpointRegistryURL: registry)
        XCTAssertEqual(workspace.depthChoice, .materialCheckpoint)
        workspace.depthChoice = .model
        workspace.reloadSelectedCheckpoint()
        XCTAssertEqual(workspace.depthChoice, .model)
        let new = selection(path: "/trained/new.pt", hash: "new")
        try new.save(to: registry)
        workspace.reloadSelectedCheckpoint()
        XCTAssertEqual(workspace.depthChoice, .materialCheckpoint)
        XCTAssertEqual(workspace.selectedMaterialCheckpoint?.sha256, new.sha256)
    }

    func testSelectionDuringImportAppliesAfterImportFinishes() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let photo = root.appendingPathComponent("photo.png")
        try CIContext().writePNGRepresentation(of: CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32)),
            to: photo, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let workspace = TextureWorkspace(checkpointRegistryURL: registry)
        workspace.importPhoto(photo)
        XCTAssertTrue(workspace.isBusy)
        let selected = selection(path: "/trained/brick.pt", hash: "brick")
        try selected.save(to: registry)
        NotificationCenter.default.post(name: SelectedMaterialCheckpoint.changeNotification, object: nil)
        for _ in 0..<1000 {
            if !workspace.isBusy { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertFalse(workspace.isBusy)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(workspace.source?.url, photo)
        XCTAssertEqual(workspace.depthChoice, .materialCheckpoint)
        XCTAssertEqual(workspace.selectedMaterialCheckpoint?.sha256, selected.sha256)
    }

    func testRecipePinsModelIdentityAndRestoresSourcePreviewOnModelChange() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let photo = root.appendingPathComponent("photo.png")
        let image = CIImage(color: CIColor(red: 0.2, green: 0.3, blue: 0.4))
            .cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32))
        try CIContext().writePNGRepresentation(of: image, to: photo, format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let modelA = selection(path: "/trained/model-a.pt", hash: "model-a")
        let modelB = SelectedMaterialCheckpoint(checkpointPath: "/trained/model-b.pt", sha256: "model-b", target: "height",
            pythonPath: "/trusted-local/python", workspacePath: "/trusted-local/workspace",
            modelDirectory: "/trusted-local/encoder", codeDirectory: "/trusted-local/code")
        try modelB.save(to: registry)
        var recipe = TextureRecipe(photoPath: photo.path, depthChoice: .materialCheckpoint,
            modelID: LocalModelDescriptor.da3GiantID, customInverseDepth: true, settings: TextureSettings())
        recipe.materialCheckpoint = MaterialCheckpointIdentity(modelA)
        let recipeURL = root.appendingPathComponent("recipe.json")
        let encoded = try JSONEncoder().encode(recipe)
        XCTAssertFalse(String(decoding: encoded, as: UTF8.self).contains("pythonPath"), "An imported recipe must not select an executable")
        try encoded.write(to: recipeURL)
        let workspace = TextureWorkspace(checkpointRegistryURL: registry)
        workspace.openRecipe(recipeURL)
        for _ in 0..<1000 {
            if !workspace.isBusy { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(workspace.selectedMaterialCheckpoint?.sha256, "model-a")
        XCTAssertEqual(workspace.selectedMaterialCheckpoint?.pythonPath, modelB.pythonPath)
        workspace.reloadSelectedCheckpoint()
        XCTAssertEqual(workspace.selectedMaterialCheckpoint?.sha256, "model-a", "Reactivation must retain the recipe's pinned model")
        XCTAssertEqual(try workspace.makeRecipe().materialCheckpoint?.sha256, "model-a")
        let metadata = try XCTUnwrap(JSONSerialization.jsonObject(with: workspace.materialCheckpointProvenance()) as? [String: Any])
        XCTAssertEqual(metadata["checkpoint_sha256"] as? String, "model-a")
        XCTAssertEqual(metadata["base_encoder"] as? String, "facebook/dinov2-base")
        let original = try XCTUnwrap(workspace.preview)
        workspace.preview = CIContext().createCGImage(CIImage(color: .white).cropped(to: image.extent), from: image.extent)
        workspace.renderedPreview = .height
        workspace.reloadSelectedCheckpoint(activate: true)
        XCTAssertEqual(workspace.selectedMaterialCheckpoint?.sha256, "model-b")
        XCTAssertEqual(workspace.renderedPreview, .source)
        XCTAssertTrue(workspace.preview === original, "A new model must never leave the old model's map displayed as the source")
    }

    func testDuplicateProcessNotificationDoesNotInvalidateTwice() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        try selection(path: "/trained/model.pt", hash: "model").save(to: registry)
        let workspace = TextureWorkspace(checkpointRegistryURL: registry)
        let information = ["selectionID": UUID().uuidString]
        NotificationCenter.default.post(name: SelectedMaterialCheckpoint.changeNotification, object: nil, userInfo: information)
        workspace.depthChoice = .model
        NotificationCenter.default.post(name: SelectedMaterialCheckpoint.changeNotification, object: nil, userInfo: information)
        XCTAssertEqual(workspace.depthChoice, .model, "Local and distributed copies of one selection event are handled once")
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
