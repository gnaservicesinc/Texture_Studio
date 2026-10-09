import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialSelectionTests: XCTestCase {
    func testNativeAdapterRetainsItsExactCustomBaseAcrossReopenAndStudioSelection() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let suite = "material-base-selection-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let fullURL = root.appendingPathComponent("full.safetensors")
        let adapterURL = root.appendingPathComponent("adapter.safetensors")
        let fullSHA = String(repeating: "a", count: 64)
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, selectedCheckpointRegistryURL: registry,
            workerOverride: { args, _ in
                let path = args[try XCTUnwrap(args.firstIndex(of: "--checkpoint")) + 1]
                let full = path == fullURL.path
                return "{\"checkpoint_path\":\"\(path)\",\"sha256\":\"\(full ? fullSHA : String(repeating: "b", count: 64))\",\"schema\":\"texture-studio-material-\(full ? "checkpoint" : "lora")-v1\",\"target\":\"height\",\"step\":1,\"compatible\":true,\"variant\":\"\(full ? "full" : "lora")\",\"base\":{\"sha256\":\"\(fullSHA)\"}}"
            })
        try await store.loadCheckpoint(fullURL)
        try await store.loadCheckpoint(adapterURL)
        let adapter = try XCTUnwrap(store.selectedCheckpoint)
        XCTAssertEqual(store.dependencyArguments(for: adapter), ["--model-directory", fullURL.path])
        store.useSelectedInStudio()
        store.workspacePath = root.appendingPathComponent("new-workspace").path
        store.saveConfiguration()
        XCTAssertEqual(try SelectedMaterialCheckpoint.read(from: registry).modelDirectory, fullURL.path)
        let reopened = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root)
        XCTAssertEqual(reopened.baseDirectory(for: adapter), fullURL.path)
    }

    func testMaterialCheckpointActivatesAndPinsRecipeIdentity() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let selected = selection(path: root.appendingPathComponent("model.safetensors").path, hash: "exact-model")
        try selected.save(to: registry)
        let workspace = TextureWorkspace(checkpointRegistryURL: registry)
        XCTAssertEqual(workspace.depthChoice, .materialCheckpoint)
        workspace.source = TextureSource(url: root.appendingPathComponent("surface.png"),
            orientedImage: CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32)),
            camera: CameraMetadata(), pixelWidth: 32, pixelHeight: 32)
        let recipe = try workspace.makeRecipe()
        XCTAssertEqual(recipe.materialCheckpoint?.sha256, "exact-model")
        XCTAssertEqual(workspace.activeHeightSourceLabel, selected.title)
    }

    func testSelectingFullCheckpointWritesExactRuntimeAndIdentity() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let suite = "material-selection-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, selectedCheckpointRegistryURL: registry)
        let model = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
        {"checkpoint_path":"/models/soil/model.safetensors","sha256":"soil-sha","schema":"texture-studio-material-checkpoint-v1","target":"height","step":120,"compatible":true,"variant":"full","supports_training_warm_start":true}
        """)
        store.checkpoints = [model]; store.selectedCheckpointId = model.id
        store.useSelectedInStudio()
        XCTAssertNil(store.error)
        let saved = try SelectedMaterialCheckpoint.read(from: registry)
        XCTAssertEqual(saved.sha256, "soil-sha")
        XCTAssertTrue(model.supportsStudioInference)
        XCTAssertTrue(model.supportsTrainingWarmStart)
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
        SelectedMaterialCheckpoint(checkpointPath: path, sha256: hash, target: "height",
            workspacePath: "/workspace", modelDirectory: "/encoder")
    }
    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("selection-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
