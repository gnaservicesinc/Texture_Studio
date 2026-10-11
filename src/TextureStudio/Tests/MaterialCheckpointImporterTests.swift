import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialCheckpointImporterTests: XCTestCase {
    func testStandaloneScalarAndNormalModelsActivateFromFilesAndExportFoldersWithoutBaseWeights() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "material-model-import-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let registry = root.appendingPathComponent("selected.json")
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, selectedCheckpointRegistryURL: registry)
        store.modelDirectory = root.appendingPathComponent("no-pbr-base").path

        for target in ["height", "roughness", "normal"] {
            let package = root.appendingPathComponent(target)
            try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
            let model = try NativeCompactMaterialModel(target: target, width: 16, seed: 29)
            let configuration = try model.checkpointConfiguration(size: 512, step: 6425)
            let metadata = String(decoding: try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys]), as: UTF8.self)
            let file = package.appendingPathComponent("model.safetensors")
            try NativeSafetensors.write(tensors: model.weights, metadata: ["configuration": metadata], to: file)
            let digest = try NativeMaterialTransfer.hash(file)
            let imported = try await MaterialCheckpointImporter.activate(target == "height" ? file : package, using: store)
            let saved = try SelectedMaterialCheckpoint.read(from: SelectedMaterialCheckpoint.registryURL(for: target, heightRegistryURL: registry))
            XCTAssertEqual(imported.modelFamily, target == "normal" ? .compactNormal : .compactScalar)
            XCTAssertEqual(imported.step, 6425)
            XCTAssertEqual(imported.checkpointPath, file.path)
            XCTAssertEqual(saved.target, target)
            XCTAssertEqual(saved.sha256, digest)
            XCTAssertEqual(saved.checkpointPath, file.path)
            XCTAssertTrue(saved.supportsStudioInference)
            XCTAssertEqual(try NativeMaterialTransfer.hash(file), digest, "Import must retain the original model bytes.")
            XCTAssertEqual(store.dependencyArguments(for: imported), [])
            XCTAssertFalse(FileManager.default.fileExists(atPath: store.modelDirectory))
        }
        XCTAssertEqual(Set(SelectedMaterialCheckpoint.readAll(heightRegistryURL: registry).keys), ["height", "roughness", "normal"])
    }

    func testInvalidWeightsOrSchemaCannotReplacePreviouslySelectedModel() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "material-model-invalid-import-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let registry = root.appendingPathComponent("selected.json")
        let previous = SelectedMaterialCheckpoint(checkpointPath: root.appendingPathComponent("previous.safetensors").path,
            sha256: String(repeating: "a", count: 64), target: "height", workspacePath: root.path, modelDirectory: "")
        try previous.save(to: registry)
        let originalRegistry = try Data(contentsOf: registry)
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, selectedCheckpointRegistryURL: registry)
        let model = try NativeCompactMaterialModel(target: "height", width: 16)
        let configuration = try model.checkpointConfiguration(size: 512, step: 1)
        let invalid = root.appendingPathComponent("invalid.safetensors")
        for badSchema in [false, true] {
            var recorded = configuration
            if badSchema { recorded["schema"] = "unknown-material-model" }
            var weights = model.weights
            if !badSchema { weights.removeValue(forKey: "input.weight") }
            let metadata = String(decoding: try JSONSerialization.data(withJSONObject: recorded, options: [.sortedKeys]), as: UTF8.self)
            try NativeSafetensors.write(tensors: weights, metadata: ["configuration": metadata], to: invalid)
            do {
                try await MaterialCheckpointImporter.activate(invalid, using: store)
                XCTFail("Invalid native material model was activated.")
            } catch {
                XCTAssertEqual(try Data(contentsOf: registry), originalRegistry)
                XCTAssertNil(store.selectedCheckpoint)
                XCTAssertTrue(store.checkpoints.isEmpty)
            }
        }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("material-import-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
