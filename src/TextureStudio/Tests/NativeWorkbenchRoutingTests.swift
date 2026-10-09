import Foundation
import XCTest
@testable import TextureStudio

@MainActor final class NativeWorkbenchRoutingTests: XCTestCase {
    func testDatasetAndCapabilityWorkflowsRunNatively() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "org.ipde.native-routing.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root)
        let capabilities = try WorkbenchResult.decode(WorkbenchTrainingCapabilities.self,
            output: await store.worker(["capabilities", "--scope", "final-map"]))
        XCTAssertEqual(capabilities.trainingSizes, [256, 512, 1024, 2048, 4096])
        let datasetURL = root.appendingPathComponent("test-dataset")
        let created = try WorkbenchResult.decode(WorkbenchDataset.self, output: await store.worker([
            "create-dataset", "--dataset", datasetURL.path, "--name", "Native dataset", "--description", "Original samples stay untouched",
            "--training-size", "1024"
        ]))
        XCTAssertEqual(created.name, "Native dataset")
        try await store.loadDataset(datasetURL)
        XCTAssertEqual(store.dataset?.datasetPath, datasetURL.path)
        XCTAssertEqual(store.dataset?.name, "Native dataset")
    }

    func testCheckpointLibraryInspectionRunsNatively() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration: [String: Any] = ["schema": "texture-studio-material-checkpoint-v1", "architecture": "pbrnxt-native-v1",
            "target": "height", "step": 7, "training_size": 1024, "image_padding": false, "image_resizing": false,
            "base": ["sha256": String(repeating: "a", count: 64)]]
        let configurationJSON = String(decoding: try JSONSerialization.data(withJSONObject: configuration), as: UTF8.self)
        let header = try JSONSerialization.data(withJSONObject: ["__metadata__": ["configuration": configurationJSON],
            "weight": ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]]])
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header); file.append(contentsOf: [0, 0, 128, 63])
        let checkpointURL = root.appendingPathComponent("model.safetensors")
        try file.write(to: checkpointURL)
        let suite = "org.ipde.native-checkpoint-routing.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root)
        let result = try WorkbenchResult.decode(WorkbenchCheckpoint.self,
            output: await store.worker(["checkpoint", "--checkpoint", checkpointURL.path]))
        XCTAssertEqual(result.checkpointPath, checkpointURL.path)
        XCTAssertEqual(result.step, 7)
        XCTAssertEqual(result.target, "height")
    }
}
