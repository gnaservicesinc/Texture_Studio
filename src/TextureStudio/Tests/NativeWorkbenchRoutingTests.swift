import Foundation
import XCTest
@testable import TextureStudio

@MainActor final class NativeWorkbenchRoutingTests: XCTestCase {
    func testNativeEventLogBatchesBurstWithoutLosingDiskOrProgressChunks() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".log")
        defer { try? FileManager.default.removeItem(at: url) }
        let log = NativeWorkbenchLog(url: url)
        let chunks = (0..<5000).map { "{\"event\":\"operation_progress\",\"completed\":\($0),\"operation\":\"Backward pass 🪵\"}\n" }
        let expected = chunks.joined()
        var reservations = 0
        for chunk in chunks { if log.append(chunk) { reservations += 1 } }
        XCTAssertEqual(reservations, 1, "A burst reserves one main-actor delivery")
        XCTAssertEqual(log.drainPending(), expected, "Every event is delivered, including those older than the display tail")
        XCTAssertEqual(log.drainPending(), "", "A completed drain cannot replay events")
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), expected, "The disk log retains the full event stream")
        XCTAssertLessThanOrEqual(log.text.utf8.count, 100000)
        XCTAssertTrue(expected.hasSuffix(log.text))
        XCTAssertFalse(log.text.contains("�"))
    }

    func testNativeEventLogFinalDrainResetsReservationWithoutReplayingTail() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".log")
        defer { try? FileManager.default.removeItem(at: url) }
        let log = NativeWorkbenchLog(url: url, displayByteLimit: 64)
        XCTAssertFalse(log.append(""))
        XCTAssertEqual(log.drainPending(), "")
        let first = "{\"event\":\"update\",\"completed_updates\":1}\n"
        let final = "{\"event\":\"training_completed\",\"status\":\"completed\"}\n"
        XCTAssertTrue(log.append(first))
        let firstDelivery = log.drainPending()
        XCTAssertTrue(log.append(final), "New events reserve another delivery after a drain")
        let finalDelivery = log.drainPending()
        XCTAssertEqual(firstDelivery + finalDelivery, first + final)
        XCTAssertEqual(finalDelivery, final, "The final flush only receives undelivered events")
        XCTAssertEqual(log.drainPending(), "", "A delayed task after final flush has nothing to replay")
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), first + final)
        XCTAssertTrue((first + final).hasSuffix(log.text))
    }

    func testNativeEventLogUnicodeTailRemainsValidAcrossWrapAndLargeChunks() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".log")
        defer { try? FileManager.default.removeItem(at: url) }
        let log = NativeWorkbenchLog(url: url, displayByteLimit: 7)
        let chunks = ["abc🪵", "✓", "😀😀🪵", "ab", "✓\n"]
        var all = ""
        for chunk in chunks {
            log.append(chunk); all += chunk
            XCTAssertLessThanOrEqual(log.text.utf8.count, 7)
            XCTAssertTrue(all.hasSuffix(log.text))
            XCTAssertFalse(log.text.contains("�"))
        }
        XCTAssertEqual(log.text, "ab✓\n")
        XCTAssertEqual(log.drainPending(), chunks.joined())
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), chunks.joined())
        XCTAssertTrue(log.append("last"), "Delivery reservation resets even after a wrapped display tail")
        XCTAssertEqual(log.drainPending(), "last")
        XCTAssertEqual(log.drainPending(), "")
    }

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
            "target": "height", "model_name": "Stone / Displacement", "step": 7, "training_size": 1024, "image_padding": false, "image_resizing": false,
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
        XCTAssertEqual(result.modelName, "Stone / Displacement")
        XCTAssertEqual(result.title, "Stone / Displacement · model.safetensors")
        store.checkpoints = [result]
        store.selectedCheckpointId = result.id
        store.training.useWarmStart = true
        XCTAssertEqual(store.effectiveTrainingModelName, "Stone / Displacement")
        store.training.modelName = "New refinement"
        XCTAssertEqual(store.effectiveTrainingModelName, "New refinement")
    }
}
