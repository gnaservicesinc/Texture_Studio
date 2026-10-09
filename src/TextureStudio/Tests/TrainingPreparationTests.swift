import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class TrainingPreparationTests: XCTestCase {
    func testPreparationUsesSupportedNativeCropGrid() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        XCTAssertEqual(store.supportedTrainingSizes, [512, 1024])
        store.selectTrainingSize(1024)
        try await settled(store)
        XCTAssertFalse(fixture.calls.contains { $0.first == "prepare-size" }, "Choosing a preview grid cannot allocate a training dataset")
        store.prepareTrainingDataset()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertTrue(store.dataset?.hasNativeSize(1024) == true)
        XCTAssertEqual(fixture.calls.last?.first, "prepare-size")
        XCTAssertEqual(value("--size", in: fixture.calls.last!), "1024")
        XCTAssertEqual(value("--expected-index-sha256", in: fixture.calls.last!), "source-sha")
        XCTAssertEqual(try Data(contentsOf: fixture.original.appendingPathComponent("dataset.json")), Data("original".utf8))
        let count = fixture.calls.count
        store.selectTrainingSize(2048)
        XCTAssertNotNil(store.error)
        XCTAssertEqual(fixture.calls.count, count, "An unsupported size never allocates training images")
    }

    func testSelectedCenterCropNeedsNoHeldOutMaterialToBeReady() throws {
        let document = """
        {"dataset_path":"/stage","index_sha256":"bound","materials":[{"material_id":"soil_4k",
        "samples":[{"sample_id":"soil_4k_center","status":"approved","split":"train","width":2048,"height":2048,
        "maps":{"input":{"path":"/crop/diffuse.png","width":2048,"height":2048},
        "height":{"path":"/crop/height.png","width":2048,"height":2048}}}]}],
        "automatic_validation":{"policy":"subject-extra-crops-v2","material_ids":[],
        "quick_fit_material_id":"soil_4k","target":"height"}}
        """.replacingOccurrences(of: "\n", with: "")
        let dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: document)
        XCTAssertTrue(dataset.readyForTraining(size: 2048, material: "soil_4k", target: "height"))
        XCTAssertFalse(dataset.readyForTraining(size: 2048, material: nil, target: "height"))
        XCTAssertTrue(dataset.readyForTraining(size: 2048, material: "soil_4k", target: "normal"), "Prepared crops are shared across targets")
        XCTAssertFalse(dataset.readyForTraining(size: 2048, material: "other", target: "height"))
    }

    func testMismatchedMapGridCannotBecomeTrainingInput() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.wrongGrid = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.selectTrainingSize(1024)
        try await settled(store)
        store.prepareTrainingDataset()
        try await settled(store)
        XCTAssertNotNil(store.error)
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
        XCTAssertFalse(fixture.calls.contains { $0.first == "train" })
    }

    func testTrainingDispatchesExactGridThenPurgesStagingAndReconnectsSources() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let oldDeveloperMode = StudioPreferences.defaults.object(forKey: StudioPreferences.developerModeKey)
        StudioPreferences.defaults.set(false, forKey: StudioPreferences.developerModeKey)
        defer { StudioPreferences.defaults.set(oldDeveloperMode, forKey: StudioPreferences.developerModeKey) }
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        try await settled(store)
        XCTAssertNil(store.error)
        let train = try XCTUnwrap(fixture.calls.first { $0.first == "train" })
        XCTAssertTrue(train.contains("--whole-maps"))
        XCTAssertEqual(value("--size", in: train), "1024")
        XCTAssertEqual(value("--dataset", in: train), fixture.prepared.path)
        XCTAssertFalse(train.contains("--developer-mode"))
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
        XCTAssertEqual(store.selectedCheckpoint?.url.pathExtension, "safetensors")
        XCTAssertTrue(store.selectedCheckpoint?.supportsTrainingWarmStart == true)
    }

    func testStopAndSaveIsEnabledOnlyAfterTrainingStartsAndAllowsFinalAdapterResult() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.canStopAndSave)
        store.recordTrainingProgress("{\"event\":\"training_")
        XCTAssertFalse(store.canStopAndSave, "Partial log chunks cannot enable saving")
        store.recordTrainingProgress("started\"}\n")
        XCTAssertTrue(store.canStopAndSave)
        store.stopAndSave()
        XCTAssertTrue(store.isStopping)
        XCTAssertTrue(store.isSavingTraining)
        fixture.continuation?.resume()
        fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertNotNil(store.selectedCheckpoint)
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
    }

    func testCheckpointRequestKeepsTrainingActiveAndRegistersSavedModel() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.uploadAfterTraining = false
        store.training.size = 1024
        store.startTraining()
        for _ in 0..<200 {
            if fixture.continuation != nil { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertNotNil(fixture.continuation)
        store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
        store.saveCheckpointNow()
        XCTAssertTrue(store.isCheckpointPending)
        XCTAssertFalse(store.isStopping)
        store.recordTrainingProgress("{\"event\":\"validation\",\"scope\":\"full\",\"sample_count\":5,\"pool_count\":5,\"mae\":0.025}\n")
        let event: [String: Any] = ["event": "checkpoint_saved", "checkpoint_path": "/tmp/step-2.safetensors",
            "sha256": "checkpoint-proof", "schema": "texture-studio-material-lora-v1", "target": "height",
            "step": 2, "compatible": true, "variant": "lora", "supports_training_warm_start": true]
        store.recordTrainingProgress(String(decoding: try JSONSerialization.data(withJSONObject: event), as: UTF8.self) + "\n")
        XCTAssertFalse(store.isCheckpointPending)
        XCTAssertTrue(store.isTraining)
        XCTAssertTrue(store.validationSummary.contains("5/5"))
        XCTAssertEqual(store.checkpoints.last?.step, 2)
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
    }

    func testStopDuringDatasetPreparationPreventsTrainingFromLaunching() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdPreparation = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.canStopAndSave)
        store.stop()
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertFalse(fixture.calls.contains { $0.first == "train" })
        XCTAssertNil(store.selectedCheckpoint)
        XCTAssertEqual(try Data(contentsOf: fixture.original.appendingPathComponent("dataset.json")), Data("original".utf8))
    }

    func testPreparationReportsParallelProgressAndIgnoresEventsAfterStopping() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdPreparation = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.prepareTrainingDataset()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        let initialActivity = store.activity
        store.recordTrainingProgress("{\"event\":\"preparation_progress\",\"completed\":2,")
        XCTAssertEqual(store.activity, initialActivity)
        store.recordTrainingProgress("\"total\":8,\"worker_count\":4,\"training_size\":1024}\n")
        XCTAssertTrue(store.activity.contains("2/8 materials"))
        XCTAssertTrue(store.activity.contains("4 workers"))
        XCTAssertFalse(store.hasTrainingStarted)
        store.stop()
        let stoppedActivity = store.activity
        store.recordTrainingProgress("{\"event\":\"preparation_completed\",\"completed\":8,\"total\":8,\"worker_count\":4,\"training_size\":1024}\n")
        XCTAssertEqual(store.activity, stoppedActivity)
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
    }

    func testSaveRequestDuringModelSetupAbortsWithoutLoadingAnAdapter() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        store.stopAndSave()
        XCTAssertTrue(store.isStopping)
        XCTAssertFalse(store.isSavingTraining)
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertNil(store.selectedCheckpoint)
        XCTAssertFalse(fixture.calls.contains { $0.first == "checkpoint" })
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
    }

    func testStopCanAbortPendingSaveWithoutLoadingOrUploadingAnAdapter() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
        store.stopAndSave()
        XCTAssertTrue(store.isSavingTraining)
        store.stop()
        XCTAssertFalse(store.isSavingTraining)
        store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
        XCTAssertFalse(store.canStopAndSave, "Delayed worker logs cannot reenable a cancelled run")
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertNil(store.selectedCheckpoint)
        XCTAssertFalse(fixture.calls.contains { $0.first == "checkpoint" || $0.first == "upload-selected" })
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
    }

    func testMissingOnlyRemovalNeverRemovesExistingSource() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        try await store.loadDataset(fixture.original)
        let source = fixture.root.appendingPathComponent("source.png")
        try Data("original image".utf8).write(to: source)
        let count = fixture.calls.count
        store.removeMissingSource(source, sampleID: "soil")
        XCTAssertEqual(fixture.calls.count, count)
        try FileManager.default.removeItem(at: source)
        store.removeMissingSource(source, sampleID: "soil")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.calls.suffix(2).compactMap(\.first), ["remove-missing", "dataset"])
    }

    private func value(_ key: String, in args: [String]) -> String? {
        guard let i = args.firstIndex(of: key), args.indices.contains(i + 1) else { return nil }
        return args[i + 1]
    }
    private func settled(_ store: WorkbenchStore) async throws {
        for _ in 0..<1000 {
            if !store.isBusy { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("Worker did not finish")
    }
    @MainActor private final class Fixture {
        let root: URL
        let original: URL
        let prepared: URL
        let defaults: UserDefaults
        let suite = "training-grid-\(UUID().uuidString)"
        var calls: [[String]] = []
        var wrongGrid = false
        var holdTraining = false
        var holdPreparation = false
        var continuation: CheckedContinuation<Void, Never>?
        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("training-grid-\(UUID().uuidString)")
            original = root.appendingPathComponent("original")
            prepared = root.appendingPathComponent(".training-data/grid-1024")
            defaults = UserDefaults(suiteName: suite)!
            try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
            try Data("original".utf8).write(to: original.appendingPathComponent("dataset.json"))
            defaults.set(root.path, forKey: "workspace")
        }
        func remove() { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        func json(_ value: [String: Any]) throws -> String {
            String(decoding: try JSONSerialization.data(withJSONObject: value), as: UTF8.self)
        }
        func dataset(prepared isPrepared: Bool) throws -> String {
            let size = isPrepared ? 1024 : 2048
            let maps: [String: Any] = ["input": ["path": root.appendingPathComponent("diffuse.png").path, "width": size, "height": size],
                "height": ["path": root.appendingPathComponent("height.png").path, "width": wrongGrid && isPrepared ? 512 : size, "height": size]]
            var value: [String: Any] = ["dataset_path": isPrepared ? prepared.path : original.path,
                "index_sha256": isPrepared ? "prepared-sha" : "source-sha", "supported_training_sizes": [512, 1024, 2048],
                "automatic_validation": ["policy": "subject-extra-crops-v2", "material_ids": ["soil"]],
                "materials": [["material_id": "soil", "samples": [["sample_id": "soil", "status": "approved", "split": "train", "width": size, "height": size, "maps": maps]]]]]
            if isPrepared { value["preparation"] = ["source_dataset_path": original.path, "source_index_sha256": "source-sha",
                "prepared_dataset_path": prepared.path, "crop_size": size, "reused": false, "target_resized": false, "target_cropped": true,
                "original_dataset_modified": false] }
            return try json(value)
        }
        func store() -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { args, script in
                self.calls.append(args)
                switch args.first {
                case "capabilities":
                    XCTAssertEqual(script, args.first)
                    return "{\"training_sizes\":[512,1024]}"
                case "dataset", "edit-dataset": return try self.dataset(prepared: false)
                case "prepare-size":
                    if self.holdPreparation { await withCheckedContinuation { self.continuation = $0 } }
                    return try self.dataset(prepared: true)
                case "train":
                    if self.holdTraining { await withCheckedContinuation { self.continuation = $0 } }
                    return try self.json(["checkpoint_path": self.root.appendingPathComponent("adapter.safetensors").path, "package_path": self.root.path])
                case "checkpoint": return try self.json(["checkpoint_path": self.root.appendingPathComponent("adapter.safetensors").path,
                    "sha256": "exact", "schema": "texture-studio-material-lora-v1", "target": "height", "step": 3,
                    "compatible": true, "variant": "lora", "supports_training_warm_start": true])
                case "cleanup-size": return try self.json(["dataset_path": self.prepared.path, "source_dataset_path": self.original.path, "removed": true])
                case "remove-missing": return "{\"removed\":true}"
                default: throw StudioError("Unexpected worker command")
                }
            })
        }
    }
}
