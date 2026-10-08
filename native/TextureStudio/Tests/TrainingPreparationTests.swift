import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class TrainingPreparationTests: XCTestCase {
    func testChangingSizePreparesAndSwitchingBackReopensOriginalWithoutChangingRequestedSize() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in
            recorder.arguments.append(args)
            if args.first == "prepare-size" {
                let size = Int(args[try XCTUnwrap(args.firstIndex(of: "--size")) + 1])!
                return try fixture.result(size: size, prepared: true)
            }
            return try fixture.result(size: 1024)
        }
        try await store.loadDataset(fixture.original)
        store.selectedSampleId = "stucco-1024-validation"
        store.selectTrainingSize(2048)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.training.size, 2048)
        XCTAssertEqual(store.dataset?.datasetPath, fixture.prepared.path)
        XCTAssertEqual(store.selectedMaterialId, "stucco")
        XCTAssertTrue(store.dataset?.hasNativeSize(2048) == true)
        let prepare = try XCTUnwrap(recorder.arguments.last)
        XCTAssertEqual(value("--dataset", in: prepare), fixture.original.path)
        XCTAssertEqual(value("--expected-index-sha256", in: prepare), "source-sha")

        store.selectTrainingSize(1024)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
        XCTAssertEqual(store.training.size, 1024)
        XCTAssertEqual(store.selectedMaterialId, "stucco")
        XCTAssertEqual(value("--expected-index-sha256", in: try XCTUnwrap(recorder.arguments.last)), "prepared-sha")
        XCTAssertEqual(try Data(contentsOf: fixture.original.appendingPathComponent("dataset.json")), fixture.originalBytes)
    }

    func testInvalidPreparationProofRetainsSelectedDatasetAndRequestedSize() async throws {
        for fault in ["wrong-size", "wrong-normal-size", "resized", "modified", "wrong-path", "missing-proof"] {
            let fixture = try Fixture()
            defer { fixture.remove() }
            let store = fixture.store { args, _ in
                try fixture.result(size: args.first == "prepare-size" ? 2048 : 1024,
                                   prepared: args.first == "prepare-size", fault: args.first == "prepare-size" ? fault : nil)
            }
            try await store.loadDataset(fixture.original)
            store.selectTrainingSize(2048)
            try await settled(store)
            XCTAssertNotNil(store.error, fault)
            XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path, fault)
            XCTAssertEqual(store.training.size, 2048, "A failed preparation must never silently downgrade the requested size")
            XCTAssertFalse(store.isPreparingDataset)
        }
    }

    func testStoppingPreparationNeverAdoptsItsLateResultOrStartsTraining() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let gate = PreparationGate()
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in
            recorder.arguments.append(args)
            if args.first == "prepare-size" {
                await withCheckedContinuation { gate.continuation = $0 }
                return try fixture.result(size: 2048, prepared: true)
            }
            return try fixture.result(size: 1024)
        }
        try await store.loadDataset(fixture.original)
        store.training.size = 2048
        store.prepareTrainingDataset()
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while gate.continuation == nil, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        guard let continuation = gate.continuation else { XCTFail("Preparation did not reach its worker"); store.stop(); return }
        XCTAssertTrue(store.isPreparingDataset)
        XCTAssertFalse(store.isTraining)
        store.stop()
        continuation.resume()
        gate.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
        XCTAssertEqual(recorder.arguments.compactMap(\.first), ["dataset", "prepare-size"])
        XCTAssertNil(store.lastOutputURL)
        XCTAssertFalse(store.isPreparingDataset)
    }

    func testPreparingRequestedNativeSizeRemainsAvailableWithoutTraining() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, script in
            recorder.arguments.append(args)
            recorder.scripts.append(script)
            return try fixture.result(size: args.first == "prepare-size" ? 2048 : 1024,
                                      prepared: args.first == "prepare-size")
        }
        try await store.loadDataset(fixture.original)
        store.training.size = 2048
        store.training.useSelectedMaterialOnly = true
        store.selectedSampleId = "stucco-1024-validation"
        store.prepareTrainingDataset()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(recorder.arguments.compactMap(\.first), ["dataset", "prepare-size"])
        let prepare = try XCTUnwrap(recorder.arguments.last)
        XCTAssertEqual(value("--dataset", in: prepare), fixture.original.path)
        XCTAssertEqual(value("--size", in: prepare), "2048")
        XCTAssertEqual(value("--material", in: prepare), "stucco")
        XCTAssertEqual(store.dataset?.datasetPath, fixture.prepared.path)
        XCTAssertTrue(store.dataset?.hasNativeSize(2048) == true)
        XCTAssertFalse(recorder.scripts.contains("material_training_cycle.py"))
    }

    func testRetiredTrainingNeverLaunchesWithOrWithoutASelectedCheckpoint() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in recorder.arguments.append(args); return try fixture.result(size: 1024) }
        try await store.loadDataset(fixture.original)
        store.training.useWarmStart = true
        store.startTraining()
        XCTAssertEqual(store.error, MaterialTrainingPolicy.trainingIssue)
        let checkpoint = try WorkbenchProcess.decode(WorkbenchCheckpoint.self, output: "{\"checkpoint_path\":\"/chosen/head.pt\",\"sha256\":\"chosen-sha\",\"schema\":\"material-native-map-cycle-v1\",\"target\":\"height\",\"step\":800,\"compatible\":true,\"variant\":\"frozen\"}")
        store.checkpoints = [checkpoint]
        store.selectedCheckpointId = checkpoint.id
        store.startTraining()
        try await settled(store)
        XCTAssertEqual(store.error, MaterialTrainingPolicy.trainingIssue)
        XCTAssertEqual(recorder.arguments.compactMap(\.first), ["dataset"])
        XCTAssertNil(store.lastOutputURL)
    }

    func testLegacyLoRAAndFrozenCheckpointsCannotAdvertiseRetiredRefinement() throws {
        for variant in ["lora", "frozen"] {
            for supported in [true, false] {
                let document: [String: Any] = ["checkpoint_path": "/chosen/adapted.pt", "sha256": "adapted-sha",
                    "schema": "material-adaptation-diagnostic-v1", "target": "height", "step": 1200,
                    "compatible": true, "variant": variant, "supports_training_warm_start": supported]
                let checkpoint = try WorkbenchProcess.decode(WorkbenchCheckpoint.self,
                    output: String(decoding: JSONSerialization.data(withJSONObject: document), as: UTF8.self))
                XCTAssertFalse(checkpoint.supportsTrainingWarmStart,
                               "Saved backend capability cannot reactivate the removed DINO training route")
            }
        }
    }

    func testRestoreRunsOnlyOnceWithoutResettingSelectionOrTrainingOptions() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.preferences.set(fixture.original.path, forKey: "dataset")
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in recorder.arguments.append(args); return try fixture.result(size: 1024) }
        store.restore()
        try await settled(store)
        store.selectedSampleId = "stucco-1024-validation"
        store.training.updatesPerCrop = 777
        store.training.useWarmStart = true
        store.restore()
        try await settled(store)
        XCTAssertEqual(recorder.arguments.count, 1)
        XCTAssertEqual(store.selectedSampleId, "stucco-1024-validation")
        XCTAssertEqual(store.training.updatesPerCrop, 777)
        XCTAssertTrue(store.training.useWarmStart)
        store.openDataset(fixture.original)
        try await settled(store)
        XCTAssertEqual(recorder.arguments.count, 2, "Explicit Open remains available after restoration")
    }

    func testNonFiniteResourceLimitRefusesBeforePreparingOrStarting() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in recorder.arguments.append(args); return try fixture.result(size: 1024) }
        try await store.loadDataset(fixture.original)
        store.training.size = 2048
        store.training.memoryGB = .nan
        store.startTraining()
        XCTAssertNotNil(store.error)
        XCTAssertEqual(recorder.arguments.count, 1)
    }

    func testHardwareResourceAllowancePersistsWithoutReactivatingRetiredTraining() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let resources = MachineResources(physicalBytes: 64 * MachineResources.gibibyte,
                                         metalRecommendedBytes: 52 * MachineResources.gibibyte)
        let store = fixture.store(resources: resources) { args, _ in
            recorder.arguments.append(args)
            return try fixture.result(size: 1024)
        }
        XCTAssertEqual(store.training.memoryGB, 51.2, accuracy: 0.000_001)
        try await store.loadDataset(fixture.original)
        for memory in [56.0, 64.0] {
            store.training.memoryGB = memory
            XCTAssertEqual(store.trainingConfigurationIssue, MaterialTrainingPolicy.trainingIssue)
            store.startTraining()
            store.resumeTraining(from: fixture.root.appendingPathComponent("checkpoint.latest.pt"))
            try await settled(store)
            XCTAssertEqual(store.error, MaterialTrainingPolicy.trainingIssue)
            XCTAssertEqual(recorder.arguments.compactMap(\.first), ["dataset"])
        }
    }

    func testSelectedMaterialAndReviewOptionsCannotBypassTrainingRetirement() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in
            recorder.arguments.append(args)
            return try fixture.result(size: 1024, fault: "unreviewed-stucco")
        }
        try await store.loadDataset(fixture.original)
        store.training.allowUnreviewed = false
        for selectedOnly in [false, true] {
            store.training.useSelectedMaterialOnly = selectedOnly
            store.selectedSampleId = "soil-1024-train"
            XCTAssertEqual(store.trainingConfigurationIssue, MaterialTrainingPolicy.trainingIssue)
            store.startTraining()
            try await settled(store)
            XCTAssertEqual(store.error, MaterialTrainingPolicy.trainingIssue)
            XCTAssertEqual(recorder.arguments.compactMap(\.first), ["dataset"])
        }
    }

    func testRuntimeDiscoveryReusesExecutableStudioRuntimeWithoutClaimingMissingOne() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let registry = fixture.root.appendingPathComponent("python-runtime.json")
        try JSONSerialization.data(withJSONObject: ["path": "/bin/sh", "managed": false]).write(to: registry)
        XCTAssertEqual(MaterialWorkbenchRuntime.defaultPython(workspace: fixture.root, registry: registry), "/bin/sh")
        try JSONSerialization.data(withJSONObject: ["path": "/missing/python", "managed": false]).write(to: registry)
        XCTAssertEqual(MaterialWorkbenchRuntime.defaultPython(workspace: fixture.root, registry: registry), fixture.root.appendingPathComponent(".venv/bin/python").path)
        try Data("not a registry".utf8).write(to: registry)
        XCTAssertEqual(MaterialWorkbenchRuntime.defaultPython(workspace: fixture.root, registry: registry), fixture.root.appendingPathComponent(".venv/bin/python").path)
    }

    func testResumeNeverStartsTheRetiredWorkerRegardlessOfSavedFormOptions() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in recorder.arguments.append(args); return "unexpected retired worker" }
        for memory in [Double.infinity, 12.0] {
            store.training.memoryGB = memory
            store.training.target = "normal"
            store.training.useWarmStart = true
            store.training.updatesPerCrop = 900
            store.resumeTraining(from: fixture.root.appendingPathComponent("checkpoint.latest.pt"))
            try await settled(store)
            XCTAssertEqual(store.error, MaterialTrainingPolicy.trainingIssue)
            XCTAssertTrue(recorder.arguments.isEmpty)
            XCTAssertNil(store.lastOutputURL)
            XCTAssertFalse(store.isResumingTraining)
        }
    }

    func testReopenedNativeDatasetRetainsCrossSizeValidationNotice() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store { _, _ in try fixture.result(size: 2048) }
        try await store.loadDataset(fixture.prepared)
        XCTAssertNil(store.dataset?.preparation)
        XCTAssertEqual(store.dataset?.crossSizeValidationNotice, "Separate native crop size uses a separate split lineage.")
        XCTAssertTrue(store.dataset?.hasNativeSize(2048) == true)
    }

    func testRetiredResumeDoesNotCreateManagedOrUserSelectedWorkingFolders() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.preferences.set("", forKey: "workspace")
        let managed = fixture.root.appendingPathComponent("Application Support/Material Workspace")
        let recorder = WorkerRecorder()
        let store = WorkbenchStore(preferences: fixture.preferences, managedWorkspaceURL: managed,
                                   workerOverride: { args, _ in recorder.arguments.append(args); return "unexpected retired worker" })
        XCTAssertEqual(store.workspaceURL, managed.standardizedFileURL)
        store.resumeTraining(from: fixture.root.appendingPathComponent("checkpoint.latest.pt"))
        try await settled(store)
        XCTAssertEqual(store.error, MaterialTrainingPolicy.trainingIssue)
        XCTAssertFalse(FileManager.default.fileExists(atPath: managed.path))
        let missing = fixture.root.appendingPathComponent("missing-explicit-working-folder")
        fixture.preferences.set(missing.path, forKey: "workspace")
        let explicit = WorkbenchStore(preferences: fixture.preferences, managedWorkspaceURL: managed,
                                     workerOverride: { args, _ in recorder.arguments.append(args); return "unexpected retired worker" })
        explicit.resumeTraining(from: fixture.root.appendingPathComponent("checkpoint.latest.pt"))
        try await settled(explicit)
        XCTAssertEqual(explicit.error, MaterialTrainingPolicy.trainingIssue)
        XCTAssertTrue(recorder.arguments.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing.path))
    }

    func testSameSizeAlsoPreparesAutomaticChecksWithoutLosingOtherMaterials() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in
            recorder.arguments.append(args)
            if args.first == "train" { return "finished\n" }
            return try fixture.result(size: 1024, prepared: args.first == "prepare-size",
                                      fault: args.first == "dataset" ? "missing-automatic" : nil)
        }
        try await store.loadDataset(fixture.original)
        store.training.useSelectedMaterialOnly = true
        store.selectedSampleId = "stucco-1024-train"
        store.prepareTrainingDataset()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(recorder.arguments.compactMap(\.first), ["dataset", "prepare-size"])
        let prepare = recorder.arguments[1]
        XCTAssertTrue(prepare.contains("--automatic-validation"))
        XCTAssertEqual(value("--material", in: prepare), "stucco")
        XCTAssertEqual(store.dataset?.materials.map(\.id), ["soil", "stucco"])
    }

    private func settled(_ store: WorkbenchStore) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while store.isBusy, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.isBusy, "Operation did not settle. Activity: \(store.activity); error: \(store.error ?? "none")")
    }
    private func value(_ flag: String, in args: [String]) -> String? {
        guard let index = args.firstIndex(of: flag), args.indices.contains(index + 1) else { return nil }
        return args[index + 1]
    }
}

@MainActor private final class WorkerRecorder {
    var arguments: [[String]] = []
    var scripts: [String] = []
}

@MainActor private final class PreparationGate {
    var continuation: CheckedContinuation<Void, Never>?
}

@MainActor private final class Fixture {
    let root: URL
    let original: URL
    let prepared: URL
    let preferences: UserDefaults
    let suite = "org.ipde.training-preparation-tests.\(UUID().uuidString)"
    let originalBytes = Data("original dataset remains intact\n".utf8)
    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("native-training-tests-\(UUID().uuidString)")
        original = root.appendingPathComponent("original")
        prepared = root.appendingPathComponent("prepared-2k")
        preferences = UserDefaults(suiteName: suite)!
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try originalBytes.write(to: original.appendingPathComponent("dataset.json"))
    }
    func remove() {
        preferences.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: root)
    }
    func store(resources: MachineResources = .current, worker: @escaping @MainActor ([String], String) async throws -> String) -> WorkbenchStore {
        let result = WorkbenchStore(preferences: preferences, resources: resources, workerOverride: worker)
        result.workspacePath = root.path
        return result
    }
    func result(size: Int, prepared isPrepared: Bool = false, fault: String? = nil) throws -> String {
        let actualSize = fault == "wrong-size" ? 1024 : size
        let normalSize = fault == "wrong-normal-size" ? 1024 : actualSize
        let path = size == 2048 ? prepared : original
        let materials: [[String: Any]] = ["soil", "stucco"].map { material in
            ["material_id": material, "samples": ["train", "validation"].map { split in
                ["sample_id": "\(material)-\(actualSize)-\(split)", "status": material == "stucco" && fault == "unreviewed-stucco" ? "unreviewed" : "approved", "split": split,
                 "width": actualSize, "height": actualSize,
                 "maps": ["input": ["path": "/test/input.png", "width": actualSize, "height": actualSize],
                          "height": ["path": "/test/height.png", "width": actualSize, "height": actualSize],
                          "normal": ["path": "/test/normal.png", "width": normalSize, "height": normalSize],
                          "roughness": ["path": "/test/roughness.png", "width": actualSize, "height": actualSize]]] as [String: Any]
            }] as [String: Any]
        }
        var document: [String: Any] = ["dataset_path": path.path, "index_sha256": size == 2048 ? "prepared-sha" : "source-sha", "materials": materials, "automatic_validation": ["policy": "automatic-material-check-5pct-v1", "material_ids": ["soil", "stucco"]]]
        if fault == "missing-automatic" { document.removeValue(forKey: "automatic_validation") }
        if size == 2048 { document["cross_size_validation_notice"] = "Separate native crop size uses a separate split lineage." }
        if isPrepared, fault != "missing-proof" {
            document["preparation"] = ["source_dataset_path": original.path, "source_index_sha256": "source-sha",
                "prepared_dataset_path": fault == "wrong-path" ? "/another/dataset" : path.path,
                "crop_size": size, "reused": size == 1024, "target_resized": fault == "resized",
                "original_dataset_modified": fault == "modified", "split_lineage_changed": size != 1024,
                "cross_size_validation_notice": "Separate native crop size uses a separate split lineage."]
        }
        return String(decoding: try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys]), as: UTF8.self)
    }
}
