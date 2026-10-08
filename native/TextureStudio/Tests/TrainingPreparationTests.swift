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
        for fault in ["wrong-size", "resized", "modified", "wrong-path", "missing-proof"] {
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
        store.startTraining()
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

    func testStartPreparesRequestedNativeSizeBeforeLaunchingNewBaseHead() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, script in
            recorder.arguments.append(args)
            recorder.scripts.append(script)
            if args.first == "train" { return "saved test run\n" }
            return try fixture.result(size: args.first == "prepare-size" ? 2048 : 1024, prepared: args.first == "prepare-size")
        }
        try await store.loadDataset(fixture.original)
        XCTAssertFalse(store.training.useWarmStart)
        store.training.size = 2048 // Also cover callers that bypass the picker.
        store.training.useSelectedMaterialOnly = true
        store.selectedSampleId = "stucco-1024-validation"
        store.startTraining()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(recorder.arguments.compactMap(\.first), ["dataset", "prepare-size", "train"])
        let train = try XCTUnwrap(recorder.arguments.last)
        XCTAssertEqual(value("--dataset", in: train), fixture.prepared.path)
        XCTAssertEqual(value("--expected-size", in: train), "2048")
        XCTAssertEqual(value("--material", in: train), "stucco")
        XCTAssertFalse(train.contains("--warm-start"))
        XCTAssertEqual(recorder.scripts.last, "material_training_cycle.py")
    }

    func testRefinementRequiresCheckpointAndUsesExactSelectedIdentity() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in
            recorder.arguments.append(args)
            return args.first == "train" ? "finished\n" : try fixture.result(size: 1024)
        }
        try await store.loadDataset(fixture.original)
        store.training.useWarmStart = true
        store.startTraining()
        XCTAssertTrue(store.error?.contains("Locate a checkpoint") == true)
        XCTAssertEqual(recorder.arguments.count, 1)
        let checkpoint = try WorkbenchProcess.decode(WorkbenchCheckpoint.self, output: "{\"checkpoint_path\":\"/chosen/head.pt\",\"sha256\":\"chosen-sha\",\"schema\":\"material-native-map-cycle-v1\",\"target\":\"height\",\"step\":800,\"compatible\":true,\"variant\":\"frozen\"}")
        store.checkpoints = [checkpoint]
        store.selectedCheckpointId = checkpoint.id
        store.startTraining()
        try await settled(store)
        XCTAssertNil(store.error)
        let train = try XCTUnwrap(recorder.arguments.last)
        XCTAssertEqual(value("--warm-start", in: train), checkpoint.checkpointPath)
        XCTAssertEqual(value("--warm-start-sha256", in: train), checkpoint.sha256)
    }

    func testLoRAHeadRefinementRequiresBackendCapabilityAndKeepsExactCheckpoint() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        func checkpoint(variant: String, supported: Bool?) throws -> WorkbenchCheckpoint {
            var document: [String: Any] = ["checkpoint_path": "/chosen/adapted.pt", "sha256": "adapted-sha",
                "schema": "material-adaptation-diagnostic-v1", "target": "height", "step": 1200,
                "compatible": true, "variant": variant]
            if let supported { document["supports_training_warm_start"] = supported }
            document["refinement_policy"] = "refine_material_head_with_frozen_lora_encoder"
            return try WorkbenchProcess.decode(WorkbenchCheckpoint.self,
                output: String(decoding: JSONSerialization.data(withJSONObject: document), as: UTF8.self))
        }
        XCTAssertFalse(try checkpoint(variant: "lora", supported: nil).supportsTrainingWarmStart,
                       "Legacy responses must not silently discard adapted encoder weights")
        XCTAssertFalse(try checkpoint(variant: "frozen", supported: false).supportsTrainingWarmStart)
        XCTAssertTrue(try checkpoint(variant: "frozen", supported: nil).supportsTrainingWarmStart)
        let adapted = try checkpoint(variant: "lora", supported: true)
        XCTAssertTrue(adapted.supportsTrainingWarmStart)
        XCTAssertEqual(adapted.refinementPolicy, "refine_material_head_with_frozen_lora_encoder")
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in recorder.arguments.append(args); return args.first == "train" ? "finished\n" : try fixture.result(size: 1024) }
        try await store.loadDataset(fixture.original)
        store.training.useWarmStart = true
        store.checkpoints = [try checkpoint(variant: "lora", supported: nil)]
        store.selectedCheckpointId = adapted.id
        store.startTraining()
        XCTAssertNotNil(store.error)
        XCTAssertEqual(recorder.arguments.count, 1)
        store.checkpoints = [adapted]
        store.startTraining()
        try await settled(store)
        XCTAssertNil(store.error)
        let train = try XCTUnwrap(recorder.arguments.last)
        XCTAssertEqual(value("--warm-start", in: train), adapted.checkpointPath)
        XCTAssertEqual(value("--warm-start-sha256", in: train), adapted.sha256)
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

    func testReviewRequirementAppliesToSelectedMaterialsBeforeLaunching() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in
            recorder.arguments.append(args)
            return args.first == "train" ? "finished\n" : try fixture.result(size: 1024, fault: "unreviewed-stucco")
        }
        try await store.loadDataset(fixture.original)
        store.training.allowUnreviewed = false
        XCTAssertTrue(store.trainingConfigurationIssue?.contains("awaiting review") == true)
        store.startTraining()
        try await settled(store)
        XCTAssertNotNil(store.error)
        XCTAssertEqual(recorder.arguments.count, 1)
        store.training.useSelectedMaterialOnly = true
        store.selectedSampleId = "soil-1024-train"
        XCTAssertNil(store.trainingConfigurationIssue, "Unselected materials do not block an approved selected material")
        store.startTraining()
        try await settled(store)
        XCTAssertNil(store.error)
        let train = try XCTUnwrap(recorder.arguments.last)
        XCTAssertEqual(value("--material", in: train), "soil")
        XCTAssertFalse(train.contains("--allow-unreviewed"))
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

    func testResumeRestoresSavedRunInsteadOfApplyingVisibleDatasetTargetOrWarmStart() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let recorder = WorkerRecorder()
        let store = fixture.store { args, _ in recorder.arguments.append(args); return "saved resumed run\n" }
        store.training.memoryGB = .infinity
        store.resumeTraining(from: fixture.root.appendingPathComponent("checkpoint.latest.pt"))
        XCTAssertNotNil(store.error)
        XCTAssertTrue(recorder.arguments.isEmpty)
        store.training.memoryGB = 12
        store.training.target = "normal"
        store.training.size = 1024
        store.training.useWarmStart = true // A separate form choice does not affect Resume.
        store.training.updatesPerCrop = 900
        let checkpoint = fixture.root.appendingPathComponent("checkpoint.latest.pt")
        store.resumeTraining(from: checkpoint)
        try await settled(store)
        XCTAssertNil(store.error)
        let args = try XCTUnwrap(recorder.arguments.last)
        XCTAssertEqual(args.first, "resume")
        XCTAssertEqual(value("--resume-checkpoint", in: args), checkpoint.path)
        XCTAssertEqual(value("--updates-per-crop", in: args), "900")
        XCTAssertEqual(value("--max-driver-bytes", in: args), "12000000000")
        for flag in ["--dataset", "--expected-size", "--target", "--warm-start"] { XCTAssertFalse(args.contains(flag)) }
        XCTAssertTrue(store.lastOutputURL?.lastPathComponent.hasPrefix("resumed-material-run-") == true)
        XCTAssertFalse(store.isResumingTraining)
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

    func testFirstRunCreatesOnlyItsManagedWorkingFolderLazily() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.preferences.set("", forKey: "workspace") // Unconfigured distribution; do not use the Debug build's repository.
        let managed = fixture.root.appendingPathComponent("Application Support/Material Workspace")
        let store = WorkbenchStore(preferences: fixture.preferences, managedWorkspaceURL: managed,
                                   workerOverride: { _, _ in "saved resumed run\n" })
        XCTAssertEqual(store.workspaceURL, managed.standardizedFileURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: managed.path), "Discovery must not create a working folder")
        store.resumeTraining(from: fixture.root.appendingPathComponent("checkpoint.latest.pt"))
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertTrue(FileManager.default.fileExists(atPath: managed.appendingPathComponent("out/material-training").path))

        let missing = fixture.root.appendingPathComponent("missing-explicit-working-folder")
        fixture.preferences.set(missing.path, forKey: "workspace")
        let recorder = WorkerRecorder()
        let explicit = WorkbenchStore(preferences: fixture.preferences, managedWorkspaceURL: managed,
                                     workerOverride: { args, _ in recorder.arguments.append(args); return "unexpected worker" })
        explicit.resumeTraining(from: fixture.root.appendingPathComponent("checkpoint.latest.pt"))
        try await settled(explicit)
        XCTAssertTrue(explicit.error?.contains("working folder is missing") == true)
        XCTAssertTrue(recorder.arguments.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing.path), "A missing user-selected folder must not be silently recreated elsewhere")
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
        store.startTraining()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(recorder.arguments.compactMap(\.first), ["dataset", "prepare-size", "train"])
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
    func store(worker: @escaping @MainActor ([String], String) async throws -> String) -> WorkbenchStore {
        let result = WorkbenchStore(preferences: preferences, workerOverride: worker)
        result.workspacePath = root.path
        return result
    }
    func result(size: Int, prepared isPrepared: Bool = false, fault: String? = nil) throws -> String {
        let actualSize = fault == "wrong-size" ? 1024 : size
        let path = size == 2048 ? prepared : original
        let materials: [[String: Any]] = ["soil", "stucco"].map { material in
            ["material_id": material, "samples": ["train", "validation"].map { split in
                ["sample_id": "\(material)-\(actualSize)-\(split)", "status": material == "stucco" && fault == "unreviewed-stucco" ? "unreviewed" : "approved", "split": split,
                 "width": actualSize, "height": actualSize,
                 "maps": ["input": ["path": "/test/input.png"], "height": ["path": "/test/height.png"],
                          "normal": ["path": "/test/normal.png"], "roughness": ["path": "/test/roughness.png"]]] as [String: Any]
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
