import Foundation
import XCTest
@testable import TextureStudio

@MainActor final class FreshMaterialTrainingLaunchTests: XCTestCase {
    func testDatasetManifestSymlinkResolvesToSavedDataset() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let manifest = fixture.dataset.appendingPathComponent("dataset.json")
        let alias = fixture.root.appendingPathComponent("chosen-dataset")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: manifest)
        let launch = try XCTUnwrap(FreshMaterialTrainingLaunch.parse(arguments: ["app", "--fresh-training", "--dataset", alias.path]))
        XCTAssertEqual(launch.datasetURL, manifest.resolvingSymlinksInPath().standardizedFileURL)
    }

    func testFreshRestorePersistsRGBHeightSettingsWithoutLoadingOldWeightsOrDepthPro() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let original = fixture.seedStalePreferences()
        var calls: [[String]] = []
        let store = fixture.store { arguments, _ in
            calls.append(arguments)
            switch arguments.first {
            case "capabilities": return "{\"training_sizes\":[256,512,1024,2048]}"
            case "dataset": return try fixture.datasetJSON()
            default: XCTFail("Fresh launch must only inspect capabilities and the chosen dataset: \(arguments)"); return "{}"
            }
        }
        store.specialistTrainingContext = .init(workspacePath: "/old/library", specialistID: "old", name: "Old",
            target: "height", datasetPath: "/old/dataset", membershipSHA256: "old-members")
        store.restore(arguments: fixture.freshArguments + ["--training-size", "1024", "--learning-rate", "0.0001", "--updates-per-map", "100"])
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["capabilities", "dataset"])
        XCTAssertEqual(value("--model-family", in: calls[0]), "compact-scalar")
        XCTAssertEqual(value("--scope", in: calls[0]), "full-model")
        XCTAssertEqual(value("--dataset", in: calls[1]), fixture.dataset.path)
        XCTAssertEqual(value("--target", in: calls[1]), "height")
        XCTAssertEqual(value("--review-size", in: calls[1]), "1024")
        XCTAssertFalse(calls.joined().contains("--depthpro-model"))
        XCTAssertFalse(calls.joined().contains("--model-directory"))
        XCTAssertEqual(store.dataset?.datasetPath, fixture.dataset.path)
        XCTAssertEqual(store.training.modelFamily, .compactScalar)
        XCTAssertEqual(store.training.target, "height")
        XCTAssertEqual(store.training.scope, "full-model")
        XCTAssertEqual(store.training.size, 1024, "A dataset's old 2K review size cannot replace the explicit launch size")
        XCTAssertEqual(store.training.learningRate, 0.0001)
        XCTAssertEqual(store.training.updatesPerCrop, 100)
        XCTAssertEqual(store.training.maxMinutes, original.maxMinutes)
        XCTAssertEqual(store.training.randomizeCrops, original.randomizeCrops)
        XCTAssertEqual(store.training.randomFlip, original.randomFlip)
        XCTAssertEqual(store.training.shuffleTrainingOrder, original.shuffleTrainingOrder)
        XCTAssertEqual(store.training.exaggerateFeatures, original.exaggerateFeatures)
        XCTAssertEqual(store.training.checkpointEvery, original.checkpointEvery)
        XCTAssertEqual(store.training.seed, original.seed)
        XCTAssertFalse(store.training.useWarmStart)
        XCTAssertTrue(store.checkpoints.isEmpty)
        XCTAssertNil(store.selectedCheckpointId)
        XCTAssertTrue(store.comparisonCheckpointIds.isEmpty)
        XCTAssertFalse(store.comparisonIncludesBase)
        XCTAssertTrue(store.comparisonCandidates.isEmpty)
        XCTAssertNil(store.sourceImageURL)
        XCTAssertNil(store.selectedSampleId)
        XCTAssertNil(store.selectedInputVariantId)
        XCTAssertNil(store.lastOutputURL)
        XCTAssertNil(store.lastLogURL)
        XCTAssertNil(store.lastPackageURL)
        XCTAssertNil(store.lastPackageCheckpointId)
        XCTAssertNil(store.specialistTrainingContext, "An ordinary dataset cannot inherit an unrelated specialist context")
        XCTAssertEqual(store.trainingConfigurationIssue, "Choose a material category in Loom Room.")
        store.startTraining()
        XCTAssertEqual(store.error, "Choose a material category in Loom Room.")
        XCTAssertEqual(calls.map { $0[0] }, ["capabilities", "dataset"], "Opening an ordinary folder cannot create a global training path")
        XCTAssertFalse(store.isBusy)
        XCTAssertEqual(fixture.defaults.stringArray(forKey: "checkpoints"), [])
        XCTAssertEqual(fixture.defaults.string(forKey: "dataset"), fixture.dataset.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.oldCheckpoint.path), "Fresh launch disconnects the old model without deleting it")
        let saved = WorkbenchPreferences.load(from: fixture.defaults)
        XCTAssertEqual(saved.training, store.training)
        XCTAssertNil(saved.selectedCheckpointId)
        XCTAssertNil(saved.selectedSampleId)
        XCTAssertEqual(saved.comparisonCheckpointIds, [])
        XCTAssertNil(saved.sourceImagePath)
        XCTAssertNil(saved.lastOutputPath)
        XCTAssertNil(saved.lastLogPath)
        XCTAssertNil(saved.lastPackagePath)
        let reopened = fixture.store { _, _ in XCTFail("Constructing a store must not launch a worker"); return "{}" }
        XCTAssertEqual(reopened.training, store.training)
        XCTAssertNil(reopened.selectedCheckpointId)
        XCTAssertNil(reopened.lastOutputURL)
        XCTAssertTrue(reopened.comparisonCheckpointIds.isEmpty)
        XCTAssertFalse(reopened.isTraining)
    }

    func testInvalidFreshArgumentsDoNotMutateSavedSettingsOrStartWorkers() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        fixture.seedStalePreferences()
        var calls = 0
        let store = fixture.store { arguments, _ in
            calls += 1
            return arguments.first == "capabilities" ? "{\"training_sizes\":[512,1024,2048]}" : try fixture.datasetJSON()
        }
        let savedData = fixture.defaults.data(forKey: WorkbenchPreferences.key)
        let original = store.training
        let checkpoint = store.selectedCheckpointId
        let originalDataset = fixture.defaults.string(forKey: "dataset")
        let originalCheckpoints = fixture.defaults.stringArray(forKey: "checkpoints")
        let unrelated = fixture.root.appendingPathComponent("diffuse.png")
        try Data("unrelated source image".utf8).write(to: unrelated)
        let invalid: [[String]] = [
            ["app", "--fresh-training"], ["app", "--fresh-training", "--dataset"],
            ["app", "--fresh-training", "--dataset", ""],
            ["app", "--fresh-training", "--dataset", fixture.root.appendingPathComponent("missing").path],
            ["app", "--fresh-training", "--dataset", unrelated.path],
            ["app", "--fresh-training", "--dataset", fixture.oldCheckpoint.path],
            fixture.freshArguments + ["--fresh-training"],
            fixture.freshArguments + ["--dataset", fixture.dataset.path],
            fixture.freshArguments + ["--training-size"],
            fixture.freshArguments + ["--training-size", "4096"],
            fixture.freshArguments + ["--training-size", "-1"],
            fixture.freshArguments + ["--training-size", "1024", "--training-size", "512"],
            fixture.freshArguments + ["--learning-rate", "0"],
            fixture.freshArguments + ["--learning-rate", "nan"],
            fixture.freshArguments + ["--learning-rate", "1e-100"],
            fixture.freshArguments + ["--learning-rate", "1e100"],
            fixture.freshArguments + ["--learning-rate", "0.1", "--learning-rate", "0.2"],
            fixture.freshArguments + ["--updates-per-map", "0"],
            fixture.freshArguments + ["--updates-per-map", "-10"],
            fixture.freshArguments + ["--updates-per-map", "1.5"],
            fixture.freshArguments + ["--updates-per-map", "100", "--updates-per-map", "200"]
        ]
        for arguments in invalid {
            store.restore(arguments: arguments)
            XCTAssertNotNil(store.error, arguments.description)
            XCTAssertFalse(store.isBusy, arguments.description)
            XCTAssertEqual(calls, 0, arguments.description)
            XCTAssertEqual(store.training, original, arguments.description)
            XCTAssertEqual(store.selectedCheckpointId, checkpoint, arguments.description)
            XCTAssertEqual(fixture.defaults.data(forKey: WorkbenchPreferences.key), savedData, arguments.description)
            XCTAssertEqual(fixture.defaults.string(forKey: "dataset"), originalDataset, arguments.description)
            XCTAssertEqual(fixture.defaults.stringArray(forKey: "checkpoints"), originalCheckpoints, arguments.description)
        }
        store.restore(arguments: fixture.freshArguments)
        try await settled(store)
        XCTAssertNil(store.error, "An invalid launch must allow a subsequent corrected request")
        XCTAssertEqual(calls, 2)
        XCTAssertEqual(store.training.size, original.size, "Omitting overrides keeps supported saved training settings")
        XCTAssertEqual(store.training.learningRate, original.learningRate)
    }

    func testFreshSpecialistDatasetRestoresOnlyMatchingHeightRegistrationContext() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        fixture.seedStalePreferences()
        try fixture.bindSpecialist()
        var calls: [String] = []
        let store = fixture.store { arguments, _ in
            calls.append(arguments[0])
            return arguments.first == "capabilities" ? "{\"training_sizes\":[512,1024,2048]}" : try fixture.datasetJSON()
        }
        store.restore(arguments: fixture.freshArguments)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls, ["capabilities", "dataset"])
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "class-bark")
        XCTAssertEqual(store.specialistTrainingContext?.datasetPath, fixture.dataset.path)
        XCTAssertEqual(store.specialistTrainingContext?.target, "height")
        XCTAssertEqual(store.specialistTrainingContext?.membershipSHA256, "current-bark-height-members")
        XCTAssertEqual(store.dataset?.sourceFolderPath, fixture.sources.path)
        XCTAssertEqual(store.modelWorkshopSourcePath, fixture.sources.path)
        XCTAssertNil(store.trainingConfigurationIssue)
        XCTAssertEqual(store.training.modelName, "Bark Displacement")
        XCTAssertFalse(store.training.useWarmStart)
        XCTAssertTrue(store.checkpoints.isEmpty, "Restoring the specialist registration context cannot load its old models")

        fixture.defaults.set("other-specialist", forKey: "selectedSpecialist")
        let unrelated = fixture.store { arguments, _ in
            arguments.first == "capabilities" ? "{\"training_sizes\":[512,1024,2048]}" : try fixture.datasetJSON()
        }
        unrelated.restore(arguments: fixture.freshArguments)
        try await settled(unrelated)
        XCTAssertNil(unrelated.specialistTrainingContext, "The dataset binding must match the selected specialist exactly")
    }

    func testFreshBoundCategoryWithCachedZeroCountRestoresIdentityAndRefreshesBeforeTraining() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        fixture.seedStalePreferences()
        try fixture.bindSpecialist(heightCount: 0, heightMembership: "")
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            switch args.first {
            case "capabilities": return "{\"training_sizes\":[256,512,1024,2048]}"
            case "dataset": return try fixture.datasetJSON()
            case "refresh-specialists": throw StudioError("Reached current category tag refresh.")
            default: XCTFail("Fresh launch must resolve its category before preparing or training: \(args)"); return "{}"
            }
        }
        defer { store.observeSourceFolder(nil) }
        store.restore(arguments: fixture.freshArguments + ["--training-size", "2048"])
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "class-bark")
        XCTAssertEqual(store.specialistTrainingContext?.membershipSHA256, "")
        XCTAssertEqual(store.training.size, 2048)
        XCTAssertNil(store.trainingConfigurationIssue, "A cached empty category cannot block current paired maps")
        store.startTraining()
        try await settled(store)
        let refresh = try XCTUnwrap(calls.first { $0.first == "refresh-specialists" })
        XCTAssertEqual(value("--dataset", in: refresh), fixture.sources.path)
        XCTAssertEqual(store.error, "Reached current category tag refresh.")
        XCTAssertFalse(calls.contains { ["prepare-size", "train", "refine"].contains($0.first ?? "") })
        XCTAssertEqual(store.training.size, 2048)
    }

    private func value(_ flag: String, in arguments: [String]) -> String? {
        arguments.firstIndex(of: flag).flatMap { arguments.indices.contains($0 + 1) ? arguments[$0 + 1] : nil }
    }
    private func settled(_ store: WorkbenchStore) async throws {
        let deadline = Date().addingTimeInterval(5)
        while store.isBusy, Date() < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.isBusy)
    }

    @MainActor private struct Fixture {
        let root: URL
        let dataset: URL
        let sources: URL
        let oldCheckpoint: URL
        let suite = "org.ipde.fresh-launch.\(UUID().uuidString)"
        let defaults: UserDefaults
        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("fresh-launch-\(UUID().uuidString)")
            dataset = root.appendingPathComponent("new-bark")
            sources = root.appendingPathComponent("originals")
            oldCheckpoint = root.appendingPathComponent("gray.safetensors")
            defaults = UserDefaults(suiteName: suite)!
            try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: dataset.appendingPathComponent("dataset.json"))
            try Data("old gray model".utf8).write(to: oldCheckpoint)
            defaults.set(root.path, forKey: "workspace")
        }
        var freshArguments: [String] { ["app", "--trainer", "--fresh-training", "--dataset", dataset.path] }
        @discardableResult func seedStalePreferences() -> MaterialTrainingOptions {
            var training = MaterialTrainingOptions()
            training.modelFamily = .depthProHeightRefiner
            training.scope = "full-model"
            training.size = 512
            training.learningRate = 0.0007
            training.updatesPerCrop = 25
            training.maxMinutes = 80
            training.useWarmStart = true
            training.randomizeCrops = .epoch
            training.randomFlip = true
            training.shuffleTrainingOrder = false
            training.exaggerateFeatures = true
            training.checkpointEvery = 3
            training.seed = 91
            WorkbenchPreferences(training: training, selectedSampleId: "old-gray-sample", selectedRole: "normal",
                selectedInputVariantId: "old-input", selectedCheckpointId: "gray",
                comparisonCheckpointIds: ["gray"], comparisonIncludesBase: true,
                sourceImagePath: root.appendingPathComponent("old-photo.png").path,
                lastOutputPath: root.appendingPathComponent("old-inference").path,
                lastLogPath: root.appendingPathComponent("old-log").path,
                lastPackagePath: root.appendingPathComponent("old-package").path,
                lastPackageCheckpointId: "gray").save(to: defaults)
            defaults.set([oldCheckpoint.path], forKey: "checkpoints")
            defaults.set(root.appendingPathComponent("old-dataset").path, forKey: "dataset")
            return training
        }
        func store(worker: @escaping @MainActor ([String], String) async throws -> String) -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, managedWorkspaceURL: root,
                selectedCheckpointRegistryURL: root.appendingPathComponent("selected.json"), workerOverride: worker)
        }
        func datasetJSON() throws -> String {
            var object: [String: Any] = ["dataset_path": dataset.path, "index_sha256": "new-index",
                "training_size": 2048, "supported_training_sizes": [256, 512, 1024, 2048],
                "materials": [["material_id": "sycamore", "samples": [["sample_id": "new-sycamore",
                    "status": "approved", "split": "train", "width": 2048, "height": 4096,
                    "maps": ["input": ["path": dataset.appendingPathComponent("diffuse.png").path, "width": 2048, "height": 4096],
                             "height": ["path": dataset.appendingPathComponent("height.png").path, "width": 2048, "height": 4096,
                                        "source_bits": 16]]]]]]]
            if FileManager.default.fileExists(atPath: dataset.appendingPathComponent(".specialist-binding.json").path) {
                object["source_folder_path"] = sources.path
            }
            return String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        }
        func bindSpecialist(heightCount: Int = 1, heightMembership: String = "current-bark-height-members") throws {
            let workspace = root.appendingPathComponent("specialists")
            try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
            defaults.set(workspace.path, forKey: "specialistWorkspace")
            defaults.set("class-bark", forKey: "selectedSpecialist")
            defaults.set(sources.path, forKey: "specialistSourceDataset")
            let binding: [String: Any] = ["schema": NativeMaterialSpecialistService.schema,
                "specialist_id": "class-bark", "workspace_path": workspace.path]
            try JSONSerialization.data(withJSONObject: binding).write(to: dataset.appendingPathComponent(".specialist-binding.json"))
            let registry: [String: Any] = ["workspace_path": workspace.path, "source_dataset_path": sources.path,
                "classifier_sha256": "classifier", "classifier_labels": ["Bark"], "updated_at": "2026-10-12",
                "classifier_only_count": 0, "material_count": 1, "family_count": 1, "warnings": [],
                "specialists": [["id": "class-bark", "name": "Bark", "class_label": "Bark", "is_custom": false,
                    "dataset_path": dataset.path, "material_ids": ["sycamore"], "family_ids": ["sycamore"],
                    "targets": [["target": "height", "material_count": heightCount, "status": heightCount > 0 ? "ready" : "no_targets",
                        "membership_sha256": heightMembership]],
                    "models": [["id": "old", "target": "height", "checkpoint_path": oldCheckpoint.path,
                        "checkpoint_sha256": "old", "training_size": 1024, "completed_at": "2026-10-10", "membership_sha256": "old-members"]]]]]
            try JSONSerialization.data(withJSONObject: registry).write(to: workspace.appendingPathComponent("specialists.json"))
        }
        func remove() {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }
    }
}
