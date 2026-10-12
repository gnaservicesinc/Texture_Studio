import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchPreferencesTests: XCTestCase {
    func testFreshWorkspaceUsesUserStorageAndPersistsDefaultWithoutCreatingRepositoryOutput() throws {
        let fixture = try PreferencesFixture(); defer { fixture.remove() }
        fixture.defaults.removeObject(forKey: "workspace")
        let managed = fixture.root.appendingPathComponent("user-support/Material Workspace")
        let store = WorkbenchStore(preferences: fixture.defaults, managedWorkspaceURL: managed)
        XCTAssertEqual(store.workspacePath, managed.path)
        XCTAssertEqual(fixture.defaults.string(forKey: "workspace"), managed.path)
        XCTAssertEqual(store.modelDirectory, managed.appendingPathComponent("out/material-training/transfer-models/pbrnxt-base").path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: managed.path), "Opening settings alone need not create output")
    }

    func testRepositoryWorkspacePreferencesIncludingNestedPathsAndAliasesMigrateWithoutMovingOldRuns() throws {
        let fixture = try PreferencesFixture(); defer { fixture.remove() }
        let checkout = fixture.root.appendingPathComponent("checkout"), managed = fixture.root.appendingPathComponent("user-support")
        try FileManager.default.createDirectory(at: checkout.appendingPathComponent(".git"), withIntermediateDirectories: true)
        let oldRun = checkout.appendingPathComponent("out/material-training/previous/model.safetensors")
        try FileManager.default.createDirectory(at: oldRun.deletingLastPathComponent(), withIntermediateDirectories: true)
        let original = Data("previous completed run".utf8); try original.write(to: oldRun)
        let alias = fixture.root.appendingPathComponent("checkout-alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: checkout)
        for path in [checkout, checkout.appendingPathComponent("out"), alias, alias.appendingPathComponent("missing/nested")] {
            fixture.defaults.set(path.path, forKey: "workspace")
            let store = WorkbenchStore(preferences: fixture.defaults, managedWorkspaceURL: managed)
            XCTAssertEqual(store.workspacePath, managed.path, path.path)
            XCTAssertEqual(fixture.defaults.string(forKey: "workspace"), managed.path)
            XCTAssertEqual(try Data(contentsOf: oldRun), original)
        }
        let worktree = fixture.root.appendingPathComponent("worktree")
        try FileManager.default.createDirectory(at: worktree, withIntermediateDirectories: true)
        try Data("gitdir: ../checkout/.git/worktrees/development\n".utf8).write(to: worktree.appendingPathComponent(".git"))
        fixture.defaults.set(worktree.path, forKey: "workspace")
        XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, managedWorkspaceURL: managed).workspacePath, managed.path)
    }

    func testExplicitUserWorkspaceOutsideRepositoryIsPreserved() throws {
        let fixture = try PreferencesFixture(); defer { fixture.remove() }
        let custom = fixture.root.appendingPathComponent("chosen-folder")
        try FileManager.default.createDirectory(at: custom, withIntermediateDirectories: true)
        fixture.defaults.set(custom.path, forKey: "workspace")
        let managed = fixture.root.appendingPathComponent("user-support")
        XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, managedWorkspaceURL: managed).workspacePath, custom.path)
        XCTAssertEqual(fixture.defaults.string(forKey: "workspace"), custom.path)
    }

    func testMissingRepositoryModelDownloadLocationMigratesWhileExistingDeveloperWeightsRemainSelected() throws {
        let fixture = try PreferencesFixture(); defer { fixture.remove() }
        let checkout = fixture.root.appendingPathComponent("checkout"), managed = fixture.root.appendingPathComponent("user-support")
        try FileManager.default.createDirectory(at: checkout.appendingPathComponent(".git"), withIntermediateDirectories: true)
        let oldBase = checkout.appendingPathComponent("out/material-training/transfer-models/pbrnxt-base")
        fixture.defaults.set(checkout.path, forKey: "workspace")
        fixture.defaults.set(oldBase.path, forKey: "materialModelDirectory")
        let migrated = WorkbenchStore(preferences: fixture.defaults, managedWorkspaceURL: managed)
        XCTAssertEqual(migrated.modelDirectory, managed.appendingPathComponent("out/material-training/transfer-models/pbrnxt-base").path)
        XCTAssertEqual(fixture.defaults.string(forKey: "materialModelDirectory"), migrated.modelDirectory)
        try FileManager.default.createDirectory(at: oldBase, withIntermediateDirectories: true)
        fixture.defaults.set(oldBase.path, forKey: "materialModelDirectory")
        XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, managedWorkspaceURL: managed).modelDirectory, oldBase.path)
    }

    func testCropTimingAndOrientationChoicesPersistWhileNeverSuppressesEffectiveFlags() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(store.training.randomizeCrops, .run)
        XCTAssertFalse(store.training.randomFlip)
        XCTAssertFalse(store.training.randomSpinCentercut)
        store.training.randomizeCrops = .epoch
        store.training.randomFlip = true
        store.training.randomSpinCentercut = true
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.training.randomizeCrops, .epoch)
        XCTAssertTrue(reopened.training.randomFlip)
        XCTAssertTrue(reopened.training.randomSpinCentercut)
        XCTAssertEqual(reopened.training.cropArguments,
            ["--randomize-crops", "epoch", "--random-flip", "--random-spin-centercut"])
        reopened.training.randomizeCrops = .never
        XCTAssertFalse(reopened.training.cropAugmentationEnabled)
        XCTAssertEqual(reopened.training.cropArguments, ["--randomize-crops", "never"])
        let disabled = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(disabled.training.randomizeCrops, .never)
        XCTAssertTrue(disabled.training.randomFlip, "Changing timing must preserve the user's flip selection")
        XCTAssertTrue(disabled.training.randomSpinCentercut, "Changing timing must preserve the user's spin selection")
        disabled.training.randomizeCrops = .run
        XCTAssertEqual(disabled.training.cropArguments,
            ["--randomize-crops", "run", "--random-flip", "--random-spin-centercut"])
    }

    func testLegacyTrainingOptionsDefaultCropTimingAndKeepExplicitFlipSelection() throws {
        let decoder = JSONDecoder()
        let legacy = try decoder.decode(MaterialTrainingOptions.self, from: Data("{}".utf8))
        XCTAssertEqual(legacy.randomizeCrops, .run)
        XCTAssertFalse(legacy.randomFlip)
        XCTAssertFalse(legacy.randomSpinCentercut)
        XCTAssertEqual(legacy.cropArguments, ["--randomize-crops", "run"])
        let flip = try decoder.decode(MaterialTrainingOptions.self, from: Data("{\"randomFlip\":true}".utf8))
        XCTAssertEqual(flip.randomizeCrops, .run)
        XCTAssertEqual(flip.cropArguments, ["--randomize-crops", "run", "--random-flip"])
        for timing in MaterialTrainingCropRandomization.allCases {
            var options = flip
            options.randomizeCrops = timing
            options.randomSpinCentercut = true
            XCTAssertEqual(try decoder.decode(MaterialTrainingOptions.self, from: JSONEncoder().encode(options)), options)
        }
    }

    func testDepthProModelPreferenceSharesCanonicalSuiteAndPreservesConfiguredPath() throws {
        let tools = try PreferencesFixture(), model = try PreferencesFixture()
        defer { tools.remove(); model.remove() }
        tools.defaults.set("/legacy/DepthPro.mlpackage", forKey: NativeDepthProContract.modelPathKey)
        model.defaults.set("/configured/DepthPro.mlpackage", forKey: NativeDepthProContract.modelPathKey)
        let store = WorkbenchStore(preferences: tools.defaults, depthProPreferences: model.defaults, resources: tools.resources)
        XCTAssertEqual(store.depthProModelPath, "/configured/DepthPro.mlpackage")
        store.depthProModelPath = "/selected/DepthPro.mlpackage"
        XCTAssertEqual(NativeDepthProContract.configuredModelURL(in: model.defaults).path, store.depthProModelPath,
                       "Studio inference and the workbench must resolve the same selected model")
        XCTAssertEqual(tools.defaults.string(forKey: NativeDepthProContract.modelPathKey), "/legacy/DepthPro.mlpackage")
        let reopened = WorkbenchStore(preferences: tools.defaults, depthProPreferences: model.defaults, resources: tools.resources)
        XCTAssertEqual(reopened.depthProModelPath, store.depthProModelPath)
    }

    func testDepthProModelPreferenceMigratesLegacyWorkbenchValue() throws {
        let tools = try PreferencesFixture(), model = try PreferencesFixture()
        defer { tools.remove(); model.remove() }
        tools.defaults.set("/legacy/DepthPro.mlpackage", forKey: NativeDepthProContract.modelPathKey)
        let store = WorkbenchStore(preferences: tools.defaults, depthProPreferences: model.defaults, resources: tools.resources)
        XCTAssertEqual(store.depthProModelPath, "/legacy/DepthPro.mlpackage")
        XCTAssertEqual(NativeDepthProContract.configuredModelURL(in: model.defaults).path, store.depthProModelPath)
        XCTAssertEqual(model.defaults.string(forKey: NativeDepthProContract.modelPathKey), store.depthProModelPath)
    }

    func testInjectedWorkbenchPreferencesKeepDepthProEditsIsolated() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let configuredAppPath = StudioPreferences.defaults.string(forKey: NativeDepthProContract.modelPathKey)
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        store.depthProModelPath = "/isolated/DepthPro.mlpackage"
        XCTAssertEqual(fixture.defaults.string(forKey: NativeDepthProContract.modelPathKey), store.depthProModelPath)
        XCTAssertEqual(NativeDepthProContract.configuredModelURL(in: fixture.defaults).path, store.depthProModelPath)
        XCTAssertEqual(StudioPreferences.defaults.string(forKey: NativeDepthProContract.modelPathKey), configuredAppPath,
                       "Injected workbench preferences must not alter the production model selection")
    }

    func testCompactFamilySelectionPersistsAndPreservesTypedSettingsAfterTransition() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(first.training.modelFamily, .pbrnxt)
        first.training.size = 512
        first.selectTrainingModelFamily(.compactScalar)
        XCTAssertEqual(first.training.target, "height")
        XCTAssertEqual(first.training.size, 512)
        XCTAssertEqual(first.training.learningRate, 0.001)
        XCTAssertEqual(first.training.effectiveScope, "full-model")
        first.training.learningRate = 0.0007
        first.selectTrainingModelFamily(.compactScalar)
        XCTAssertEqual(first.training.learningRate, 0.0007)
        first.selectTrainingTarget("normal")
        XCTAssertEqual(first.training.modelFamily, .compactNormal)
        XCTAssertEqual(first.training.learningRate, 0.0007)
        first.selectTrainingTarget("roughness")
        XCTAssertEqual(first.training.modelFamily, .compactScalar)
        XCTAssertEqual(first.training.learningRate, 0.0007)
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.training, first.training)
        reopened.selectTrainingModelFamily(.pbrnxt)
        XCTAssertEqual(reopened.training.target, "roughness")
        XCTAssertEqual(reopened.training.learningRate, 0.00001)
    }

    func testCompactOptionsDefaultLearningRateWithoutChangingLegacyPBRChoices() throws {
        let decoder = JSONDecoder()
        let old = try decoder.decode(MaterialTrainingOptions.self, from: Data("{\"target\":\"normal\",\"learningRate\":0.0002}".utf8))
        XCTAssertEqual(old.modelFamily, .pbrnxt)
        XCTAssertEqual(old.learningRate, 0.0002)
        let compact = try decoder.decode(MaterialTrainingOptions.self,
            from: Data("{\"modelFamily\":\"compact-scalar\",\"target\":\"roughness\"}".utf8))
        XCTAssertEqual(compact.modelFamily, .compactScalar)
        XCTAssertEqual(compact.learningRate, 0.001)
        XCTAssertNil(compact.configurationIssue)
        var invalid = compact
        invalid.target = "normal"
        XCTAssertNotNil(invalid.configurationIssue)
        invalid.modelFamily = .compactNormal
        invalid.loraRank = 0
        invalid.loraAlpha = 0
        XCTAssertNil(invalid.configurationIssue, "Full compact training does not validate hidden LoRA settings")
    }

    func testPreparationWorkersDefaultToAvailableCoresAndSaveImmediately() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let resources = MachineResources(physicalBytes: 32 * MachineResources.gibibyte, availableProcessorCount: 12)
        let first = WorkbenchStore(preferences: fixture.defaults, resources: resources)
        XCTAssertEqual(first.preparationWorkers, 12)
        first.preparationWorkers = 3
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: resources)
        XCTAssertEqual(reopened.preparationWorkers, 3)
        for value in [0, -1, Int.min] {
            reopened.preparationWorkers = value
            XCTAssertEqual(reopened.preparationWorkers, 1)
            XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, resources: resources).preparationWorkers, 1)
        }
    }

    func testPreparationWorkersAdaptToFewerAvailableCores() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        fixture.defaults.set(24, forKey: "preparationWorkers")
        let resources = MachineResources(physicalBytes: 16 * MachineResources.gibibyte, availableProcessorCount: 6)
        let store = WorkbenchStore(preferences: fixture.defaults, resources: resources)
        XCTAssertEqual(store.preparationWorkers, 6)
        for value in [100, Int.max] {
            store.preparationWorkers = value
            XCTAssertEqual(store.preparationWorkers, 6)
            XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, resources: resources).preparationWorkers, 6)
        }
    }

    func testFormAndComparisonChoicesSaveWithoutStartingWorkerOrClosingWindow() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        first.training.modelName = "Stone / warm evening"
        first.training.target = "normal"
        first.training.size = 2048
        first.training.updatesPerCrop = 875
        first.training.maxMinutes = 120
        first.training.useSelectedMaterialOnly = true
        first.training.useWarmStart = true
        first.selectedSampleId = "soil_crop_002"
        first.selectedRole = "roughness"
        first.selectedCheckpointId = "second-checkpoint"
        first.comparisonCheckpointIds = ["first-checkpoint", "second-checkpoint"]
        first.comparisonIncludesBase = false
        first.sourceImageURL = fixture.root.appendingPathComponent("source.png")
        first.lastPackageURL = fixture.root.appendingPathComponent("exported-model")
        first.lastPackageCheckpointId = "second-checkpoint"

        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.training, first.training)
        XCTAssertEqual(reopened.training.modelName, "Stone / warm evening")
        XCTAssertEqual(reopened.selectedSampleId, "soil_crop_002")
        XCTAssertEqual(reopened.selectedRole, "roughness")
        XCTAssertEqual(reopened.selectedCheckpointId, "second-checkpoint")
        XCTAssertEqual(reopened.comparisonCheckpointIds, ["first-checkpoint", "second-checkpoint"])
        XCTAssertFalse(reopened.comparisonIncludesBase)
        XCTAssertEqual(reopened.sourceImageURL, first.sourceImageURL)
        XCTAssertEqual(reopened.lastPackageURL, first.lastPackageURL)
        XCTAssertEqual(reopened.lastPackageCheckpointId, "second-checkpoint")
    }

    func testRestoreKeepsSelectedCropAndCheckpointInsteadOfLastLoadedRow() async throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let dataset = fixture.root.appendingPathComponent("dataset.json")
        let firstPath = fixture.root.appendingPathComponent("first.pt")
        let secondPath = fixture.root.appendingPathComponent("second.pt")
        for url in [dataset, firstPath, secondPath] { try Data().write(to: url) }
        fixture.defaults.set(dataset.path, forKey: "dataset")
        fixture.defaults.set([firstPath.path, secondPath.path], forKey: "checkpoints")
        let original = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        original.selectedSampleId = "soil_002"
        original.selectedCheckpointId = "first"
        original.comparisonCheckpointIds = ["second"]
        original.comparisonIncludesBase = false
        original.training.size = 2048
        var calls: [String] = []
        let restored = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources, workerOverride: { arguments, _ in
            calls.append(arguments[0])
            if arguments.first == "capabilities" { return "{\"training_sizes\":[512,1024,2048]}" }
            if arguments.first == "dataset" { return try Self.datasetJSON() }
            if arguments.first == "prepare-size" { return try Self.datasetJSON(size: 2048, prepared: true) }
            let path = arguments[try XCTUnwrap(arguments.firstIndex(of: "--checkpoint")) + 1]
            let id = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
            return """
            {"checkpoint_path":"\(path)","sha256":"\(id)","schema":"texture-studio-material-lora-v1","target":"height","step":42,"compatible":true}
            """
        })
        restored.restore()
        try await settled(restored)
        XCTAssertNil(restored.error)
        XCTAssertEqual(restored.selectedSampleId, "soil_002")
        XCTAssertEqual(restored.selectedCheckpointId, "first")
        XCTAssertEqual(restored.comparisonCheckpointIds, ["second"])
        XCTAssertFalse(restored.comparisonIncludesBase)
        XCTAssertEqual(restored.training.size, 2048, "Reopening a 1K dataset cannot reset the requested 2K size")
        XCTAssertEqual(calls, ["capabilities", "dataset", "checkpoint", "checkpoint"])
        XCTAssertTrue(restored.dataset?.hasNativeSize(1024) == true, "Opening preserves source references and allocates no rescaled dataset")
        XCTAssertEqual(fixture.defaults.stringArray(forKey: "checkpoints"), [firstPath.path, secondPath.path])
        restored.restore()
        XCTAssertEqual(calls.count, 4, "A view appearing again must not reload and overwrite selections")
        let next = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(next.selectedCheckpointId, "first")
        XCTAssertEqual(next.selectedSampleId, "soil_002")
        XCTAssertEqual(next.comparisonCheckpointIds, ["second"])
    }

    func testBareSourceStartupDefersSavedModelsAndKeepsSelectionsUntilModelNavigation() async throws {
        let fixture = try PreferencesFixture(); defer { fixture.remove() }
        let first = fixture.root.appendingPathComponent("first.safetensors")
        let second = fixture.root.appendingPathComponent("second.safetensors")
        let category = fixture.root.appendingPathComponent("category.safetensors")
        for file in [first, second, category] { try Data("Stored model bytes".utf8).write(to: file) }
        fixture.defaults.set(fixture.root.path, forKey: "dataset")
        fixture.defaults.set([first.path, second.path], forKey: "checkpoints")
        WorkbenchPreferences(selectedCheckpointId: "first", comparisonCheckpointIds: ["second"], comparisonIncludesBase: false).save(to: fixture.defaults)
        var calls: [String] = []
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources, workerOverride: { args, _ in
            calls.append(args[0])
            if args.first == "capabilities" { return "{\"training_sizes\":[256,512,1024,2048]}" }
            if args.first == "dataset" { return """
                {"dataset_path":"\(fixture.root.path)","source_folder_path":"\(fixture.root.path)","is_source_folder":true,
                 "index_sha256":"source","training_size":2048,"materials":[]}
                """ }
            let path = args[try XCTUnwrap(args.firstIndex(of: "--checkpoint")) + 1]
            let id = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
            return """
                {"checkpoint_path":"\(path)","sha256":"\(id)","schema":"texture-studio-compact-material-v1",
                 "target":"height","compatible":true,"step":1}
                """
        })
        store.restore(arguments: ["Texture Studio", "--dataset", fixture.root.path], sourceBrowsing: true)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls, ["capabilities", "dataset"], "Opening source folders cannot inspect saved model weights")
        XCTAssertTrue(store.checkpoints.isEmpty)
        XCTAssertEqual(store.selectedCheckpointId, "first")
        XCTAssertEqual(store.comparisonCheckpointIds, ["second"])
        XCTAssertEqual(fixture.defaults.stringArray(forKey: "checkpoints"), [first.path, second.path])
        XCTAssertEqual(WorkbenchPreferences.load(from: fixture.defaults).comparisonCheckpointIds, ["second"])

        try await store.loadCheckpoint(category)
        XCTAssertEqual(fixture.defaults.stringArray(forKey: "checkpoints"), [first.path, second.path, category.path],
                       "Loading one category model must retain deferred library paths")
        XCTAssertEqual(store.comparisonCheckpointIds, ["second", "category"])
        store.reconnectSavedCheckpoints()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls, ["capabilities", "dataset", "checkpoint", "checkpoint", "checkpoint"])
        XCTAssertEqual(Set(store.checkpoints.map(\.id)), ["first", "second", "category"])
        XCTAssertEqual(store.selectedCheckpointId, "category")
        XCTAssertEqual(store.comparisonCheckpointIds, ["second", "category"])
        store.reconnectSavedCheckpoints()
        XCTAssertFalse(store.isBusy, "Repeated model navigation reuses connected checkpoints")
        XCTAssertEqual(calls.count, 5)
    }

    func testBareSourceModelStartupReconnectsOnlyWhenModelNavigationIsRequested() async throws {
        let fixture = try PreferencesFixture(); defer { fixture.remove() }
        let model = fixture.root.appendingPathComponent("saved.safetensors")
        try Data("Stored model bytes".utf8).write(to: model)
        fixture.defaults.set(fixture.root.path, forKey: "dataset")
        fixture.defaults.set([model.path], forKey: "checkpoints")
        var calls: [String] = []
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources, workerOverride: { args, _ in
            calls.append(args[0])
            if args.first == "capabilities" { return "{\"training_sizes\":[2048]}" }
            if args.first == "dataset" { return """
                {"dataset_path":"\(fixture.root.path)","is_source_folder":true,"index_sha256":"source","materials":[]}
                """ }
            return """
                {"checkpoint_path":"\(model.path)","sha256":"saved","schema":"texture-studio-compact-material-v1",
                 "target":"height","compatible":true,"step":1}
                """
        })
        store.restore(arguments: ["Texture Studio"], sourceBrowsing: false)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls, ["capabilities", "dataset", "checkpoint"])
        XCTAssertEqual(store.selectedCheckpointId, "saved")
    }

    func testCapabilityDiscoveryPreservesSupportedResolutionSelection() async throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources, workerOverride: { arguments, _ in
            return "{\"training_sizes\":[1024,2048]}"
        })
        store.training.size = 2048
        try await store.loadTrainingCapabilities()
        XCTAssertEqual(store.training.size, 2048)
        store.selectTrainingSize(1024)
        XCTAssertEqual(store.training.size, 1024)
        store.selectTrainingSize(2048)
        XCTAssertEqual(store.training.size, 2048)
        try await store.loadTrainingCapabilities()
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.training.size, 2048)
    }

    func testValidDatasetCanRunBeforeCapabilityDiscovery() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let resources = MachineResources(physicalBytes: 8 * MachineResources.gibibyte)
        let store = WorkbenchStore(preferences: fixture.defaults, resources: resources)
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.datasetJSON())
        store.training.size = 1024
        XCTAssertNil(store.trainingConfigurationIssue)
        XCTAssertTrue(store.supportedTrainingSizes.contains(1024))
    }

    func testPreparedFolderSelectionAcceptsHigherTierWhenRequestedOriginalIsUndersized() throws {
        let originalID = "stone_1k"
        var object: [String: Any] = ["dataset_path": "/dataset/.training-data/2048", "index_sha256": "prepared-sha",
            "materials": [["material_id": "stone_2k", "samples": [["sample_id": "stone_2k_full",
                "status": "approved", "split": "train", "width": 2048, "height": 2048,
                "maps": ["input": ["path": "/sources/stone_diff_2k.png", "width": 2048, "height": 2048],
                         "height": ["path": "/sources/stone_disp_2k.png", "width": 2048, "height": 2048]]]]]],
            "automatic_validation": ["policy": "subject-extra-crops-v2", "target": "height",
                "material_ids": ["stone_2k"], "quick_fit_material_id": originalID],
            "preparation": ["source_dataset_path": "/dataset", "source_index_sha256": "source-sha",
                "prepared_dataset_path": "/dataset/.training-data/2048", "crop_size": 2048, "reused": false,
                "target_resized": false, "original_dataset_modified": false]]
        func decode() throws -> WorkbenchDataset {
            try WorkbenchResult.decode(WorkbenchDataset.self,
                output: String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self))
        }
        let prepared = try decode()
        XCTAssertTrue(prepared.readyForTraining(size: 2048, material: originalID, target: "height"))
        XCTAssertFalse(prepared.readyForTraining(size: 2048, material: "other_1k", target: "height"))
        XCTAssertFalse(prepared.readyForTraining(size: 2048, material: nil, target: "height"))
        XCTAssertFalse(prepared.readyForTraining(size: 2048, material: originalID, target: "normal"))
        object.removeValue(forKey: "preparation")
        XCTAssertFalse(try decode().readyForTraining(size: 2048, material: originalID, target: "height"),
                       "Unprepared datasets cannot use another resolution's samples as a folder-selection proof")
        object.removeValue(forKey: "automatic_validation")
        XCTAssertFalse(try decode().readyForTraining(size: 2048, material: originalID, target: "height"))
    }

    func testSelectedOriginalCanTrainFromHigherResolutionInTheSameCanonicalFolder() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let folder = fixture.root.appendingPathComponent("sources/stone").path
        let materials = [
            Self.sourceMaterial(id: "stone_1k", folder: folder, filename: "diffuse-small.png", pixels: 1024, height: false),
            Self.sourceMaterial(id: "stone_2k", folder: folder + "/../stone", filename: "diffuse-large.png", pixels: 2048, height: true)]
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        store.adoptDataset(try Self.sourceDataset(materials: materials, sizes: [1024, 2048]))
        store.selectedSampleId = "stone_1k_sample"
        store.training.useSelectedMaterialOnly = true
        store.selectTrainingModelFamily(.compactScalar)
        store.training.size = 2048
        XCTAssertEqual(Set(store.trainingFolderSamples.map(\.id)), ["stone_1k_sample", "stone_2k_sample"])
        XCTAssertEqual(store.datasetInventorySummary, "1 folder · 3 original files · 2 paired sets")
        XCTAssertEqual(store.trainingCurrentCropSummary, "Currently available: 1 height training crop at 2048 × 2048")
        XCTAssertTrue(store.trainingScopeSummary.contains("includes all resolution tiers"))
        XCTAssertNil(store.trainingConfigurationIssue,
                     "An undersized selected original must not discard another eligible tier in its source folder")
    }

    func testRegisteredProviderWithoutTargetOrSizeSupportCanAttemptPreparation() async throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let folder = fixture.root.appendingPathComponent("sources/stone").path
        var calls: [[String]] = []
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources,
            workerOverride: { arguments, _ in
                calls.append(arguments)
                if arguments.first == "capabilities" { return "{\"training_sizes\":[1024,2048]}" }
                if arguments.first == "prepare-size" { throw StudioError("The provider did not publish the requested height tier.") }
                throw StudioError("Unexpected test command")
            })
        store.selectTrainingModelFamily(.compactScalar)
        store.adoptDataset(try Self.sourceDataset(materials: [
            Self.sourceMaterial(id: "stone_1k", folder: folder, filename: "stone_diff_1k.png", pixels: 1024, height: false)], sizes: []))
        XCTAssertTrue(store.supportedTrainingSizes.contains(1024))
        XCTAssertTrue(store.supportedTrainingSizes.contains(2048))
        try await store.loadTrainingCapabilities()
        XCTAssertEqual(store.supportedTrainingSizes, [1024, 2048])
        store.training.size = 2048
        store.training.useSelectedMaterialOnly = true
        XCTAssertNil(store.trainingConfigurationIssue)
        store.startTraining()
        try await settled(store)
        let preparation = try XCTUnwrap(calls.first { $0.first == "prepare-size" })
        XCTAssertEqual(preparation[try XCTUnwrap(preparation.firstIndex(of: "--size")) + 1], "2048")
        XCTAssertEqual(preparation[try XCTUnwrap(preparation.firstIndex(of: "--target")) + 1], "height")
        XCTAssertEqual(preparation[try XCTUnwrap(preparation.firstIndex(of: "--material")) + 1], "stone_1k")
        XCTAssertEqual(store.error, "The provider did not publish the requested height tier.")
        XCTAssertFalse(calls.contains { $0.first == "train" })
    }

    func testSelectedFolderDoesNotBorrowTargetFromAnUnrelatedFolder() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        store.adoptDataset(try Self.sourceDataset(materials: [
            Self.sourceMaterial(id: "stone_1k", folder: fixture.root.appendingPathComponent("sources/selected").path,
                filename: "diffuse-small.png", pixels: 1024, height: false),
            Self.sourceMaterial(id: "stone_2k", folder: fixture.root.appendingPathComponent("sources/unrelated").path,
                filename: "diffuse-large.png", pixels: 2048, height: true)], sizes: [1024, 2048]))
        store.selectedSampleId = "stone_1k_sample"
        store.selectTrainingModelFamily(.compactScalar)
        store.training.size = 2048
        store.training.useSelectedMaterialOnly = true
        XCTAssertEqual(store.trainingFolderSamples.map(\.id), ["stone_1k_sample"])
        XCTAssertEqual(store.datasetInventorySummary, "2 folders · 3 original files · 2 paired sets")
        XCTAssertEqual(store.trainingCurrentCropSummary, "Currently available: 0 height training crops at 2048 × 2048")
        XCTAssertNotNil(store.trainingConfigurationIssue)
        store.training.useSelectedMaterialOnly = false
        XCTAssertEqual(store.trainingCurrentCropSummary, "Currently available: 1 height training crop at 2048 × 2048")
        XCTAssertNil(store.trainingConfigurationIssue, "The unrelated target remains eligible when the full dataset is selected")
    }

    func testProviderFilenameWithWrongNominalDimensionsDoesNotEnableLargerTiers() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        store.adoptDataset(try Self.sourceDataset(materials: [
            Self.sourceMaterial(id: "stone_1k", folder: fixture.root.appendingPathComponent("sources/stone").path,
                filename: "stone_diff_1k.png", pixels: 256, height: false)], sizes: []))
        store.selectTrainingModelFamily(.compactScalar)
        store.training.size = 2048
        XCTAssertFalse(store.supportedTrainingSizes.contains(1024))
        XCTAssertFalse(store.supportedTrainingSizes.contains(2048))
        XCTAssertNotNil(store.trainingConfigurationIssue,
                        "Provider naming alone cannot establish a genuine original resolution")
    }

    func testAmbientColorOriginalCanRecoverTiersOnlyAtItsVerifiedNominalDimensions() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        let folder = fixture.root.appendingPathComponent("sources/Wicker013").path
        store.selectTrainingModelFamily(.compactScalar)
        store.adoptDataset(try Self.sourceDataset(materials: [
            Self.sourceMaterial(id: "wicker_1k", folder: folder, filename: "Wicker013_1K-PNG_Color.png",
                pixels: 1024, height: false)], sizes: []))
        store.training.size = 2048
        XCTAssertTrue(store.supportedTrainingSizes.contains(2048))
        XCTAssertNil(store.trainingConfigurationIssue)
        store.adoptDataset(try Self.sourceDataset(materials: [
            Self.sourceMaterial(id: "wicker_1k", folder: folder, filename: "Wicker013_1K-PNG_Color.png",
                pixels: 256, height: false)], sizes: []))
        XCTAssertFalse(store.supportedTrainingSizes.contains(2048))
        XCTAssertNotNil(store.trainingConfigurationIssue)
    }

    func testPublishedRectangularProviderFieldsCanRecoverMissingCropTiers() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        store.selectTrainingModelFamily(.compactScalar)
        let fields = [("jersey_diff_4k.png", 4096, 3893), ("scuba_diff_4k.png", 4096, 4005),
                      ("wool_diff_2k.png", 2048, 2538), ("wool_diff_4k.png", 4096, 5076),
                      ("portrait_diff_4k.png", 3893, 4096)]
        for (filename, width, height) in fields {
            store.adoptDataset(try Self.sourceDataset(materials: [
                Self.sourceMaterial(id: "field", folder: fixture.root.appendingPathComponent("sources/field").path,
                    filename: filename, pixels: width, height: false, heightPixels: height)], sizes: []))
            store.training.size = 2048
            XCTAssertTrue(store.supportedTrainingSizes.contains(2048), filename)
            XCTAssertNil(store.trainingConfigurationIssue, filename)
        }
    }

    func testPreparedInventoryCountsResolutionGroupsSeparatelyFromDiagnosticCrops() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let folder = fixture.root.appendingPathComponent("sources/stone").path
        var materials = [1, 2, 4].map { tier in
            Self.sourceMaterial(id: "stone_\(tier)k", folder: folder, filename: "stone_diff_\(tier)k.png",
                pixels: 1024, height: true)
        }
        var larger = materials[2]
        var samples = try XCTUnwrap(larger["samples"] as? [[String: Any]])
        var diagnostic = samples[0]
        diagnostic["sample_id"] = "stone_4k_validation"
        diagnostic["split"] = "validation"
        samples.append(diagnostic)
        larger["samples"] = samples
        materials[2] = larger
        let document: [String: Any] = ["dataset_path": "/dataset/prepared", "index_sha256": "prepared-sha",
            "source_set_count": 4, "training_size": 1024, "materials": materials,
            "preparation": ["source_dataset_path": "/dataset", "source_index_sha256": "source-sha",
                "prepared_dataset_path": "/dataset/prepared", "crop_size": 1024,
                "reused": false, "target_resized": false, "original_dataset_modified": false]]
        let dataset = try WorkbenchResult.decode(WorkbenchDataset.self,
            output: String(decoding: JSONSerialization.data(withJSONObject: document), as: UTF8.self))
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        store.adoptDataset(dataset)
        store.selectTrainingModelFamily(.compactScalar)
        store.training.useSelectedMaterialOnly = false
        XCTAssertEqual(store.datasetInventorySummary, "1 folder · 8 original files · 3 paired sets")
        XCTAssertEqual(store.trainingCurrentCropSummary, "Prepared: 3 height training crops at 1024 × 1024")
    }

    func testRuntimeAndUploadEditsSaveImmediately() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        first.workspacePath = fixture.root.path
        first.modelDirectory = "/located/model"
        first.uploadRepo = "artist/material-height"
        first.uploadPublic = true
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.workspacePath, fixture.root.path)
        XCTAssertEqual(reopened.modelDirectory, "/located/model")
        XCTAssertEqual(reopened.uploadRepo, "artist/material-height")
        XCTAssertTrue(reopened.uploadPublic)
    }

    func testPartialTrainingPreferencesKeepSupportedChoices() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let options = try JSONDecoder().decode(MaterialTrainingOptions.self,
            from: Data("{\"size\":2048}".utf8))
        XCTAssertEqual(options.size, 2048)
        XCTAssertEqual(options.modelName, "")
        XCTAssertEqual(options.target, "height")
        XCTAssertEqual(options.validationEvery, 0)
        XCTAssertEqual(options.validationUnit, .epoch)
        XCTAssertEqual(options.checkpointUnit, .epoch)
        XCTAssertEqual(options.restored(for: fixture.resources), options)
        fixture.defaults.set(Data("{\"training\":{\"size\":2048}}".utf8), forKey: WorkbenchPreferences.key)
        let restored = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(restored.training.size, 2048)
        XCTAssertTrue(restored.comparisonIncludesBase)
    }

    func testTrainingIntervalsDefaultToEpochsAndPersistIndependentUnits() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(first.training.validationEvery, 0)
        XCTAssertEqual(first.training.validationUnit, .epoch)
        XCTAssertEqual(first.training.checkpointEvery, 0)
        XCTAssertEqual(first.training.checkpointUnit, .epoch)
        first.training.validationEvery = 3
        first.training.validationUnit = .step
        first.training.checkpointEvery = 2
        first.training.checkpointUnit = .epoch
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.training, first.training)
        let data = try XCTUnwrap(fixture.defaults.data(forKey: WorkbenchPreferences.key))
        let preferences = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let training = try XCTUnwrap(preferences["training"] as? [String: Any])
        XCTAssertEqual(training["validationUnit"] as? String, "step")
        XCTAssertEqual(training["checkpointUnit"] as? String, "epoch")
    }

    func testLegacyActiveIntervalsRemainStepsAndDisabledIntervalsDefaultToEpochs() throws {
        for (document, unit) in [("{\"validationEvery\":91,\"checkpointEvery\":125}", MaterialTrainingIntervalUnit.step),
                                 ("{\"validationEvery\":0,\"checkpointEvery\":0}", .epoch)] {
            let options = try JSONDecoder().decode(MaterialTrainingOptions.self, from: Data(document.utf8))
            XCTAssertEqual(options.validationUnit, unit)
            XCTAssertEqual(options.checkpointUnit, unit)
            let restored = try JSONDecoder().decode(MaterialTrainingOptions.self, from: JSONEncoder().encode(options))
            XCTAssertEqual(restored, options)
        }
        let partial = try JSONDecoder().decode(MaterialTrainingOptions.self, from: Data("{\"validationEvery\":91}".utf8))
        XCTAssertEqual(partial.validationEvery, 91)
        XCTAssertEqual(partial.validationUnit, .step)
        XCTAssertEqual(partial.checkpointUnit, .epoch)
        let explicit = try JSONDecoder().decode(MaterialTrainingOptions.self,
            from: Data("{\"validationEvery\":91,\"validationUnit\":\"epoch\",\"checkpointEvery\":125,\"checkpointUnit\":\"step\"}".utf8))
        XCTAssertEqual(explicit.validationEvery, 91)
        XCTAssertEqual(explicit.validationUnit, .epoch)
        XCTAssertEqual(explicit.checkpointEvery, 125)
        XCTAssertEqual(explicit.checkpointUnit, .step)
    }

    func testReopeningPreservesZeroQuickChecksAndValuesBeyondFormerUICaps() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        first.training.validationEvery = 0
        first.training.checkpointEvery = 200_000
        first.training.updatesPerCrop = 50_000
        first.training.maxMinutes = 720
        first.training.loraRank = 128
        first.training.loraAlpha = 256
        XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources).training, first.training)
        first.training.validationEvery = Int.max
        first.training.maxMinutes = 0.1
        first.training.loraAlpha = 0.001
        XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources).training, first.training)
    }

    func testReopeningOnSmallerMacPreservesSettingsWithoutMemoryAdmissionAdaptation() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        var options = MaterialTrainingOptions()
        options.size = 2048
        options.updatesPerCrop = 800
        options.maxMinutes = 150
        WorkbenchPreferences(training: options).save(to: fixture.defaults)
        let smaller = MachineResources(physicalBytes: 16 * MachineResources.gibibyte)
        let restored = WorkbenchStore(preferences: fixture.defaults, resources: smaller)
        XCTAssertEqual(restored.training.size, 2048)
        XCTAssertEqual(restored.training.updatesPerCrop, 800)
        XCTAssertEqual(restored.training.maxMinutes, 150)
        XCTAssertTrue(restored.activity.isEmpty)
        fixture.defaults.set(Data("not JSON".utf8), forKey: WorkbenchPreferences.key)
        let fallback = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(fallback.training.size, MaterialTrainingOptions().size)
    }

    func testStudioPreferencesRoundTripAndMissingFieldsKeepDefaultsAvailable() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        var value = StudioPreferences()
        var settings = TextureSettings()
        settings.outputSize = 2048
        value.settings = settings
        value.depthChoice = .photoDetail
        value.modelID = "custom-local-model"
        value.customInverseDepth = false
        value.selectedPreview = "Normal"
        value.showInspector = false
        value.exportDirectory = fixture.root.path
        value.save(to: fixture.defaults)
        XCTAssertEqual(StudioPreferences.load(from: fixture.defaults), value)
        fixture.defaults.set(Data("{\"showInspector\":false}".utf8), forKey: StudioPreferences.key)
        let partial = StudioPreferences.load(from: fixture.defaults)
        XCTAssertEqual(partial.showInspector, false)
        XCTAssertNil(partial.settings)
        XCTAssertNil(partial.depthChoice)
    }

    private func settled(_ store: WorkbenchStore) async throws {
        let deadline = Date().addingTimeInterval(5)
        while store.isBusy, Date() < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.isBusy)
    }
    private static func sourceMaterial(id: String, folder: String, filename: String, pixels: Int, height: Bool,
                                       heightPixels: Int? = nil) -> [String: Any] {
        let imageHeight = heightPixels ?? pixels
        var maps: [String: Any] = ["input": ["path": folder + "/" + filename, "width": pixels, "height": imageHeight,
            "original_source_path": folder + "/" + filename, "original_source_width": pixels, "original_source_height": imageHeight]]
        if height { maps["height"] = ["path": folder + "/height.png", "width": pixels, "height": imageHeight, "source_bits": 16] }
        return ["material_id": id, "source_directory": folder, "samples": [["sample_id": id + "_sample",
            "source_family_id": "stone", "status": "approved", "split": "train", "width": pixels, "height": imageHeight,
            "maps": maps, "available_targets": height ? ["height"] : [String]()]]]
    }
    private static func sourceDataset(materials: [[String: Any]], sizes: [Int]) throws -> WorkbenchDataset {
        let object: [String: Any] = ["dataset_path": "/dataset", "index_sha256": "source-sha",
            "supported_training_sizes": sizes, "materials": materials]
        return try WorkbenchResult.decode(WorkbenchDataset.self,
            output: String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self))
    }
    private static func datasetJSON(size: Int = 1024, prepared: Bool = false) throws -> String {
        let source = "/dataset/dataset.json"
        let path = prepared ? "/dataset/prepared-2048" : source
        let samples: [[String: Any]] = ["soil_001", "soil_002"].map { sample in
            ["sample_id": sample, "status": "approved", "split": "train", "width": size, "height": size,
             "maps": ["input": ["path": "/dataset/\(sample).png", "width": size, "height": size],
                      "height": ["path": "/dataset/\(sample)-height.png", "width": size, "height": size]]]
        }
        var document: [String: Any] = ["dataset_path": path, "index_sha256": prepared ? "prepared-sha" : "source-sha",
            "materials": [["material_id": "soil", "samples": samples]]]
        if prepared {
            document["automatic_validation"] = ["policy": "subject-extra-crops-v2", "material_ids": ["soil"]]
            document["preparation"] = ["source_dataset_path": source, "source_index_sha256": "source-sha",
                "prepared_dataset_path": path, "crop_size": size, "reused": true, "target_resized": false,
                "original_dataset_modified": false, "split_lineage_changed": true]
        }
        return String(decoding: try JSONSerialization.data(withJSONObject: document), as: UTF8.self)
    }

}

@MainActor private struct PreferencesFixture {
    let root: URL
    let defaults: UserDefaults
    let suite = "org.ipde.preference-tests.\(UUID().uuidString)"
    let resources = MachineResources(physicalBytes: 64 * MachineResources.gibibyte,
                                     metalRecommendedBytes: 48 * MachineResources.gibibyte)
    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("preference-tests-\(UUID().uuidString)")
        defaults = UserDefaults(suiteName: suite)!
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defaults.set(root.path, forKey: "workspace")
    }
    func remove() {
        defaults.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: root)
    }
}
