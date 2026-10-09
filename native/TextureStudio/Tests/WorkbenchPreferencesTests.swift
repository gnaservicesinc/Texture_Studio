import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchPreferencesTests: XCTestCase {
    func testFormAndComparisonChoicesSaveWithoutStartingWorkerOrClosingWindow() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        first.training.target = "normal"
        first.training.size = 2048
        first.training.updatesPerCrop = 875
        first.training.maxMinutes = 120
        first.training.memoryGB = 56
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
        XCTAssertEqual(reopened.training.memoryGB, 56, "A valid setting above the recommendation must stay selected")
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

    func testAutomaticBudgetFollowsResolutionAndManualOverridePersists() async throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        var budgets: [Double] = []
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources, workerOverride: { arguments, _ in
            let index = try XCTUnwrap(arguments.firstIndex(of: "--memory-gib"))
            budgets.append(try XCTUnwrap(Double(arguments[index + 1])))
            return """
            {"training_sizes":[1024,2048],"memory_plans":{
                "1024":{"required_memory_gib":15,"recommended_memory_gib":18},
                "2048":{"required_memory_gib":47.5,"recommended_memory_gib":48}}}
            """.replacingOccurrences(of: "\n", with: "")
        })
        store.training.size = 2048
        store.training.memoryGB = 8
        try await store.loadTrainingCapabilities()
        XCTAssertEqual(store.training.size, 2048, "A stale small budget cannot hide a resolution the machine can train")
        XCTAssertEqual(store.training.memoryGB, 48)
        XCTAssertEqual(budgets.last, fixture.resources.maximumTrainingGiB)
        store.selectTrainingSize(1024)
        XCTAssertEqual(store.training.memoryGB, 18)
        store.selectTrainingSize(2048)
        XCTAssertEqual(store.training.memoryGB, 48)
        store.training.automaticMemory = false
        store.training.memoryGB = 50
        try await store.loadTrainingCapabilities()
        XCTAssertEqual(budgets.last, 50)
        XCTAssertEqual(store.training.memoryGB, 50)
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertFalse(reopened.training.automaticMemory)
        XCTAssertEqual(reopened.training.memoryGB, 50)
    }

    func testRuntimeAndUploadEditsSaveImmediately() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        first.workspacePath = fixture.root.path
        first.pythonPath = "/located/python"
        first.modelDirectory = "/located/model"
        first.codeDirectory = "/located/code"
        first.uploadRepo = "artist/material-height"
        first.uploadPublic = true
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.workspacePath, fixture.root.path)
        XCTAssertEqual(reopened.pythonPath, "/located/python")
        XCTAssertEqual(reopened.modelDirectory, "/located/model")
        XCTAssertEqual(reopened.codeDirectory, "/located/code")
        XCTAssertEqual(reopened.uploadRepo, "artist/material-height")
        XCTAssertTrue(reopened.uploadPublic)
    }

    func testPartialTrainingPreferencesKeepSupportedChoices() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let options = try JSONDecoder().decode(MaterialTrainingOptions.self,
            from: Data("{\"size\":2048,\"memoryGB\":56}".utf8))
        XCTAssertEqual(options.size, 2048)
        XCTAssertEqual(options.memoryGB, 56)
        XCTAssertEqual(options.target, "height")
        XCTAssertEqual(options.restored(for: fixture.resources), options)
        fixture.defaults.set(Data("{\"training\":{\"size\":2048,\"memoryGB\":56}}".utf8), forKey: WorkbenchPreferences.key)
        let restored = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(restored.training.size, 2048)
        XCTAssertEqual(restored.training.memoryGB, 56)
        XCTAssertTrue(restored.comparisonIncludesBase)
    }

    func testOnlyUnsupportedStoredResourceChoicesAreAdaptedToThisMac() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        var options = MaterialTrainingOptions()
        options.memoryGB = 56
        options.size = 2048
        options.updatesPerCrop = 800
        options.maxMinutes = 150
        WorkbenchPreferences(training: options).save(to: fixture.defaults)
        let smaller = MachineResources(physicalBytes: 16 * MachineResources.gibibyte)
        let restored = WorkbenchStore(preferences: fixture.defaults, resources: smaller)
        XCTAssertEqual(restored.training.memoryGB, smaller.maximumTrainingGiB)
        XCTAssertEqual(restored.training.size, 2048)
        XCTAssertEqual(restored.training.updatesPerCrop, 800)
        XCTAssertEqual(restored.training.maxMinutes, 150)
        XCTAssertTrue(restored.activity.contains("adapted"), "A resource adaptation must be explained")
        fixture.defaults.set(Data("not JSON".utf8), forKey: WorkbenchPreferences.key)
        let fallback = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(fallback.training.memoryGB, fixture.resources.defaultTrainingGiB)
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
