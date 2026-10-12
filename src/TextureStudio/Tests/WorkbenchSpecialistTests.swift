import Foundation
import XCTest
@testable import TextureStudio

@MainActor final class WorkbenchSpecialistTests: XCTestCase {
    func testRefreshKeepsMainDatasetAndPersistsIndependentSpecialistPaths() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in calls.append(args); return Self.registry }
        store.specialistSourceDatasetPath = "/original/main"
        store.specialistWorkspacePath = "/specialist/library"
        store.specialistClassifierPath = "/models/Classifier.mlmodel"
        store.classifierDatasetPath = "/curated/classes"
        store.classifierSourceFolderPath = "/extra/materials"
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.dataset(path: "/original/main"))
        store.refreshSpecialists()
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.first, "refresh-specialists")
        XCTAssertTrue(calls[0].contains("/models/Classifier.mlmodel"))
        XCTAssertTrue(calls[0].contains("--source-folder"))
        XCTAssertEqual(store.dataset?.datasetPath, "/original/main")
        XCTAssertEqual(store.specialistRegistry?.classifierOnlyCount, 7)
        XCTAssertEqual(store.selectedSpecialist?.name, "Wood")
        let reopened = fixture.store { _, _ in "" }
        XCTAssertEqual(reopened.specialistWorkspacePath, "/specialist/library")
        XCTAssertEqual(reopened.specialistSourceDatasetPath, "/original/main")
        XCTAssertEqual(reopened.specialistClassifierPath, "/models/Classifier.mlmodel")
    }

    func testGuidedTrainerSelectsCategoryTargetAndContinuationWithoutPBRBase() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            switch args.first {
            case "capabilities": return "{\"training_sizes\":[256,512,1024,2048]}"
            case "dataset": return Self.dataset(path: "/specialist/wood")
            case "checkpoint": return """
            {"checkpoint_path":"/checkpoints/wood.safetensors","sha256":"model-sha","schema":"texture-studio-compact-material-v1",
            "architecture":"texture-studio-compact-normal-native-v1","target":"normal","step":4,"compatible":true,"variant":"full","supports_training_warm_start":true}
            """
            default: XCTFail("Unexpected operation \(args)"); return "{}"
            }
        }
        store.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self, output: Self.registry)
        store.specialistTarget = "normal"
        store.openSpecialistTrainer(continuing: true)
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["capabilities", "dataset", "checkpoint"])
        XCTAssertFalse(calls.joined().contains("--model-directory"))
        XCTAssertEqual(store.training.modelFamily, .compactNormal)
        XCTAssertEqual(store.training.target, "normal")
        XCTAssertEqual(store.training.modelName, "Wood Normals")
        XCTAssertFalse(store.training.useSelectedMaterialOnly)
        XCTAssertTrue(store.training.useWarmStart)
        XCTAssertEqual(store.dataset?.datasetPath, "/specialist/wood")
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "wood")
        XCTAssertEqual(store.specialistTrainingContext?.membershipSHA256, "normal-members")
    }

    func testMissingPairedMapStopsSetupAndNextSpecialistSkipsFinishedModels() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store { _, _ in XCTFail("Missing maps must not launch a worker"); return "" }
        store.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self, output: Self.registry)
        store.specialistTarget = "roughness"
        store.openSpecialistTrainer()
        XCTAssertFalse(store.isBusy)
        XCTAssertNotNil(store.error)
        store.specialistTarget = "height"
        store.selectedSpecialistID = "wood"
        store.selectNextSpecialist()
        XCTAssertEqual(store.selectedSpecialistID, "bark")
    }

    func testOriginalLibraryChoiceSurvivesTrainingDatasetSelectionAndRelaunch() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.defaults.set("/original/main", forKey: "dataset")
        let initial = fixture.store { _, _ in "" }
        XCTAssertEqual(initial.specialistSourceDatasetPath, "/original/main")
        fixture.defaults.set("/specialist/wood", forKey: "dataset")
        let reopened = fixture.store { _, _ in "" }
        XCTAssertEqual(reopened.specialistSourceDatasetPath, "/original/main")
        XCTAssertEqual(reopened.specialistWorkspacePath, initial.specialistWorkspacePath)
    }

    func testCompletedModelRegistrationCarriesCapturedMembershipAndActualUpdates() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in calls.append(args); return Self.registry }
        let context = WorkbenchSpecialistTrainingContext(workspacePath: "/library", specialistID: "wood", name: "Wood",
            target: "normal", datasetPath: "/specialist/wood", membershipSHA256: "captured-members")
        try await store.recordSpecialistCheckpoint(context: context, checkpointPath: "/trained/model.safetensors",
            trainingSize: 1024, completedUpdates: 42)
        XCTAssertEqual(calls[0].first, "record-specialist-model")
        XCTAssertEqual(value("--completed-updates", in: calls[0]), "42")
        XCTAssertEqual(value("--membership-sha256", in: calls[0]), "captured-members")
        XCTAssertEqual(value("--target", in: calls[0]), "normal")
    }

    private func value(_ flag: String, in args: [String]) -> String? {
        args.firstIndex(of: flag).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil }
    }
    private func wait(_ store: WorkbenchStore) async throws {
        for _ in 0..<500 where store.isBusy { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isBusy)
    }
    @MainActor private struct Fixture {
        let root: URL
        let defaults: UserDefaults
        let suite: String
        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("specialist-ui-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            suite = "specialist-ui-\(UUID().uuidString)"
            defaults = UserDefaults(suiteName: suite)!
        }
        func store(worker: @escaping @MainActor ([String], String) async throws -> String) -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, managedWorkspaceURL: root,
                selectedCheckpointRegistryURL: root.appendingPathComponent("selected.json"), workerOverride: worker)
        }
        func remove() { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
    }
    private static func dataset(path: String) -> String {
        """
        {"dataset_path":"\(path)","index_sha256":"dataset-sha","supported_training_sizes":[256,512,1024],"materials":[]}
        """
    }
    private static let registry = """
    {"workspace_path":"/library","source_dataset_path":"/original/main","classifier_dataset_path":"/curated/classes","classifier_path":"/models/Classifier.mlmodel",
    "classifier_sha256":"classifier-sha","classifier_labels":["Wood","Bark"],"updated_at":"2026-10-11","classifier_only_count":7,"material_count":12,"family_count":4,"warnings":[],
    "specialists":[{"id":"wood","name":"Wood","class_label":"Wood","subtype":null,"is_custom":false,"dataset_path":"/specialist/wood","material_ids":["planks_1k"],"family_ids":["planks"],
    "targets":[{"target":"height","material_count":1,"status":"ready","membership_sha256":"height-members"},{"target":"normal","material_count":1,"status":"ready","membership_sha256":"normal-members"}],
    "models":[{"id":"model-sha","target":"normal","checkpoint_path":"/checkpoints/wood.safetensors","checkpoint_sha256":"model-sha","training_size":1024,"completed_at":"2026-10-11","membership_sha256":"normal-members","approved":false}]},
    {"id":"bark","name":"Bark","class_label":"Bark","subtype":null,"is_custom":false,"dataset_path":"/specialist/bark","material_ids":["bark_1k"],"family_ids":["bark"],
    "targets":[{"target":"height","material_count":1,"status":"ready","membership_sha256":"bark-members"}],"models":[]}]}
    """
}
