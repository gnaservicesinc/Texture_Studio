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
        store.classifierUseSourceFolder = true
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

    func testModelWorkshopUsesOpenSourceFolderAndIgnoresUnselectedGlobalCorpus() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in calls.append(args); return Self.registry }
        store.specialistSourceDatasetPath = "/stale/dataset"
        store.classifierSourceFolderPath = "/global/extra"
        store.classifierUseSourceFolder = false
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
            {"dataset_path":"/managed/reference","index_sha256":"source-sha","materials":[],
             "is_source_folder":true,"source_folder_path":"/original/photos"}
            """)
        XCTAssertEqual(store.modelWorkshopSourcePath, "/original/photos")
        store.refreshSpecialists()
        try await wait(store)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(value("--dataset", in: call), "/original/photos")
        XCTAssertFalse(call.contains("--source-folder"), "An unrelated global corpus cannot silently enter the open folder's workshop")
        XCTAssertEqual(store.specialistSourceDatasetPath, "/original/photos")
    }

    func testModelWorkshopReturnsToOriginalSourcesAfterCategoryTrainingSelection() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            return Self.dataset(path: "/original/main")
        }
        store.specialistSourceDatasetPath = "/original/main"
        store.specialistTrainingContext = .init(workspacePath: "/library", specialistID: "wood", name: "Wood",
            target: "normal", datasetPath: "/specialist/wood", membershipSHA256: "members")
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.dataset(path: "/specialist/wood"))
        store.openModelWorkshopSources()
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.dataset?.datasetPath, "/original/main")
        XCTAssertNil(store.specialistTrainingContext)
        XCTAssertEqual(store.datasetBrowserMode, .originals)
        XCTAssertEqual(calls.first?.first, "dataset")
        XCTAssertTrue(calls.first?.contains("--skip-source-recovery") == true)
    }

    func testReturningToWorkshopFromCategoryComparisonRestoresOriginalRoot() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in calls.append(args); return Self.originalFolder(path: "/original/main") }
        store.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self, output: Self.registry)
        store.specialistSourceDatasetPath = "/original/main"
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.dataset(path: "/specialist/wood"))
        XCTAssertNil(store.specialistTrainingContext, "Comparison may select a category without a training context")
        XCTAssertEqual(store.modelWorkshopSourcePath, "/original/main")
        store.returnToModelWorkshop()
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(value("--dataset", in: try XCTUnwrap(calls.first)), "/original/main")
        XCTAssertEqual(store.dataset?.datasetPath, "/original/main")
        XCTAssertNotNil(store.specialistNavigationRequest)
        XCTAssertEqual(store.datasetBrowserMode, .originals)
    }

    func testWorkshopAddCopiesIntoOriginalRootAfterCategoryTrainingSelection() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let source = fixture.root.appendingPathComponent("Photo.jpg")
        try Data("Original image bytes".utf8).write(to: source)
        let store = fixture.store { args, _ in
            calls.append(args)
            return args.first == "add-source" ? "{\"completed\":true}" : Self.originalFolder(path: "/original/main")
        }
        store.specialistSourceDatasetPath = "/original/main"
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.dataset(path: "/specialist/wood"))
        store.specialistTrainingContext = .init(workspacePath: "/library", specialistID: "wood", name: "Wood",
            target: "height", datasetPath: "/specialist/wood", membershipSHA256: "members")
        store.addModelWorkshopSourceFiles([source])
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["dataset", "add-source", "dataset"])
        XCTAssertTrue(calls.allSatisfy { value("--dataset", in: $0) == "/original/main" })
        XCTAssertNil(store.specialistTrainingContext)
        XCTAssertEqual(store.dataset?.datasetPath, "/original/main")
        XCTAssertEqual(try Data(contentsOf: source), Data("Original image bytes".utf8))
    }

    func testWorkshopClassifierExamplesUseOriginalRootAndItsGridAfterCategorySelection() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            if args.first == "dataset" { return Self.originalFolder(path: "/original/main") }
            return """
                {"dataset_path":"/classes","manifest_path":"/classes/manifest","review_path":"/classes/review.html",
                 "total":0,"added":0,"preserved":0,"needs_review":0,"classes":[],"warnings":[]}
                """
        }
        store.specialistSourceDatasetPath = "/original/main"
        store.classifierUseSourceFolder = false
        store.classifierDatasetPath = "/classes"
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.dataset(path: "/specialist/wood"))
        store.specialistTrainingContext = .init(workspacePath: "/library", specialistID: "wood", name: "Wood",
            target: "height", datasetPath: "/specialist/wood", membershipSHA256: "members")
        store.refreshClassifierDataset()
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["dataset", "refresh-classifier-dataset"])
        let refresh = try XCTUnwrap(calls.last)
        XCTAssertEqual(value("--dataset", in: refresh), "/original/main")
        XCTAssertEqual(value("--size", in: refresh), "2048")
        XCTAssertNil(store.specialistTrainingContext)
    }

    func testOpeningAnotherSourceFolderClearsOldWorkshopTrainingContext() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            return args.first == "refresh-classifier-dataset" ? Self.classifierExamples : Self.originalFolder(path: "/new/originals")
        }
        store.specialistSourceDatasetPath = "/old/originals"
        store.classifierUseSourceFolder = true
        store.classifierSourceFolderPath = "/stale/1k-sources"
        store.classifierSourceMinimumSize = 1024
        store.specialistTrainingContext = .init(workspacePath: "/library", specialistID: "wood", name: "Wood",
            target: "height", datasetPath: "/specialist/wood", membershipSHA256: "members")
        store.openDataset(URL(fileURLWithPath: "/new/originals"))
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertNil(store.specialistTrainingContext)
        XCTAssertEqual(store.specialistSourceDatasetPath, "/new/originals")
        XCTAssertEqual(store.modelWorkshopSourcePath, "/new/originals")
        XCTAssertFalse(store.classifierUseSourceFolder)
        store.refreshClassifierDataset()
        try await wait(store)
        XCTAssertNil(store.error)
        let refresh = try XCTUnwrap(calls.last)
        XCTAssertEqual(value("--dataset", in: refresh), "/new/originals")
        XCTAssertEqual(value("--size", in: refresh), "2048")
        XCTAssertFalse(refresh.contains("--source-folder"))
    }

    func testCreatingBareSourceFolderReplacesStaleClassifierOverrideAndCategoryContext() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let newFolder = fixture.root.appendingPathComponent("New originals")
        let store = fixture.store { args, _ in
            calls.append(args)
            return args.first == "refresh-classifier-dataset" ? Self.classifierExamples : Self.originalFolder(path: newFolder.path)
        }
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.dataset(path: "/specialist/wood"))
        store.specialistSourceDatasetPath = "/old/originals"
        store.specialistTrainingContext = .init(workspacePath: "/library", specialistID: "wood", name: "Wood",
            target: "height", datasetPath: "/specialist/wood", membershipSHA256: "members")
        store.classifierUseSourceFolder = true
        store.classifierSourceFolderPath = "/stale/1k-sources"
        store.classifierSourceMinimumSize = 1024
        store.createDataset(name: "New originals", description: "", parentURL: fixture.root)
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertFalse(store.classifierUseSourceFolder)
        XCTAssertNil(store.specialistTrainingContext)
        XCTAssertEqual(store.specialistSourceDatasetPath, newFolder.path)
        XCTAssertEqual(store.datasetResolution, 2048)
        store.refreshClassifierDataset()
        try await wait(store)
        XCTAssertNil(store.error)
        let refresh = try XCTUnwrap(calls.last)
        XCTAssertEqual(value("--dataset", in: refresh), newFolder.path)
        XCTAssertEqual(value("--size", in: refresh), "2048")
        XCTAssertFalse(refresh.contains("--source-folder"))
    }

    func testInternalCategoryLoadingPreservesExplicitClassifierOverride() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = fixture.store { _, _ in Self.dataset(path: "/specialist/wood") }
        store.classifierUseSourceFolder = true
        store.classifierSourceFolderPath = "/chosen/extra"
        store.classifierSourceMinimumSize = 4096
        try await store.loadDataset(URL(fileURLWithPath: "/specialist/wood"))
        XCTAssertTrue(store.classifierUseSourceFolder)
        XCTAssertEqual(store.classifierSourceFolderPath, "/chosen/extra")
        XCTAssertEqual(store.classifierSourceMinimumSize, 4096)
    }

    func testModelWorkshopShowsImageTagsOnceAcrossSubtypeCategories() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(Self.registry.utf8)) as? [String: Any])
        var specialists = try XCTUnwrap(object["specialists"] as? [[String: Any]])
        let assignment: [String: Any] = ["family_id": "wood_planks", "input_path": "/sources/wood_basecolor.png",
            "class_label": "Wood", "subtype": "planks", "semantic_terms": ["boards", "wood"]]
        for index in specialists.indices { specialists[index]["assignments"] = [assignment] }
        object["specialists"] = specialists
        let store = fixture.store { _, _ in "" }
        store.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self,
            output: String(decoding: JSONSerialization.data(withJSONObject: object), as: UTF8.self))
        XCTAssertEqual(store.modelWorkshopImageTags.count, 1)
        XCTAssertEqual(store.modelWorkshopImageTags.first?.tags, ["Wood", "planks"])
    }

    func testModelWorkshopShowsSavedOriginalLabelsWithoutClassifierRegistry() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = fixture.store { _, _ in "" }
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
            {"dataset_path":"/sources","index_sha256":"inventory-sha","materials":[],
             "source_inventory":[{"path":"/sources/leather.png","role":"input","class_label":"Leather","source_verification":"verified"},
               {"path":"/sources/unknown.jpg","role":"unknown","class_label":"Custom Material","source_verification":"header_inspected"},
               {"path":"/sources/unlabeled.png","role":"unknown"}]}
            """)
        XCTAssertNil(store.specialistRegistry)
        XCTAssertEqual(store.modelWorkshopTaggedOriginals.count, 2)
        XCTAssertEqual(store.modelWorkshopTaggedOriginals.first?.classLabel, "Leather")
        XCTAssertEqual(store.modelWorkshopTaggedOriginals.first?.sourceVerification, "verified")
        XCTAssertEqual(store.modelWorkshopTaggedOriginals.last?.role, "unknown")
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
    private static func originalFolder(path: String) -> String {
        """
        {"dataset_path":"\(path)","source_folder_path":"\(path)","is_source_folder":true,
         "index_sha256":"original-sha","training_size":2048,"materials":[]}
        """
    }
    private static let classifierExamples = """
    {"dataset_path":"/classes","manifest_path":"/classes/manifest","review_path":"/classes/review.html",
     "total":0,"added":0,"preserved":0,"needs_review":0,"classes":[],"warnings":[]}
    """
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
