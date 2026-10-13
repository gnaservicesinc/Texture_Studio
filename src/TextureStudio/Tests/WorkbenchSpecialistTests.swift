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
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
            {"dataset_path":"/specialist/wood","source_folder_path":"/original/main","is_source_folder":false,
             "index_sha256":"current","training_size":2048,"supported_training_sizes":[1024,2048],"materials":[]}
            """)
        XCTAssertTrue(store.modelWorkshopNeedsOriginalSources, "A category's original-source link does not make its projection the original folder")
        store.openModelWorkshopSources()
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.dataset?.datasetPath, "/original/main")
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "wood",
                       "Browsing original files must preserve the trainer's tag category")
        XCTAssertEqual(store.datasetBrowserMode, .originals)
        XCTAssertEqual(calls.first?.first, "dataset")
        XCTAssertTrue(calls.first?.contains("--skip-source-recovery") == true)
    }

    func testTrainerCategoryAndGridSurviveOriginalBrowsingAndRelaunch() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            return Self.originalFolder(path: self.value("--dataset", in: args) ?? "")
        }
        store.specialistSourceDatasetPath = "/original/main"
        store.specialistTrainingContext = .init(workspacePath: "/library", specialistID: "wood", name: "Wood",
            target: "height", datasetPath: "/specialist/wood", membershipSHA256: "members")
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.dataset(path: "/specialist/wood"))
        store.training.size = 2048
        store.returnToModelWorkshop()
        try await wait(store)
        XCTAssertEqual(store.dataset?.datasetPath, "/original/main")
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "wood")
        XCTAssertEqual(store.training.size, 2048)
        store.openActiveTrainerDataset()
        try await wait(store)
        XCTAssertEqual(store.dataset?.datasetPath, "/specialist/wood")
        XCTAssertEqual(value("--review-size", in: try XCTUnwrap(calls.last)), "2048")
        XCTAssertEqual(fixture.store { _, _ in "" }.specialistTrainingContext?.specialistID, "wood")
    }

    func testCategoryResolutionSelectionReadsProjectionWithoutEditingMembership() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let json = """
            {"dataset_path":"/specialist/wood","source_folder_path":"/original/main","is_source_folder":false,
             "index_sha256":"current","training_size":1024,"supported_training_sizes":[1024,2048],"materials":[]}
            """
        let store = fixture.store { args, _ in calls.append(args); return json }
        store.adoptDataset(try WorkbenchResult.decode(WorkbenchDataset.self, output: json))
        store.selectTrainingSize(2048)
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["dataset"])
        XCTAssertEqual(value("--review-size", in: calls[0]), "2048")
        XCTAssertEqual(store.training.size, 2048)
    }

    func testTrainingRefreshesTagMembershipAndPreservesRequestedGrid() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            return args.first == "refresh-specialists" ? Self.registry : Self.dataset(path: "/specialist/wood")
        }
        store.specialistSourceDatasetPath = "/original/main"
        let context = WorkbenchSpecialistTrainingContext(workspacePath: "/library", specialistID: "wood", name: "Wood",
            target: "height", datasetPath: "/old/wood", membershipSHA256: "stale")
        store.specialistTrainingContext = context
        var options = store.training
        options.size = 2048
        let updated = try await store.refreshActiveTrainingCategory(context, options: options)
        XCTAssertEqual(updated?.membershipSHA256, "height-members")
        XCTAssertEqual(updated?.datasetPath, "/specialist/wood")
        XCTAssertEqual(calls.map { $0[0] }, ["refresh-specialists", "dataset"])
        XCTAssertEqual(value("--dataset", in: calls[0]), "/original/main")
        XCTAssertEqual(value("--review-size", in: calls[1]), "2048")
        XCTAssertEqual(store.training.size, 2048)
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
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "wood")
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
        XCTAssertEqual(value("--size", in: refresh), "1024")
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "wood")
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
        store.training.size = 2048
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

    func testOpeningUnrelatedSourceAutomaticallySelectsAndReusesItsOwnWorkspace() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let first = fixture.root.appendingPathComponent("First sources"), second = fixture.root.appendingPathComponent("Second sources")
        for source in [first, second] { try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true) }
        let originalWorkspace = fixture.root.appendingPathComponent("Original model workspace")
        try writeModelWorkspace(originalWorkspace, source: first)
        let originalRegistry = try Data(contentsOf: originalWorkspace.appendingPathComponent("specialists.json"))
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            if args.first == "refresh-specialists" {
                let workspace = URL(fileURLWithPath: try XCTUnwrap(self.value("--workspace", in: args)))
                let source = URL(fileURLWithPath: try XCTUnwrap(self.value("--dataset", in: args)))
                XCTAssertTrue(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: source))
                return try self.modelWorkspaceDocument(workspace, source: source)
            }
            return Self.originalFolder(path: try XCTUnwrap(self.value("--dataset", in: args)))
        }
        store.specialistWorkspacePath = originalWorkspace.path
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.originalFolder(path: first.path))
        await store.loadSpecialistRegistry()
        XCTAssertEqual(store.selectedSpecialist?.name, "Wood")
        store.openDataset(second)
        try await wait(store)
        XCTAssertNil(store.error)
        let secondWorkspace = store.specialistWorkspacePath
        XCTAssertNotEqual(secondWorkspace, originalWorkspace.path)
        XCTAssertNil(store.specialistRegistry, "Categories from another source must disappear immediately after opening")
        XCTAssertNil(store.selectedSpecialistID)
        XCTAssertEqual(calls.map { $0[0] }, ["dataset"], "Opening selects metadata without classification or model inspection")
        store.refreshSpecialists()
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(value("--workspace", in: try XCTUnwrap(calls.last)), secondWorkspace)
        XCTAssertEqual(value("--dataset", in: try XCTUnwrap(calls.last)), second.path)
        try writeModelWorkspace(URL(fileURLWithPath: secondWorkspace), source: second)
        store.openDataset(first)
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.specialistWorkspacePath, originalWorkspace.path)
        XCTAssertEqual(store.selectedSpecialist?.name, "Wood")
        store.openDataset(second)
        try await wait(store)
        XCTAssertEqual(store.specialistWorkspacePath, secondWorkspace)
        XCTAssertEqual(try Data(contentsOf: originalWorkspace.appendingPathComponent("specialists.json")), originalRegistry,
                       "Switching source roots must preserve the other workspace")
    }

    func testLegacyExternalReferenceListCannotBindAnEmptyFolderToOriginalCorpus() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let originals = fixture.root.appendingPathComponent("Original images"), legacy = fixture.root.appendingPathComponent("Legacy dataset")
        try FileManager.default.createDirectory(at: originals, withIntermediateDirectories: true)
        let image = originals.appendingPathComponent("wood_input.png")
        try Data("Original pixel bytes are never read by workspace selection".utf8).write(to: image)
        let sample = legacy.appendingPathComponent("samples/wood")
        try FileManager.default.createDirectory(at: sample, withIntermediateDirectories: true)
        let sha = String(repeating: "a", count: 64)
        let source: [String: Any] = ["path": image.path, "width": 256, "height": 256, "file_sha256": sha]
        let details: [String: Any] = ["storage": "source_reference", "source": source, "sample_sha256": sha]
        let record: [String: Any] = ["material_id": "wood", "sample_pixel_dimensions": [256, 256],
            "maps": ["input": image.path], "map_metadata": ["input": details]]
        try JSONSerialization.data(withJSONObject: record).write(to: sample.appendingPathComponent("sample.json"))
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "storage_policy": "original-source-references-v1",
            "samples": [["path": "samples/wood"]]]).write(to: legacy.appendingPathComponent("dataset.json"))
        let workspace = fixture.root.appendingPathComponent("Existing models")
        try writeModelWorkspace(workspace, source: legacy)
        let before = try Data(contentsOf: workspace.appendingPathComponent("specialists.json"))
        var calls: [[String]] = []
        let store = fixture.store { args, _ in calls.append(args); return Self.originalFolder(path: originals.path) }
        store.specialistWorkspacePath = workspace.path
        store.openDataset(originals)
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertNotEqual(store.specialistWorkspacePath, workspace.path,
            "External paths in an obsolete member list cannot establish physical folder membership")
        XCTAssertNil(store.specialistRegistry)
        XCTAssertEqual(store.modelWorkshopSourcePath, originals.path)
        XCTAssertEqual(calls.map { $0[0] }, ["dataset"])
        XCTAssertEqual(try Data(contentsOf: workspace.appendingPathComponent("specialists.json")), before,
                       "Opening only selects the compatible workspace; explicit refresh performs migration")
        XCTAssertEqual(try Data(contentsOf: image), Data("Original pixel bytes are never read by workspace selection".utf8))
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
            case "refresh-specialists": return Self.registry
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
        XCTAssertEqual(calls.map { $0[0] }, ["capabilities", "refresh-specialists", "dataset", "checkpoint"])
        XCTAssertFalse(calls.joined().contains("--model-directory"))
        XCTAssertEqual(store.training.modelFamily, .compactNormal)
        XCTAssertEqual(store.training.target, "normal")
        XCTAssertEqual(store.training.modelName, "Wood Normals")
        XCTAssertTrue(store.training.useWarmStart)
        XCTAssertEqual(store.dataset?.datasetPath, "/specialist/wood")
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "wood")
        XCTAssertEqual(store.specialistTrainingContext?.membershipSHA256, "normal-members")
    }

    func testCategoryTrainerUsesOriginal2048GridInsteadOfLegacy1024Default() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            if args.first == "capabilities" { return "{\"training_sizes\":[256,512,1024,2048]}" }
            if args.first == "refresh-specialists" { return Self.registry }
            if self.value("--dataset", in: args) == "/original/main" { return Self.originalFolder(path: "/original/main") }
            return """
                {"dataset_path":"/specialist/wood","index_sha256":"category-sha","training_size":1024,
                 "review_size":2048,"supported_training_sizes":[1024,2048],
                 "materials":[{"material_id":"wood_2k","samples":[{"sample_id":"wood_2k_full","status":"unreviewed",
                   "split":"train","width":2048,"height":2048,"available_targets":["height"],
                   "maps":{"input":{"path":"/sources/wood_diff_2k.png","width":2048,"height":2048},
                           "height":{"path":"/sources/wood_disp_2k.png","width":2048,"height":2048,"source_bits":16}}}]}],
                 "original_materials":[{"material_id":"wood_1k","samples":[{"sample_id":"wood_1k_full","status":"unreviewed",
                   "split":"original","width":1024,"height":1024,"maps":{}}]}]}
                """
        }
        store.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self, output: Self.registry)
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.originalFolder(path: "/original/main"))
        store.training.size = 2048
        store.specialistTarget = "height"
        store.openSpecialistTrainer()
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["capabilities", "refresh-specialists", "dataset"])
        XCTAssertEqual(value("--review-size", in: try XCTUnwrap(calls.last)), "2048")
        XCTAssertEqual(store.training.size, 2048)
        XCTAssertEqual(store.datasetResolution, 2048)
        XCTAssertEqual(store.dataset?.reviewSize, 2048)
        XCTAssertFalse(store.samples.isEmpty)
        XCTAssertTrue(store.samples.allSatisfy { $0.width == 2048 && $0.height == 2048 })
        XCTAssertFalse(store.samples.contains { $0.width == 1024 || $0.height == 1024 })
        XCTAssertEqual(store.dataset?.originalSamples.first?.width, 1024, "Originals remain visible separately from the selected training projection")
        XCTAssertNil(store.trainingConfigurationIssue)

        calls = []
        store.specialistTrainingContext = nil
        store.dataset?.trainingSize = 1024
        store.training.size = 1024
        store.openSpecialistTrainer()
        try await wait(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["dataset", "capabilities", "refresh-specialists", "dataset"],
                       "A category comparison restores the original source grid before another training handoff")
        XCTAssertEqual(value("--dataset", in: try XCTUnwrap(calls.first)), "/original/main")
        XCTAssertEqual(value("--review-size", in: try XCTUnwrap(calls.last)), "1024")
        XCTAssertEqual(store.training.size, 1024, "Explicit training choice survives original-folder navigation")
    }

    func testLiveMissingPairedMapStopsSetupAndNextSpecialistSkipsFinishedModels() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            if args.first == "capabilities" { return "{\"training_sizes\":[256,512,1024,2048]}" }
            XCTAssertEqual(args.first, "refresh-specialists")
            return Self.registry
        }
        store.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self, output: Self.registry)
        store.specialistTarget = "roughness"
        store.openSpecialistTrainer()
        try await wait(store)
        XCTAssertFalse(store.isBusy)
        XCTAssertTrue(store.error?.contains("no included paired roughness maps") == true)
        XCTAssertEqual(calls.map { $0[0] }, ["capabilities", "refresh-specialists"])
        store.specialistTarget = "height"
        store.selectedSpecialistID = "wood"
        store.selectNextSpecialist()
        XCTAssertEqual(store.selectedSpecialistID, "bark")
    }

    func testOpeningCachedEmptyCategoryChecksRetaggedAndNewPhysicalMaterialsAtSavedGrid() async throws {
        for retagExisting in [true, false] {
            let fixture = try Fixture(); defer { fixture.remove() }
            let originals = fixture.root.appendingPathComponent("Originals")
            let workspace = fixture.root.appendingPathComponent("Models")
            try FileManager.default.createDirectory(at: originals, withIntermediateDirectories: true)
            let existing = try writeTaggedMaterial("Existing", label: "Bark", in: originals)
            let classifier = NativeMaterialSpecialistService.Classifier(path: "/test/offline", sha256: String(repeating: "a", count: 64),
                labels: ["Wood", "Bark"], minimumImageSize: 0) { _ in
                    throw StudioError("Explicit image tags must determine this category without a prediction")
                }
            let previous = try await NativeMaterialSpecialistService.refresh(datasetURL: originals,
                workspaceURL: workspace, classifier: classifier)
            let emptyWood = try XCTUnwrap(previous.specialists.first { $0.classLabel == "Wood" })
            XCTAssertEqual(emptyWood.targets.first { $0.target == "height" }?.materialCount, 0)
            let selectedInput: URL
            if retagExisting {
                for file in existing { try NativeSourceFileTags.update(file, classLabel: "Wood", classifierIdentity: "user") }
                selectedInput = existing[0]
            } else {
                selectedInput = try writeTaggedMaterial("Added", label: "Wood", in: originals)[0]
            }
            let originalBytes = try Data(contentsOf: selectedInput)
            let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase
            var calls: [[String]] = []
            let store = fixture.store { args, _ in
                calls.append(args)
                switch args.first {
                case "capabilities": return "{\"training_sizes\":[256,512,1024,2048]}"
                case "refresh-specialists":
                    XCTAssertEqual(self.value("--dataset", in: args), originals.path)
                    let current = try await NativeMaterialSpecialistService.refresh(datasetURL: originals,
                        workspaceURL: workspace, classifier: classifier)
                    return String(decoding: try encoder.encode(current), as: UTF8.self)
                case "dataset":
                    let response = try await NativeMaterialDatasetService.run(arguments: args)
                    return try XCTUnwrap(response)
                default: throw StudioError("Unexpected opening command: \(args)")
                }
            }
            defer { store.observeSourceFolder(nil) }
            store.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self,
                output: String(decoding: encoder.encode(previous), as: UTF8.self))
            store.selectedSpecialistID = emptyWood.id
            store.specialistWorkspacePath = workspace.path
            store.specialistSourceDatasetPath = originals.path
            store.specialistTarget = "height"
            store.training.size = 2048
            let previousNavigation = store.trainingNavigationRequest
            store.openSpecialistTrainer()
            try await wait(store)
            XCTAssertNil(store.error, retagExisting ? "Retagged material" : "New material")
            XCTAssertEqual(calls.map { $0[0] }, ["capabilities", "refresh-specialists", "dataset"])
            XCTAssertEqual(value("--review-size", in: try XCTUnwrap(calls.last)), "2048")
            XCTAssertEqual(store.training.size, 2048)
            XCTAssertEqual(store.dataset?.datasetPath, emptyWood.datasetPath)
            XCTAssertEqual(store.dataset?.sourceFolderPath, originals.resolvingSymlinksInPath().path)
            XCTAssertEqual(store.specialistTrainingContext?.specialistID, emptyWood.id)
            XCTAssertEqual(store.selectedSpecialist?.targetInfo("height")?.materialCount, 1)
            XCTAssertNotEqual(store.specialistTrainingContext?.membershipSHA256,
                emptyWood.targets.first { $0.target == "height" }?.membershipSha256)
            XCTAssertEqual(Set(store.dataset?.originalSamples.compactMap { $0.maps["input"]?.originalSourcePath } ?? []),
                [selectedInput.resolvingSymlinksInPath().path])
            XCTAssertEqual(try Data(contentsOf: selectedInput), originalBytes)
            XCTAssertNil(store.trainingConfigurationIssue)
            XCTAssertNotEqual(store.trainingNavigationRequest, previousNavigation)
        }
    }

    private func writeTaggedMaterial(_ name: String, label: String, in originals: URL) throws -> [URL] {
        let folder = originals.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let size = 2048
        var files: [URL] = []
        for (filename, role, channels, bits) in [("BaseColor.png", "input", 3, 8), ("Height.png", "height", 1, 16)] {
            let file = folder.appendingPathComponent(filename)
            try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels,
                color: channels == 1 ? 0 : 2, interlace: 0), pixels: Data(repeating: 91,
                    count: size * size * channels * bits / 8), colorChunks: []).encoded().write(to: file)
            try NativeSourceFileTags.update(file, role: role, classLabel: label, classifierIdentity: "user")
            files.append(file)
        }
        return files
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
    private func modelWorkspaceDocument(_ workspace: URL, source: URL) throws -> String {
        var document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(Self.registry.utf8)) as? [String: Any])
        document["schema"] = NativeMaterialSpecialistService.schema
        document["workspace_path"] = workspace.path
        document["source_dataset_path"] = source.path
        document["classifier_only_assets"] = []
        var specialists = try XCTUnwrap(document["specialists"] as? [[String: Any]])
        for index in specialists.indices {
            specialists[index]["dataset_path"] = workspace.appendingPathComponent("datasets/" + (specialists[index]["id"] as! String)).path
            specialists[index]["models"] = []
            specialists[index]["assignments"] = []
            specialists[index]["automatically_managed_material_ids"] = []
            specialists[index]["automatically_removed_material_ids"] = []
        }
        document["specialists"] = specialists
        return String(decoding: try JSONSerialization.data(withJSONObject: document), as: UTF8.self)
    }
    private func writeModelWorkspace(_ workspace: URL, source: URL) throws {
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let document = try modelWorkspaceDocument(workspace, source: source)
        try Data(document.utf8).write(to: workspace.appendingPathComponent("specialists.json"))
        try JSONSerialization.data(withJSONObject: ["schema": NativeMaterialSpecialistService.schema,
            "workspace_path": workspace.path, "source_dataset_path": source.path]).write(to: workspace.appendingPathComponent(".specialist-workspace.json"))
        let registry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self, output: document)
        for category in registry.specialists {
            let dataset = URL(fileURLWithPath: category.datasetPath)
            try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: dataset.appendingPathComponent("dataset.json"))
            try JSONSerialization.data(withJSONObject: ["schema": NativeMaterialSpecialistService.schema,
                "workspace_path": workspace.path, "specialist_id": category.id]).write(to: dataset.appendingPathComponent(".specialist-binding.json"))
        }
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
