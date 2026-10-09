import Foundation
import CryptoKit
import XCTest
@testable import TextureStudio

@MainActor
final class DatasetManagementTests: XCTestCase {
    func testOpenFolderAndRestoreDoNotRequireModelCapabilities() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.defaults.set(fixture.dataset.path, forKey: "dataset")
        let store = fixture.store()
        store.restore()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.datasetName, "Test Dataset")
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["capabilities", "dataset"])
        fixture.calls = []
        store.openDataset(fixture.dataset)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["dataset"], "Folder opening must not import the model runtime")
        XCTAssertEqual(store.recentDatasets.count, 1)
    }

    func testCreateRenameAndReopenKeepOneNamedLibraryEntry() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.showNewDatasetSheet = true
        store.createDataset(name: "Test Dataset", description: "Native original maps", parentURL: fixture.root)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertFalse(store.showNewDatasetSheet)
        XCTAssertEqual(store.datasetName, "Test Dataset")
        XCTAssertEqual(fixture.calls.last?.first, "create-dataset")
        XCTAssertEqual(fixture.value("--dataset", in: fixture.calls.last!), fixture.root.appendingPathComponent("Test Dataset").path)
        store.showDatasetInfoSheet = true
        store.updateDatasetInfo(name: "Renamed Dataset", description: "Updated info")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.datasetName, "Renamed Dataset")
        XCTAssertEqual(store.datasetDescription, "Updated info")
        XCTAssertFalse(store.showDatasetInfoSheet)
        XCTAssertEqual(store.recentDatasets.count, 1)
        XCTAssertEqual(store.recentDatasets.first?.name, "Renamed Dataset")
        XCTAssertEqual(fixture.value("--expected-index-sha256", in: fixture.calls.last!), "source-hash")
        let restored = fixture.store()
        XCTAssertEqual(restored.recentDatasets, store.recentDatasets)
        XCTAssertEqual(try Data(contentsOf: fixture.original), Data("original map bytes".utf8))
    }

    func testMetadataConflictKeepsDatasetAndEditorOpen() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        try await store.loadDataset(fixture.dataset)
        fixture.failEdit = true
        store.showDatasetInfoSheet = true
        store.updateDatasetInfo(name: "Conflicting", description: "")
        try await settled(store)
        XCTAssertNotNil(store.error)
        XCTAssertTrue(store.showDatasetInfoSheet)
        XCTAssertEqual(store.datasetName, "Test Dataset")
        XCTAssertEqual(store.recentDatasets.first?.name, "Test Dataset")
    }

    func testLifecycleChangesReconnectDurableSourceBeforeEditingPreparedView() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.dataset = try WorkbenchProcess.decode(WorkbenchDataset.self, output: fixture.document(prepared: true))
        store.updateDatasetInfo(name: "Source renamed", description: "")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["cleanup-size", "dataset", "edit-dataset"])
        XCTAssertEqual(fixture.value("--dataset", in: fixture.calls.last!), fixture.dataset.path)
        XCTAssertEqual(store.datasetFolderURL?.path, fixture.dataset.path)
        XCTAssertEqual(store.datasetName, "Source renamed")
    }

    func testTrashUsesVerifiedMetadataScopeAndClearsLibraryOnlyAfterSuccess() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        var removed: [URL] = []
        let store = fixture.store(trash: { removed.append($0) })
        try await store.loadDataset(fixture.dataset)
        store.trashDataset()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(removed, [fixture.dataset.appendingPathComponent("dataset.json")])
        XCTAssertNil(store.dataset)
        XCTAssertNil(store.selectedSampleId)
        XCTAssertNil(fixture.defaults.string(forKey: "dataset"))
        XCTAssertTrue(store.recentDatasets.isEmpty)
        XCTAssertEqual(try Data(contentsOf: fixture.original), Data("original map bytes".utf8))

        let failing = fixture.store(trash: { _ in throw StudioError("Trash unavailable") })
        try await failing.loadDataset(fixture.dataset)
        failing.trashDataset()
        try await settled(failing)
        XCTAssertNotNil(failing.error)
        XCTAssertNotNil(failing.dataset)
        XCTAssertEqual(failing.recentDatasets.count, 1)
        XCTAssertNotNil(fixture.defaults.string(forKey: "dataset"))
    }

    func testDeletionRejectsStaleHashAndPathsOutsideOwnedScope() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        var called = false
        let stale = WorkbenchDatasetDeletion(datasetPath: fixture.dataset.path, indexSha256: "stale",
            safeToTrashFolder: false, trashPaths: [fixture.dataset.appendingPathComponent("dataset.json").path])
        XCTAssertThrowsError(try DatasetFileOperations.trash(stale, expectedDataset: fixture.dataset) { _ in called = true })
        let wrongPath = WorkbenchDatasetDeletion(datasetPath: fixture.dataset.path, indexSha256: fixture.hash,
            safeToTrashFolder: false, trashPaths: [fixture.original.path])
        XCTAssertThrowsError(try DatasetFileOperations.trash(wrongPath, expectedDataset: fixture.dataset) { _ in called = true })
        XCTAssertFalse(called)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.original.path))
    }

    func testOwnedFolderTrashRejectsNewOriginalAddedAfterPlan() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let plan = WorkbenchDatasetDeletion(datasetPath: fixture.dataset.path, indexSha256: fixture.hash,
            safeToTrashFolder: true, trashPaths: [fixture.dataset.path])
        var called = false
        try DatasetFileOperations.trash(plan, expectedDataset: fixture.dataset) { _ in called = true }
        XCTAssertTrue(called)
        called = false
        try Data("new original bytes".utf8).write(to: fixture.dataset.appendingPathComponent("original.png"))
        XCTAssertThrowsError(try DatasetFileOperations.trash(plan, expectedDataset: fixture.dataset) { _ in called = true })
        XCTAssertFalse(called)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.dataset.appendingPathComponent("original.png").path))
    }

    func testNativeLifecycleAgainstBundledPythonContractKeepsOriginalBytes() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let python = repository.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw XCTSkip("Local backend runtime is unavailable") }
        let fixture = try Fixture()
        defer { fixture.remove() }
        let source = fixture.root.appendingPathComponent("Sources")
        let generator = WorkbenchProcess()
        _ = try await generator.run(executable: python, arguments: ["-B", "-c", """
            import sys
            from pathlib import Path
            sys.path.insert(0, sys.argv[1])
            import numpy as np
            from material_dataset import write_png
            root = Path(sys.argv[2]); root.mkdir()
            write_png(root/'diffuse.png', np.full((256,256,3), 127, np.uint8))
            write_png(root/'height.png', np.arange(65536, dtype=np.uint16).reshape(256,256,1))
            """, repository.appendingPathComponent("scripts").path, source.path], directory: repository,
            log: fixture.root.appendingPathComponent("generate.log"), onLog: { _ in })
        let input = source.appendingPathComponent("diffuse.png")
        let height = source.appendingPathComponent("height.png")
        let originalBytes = try Data(contentsOf: height)
        let backend = repository.appendingPathComponent("scripts/material_workbench.py")
        var trashed: [URL] = []
        let store = WorkbenchStore(preferences: fixture.defaults, trashHandler: { url in
            trashed.append(url)
            try FileManager.default.moveItem(at: url, to: fixture.root.appendingPathComponent("Recovered Dataset"))
        }, workerOverride: { args, script in
            XCTAssertEqual(script, "material_workbench.py")
            return try await WorkbenchProcess().run(executable: python, arguments: ["-B", backend.path] + args,
                directory: repository, log: fixture.root.appendingPathComponent(UUID().uuidString + ".log"), onLog: { _ in })
        })
        store.training.size = 1024
        store.openDataset(source)
        try await settled(store)
        XCTAssertEqual(store.error, "This folder does not contain a material dataset. Create a dataset or import material maps into one.")
        store.createDataset(name: "Native Lifecycle", description: "Exact originals", parentURL: fixture.root)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.samples.count, 0)
        store.addMaterial(name: "壁材", input: input, height: height, roughness: nil, normal: nil)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.samples.count, 1, "A source smaller than the chosen training grid remains manageable")
        XCTAssertEqual(store.selectedSample?.width, 256)
        XCTAssertEqual(store.selectedMaterialName, "壁材")
        XCTAssertNil(store.datasetDisplayTransform(try XCTUnwrap(store.selectedMap), role: "height"))
        store.updateDatasetInfo(name: "Renamed Native Dataset", description: "Edited description")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.datasetDescription, "Edited description")
        store.curateSelected(status: "approved", split: "validation", note: "Retain native precision")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.selectedSample?.split, "validation")
        XCTAssertEqual(store.selectedSample?.note, "Retain native precision")
        store.curateSelected(status: "approved", note: "")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.selectedSample?.note, "")
        store.removeSelectedMaterial()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.samples.count, 0)
        store.importMaterialFolder(source)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.samples.count, 1)
        let folder = try XCTUnwrap(store.datasetFolderURL)
        let expectedTrashedPath = folder.resolvingSymlinksInPath().standardizedFileURL.path
        store.closeDataset()
        store.openDataset(folder)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.datasetName, "Renamed Native Dataset")
        store.trashDataset()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(trashed.map(\.path), [expectedTrashedPath])
        XCTAssertNil(store.dataset)
        XCTAssertEqual(try Data(contentsOf: height), originalBytes)
        XCTAssertTrue(FileManager.default.fileExists(atPath: input.path))
    }

    func testTrainerExplainsWhenAllMaterialsAreExplicitlyHeldOut() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, workerOverride: { args, _ in
            if args.first == "capabilities" { return "{\"training_sizes\":[256]}" }
            return """
                {"dataset_path":"\(fixture.dataset.path)","index_sha256":"source-hash","supported_training_sizes":[256],
                "materials":[{"material_id":"wall","samples":[{"sample_id":"wall_full","status":"approved",
                "split":"validation","split_assignment":"manual","width":256,"height":256,
                "maps":{"height":{"path":"\(fixture.original.path)","width":256,"height":256}}}]}]}
                """.replacingOccurrences(of: "\n", with: "")
        })
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.dataset)
        XCTAssertTrue(store.trainingConfigurationIssue?.contains("All available materials are assigned to Validation") == true)
        store.startTraining()
        XCTAssertFalse(store.isBusy)
        XCTAssertNotNil(store.error)
    }

    private func settled(_ store: WorkbenchStore) async throws {
        for _ in 0..<1000 {
            if !store.isBusy { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("Dataset operation did not finish")
    }
    @MainActor private final class Fixture {
        let root: URL
        let dataset: URL
        let original: URL
        let defaults: UserDefaults
        let suite = "dataset-management-\(UUID().uuidString)"
        var calls: [[String]] = []
        var name = "Test Dataset"
        var description = "Native original maps"
        var failEdit = false
        var hash: String {
            SHA256.hash(data: try! Data(contentsOf: dataset.appendingPathComponent("dataset.json")))
                .map { String(format: "%02x", $0) }.joined()
        }
        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("dataset-management-\(UUID().uuidString)")
            dataset = root.appendingPathComponent("Test Dataset")
            original = root.appendingPathComponent("original.png")
            defaults = UserDefaults(suiteName: suite)!
            try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
            try Data("index bytes".utf8).write(to: dataset.appendingPathComponent("dataset.json"))
            try Data("original map bytes".utf8).write(to: original)
            defaults.set(root.path, forKey: "workspace")
        }
        func remove() { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        func value(_ flag: String, in args: [String]) -> String? {
            guard let i = args.firstIndex(of: flag), args.indices.contains(i + 1) else { return nil }
            return args[i + 1]
        }
        func document(prepared: Bool = false) throws -> String {
            var object: [String: Any] = ["dataset_path": prepared ? root.appendingPathComponent("prepared").path : dataset.path,
                "index_sha256": "source-hash", "name": name, "description": description, "materials": []]
            if prepared { object["preparation"] = ["source_dataset_path": dataset.path, "source_index_sha256": "source-hash",
                "prepared_dataset_path": root.appendingPathComponent("prepared").path, "crop_size": 1024,
                "reused": false, "target_resized": false, "original_dataset_modified": false] }
            return String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        }
        func store(trash: @escaping (URL) throws -> Void = { _ in }) -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, trashHandler: trash, workerOverride: { args, script in
                self.calls.append(args)
                if args.first == "capabilities" { throw StudioError("Model runtime is not installed") }
                XCTAssertEqual(script, "material_workbench.py")
                if args.first == "edit-dataset" {
                    if self.failEdit { throw StudioError("Dataset changed since selection") }
                    self.name = self.value("--name", in: args) ?? self.name
                    self.description = self.value("--description", in: args) ?? self.description
                }
                if args.first == "validate-delete" {
                    return String(decoding: try JSONSerialization.data(withJSONObject: ["dataset_path": self.dataset.path,
                        "index_sha256": self.hash, "safe_to_trash_folder": false,
                        "trash_paths": [self.dataset.appendingPathComponent("dataset.json").path]]), as: UTF8.self)
                }
                if args.first == "cleanup-size" { return "{\"removed\":true}" }
                return try self.document()
            })
        }
    }
}
