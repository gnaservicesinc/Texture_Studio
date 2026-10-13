import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchSourceFolderRefreshTests: XCTestCase {
    func testRestoreCategoryKeepsSavedGridWhenManifestRetainsLegacyGrid() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let category = fixture.root.appendingPathComponent("Models/datasets/wood")
        let originals = fixture.root.appendingPathComponent("Originals")
        for folder in [category, originals] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
        let seeded = fixture.store { _, _ in "" }
        seeded.training.modelFamily = .compactScalar
        seeded.training.size = 2048
        seeded.specialistSourceDatasetPath = originals.path
        seeded.specialistTrainingContext = .init(workspacePath: fixture.root.appendingPathComponent("Models").path,
            specialistID: "wood", name: "Wood", target: "height", datasetPath: category.path, membershipSHA256: "members")
        fixture.defaults.set(category.path, forKey: "dataset")
        var calls: [[String]] = []
        let reopened = fixture.store { args, _ in
            calls.append(args)
            if args.first == "capabilities" { return "{\"training_sizes\":[256,512,1024,2048]}" }
            return try self.json(["dataset_path": category.path, "source_folder_path": originals.path,
                "is_source_folder": false, "index_sha256": "legacy-category", "training_size": 1024,
                "review_size": 2048, "supported_training_sizes": [1024, 2048], "materials": []])
        }
        defer { reopened.observeSourceFolder(nil) }
        reopened.restore(arguments: ["TextureStudio"], sourceBrowsing: false)
        try await waitForIdle(reopened)
        XCTAssertNil(reopened.error)
        let read = try XCTUnwrap(calls.first { $0.first == "dataset" })
        XCTAssertEqual(value("--dataset", in: read), category.path)
        XCTAssertEqual(value("--review-size", in: read), "2048")
        XCTAssertEqual(reopened.training.size, 2048)
        XCTAssertEqual(reopened.dataset?.trainingSize, 2048)
        XCTAssertEqual(reopened.dataset?.reviewSize, 2048)
        XCTAssertEqual(reopened.specialistTrainingContext?.datasetPath, category.path)
        XCTAssertEqual(reopened.sourceFolderObserver?.root.path, originals.resolvingSymlinksInPath().path)
    }

    func testPreparedCategoryRefreshReadsCurrentTagsAndKeepsRequestedGrid() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let originals = fixture.root.appendingPathComponent("Originals")
        let workspace = fixture.root.appendingPathComponent("Models")
        let category = workspace.appendingPathComponent("datasets/wood")
        let prepared = category.appendingPathComponent(".training-data/prepared")
        for folder in [originals, prepared] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
        let previous = try writeMaterial("Previous", label: "Wood", in: originals)
        let current = try writeMaterial("Current", label: "Bark", in: originals)
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "name": "Wood", "training_size": 1024,
            "samples": [], "validation": ["enabled": false]])
            .write(to: category.appendingPathComponent("dataset.json"))
        try JSONSerialization.data(withJSONObject: ["schema": NativeMaterialSpecialistService.schema,
            "specialist_id": "wood", "workspace_path": workspace.path, "source_folder_path": originals.path,
            "class_label": "Wood", "subtype": ""])
            .write(to: category.appendingPathComponent(".specialist-binding.json"))
        var calls: [[String]] = []
        let store = fixture.store { args, _ in
            calls.append(args)
            let response = try await NativeMaterialDatasetService.run(arguments: args)
            return try XCTUnwrap(response)
        }
        defer { store.observeSourceFolder(nil) }
        store.specialistSourceDatasetPath = originals.path
        store.specialistTrainingContext = .init(workspacePath: workspace.path, specialistID: "wood", name: "Wood",
            target: "height", datasetPath: category.path, membershipSHA256: "old-members")
        let initialResponse = try await NativeMaterialDatasetService.run(
            arguments: ["dataset", "--dataset", category.path, "--review-size", "2048"])
        var document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(initialResponse).utf8)) as? [String: Any])
        document["dataset_path"] = prepared.path
        document["training_size"] = 2048
        document["preparation"] = ["source_dataset_path": category.path, "source_index_sha256": "category-sha",
            "prepared_dataset_path": prepared.path, "crop_size": 2048, "reused": false,
            "target_resized": false, "original_dataset_modified": false]
        store.adoptDataset(try WorkbenchResult.decode(WorkbenchDataset.self, output: json(document)))
        XCTAssertEqual(store.training.size, 2048)
        XCTAssertEqual(Set(store.dataset?.originalSamples.compactMap { $0.maps["input"]?.originalSourcePath } ?? []), [previous.path])
        try NativeSourceFileTags.update(previous, classLabel: "Bark", classifierIdentity: "user")
        try NativeSourceFileTags.update(current, classLabel: "Wood", classifierIdentity: "user")
        // Exercise the event consumer directly; OS event delivery is a separate
        // observer concern, and this verifies the route from prepared to live data.
        store.sourceFolderNeedsRefresh = true
        store.refreshSourceFolderIfNeeded()
        for _ in 0..<500 where store.isRefreshingSourceFolder { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isRefreshingSourceFolder)
        XCTAssertEqual(value("--dataset", in: try XCTUnwrap(calls.last)), category.path,
                       "A source change refreshes the active category rather than broadening training to every original")
        XCTAssertEqual(value("--review-size", in: try XCTUnwrap(calls.last)), "2048")
        XCTAssertEqual(store.dataset?.datasetPath, category.path)
        XCTAssertNil(store.dataset?.preparation)
        XCTAssertEqual(store.dataset?.sourceFolderPath, originals.path)
        XCTAssertEqual(store.training.size, 2048)
        XCTAssertEqual(store.dataset?.trainingSize, 2048)
        XCTAssertEqual(store.dataset?.reviewSize, 2048)
        XCTAssertEqual(Set(store.dataset?.originalSamples.compactMap { $0.maps["input"]?.originalSourcePath } ?? []), [current.path])
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "wood")
    }

    func testOlderObserverReadCannotResetANewTrainingSizeAndRequeuesCurrentGrid() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let originals = fixture.root.appendingPathComponent("Originals")
        let category = fixture.root.appendingPathComponent("Models/datasets/wood")
        for folder in [originals, category] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
        func document(size: Int) throws -> String {
            try json(["dataset_path": category.path, "source_folder_path": originals.path,
                "index_sha256": "current-tags", "training_size": size, "review_size": size,
                "supported_training_sizes": [1024, 2048], "materials": []])
        }
        var reads: [[String]] = []
        var heldRead: CheckedContinuation<String, Never>?
        let store = fixture.store { args, _ in
            reads.append(args)
            if reads.count == 1 { return await withCheckedContinuation { heldRead = $0 } }
            XCTAssertEqual(self.value("--review-size", in: args), "2048")
            return try document(size: 2048)
        }
        defer { store.observeSourceFolder(nil) }
        store.specialistSourceDatasetPath = originals.path
        store.specialistTrainingContext = .init(workspacePath: fixture.root.appendingPathComponent("Models").path,
            specialistID: "wood", name: "Wood", target: "height", datasetPath: category.path, membershipSHA256: "members")
        store.adoptDataset(try WorkbenchResult.decode(WorkbenchDataset.self, output: document(size: 1024)))
        store.sourceFolderNeedsRefresh = true
        store.refreshSourceFolderIfNeeded()
        for _ in 0..<500 where heldRead == nil { try await Task.sleep(for: .milliseconds(10)) }
        let continuation = try XCTUnwrap(heldRead)
        XCTAssertEqual(value("--review-size", in: try XCTUnwrap(reads.first)), "1024")
        // A size selection can finish while the background observer's metadata
        // read is waiting. Its older DTO must not undo the newer user choice.
        store.training.size = 2048
        continuation.resume(returning: try document(size: 1024))
        heldRead = nil
        for _ in 0..<500 where store.isRefreshingSourceFolder { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isRefreshingSourceFolder)
        XCTAssertEqual(reads.count, 2)
        XCTAssertTrue(reads.allSatisfy { value("--dataset", in: $0) == category.path })
        XCTAssertEqual(store.training.size, 2048)
        XCTAssertEqual(store.dataset?.trainingSize, 2048)
        XCTAssertEqual(store.dataset?.reviewSize, 2048)
        XCTAssertEqual(store.specialistTrainingContext?.specialistID, "wood")
        XCTAssertFalse(store.sourceFolderNeedsRefresh)
    }

    private func writeMaterial(_ name: String, label: String, in root: URL) throws -> URL {
        let folder = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let size = 2048, input = folder.appendingPathComponent("BaseColor.png"), height = folder.appendingPathComponent("Height.png")
        for (file, role, channels, bits) in [(input, "input", 3, 8), (height, "height", 1, 16)] {
            try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels,
                color: channels == 1 ? 0 : 2, interlace: 0), pixels: Data(repeating: 93,
                    count: size * size * channels * bits / 8), colorChunks: []).encoded().write(to: file)
            try NativeSourceFileTags.update(file, role: role, classLabel: label, classifierIdentity: "user")
        }
        return input
    }
    private func json(_ document: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: document), as: UTF8.self)
    }
    private func value(_ flag: String, in args: [String]) -> String? {
        args.firstIndex(of: flag).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil }
    }
    private func waitForIdle(_ store: WorkbenchStore) async throws {
        for _ in 0..<500 where store.isBusy { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isBusy)
    }
    @MainActor private struct Fixture {
        let root: URL
        let defaults: UserDefaults
        let suite: String
        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("source-refresh-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            suite = "source-refresh-\(UUID().uuidString)"
            defaults = UserDefaults(suiteName: suite)!
        }
        func store(worker: @escaping @MainActor ([String], String) async throws -> String) -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, managedWorkspaceURL: root,
                selectedCheckpointRegistryURL: root.appendingPathComponent("selected.json"), workerOverride: worker)
        }
        func remove() { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
    }
}
