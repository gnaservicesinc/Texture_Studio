import AppKit
import SwiftUI
import Foundation
import CryptoKit
import XCTest
@testable import TextureStudio

@MainActor
final class DatasetManagementTests: XCTestCase {
    func testNewDatasetDefaultsTo2048IndependentlyOfPreviousTrainingSize() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = fixture.store()
        store.training.size = 512
        XCTAssertEqual(store.datasetResolution, 2048)
        store.createDataset(name: "New originals", description: "", parentURL: fixture.root)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.value("--training-size", in: try XCTUnwrap(fixture.calls.first)), "2048")
        XCTAssertEqual(store.datasetResolution, 2048)
        XCTAssertFalse(store.showAddMaterialSheet, "Creating an empty dataset must not force manual map pairing")
    }

    func testManagedBrowserKeepsUndersizedOriginalsAndUnassignedFilesSeparateFromCrops() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = fixture.store()
        let dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: managedBrowserDocument(fixture))
        store.adoptDataset(dataset)
        XCTAssertEqual(store.samples.count, 1)
        XCTAssertEqual(store.samples.first?.width, 2048)
        XCTAssertEqual(store.datasetBrowserSamples.count, 2)
        XCTAssertEqual(store.datasetBrowserMode, .originals)
        let smaller = try XCTUnwrap(store.datasetBrowserSamples.first { $0.width == 1024 })
        store.selectDatasetSample(smaller)
        XCTAssertEqual(store.selectedSample?.split, "original")
        XCTAssertEqual(store.selectedSample?.width, 1024)
        XCTAssertNil(store.datasetDisplayTransform(try XCTUnwrap(store.selectedMap), role: "input"))
        XCTAssertEqual(store.datasetSourceFolderCount, 2)
        XCTAssertEqual(store.dataset?.sourceInventory?.filter { $0.state == "needs_assignment" }.count, 1)
        XCTAssertTrue(store.datasetInventorySummary.contains("4 original files"))
        store.datasetBrowserMode = .crops
        XCTAssertEqual(store.datasetBrowserSamples.count, 1)
        XCTAssertEqual(store.selectedSample?.split, "train")
        XCTAssertTrue(store.dataset?.hasNativeSize(2048) == true)
        store.datasetBrowserMode = .originals
        XCTAssertEqual(store.datasetBrowserSamples.count, 2, "Changing the grid view never removes originals")
    }

    func testMissingOriginalInspectionRequestsLiveFolderRefreshWithoutDeletingMetadata() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = fixture.store()
        store.adoptDataset(try WorkbenchResult.decode(WorkbenchDataset.self, output: managedBrowserDocument(fixture)))
        let original = try XCTUnwrap(store.datasetBrowserSamples.first)
        store.selectDatasetSample(original)
        let index = store.dataset?.indexSha256
        let notice = store.reportMissingDatasetSource(URL(fileURLWithPath: "/unavailable/original.png"))
        XCTAssertTrue(notice.contains("moved or deleted"))
        XCTAssertTrue(store.sourceFolderNeedsRefresh)
        XCTAssertEqual(store.dataset?.indexSha256, index)
        XCTAssertEqual(store.datasetBrowserSamples.count, 2)
        XCTAssertEqual(store.selectedSample?.id, original.id)
        XCTAssertTrue(fixture.calls.isEmpty, "Inspection cannot unregister an unavailable original")
    }

    func testAssignedOriginalsWithMissingAmbiguousOrInvalidHeadersStayInInventoryReview() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(managedBrowserDocument(fixture).utf8)) as? [String: Any])
        let assigned = fixture.root.appendingPathComponent("Wall/wall_1k_basecolor.png").path
        let cases = ["paired", "header_error", "missing", "ambiguous"]
        for state in cases {
            object["source_inventory"] = [["path": assigned, "state": state, "issue": state == "paired" ? "" : "Needs review"]]
            let dataset = try WorkbenchResult.decode(WorkbenchDataset.self,
                output: String(decoding: JSONSerialization.data(withJSONObject: object), as: UTF8.self))
            XCTAssertEqual(dataset.browserInventory().count, state == "paired" ? 0 : 1,
                           "An assigned map with a source error must remain independently visible")
            XCTAssertEqual(dataset.browserInventory(query: "basecolor").count, state == "paired" ? 0 : 1)
            XCTAssertTrue(dataset.browserInventory(query: "unrelated").isEmpty)
        }
    }

    func testPreparedDatasetBrowserKeepsCanonicalOriginalsSeparateFromTrainingMaps() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(managedBrowserDocument(fixture).utf8)) as? [String: Any])
        object["dataset_path"] = fixture.root.appendingPathComponent("Prepared").path
        object["preparation"] = ["source_dataset_path": fixture.dataset.path, "source_index_sha256": "managed-hash",
            "prepared_dataset_path": fixture.root.appendingPathComponent("Prepared").path, "crop_size": 2048,
            "reused": true, "target_resized": false, "original_dataset_modified": false]
        let store = fixture.store()
        store.adoptDataset(try WorkbenchResult.decode(WorkbenchDataset.self,
            output: String(decoding: JSONSerialization.data(withJSONObject: object), as: UTF8.self)))
        XCTAssertEqual(store.datasetBrowserMode, .originals)
        XCTAssertEqual(store.datasetBrowserSamples.map(\.width), [1024, 2048])
        XCTAssertEqual(store.samples.map(\.width), [2048])
        XCTAssertEqual(store.datasetFolderURL?.resolvingSymlinksInPath().standardizedFileURL.path,
                       fixture.dataset.resolvingSymlinksInPath().standardizedFileURL.path)
        XCTAssertEqual(store.selectedSample?.split, "original")
    }

    func testOriginalAndCropCurationUseSeparateReviewRoutes() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, workerOverride: { args, _ in
            fixture.calls.append(args)
            return try self.managedBrowserDocument(fixture)
        })
        try await store.loadDataset(fixture.dataset)
        store.curateSelected(status: "approved", note: "Original review")
        try await settled(store)
        let originalCall = try XCTUnwrap(fixture.calls.first { $0.first == "curate" })
        XCTAssertEqual(fixture.value("--original-source", in: originalCall), "true")
        XCTAssertEqual(fixture.value("--note", in: originalCall), "Original review")
        fixture.calls = []
        store.datasetBrowserMode = .crops
        store.curateSelected(status: "approved", note: "Crop review")
        try await settled(store)
        let cropCall = try XCTUnwrap(fixture.calls.first { $0.first == "curate" })
        XCTAssertFalse(cropCall.contains("--original-source"))
        XCTAssertEqual(fixture.value("--note", in: cropCall), "Crop review")
    }

    func testLooseSourceFilesKeepTheirOriginalParentAndProviderTierGroups() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let first = fixture.root.appendingPathComponent("Download A/Material")
        let second = fixture.root.appendingPathComponent("Download B/Material")
        let standalone = fixture.root.appendingPathComponent("Download A/Standalone")
        for parent in [first, second, standalone] {
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        }
        let urls = [first.appendingPathComponent("wood_diff_2k.png"), first.appendingPathComponent("wood_disp_2k.png"),
                    first.appendingPathComponent("wood_ao_2k.png"), first.appendingPathComponent("wood_diff_4k.png"),
                    first.appendingPathComponent("wood_disp_4k.png"), second.appendingPathComponent("wood_diff_2k.png"),
                    second.appendingPathComponent("wood_disp_2k.png"), standalone.appendingPathComponent("BaseColor.png"),
                    standalone.appendingPathComponent("Height.png")]
        for url in urls { try Data("Unchanged original".utf8).write(to: url) }
        let store = fixture.store()
        store.adoptDataset(try WorkbenchResult.decode(WorkbenchDataset.self, output: fixture.document()))
        try await store.registerSourceFiles(urls)
        let added = fixture.calls.filter { $0.first == "add-source" }
        XCTAssertEqual(added.count, urls.count)
        func folder(_ url: URL) throws -> String {
            let call = try XCTUnwrap(added.first { fixture.value("--source", in: $0) == url.path })
            XCTAssertEqual(fixture.value("--defer-refresh", in: call), "true")
            return try XCTUnwrap(fixture.value("--folder-name", in: call))
        }
        XCTAssertEqual(try folder(urls[0]), try folder(urls[1]))
        XCTAssertEqual(try folder(urls[0]), try folder(urls[2]), "AO stays with its source family without becoming diffuse")
        XCTAssertEqual(try folder(urls[3]), try folder(urls[4]))
        XCTAssertNotEqual(try folder(urls[0]), try folder(urls[3]), "Provider resolution tiers stay separate")
        XCTAssertEqual(try folder(urls[5]), try folder(urls[6]))
        XCTAssertNotEqual(try folder(urls[0]), try folder(urls[5]), "Identically named materials from distinct real parents stay separate")
        XCTAssertEqual(try folder(urls[7]), try folder(urls[8]), "Standalone map roles share their original material folder")
        XCTAssertEqual(fixture.calls.filter { $0.first == "dataset" }.count, 1, "A batch refreshes only once")
        XCTAssertFalse(fixture.calls.contains { ["scan-folder", "prepare-size"].contains($0.first ?? "") })
    }

    func testDirectAddFolderRegistersFilesWithoutPreviewOrTrainingWork() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let folder = fixture.root.appendingPathComponent("Unassigned photos")
        let store = WorkbenchStore(preferences: fixture.defaults, workerOverride: { args, _ in
            fixture.calls.append(args)
            return try self.managedBrowserDocument(fixture)
        })
        try await store.loadDataset(fixture.dataset)
        fixture.calls = []
        store.addMaterialFolders([folder])
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["import-folder"])
        let call = try XCTUnwrap(fixture.calls.first)
        XCTAssertEqual(fixture.value("--managed-import", in: call), "true")
        XCTAssertTrue(call.contains("--skip-source-recovery"))
        XCTAssertFalse(call.contains("--plan"))
        XCTAssertFalse(store.showImportFolderSheet)
        XCTAssertEqual(store.datasetBrowserSamples.count, 2)
        XCTAssertEqual(store.dataset?.sourceInventory?.count, 4)
        XCTAssertEqual(store.datasetResolution, 2048)
    }

    func testOpeningDatasetReadsLocalInventoryWithoutImplicitTierRecovery() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = fixture.store()
        try await store.loadDataset(fixture.dataset)
        XCTAssertEqual(fixture.calls.count, 1)
        XCTAssertTrue(try XCTUnwrap(fixture.calls.first).contains("--skip-source-recovery"))
    }

    func testDeferredVerificationReportsActualProgressDuringPreparation() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = fixture.store()
        store.operation("Preparing") {
            store.recordTrainingProgress("{\"event\":\"source_verification_started\",\"completed\":0,\"total\":3}\n")
            XCTAssertEqual(store.activity, "Verifying originals for training: 0 / 3 files")
            store.recordTrainingProgress("{\"event\":\"source_verification_progress\",\"completed\":1,\"total\":3,\"source_path\":\"/sources/wall_basecolor.png\"}\n")
            XCTAssertTrue(store.activity.contains("1 / 3 files · wall_basecolor.png"))
            store.recordTrainingProgress("{\"event\":\"source_verification_completed\",\"completed\":3,\"total\":3}\n")
            XCTAssertEqual(store.activity, "Verifying originals for training: 3 / 3 files")
        }
        try await settled(store)
    }

    func testSourceTierCountIsSavedFromDatasetSettingsAndPreservedByResolutionEdits() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let store = fixture.store()
        store.createDataset(name: "Bounded dataset", description: "", parentURL: fixture.root, size: 2048, sourceTierCount: 2)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.value("--source-tier-count", in: fixture.calls.last!), "2")
        XCTAssertEqual(store.datasetSourceTierCount, 2)
        store.updateDatasetInfo(name: store.datasetName, description: "", sourceTierCount: 1)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.datasetSourceTierCount, 1)
        store.updateDatasetInfo(name: store.datasetName, description: "", size: 1024)
        try await settled(store)
        XCTAssertEqual(fixture.value("--source-tier-count", in: fixture.calls.last!), "1")
        XCTAssertEqual(store.datasetSourceTierCount, 1)
    }

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
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: fixture.document(prepared: true))
        store.updateDatasetInfo(name: "Source renamed", description: "")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["cleanup-size", "dataset", "edit-dataset"])
        XCTAssertTrue(fixture.calls[1].contains("--skip-source-recovery"), "Snapshot reconnection must not add new source tiers before checking its proof")
        XCTAssertEqual(fixture.value("--dataset", in: fixture.calls.last!), fixture.dataset.path)
        XCTAssertEqual(store.datasetFolderURL?.path, fixture.dataset.path)
        XCTAssertEqual(store.datasetName, "Source renamed")
    }

    func testPreparedManagementRefreshesCurrentSourceInsteadOfRejectingObsoleteIndex() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: fixture.document(prepared: true))
        try Data("changed durable dataset".utf8).write(to: fixture.dataset.appendingPathComponent("dataset.json"))
        store.updateDatasetInfo(name: "Source renamed", description: "")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["cleanup-size", "dataset", "edit-dataset"])
        XCTAssertNil(store.dataset?.preparation)
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

    func testNativeLifecycleKeepsOriginalBytes() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let source = fixture.root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        let input = source.appendingPathComponent("diffuse.png"), height = source.appendingPathComponent("height.png")
        let diffuse = NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0),
            pixels: Data(repeating: 127, count: 256 * 256 * 3), colorChunks: [])
        var samples = Data()
        for code in 0..<65536 { samples.append(UInt8(code >> 8)); samples.append(UInt8(code & 255)) }
        let scalar = NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0), pixels: samples, colorChunks: [])
        let inputBytes = try diffuse.encoded(), originalBytes = try scalar.encoded()
        try inputBytes.write(to: input); try originalBytes.write(to: height)
        let store = WorkbenchStore(preferences: fixture.defaults)
        store.training.size = 1024
        store.openDataset(source)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertFalse(store.showNewDatasetSheet)
        XCTAssertTrue(store.dataset?.isSourceFolder == true)
        XCTAssertEqual(store.dataset?.sourceSetCount, 1)
        store.createDataset(name: "Native Lifecycle", description: "", parentURL: fixture.root, size: 1024)
        try await settled(store)
        XCTAssertNil(store.error)
        let folder = try XCTUnwrap(store.datasetFolderURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.appendingPathComponent("dataset.json").path))
        XCTAssertEqual(store.dataset?.sourceSetCount, 0)
        store.updateDatasetInfo(name: store.datasetName, description: "", size: 1024)
        try await settled(store)
        store.addMaterial(name: "壁材", input: input, height: height, roughness: nil, normal: nil)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.samples.count, 0, "A 256-pixel original cannot supply a 1024 training crop")
        XCTAssertEqual(store.dataset?.sourceSetCount, 1)
        XCTAssertEqual(store.datasetResolution, 1024)
        XCTAssertEqual(store.datasetBrowserSamples.count, 1)
        let copiedFolder = folder.appendingPathComponent("壁材")
        let copiedInput = copiedFolder.appendingPathComponent(input.lastPathComponent), copiedHeight = copiedFolder.appendingPathComponent(height.lastPathComponent)
        XCTAssertEqual(try Data(contentsOf: copiedInput), inputBytes)
        XCTAssertEqual(try Data(contentsOf: copiedHeight), originalBytes)
        XCTAssertEqual(NativeSourceFileTags.read(copiedInput)?.role, "input")
        XCTAssertEqual(NativeSourceFileTags.read(copiedHeight)?.role, "height")
        XCTAssertTrue((store.dataset?.sourceInventory ?? []).allSatisfy { $0.path.hasPrefix(folder.path + "/") })
        store.updateDatasetInfo(name: store.datasetName, description: "", size: 256)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.samples.count, 1)
        XCTAssertEqual(store.datasetResolution, 256)
        XCTAssertEqual(store.training.size, 256)
        _ = try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", folder.path,
            "--size", "256", "--target", "height", "--skip-source-recovery"])
        try await store.loadDataset(folder, recoverSources: false, reviewSize: 256)
        store.datasetBrowserMode = .crops
        store.selectedSampleId = store.samples.first?.id
        store.selectedRole = "height"
        XCTAssertEqual(store.selectedSample?.width, 256)
        let display = try XCTUnwrap(store.datasetDisplayTransform(try XCTUnwrap(store.selectedMap), role: "height"))
        XCTAssertEqual(display.size, 256)
        XCTAssertEqual(display.cropRectangle, [0, 0, 256, 256])
        XCTAssertEqual(display.algorithm, MapReviewDisplayTransform.exactCrop)
        XCTAssertEqual(display.sourceSHA256, NativeMaterialTrainer.checksum(originalBytes))
        store.curateSelected(status: "approved", split: "validation", note: "Retain native precision")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.selectedSample?.split, "validation")
        XCTAssertEqual(store.selectedSample?.splitAssignment, "manual")
        XCTAssertEqual(store.selectedSample?.note, "Retain native precision")
        store.curateSelected(status: "approved", note: "")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.selectedSample?.note, "")
        XCTAssertEqual(store.selectedSample?.split, "validation")
        store.closeDataset()
        store.openDataset(folder)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.datasetResolution, 256)
        XCTAssertEqual(store.dataset?.sourceSetCount, 1)
        store.renameSourceFolder(copiedFolder, name: "Renamed Wall")
        try await settled(store)
        XCTAssertNil(store.error)
        let renamed = folder.appendingPathComponent("Renamed Wall")
        XCTAssertFalse(FileManager.default.fileExists(atPath: copiedFolder.path))
        XCTAssertEqual(try Data(contentsOf: renamed.appendingPathComponent(height.lastPathComponent)), originalBytes)
        XCTAssertEqual(store.dataset?.sourceSetCount, 1)
        let moved = fixture.root.appendingPathComponent("Moved Wall")
        store.closeDataset()
        try FileManager.default.moveItem(at: renamed, to: moved)
        store.openDataset(folder)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.dataset?.sourceSetCount, 0, "Moving physical originals out removes their membership")
        XCTAssertEqual(store.dataset?.sourceInventory?.count, 0)
        XCTAssertEqual(try Data(contentsOf: moved.appendingPathComponent(height.lastPathComponent)), originalBytes)
        XCTAssertEqual(try Data(contentsOf: height), originalBytes)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
    }

    func testTrainerRequiresCategoryBeforeUsingCachedValidationInventory() async throws {
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
        store.training.size = 256
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.dataset)
        XCTAssertEqual(store.trainingConfigurationIssue, "Choose a material category in Loom Room.")
        store.startTraining()
        XCTAssertFalse(store.isBusy)
        XCTAssertNotNil(store.error)
    }

    func testResolutionIsSavedWithDatasetAndNeverSilentlyReducedByCapabilities() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.createDataset(name: "High detail", description: "", parentURL: fixture.root, size: 2048)
        try await settled(store)
        XCTAssertEqual(store.datasetResolution, 2048)
        XCTAssertEqual(store.training.size, 2048)
        store.updateDatasetInfo(name: "High detail", description: "", size: 512)
        try await settled(store)
        XCTAssertEqual(store.training.size, 512)
        XCTAssertEqual(fixture.value("--training-size", in: fixture.calls.last!), "512")
        let smaller = WorkbenchStore(preferences: fixture.defaults, workerOverride: { _, _ in "{\"training_sizes\":[256]}" })
        smaller.adoptDataset(try WorkbenchResult.decode(WorkbenchDataset.self, output: fixture.document()))
        try await smaller.loadTrainingCapabilities()
        XCTAssertEqual(smaller.training.size, 512, "Capabilities must report an incompatible budget without changing the dataset grid")
        XCTAssertNotNil(smaller.trainingConfigurationIssue)
        let reopened = fixture.store()
        try await reopened.loadDataset(fixture.dataset)
        XCTAssertEqual(reopened.training.size, 512)
    }

    func testRawFolderOpensDirectlyWithoutSetupOrImport() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let source = fixture.root.appendingPathComponent("Nested material sources")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let store = fixture.store()
        store.openDataset(source)
        try await settled(store)
        XCTAssertFalse(store.showNewDatasetSheet)
        XCTAssertNil(store.pendingSourceFolder)
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["dataset"])
        XCTAssertEqual(fixture.value("--dataset", in: try XCTUnwrap(fixture.calls.first)), source.path)
        XCTAssertNil(store.error)
    }

    func testOpeningFolderDuringRestoreIsQueuedAndSheetsHaveOneRoute() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let source = fixture.root.appendingPathComponent("Originals")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let store = fixture.store()
        store.operation("Restoring") { try await Task.sleep(for: .milliseconds(50)) }
        store.receiveDataset(source)
        try await settled(store)
        XCTAssertFalse(store.showNewDatasetSheet)
        XCTAssertNil(store.pendingSourceFolder)
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["dataset"])
        XCTAssertNil(store.error)
        store.showAddMaterialSheet = true
        XCTAssertFalse(store.showNewDatasetSheet)
        store.showImportFolderSheet = true
        XCTAssertFalse(store.showAddMaterialSheet)
        XCTAssertEqual(store.datasetSheet, .folder)
    }

    func testDatasetSetupAndImportSheetsRenderAtUsableWindowSizes() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        try await store.loadDataset(fixture.dataset)
        let plan: [String: Any] = ["size": 2048, "crop_count": 329, "source_set_count": 227,
            "train_count": 308, "validation_count": 21, "excluded_count": 0,
            "unavailable_target_count": 0, "undersized_source_set_count": 4, "regional_families": [],
            "subject_count": 121, "shared_validation_count": 2, "validation_limit": 2,
            "validation_candidate_count": 121]
        let summary = try WorkbenchResult.decode(WorkbenchDatasetPlan.self,
            output: String(decoding: JSONSerialization.data(withJSONObject: plan), as: UTF8.self))
        XCTAssertTrue(summary.sourceIssues.isEmpty, "Older preview results must remain readable without source issues")
        XCTAssertNil(summary.smallerAlternateSourceSetCount, "Older preview results must remain readable without alternate-size counts")
        store.dataset?.resolutionPlans = ["2048": ["height": summary, "roughness": summary, "normal": summary]]
        store.dataset?.trainingSize = 2048
        store.training.size = 2048
        store.folderImportSize = 2048
        store.folderImportURL = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let missingSources = [
            ("fabric_pattern_05_2k", "Fabric Pattern 05/fabric_pattern_05_col_01_2k.png"),
            ("fabric_pattern_05_4k", "Fabric Pattern 05/fabric_pattern_05_col_01_4k.png"),
            ("fabric_pattern_07_2k", "Fabric Pattern 07/fabric_pattern_07_col_1_2k.png"),
            ("fabric_pattern_07_4k", "Fabric Pattern 07/fabric_pattern_07_col_1_4k.png")
        ]
        var heightPlan = plan
        heightPlan["smaller_alternate_source_set_count"] = 4
        heightPlan["unavailable_target_count"] = 4
        heightPlan["source_issues"] = missingSources.map { material, relativePath in
            ["material_id": material, "source_path": "/opt/ipde/sources_mats/\(relativePath)",
             "target": "height", "code": "not_published",
             "reason": "Poly Haven does not publish displacement for this material. It is skipped for displacement training; available normal and roughness maps remain usable.",
             "crop_count": 1] as [String: Any]
        }
        var roughnessPlan = plan
        roughnessPlan["unavailable_target_count"] = 1
        roughnessPlan["source_issues"] = [["material_id": "roughness_precision_4k",
            "source_path": "/opt/ipde/sources_mats/roughness_precision/roughness_precision_rough_4k.png",
            "target": "roughness", "code": "unavailable_target",
            "reason": "This roughness map cannot be used for training. Choose a supported original map and scan again.",
            "crop_count": 1] as [String: Any]]
        let unusualPaths = [
            "/opt/ipde/sources_mats/Granite Tile 04/granite_tile_04_rough_4k.txt",
            "/opt/ipde/sources_mats/Leaves Forest Ground/leaves_forest_ground_disp_4k.txt",
            "/opt/ipde/sources_mats/metal_plate/metal_plate_disp_4k.txt",
            "/opt/ipde/sources_mats/forest_leaves_03/forest_leaves_03_nor_gl_4k.txt"
        ]
        store.folderImport = try WorkbenchResult.decode(WorkbenchFolderImport.self, output: String(decoding: JSONSerialization.data(withJSONObject: [
            "folder_path": "/opt/ipde/sources_mats", "plan_path": "/tmp/preview.json", "plan_sha256": "proof",
            "index_sha256": "source-hash", "source_set_count": 231, "added_material_count": 231,
            "duplicate_material_count": 0, "ignored_file_count": 18, "recovered_map_count": 2,
            "warnings": unusualPaths.map { "\($0): PNG content detected despite .txt extension; read as PNG." },
            "plans": ["2048": ["height": heightPlan, "roughness": roughnessPlan, "normal": plan]]]), as: UTF8.self))
        let decodedIssues = try XCTUnwrap(store.folderImport?.plans["2048"]?["height"]?.sourceIssues)
        XCTAssertEqual(decodedIssues.count, missingSources.count)
        XCTAssertEqual(decodedIssues.first?.materialId, missingSources.first?.0)
        XCTAssertEqual(decodedIssues.first?.sourcePath, "/opt/ipde/sources_mats/Fabric Pattern 05/fabric_pattern_05_col_01_2k.png")
        XCTAssertEqual(decodedIssues.first?.target, "height")
        XCTAssertEqual(decodedIssues.first?.code, "not_published")
        XCTAssertEqual(decodedIssues.first?.cropCount, 1)
        XCTAssertEqual(store.folderImport?.plans["2048"]?["height"]?.smallerAlternateSourceSetCount, 4)
        XCTAssertEqual(store.folderImport?.recoveredMapCount, 2)
        try await snapshotSheet(NewMaterialDatasetSheet(store: store), name: "new-dataset", size: NSSize(width: 660, height: 390), minimumHeight: 380)
        try await snapshotSheet(MaterialDatasetInfoSheet(store: store), name: "dataset-info", size: NSSize(width: 700, height: 700))
        try await snapshotSheet(ImportDatasetFolderSheet(store: store), name: "folder-import", size: NSSize(width: 780, height: 700))
        var precisionPlan = plan
        precisionPlan["unavailable_target_count"] = 1
        precisionPlan["source_issues"] = [["material_id": "height_precision_4k",
            "source_path": "/opt/ipde/sources_mats/height_precision/height_precision_disp_4k.png",
            "target": "height", "code": "unsupported_precision",
            "reason": "The displacement map is 8-bit; displacement training requires a 16-bit PNG. Choose a 16-bit original and scan again.",
            "crop_count": 1] as [String: Any]]
        let precisionSummary = try WorkbenchResult.decode(WorkbenchDatasetPlan.self,
            output: String(decoding: JSONSerialization.data(withJSONObject: precisionPlan), as: UTF8.self))
        try await snapshotSheet(VStack(alignment: .leading) {
            DatasetPlanSummary(plans: ["height": precisionSummary])
            Spacer()
        }.padding(24).frame(width: 700, height: 600),
                                name: "folder-target-precision", size: NSSize(width: 700, height: 600))
    }

    func testFolderScanRetainsResolutionEditsAndRefreshesTiersBeforeProofImport() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.trainingSize = 2048
        let folder = fixture.root.appendingPathComponent("Original subjects")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let store = WorkbenchStore(preferences: fixture.defaults, workerOverride: { args, _ in
            fixture.calls.append(args)
            if args[0] == "scan-folder" {
                try await Task.sleep(for: .milliseconds(150))
                let document: [String: Any] = ["folder_path": folder.path, "plan_path": "/tmp/test-plan.json",
                    "plan_sha256": "plan-proof", "index_sha256": "source-hash", "source_set_count": 4,
                    "added_material_count": 4, "duplicate_material_count": 0, "ignored_file_count": 0,
                    "warnings": [], "plans": [:]]
                return String(decoding: try JSONSerialization.data(withJSONObject: document), as: UTF8.self)
            }
            if args[0] == "import-folder" { fixture.trainingSize = Int(fixture.value("--training-size", in: args)!)! }
            return try fixture.document()
        })
        store.training.size = 2048
        try await store.loadDataset(fixture.dataset)
        store.importMaterialFolder(folder)
        for _ in 0..<100 {
            if store.isScanningFolder { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertTrue(store.isScanningFolder)
        XCTAssertEqual(store.folderImportSize, 2048, "The sheet must open at the saved dataset resolution")
        store.folderImportSize = 512
        try await settled(store)
        XCTAssertEqual(store.folderImportSize, 512, "Scan completion cannot reset the user's choice")
        XCTAssertNil(store.folderImport?.recoveredMapCount, "Older scan results must remain readable without a recovery count")
        store.commitFolderImport(size: store.folderImportSize)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.datasetResolution, 512)
        let scans = fixture.calls.filter { $0[0] == "scan-folder" }
        XCTAssertEqual(scans.count, 2)
        XCTAssertEqual(fixture.value("--size", in: scans[0]), "2048")
        XCTAssertEqual(fixture.value("--size", in: scans[1]), "512")
        XCTAssertEqual(fixture.value("--target", in: scans[1]), "height")
        XCTAssertEqual(fixture.value("--folder", in: fixture.calls.last!), folder.path)
    }

    func testNewDatasetAddsPendingSourceFolderDirectlyWithoutTierRecoveryOrPreview() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let folder = fixture.root.appendingPathComponent("Original subjects")
        var scans: [[String]] = []
        let store = WorkbenchStore(preferences: fixture.defaults, workerOverride: { args, _ in
            fixture.calls.append(args)
            if args[0] == "create-dataset" { fixture.trainingSize = Int(fixture.value("--training-size", in: args)!)! }
            if args[0] == "scan-folder" {
                scans.append(args)
                let document: [String: Any] = ["folder_path": folder.path, "plan_path": "/tmp/tier-plan.json",
                    "plan_sha256": "tier-proof", "index_sha256": "post-recovery-hash", "source_set_count": 3,
                    "added_material_count": 3, "duplicate_material_count": 0, "ignored_file_count": 0,
                    "warnings": [], "plans": [:], "recovered_map_count": 4,
                    "source_recovery": ["completed": 4, "total": 4, "training_size": 1024, "recovered": 4,
                        "not_published": 0, "failed": 0, "already_present": 0]]
                return String(decoding: try JSONSerialization.data(withJSONObject: document), as: UTF8.self)
            }
            return try fixture.document()
        })
        store.pendingSourceFolder = folder
        store.createDataset(name: "Balanced", description: "", parentURL: fixture.root, size: 1024, sourceFolder: false)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertTrue(scans.isEmpty)
        XCTAssertNil(store.folderImportScannedSize)
        XCTAssertNil(store.folderImport)
        XCTAssertFalse(store.showImportFolderSheet)
        XCTAssertTrue(store.sourceRecoverySummary.isEmpty)
        XCTAssertFalse(fixture.calls.contains { ["train", "prepare-size"].contains($0[0]) })
        let imported = try XCTUnwrap(fixture.calls.first { $0[0] == "import-folder" })
        XCTAssertEqual(fixture.value("--expected-index-sha256", in: imported), "source-hash")
        XCTAssertEqual(fixture.value("--managed-import", in: imported), "true")
        XCTAssertTrue(imported.contains("--skip-source-recovery"))
        XCTAssertEqual(fixture.calls.map { $0[0] }, ["create-dataset", "import-folder"])
    }

    func testRecoveryProgressShowsPendingCountsAndRetainsActualFailures() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.operation("Setting up tiers") {
            store.recordTrainingProgress("{\"event\":\"source_recovery_started\",\"completed\":0,\"total\":6,\"training_size\":1024}\n")
            XCTAssertEqual(store.sourceRecoverySummary, "Tier setup: 0 / 6 maps checked")
            store.recordTrainingProgress("{\"event\":\"source_recovery_progress\",\"completed\":2,\"total\":6,\"message\":\"Fetching stone 4k height\"}\n")
            XCTAssertEqual(store.sourceRecoverySummary, "Tier setup: 2 / 6 maps checked")
            XCTAssertEqual(store.activity, "Fetching stone 4k height")
            store.recordTrainingProgress("{\"event\":\"source_recovery_completed\",\"completed\":6,\"total\":6,\"recovered\":3,\"not_published\":2,\"failed\":1}\n")
            XCTAssertEqual(store.sourceRecoverySummary, "Tier setup: 6 / 6 maps checked · 3 fetched · 2 unavailable · 1 failed")
            XCTAssertEqual(store.activity, store.sourceRecoverySummary)
        }
        try await settled(store)
        XCTAssertTrue(store.sourceRecoverySummary.contains("1 failed"))
    }

    func testChangingTrainingTargetDoesNotReloadOrReassignDataset() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        try await store.loadDataset(fixture.dataset)
        let calls = fixture.calls.count
        let hash = store.dataset?.indexSha256
        for target in ["height", "roughness", "normal"] { store.selectTrainingTarget(target) }
        XCTAssertEqual(fixture.calls.count, calls)
        XCTAssertEqual(store.dataset?.indexSha256, hash)
        XCTAssertFalse(store.isBusy)
    }

    private func managedBrowserDocument(_ fixture: Fixture) throws -> String {
        let wall = fixture.root.appendingPathComponent("Wall")
        let photos = fixture.root.appendingPathComponent("Photos")
        func map(_ path: URL, size: Int) -> [String: Any] {
            ["path": path.path, "width": size, "height": size, "sha256": "unchanged-original", "source_bits": 8]
        }
        func material(_ id: String, size: Int, split: String) -> [String: Any] {
            ["material_id": id, "subject_id": wall.path, "source_directory": wall.path,
             "samples": [["sample_id": id + "_full", "status": "unreviewed", "split": split,
                          "width": size, "height": size, "source_region_id": split == "original" ? "original" : "full",
                          "maps": ["input": map(wall.appendingPathComponent(id + "_basecolor.png"), size: size),
                                   "roughness": map(wall.appendingPathComponent(id + "_roughness.png"), size: size)]]]]
        }
        let inventory: [[String: Any]] = [
            ["path": wall.appendingPathComponent("wall_1k_basecolor.png").path, "directory": wall.path, "role": "input", "state": "paired", "width": 1024, "height": 1024],
            ["path": wall.appendingPathComponent("wall_2k_basecolor.png").path, "directory": wall.path, "role": "input", "state": "paired", "width": 2048, "height": 2048],
            ["path": wall.appendingPathComponent("wall_ao.png").path, "directory": wall.path, "role": "ancillary", "state": "paired", "width": 2048, "height": 2048],
            ["path": photos.appendingPathComponent("photo.jpg").path, "directory": photos.path, "role": "unknown", "state": "needs_assignment", "width": 2048, "height": 3072,
             "issue": "Assign a map role before material training."]]
        let document: [String: Any] = ["dataset_path": fixture.dataset.path, "index_sha256": "managed-hash", "name": "Managed folders",
            "training_size": 2048, "source_set_count": 2, "registered_folder_count": 2,
            "added_file_count": 4, "added_material_count": 2, "source_inventory": inventory,
            "original_materials": [material("wall_1k", size: 1024, split: "original"), material("wall_2k", size: 2048, split: "original")],
            "materials": [material("wall_2k", size: 2048, split: "train")]]
        return String(decoding: try JSONSerialization.data(withJSONObject: document), as: UTF8.self)
    }

    private func snapshotSheet<V: View>(_ view: V, name: String, size: NSSize, minimumHeight: CGFloat = 500) async throws {
        let host = NSHostingView(rootView: view.environment(\.colorScheme, .dark))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderFront(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(250))
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let image = NSImage(cgImage: try XCTUnwrap(bitmap.cgImage), size: host.bounds.size)
        let attachment = XCTAttachment(image: image)
        attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent("ipde-\(name)-validation.png")
        try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: destination)
        print("DATASET_LAYOUT_SNAPSHOT=\(destination.path)")
        XCTAssertGreaterThanOrEqual(host.bounds.height, minimumHeight)
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
        var trainingSize = 1024
        var sourceTierCount = 1
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
        func document(prepared: Bool = false, indexHash: String = "source-hash") throws -> String {
            var object: [String: Any] = ["dataset_path": prepared ? root.appendingPathComponent("prepared").path : dataset.path,
                "index_sha256": indexHash, "name": name, "description": description, "training_size": trainingSize, "source_tier_count": sourceTierCount, "materials": []]
            if prepared { object["preparation"] = ["source_dataset_path": dataset.path, "source_index_sha256": hash,
                "prepared_dataset_path": root.appendingPathComponent("prepared").path, "crop_size": 1024,
                "reused": false, "target_resized": false, "original_dataset_modified": false] }
            return String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        }
        func store(trash: @escaping (URL) throws -> Void = { _ in }) -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, trashHandler: trash, workerOverride: { args, script in
                self.calls.append(args)
                if ["edit-dataset", "create-dataset"].contains(args[0]), !self.failEdit,
                   let size = self.value("--training-size", in: args).flatMap(Int.init) { self.trainingSize = size }
                if ["edit-dataset", "create-dataset"].contains(args[0]), !self.failEdit,
                   let count = self.value("--source-tier-count", in: args).flatMap(Int.init) { self.sourceTierCount = count }
                if args.first == "capabilities" { throw StudioError("Model runtime is not installed") }
                XCTAssertEqual(script, args.first)
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
                if args.first == "dataset", args.contains("--skip-source-recovery") { return try self.document(indexHash: self.hash) }
                return try self.document()
            })
        }
    }
}
