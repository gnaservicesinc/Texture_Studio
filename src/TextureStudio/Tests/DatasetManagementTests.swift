import AppKit
import SwiftUI
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
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: fixture.document(prepared: true))
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

    func testNativeLifecycleKeepsOriginalBytes() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let source = fixture.root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        let input = source.appendingPathComponent("diffuse.png")
        let height = source.appendingPathComponent("height.png")
        let diffuse = NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0),
            pixels: Data(repeating: 127, count: 256 * 256 * 3), colorChunks: [])
        var samples = Data()
        for code in 0..<65536 { samples.append(UInt8(code >> 8)); samples.append(UInt8(code & 255)) }
        let scalar = NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0), pixels: samples, colorChunks: [])
        try diffuse.encoded().write(to: input)
        try scalar.encoded().write(to: height)
        let originalBytes = try Data(contentsOf: height)
        var trashed: [URL] = []
        let store = WorkbenchStore(preferences: fixture.defaults, trashHandler: { url in
            trashed.append(url)
            try FileManager.default.moveItem(at: url, to: fixture.root.appendingPathComponent("Recovered Dataset"))
        })
        store.training.size = 1024
        store.openDataset(source)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertTrue(store.showNewDatasetSheet)
        XCTAssertEqual(store.pendingSourceFolder, source)
        store.pendingSourceFolder = nil
        store.showNewDatasetSheet = false
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
        XCTAssertEqual(store.selectedSample?.split, "train")
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
        XCTAssertTrue(store.showImportFolderSheet)
        XCTAssertEqual(store.folderImport?.addedMaterialCount, 1)
        XCTAssertEqual(store.samples.count, 0, "Scanning is read-only until Import is pressed")
        let indexURL = try XCTUnwrap(store.datasetFolderURL).appendingPathComponent("dataset.json")
        var edited = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any])
        edited["description"] = "Changed in another window"
        try JSONSerialization.data(withJSONObject: edited).write(to: indexURL, options: .atomic)
        store.commitFolderImport(size: 256)
        try await settled(store)
        XCTAssertNotNil(store.error)
        XCTAssertTrue(store.showImportFolderSheet)
        XCTAssertEqual(store.samples.count, 0)
        store.importMaterialFolder(source)
        try await settled(store)
        XCTAssertNil(store.error, "Scan Again must reconnect a changed dataset, not repeat the stale-index failure")
        store.commitFolderImport(size: 256)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertFalse(store.showImportFolderSheet)
        XCTAssertEqual(store.datasetResolution, 256)
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

    func testRawFolderStartsSetupWithoutSendingItToDatasetReader() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let source = fixture.root.appendingPathComponent("Nested material sources")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let store = fixture.store()
        store.openDataset(source)
        XCTAssertTrue(store.showNewDatasetSheet)
        XCTAssertEqual(store.pendingSourceFolder, source)
        XCTAssertTrue(fixture.calls.isEmpty)
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
        XCTAssertTrue(store.showNewDatasetSheet)
        XCTAssertEqual(store.pendingSourceFolder, source)
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
        store.dataset?.resolutionPlans = ["2048": ["height": summary, "roughness": summary, "normal": summary]]
        store.dataset?.trainingSize = 2048
        store.training.size = 2048
        store.folderImportSize = 2048
        store.folderImportURL = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let missingSources = [
            ("fabric_pattern_05_2k", "Fabric Pattern 05/fabric_pattern_05_col_01_2k.png"),
            ("fabric_pattern_05_4k", "Fabric Pattern 05/fabric_pattern_05_col_01_4k.png"),
            ("fabric_pattern_07_2k", "Fabric Pattern 07/fabric_pattern_07_col_1_2k.png"),
            ("fabric_pattern_07_4k", "Fabric Pattern 07/fabric_pattern_07_col_1_4k.png"),
            ("granite_tile_04_4k", "Granite Tile 04/granite_tile_04_diff_4k.png"),
            ("leather_red_03_4k", "Leather Red 03/leather_red_03_coll1_4k.png")
        ]
        var heightPlan = plan
        heightPlan["unavailable_target_count"] = 7
        heightPlan["source_issues"] = missingSources.map { material, relativePath in
            ["material_id": material, "source_path": "/opt/ipde/sources_mats/\(relativePath)",
             "target": "height", "code": "missing_target",
             "reason": "No displacement map was found for this source set. Add a matching displacement PNG to this folder and scan again.",
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
            "duplicate_material_count": 0, "ignored_file_count": 18,
            "warnings": unusualPaths.map { "\($0): PNG content detected despite .txt extension; read as PNG." },
            "plans": ["2048": ["height": heightPlan, "roughness": roughnessPlan, "normal": plan]]]), as: UTF8.self))
        let decodedIssues = try XCTUnwrap(store.folderImport?.plans["2048"]?["height"]?.sourceIssues)
        XCTAssertEqual(decodedIssues.count, missingSources.count)
        XCTAssertEqual(decodedIssues.first?.materialId, missingSources.first?.0)
        XCTAssertEqual(decodedIssues.first?.sourcePath, "/opt/ipde/sources_mats/Fabric Pattern 05/fabric_pattern_05_col_01_2k.png")
        XCTAssertEqual(decodedIssues.first?.target, "height")
        XCTAssertEqual(decodedIssues.first?.code, "missing_target")
        XCTAssertEqual(decodedIssues.first?.cropCount, 1)
        try await snapshotSheet(NewMaterialDatasetSheet(store: store), name: "new-dataset", size: NSSize(width: 660, height: 600))
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

    func testFolderScanRetainsResolutionEditsWithoutRepeatingScan() async throws {
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
        store.commitFolderImport(size: store.folderImportSize)
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.datasetResolution, 512)
        XCTAssertEqual(fixture.calls.filter { $0[0] == "scan-folder" }.count, 1)
        XCTAssertEqual(fixture.value("--folder", in: fixture.calls.last!), folder.path)
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

    private func snapshotSheet<V: View>(_ view: V, name: String, size: NSSize) async throws {
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
        XCTAssertGreaterThan(host.bounds.height, 500)
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
                "index_sha256": "source-hash", "name": name, "description": description, "training_size": trainingSize, "materials": []]
            if prepared { object["preparation"] = ["source_dataset_path": dataset.path, "source_index_sha256": "source-hash",
                "prepared_dataset_path": root.appendingPathComponent("prepared").path, "crop_size": 1024,
                "reused": false, "target_resized": false, "original_dataset_modified": false] }
            return String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        }
        func store(trash: @escaping (URL) throws -> Void = { _ in }) -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, trashHandler: trash, workerOverride: { args, script in
                self.calls.append(args)
                if ["edit-dataset", "create-dataset"].contains(args[0]), !self.failEdit,
                   let size = self.value("--training-size", in: args).flatMap(Int.init) { self.trainingSize = size }
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
                return try self.document()
            })
        }
    }
}
