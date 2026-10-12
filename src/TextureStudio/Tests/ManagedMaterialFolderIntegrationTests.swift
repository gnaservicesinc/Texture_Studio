import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class ManagedMaterialFolderIntegrationTests: XCTestCase {
    func testDeveloperClassifierDiscoveryUsesNativeImagesAtOrAboveTheSelectedGrid() throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_MANAGED_FOLDER_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in header-only classifier discovery against the real corpus.")
        }
        let started = ProcessInfo.processInfo.systemUptime
        let report = try NativeDiffuseClassifierDatasetService.discoverReport(dataset: nil,
            sourceFolder: URL(fileURLWithPath: "/opt/ipde/sources_mats"), minimumSize: 2048)
        let elapsed = ProcessInfo.processInfo.systemUptime - started
        XCTAssertGreaterThan(report.sources.count, 140)
        XCTAssertTrue(report.sources.allSatisfy { $0.width >= 2048 && $0.height >= 2048 })
        XCTAssertGreaterThan(report.undersizedCount, 0)
        XCTAssertEqual(report.unreadableCount, 0)
        try JSONSerialization.data(withJSONObject: ["minimum_native_size": report.minimumSize,
            "selected_native_originals": report.sources.count, "discovered_diffuse_originals": report.discoveredCount,
            "undersized_originals": report.undersizedCount, "unreadable_originals": report.unreadableCount,
            "seconds": elapsed, "source_files_modified": false,
            "chosen_source_paths": report.sources.map { $0.url.path }], options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-managed-classifier-proof.json"))
        XCTAssertLessThan(elapsed, 1, "Classifier selection should use local headers, without network or pixel-body work")
    }

    func testDeveloperLegacyDatasetManagedAdditionKeepsExistingMetadataAndAddsMissingFolders() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_MANAGED_FOLDER_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in migration proof using a temporary copy of real dataset metadata.")
        }
        let live = URL(fileURLWithPath: "/Users/andrewsmith/Documents/Texture Studio Datasets/MatterLoom 2k")
        guard FileManager.default.fileExists(atPath: live.appendingPathComponent("dataset.json").path) else {
            throw XCTSkip("The optional existing dataset fixture is unavailable.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("managed-folder-legacy-proof-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let indexBytes = try Data(contentsOf: live.appendingPathComponent("dataset.json"))
        let index = try XCTUnwrap(JSONSerialization.jsonObject(with: indexBytes) as? [String: Any])
        let entries = try XCTUnwrap(index["samples"] as? [[String: Any]])
        try indexBytes.write(to: root.appendingPathComponent("dataset.json"))
        var metadataBefore: [String: Data] = [:]
        for entry in entries {
            let relative = try XCTUnwrap(entry["path"] as? String)
            XCTAssertFalse(relative.hasPrefix("/")); XCTAssertFalse(relative.split(separator: "/").contains(".."))
            let bytes = try Data(contentsOf: live.appendingPathComponent(relative).appendingPathComponent("sample.json"))
            let folder = root.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try bytes.write(to: folder.appendingPathComponent("sample.json"))
            metadataBefore[relative] = bytes
        }
        let review = live.appendingPathComponent(".material-size-reviews.json")
        if FileManager.default.fileExists(atPath: review.path) {
            try FileManager.default.copyItem(at: review, to: root.appendingPathComponent(review.lastPathComponent))
        }
        let result = try await command(["import-folder", "--dataset", root.path, "--folder", "/opt/ipde/sources_mats",
                                       "--managed-import", "true", "--review-size", "2048"])
        let originals = try XCTUnwrap(result["original_materials"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(originals.count, 459)
        XCTAssertEqual(result["added_material_count"] as? Int, originals.count - entries.count, "Existing verified sets must be recognized by their paths; new source folders and retained collision tiers add only their distinct sets")
        for (relative, bytes) in metadataBefore {
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent(relative).appendingPathComponent("sample.json")), bytes)
        }
        XCTAssertEqual(try Data(contentsOf: live.appendingPathComponent("dataset.json")), indexBytes)
    }

    func testDeveloperCorpusManagedAdditionAccountsForEveryPNGAndRetainsBothPineBarkFolders() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_MANAGED_FOLDER_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in managed-folder registration of the real corpus; writes only a temporary dataset.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("managed-folder-proof-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), source = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Managed corpus", "--training-size", "2048"])
        let pngPaths = try pngInventory(source)
        let started = ProcessInfo.processInfo.systemUptime
        let added = try await command(["import-folder", "--dataset", dataset.path, "--folder", source.path,
                                       "--managed-import", "true", "--review-size", "2048"])
        let additionSeconds = ProcessInfo.processInfo.systemUptime - started
        let inventory = try XCTUnwrap(added["source_inventory"] as? [[String: Any]])
        let paths = Set(inventory.compactMap { $0["path"] as? String })
        XCTAssertTrue(pngPaths.isSubset(of: paths), "Every PNG must remain accounted for, including supplemental and unassigned files")
        XCTAssertGreaterThanOrEqual(pngPaths.count, 1885)
        XCTAssertEqual(inventory.filter { ($0["path"] as? String)?.lowercased().hasSuffix(".png") == true }.count, pngPaths.count, "The managed catalog must account for the current live corpus, including user additions")
        let originals = try XCTUnwrap(added["original_materials"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(originals.count, 459, "Distinct folders with normalized names must not discard three tiers")
        let originalRows = originals.flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertTrue(originalRows.contains { $0["width"] as? Int == 1024 })
        let crops = (added["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertTrue(crops.allSatisfy { $0["width"] as? Int == 2048 && $0["height"] as? Int == 2048 })
        for row in originalRows {
            for map in (row["maps"] as? [String: [String: Any]] ?? [:]).values {
                XCTAssertNil(map["sha256"] as? String, "Managed addition must not hash pixel bodies")
            }
        }
        let indexBeforeReload = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let reloadStarted = ProcessInfo.processInfo.systemUptime
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        let reloadSeconds = ProcessInfo.processInfo.systemUptime - reloadStarted
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), indexBeforeReload,
                       "Browsing must not run implicit inventory maintenance or verification")
        XCTAssertEqual((reopened["source_inventory"] as? [Any])?.count, inventory.count)
        let report: [String: Any] = ["folder": source.path, "png_count": pngPaths.count,
            "accounted_file_count": inventory.count, "registered_original_sets": originals.count,
            "visible_exact_2048_crops": crops.count, "addition_seconds": additionSeconds,
            "reload_seconds": reloadSeconds, "source_body_hashes_during_addition": 0,
            "source_files_modified": false, "warnings": added["warnings"] ?? []]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-managed-folder-proof.json"))
        print("Managed corpus addition: \(additionSeconds)s; reload: \(reloadSeconds)s; files: \(inventory.count); originals: \(originals.count)")
        XCTAssertLessThan(additionSeconds, 3, "Adding folder references should be bounded to a few seconds, without full image verification")
    }

    private func pngInventory(_ root: URL) throws -> Set<String> {
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]))
        var paths = Set<String>()
        for case let path as URL in enumerator where path.pathExtension.lowercased() == "png" {
            paths.insert(path.resolvingSymlinksInPath().standardizedFileURL.path)
        }
        return paths
    }
    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let output = try await NativeMaterialDatasetService.run(arguments: arguments + ["--skip-source-recovery"])
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(output).utf8)) as? [String: Any])
    }
}
