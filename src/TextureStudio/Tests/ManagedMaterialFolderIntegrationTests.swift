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
        XCTAssertFalse(report.sources.isEmpty)
        XCTAssertTrue(report.sources.allSatisfy { $0.width >= 2048 && $0.height >= 2048 })
        try JSONSerialization.data(withJSONObject: ["minimum_native_size": report.minimumSize,
            "selected_native_originals": report.sources.count, "discovered_diffuse_originals": report.discoveredCount,
            "undersized_originals": report.undersizedCount, "unreadable_originals": report.unreadableCount,
            "seconds": elapsed, "source_files_modified": false,
            "chosen_source_paths": report.sources.map { $0.url.path }], options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-managed-classifier-proof.json"))
    }

    func testDeveloperCorpusFolderReadAccountsForEveryCurrentPNG() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_MANAGED_FOLDER_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in read-only inspection of the real source folder.")
        }
        let source = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let pngPaths = try pngInventory(source)
        let manifest = source.appendingPathComponent("dataset.json")
        let manifestBefore = try? Data(contentsOf: manifest)
        let started = ProcessInfo.processInfo.systemUptime
        let info = try await command(["dataset", "--dataset", source.path, "--review-size", "2048"])
        let firstReadSeconds = ProcessInfo.processInfo.systemUptime - started
        let inventory = try XCTUnwrap(info["source_inventory"] as? [[String: Any]])
        let paths = Set(inventory.compactMap { $0["path"] as? String })
        XCTAssertFalse(pngPaths.isEmpty)
        XCTAssertTrue(pngPaths.isSubset(of: paths), "Every physical PNG remains visible, including supplemental and unassigned files")
        XCTAssertTrue(paths.allSatisfy { $0.hasPrefix(source.path + "/") })
        XCTAssertEqual(inventory.filter { ($0["path"] as? String)?.lowercased().hasSuffix(".png") == true }.count, pngPaths.count)
        let originals = try XCTUnwrap(info["original_materials"] as? [[String: Any]])
        XCTAssertFalse(originals.isEmpty)
        let crops = (info["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertTrue(crops.allSatisfy { $0["width"] as? Int == 2048 && $0["height"] as? Int == 2048 })
        let reloadStarted = ProcessInfo.processInfo.systemUptime
        let reopened = try await command(["dataset", "--dataset", source.path, "--review-size", "2048"])
        let reloadSeconds = ProcessInfo.processInfo.systemUptime - reloadStarted
        XCTAssertEqual(try? Data(contentsOf: manifest), manifestBefore, "Browsing never writes a dataset reference manifest into the source folder")
        XCTAssertEqual((reopened["source_inventory"] as? [Any])?.count, inventory.count)
        let report: [String: Any] = ["folder": source.path, "png_count": pngPaths.count,
            "accounted_file_count": inventory.count, "current_original_sets": originals.count,
            "visible_exact_2048_crops": crops.count, "first_read_seconds": firstReadSeconds,
            "reload_seconds": reloadSeconds, "source_files_modified": false, "warnings": info["warnings"] ?? []]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-managed-folder-proof.json"))
        print("Source corpus read: \(firstReadSeconds)s; reload: \(reloadSeconds)s; files: \(inventory.count); originals: \(originals.count)")
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
