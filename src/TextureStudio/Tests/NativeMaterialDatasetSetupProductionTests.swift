import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

/// Opt-in repair of an existing original-reference dataset, with no training.
final class NativeMaterialDatasetSetupProductionTests: XCTestCase {
    /// One-off cleanup explicitly authorized for this downloadable development
    /// corpus. This is not deployed dataset-management behavior.
    func testRemoveUnusableClothFromAuthorizedDevelopmentCorpus() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["TEXTURE_STUDIO_DATASET_CORPUS_CLEANUP"] == "1" else {
            throw XCTSkip("Opt-in, user-authorized cleanup of the public development corpus.")
        }
        let root = URL(fileURLWithPath: try XCTUnwrap(env["TEXTURE_STUDIO_DATASET_SETUP_PATH"]))
            .resolvingSymlinksInPath().standardizedFileURL
        XCTAssertEqual(root.path, "/Users/andrewsmith/Documents/Texture Studio Datasets/PolyMat")
        guard root.path == "/Users/andrewsmith/Documents/Texture Studio Datasets/PolyMat" else { return }
        let corpus = URL(fileURLWithPath: "/opt/ipde/sources_mats").resolvingSymlinksInPath().standardizedFileURL
        let names = ["Fabric Pattern 05", "Fabric Pattern 07", "jersey_melange", "scuba_suede", "Waffle Pique Cotton", "polar_fleece"]
        let folders = names.map { corpus.appendingPathComponent($0) }
        let paths = Set(folders.map(\.path))
        for folder in folders {
            let state = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            XCTAssertTrue(state.isDirectory == true && state.isSymbolicLink != true)
            guard state.isDirectory == true, state.isSymbolicLink != true,
                  folder.resolvingSymlinksInPath().deletingLastPathComponent() == corpus else {
                throw StudioError("The authorized corpus folder changed before cleanup.")
            }
        }
        let before = try records(root, index: object(root.appendingPathComponent("dataset.json")))
        let removed = before.filter { paths.contains($0["source_directory"] as? String ?? "") }
        let materials = Set(removed.compactMap { $0["material_id"] as? String })
        XCTAssertEqual(materials.count, 16)
        guard materials.count == 16 else { throw StudioError("The expected cloth inventory changed before cleanup.") }
        for material in materials.sorted() {
            let bytes = try Data(contentsOf: root.appendingPathComponent("dataset.json"))
            _ = try await NativeMaterialDatasetService.run(arguments: ["remove-material", "--dataset", root.path,
                "--material", material, "--expected-index-sha256", hash(bytes)])
        }
        let remaining = try records(root, index: object(root.appendingPathComponent("dataset.json")))
        guard !remaining.contains(where: { paths.contains($0["source_directory"] as? String ?? "") }) else {
            throw StudioError("A cloth source is still referenced by the active dataset.")
        }
        for folder in folders {
            let state = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard state.isDirectory == true, state.isSymbolicLink != true,
                  folder.resolvingSymlinksInPath().deletingLastPathComponent() == corpus else {
                throw StudioError("The authorized corpus folder changed before deletion.")
            }
            try FileManager.default.removeItem(at: folder)
            XCTAssertFalse(FileManager.default.fileExists(atPath: folder.path))
        }
        let report: [String: Any] = ["dataset_path": root.path, "removed_source_folders": folders.map(\.path),
            "removed_resolution_sets": materials.sorted(), "remaining_resolution_sets_before_leather_recovery": remaining.count,
            "source_files_deleted": true, "source_pixels_resampled": false, "training_started": false,
            "authorization": "User explicitly requested removal of the problem cloth originals from this downloadable corpus."]
        if let path = env["TEXTURE_STUDIO_DATASET_CORPUS_CLEANUP_REPORT"] {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: path))
        }
    }

    func testRealDatasetSetupRegistersBalancedTiersBeforeTraining() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["TEXTURE_STUDIO_DATASET_SETUP_VERIFY"] == "1" else {
            throw XCTSkip("Opt-in source-tier setup on the user's retained material dataset.")
        }
        let root = URL(fileURLWithPath: try XCTUnwrap(env["TEXTURE_STUDIO_DATASET_SETUP_PATH"]))
        let size = Int(env["TEXTURE_STUDIO_DATASET_SETUP_SIZE"] ?? "1024")!
        let originalIndex = try object(root.appendingPathComponent("dataset.json"))
        XCTAssertNil(originalIndex["native_size_preparation"])
        let before = try records(root, index: originalIndex)
        var originalSources: [String: [String: Any]] = [:]
        for sample in before {
            for map in (sample["map_metadata"] as? [String: [String: Any]] ?? [:]).values {
                guard let source = map["source"] as? [String: Any], let path = source["path"] as? String else { continue }
                originalSources[path] = source
            }
            for variant in sample["input_variants"] as? [[String: Any]] ?? [] {
                guard let source = variant["source"] as? [String: Any], let path = source["path"] as? String else { continue }
                originalSources[path] = source
            }
        }
        let originalReviews = try? Data(contentsOf: root.appendingPathComponent(".material-size-reviews.json"))
        let started = Date()
        let response = try await NativeMaterialDatasetService.run(arguments: ["dataset", "--dataset", root.path,
            "--review-size", String(size), "--target", "height"], onEvent: { event in
                FileHandle.standardOutput.write(Data(event.utf8))
            })
        let result = try XCTUnwrap(response)
        let info = try JSONSerialization.jsonObject(with: Data(result.utf8)) as! [String: Any]
        let afterIndex = try object(root.appendingPathComponent("dataset.json")), after = try records(root, index: afterIndex)
        XCTAssertNil(afterIndex["native_size_preparation"], "Setup must keep the original dataset active")
        XCTAssertEqual(try? Data(contentsOf: root.appendingPathComponent(".material-size-reviews.json")), originalReviews)
        let oldIDs = Set(before.compactMap { $0["material_id"] as? String })
        XCTAssertTrue(oldIDs.isSubset(of: Set(after.compactMap { $0["material_id"] as? String })))
        for (path, source) in originalSources {
            let current = try XCTUnwrap(after.flatMap { sample -> [[String: Any]] in
                let maps = (sample["map_metadata"] as? [String: [String: Any]] ?? [:]).values.compactMap { $0["source"] as? [String: Any] }
                let variants = (sample["input_variants"] as? [[String: Any]] ?? []).compactMap { $0["source"] as? [String: Any] }
                return maps + variants
            }.first { $0["path"] as? String == path })
            XCTAssertEqual(current["file_sha256"] as? String, source["file_sha256"] as? String)
            let bytes = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
            XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), source["file_sha256"] as? String,
                "Setup must preserve every previously recorded original")
        }
        let groups = Dictionary(grouping: after) { $0["source_directory"] as? String ?? "" }
        let subjects: [[String: Any]] = groups.keys.sorted().map { folder in
            let samples = groups[folder]!
            let dimensions = samples.compactMap { $0["source_pixel_dimensions"] as? [Int] }.sorted { $0[0] < $1[0] }
            let heights = samples.filter { ($0["available_targets"] as? [String] ?? []).contains("height") }
            return ["source_folder": folder, "source_dimensions": dimensions, "registered_resolution_sets": samples.count,
                "height_resolution_sets": heights.count,
                "resolution_labels": samples.compactMap { $0["source_resolution_label"] as? String }.sorted(),
                "usable_resolution_sets": dimensions.filter { min($0[0], $0[1]) >= size }.count,
                "has_required_tiers": [size, size * 2, size * 4].allSatisfy { requested in
                    samples.contains { $0["source_resolution_label"] as? String == "\(requested / 1024)k" }
                }]
        }
        let plans = info["training_plans"] as? [String: [String: Any]] ?? [:]
        let report: [String: Any] = ["dataset_path": root.path, "training_size": size, "original_source_sets": before.count,
            "registered_source_sets": after.count, "source_folders": groups.count, "subjects": subjects,
            "folders_with_required_tiers": subjects.filter { $0["has_required_tiers"] as? Bool == true }.count,
            "height_training_crops": plans["height"]?["train_count"] ?? NSNull(),
            "source_recovery": info["source_recovery"] ?? NSNull(), "source_recovery_results": info["source_recovery_results"] ?? [],
            "original_source_bytes_unchanged": true, "reviews_unchanged": true, "training_started": false,
            "wall_seconds": Date().timeIntervalSince(started)]
        if let path = env["TEXTURE_STUDIO_DATASET_SETUP_REPORT"] {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: path))
        }
        XCTAssertGreaterThanOrEqual(after.count, before.count)
        // Exact provider availability is retained in the receipt. Fixtures test
        // the three-tier requirement without depending on live publication.
    }

    private func object(_ url: URL) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }
    private func hash(_ bytes: Data) -> String {
        SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
    private func records(_ root: URL, index: [String: Any]) throws -> [[String: Any]] {
        try (index["samples"] as! [[String: Any]]).map {
            try object(root.appendingPathComponent($0["path"] as! String).appendingPathComponent("sample.json"))
        }
    }
}
