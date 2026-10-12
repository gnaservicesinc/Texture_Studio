import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialQuickScanTests: XCTestCase {
    func testQuickPreviewAndImportDoNotInvalidateOwnProofThroughInventoryMaintenance() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Bound preview", "--training-size", "256"])
        let owned = dataset.appendingPathComponent("sources/Owned")
        for (folder, family) in [(owned, "owned"), (sources, "external")] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try image(channels: 3, bits: 8).write(to: folder.appendingPathComponent(family + "_diff_1k.png"))
            try image(channels: 1, bits: 16).write(to: folder.appendingPathComponent(family + "_disp_1k.png"))
        }
        let indexBefore = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let plan = root.appendingPathComponent("preview.json")
        let preview = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                        "--plan", plan.path, "--size", "256", "--quick-preview", "true"])
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), indexBefore)
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path,
                                         "--plan", plan.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String)])
        XCTAssertEqual(imported["added_material_count"] as? Int, 1)
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(reopened["source_set_count"] as? Int, 2, "Deferred owned-source discovery remains available on reload")
    }
    func testQuickPreviewDoesNotFetchMissingProviderTiers() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        try image(channels: 3, bits: 8, size: 1024).write(to: sources.appendingPathComponent("brick_diff_1k.png"))
        try image(channels: 1, bits: 16, size: 1024).write(to: sources.appendingPathComponent("brick_disp_1k.png"))
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Local preview", "--training-size", "2048"])
        let probe = QuickScanNetworkProbe()
        let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { _ in
            await probe.record(); throw StudioError("Quick preview must stay local")
        }, download: { _, _ in
            await probe.record(); throw StudioError("Quick preview must stay local")
        }))
        let text = try await NativeMaterialDatasetService.run(arguments: ["scan-folder", "--dataset", dataset.path,
            "--folder", sources.path, "--size", "2048", "--plan", root.appendingPathComponent("preview.json").path,
            "--quick-preview", "true"], sourceRecovery: recovery)
        let preview = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(text).utf8)) as? [String: Any])
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        let requests = await probe.count()
        XCTAssertEqual(requests, 0)
    }
    func testQuickPreviewReadsHeadersAndImportVerifiesFullSourceBytes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let rgb = try image(channels: 3, bits: 8), scalar = try image(channels: 1, bits: 16)
        let input = sources.appendingPathComponent("Carpet_COMPILED_baseColor_1k.png")
        let height = sources.appendingPathComponent("Carpet_COMPILED_height_1k.png")
        try rgb.write(to: input); try scalar.write(to: height)
        try Data([1, 2, 3]).write(to: sources.appendingPathComponent("Carpet_COMPILED_ambientOcclusion_1k.png"))
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Quick scan", "--training-size", "256"])
        let plan = root.appendingPathComponent("preview.json")
        let preview = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                        "--plan", plan.path, "--size", "256", "--quick-preview", "true"])
        XCTAssertEqual(preview["warnings"] as? [String], [])
        XCTAssertEqual(preview["added_material_count"] as? Int, 1)
        XCTAssertEqual(Set((preview["plans"] as? [String: Any] ?? [:]).keys), ["256"])
        let proof = try object(plan)
        XCTAssertEqual(proof["source_verification"] as? String, "headers-only-v1")
        let maps = try XCTUnwrap(((proof["materials"] as? [[String: Any]])?.first)?["maps"] as? [String: [String: Any]])
        XCTAssertNil(maps["input"]?["file_sha256"], "A new header preview must not read and hash compressed pixel bytes")
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path,
                                         "--plan", plan.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String)])
        XCTAssertEqual(imported["added_material_count"] as? Int, 1)
        let index = try object(dataset.appendingPathComponent("dataset.json"))
        let entry = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
        let sample = try object(dataset.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json"))
        let durableMaps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
        XCTAssertEqual((durableMaps["input"]?["source"] as? [String: Any])?["file_sha256"] as? String, checksum(rgb))
        XCTAssertEqual((durableMaps["height"]?["source"] as? [String: Any])?["file_sha256"] as? String, checksum(scalar))
        let repeated = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                         "--plan", root.appendingPathComponent("repeat.json").path, "--size", "256", "--quick-preview", "true"])
        XCTAssertEqual(repeated["duplicate_material_count"] as? Int, 1)
        XCTAssertEqual(repeated["added_material_count"] as? Int, 0)
        XCTAssertEqual(try Data(contentsOf: input), rgb); XCTAssertEqual(try Data(contentsOf: height), scalar)
    }

    func testQuickHeaderPreviewCannotRegisterCorruptPixelChunks() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        var rgb = try image(channels: 3, bits: 8); rgb[45] ^= 1
        try rgb.write(to: sources.appendingPathComponent("brick_diff_1k.png"))
        try image(channels: 1, bits: 16).write(to: sources.appendingPathComponent("brick_disp_1k.png"))
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Invalid pixel chunks", "--training-size", "256"])
        let plan = root.appendingPathComponent("preview.json")
        let preview = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                        "--plan", plan.path, "--size", "256", "--quick-preview", "true"])
        XCTAssertEqual(preview["added_material_count"] as? Int, 1)
        do {
            _ = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path,
                                   "--plan", plan.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String)])
            XCTFail("Header-only inspection is not enough to register a source")
        } catch { XCTAssertTrue(String(describing: error).contains("verification")) }
        XCTAssertEqual((try object(dataset.appendingPathComponent("dataset.json"))["samples"] as? [Any])?.count, 0)
    }

    func testDeveloperCorpusQuickPreviewTiming() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_QUICK_SCAN_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in timing of the developer's full source corpus without source writes or downloads.")
        }
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Corpus timing", "--training-size", "2048"])
        var timings: [Double] = [], setCounts: [Int] = [], warningCounts: [Int] = [], warnings: [String] = []
        for index in 0..<3 {
            let started = ProcessInfo.processInfo.systemUptime
            let preview = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                            "--plan", root.appendingPathComponent("preview-\(index).json").path,
                                            "--size", "2048", "--quick-preview", "true"])
            timings.append(ProcessInfo.processInfo.systemUptime - started)
            setCounts.append(preview["source_set_count"] as? Int ?? 0)
            warningCounts.append((preview["warnings"] as? [String])?.count ?? 0)
            warnings = preview["warnings"] as? [String] ?? []
        }
        let report: [String: Any] = ["folder": sources.path, "seconds": timings, "source_sets": setCounts,
                                   "warning_counts": warningCounts, "warnings": warnings, "training_size": 2048]
        let reportURL = URL(fileURLWithPath: "/tmp/texture-quick-scan-benchmark.json")
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: reportURL)
        print("Quick scan seconds: \(timings); source sets: \(setCounts); warnings: \(warningCounts)")
        XCTAssertTrue(setCounts.allSatisfy { $0 >= 400 })
        XCTAssertLessThan(timings.max() ?? .infinity, 1, "Full input preview must complete in under one second")
    }

    func testDeveloperExistingDatasetQuickPreviewTiming() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_QUICK_SCAN_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in timing against the existing developer dataset without registration changes.")
        }
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = URL(fileURLWithPath: "/Users/andrewsmith/Documents/Texture Studio Datasets/BadMonkey")
        let sources = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let indexBefore = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        var timings: [Double] = [], duplicates: [Int] = []
        for index in 0..<3 {
            let started = ProcessInfo.processInfo.systemUptime
            let preview = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                            "--plan", root.appendingPathComponent("preview-\(index).json").path,
                                            "--size", "2048", "--quick-preview", "true"])
            timings.append(ProcessInfo.processInfo.systemUptime - started)
            duplicates.append(preview["duplicate_material_count"] as? Int ?? 0)
            XCTAssertEqual(preview["added_material_count"] as? Int, 0)
        }
        let report: [String: Any] = ["dataset_path": dataset.path, "folder": sources.path,
                                   "seconds": timings, "duplicate_source_sets": duplicates, "training_size": 2048]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-existing-dataset-quick-scan-benchmark.json"))
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), indexBefore)
        XCTAssertTrue(duplicates.allSatisfy { $0 >= 400 })
        XCTAssertLessThan(timings.max() ?? .infinity, 1, "Preview against a populated dataset must complete in under one second")
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-quick-scan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
    private func image(channels: Int, bits: Int, size: Int = 256) throws -> Data {
        try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels,
                                    color: channels == 1 ? 0 : 2, interlace: 0),
                      pixels: Data(repeating: 91, count: size * size * channels * bits / 8), colorChunks: []).encoded()
    }
    private func checksum(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private func object(_ url: URL) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]) }
    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let output = try await NativeMaterialDatasetService.run(arguments: arguments + ["--skip-source-recovery"])
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(output).utf8)) as? [String: Any])
    }
}

private actor QuickScanNetworkProbe {
    private var requests = 0
    func record() { requests += 1 }
    func count() -> Int { requests }
}
