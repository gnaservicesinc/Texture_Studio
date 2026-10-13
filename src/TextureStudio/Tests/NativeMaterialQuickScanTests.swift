import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialQuickScanTests: XCTestCase {
    func testFolderPreviewReadsCurrentMembershipWithoutCreatingAnIndex() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources"), outside = root.appendingPathComponent("Outside")
        for folder in [sources, outside] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false) }
        let material = sources.appendingPathComponent("Brick")
        try FileManager.default.createDirectory(at: material, withIntermediateDirectories: false)
        let rgb = try image(channels: 3, bits: 8), scalar = try image(channels: 1, bits: 16)
        try rgb.write(to: material.appendingPathComponent("BaseColor.png")); try scalar.write(to: material.appendingPathComponent("Height.png"))
        let first = try await command(["dataset", "--dataset", sources.path, "--review-size", "256"])
        let second = try await command(["dataset", "--dataset", sources.path, "--review-size", "256"])
        XCTAssertEqual(first["source_set_count"] as? Int, 1)
        XCTAssertEqual(first["index_sha256"] as? String, second["index_sha256"] as? String)
        XCTAssertFalse(FileManager.default.fileExists(atPath: sources.appendingPathComponent("dataset.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: sources.appendingPathComponent("samples").path))
        try FileManager.default.moveItem(at: material, to: outside.appendingPathComponent("Brick"))
        let moved = try await command(["dataset", "--dataset", sources.path, "--review-size", "256"])
        XCTAssertEqual(moved["source_set_count"] as? Int, 0, "A folder moved out of the dataset immediately leaves it")
        XCTAssertEqual(try Data(contentsOf: outside.appendingPathComponent("Brick/BaseColor.png")), rgb)
        XCTAssertEqual(try Data(contentsOf: outside.appendingPathComponent("Brick/Height.png")), scalar)
    }

    func testFolderPreviewDoesNotFetchMissingProviderTiers() async throws {
        let sources = try temporary(); defer { try? FileManager.default.removeItem(at: sources) }
        try image(channels: 3, bits: 8, size: 1024).write(to: sources.appendingPathComponent("brick_diff_1k.png"))
        try image(channels: 1, bits: 16, size: 1024).write(to: sources.appendingPathComponent("brick_disp_1k.png"))
        let probe = QuickScanNetworkProbe()
        let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { _ in
            await probe.record(); throw StudioError("Folder preview must stay local")
        }, download: { _, _ in
            await probe.record(); throw StudioError("Folder preview must stay local")
        }))
        let text = try await NativeMaterialDatasetService.run(arguments: ["dataset", "--dataset", sources.path,
            "--review-size", "2048"], sourceRecovery: recovery)
        let preview = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(text).utf8)) as? [String: Any])
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        XCTAssertEqual((preview["materials"] as? [Any])?.count, 0, "Small originals remain inspectable without pretending to be 2048 crops")
        let requests = await probe.count()
        XCTAssertEqual(requests, 0)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: sources.path).count, 2)
    }

    func testFolderPreviewReadsHeadersAndPreparationVerifiesFullSourceBytes() async throws {
        let sources = try temporary(); defer { try? FileManager.default.removeItem(at: sources) }
        let rgb = try image(channels: 3, bits: 8), scalar = try image(channels: 1, bits: 16)
        let input = sources.appendingPathComponent("Carpet_COMPILED_baseColor_1k.png")
        let height = sources.appendingPathComponent("Carpet_COMPILED_height_1k.png")
        try rgb.write(to: input); try scalar.write(to: height)
        let preview = try await command(["dataset", "--dataset", sources.path, "--review-size", "256"])
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        let inventory = try XCTUnwrap(preview["source_inventory"] as? [[String: Any]])
        XCTAssertTrue(inventory.allSatisfy { $0["source_verification"] as? String == "header_inspected" })
        let records = try NativeMaterialDatasetService.sourceFolderRecords(sources)
        let maps = try XCTUnwrap(records.first?.sample["map_metadata"] as? [String: [String: Any]])
        XCTAssertNil(maps["input"]?["sample_sha256"] as? String, "Opening a folder does not hash compressed pixels")
        let result = try await command(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "height"])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        defer { _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        let sample = try XCTUnwrap(NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").first)
        XCTAssertEqual(sample.inputSHA256, checksum(rgb)); XCTAssertEqual(sample.targetSHA256, checksum(scalar))
        XCTAssertEqual(NativeSourceFileTags.read(input)?.verifiedSHA256, checksum(rgb))
        XCTAssertEqual(NativeSourceFileTags.read(height)?.verifiedSHA256, checksum(scalar))
        XCTAssertEqual(try Data(contentsOf: input), rgb); XCTAssertEqual(try Data(contentsOf: height), scalar)
        XCTAssertFalse(FileManager.default.fileExists(atPath: sources.appendingPathComponent("dataset.json").path))
    }

    func testHeaderPreviewCannotAuthorizeCorruptPixelChunksForTraining() async throws {
        let sources = try temporary(); defer { try? FileManager.default.removeItem(at: sources) }
        var rgb = try image(channels: 3, bits: 8); rgb[45] ^= 1
        let input = sources.appendingPathComponent("brick_diff_1k.png")
        try rgb.write(to: input)
        try image(channels: 1, bits: 16).write(to: sources.appendingPathComponent("brick_disp_1k.png"))
        let preview = try await command(["dataset", "--dataset", sources.path, "--review-size", "256"])
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        do {
            _ = try await command(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "height"])
            XCTFail("Header-only inspection cannot authorize corrupted compressed pixels")
        } catch { XCTAssertFalse(error is CancellationError) }
        XCTAssertEqual(try Data(contentsOf: input), rgb)
        XCTAssertNil(NativeSourceFileTags.read(input)?.verifiedSHA256)
        let cache = try NativeMaterialDatasetService.trainingCacheRoot(datasetURL: sources)
        XCTAssertEqual((try? FileManager.default.contentsOfDirectory(atPath: cache.path)) ?? [], [])
    }

    func testDeveloperCorpusFolderPreviewTiming() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_QUICK_SCAN_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in timing of the developer source corpus without source writes or downloads.")
        }
        let sources = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        var timings: [Double] = [], setCounts: [Int] = []
        for _ in 0..<3 {
            let started = ProcessInfo.processInfo.systemUptime
            let preview = try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: sources, size: 1024)
            timings.append(ProcessInfo.processInfo.systemUptime - started)
            setCounts.append(preview["source_set_count"] as? Int ?? 0)
        }
        let report: [String: Any] = ["folder": sources.path, "seconds": timings, "source_sets": setCounts, "training_size": 1024]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-folder-preview-benchmark.json"))
        XCTAssertTrue(setCounts.allSatisfy { $0 >= 400 })
        XCTAssertLessThan(timings.max() ?? .infinity, 1, "Full source-folder preview must complete in under one second")
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-quick-scan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
    private func image(channels: Int, bits: Int, size: Int = 256) throws -> Data {
        try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels,
            color: channels == 1 ? 0 : 2, interlace: 0), pixels: Data(repeating: 91,
            count: size * size * channels * bits / 8), colorChunks: []).encoded()
    }
    private func checksum(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let output = try await NativeMaterialDatasetService.run(arguments: arguments)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(output).utf8)) as? [String: Any])
    }
}

private actor QuickScanNetworkProbe {
    private var requests = 0
    func record() { requests += 1 }
    func count() -> Int { requests }
}
