import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialCropSizingTests: XCTestCase {
    func testTwoKReviewOmitsOneKOriginalsAndSwitchingGridRestoresTheirExactCrops() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset.appendingPathComponent("Anti Skid Tiles")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: true)
        _ = try await command(["edit-dataset", "--dataset", dataset.path, "--training-size", "2048", "--validation-enabled", "no"])
        var originals: [URL: Data] = [:]
        for size in [1024, 2048] {
            for (role, bits, channels) in [("diff", 8, 3), ("disp", 16, 1)] {
                let path = sources.appendingPathComponent("anti_skid_tiles_\(role)_\(size / 1024)k.png")
                let bytes = try png(width: size, height: size, bits: bits, channels: channels)
                try bytes.write(to: path); originals[path] = bytes
            }
        }
        let imported = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(imported["source_set_count"] as? Int, 2, "Both original resolutions remain registered")
        let snapshot = try metadataSnapshot(dataset)
        for _ in 0..<2 {
            let info = try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: dataset, size: 2048)
            let rows = samples(info)
            XCTAssertEqual(rows.count, 1)
            XCTAssertEqual(rows.first?["sample_id"] as? String, "anti_skid_tiles_2k_full")
            XCTAssertTrue(rows.allSatisfy { $0["width"] as? Int == 2048 && $0["height"] as? Int == 2048 })
        }
        let smallGrid = try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: dataset, size: 1024)
        XCTAssertEqual(samples(smallGrid).count, 2, "Changing the grid rebuilds eligible crops from retained source references")
        XCTAssertTrue(samples(smallGrid).allSatisfy { $0["width"] as? Int == 1024 && $0["height"] as? Int == 1024 })
        let smallerPreparedInfo = try await command(["prepare-size", "--dataset", dataset.path, "--size", "1024", "--target", "height"])
        XCTAssertEqual(samples(smallerPreparedInfo).count, 2, "A prepared dataset uses its own exact grid even when the original dataset selects 2048")
        XCTAssertTrue(samples(smallerPreparedInfo).allSatisfy { $0["width"] as? Int == 1024 && $0["height"] as? Int == 1024 })
        XCTAssertEqual(smallerPreparedInfo["training_size"] as? Int, 1024)
        XCTAssertEqual(smallerPreparedInfo["review_size"] as? Int, 1024)
        let preparedDataset = try decode(smallerPreparedInfo)
        XCTAssertTrue(preparedDataset.readyForTraining(size: 1024, material: nil, target: "height"))
        XCTAssertFalse(preparedDataset.readyForTraining(size: 2048, material: nil, target: "height"))
        let suite = "NativeMaterialCropSizingTests-" + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite)); defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.appendingPathComponent("Workspace").path, forKey: "workspace")
        let store = WorkbenchStore(preferences: defaults)
        store.training.size = 2048
        store.adoptDataset(preparedDataset)
        XCTAssertEqual(store.training.size, 1024, "Adopting prepared crops must use their actual grid")
        var wrongGridMetadata = smallerPreparedInfo
        wrongGridMetadata["training_size"] = 2048
        XCTAssertFalse(try decode(wrongGridMetadata).readyForTraining(size: 1024, material: nil, target: "height"))
        wrongGridMetadata = smallerPreparedInfo
        var wrongPreparation = try XCTUnwrap(wrongGridMetadata["preparation"] as? [String: Any])
        wrongPreparation["crop_size"] = 2048; wrongGridMetadata["preparation"] = wrongPreparation
        XCTAssertFalse(try decode(wrongGridMetadata).readyForTraining(size: 1024, material: nil, target: "height"))
        var wrongVariant = smallerPreparedInfo
        var materials = try XCTUnwrap(wrongVariant["materials"] as? [[String: Any]])
        var rows = try XCTUnwrap(materials[0]["samples"] as? [[String: Any]])
        rows[0]["input_variants"] = [["path": "/wrong-grid.png", "width": 512, "height": 512]]
        materials[0]["samples"] = rows; wrongVariant["materials"] = materials
        XCTAssertFalse(try decode(wrongVariant).readyForTraining(size: 1024, material: nil, target: "height"), "An undersized alternate diffuse must not bypass map sizing")
        let preparedInfo = try await command(["prepare-size", "--dataset", dataset.path, "--size", "2048", "--target", "height"])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(preparedInfo["dataset_path"] as? String))
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 2048, target: "height")
        XCTAssertEqual(descriptors.map(\.id), ["anti_skid_tiles_2k_full"])
        XCTAssertEqual(try metadataSnapshot(dataset), snapshot, "Projection and preparation preserve original registration")
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
    }

    func testManualNearSizedMapsUsePerMapExactCropWhenCommonIntersectionEqualsGrid() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: false)
        _ = try await command(["edit-dataset", "--dataset", dataset.path, "--training-size", "256", "--validation-enabled", "no"])
        let fixtures = [("input", 264, 262, 8, 3), ("height", 256, 256, 16, 1),
                        ("roughness", 260, 258, 8, 1), ("normal", 258, 264, 8, 3)]
        var paths: [String: URL] = [:], originals: [URL: Data] = [:]
        for (role, width, height, bits, channels) in fixtures {
            let path = dataset.appendingPathComponent(role + ".png"), bytes = try png(width: width, height: height, bits: bits, channels: channels, varying: true)
            try bytes.write(to: path); paths[role] = path; originals[path] = bytes
        }
        let arguments = paths.keys.sorted().flatMap { ["--" + $0, paths[$0]!.path] }
        let added = try await command(["add-material", "--dataset", dataset.path, "--name", "Near size"] + arguments)
        XCTAssertEqual(samples(added).first?["width"] as? Int, 256)
        let snapshot = try metadataSnapshot(dataset)
        XCTAssertThrowsError(try NativeMaterialDatasetService.trainingSamples(datasetURL: dataset, size: 256, target: "height"),
                             "The common intersection cannot pretend a larger original reference is already a tensor of the selected size")
        for target in ["height", "roughness", "normal"] {
            let preparedInfo = try await command(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target])
            let prepared = URL(fileURLWithPath: try XCTUnwrap(preparedInfo["dataset_path"] as? String))
            let sample = try XCTUnwrap(NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: target).first)
            XCTAssertEqual(sample.cropRectangle, [0, 0, 256, 256])
            for (preparedPath, originalPath) in [(sample.inputURL, paths["input"]!), (sample.targetURL, paths[target]!)] {
                let actual = try NativePNG.decode(Data(contentsOf: preparedPath))
                let expected = try NativePNG.decode(try XCTUnwrap(originals[originalPath])).crop([0, 0, 256, 256])
                XCTAssertEqual(actual.header.width, 256); XCTAssertEqual(actual.header.height, 256)
                XCTAssertEqual(actual.pixels, expected.pixels, "Every map crops its own original native integer codes")
            }
            XCTAssertEqual(sample.targetURL == paths[target], target == "height", "Only an original already equal to the selected rectangle can be referenced directly")
            let preparedIndex = try object(prepared.appendingPathComponent("dataset.json"))
            XCTAssertEqual((preparedIndex["native_size_preparation"] as? [String: Any])?["target_cropped"] as? Bool, true)
        }
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(samples(reopened).first?["width"] as? Int, 256)
        XCTAssertEqual(try metadataSnapshot(dataset), snapshot)
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
    }

    func testCommonPixelsRejectWideMismatchesAndDoNotInflateAnUndersizedSource() throws {
        XCTAssertEqual(try NativeMaterialDatasetService.commonNativeDimensions([
            ["width": 2048, "height": 2050], ["width": 2056, "height": 2048]
        ]), [2048, 2048])
        XCTAssertEqual(try NativeMaterialDatasetService.commonNativeDimensions([
            ["width": 2048, "height": 2048], ["width": 2047, "height": 2048]
        ]), [2047, 2048], "A missing pixel cannot be scaled or padded to make a 2048 crop")
        XCTAssertTrue(NativeMaterialDatasetService.cropLayout([2047, 2048], size: 2048).isEmpty)
        XCTAssertThrowsError(try NativeMaterialDatasetService.commonNativeDimensions([
            ["width": 2048, "height": 2048], ["width": 2057, "height": 2048]
        ]))
        XCTAssertThrowsError(try NativeMaterialDatasetService.commonNativeDimensions([
            ["width": 2048, "height": 2048], ["width": 1024, "height": 1024]
        ]))
    }

    private func samples(_ info: [String: Any]) -> [[String: Any]] {
        (info["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
    }
    private func decode(_ info: [String: Any]) throws -> WorkbenchDataset {
        try WorkbenchResult.decode(WorkbenchDataset.self, output: String(decoding: JSONSerialization.data(withJSONObject: info), as: UTF8.self))
    }
    private func metadataSnapshot(_ root: URL) throws -> [String: String] {
        let files = try XCTUnwrap(FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]))
        var snapshot: [String: String] = [:]
        for case let file as URL in files where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            snapshot[file.path] = SHA256.hash(data: try Data(contentsOf: file)).map { String(format: "%02x", $0) }.joined()
        }
        return snapshot
    }
    private func png(width: Int, height: Int, bits: Int, channels: Int, varying: Bool = false) throws -> Data {
        let count = width * height * channels * bits / 8
        let pixels = varying ? Data((0..<count).map { UInt8(truncatingIfNeeded: $0 * 97) }) : Data(repeating: 73, count: count)
        return try NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels,
                                          color: channels == 1 ? 0 : 2, interlace: 0), pixels: pixels, colorChunks: []).encoded()
    }
    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments + ["--skip-source-recovery"])
        let text = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
    private func object(_ url: URL) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-crop-sizing-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
}
