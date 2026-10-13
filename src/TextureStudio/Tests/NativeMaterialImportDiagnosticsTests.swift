import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialImportDiagnosticsTests: XCTestCase {
    func testDistinctFoldersWithTheSameNormalizedFamilyBothRemainDiscoverable() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await create(dataset)
        for name in ["Pine Bark", "pine_bark"] {
            let folder = dataset.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            _ = try writeCompleteFamily(name, folder: folder, width: 256, height: 256)
        }
        let info = try await inspect(dataset), records = try sourceRecords(dataset)
        XCTAssertEqual(info["source_set_count"] as? Int, 2)
        XCTAssertEqual(info["warnings"] as? [String], [])
        XCTAssertEqual(Set(records.compactMap { $0["material_id"] as? String }).count, 2)
        XCTAssertEqual(Set(records.compactMap { $0["source_family_id"] as? String }), ["pine_bark"])
        XCTAssertEqual(Set(records.compactMap { $0["source_directory"] as? String }).count, 2)
    }

    func testNamedAmbientOcclusionDoesNotHideStandaloneBaseColorAndHeightMaps() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Carpet")
        _ = try await create(dataset)
        let input = try writeMap("BaseColor.png", folder: dataset, width: 256, height: 256, bits: 8, channels: 3)
        _ = try writeMap("Height.png", folder: dataset, width: 256, height: 256, bits: 16, channels: 1)
        let ancillary = dataset.appendingPathComponent("carpet_ao_2k.png")
        try Data([1, 2, 3]).write(to: ancillary)
        let info = try await inspect(dataset), record = try XCTUnwrap(try sourceRecords(dataset).first)
        XCTAssertEqual(info["source_set_count"] as? Int, 1)
        XCTAssertEqual(info["warnings"] as? [String], [])
        XCTAssertEqual(record["maps"] as? [String: String], ["input": input.path, "height": dataset.appendingPathComponent("Height.png").path])
        let catalog = try XCTUnwrap(info["source_inventory"] as? [[String: Any]])
        XCTAssertEqual(catalog.first { $0["path"] as? String == ancillary.path }?["state"] as? String, "header_error")
        XCTAssertEqual(catalog.count, 3, "An unreadable supplemental image remains visible without hiding paired maps")
    }

    func testFolderInspectionReadsHeadersAndPreparationRejectsCorruptPixelChunks() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await create(dataset)
        let input = try writeMap("BaseColor.png", folder: dataset, width: 256, height: 256, bits: 8, channels: 3)
        _ = try writeMap("Height.png", folder: dataset, width: 256, height: 256, bits: 16, channels: 1)
        var corrupt = try Data(contentsOf: input); corrupt[45] ^= 1
        try corrupt.write(to: input)
        let info = try await inspect(dataset)
        XCTAssertEqual(info["source_set_count"] as? Int, 1, "Browsing inspects headers without decoding every source image")
        XCTAssertEqual((info["source_inventory"] as? [Any])?.count, 2)
        do {
            _ = try await command(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
            XCTFail("Corrupt source pixels must never produce training data")
        } catch {
            XCTAssertFalse(error is CancellationError)
        }
        XCTAssertEqual(try Data(contentsOf: input), corrupt)
    }

    func testCarpetAmbientOcclusionIsSupplementalAndBaseColorAloneSuppliesDiffuseInputs() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("CarpetSubstance001")
        _ = try await create(dataset)
        var originals: [URL: Data] = [:]
        for (offset, label) in ["1k", "2k", "4k", "8k"].enumerated() {
            for (role, bits, channels) in [("baseColor", 8, 3), ("height", 16, 1), ("normal", 8, 3), ("roughness", 8, 1), ("ambientOcclusion", 8, 1)] {
                let path = try writeMap("CarpetSubstance001_COMPILED_" + role + "_" + label + ".png", folder: dataset,
                    width: 256 + offset * 16, height: 256 + offset * 16, bits: bits, channels: channels)
                originals[path] = try Data(contentsOf: path)
            }
        }
        let info = try await inspect(dataset), records = try sourceRecords(dataset)
        XCTAssertEqual(info["source_set_count"] as? Int, 4)
        XCTAssertEqual(info["warnings"] as? [String], [])
        XCTAssertEqual((info["source_inventory"] as? [[String: Any]])?.filter { $0["role"] as? String == "ancillary" }.count, 4)
        for record in records {
            XCTAssertEqual(Set((record["maps"] as? [String: String] ?? [:]).keys), ["input", "height", "normal", "roughness"])
            let inputs = try XCTUnwrap(record["input_variants"] as? [[String: Any]])
            XCTAssertEqual(inputs.count, 1)
            XCTAssertTrue(inputs.allSatisfy { ($0["path"] as? String)?.contains("_baseColor_") == true })
        }
        let classified = try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: dataset, minimumSize: 32)
        XCTAssertEqual(classified.count, 1)
        XCTAssertTrue(classified.allSatisfy { $0.url.lastPathComponent.contains("_baseColor_") })
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
    }

    func testNearSizedMapsPairAtCommonCropRectangleWithoutJoiningDifferentExplicitTiers() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await create(dataset)
        let dimensions = ["baseColor": [264, 266], "height": [260, 262], "normal": [262, 264], "roughness": [261, 263]]
        var originals: [URL: Data] = [:]
        for label in ["1K", "2K"] {
            let offset = label == "1K" ? 0 : 16
            for (role, bits, channels) in [("baseColor", 8, 3), ("height", 16, 1), ("normal", 8, 3), ("roughness", 8, 1)] {
                let size = try XCTUnwrap(dimensions[role])
                let path = try writeMap("authored_near_" + role + "_" + label + ".png", folder: dataset,
                    width: size[0] + offset, height: size[1] + offset, bits: bits, channels: channels)
                originals[path] = try Data(contentsOf: path)
            }
        }
        let info = try await inspect(dataset), records = try sourceRecords(dataset)
        XCTAssertEqual(info["source_set_count"] as? Int, 2)
        XCTAssertEqual(info["warnings"] as? [String], [])
        XCTAssertEqual(Set(records.compactMap { $0["source_resolution_label"] as? String }), ["1k", "2k"])
        for record in records {
            let offset = record["source_resolution_label"] as? String == "1k" ? 0 : 16
            XCTAssertEqual(record["source_pixel_dimensions"] as? [Int], [260 + offset, 262 + offset])
            let maps = try XCTUnwrap(record["map_metadata"] as? [String: [String: Any]])
            XCTAssertEqual((maps["input"]?["source"] as? [String: Any])?["width"] as? Int, 264 + offset)
        }
        let prepared = try await command(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(samples.count, 2)
        for sample in samples {
            let rectangle = try XCTUnwrap(sample.cropRectangle)
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.inputURL)).pixels,
                try NativePNG.crop(try XCTUnwrap(sample.augmentationSourceInputURL), rectangle: rectangle).pixels)
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.targetURL)).pixels,
                try NativePNG.crop(try XCTUnwrap(sample.augmentationSourceTargetURL), rectangle: rectangle).pixels)
        }
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
    }

    func testNamingStylesRetainEveryPhysicalResolutionWithoutChangingSources() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await create(dataset)
        var originals: [URL: Data] = [:]
        for (family, labels, roles) in [
            ("FabricSubstance003_COMPILED", ["515", "1k", "2k", "4k"], ["baseColor", "height", "normal", "roughness"]),
            ("pine_bark", ["1k", "2k", "4k"], ["diff", "disp", "nor_gl", "rough"])
        ] {
            let folder = dataset.appendingPathComponent(family)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            for (offset, label) in labels.enumerated() {
                let dimension = label == "515" ? 512 : 256 + offset * 2
                for (role, bits, channels) in [(roles[0], 8, 3), (roles[1], 16, 1), (roles[2], 8, 3), (roles[3], 8, 1)] {
                    let path = try writeMap(family + "_" + role + "_" + label + ".png", folder: folder,
                        width: dimension, height: dimension, bits: bits, channels: channels)
                    originals[path] = try Data(contentsOf: path)
                }
            }
        }
        let info = try await inspect(dataset), records = try sourceRecords(dataset)
        XCTAssertEqual(info["source_set_count"] as? Int, 7)
        XCTAssertEqual(info["warnings"] as? [String], [])
        let unusual = try XCTUnwrap(records.first { $0["material_id"] as? String == "fabricsubstance003_compiled_515" })
        XCTAssertEqual(unusual["source_pixel_dimensions"] as? [Int], [512, 512])
        for record in records {
            XCTAssertEqual(Set((record["maps"] as? [String: String] ?? [:]).keys), ["input", "height", "normal", "roughness"])
            for detail in (record["map_metadata"] as? [String: [String: Any]] ?? [:]).values {
                let source = try XCTUnwrap(detail["source"] as? [String: Any])
                XCTAssertNil(source["file_sha256"], "Folder browsing must not hash every image body")
                XCTAssertNil(source["provider"], "A filename does not verify provider provenance")
            }
        }
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
    }

    func testStandaloneAliasesKeepDirectXConventionAndReportConflictingTargets() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await create(dataset)
        let complete = dataset.appendingPathComponent("Author supplied maps"), ambiguous = dataset.appendingPathComponent("Conflicting maps")
        for folder in [complete, ambiguous] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false) }
        for (name, bits, channels) in [("dif", 8, 3), ("displ", 16, 1), ("normal_directx", 8, 3), ("roughness_map", 8, 1)] {
            _ = try writeMap(name + ".png", folder: complete, width: 260, height: 262, bits: bits, channels: channels)
        }
        for (name, bits, channels) in [("base_colour", 8, 3), ("displacement_map", 16, 1), ("height", 16, 1)] {
            _ = try writeMap(name + ".png", folder: ambiguous, width: 260, height: 262, bits: bits, channels: channels)
        }
        let info = try await inspect(dataset), record = try XCTUnwrap(try sourceRecords(dataset).first)
        XCTAssertEqual(info["source_set_count"] as? Int, 1)
        XCTAssertTrue((info["warnings"] as? [String] ?? []).contains {
            $0.contains("Conflicting maps") && $0.localizedCaseInsensitiveContains("multiple height maps need explicit selection")
        })
        let maps = try XCTUnwrap(record["map_metadata"] as? [String: [String: Any]])
        XCTAssertEqual((maps["normal"]?["source"] as? [String: Any])?["suffix"] as? String, "nor_dx")
    }

    func testGrayAndGrayAlphaFamiliesRetainNativePrecisionsAndAllAvailableTargets() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await create(dataset)
        var originals: [URL: Data] = [:]
        for bits in [8, 16] {
            for channels in [1, 2] {
                let files = try writeCompleteFamily("authored_gray_\(bits)_\(channels)", folder: dataset,
                    width: 260, height: 262, diffuseBits: bits, diffuseChannels: channels, normalChannels: channels + 2)
                for file in files { originals[file] = try Data(contentsOf: file) }
            }
        }
        let info = try await inspect(dataset), records = try sourceRecords(dataset)
        XCTAssertEqual(info["source_set_count"] as? Int, 4)
        XCTAssertEqual(info["warnings"] as? [String], [])
        for record in records {
            XCTAssertEqual(record["source_pixel_dimensions"] as? [Int], [260, 262])
            XCTAssertEqual(Set(record["available_targets"] as? [String] ?? []), ["height", "normal", "roughness"])
            for detail in (record["map_metadata"] as? [String: [String: Any]] ?? [:]).values {
                let source = try XCTUnwrap(detail["source"] as? [String: Any])
                let path = URL(fileURLWithPath: try XCTUnwrap(source["path"] as? String))
                let native = try NativePNG.decode(Data(contentsOf: path))
                XCTAssertEqual(source["channels"] as? Int, native.header.channels)
                XCTAssertEqual(source["sample_bits"] as? Int, native.header.bits)
            }
        }
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
    }

    func testRecognizedSurfaceMapsProduceOneGroupedMissingDiffuseDiagnostic() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        _ = try await create(dataset)
        _ = try writeCompleteFamily("complete_surface", folder: sources, width: 256, height: 256)
        var unmatched: [URL] = []
        for (role, bits, channels) in [("height", 16, 1), ("normal", 8, 3), ("roughness", 8, 1)] {
            unmatched.append(try writeMap("authored_surface_" + role + "_2K.png", folder: sources,
                                          width: 384, height: 386, bits: bits, channels: channels))
        }
        let preview = try await inspect(dataset)
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        let warnings = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertEqual(warnings.count, 1, "Recognized maps belong to one incomplete source set, not three unrecognized files: \(warnings)")
        let notice = try XCTUnwrap(warnings.first)
        XCTAssertTrue(notice.contains("authored_surface_2k (384 × 386)"))
        XCTAssertTrue(notice.contains("height, normal, roughness maps were recognized"))
        XCTAssertTrue(notice.contains("matching diffuse map at these pixel dimensions is missing"))
        for file in unmatched { XCTAssertTrue(notice.contains(file.path)) }
        assertNoMisleadingFormatDiagnostic(warnings)
    }

    func testRecognizedDiffuseOnlyProducesPreciseMissingSurfaceMapsDiagnostic() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        _ = try await create(dataset)
        _ = try writeCompleteFamily("complete_surface", folder: sources, width: 256, height: 256)
        let diffuse = try writeMap("authored_paper_BaseColor_1K.png", folder: sources,
                                   width: 260, height: 264, bits: 16, channels: 2)
        let preview = try await inspect(dataset)
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        let warnings = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertEqual(warnings.count, 1, "\(warnings)")
        let notice = try XCTUnwrap(warnings.first)
        XCTAssertTrue(notice.contains("authored_paper_1k (260 × 264)"))
        XCTAssertTrue(notice.contains("diffuse map was recognized"))
        XCTAssertTrue(notice.contains("matching height, normal or roughness maps at these pixel dimensions are missing"))
        XCTAssertTrue(notice.contains(diffuse.path))
        assertNoMisleadingFormatDiagnostic(warnings)
    }

    func testDifferentNativeDimensionsExplainBothUnpairedGroupsWithoutRejectingTheirFormats() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        _ = try await create(dataset)
        _ = try writeCompleteFamily("complete_surface", folder: sources, width: 256, height: 256)
        let diffuse = try writeMap("authored_mismatch_color_1K.png", folder: sources,
                                   width: 272, height: 274, bits: 8, channels: 1)
        let height = try writeMap("authored_mismatch_height_1K.png", folder: sources,
                                  width: 260, height: 262, bits: 16, channels: 1)
        let preview = try await inspect(dataset)
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        let warnings = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertEqual(warnings.count, 2, "\(warnings)")
        XCTAssertTrue(warnings.contains { $0.contains("260 × 262") && $0.contains(height.path) && $0.contains("matching diffuse map") })
        XCTAssertTrue(warnings.contains { $0.contains("272 × 274") && $0.contains(diffuse.path) && $0.contains("matching height, normal or roughness") })
        assertNoMisleadingFormatDiagnostic(warnings)
    }

    func testGrayNormalIsRejectedWhileGrayDiffuseRemainsARecognizedValidSource() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        _ = try await create(dataset)
        _ = try writeCompleteFamily("complete_surface", folder: sources, width: 256, height: 256)
        _ = try writeMap("authored_bad_normal_color_1K.png", folder: sources,
                         width: 260, height: 262, bits: 8, channels: 1)
        _ = try writeMap("authored_bad_normal_normal_1K.png", folder: sources,
                         width: 260, height: 262, bits: 8, channels: 1)
        let preview = try await inspect(dataset)
        XCTAssertEqual(preview["source_set_count"] as? Int, 1, "A scalar normal cannot masquerade as RGB vector data")
        let warnings = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertTrue(warnings.contains { $0.contains("Skipped authored_bad_normal_1k") && $0.contains("normal maps need RGB or RGBA channels") })
        XCTAssertFalse(warnings.contains { $0.contains("matching diffuse map") }, "A valid grayscale diffuse was found")
    }

    private func create(_ dataset: URL) async throws -> [String: Any] {
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--source-folder", "true"])
        return try await command(["edit-dataset", "--dataset", dataset.path,
            "--training-size", "256", "--subject-validation", "disabled"])
    }
    private func inspect(_ dataset: URL) async throws -> [String: Any] {
        try await command(["dataset", "--dataset", dataset.path, "--review-size", "256"])
    }
    private func sourceRecords(_ dataset: URL) throws -> [[String: Any]] {
        try NativeMaterialDatasetService.sourceFolderRecords(dataset).map(\.sample)
    }
    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let output = try await NativeMaterialDatasetService.run(arguments: arguments + ["--skip-source-recovery"])
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(output).utf8)) as? [String: Any])
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-folder-diagnostics-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
    private func assertNoMisleadingFormatDiagnostic(_ warnings: [String], file: StaticString = #filePath, line: UInt = #line) {
        for warning in warnings {
            XCTAssertFalse(warning.contains("No recognized paired"), file: file, line: line)
            XCTAssertFalse(warning.contains("Assign its maps with Add Materials"), file: file, line: line)
            XCTAssertFalse(warning.contains("Invalid native material"), file: file, line: line)
            XCTAssertFalse(warning.contains("channels"), file: file, line: line)
        }
    }

    private func writeCompleteFamily(_ family: String, folder: URL, width: Int, height: Int,
                                     diffuseBits: Int = 8, diffuseChannels: Int = 3, normalChannels: Int = 3) throws -> [URL] {
        try [("color", diffuseBits, diffuseChannels), ("height", 16, 1), ("normal", 16, normalChannels), ("roughness", 8, 1)].map {
            try writeMap(family + "_" + $0.0 + "_1K.png", folder: folder, width: width, height: height, bits: $0.1, channels: $0.2)
        }
    }
    private func writeMap(_ name: String, folder: URL, width: Int, height: Int, bits: Int, channels: Int) throws -> URL {
        let path = folder.appendingPathComponent(name)
        let native = NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels,
                                             color: [1: UInt8(0), 2: 4, 3: 2, 4: 6][channels]!, interlace: 0),
                               pixels: Data(repeating: 93, count: width * height * channels * bits / 8), colorChunks: [])
        try native.encoded().write(to: path)
        return path
    }
}
