import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialImportDiagnosticsTests: XCTestCase {
    func testFabricSubstanceAndPineBarkNamingStylesImportEveryResolutionWithoutChangingSources() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await create(dataset)
        var originals: [URL: Data] = [:]
        for (family, labels, roles) in [
            ("FabricSubstance003_COMPILED", ["515", "1k", "2k", "4k"], ["baseColor", "height", "normal", "roughness"]),
            ("pine_bark", ["1k", "2k", "4k"], ["diff", "disp", "nor_gl", "rough"])
        ] {
            let folder = sources.appendingPathComponent(family)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for (offset, label) in labels.enumerated() {
                // Different actual dimensions separate sets; a misleading 515
                // filename must still pair maps by their actual 512 pixels.
                let dimension = label == "515" ? 512 : 256 + offset * 2
                for (role, bits, channels) in [(roles[0], 8, 3), (roles[1], 16, 1), (roles[2], 8, 3), (roles[3], 8, 1)] {
                    let path = try writeMap(family + "_" + role + "_" + label + ".png", folder: folder,
                                            width: dimension, height: dimension, bits: bits, channels: channels)
                    originals[path] = try Data(contentsOf: path)
                }
            }
        }
        let preview = try await scan(dataset, sources: sources, root: root)
        XCTAssertEqual(preview["source_set_count"] as? Int, 7)
        XCTAssertEqual(preview["ignored_file_count"] as? Int, 0)
        XCTAssertEqual(preview["warnings"] as? [String], [])
        let plan = try object(root.appendingPathComponent("preview.json")), materials = try XCTUnwrap(plan["materials"] as? [[String: Any]])
        XCTAssertEqual(materials.count, 7)
        let unusual = try XCTUnwrap(materials.first { $0["material_id"] as? String == "fabricsubstance003_compiled_515" })
        XCTAssertEqual(unusual["common_pixel_dimensions"] as? [Int], [512, 512])
        for material in materials {
            let maps = try XCTUnwrap(material["maps"] as? [String: [String: Any]])
            XCTAssertEqual(Set(maps.keys), ["input", "height", "normal", "roughness"])
            for source in maps.values {
                let path = URL(fileURLWithPath: try XCTUnwrap(source["path"] as? String))
                XCTAssertEqual(source["file_sha256"] as? String, NativeMaterialTrainer.checksum(try XCTUnwrap(originals[path])))
                XCTAssertNil(source["provider"], "A name alone must not claim verified provider provenance")
            }
        }
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path,
                                           "--plan", root.appendingPathComponent("preview.json").path,
                                           "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String),
                                           "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--skip-source-recovery"])
        XCTAssertEqual(imported["added_material_count"] as? Int, 7)
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
    }

    func testCommonStandaloneAliasesKeepDirectXConventionAndRejectConflictingTargetAliases() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await create(dataset)
        let complete = sources.appendingPathComponent("Author supplied maps"), ambiguous = sources.appendingPathComponent("Conflicting maps")
        for folder in [complete, ambiguous] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
        for (name, bits, channels) in [("dif", 8, 3), ("displ", 16, 1), ("normal_directx", 8, 3), ("roughness_map", 8, 1)] {
            _ = try writeMap(name + ".png", folder: complete, width: 260, height: 262, bits: bits, channels: channels)
        }
        _ = try writeMap("base_colour.png", folder: ambiguous, width: 260, height: 262, bits: 8, channels: 3)
        _ = try writeMap("displacement_map.png", folder: ambiguous, width: 260, height: 262, bits: 16, channels: 1)
        _ = try writeMap("height.png", folder: ambiguous, width: 260, height: 262, bits: 16, channels: 1)
        let preview = try await scan(dataset, sources: sources, root: root)
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        let warnings = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertTrue(warnings.contains { $0.contains("Conflicting maps") && $0.contains("multiple height maps need explicit selection") })
        let plan = try object(root.appendingPathComponent("preview.json")), material = try XCTUnwrap((plan["materials"] as? [[String: Any]])?.first)
        let maps = try XCTUnwrap(material["maps"] as? [String: [String: Any]])
        XCTAssertEqual(Set(maps.keys), ["input", "height", "normal", "roughness"])
        XCTAssertEqual(maps["normal"]?["suffix"] as? String, "nor_dx")
    }

    func testDeveloperReportedFabricAndBarkFoldersPreviewOfflineWithOriginalHashesPreserved() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_IMPORT_SYNONYMS_PROOF"] == "1" else {
            throw XCTSkip("Set TEXTURE_STUDIO_IMPORT_SYNONYMS_PROOF=1 to inspect the user's reported original folders without changing them.")
        }
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        var reports: [[String: Any]] = []
        for (offset, name, expected) in [(0, "FabricSubstance003", 4), (1, "Pine Bark", 3), (2, "pine_bark", 3)] {
            let source = URL(fileURLWithPath: "/opt/ipde/sources_mats/" + name), dataset = root.appendingPathComponent("Dataset-\(offset)")
            let paths = try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil).filter { $0.pathExtension.lowercased() == "png" }
            let hashes = try Dictionary(uniqueKeysWithValues: paths.map { ($0, NativeMaterialTrainer.checksum(try Data(contentsOf: $0))) })
            _ = try await create(dataset)
            let preview = try await scan(dataset, sources: source, root: root)
            XCTAssertEqual(preview["source_set_count"] as? Int, expected, source.path)
            XCTAssertEqual(preview["warnings"] as? [String], [], source.path)
            XCTAssertEqual(preview["ignored_file_count"] as? Int, 0, source.path)
            let plan = try object(root.appendingPathComponent("preview.json")), materials = try XCTUnwrap(plan["materials"] as? [[String: Any]])
            for material in materials { XCTAssertEqual(Set((material["maps"] as? [String: [String: Any]] ?? [:]).keys), ["input", "height", "normal", "roughness"]) }
            for (path, hash) in hashes { XCTAssertEqual(NativeMaterialTrainer.checksum(try Data(contentsOf: path)), hash, path.path) }
            reports.append(["folder": source.path, "source_set_count": expected, "original_sha256_checked_count": hashes.count,
                            "warnings": preview["warnings"] ?? [], "sets": materials.map { ["material_id": $0["material_id"] ?? "", "dimensions": $0["common_pixel_dimensions"] ?? []] }])
        }
        try JSONSerialization.data(withJSONObject: ["offline_preview": true, "source_bytes_unchanged": true, "folders": reports], options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-import-synonyms-proof.json"))
    }

    func testRecognizedSurfaceMapsProduceOneGroupedMissingDiffuseDiagnostic() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await create(dataset)
        _ = try writeCompleteFamily("complete_surface", folder: sources, width: 256, height: 256)
        var unmatched: [URL] = []
        for (role, bits, channels) in [("height", 16, 1), ("normal", 8, 3), ("roughness", 8, 1)] {
            unmatched.append(try writeMap("authored_surface_" + role + "_2K.png", folder: sources,
                                          width: 384, height: 386, bits: bits, channels: channels))
        }
        let preview = try await scan(dataset, sources: sources, root: root)
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
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await create(dataset)
        _ = try writeCompleteFamily("complete_surface", folder: sources, width: 256, height: 256)
        let diffuse = try writeMap("authored_paper_BaseColor_1K.png", folder: sources,
                                   width: 260, height: 264, bits: 16, channels: 2)
        let preview = try await scan(dataset, sources: sources, root: root)
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
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await create(dataset)
        _ = try writeCompleteFamily("complete_surface", folder: sources, width: 256, height: 256)
        let diffuse = try writeMap("authored_mismatch_color_1K.png", folder: sources,
                                   width: 264, height: 266, bits: 8, channels: 1)
        let height = try writeMap("authored_mismatch_height_1K.png", folder: sources,
                                  width: 260, height: 262, bits: 16, channels: 1)
        let preview = try await scan(dataset, sources: sources, root: root)
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        let warnings = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertEqual(warnings.count, 2, "\(warnings)")
        XCTAssertTrue(warnings.contains { $0.contains("260 × 262") && $0.contains(height.path) && $0.contains("matching diffuse map") })
        XCTAssertTrue(warnings.contains { $0.contains("264 × 266") && $0.contains(diffuse.path) && $0.contains("matching height, normal or roughness") })
        assertNoMisleadingFormatDiagnostic(warnings)
    }

    func testCompleteNativeGrayAndGrayAlphaFamiliesScanAndImportAllOriginalMapsWithoutWarnings() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await create(dataset)
        var originals: [URL: Data] = [:]
        for bits in [8, 16] {
            for channels in [1, 2] {
                let files = try writeCompleteFamily("authored_gray_\(bits)_\(channels)", folder: sources,
                                                    width: 260, height: 262, diffuseBits: bits, diffuseChannels: channels,
                                                    normalChannels: channels + 2)
                for file in files { originals[file] = try Data(contentsOf: file) }
            }
        }
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await scan(dataset, sources: sources, root: root)
        XCTAssertEqual(preview["source_set_count"] as? Int, 4)
        XCTAssertEqual(preview["added_material_count"] as? Int, 4)
        XCTAssertEqual(preview["warnings"] as? [String], [])
        let plan = try object(planURL), materials = try XCTUnwrap(plan["materials"] as? [[String: Any]])
        XCTAssertEqual(materials.count, 4)
        for material in materials {
            XCTAssertEqual(material["common_pixel_dimensions"] as? [Int], [260, 262])
            let maps = try XCTUnwrap(material["maps"] as? [String: [String: Any]])
            XCTAssertEqual(Set(maps.keys), ["input", "height", "normal", "roughness"])
            for source in maps.values {
                let path = URL(fileURLWithPath: try XCTUnwrap(source["path"] as? String))
                let native = try NativePNG.decode(try Data(contentsOf: path))
                XCTAssertEqual(source["channels"] as? Int, native.header.channels)
                XCTAssertEqual(source["sample_bits"] as? Int, native.header.bits)
                XCTAssertEqual(source["file_sha256"] as? String, NativeMaterialTrainer.checksum(try Data(contentsOf: path)))
            }
        }
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path,
                                           "--plan", planURL.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String),
                                           "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--skip-source-recovery"])
        XCTAssertEqual(imported["source_set_count"] as? Int, 4)
        XCTAssertEqual(imported["added_material_count"] as? Int, 4)
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes, "Import preserves native source files") }
        let index = try object(dataset.appendingPathComponent("dataset.json"))
        let entries = try XCTUnwrap(index["samples"] as? [[String: Any]])
        XCTAssertEqual(entries.count, 4)
        for entry in entries {
            let sample = try object(dataset.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json"))
            XCTAssertEqual(Set(sample["available_targets"] as? [String] ?? []), ["height", "normal", "roughness"])
            XCTAssertEqual(sample["source_pixel_dimensions"] as? [Int], [260, 262])
        }
    }

    func testGrayNormalIsRejectedWhileGrayDiffuseRemainsARecognizedValidSource() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await create(dataset)
        _ = try writeCompleteFamily("complete_surface", folder: sources, width: 256, height: 256)
        _ = try writeMap("authored_bad_normal_color_1K.png", folder: sources,
                         width: 260, height: 262, bits: 8, channels: 1)
        _ = try writeMap("authored_bad_normal_normal_1K.png", folder: sources,
                         width: 260, height: 262, bits: 8, channels: 1)
        let preview = try await scan(dataset, sources: sources, root: root)
        XCTAssertEqual(preview["source_set_count"] as? Int, 1, "A scalar normal cannot masquerade as RGB vector data")
        let warnings = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertTrue(warnings.contains { $0.contains("Skipped authored_bad_normal_1k") && $0.contains("normal maps need RGB or RGBA channels") })
        XCTAssertFalse(warnings.contains { $0.contains("matching diffuse map") }, "A valid grayscale diffuse was found")
    }

    func testSkipSourceRecoveryPreviewsAndImportsLocalProviderSetWithoutTierDownloads() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let probe = NetworkProbe()
        let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            await probe.record(url); throw ProbeError.unexpectedRequest
        }, download: { url, _ in
            await probe.record(url); throw ProbeError.unexpectedRequest
        }))
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Local provider sources",
                               "--training-size", "1024", "--subject-validation", "disabled", "--skip-source-recovery"], recovery: recovery)
        let diffuse = try writeMap("Paper003_1K-PNG_Color.png", folder: sources,
                                   width: 1024, height: 1024, bits: 8, channels: 1)
        let height = try writeMap("Paper003_1K-PNG_Displacement.png", folder: sources,
                                  width: 1024, height: 1024, bits: 16, channels: 1)
        let originalDiffuse = try Data(contentsOf: diffuse), originalHeight = try Data(contentsOf: height)
        let preview = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                         "--plan", root.appendingPathComponent("preview.json").path,
                                         "--size", "1024", "--target", "height", "--skip-source-recovery"], recovery: recovery)
        XCTAssertEqual(preview["source_set_count"] as? Int, 1)
        XCTAssertEqual(preview["warnings"] as? [String], [])
        let previewRequests = await probe.urls()
        XCTAssertEqual(previewRequests, [], "Skipping source recovery applies to import preview as well as dataset reads")
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path,
                                           "--target", "height", "--training-size", "1024", "--skip-source-recovery"], recovery: recovery)
        XCTAssertEqual(imported["source_set_count"] as? Int, 1)
        XCTAssertEqual(imported["added_material_count"] as? Int, 1)
        let importRequests = await probe.urls()
        XCTAssertEqual(importRequests, [], "A direct local import must honor the same explicit skip flag")
        XCTAssertEqual(try Data(contentsOf: diffuse), originalDiffuse)
        XCTAssertEqual(try Data(contentsOf: height), originalHeight)
        XCTAssertFalse(FileManager.default.fileExists(atPath: sources.appendingPathComponent("Paper003_2K-PNG_Color.png").path))
    }

    private func assertNoMisleadingFormatDiagnostic(_ warnings: [String], file: StaticString = #filePath, line: UInt = #line) {
        for warning in warnings {
            XCTAssertFalse(warning.contains("No recognized paired"), file: file, line: line)
            XCTAssertFalse(warning.contains("Assign its maps with Add Materials"), file: file, line: line)
            XCTAssertFalse(warning.contains("Invalid native material"), file: file, line: line)
            XCTAssertFalse(warning.contains("channels"), file: file, line: line)
        }
    }

    private func create(_ dataset: URL) async throws -> [String: Any] {
        try await command(["create-dataset", "--dataset", dataset.path, "--name", "Import diagnostics",
                           "--training-size", "256", "--subject-validation", "disabled", "--skip-source-recovery"])
    }
    private func scan(_ dataset: URL, sources: URL, root: URL) async throws -> [String: Any] {
        try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                           "--plan", root.appendingPathComponent("preview.json").path, "--skip-source-recovery"])
    }
    private func command(_ arguments: [String], recovery: NativeMaterialSourceRecovery = NativeMaterialSourceRecovery()) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments, sourceRecovery: recovery)
        let text = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-import-diagnostics-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
    private func object(_ url: URL) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
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
    private actor NetworkProbe {
        private var requestedURLs: [String] = []
        func record(_ url: URL) { requestedURLs.append(url.absoluteString) }
        func urls() -> [String] { requestedURLs }
    }
    private enum ProbeError: Error { case unexpectedRequest }
}
