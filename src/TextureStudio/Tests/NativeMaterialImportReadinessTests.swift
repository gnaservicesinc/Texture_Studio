import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

/// Import readiness is checked through the same preview, registration and crop
/// commands used by Dataset Studio, with a transport that rejects every request.
@MainActor
final class NativeMaterialImportReadinessTests: XCTestCase {
    func testCompleteLocalProviderAndAuthoredSetsImportWithoutNetworkAndPreserveNativeMaps() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        let probe = ImportReadinessNetworkProbe(), recovery = rejectingRecovery(probe)
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Local originals", "--training-size", "1024", "--subject-validation", "disabled"], recovery: recovery)
        var originals: [URL: Data] = [:]
        for (number, bits, channels, ambient) in [(1, 8, 1, true), (2, 8, 2, false), (3, 16, 1, true), (4, 16, 2, false)] {
            let asset = ambient ? String(format: "Paper%03d", number) : "gray_surface_\(number)"
            let folder = sources.appendingPathComponent(asset)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for scale in [1, 2, 4] {
                for (role, suffix, precision, count) in [("input", ambient ? "Color" : "diff", bits, channels), ("height", ambient ? "Displacement" : "disp", 16, 1), ("normal", ambient ? "NormalGL" : "nor_gl", 16, 3), ("roughness", ambient ? "Roughness" : "rough", 8, 1)] {
                    let filename = ambient ? "\(asset)_\(scale)K-PNG_\(suffix).png" : "\(asset)_\(suffix)_\(scale)k.png"
                    let path = folder.appendingPathComponent(filename)
                    let bytes = try image(width: scale * 1024, height: scale * 1024, bits: precision, channels: count, role: role).encoded()
                    try bytes.write(to: path); originals[path] = bytes
                }
            }
        }
        // Authored names do not authorize a provider. Their dimension labels
        // remain descriptive; pairing uses actual dimensions and explicit roles.
        for (asset, size, suffixFirst) in [("Cardboard_01", 1030, true), ("Wood Boards", 1032, false)] {
            let folder = sources.appendingPathComponent(asset)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for (role, suffix, bits, channels) in [("input", "BaseColor", 16, 4), ("height", "Height", 16, 1), ("normal", "NormalGL", 8, 3), ("roughness", "Roughness", 8, 1)] {
                let filename = suffixFirst ? "\(asset)_\(suffix)_1K.png" : "\(asset)_1K_\(suffix).png"
                let path = folder.appendingPathComponent(filename), bytes = try image(width: size, height: size + 2, bits: bits, channels: channels, role: role).encoded()
                try bytes.write(to: path); originals[path] = bytes
            }
        }
        let plan = root.appendingPathComponent("preview.json")
        let scanned = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", plan.path, "--training-size", "1024"], recovery: recovery)
        XCTAssertEqual(scanned["source_set_count"] as? Int, 14)
        XCTAssertEqual(scanned["added_material_count"] as? Int, 14)
        XCTAssertEqual(scanned["warnings"] as? [String], [])
        XCTAssertEqual(scanned["ignored_file_count"] as? Int, 0)
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", plan.path, "--expected-plan-sha256", try XCTUnwrap(scanned["plan_sha256"] as? String), "--expected-index-sha256", try XCTUnwrap(scanned["index_sha256"] as? String), "--training-size", "1024"], recovery: recovery)
        XCTAssertEqual(imported["source_set_count"] as? Int, 14)
        XCTAssertEqual(imported["added_material_count"] as? Int, 14)
        for target in ["height", "normal", "roughness"] {
            let prepared = try await command(["prepare-size", "--dataset", dataset.path, "--size", "1024", "--target", target], recovery: recovery)
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 1024, target: target)
            XCTAssertEqual(samples.count, 14)
            for sample in samples {
                try assertExactCrop(sample.inputURL, source: XCTUnwrap(sample.augmentationSourceInputURL), rectangle: XCTUnwrap(sample.cropRectangle), originals: originals)
                try assertExactCrop(sample.targetURL, source: XCTUnwrap(sample.augmentationSourceTargetURL), rectangle: XCTUnwrap(sample.cropRectangle), originals: originals)
            }
            _ = try await command(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
        }
        let calls = await probe.calls()
        XCTAssertEqual(calls, [], "Complete native local source tiers never fetch metadata or download maps")
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes, path.path) }
    }

    /// Opt in only for the real developer corpus. This creates an isolated
    /// reference dataset. The one absent original is recovered through the
    /// native verified provider path before an entirely offline import proof.
    func testDeveloperCorpusFullFolderImportAndPaperGrayOneKCropsOffline() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_IMPORT_CORPUS_PROOF"] == "1" else {
            throw XCTSkip("Set TEXTURE_STUDIO_IMPORT_CORPUS_PROOF=1 after verified corpus recovery to exercise the real source library.")
        }
        let root = try temporary(), sources = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), plan = root.appendingPathComponent("preview.json")
        let probe = ImportReadinessNetworkProbe(), recovery = rejectingRecovery(probe)
        let pngs = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
            .compactMap { $0 as? URL }.filter { $0.pathExtension.lowercased() == "png" }
        var originalHashes: [URL: String] = [:]
        for file in pngs { originalHashes[file] = checksum(try Data(contentsOf: file)) }
        let recoveredColor = sources.appendingPathComponent("Paper003/Paper003_2K-PNG_Color.png")
        let result = try await NativeMaterialSourceRecovery().recover(.init(asset: "Paper003", resolution: "2k", target: "input", destination: recoveredColor, width: 2048, height: 2048, provider: .ambientCG, preserveOriginalDimensions: true))
        var recoveryEvidence: [String: Any] = [:]
        switch result {
        case .recovered(let provenance):
            recoveryEvidence = ["recovered": true, "destination": provenance.destination.path, "download_url": provenance.downloadURL.absoluteString, "archive_sha256": provenance.archiveSHA256 ?? "", "archive_sha1": provenance.archiveSHA1 ?? "", "archive_member": provenance.archiveMember ?? "", "source_sha256": provenance.sha256, "source_width": provenance.sourceWidth, "source_height": provenance.sourceHeight]
            XCTAssertEqual(provenance.cropRectangle, nil)
            XCTAssertEqual(provenance.sourceWidth, 2048); XCTAssertEqual(provenance.sourceHeight, 2048)
        case .alreadyPresent:
            recoveryEvidence = ["recovered": false, "already_present": true, "destination": recoveredColor.path, "source_sha256": checksum(try Data(contentsOf: recoveredColor))]
            let proofs = try FileManager.default.contentsOfDirectory(at: recoveredColor.deletingLastPathComponent(), includingPropertiesForKeys: nil).filter { $0.lastPathComponent.hasPrefix(".texture-studio-source-proof-") }
            for proof in proofs {
                guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: proof)) as? [String: Any],
                      value["source_path"] as? String == recoveredColor.path else { continue }
                recoveryEvidence["verified_provider_proof"] = value
            }
        case .failed(let reason): XCTFail("Verified native Paper003 grayscale recovery failed: " + reason); return
        case .notPublished: XCTFail("The published Paper003 2K color map must be recoverable"); return
        }
        var originals: [URL: Data] = [:]
        for file in Set(pngs + [recoveredColor]) where ["Paper003_", "Paper005_", "Cardboard_01_", "Cardboard002_"].contains(where: { file.lastPathComponent.hasPrefix($0) }) {
            originals[file] = try Data(contentsOf: file)
        }
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Full real original import proof", "--training-size", "1024", "--subject-validation", "disabled"], recovery: recovery)
        let preview = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", plan.path, "--training-size", "1024"], recovery: recovery)
        XCTAssertEqual(preview["source_set_count"] as? Int, 376)
        XCTAssertEqual(preview["added_material_count"] as? Int, 376)
        XCTAssertEqual(preview["warnings"] as? [String], [], "All paired provider and authored formats are recognized")
        guard preview["source_set_count"] as? Int == 376, preview["added_material_count"] as? Int == 376, preview["warnings"] as? [String] == [] else {
            throw StudioError("Real source preview is not ready; the user's aborted import must remain untouched")
        }
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", plan.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String), "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--training-size", "1024"], recovery: recovery)
        XCTAssertEqual(imported["source_set_count"] as? Int, 376)
        XCTAssertEqual(imported["added_material_count"] as? Int, 376)
        let materials = try XCTUnwrap(imported["materials"] as? [[String: Any]])
        XCTAssertEqual(materials.count, 376)
        let families = materials.flatMap { ($0["samples"] as? [[String: Any]] ?? []).compactMap { $0["source_family_id"] as? String } }
        XCTAssertEqual(Set(families).count, 122)
        guard materials.count == 376, Set(families).count == 122 else { throw StudioError("Real registered source library is incomplete") }
        for target in ["height", "normal", "roughness"] {
            let prepared = try await command(["prepare-size", "--dataset", dataset.path, "--size", "1024", "--target", target, "--material", "paper003_2k"], recovery: recovery)
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 1024, target: target)
            XCTAssertEqual(samples.count, 3)
            let twoK = try XCTUnwrap(samples.first { $0.augmentationSourceInputURL?.lastPathComponent == "Paper003_2K-PNG_Color.png" })
            let gray = try NativePNG.decode(Data(contentsOf: twoK.inputURL))
            XCTAssertEqual(gray.header.channels, 1); XCTAssertEqual(gray.header.bits, 8)
            guard samples.count == 3, gray.header.channels == 1, gray.header.bits == 8 else { throw StudioError("Real Paper003 crops did not retain the native grayscale format") }
            for sample in samples {
                try assertExactCrop(sample.inputURL, source: XCTUnwrap(sample.augmentationSourceInputURL), rectangle: XCTUnwrap(sample.cropRectangle), originals: originals)
                try assertExactCrop(sample.targetURL, source: XCTUnwrap(sample.augmentationSourceTargetURL), rectangle: XCTUnwrap(sample.cropRectangle), originals: originals)
            }
            _ = try await command(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
        }
        let secondPlan = root.appendingPathComponent("repeat-preview.json")
        let repeated = try await command(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", secondPlan.path, "--training-size", "1024"], recovery: recovery)
        XCTAssertEqual(repeated["source_set_count"] as? Int, 376)
        XCTAssertEqual(repeated["duplicate_material_count"] as? Int, 376)
        XCTAssertEqual(repeated["added_material_count"] as? Int, 0)
        XCTAssertEqual(repeated["warnings"] as? [String], [])
        let calls = await probe.calls(); XCTAssertEqual(calls, [])
        guard repeated["source_set_count"] as? Int == 376, repeated["duplicate_material_count"] as? Int == 376,
              repeated["added_material_count"] as? Int == 0, repeated["warnings"] as? [String] == [], calls.isEmpty else { throw StudioError("Real original-source rescan is not ready offline") }
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes, "Real source bytes remain unchanged") }
        for (file, hash) in originalHashes { guard checksum(try Data(contentsOf: file)) == hash else { throw StudioError("Real original source changed: " + file.path) } }
        let report: [String: Any] = ["source_path": sources.path, "source_set_count": 376, "family_count": 122, "network_calls": calls, "warnings": preview["warnings"] ?? [], "paper003_targets": ["height", "normal", "roughness"], "paper003_training_size": 1024, "native_source_bytes_preserved": true, "existing_original_sha256_checked_count": originalHashes.count, "recovery": recoveryEvidence]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: "/tmp/texture-import-readiness-corpus-proof.json"))
    }

    func testRepairUserAuthorizedAbortedWolfImportOnlyWhileStillEmpty() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_REPAIR_ABORTED_WOLF_IMPORT"] == "1" else {
            throw XCTSkip("Explicit developer repair of the user's aborted empty Wolf import is disabled")
        }
        let proofURL = URL(fileURLWithPath: "/tmp/texture-import-readiness-corpus-proof.json")
        let proof = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: proofURL)) as? [String: Any])
        guard proof["source_set_count"] as? Int == 376, proof["family_count"] as? Int == 122,
              proof["warnings"] as? [String] == [], proof["network_calls"] as? [String] == [], proof["native_source_bytes_preserved"] as? Bool == true else {
            throw StudioError("Full verified source-library proof is required before repairing the aborted import")
        }
        let dataset = URL(fileURLWithPath: "/Users/andrewsmith/Documents/Texture Studio Datasets/Wolf")
        let source = URL(fileURLWithPath: "/opt/ipde/sources_mats"), root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let index = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: dataset.appendingPathComponent("dataset.json"))) as? [String: Any])
        guard index["storage_policy"] as? String == "original-source-references-v1", let existingSamples = index["samples"] as? [[String: Any]], existingSamples.isEmpty else {
            throw StudioError("Wolf changed or is no longer an empty native original-reference dataset; repair must not replace user data")
        }
        let probe = ImportReadinessNetworkProbe(), recovery = rejectingRecovery(probe), plan = root.appendingPathComponent("wolf-preview.json")
        let preview = try await command(["scan-folder", "--dataset", dataset.path, "--folder", source.path, "--plan", plan.path, "--training-size", "1024"], recovery: recovery)
        guard preview["source_set_count"] as? Int == 376, preview["added_material_count"] as? Int == 376,
              preview["duplicate_material_count"] as? Int == 0, preview["warnings"] as? [String] == [] else {
            throw StudioError("Wolf import preview must include all 376 originals without errors before registration")
        }
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", source.path, "--plan", plan.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String), "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--training-size", "1024"], recovery: recovery)
        XCTAssertEqual(imported["source_set_count"] as? Int, 376); XCTAssertEqual(imported["added_material_count"] as? Int, 376)
        let calls = await probe.calls(); XCTAssertEqual(calls, [])
        let report: [String: Any] = ["dataset_path": dataset.path, "source_set_count": imported["source_set_count"] ?? NSNull(), "added_material_count": imported["added_material_count"] ?? NSNull(), "network_calls": calls, "warnings": preview["warnings"] ?? [], "index_sha256": imported["index_sha256"] ?? NSNull()]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: "/tmp/texture-import-readiness-wolf-repair.json"))
    }

    private func assertExactCrop(_ actualURL: URL, source: URL, rectangle: [Int], originals: [URL: Data]) throws {
        let original = try NativePNG.decode(try XCTUnwrap(originals[source], source.path)), expected = try original.crop(rectangle)
        let actual = try NativePNG.decode(Data(contentsOf: actualURL))
        XCTAssertEqual(actual.header, expected.header); XCTAssertEqual(actual.pixels, expected.pixels)
        XCTAssertEqual(actual.colorChunks.map(\.0), expected.colorChunks.map(\.0))
        XCTAssertEqual(actual.colorChunks.map(\.1), expected.colorChunks.map(\.1))
        guard actual.header == expected.header, actual.pixels == expected.pixels,
              actual.colorChunks.map(\.0) == expected.colorChunks.map(\.0), actual.colorChunks.map(\.1) == expected.colorChunks.map(\.1) else {
            throw StudioError("Native crop fidelity failed for " + source.path)
        }
    }
    private func command(_ arguments: [String], recovery: NativeMaterialSourceRecovery) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments, sourceRecovery: recovery)
        let text = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
    private func rejectingRecovery(_ probe: ImportReadinessNetworkProbe) -> NativeMaterialSourceRecovery {
        NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            await probe.record("metadata", url: url); throw StudioError("Local-ready import must not request provider metadata")
        }, download: { url, _ in
            await probe.record("download", url: url); throw StudioError("Local-ready import must not download maps")
        }))
    }
    private func image(width: Int, height: Int, bits: Int, channels: Int, role: String) -> NativePNG {
        NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels, color: [1: UInt8(0), 2: 4, 3: 2, 4: 6][channels]!, interlace: 0),
                  pixels: Data(repeating: 97, count: width * height * channels * bits / 8),
                  colorChunks: role == "input" ? [("gAMA", Data([0, 0, 177, 143])), ("sRGB", Data([0]))] : [])
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-import-ready-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); return root
    }
    private func checksum(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}

private actor ImportReadinessNetworkProbe {
    private var requested: [String] = []
    func record(_ kind: String, url: URL) { requested.append(kind + ":" + url.absoluteString) }
    func calls() -> [String] { requested }
}
