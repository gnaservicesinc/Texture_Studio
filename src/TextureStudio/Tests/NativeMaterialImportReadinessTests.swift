import Foundation
import XCTest
@testable import TextureStudio

/// Folder readiness uses current physical originals and the real crop commands.
/// A transport probe rejects any unintended source download.
@MainActor
final class NativeMaterialImportReadinessTests: XCTestCase {
    func testCompletePhysicalProviderAndAuthoredSetsPrepareWithoutNetworkAndPreserveNativeMaps() async throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        let probe = ImportReadinessNetworkProbe(), recovery = rejectingRecovery(probe)
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Local originals", "--training-size", "1024", "--subject-validation", "disabled", "--source-folder", "true"], recovery: recovery)
        _ = try await command(["edit-dataset", "--dataset", dataset.path, "--training-size", "1024", "--subject-validation", "disabled"], recovery: recovery)
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
        let inspected = try await command(["dataset", "--dataset", dataset.path, "--review-size", "1024"], recovery: recovery)
        XCTAssertEqual(inspected["source_set_count"] as? Int, 14)
        XCTAssertEqual(inspected["warnings"] as? [String], [])
        XCTAssertEqual((inspected["source_inventory"] as? [Any])?.count, originals.count)
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

    func testDeveloperCorpusFolderInspectionAccountsForCurrentOriginalsWithoutNetwork() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_IMPORT_CORPUS_PROOF"] == "1" else {
            throw XCTSkip("Opt-in read-only inspection of the real source library.")
        }
        let sources = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let probe = ImportReadinessNetworkProbe(), recovery = rejectingRecovery(probe)
        let pngs = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]))
            .compactMap { $0 as? URL }.filter { $0.pathExtension.lowercased() == "png" }
        let inspected = try await command(["dataset", "--dataset", sources.path, "--review-size", "1024"], recovery: recovery)
        let catalog = try XCTUnwrap(inspected["source_inventory"] as? [[String: Any]])
        let paths = Set(catalog.compactMap { $0["path"] as? String })
        XCTAssertTrue(Set(pngs.map { $0.resolvingSymlinksInPath().standardizedFileURL.path }).isSubset(of: paths))
        XCTAssertTrue(paths.allSatisfy { $0.hasPrefix(sources.path + "/") })
        let calls = await probe.calls()
        XCTAssertEqual(calls, [])
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
}

private actor ImportReadinessNetworkProbe {
    private var requested: [String] = []
    func record(_ kind: String, url: URL) { requested.append(kind + ":" + url.absoluteString) }
    func calls() -> [String] { requested }
}
