import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialSourceRecoveryTests: XCTestCase {
    func testRecoveryPublishesVerifiedOriginalIntoSourceFolder() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let bytes = try fixturePNG(), destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let downloadURL = providerURL(), apiURL = URL(string: "https://api.polyhaven.com/files/ribbed_corduroy")!
        let probe = RecoveryTransportProbe()
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes, probe: probe))
        let result = try await recovery.recover(request(destination))
        guard case .recovered(let provenance) = result else { return XCTFail("A missing published map must be restored") }
        XCTAssertEqual(try Data(contentsOf: destination), bytes, "Provider pixels and encoded source bytes must be preserved")
        XCTAssertEqual(provenance.destination, destination)
        XCTAssertEqual(provenance.downloadURL, downloadURL)
        XCTAssertEqual(provenance.apiURL, apiURL)
        XCTAssertEqual(provenance.sha256, sha256(bytes))
        XCTAssertEqual(provenance.publishedMD5, md5(bytes))
        XCTAssertEqual(provenance.publishedBytes, bytes.count)
        XCTAssertEqual(provenance.sourceWidth, 256); XCTAssertEqual(provenance.sourceHeight, 256)
        XCTAssertNil(provenance.cropRectangle)
        let calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1)
        XCTAssertEqual(try filenames(root), [destination.lastPathComponent], "Only the verified source may remain; no backup or staging file")
    }

    func testExistingSourceIsNeverOverwrittenOrDownloaded() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let original = Data("Local edited source".utf8); try original.write(to: destination)
        let probe = RecoveryTransportProbe()
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: try fixturePNG(), probe: probe))
        let result = try await recovery.recover(request(destination))
        guard case .alreadyPresent = result else { return XCTFail("Recovery must leave existing sources alone") }
        XCTAssertEqual(try Data(contentsOf: destination), original)
        let calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 0); XCTAssertEqual(calls.downloads, 0)
    }

    func testRecoveryRejectsBadPublishedChecksumsDimensionsAndDisplacementPrecision() async throws {
        let cases: [(String, Data, String?)] = [
            ("published checksum", try fixturePNG(), String(repeating: "0", count: 32)),
            ("dimensions", try fixturePNG(width: 512, height: 256), nil),
            ("excess over the crop tolerance", try fixturePNG(width: 263, height: 256), nil),
            ("undersized dimensions", try fixturePNG(width: 255, height: 256), nil),
            ("displacement precision", try fixturePNG(bits: 8), nil)
        ]
        for (name, bytes, publishedMD5) in cases {
            let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png")
            let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes, publishedMD5: publishedMD5))
            let result = try await recovery.recover(request(destination))
            guard case .failed = result else { XCTFail("Reject invalid \(name)"); continue }
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path), name)
            XCTAssertEqual(try filenames(root), [], "Rejected \(name) must not leave a partial original or staging file")
        }
    }

    func testRecordedSourceRecoveryRequiresExactHistoricalSHA256() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let bytes = try fixturePNG(), destination = root.appendingPathComponent("original-displacement.png")
        let probe = RecoveryTransportProbe()
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes, probe: probe))
        let result = try await recovery.recover(request(destination, expectedSHA256: String(repeating: "0", count: 64)))
        guard case .failed = result else { return XCTFail("A replacement provider version cannot silently change the dataset's source identity") }
        XCTAssertEqual(try filenames(root), [])
        let restored = try await recovery.recover(request(destination, expectedSHA256: sha256(bytes)))
        guard case .recovered = restored else { return XCTFail("The recorded source checksum must be recoverable") }
        XCTAssertEqual(try Data(contentsOf: destination), bytes)
        let calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 1, "Requests for one asset should reuse its published catalog")
        XCTAssertEqual(calls.downloads, 2)
    }

    func testRecoveryRejectsCorruptPNGAndWrongPublishedLength() async throws {
        var damaged = try fixturePNG(); damaged[29] ^= 1
        let valid = try fixturePNG()
        for (bytes, publishedLength) in [(damaged, damaged.count), (valid, valid.count + 1)] {
            let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png")
            let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes, publishedLength: publishedLength))
            let result = try await recovery.recover(request(destination))
            guard case .failed = result else { XCTFail("Never publish malformed or truncated originals"); continue }
            XCTAssertEqual(try filenames(root), [])
        }
    }

    func testRecoveryPreservesRoughnessPrecisionAndRejectsScalarNormalMap() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let roughness = root.appendingPathComponent("ribbed_corduroy_rough_1k.png"), bytes = try fixturePNG(bits: 8)
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes, target: "roughness"))
        let result = try await recovery.recover(request(roughness, target: "roughness"))
        guard case .recovered = result else { return XCTFail("Published roughness can retain its native 8-bit precision") }
        XCTAssertEqual(try Data(contentsOf: roughness), bytes)
        let normal = root.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png")
        let invalid = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes, target: "normal"))
        let invalidResult = try await invalid.recover(request(normal, target: "normal"))
        guard case .failed = invalidResult else { return XCTFail("A normal source needs RGB or RGBA channels before it is saved") }
        XCTAssertEqual(try filenames(root), [roughness.lastPathComponent])
    }

    func testHistoricalColorVariantAndDirectXNormalRestoreTheirOwnPublishedFiles() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0),
                                 pixels: Data((0..<256 * 256 * 3).map { UInt8(truncatingIfNeeded: $0 * 97) }), colorChunks: [])
        let bytes = try original.encoded()
        for (target, publishedRole, filename) in [("input", Optional("col_03"), "ribbed_corduroy_col_03_1k.png"),
                                                   ("normal_dx", nil, "ribbed_corduroy_nor_dx_1k.png")] {
            let destination = root.appendingPathComponent(filename)
            let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes, target: target, publishedRole: publishedRole))
            let result = try await recovery.recover(request(destination, target: target, expectedSHA256: sha256(bytes), publishedRole: publishedRole))
            guard case .recovered(let provenance) = result else { XCTFail("Restore the exact published role for \(filename)"); continue }
            XCTAssertEqual(provenance.downloadURL, providerURL(target: target, publishedRole: publishedRole))
            XCTAssertEqual(try Data(contentsOf: destination), bytes)
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: destination)).pixels, original.pixels,
                           "Source restoration must preserve the historical color variant and native normal convention")
        }
        XCTAssertEqual(try filenames(root), ["ribbed_corduroy_col_03_1k.png", "ribbed_corduroy_nor_dx_1k.png"])
    }

    func testSlightlyOversizedPublishedMapUsesExactCenteredIntegerCropAndSavedChecksum() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = NativePNG(header: .init(width: 258, height: 259, bits: 16, channels: 1, color: 0, interlace: 0),
                                 pixels: Data((0..<258 * 259 * 2).map { UInt8(truncatingIfNeeded: $0 * 97) }), colorChunks: [])
        let providerBytes = try original.encoded(), expected = try original.crop([1, 1, 256, 256])
        let expectedBytes = try expected.encoded(), destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: providerBytes))
        let result = try await recovery.recover(request(destination))
        guard case .recovered(let provenance) = result else { return XCTFail("A few extra provider pixels must be cropped automatically") }
        let saved = try Data(contentsOf: destination), decoded = try NativePNG.decode(saved)
        XCTAssertEqual(decoded.header.width, 256); XCTAssertEqual(decoded.header.height, 256)
        XCTAssertEqual(decoded.header.bits, 16); XCTAssertEqual(decoded.header.channels, 1)
        XCTAssertEqual(decoded.pixels, expected.pixels, "Centering uses floor division and must preserve every retained 16-bit code")
        XCTAssertEqual(saved, expectedBytes)
        XCTAssertEqual(provenance.sha256, sha256(saved), "The dataset binds the saved source after the exact crop")
        XCTAssertEqual(provenance.publishedMD5, md5(providerBytes), "Provider evidence remains bound to the original download")
        XCTAssertEqual(provenance.publishedBytes, providerBytes.count)
        XCTAssertEqual(provenance.sourceWidth, 258); XCTAssertEqual(provenance.sourceHeight, 259)
        XCTAssertEqual(provenance.cropRectangle, [1, 1, 256, 256])
        let restored = root.appendingPathComponent("recorded-height.png")
        let second = try await recovery.recover(request(restored, expectedSHA256: sha256(expectedBytes)))
        guard case .recovered = second else { return XCTFail("Recovery compares historical SHA256 with final saved source bytes") }
        XCTAssertEqual(try Data(contentsOf: restored), expectedBytes)
        XCTAssertEqual(try filenames(root), ["recorded-height.png", "ribbed_corduroy_disp_1k.png"], "No download copy or backup remains")
    }

    func testUnpublishedMapAndNetworkFailureAreSoftResults() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png"), probe = RecoveryTransportProbe()
        let absent = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            await probe.recordMetadata()
            return .init(data: Data("{}".utf8), statusCode: 200, url: url)
        }, download: { _, _ in
            XCTFail("Unpublished maps must not start a download")
            throw URLError(.unsupportedURL)
        }))
        let result = try await absent.recover(request(destination))
        guard case .notPublished = result else { return XCTFail("An unavailable provider map must be a soft result") }
        let offline = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { _ in throw URLError(.notConnectedToInternet) },
                                                                   download: { _, _ in throw URLError(.notConnectedToInternet) }))
        let offlineResult = try await offline.recover(request(destination))
        guard case .failed(let reason) = offlineResult else { return XCTFail("Offline recovery must report a soft failure") }
        XCTAssertFalse(reason.isEmpty)
        XCTAssertEqual(try filenames(root), [])
    }

    func testCancellationRemovesDownloadStageAndDoesNotPublishPartialOriginal() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png"), bytes = try fixturePNG()
        let metadata = try providerMetadata(bytes: bytes), probe = RecoveryTransportProbe()
        let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            .init(data: metadata, statusCode: 200, url: url)
        }, download: { _, stage in
            try bytes.prefix(100).write(to: stage)
            await probe.recordDownload(stage)
            try await Task.sleep(for: .seconds(60))
            throw CancellationError()
        }))
        let request = request(destination)
        let task = Task { try await recovery.recover(request) }
        let stage = await probe.waitForDownload()
        XCTAssertTrue(FileManager.default.fileExists(atPath: stage.path))
        task.cancel()
        do { _ = try await task.value; XCTFail("Cancellation must propagate to the import command") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        XCTAssertEqual(try filenames(root), [], "Cancellation must remove only its incomplete staging file")
    }

    func testSourceAppearingDuringDownloadWinsWithoutBeingOverwritten() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png"), bytes = try fixturePNG()
        let local = Data("A source created while recovery was in progress".utf8)
        let metadata = try providerMetadata(bytes: bytes)
        let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            .init(data: metadata, statusCode: 200, url: url)
        }, download: { url, stage in
            try bytes.write(to: stage)
            try local.write(to: destination)
            return .init(statusCode: 200, url: url)
        }))
        let result = try await recovery.recover(request(destination))
        guard case .alreadyPresent = result else { return XCTFail("Atomic publication must preserve a concurrent source writer") }
        XCTAssertEqual(try Data(contentsOf: destination), local)
        XCTAssertEqual(try filenames(root), [destination.lastPathComponent])
    }

    func testFolderPreviewIncludesRecoveredMapAndPreparationRestoresItsRecordedIdentity() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let input = sources.appendingPathComponent("ribbed_corduroy_diff_1k.png")
        let normal = sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png")
        let height = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let rgb = try rgbFixturePNG(), numeric = try fixturePNG(width: 1024, height: 1024)
        try rgb.write(to: input); try rgb.write(to: normal)
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: numeric))
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Automatic recovery", "--training-size", "256"], recovery: recovery)
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path], recovery: recovery)
        XCTAssertEqual(try Data(contentsOf: height), numeric, "The original source folder must be fixed before preview")
        let plan = try object(planURL), inventory = try XCTUnwrap(plan["inventory"] as? [[Any]])
        XCTAssertTrue(inventory.contains {
            guard let path = $0.first as? String else { return false }
            return URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL == height.resolvingSymlinksInPath().standardizedFileURL
        }, "Import proof must bind the source recovered before inventory")
        let pending = try XCTUnwrap((plan["materials"] as? [[String: Any]])?.first)
        let maps = try XCTUnwrap(pending["maps"] as? [String: [String: Any]])
        XCTAssertEqual(maps["height"]?["file_sha256"] as? String, sha256(numeric))
        let imported = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path,
            "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String),
            "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String)], recovery: recovery)
        XCTAssertEqual(imported["added_material_count"] as? Int, 1)
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        try FileManager.default.removeItem(at: height)
        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"], recovery: recovery)
        XCTAssertEqual(try Data(contentsOf: height), numeric, "Deleted recorded maps must restore the same original bytes")
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex,
                       "Restoring a checksum-identical original must preserve its dataset binding")
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 1)
        let descriptor = try XCTUnwrap(descriptors.first)
        XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.targetURL)).pixels,
                       try NativePNG.decode(numeric).crop([384, 384, 256, 256]).pixels)
        XCTAssertEqual(try Data(contentsOf: input), rgb); XCTAssertEqual(try Data(contentsOf: normal), rgb)
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
    }

    func testProviderResolutionLabelWithOneExtraRowStillRecoversMissingTarget() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let rgb = try rgbFixturePNG(height: 1025), numeric = try fixturePNG(width: 1024, height: 1025)
        try rgb.write(to: sources.appendingPathComponent("ribbed_corduroy_diff_1k.png"))
        try rgb.write(to: sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png"))
        let height = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: numeric))
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Provider pixel registration", "--training-size", "256"], recovery: recovery)
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path], recovery: recovery)
        XCTAssertEqual(preview["added_material_count"] as? Int, 1)
        XCTAssertEqual(try Data(contentsOf: height), numeric, "A legitimate 1k source with an extra row must still obtain its missing map")
        let material = try XCTUnwrap((try object(planURL)["materials"] as? [[String: Any]])?.first)
        XCTAssertEqual(material["common_pixel_dimensions"] as? [Int], [1024, 1025])
        let maps = try XCTUnwrap(material["maps"] as? [String: [String: Any]])
        let heightPath = try XCTUnwrap(maps["height"]?["path"] as? String)
        XCTAssertEqual(URL(fileURLWithPath: heightPath).resolvingSymlinksInPath().standardizedFileURL,
                       height.resolvingSymlinksInPath().standardizedFileURL)
    }

    func testRecoveryInsideOwnedSourcesAdvancesItsInventoryProofWithoutRejectingTheScan() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: try fixturePNG(width: 1024, height: 1024)))
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Owned original sources", "--training-size", "256"], recovery: recovery)
        let sources = dataset.appendingPathComponent("sources/Subject")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: true)
        let rgb = try rgbFixturePNG(), roughness = try fixturePNG(width: 1024, height: 1024, bits: 8)
        try rgb.write(to: sources.appendingPathComponent("ribbed_corduroy_diff_1k.png"))
        try rgb.write(to: sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png"))
        try roughness.write(to: sources.appendingPathComponent("ribbed_corduroy_rough_1k.png"))
        let before = try await output(["dataset", "--dataset", dataset.path], recovery: recovery)
        XCTAssertEqual(before["material_count"] as? Int, 1)
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", dataset.appendingPathComponent("sources").path,
            "--plan", planURL.path, "--expected-index-sha256", try XCTUnwrap(before["index_sha256"] as? String)], recovery: recovery)
        XCTAssertEqual(preview["duplicate_material_count"] as? Int, 1, "The new map belongs to the existing material")
        XCTAssertEqual(preview["added_material_count"] as? Int, 0)
        let height = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        XCTAssertTrue(FileManager.default.fileExists(atPath: height.path))
        let after = try await output(["dataset", "--dataset", dataset.path], recovery: recovery)
        XCTAssertEqual(after["index_sha256"] as? String, preview["index_sha256"] as? String,
                       "Recovery must commit its owned source inventory together with attached target metadata")
        let plans = try XCTUnwrap(preview["plans"] as? [String: [String: [String: Any]]])
        XCTAssertEqual(plans["256"]?["height"]?["train_count"] as? Int, 1)
    }

    func testPreparationAddsMissingPublishedTargetToExistingMaterialAndPreservesCuration() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let input = sources.appendingPathComponent("ribbed_corduroy_diff_1k.png")
        let normal = sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png")
        let height = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let rgb = try rgbFixturePNG(), numeric = try fixturePNG(width: 1024, height: 1024)
        try rgb.write(to: input); try rgb.write(to: normal)
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: numeric))
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Existing incomplete material", "--training-size", "256"], recovery: recovery)
        let imported = try await output(["add-material", "--dataset", dataset.path, "--name", "ribbed_corduroy",
                                         "--input", input.path, "--normal", normal.path], recovery: recovery)
        let reviewHash = try XCTUnwrap(imported["review_sha256"] as? String)
        _ = try await output(["curate", "--dataset", dataset.path, "--sample", "ribbed_corduroy_1k_center",
                              "--review-size", "256", "--status", "approved", "--note", "Keep the reviewed crop",
                              "--expected-review-sha256", reviewHash], recovery: recovery)
        let unpublished = unpublishedRecovery()
        let cached = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "normal"], recovery: unpublished)
        let cachedURL = URL(fileURLWithPath: try XCTUnwrap(cached["dataset_path"] as? String))
        let original = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"], recovery: recovery)
        let updated = try await output(["curate", "--dataset", dataset.path, "--sample", "ribbed_corduroy_1k_center",
                              "--review-size", "256", "--status", "approved", "--note", "Edited after cached normal preparation",
                              "--expected-review-sha256", try XCTUnwrap(original["review_sha256"] as? String)], recovery: recovery)
        let prepared = try await output(["prepare-size", "--dataset", cachedURL.path, "--size", "256", "--target", "height",
            "--expected-index-sha256", try XCTUnwrap(cached["index_sha256"] as? String),
            "--expected-review-sha256", try XCTUnwrap(updated["review_sha256"] as? String)], recovery: recovery)
        XCTAssertEqual(try Data(contentsOf: height), numeric)
        let inspected = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"], recovery: recovery)
        XCTAssertEqual(inspected["material_count"] as? Int, 1, "Adding a target must enrich the existing source set without a name collision")
        let material = try XCTUnwrap((inspected["materials"] as? [[String: Any]])?.first)
        let sample = try XCTUnwrap((material["samples"] as? [[String: Any]])?.first)
        XCTAssertEqual(sample["status"] as? String, "approved")
        XCTAssertEqual(sample["note"] as? String, "Edited after cached normal preparation",
                       "An old prepared dataset must not overwrite current original reviews when a recovered map enriches the source binding")
        let maps = try XCTUnwrap(sample["maps"] as? [String: [String: Any]])
        XCTAssertNotNil(maps["height"])
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 1)
        XCTAssertEqual(try Data(contentsOf: input), rgb); XCTAssertEqual(try Data(contentsOf: normal), rgb)
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
        _ = try await output(["cleanup-size", "--dataset", cachedURL.path], recovery: recovery)
    }

    func testUnpublishedDisplacementSkipsOnlyHeightTrainingAndKeepsNormalTrainingUsable() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let rgb = try rgbFixturePNG(), roughness = try fixturePNG(width: 1024, height: 1024, bits: 8)
        let height = try fixturePNG(width: 1024, height: 1024)
        for asset in ["fabric_pattern_05", "brick_complete"] {
            try rgb.write(to: sources.appendingPathComponent(asset + "_diff_1k.png"))
            try rgb.write(to: sources.appendingPathComponent(asset + "_nor_gl_1k.png"))
            try roughness.write(to: sources.appendingPathComponent(asset + "_rough_1k.png"))
        }
        try height.write(to: sources.appendingPathComponent("brick_complete_disp_1k.png"))
        let recovery = unpublishedRecovery()
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Optional published maps", "--training-size", "256"], recovery: recovery)
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path], recovery: recovery)
        XCTAssertEqual(preview["added_material_count"] as? Int, 2)
        let plans = try XCTUnwrap(preview["plans"] as? [String: [String: [String: Any]]])
        let issues = try XCTUnwrap(plans["256"]?["height"]?["source_issues"] as? [[String: Any]])
        XCTAssertEqual(issues.count, 1)
        XCTAssertEqual(issues.first?["material_id"] as? String, "fabric_pattern_05_1k")
        XCTAssertEqual(issues.first?["code"] as? String, "not_published")
        XCTAssertTrue((issues.first?["reason"] as? String)?.contains("other maps remain usable") == true)
        _ = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path,
            "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String)], recovery: recovery)
        let preparedHeight = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"], recovery: recovery)
        let heightURL = URL(fileURLWithPath: try XCTUnwrap(preparedHeight["dataset_path"] as? String))
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: heightURL, size: 256, target: "height").count, 1)
        _ = try await output(["cleanup-size", "--dataset", heightURL.path], recovery: recovery)
        let preparedNormal = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "normal"], recovery: recovery)
        let normalURL = URL(fileURLWithPath: try XCTUnwrap(preparedNormal["dataset_path"] as? String))
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: normalURL, size: 256, target: "normal").count, 2,
                       "An unpublished displacement must not prevent training on that material's normal map")
        XCTAssertFalse(FileManager.default.fileExists(atPath: sources.appendingPathComponent("fabric_pattern_05_disp_1k.png").path))
        _ = try await output(["cleanup-size", "--dataset", normalURL.path], recovery: recovery)
    }

    private func fixtureTransport(bytes: Data, target: String = "height", publishedRole: String? = nil,
                                  publishedMD5: String? = nil, publishedLength: Int? = nil,
                                  probe: RecoveryTransportProbe = RecoveryTransportProbe()) -> NativeMaterialSourceRecovery.Transport {
        let metadata = try! providerMetadata(bytes: bytes, target: target, publishedRole: publishedRole,
                                            publishedMD5: publishedMD5, publishedLength: publishedLength)
        return .init(fetchMetadata: { url in
            await probe.recordMetadata()
            return .init(data: metadata, statusCode: 200, url: url)
        }, download: { url, stage in
            await probe.recordDownload(stage)
            try bytes.write(to: stage)
            return .init(statusCode: 200, url: url)
        })
    }
    private func unpublishedRecovery() -> NativeMaterialSourceRecovery {
        NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            .init(data: Data("{}".utf8), statusCode: 200, url: url)
        }, download: { _, _ in
            XCTFail("No download exists in an unpublished provider catalog")
            throw URLError(.unsupportedURL)
        }))
    }
    private func request(_ destination: URL, target: String = "height", expectedSHA256: String? = nil,
                         publishedRole: String? = nil) -> NativeMaterialSourceRecovery.Request {
        .init(asset: "ribbed_corduroy", resolution: "1k", target: target, destination: destination,
              width: 256, height: 256, expectedSHA256: expectedSHA256, publishedRole: publishedRole)
    }
    private func providerURL(target: String = "height", publishedRole: String? = nil) -> URL {
        let suffix = publishedRole ?? ["input": "diff", "height": "disp", "roughness": "rough", "normal": "nor_gl", "normal_dx": "nor_dx"][target]!
        return URL(string: "https://dl.polyhaven.org/file/ph-assets/Textures/png/1k/ribbed_corduroy/ribbed_corduroy_\(suffix)_1k.png")!
    }
    private func providerMetadata(bytes: Data, target: String = "height", publishedRole: String? = nil,
                                  publishedMD5: String? = nil, publishedLength: Int? = nil) throws -> Data {
        let role = publishedRole ?? ["input": "Diffuse", "height": "Displacement", "roughness": "Rough", "normal": "nor_gl", "normal_dx": "nor_dx"][target]!
        return try JSONSerialization.data(withJSONObject: [role: ["1k": ["png": ["url": providerURL(target: target, publishedRole: publishedRole).absoluteString,
            "size": publishedLength ?? bytes.count, "md5": publishedMD5 ?? md5(bytes)]]]])
    }
    private func fixturePNG(width: Int = 256, height: Int = 256, bits: Int = 16) throws -> Data {
        try NativePNG(header: .init(width: width, height: height, bits: bits, channels: 1, color: 0, interlace: 0),
                      pixels: Data(repeating: 97, count: width * height * (bits / 8)), colorChunks: []).encoded()
    }
    private func rgbFixturePNG(width: Int = 1024, height: Int = 1024) throws -> Data {
        try NativePNG(header: .init(width: width, height: height, bits: 8, channels: 3, color: 2, interlace: 0),
                      pixels: Data(repeating: 127, count: width * height * 3), colorChunks: []).encoded()
    }
    private func output(_ arguments: [String], recovery: NativeMaterialSourceRecovery) async throws -> [String: Any] {
        let result = try await NativeMaterialDatasetService.run(arguments: arguments, sourceRecovery: recovery)
        let text = try XCTUnwrap(result)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
    private func object(_ url: URL) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
    private func sha256(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    private func md5(_ bytes: Data) -> String { Insecure.MD5.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    private func filenames(_ folder: URL) throws -> [String] { try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-source-recovery-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
}

private actor RecoveryTransportProbe {
    private var metadata = 0, downloads = 0
    private var stage: URL?, waiter: CheckedContinuation<URL, Never>?
    func recordMetadata() { metadata += 1 }
    func recordDownload(_ path: URL) {
        downloads += 1; stage = path
        waiter?.resume(returning: path); waiter = nil
    }
    func counts() -> (metadata: Int, downloads: Int) { (metadata, downloads) }
    func waitForDownload() async -> URL {
        if let stage { return stage }
        return await withCheckedContinuation { waiter = $0 }
    }
}
