import CryptoKit
import Foundation
import XCTest
import zlib
@testable import TextureStudio

@MainActor
final class NativeMaterialSourceRecoveryTests: XCTestCase {
    func testPolyHavenLeatherColorAliasRecoversExactPublishedVariantAndRetainsProof() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let rgb = try rgbFixturePNG(width: 256, height: 256), other = try rgbFixturePNG(width: 256, height: 257)
        for asset in ["leather_red_02", "leather_red_03"] {
            let colorURL = URL(string: "https://dl.polyhaven.org/file/ph-assets/Textures/png/1k/\(asset)/\(asset)_coll1_1k.png")!
            let secondURL = URL(string: "https://dl.polyhaven.org/file/ph-assets/Textures/png/1k/\(asset)/\(asset)_coll2_1k.png")!
            let metadata = try JSONSerialization.data(withJSONObject: [
                "coll1": ["1k": ["png": ["url": colorURL.absoluteString, "md5": md5(rgb), "size": rgb.count]]],
                "coll2": ["1k": ["png": ["url": secondURL.absoluteString, "md5": md5(other), "size": other.count]]]
            ])
            let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
                .init(data: metadata, statusCode: 200, url: url)
            }, download: { url, staged in
                XCTAssertEqual(url, colorURL, "Restore primary coll1; never silently switch to coll2")
                try rgb.write(to: staged); return .init(statusCode: 200, url: url)
            }))
            for (offset, item) in [("col_1", Optional<String>.none), ("COL_1", Optional(sha256(rgb))), ("coll1", Optional(sha256(rgb)))].enumerated() {
                let (role, expected) = item
                let destination = root.appendingPathComponent("\(asset)_alias\(offset)_1k.png")
                let result = try await recovery.recover(.init(asset: asset, resolution: "1k", target: "input", destination: destination,
                    width: 256, height: 256, expectedSHA256: expected, publishedRole: role))
                guard case .recovered(let proof) = result else { XCTFail("Resolve \(role) to exact published coll1"); continue }
                XCTAssertEqual(proof.downloadURL, colorURL)
                XCTAssertEqual(try Data(contentsOf: destination), rgb)
                XCTAssertNotNil(NativeMaterialSourceRecovery.sourceProvenance(for: destination,
                    sha256: sha256(rgb), bytes: rgb.count, width: 256, height: 256))
            }
        }
    }

    func testPolyHavenInputRoleLookupIgnoresCaseAndRejectsCompetingAliases() async throws {
        let bytes = try rgbFixturePNG(width: 256, height: 256)
        for competing in [false, true] {
            let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let base = "https://dl.polyhaven.org/file/ph-assets/Textures/png/1k/ribbed_corduroy/"
            let url = URL(string: base + "ribbed_corduroy_col_1_1k.png")!
            var catalog: [String: Any] = ["Col_1": ["1k": ["png": ["url": url.absoluteString, "md5": md5(bytes), "size": bytes.count]]]]
            if competing { catalog["coll1"] = ["1k": ["png": ["url": base + "ribbed_corduroy_coll1_1k.png", "md5": md5(bytes), "size": bytes.count]]] }
            let metadata = try JSONSerialization.data(withJSONObject: catalog)
            let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { requested in
                .init(data: metadata, statusCode: 200, url: requested)
            }, download: { requested, staged in
                XCTAssertFalse(competing, "Ambiguous aliases must fail before a download")
                try bytes.write(to: staged); return .init(statusCode: 200, url: requested)
            }))
            let destination = root.appendingPathComponent("ribbed_corduroy_col_1_1k.png")
            let result = try await recovery.recover(request(destination, target: "input", publishedRole: "col_1"))
            if competing {
                guard case .failed(let reason) = result else { XCTFail("Competing color aliases cannot choose arbitrary pixels"); continue }
                XCTAssertTrue(reason.contains("competing")); XCTAssertEqual(try filenames(root), [])
            } else {
                guard case .recovered = result else { XCTFail("Case-insensitive color roles must resolve"); continue }
                XCTAssertEqual(try Data(contentsOf: destination), bytes)
            }
        }
    }

    func testRecoveryProofSurvivesReopenAndOnlyAttachesToItsExactSource() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let bytes = try fixturePNG(width: 32, height: 33)
        let archive = try ambientZIP([("Wicker013_1K-PNG_Displacement.png", bytes)], deflated: true)
        let destination = root.appendingPathComponent("Wicker013_1K-PNG_Displacement.png")
        let recovery = NativeMaterialSourceRecovery(transport: try ambientTransport(archive))
        let result = try await recovery.recover(ambientRequest(destination, target: "height"))
        guard case .recovered(let proof) = result else { return XCTFail("The complete original and durable proof must be saved") }
        let sidecar = NativeMaterialSourceRecovery.provenanceSidecar(for: destination)
        XCTAssertEqual(proof.provenanceSidecarURL, sidecar); XCTAssertNil(proof.provenanceWarning)
        XCTAssertTrue(FileManager.default.fileExists(atPath: sidecar.path))
        let persisted = try object(sidecar)
        XCTAssertEqual(persisted["source_sha256"] as? String, sha256(bytes))
        XCTAssertEqual(persisted["source_bytes"] as? Int, bytes.count)
        XCTAssertEqual(persisted["archive_sha256"] as? String, sha256(archive))
        let rebound = try XCTUnwrap(NativeMaterialSourceRecovery.sourceProvenance(for: destination,
            sha256: sha256(bytes), bytes: bytes.count, width: 32, height: 33))
        XCTAssertEqual(rebound["provider"] as? String, "ambientCG")
        XCTAssertEqual(rebound["published_recovery_proof_path"] as? String, sidecar.path)
        let savedEvidence = try XCTUnwrap(rebound["source_recovery_provenance"] as? [String: Any])
        XCTAssertEqual(savedEvidence["archive_member"] as? String, "Wicker013_1K-PNG_Displacement.png")
        XCTAssertNil(NativeMaterialSourceRecovery.sourceProvenance(for: destination, sha256: sha256(bytes),
            bytes: bytes.count, width: 32, height: 32), "The proof cannot be attached to another grid")
        let changed = Data("replaced source".utf8); try changed.write(to: destination)
        XCTAssertNil(NativeMaterialSourceRecovery.sourceProvenance(for: destination, sha256: sha256(changed),
            bytes: changed.count, width: 32, height: 33), "Stale proof must not identify edited source bytes")
        XCTAssertEqual(try Data(contentsOf: sidecar), try JSONSerialization.data(withJSONObject: persisted, options: [.sortedKeys]))
    }

    func testRecoveryProofWriteFailureReportsSavedOriginalTruthfully() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let bytes = try fixturePNG(), destination = root.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let sidecar = NativeMaterialSourceRecovery.provenanceSidecar(for: destination)
        try FileManager.default.createDirectory(at: sidecar, withIntermediateDirectories: false)
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes))
        let result = try await recovery.recover(request(destination))
        guard case .recovered(let proof) = result else { return XCTFail("The original was saved even though the proof location is unavailable") }
        XCTAssertEqual(try Data(contentsOf: destination), bytes)
        XCTAssertNil(proof.provenanceSidecarURL)
        XCTAssertTrue(try XCTUnwrap(proof.provenanceWarning).contains("Saved the original map"))
        XCTAssertNil(NativeMaterialSourceRecovery.sourceProvenance(for: destination,
            sha256: sha256(bytes), bytes: bytes.count, width: 256, height: 256))
    }

    func testAmbientCGStoredAndDeflatedPackagesPreservePairedPNGBytesAndDownloadOnce() async throws {
        for deflated in [false, true] {
            let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let color = try rgbFixturePNG(width: 32, height: 33)
            let height = try NativePNG(header: .init(width: 32, height: 33, bits: 16, channels: 1, color: 0, interlace: 0),
                pixels: Data((0..<32 * 33 * 2).map { UInt8(truncatingIfNeeded: $0 * 97) }), colorChunks: []).encoded()
            let unrelated = try fixturePNG(width: 32, height: 33, bits: 8)
            let archive = try ambientZIP([("Wicker013_1K-PNG_Color.png", color),
                                          ("Wicker013_1K-PNG_Displacement.png", height),
                                          ("Wicker013_1K-PNG_Roughness.png", unrelated)], deflated: deflated)
            let probe = RecoveryTransportProbe(), recovery = NativeMaterialSourceRecovery(transport: try ambientTransport(archive, probe: probe))
            for (target, bytes) in [("input", color), ("height", height)] {
                let destination = root.appendingPathComponent("Wicker013_\(target)_1k.png")
                let result = try await recovery.recover(ambientRequest(destination, target: target))
                guard case .recovered(let proof) = result else { XCTFail("Recover the exact \(target) archive member"); continue }
                XCTAssertEqual(try Data(contentsOf: destination), bytes)
                XCTAssertEqual(proof.provider, .ambientCG)
                XCTAssertEqual(proof.apiURL.absoluteString, "https://ambientcg.com/api/v2/full_json?include=downloadData&id=Wicker013")
                XCTAssertEqual(proof.downloadURL.absoluteString, "https://ambientcg.com/get?file=Wicker013_1K-PNG.zip")
                XCTAssertEqual(proof.publishedMD5, "", "ambientCG's API does not publish an MD5; it must not be fabricated")
                XCTAssertEqual(proof.publishedBytes, archive.count)
                XCTAssertEqual(proof.archiveSHA256, sha256(archive))
                XCTAssertEqual(proof.archiveSHA1, sha1(archive))
                XCTAssertEqual(proof.archiveMember, "Wicker013_1K-PNG_\(target == "input" ? "Color" : "Displacement").png")
                XCTAssertEqual(proof.archiveMemberBytes, bytes.count)
                XCTAssertEqual(proof.archiveMemberCRC32, String(format: "%08x", crc(bytes)))
                XCTAssertEqual(proof.sha256, sha256(bytes)); XCTAssertNil(proof.cropRectangle)
                XCTAssertEqual(proof.sourceWidth, 32); XCTAssertEqual(proof.sourceHeight, 33)
            }
            let calls = await probe.counts()
            XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1)
            XCTAssertEqual(try filenames(root), ["Wicker013_height_1k.png", "Wicker013_input_1k.png"],
                           "Do not publish unrelated maps or retain archive/staging copies")
            let numeric = try NativePNG.decode(Data(contentsOf: root.appendingPathComponent("Wicker013_height_1k.png")))
            XCTAssertEqual(numeric.header.bits, 16)
            XCTAssertEqual(numeric.pixels, try NativePNG.decode(height).pixels, "Every original 16-bit code remains exact")
        }
    }

    func testAmbientCGGrayscaleDiffuseRecoveryPreservesEightAndSixteenBitOriginalsWithAlpha() async throws {
        for bits in [8, 16] {
            for channels in [1, 2] {
                let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
                let original = grayscaleFixture(width: 32, height: 33, bits: bits, channels: channels)
                let bytes = try original.encoded()
                let archive = try ambientZIP([("Wicker013_1K-PNG_Color.png", bytes)], deflated: true)
                let probe = RecoveryTransportProbe()
                let recovery = NativeMaterialSourceRecovery(transport: try ambientTransport(archive, probe: probe))
                let destination = root.appendingPathComponent("Wicker013_1K-PNG_Color.png")
                let result = try await recovery.recover(ambientRequest(destination, target: "input", expectedSHA256: sha256(bytes)))
                guard case .recovered(let proof) = result else {
                    XCTFail("Recover native \(bits)-bit grayscale diffuse with \(channels) channels"); continue
                }
                XCTAssertEqual(try Data(contentsOf: destination), bytes, "Do not convert, resize, or recompress a recovered grayscale source")
                let decoded = try NativePNG.decode(Data(contentsOf: destination))
                XCTAssertEqual(decoded.header, original.header)
                XCTAssertEqual(decoded.pixels, original.pixels, "Retain every original grayscale and alpha code")
                XCTAssertEqual(proof.sha256, sha256(bytes)); XCTAssertNil(proof.cropRectangle)
                XCTAssertEqual(proof.archiveMember, "Wicker013_1K-PNG_Color.png")
                XCTAssertEqual(proof.archiveMemberCRC32, String(format: "%08x", crc(bytes)))
                XCTAssertNotNil(NativeMaterialSourceRecovery.sourceProvenance(for: destination,
                    sha256: sha256(bytes), bytes: bytes.count, width: 32, height: 33))
                let samples = try decoded.modelFloatSamples(role: "input", encoding: "srgb", inputTransfer: .linear)
                let count = original.header.width * original.header.height
                XCTAssertEqual(Array(samples[0..<count]), Array(samples[count..<count * 2]))
                XCTAssertEqual(Array(samples[0..<count]), Array(samples[count * 2..<count * 3]),
                    "RGB expansion belongs only to the model tensor")
                let existing = try await recovery.recover(ambientRequest(destination, target: "input"))
                guard case .alreadyPresent = existing else { XCTFail("An existing grayscale source is authoritative"); continue }
                let calls = await probe.counts()
                XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1)
            }
        }
    }

    func testPolyHavenGrayscaleDiffuseRecoveryPreservesEightAndSixteenBitOriginalsWithAlpha() async throws {
        for bits in [8, 16] {
            for channels in [1, 2] {
                let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
                let original = grayscaleFixture(width: 256, height: 256, bits: bits, channels: channels)
                let bytes = try original.encoded(), destination = root.appendingPathComponent("ribbed_corduroy_diff_1k.png")
                let probe = RecoveryTransportProbe()
                let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: bytes, target: "input", probe: probe))
                let result = try await recovery.recover(request(destination, target: "input", expectedSHA256: sha256(bytes)))
                guard case .recovered(let proof) = result else {
                    XCTFail("Recover native \(bits)-bit grayscale diffuse with \(channels) channels"); continue
                }
                XCTAssertEqual(try Data(contentsOf: destination), bytes)
                let decoded = try NativePNG.decode(Data(contentsOf: destination))
                XCTAssertEqual(decoded.header, original.header); XCTAssertEqual(decoded.pixels, original.pixels)
                XCTAssertEqual(proof.publishedMD5, md5(bytes)); XCTAssertEqual(proof.sha256, sha256(bytes))
                XCTAssertEqual(proof.sourceWidth, 256); XCTAssertEqual(proof.sourceHeight, 256); XCTAssertNil(proof.cropRectangle)
                XCTAssertNotNil(NativeMaterialSourceRecovery.sourceProvenance(for: destination,
                    sha256: sha256(bytes), bytes: bytes.count, width: 256, height: 256))
                let existing = try await recovery.recover(request(destination, target: "input"))
                guard case .alreadyPresent = existing else { XCTFail("Do not download an already recovered grayscale source"); continue }
                let calls = await probe.counts()
                XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1)
            }
        }
    }

    func testAmbientCGScalarNormalRecoveryStillRejectsGrayscaleAndGrayAlphaForBothConventions() async throws {
        for target in ["normal", "normal_dx"] {
            for channels in [1, 2] {
                let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
                let bytes = try grayscaleFixture(width: 32, height: 33, bits: 16, channels: channels).encoded()
                let role = target == "normal" ? "NormalGL" : "NormalDX"
                let archive = try ambientZIP([("Wicker013_1K-PNG_\(role).png", bytes)])
                let recovery = NativeMaterialSourceRecovery(transport: try ambientTransport(archive))
                let destination = root.appendingPathComponent("Wicker013_1K-PNG_\(role).png")
                let result = try await recovery.recover(ambientRequest(destination, target: target))
                guard case .failed(let reason) = result else { XCTFail("A scalar image cannot supply normal vectors"); continue }
                XCTAssertTrue(reason.contains("normal map")); XCTAssertFalse(reason.contains("color or normal"))
                XCTAssertEqual(try filenames(root), [], "Reject invalid vectors without publishing partial files")
            }
        }
    }

    func testAmbientCGRejectsInvalidPackageProofAndMapContractWithoutPublishing() async throws {
        let height = try fixturePNG(width: 32, height: 33)
        let valid = try ambientZIP([("Wicker013_1K-PNG_Displacement.png", height)], deflated: true)
        let crcInvalid = try ambientZIP([("Wicker013_1K-PNG_Displacement.png", height)], deflated: true, incorrectCRC: true)
        let lowPrecision = try ambientZIP([("Wicker013_1K-PNG_Displacement.png", try fixturePNG(width: 32, height: 33, bits: 8))])
        let unsafe = try ambientZIP([("../Wicker013_1K-PNG_Displacement.png", height)])
        for (name, archive, length, digest, requestedWidth, expectedSHA) in [
            ("member CRC", crcInvalid, Optional<Int>.none, Optional<String>.none, 32, Optional<String>.none),
            ("package length", valid, valid.count + 1, nil, 32, nil),
            ("published package SHA1", valid, nil, String(repeating: "0", count: 40), 32, nil),
            ("native displacement precision", lowPrecision, nil, nil, 32, nil),
            ("unsafe archive member", unsafe, nil, nil, 32, nil),
            ("paired dimensions", valid, nil, nil, 31, nil),
            ("historical source SHA256", valid, nil, nil, 32, String(repeating: "0", count: 64))
        ] {
            let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let destination = root.appendingPathComponent("height.png")
            let recovery = NativeMaterialSourceRecovery(transport: try ambientTransport(archive, length: length, publishedSHA1: digest))
            let result = try await recovery.recover(ambientRequest(destination, target: "height", width: requestedWidth, expectedSHA256: expectedSHA))
            guard case .failed = result else { XCTFail("Reject \(name)"); continue }
            XCTAssertEqual(try filenames(root), [], "Rejected \(name) must not leave partial sources or downloads")
        }
    }

    func testAmbientCGUnpublishedTargetKeepsAvailableColorAndNormalUsable() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let color = try rgbFixturePNG(width: 32, height: 33), normal = try rgbFixturePNG(width: 32, height: 33)
        let archive = try ambientZIP([("Wicker013_1K-PNG_Color.png", color),
                                     ("Wicker013_1K-PNG_NormalGL.png", normal),
                                     ("Wicker013_1K-PNG_NormalDX.png", color)])
        let probe = RecoveryTransportProbe(), recovery = NativeMaterialSourceRecovery(transport: try ambientTransport(archive, probe: probe))
        let missing = try await recovery.recover(ambientRequest(root.appendingPathComponent("height.png"), target: "height"))
        guard case .notPublished = missing else { return XCTFail("A missing displacement member must be reported as unpublished") }
        for target in ["input", "normal", "normal_dx"] {
            let destination = root.appendingPathComponent(target + ".png")
            let result = try await recovery.recover(ambientRequest(destination, target: target))
            guard case .recovered(let proof) = result else { XCTFail("Keep \(target) usable"); continue }
            XCTAssertEqual(try Data(contentsOf: destination), color)
            XCTAssertEqual(proof.archiveMember, "Wicker013_1K-PNG_\(["input": "Color", "normal": "NormalGL", "normal_dx": "NormalDX"][target]!).png")
        }
        let calls = await probe.counts(); XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("height.png").path))
    }

    func testNewTiersPreserveFullProviderDimensionsInsteadOfCroppingToNominalSquare() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let bytes = try fixturePNG(width: 32, height: 40), archive = try ambientZIP([("Wicker013_1K-PNG_Displacement.png", bytes)])
        let recovery = NativeMaterialSourceRecovery(transport: try ambientTransport(archive))
        let destination = root.appendingPathComponent("ambient-height.png")
        let result = try await recovery.recover(.init(asset: "Wicker013", resolution: "1k", target: "height",
            destination: destination, width: 32, height: 32, provider: .ambientCG, preserveOriginalDimensions: true))
        guard case .recovered(let proof) = result else { return XCTFail("New source tiers must preserve complete provider fields") }
        XCTAssertEqual(try Data(contentsOf: destination), bytes)
        XCTAssertEqual(proof.sourceHeight, 40); XCTAssertNil(proof.cropRectangle)
        let polyBytes = try fixturePNG(width: 256, height: 300)
        let poly = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: polyBytes))
        let polyDestination = root.appendingPathComponent("poly-height.png")
        let polyResult = try await poly.recover(.init(asset: "ribbed_corduroy", resolution: "1k", target: "height",
            destination: polyDestination, width: 256, height: 256, preserveOriginalDimensions: true))
        guard case .recovered(let polyProof) = polyResult else { return XCTFail("Both providers preserve full fields for new tiers") }
        XCTAssertEqual(try Data(contentsOf: polyDestination), polyBytes)
        XCTAssertEqual(polyProof.sourceHeight, 300); XCTAssertNil(polyProof.cropRectangle)
    }

    func testAmbientCGCancellationCleansArchiveStageAndLeavesExistingFilesUnchanged() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("height.png"), original = Data("existing original".utf8)
        try original.write(to: destination)
        let archive = try ambientZIP([("Wicker013_1K-PNG_Displacement.png", try fixturePNG(width: 32, height: 33))])
        let probe = RecoveryTransportProbe(), recovery = NativeMaterialSourceRecovery(transport: try ambientTransport(archive, probe: probe))
        let existing = try await recovery.recover(ambientRequest(destination, target: "height"))
        guard case .alreadyPresent = existing else { return XCTFail("Existing originals remain authoritative") }
        XCTAssertEqual(try Data(contentsOf: destination), original)
        let calls = await probe.counts(); XCTAssertEqual(calls.metadata, 0); XCTAssertEqual(calls.downloads, 0)
        let metadata = try ambientMetadata(archive.count)
        let cancellation = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            .init(data: metadata, statusCode: 200, url: url)
        }, download: { url, staged in
            try archive.write(to: staged); throw CancellationError()
        }))
        do {
            _ = try await cancellation.recover(ambientRequest(root.appendingPathComponent("new-height.png"), target: "height"))
            XCTFail("Cancellation must propagate")
        } catch is CancellationError {}
        XCTAssertEqual(try filenames(root), ["height.png"])
    }

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

    func testExplicitRecoveryAddsPhysicalMapAndDeletedOriginalRequiresExplicitRestore() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let input = sources.appendingPathComponent("ribbed_corduroy_diff_1k.png")
        let normal = sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png")
        let height = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let rgb = try rgbFixturePNG(), numeric = try fixturePNG(width: 1024, height: 1024)
        try rgb.write(to: input); try rgb.write(to: normal)
        let probe = RecoveryTransportProbe()
        let recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: numeric, probe: probe))
        let before = try await output(["dataset", "--dataset", sources.path, "--review-size", "256"], recovery: recovery)
        XCTAssertEqual(before["source_set_count"] as? Int, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: height.path), "Opening a folder cannot manufacture absent originals")
        var calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 0); XCTAssertEqual(calls.downloads, 0)
        let restore = NativeMaterialSourceRecovery.Request(asset: "ribbed_corduroy", resolution: "1k", target: "height",
            destination: height, width: 1024, height: 1024, expectedSHA256: sha256(numeric), preserveOriginalDimensions: true)
        guard case .recovered = try await recovery.recover(restore) else { return XCTFail("Explicit recovery must publish the verified physical original") }
        let first = try await output(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "height"], recovery: recovery)
        let firstURL = URL(fileURLWithPath: try XCTUnwrap(first["dataset_path"] as? String))
        defer { _ = try? NativeMaterialDatasetService.cleanup(firstURL) }
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: firstURL, size: 256, target: "height").count, 1)
        try FileManager.default.removeItem(at: height)
        _ = try await output(["dataset", "--dataset", sources.path, "--review-size", "256"], recovery: recovery)
        XCTAssertTrue(try NativeMaterialDatasetService.trainingSamples(datasetURL: firstURL, size: 256, target: "height").isEmpty,
            "Prepared evidence cannot retain a map deleted from the source folder")
        do {
            _ = try await output(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "height"], recovery: recovery)
            XCTFail("Height training requires a physical height map")
        } catch { XCTAssertTrue(error.localizedDescription.contains("No original source set supplies")) }
        calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1, "Inspection and preparation cannot silently undo a Finder deletion")
        XCTAssertFalse(FileManager.default.fileExists(atPath: height.path))
        guard case .recovered = try await recovery.recover(restore) else { return XCTFail("An explicit restore must require the recorded original checksum") }
        let prepared = try await output(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "height"], recovery: recovery)
        XCTAssertEqual(try Data(contentsOf: height), numeric)
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 1)
        let descriptor = try XCTUnwrap(descriptors.first)
        let crop = try XCTUnwrap(descriptor.cropRectangle)
        let target = try NativePNG.decode(Data(contentsOf: descriptor.targetURL))
        XCTAssertEqual(target.header.bits, 16)
        XCTAssertEqual(target.pixels, try NativePNG.decode(numeric).crop(crop).pixels)
        XCTAssertEqual(descriptor.augmentationSourceTargetSHA256, sha256(numeric))
        XCTAssertEqual(try Data(contentsOf: input), rgb); XCTAssertEqual(try Data(contentsOf: normal), rgb)
        calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 2)
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
    }

    func testProviderResolutionLabelWithOneExtraRowStillRecoversMissingTarget() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let rgb = try rgbFixturePNG(height: 1025), numeric = try fixturePNG(width: 1024, height: 1025)
        try rgb.write(to: sources.appendingPathComponent("ribbed_corduroy_diff_1k.png"))
        try rgb.write(to: sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png"))
        let height = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let probe = RecoveryTransportProbe(), recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: numeric, probe: probe))
        _ = try await output(["dataset", "--dataset", sources.path, "--review-size", "256"], recovery: recovery)
        let before = await probe.counts()
        XCTAssertEqual(before.metadata, 0); XCTAssertEqual(before.downloads, 0)
        let result = try await recovery.recover(.init(asset: "ribbed_corduroy", resolution: "1k", target: "height",
            destination: height, width: 1024, height: 1025, preserveOriginalDimensions: true))
        guard case .recovered(let proof) = result else { return XCTFail("Explicit recovery must preserve a legitimate provider field with an extra row") }
        XCTAssertEqual(try Data(contentsOf: height), numeric)
        XCTAssertEqual(proof.sourceHeight, 1025); XCTAssertNil(proof.cropRectangle)
        let material = try XCTUnwrap(NativeMaterialDatasetService.sourceFolderRecords(sources).first?.sample)
        XCTAssertEqual(material["source_pixel_dimensions"] as? [Int], [1024, 1025])
        let maps = try XCTUnwrap(material["maps"] as? [String: String])
        let heightPath = try XCTUnwrap(maps["height"])
        XCTAssertEqual(URL(fileURLWithPath: heightPath).resolvingSymlinksInPath().standardizedFileURL,
                       height.resolvingSymlinksInPath().standardizedFileURL)
        let prepared = try await output(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "height"], recovery: recovery)
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let sample = try XCTUnwrap(try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height").first)
        XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.targetURL)).pixels,
            try NativePNG.decode(numeric).crop(try XCTUnwrap(sample.cropRectangle)).pixels)
        XCTAssertEqual(try Data(contentsOf: height), numeric)
        XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent("ribbed_corduroy_diff_1k.png")), rgb)
        XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png")), rgb)
        let calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1)
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
    }

    func testExplicitRecoveryInsideSourceFolderAdvancesItsInventoryRevision() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Sources"), numeric = try fixturePNG(width: 1024, height: 1024)
        let probe = RecoveryTransportProbe(), recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: numeric, probe: probe))
        let sources = dataset.appendingPathComponent("sources/Subject")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: true)
        let rgb = try rgbFixturePNG(), roughness = try fixturePNG(width: 1024, height: 1024, bits: 8)
        try rgb.write(to: sources.appendingPathComponent("ribbed_corduroy_diff_1k.png"))
        try rgb.write(to: sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png"))
        try roughness.write(to: sources.appendingPathComponent("ribbed_corduroy_rough_1k.png"))
        let before = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"], recovery: recovery)
        XCTAssertEqual(before["material_count"] as? Int, 1)
        let height = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        XCTAssertFalse(FileManager.default.fileExists(atPath: height.path))
        guard case .recovered = try await recovery.recover(.init(asset: "ribbed_corduroy", resolution: "1k", target: "height",
            destination: height, width: 1024, height: 1024, preserveOriginalDimensions: true)) else { return XCTFail("Explicit recovery must publish into the selected physical folder") }
        let after = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"], recovery: recovery)
        let reopened = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"], recovery: recovery)
        XCTAssertNotEqual(after["index_sha256"] as? String, before["index_sha256"] as? String)
        XCTAssertEqual(after["index_sha256"] as? String, reopened["index_sha256"] as? String,
            "A recovered physical map and discovery tags must yield a stable inventory revision")
        XCTAssertEqual(after["source_set_count"] as? Int, 1, "A new target enriches the existing physical source set")
        let plans = try XCTUnwrap(after["training_plans"] as? [String: [String: Any]])
        XCTAssertEqual(plans["height"]?["train_count"] as? Int, 1)
        XCTAssertEqual(try Data(contentsOf: height), numeric)
        let calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1)
    }

    func testExplicitTargetRecoveryPreservesCurrentOriginalCurationAcrossPreparedCache() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let input = sources.appendingPathComponent("ribbed_corduroy_diff_1k.png")
        let normal = sources.appendingPathComponent("ribbed_corduroy_nor_gl_1k.png")
        let height = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
        let rgb = try rgbFixturePNG(), numeric = try fixturePNG(width: 1024, height: 1024)
        try rgb.write(to: input); try rgb.write(to: normal)
        let probe = RecoveryTransportProbe(), recovery = NativeMaterialSourceRecovery(transport: fixtureTransport(bytes: numeric, probe: probe))
        let imported = try await output(["dataset", "--dataset", sources.path, "--review-size", "256"], recovery: recovery)
        let importedMaterials = try XCTUnwrap(imported["original_materials"] as? [[String: Any]])
        let originalSamples = try XCTUnwrap(importedMaterials.first?["samples"] as? [[String: Any]])
        let indexedSampleID = try XCTUnwrap(originalSamples.first?["sample_id"] as? String)
        _ = try await output(["curate", "--dataset", sources.path, "--sample", indexedSampleID,
            "--original-source", "true", "--status", "approved", "--note", "Keep the reviewed originals"], recovery: recovery)
        let cached = try await output(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "normal"], recovery: recovery)
        let cachedURL = URL(fileURLWithPath: try XCTUnwrap(cached["dataset_path"] as? String))
        _ = try await output(["curate", "--dataset", sources.path, "--sample", indexedSampleID,
            "--original-source", "true", "--status", "approved", "--note", "Edited after cached normal preparation"], recovery: recovery)
        var calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 0); XCTAssertEqual(calls.downloads, 0)
        guard case .recovered = try await recovery.recover(.init(asset: "ribbed_corduroy", resolution: "1k", target: "height",
            destination: height, width: 1024, height: 1024, preserveOriginalDimensions: true)) else { return XCTFail("Explicit recovery must add the published target") }
        let prepared = try await output(["prepare-size", "--dataset", cachedURL.path, "--size", "256", "--target", "height",
            "--expected-index-sha256", try XCTUnwrap(cached["index_sha256"] as? String)], recovery: recovery)
        XCTAssertEqual(try Data(contentsOf: height), numeric)
        let inspected = try await output(["dataset", "--dataset", sources.path, "--review-size", "256"], recovery: recovery)
        XCTAssertEqual(inspected["source_set_count"] as? Int, 1, "Adding a physical target must enrich the existing source set")
        let material = try XCTUnwrap((inspected["original_materials"] as? [[String: Any]])?.first)
        let sample = try XCTUnwrap((material["samples"] as? [[String: Any]])?.first)
        XCTAssertEqual(sample["sample_id"] as? String, indexedSampleID)
        XCTAssertEqual(sample["status"] as? String, "approved")
        XCTAssertEqual(sample["note"] as? String, "Edited after cached normal preparation",
                       "An old prepared dataset must not overwrite current original reviews when a recovered map enriches the source binding")
        XCTAssertEqual(NativeSourceFileTags.read(input)?.reviewStatus, "approved")
        let maps = try XCTUnwrap(sample["maps"] as? [String: [String: Any]])
        XCTAssertNotNil(maps["height"])
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 1)
        XCTAssertEqual(descriptors.first?.augmentationSourceTargetSHA256, sha256(numeric))
        XCTAssertEqual(try Data(contentsOf: input), rgb); XCTAssertEqual(try Data(contentsOf: normal), rgb)
        calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 1); XCTAssertEqual(calls.downloads, 1)
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
        _ = try await output(["cleanup-size", "--dataset", cachedURL.path], recovery: recovery)
    }

    func testExplicitUnpublishedDisplacementLeavesPhysicalNormalAndRoughnessTrainingUsable() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let rgb = try rgbFixturePNG(), roughness = try fixturePNG(width: 1024, height: 1024, bits: 8)
        let height = try fixturePNG(width: 1024, height: 1024)
        for asset in ["fabric_pattern_05", "brick_complete"] {
            try rgb.write(to: sources.appendingPathComponent(asset + "_diff_1k.png"))
            try rgb.write(to: sources.appendingPathComponent(asset + "_nor_gl_1k.png"))
            try roughness.write(to: sources.appendingPathComponent(asset + "_rough_1k.png"))
        }
        try height.write(to: sources.appendingPathComponent("brick_complete_disp_1k.png"))
        let probe = RecoveryTransportProbe(), recovery = unpublishedRecovery(probe: probe)
        let absent = sources.appendingPathComponent("fabric_pattern_05_disp_1k.png")
        guard case .notPublished = try await recovery.recover(.init(asset: "fabric_pattern_05", resolution: "1k", target: "height",
            destination: absent, width: 1024, height: 1024, preserveOriginalDimensions: true)) else { return XCTFail("The explicit provider request must truthfully report an unpublished map") }
        let inspected = try await output(["dataset", "--dataset", sources.path, "--review-size", "256"], recovery: recovery)
        XCTAssertEqual(inspected["source_set_count"] as? Int, 2)
        let plans = try XCTUnwrap(inspected["training_plans"] as? [String: [String: Any]])
        let issues = try XCTUnwrap(plans["height"]?["source_issues"] as? [[String: Any]])
        XCTAssertEqual(issues.count, 1)
        XCTAssertEqual(issues.first?["material_id"] as? String, "fabric_pattern_05_1k")
        XCTAssertEqual(issues.first?["code"] as? String, "missing_target", "Folder membership reports physical maps rather than cached provider predictions")
        XCTAssertEqual(plans["height"]?["train_count"] as? Int, 1)
        XCTAssertEqual(plans["normal"]?["train_count"] as? Int, 2)
        XCTAssertEqual(plans["roughness"]?["train_count"] as? Int, 2)
        for (target, count) in [("height", 1), ("normal", 2), ("roughness", 2)] {
            let prepared = try await output(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", target], recovery: recovery)
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: target)
            XCTAssertEqual(samples.count, count, "A missing displacement cannot block the material's other physical targets")
            if target == "height" { XCTAssertEqual(samples.first?.augmentationSourceTargetSHA256, sha256(height)) }
            _ = try await output(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: absent.path))
        for asset in ["fabric_pattern_05", "brick_complete"] {
            XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent(asset + "_diff_1k.png")), rgb)
            XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent(asset + "_nor_gl_1k.png")), rgb)
            XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent(asset + "_rough_1k.png")), roughness)
        }
        XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent("brick_complete_disp_1k.png")), height)
        let calls = await probe.counts()
        XCTAssertEqual(calls.metadata, 1, "Folder inspection and training must not request provider metadata")
        XCTAssertEqual(calls.downloads, 0)
    }

    func testExplicitProviderMetadataFailureIsRetryableAndPhysicalTargetsRemainUsable() async throws {
        let rgb = try rgbFixturePNG(), roughness = try fixturePNG(width: 1024, height: 1024, bits: 8)
        let height = try fixturePNG(width: 1024, height: 1024)
        let statusCodes: [Int?] = [nil, 503]
        for statusCode in statusCodes {
            let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let sources = root.appendingPathComponent("Sources")
            try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
            for asset in ["ribbed_corduroy", "brick_complete"] {
                try rgb.write(to: sources.appendingPathComponent(asset + "_diff_1k.png"))
                try rgb.write(to: sources.appendingPathComponent(asset + "_nor_gl_1k.png"))
                try roughness.write(to: sources.appendingPathComponent(asset + "_rough_1k.png"))
            }
            try height.write(to: sources.appendingPathComponent("brick_complete_disp_1k.png"))
            let probe = RecoveryTransportProbe()
            let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
                await probe.recordMetadata()
                guard let statusCode else { throw URLError(.notConnectedToInternet) }
                return .init(data: Data("{}".utf8), statusCode: statusCode, url: url)
            }, download: { _, _ in
                XCTFail("Failed provider metadata must not start a map download")
                throw URLError(.unsupportedURL)
            }))
            let missing = sources.appendingPathComponent("ribbed_corduroy_disp_1k.png")
            let request = NativeMaterialSourceRecovery.Request(asset: "ribbed_corduroy", resolution: "1k", target: "height",
                destination: missing, width: 1024, height: 1024, preserveOriginalDimensions: true)
            for _ in 0..<2 {
                guard case .failed(let reason) = try await recovery.recover(request) else { return XCTFail("Offline or failed provider metadata does not prove the map is unpublished") }
                XCTAssertFalse(reason.isEmpty)
            }
            let inspected = try await output(["dataset", "--dataset", sources.path, "--review-size", "256"], recovery: recovery)
            XCTAssertEqual(inspected["source_set_count"] as? Int, 2)
            let plans = try XCTUnwrap(inspected["training_plans"] as? [String: [String: Any]])
            let issues = try XCTUnwrap(plans["height"]?["source_issues"] as? [[String: Any]])
            XCTAssertEqual(issues.count, 1)
            XCTAssertEqual(issues.first?["material_id"] as? String, "ribbed_corduroy_1k")
            XCTAssertEqual(issues.first?["code"] as? String, "missing_target")
            XCTAssertFalse((issues.first?["reason"] as? String ?? "").contains("does not publish"))
            XCTAssertEqual(plans["normal"]?["train_count"] as? Int, 2)
            XCTAssertEqual(plans["roughness"]?["train_count"] as? Int, 2)
            for (target, count) in [("normal", 2), ("roughness", 2), ("height", 1)] {
                let prepared = try await output(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", target], recovery: recovery)
                let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
                XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: target).count, count,
                               "A failed recovery must skip only its missing target")
                _ = try await output(["cleanup-size", "--dataset", preparedURL.path], recovery: recovery)
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: sources.appendingPathComponent("ribbed_corduroy_disp_1k.png").path))
            for asset in ["ribbed_corduroy", "brick_complete"] {
                XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent(asset + "_diff_1k.png")), rgb)
                XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent(asset + "_nor_gl_1k.png")), rgb)
                XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent(asset + "_rough_1k.png")), roughness)
            }
            XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent("brick_complete_disp_1k.png")), height)
            let calls = await probe.counts()
            XCTAssertEqual(calls.metadata, 2, "Explicit retries refetch failed metadata; folder reads and preparation make no further requests")
            XCTAssertEqual(calls.downloads, 0)
        }
    }

    private func ambientRequest(_ destination: URL, target: String, width: Int = 32,
                                expectedSHA256: String? = nil) -> NativeMaterialSourceRecovery.Request {
        .init(asset: "Wicker013", resolution: "1k", target: target, destination: destination,
              width: width, height: 33, expectedSHA256: expectedSHA256, provider: .ambientCG)
    }
    private func ambientMetadata(_ length: Int) throws -> Data {
        let download: [String: Any] = ["downloadLink": "https://ambientcg.com/get?file=Wicker013_1K-PNG.zip",
            "fileName": "Wicker013_1K-PNG.zip", "size": length, "filetype": "zip", "attribute": "1K-PNG"]
        let folder: [String: Any] = ["downloadFiletypeCategories": ["zip": ["downloads": [download]]]]
        let asset: [String: Any] = ["assetId": "Wicker013", "downloadFolders": ["default": folder]]
        return try JSONSerialization.data(withJSONObject: ["foundAssets": [asset]])
    }
    private func ambientTransport(_ archive: Data, length: Int? = nil, publishedSHA1: String? = nil,
                                  probe: RecoveryTransportProbe = RecoveryTransportProbe()) throws -> NativeMaterialSourceRecovery.Transport {
        let metadata = try ambientMetadata(length ?? archive.count), expected = publishedSHA1 ?? sha1(archive)
        return .init(fetchMetadata: { url in
            await probe.recordMetadata()
            return .init(data: metadata, statusCode: 200, url: url)
        }, download: { url, staged in
            await probe.recordDownload(staged)
            try archive.write(to: staged)
            let final = URL(string: "https://acg-download.struffelproductions.com/file/ambientCG-Web/download/Wicker013_fixture/Wicker013_1K-PNG.zip")!
            return .init(statusCode: 200, url: final, publishedSHA1: expected)
        })
    }
    private func ambientZIP(_ members: [(String, Data)], deflated: Bool = false, incorrectCRC: Bool = false) throws -> Data {
        var archive = Data(), directory = Data()
        func number<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }
        for (filename, bytes) in members {
            let name = Data(filename.utf8), offset = UInt32(archive.count), method: UInt16 = deflated ? 8 : 0
            var compressed = bytes
            if deflated {
                var stream = z_stream()
                guard deflateInit2_(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY,
                                    ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
                    throw NSError(domain: "ZIP fixture", code: 1)
                }
                defer { deflateEnd(&stream) }
                var storage = Data(count: Int(deflateBound(&stream, uLong(bytes.count))))
                let status = storage.withUnsafeMutableBytes { output in bytes.withUnsafeBytes { input in
                    stream.next_in = UnsafeMutablePointer(mutating: input.bindMemory(to: Bytef.self).baseAddress)
                    stream.avail_in = uInt(bytes.count)
                    stream.next_out = output.bindMemory(to: Bytef.self).baseAddress
                    stream.avail_out = uInt(output.count)
                    return deflate(&stream, Z_FINISH)
                } }
                guard status == Z_STREAM_END else { throw NSError(domain: "ZIP fixture", code: 2) }
                storage.count = Int(stream.total_out); compressed = storage
            }
            let checksum = crc(bytes) ^ (incorrectCRC ? 1 : 0)
            number(UInt32(0x04034b50), to: &archive); number(UInt16(20), to: &archive)
            number(UInt16(0), to: &archive); number(method, to: &archive)
            number(UInt32(0), to: &archive); number(checksum, to: &archive)
            number(UInt32(compressed.count), to: &archive); number(UInt32(bytes.count), to: &archive)
            number(UInt16(name.count), to: &archive); number(UInt16(0), to: &archive)
            archive.append(name); archive.append(compressed)
            number(UInt32(0x02014b50), to: &directory); number(UInt16(20), to: &directory); number(UInt16(20), to: &directory)
            number(UInt16(0), to: &directory); number(method, to: &directory)
            number(UInt32(0), to: &directory); number(checksum, to: &directory)
            number(UInt32(compressed.count), to: &directory); number(UInt32(bytes.count), to: &directory)
            number(UInt16(name.count), to: &directory)
            for _ in 0..<4 { number(UInt16(0), to: &directory) }
            number(UInt32(0), to: &directory); number(offset, to: &directory); directory.append(name)
        }
        let start = UInt32(archive.count); archive.append(directory)
        number(UInt32(0x06054b50), to: &archive); number(UInt32(0), to: &archive)
        number(UInt16(members.count), to: &archive); number(UInt16(members.count), to: &archive)
        number(UInt32(directory.count), to: &archive); number(start, to: &archive); number(UInt16(0), to: &archive)
        return archive
    }
    private func crc(_ bytes: Data) -> UInt32 {
        bytes.withUnsafeBytes { UInt32(crc32_z(0, $0.bindMemory(to: Bytef.self).baseAddress, bytes.count)) }
    }
    private func sha1(_ bytes: Data) -> String { Insecure.SHA1.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }

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
    private func unpublishedRecovery(probe: RecoveryTransportProbe = RecoveryTransportProbe()) -> NativeMaterialSourceRecovery {
        NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            await probe.recordMetadata()
            return .init(data: Data("{}".utf8), statusCode: 200, url: url)
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
    private func grayscaleFixture(width: Int, height: Int, bits: Int, channels: Int) -> NativePNG {
        NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels,
                               color: channels == 1 ? 0 : 4, interlace: 0),
                  pixels: Data((0..<width * height * channels * (bits / 8)).map { UInt8(truncatingIfNeeded: $0 * 97 + 3) }),
                  colorChunks: [("sRGB", Data([0]))])
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
    private func filenames(_ folder: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: folder.path)
            .filter { !$0.hasPrefix(".texture-studio-source-proof-") }.sorted()
    }
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
