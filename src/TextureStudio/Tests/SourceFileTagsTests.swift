import Darwin
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class SourceFileTagsTests: XCTestCase {
    func testUnchangedTagsAndVerificationPreserveOriginalBytesAndModificationTime() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("Original.png")
        let bytes = try png(channels: 3, bits: 8)
        try bytes.write(to: image)
        let before = try NativeSourceFileTags.identity(image)
        let sha = NativeMaterialTrainer.checksum(bytes)
        try NativeSourceFileTags.update(image, role: "input", classLabel: "Carpet", verifiedSHA256: sha)
        let tags = try XCTUnwrap(NativeSourceFileTags.read(image))
        XCTAssertEqual(tags.role, "input")
        XCTAssertEqual(tags.classLabel, "Carpet")
        XCTAssertEqual(tags.classOrigin, "user")
        XCTAssertTrue(tags.hasManualClass, "A legacy label without classifier provenance is manual teaching")
        XCTAssertEqual(tags.verifiedSHA256, sha)
        XCTAssertEqual(try NativeSourceFileTags.identity(image), before)
        XCTAssertEqual(try Data(contentsOf: image), bytes)
    }

    func testRewritingPixelsKeepsManualTeachingAndExpiresVerification() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for classifierIdentity in [nil, "user"] as [String?] {
            let image = root.appendingPathComponent("\(classifierIdentity ?? "legacy")-Original.png")
            let before = try png(channels: 3, bits: 8, code: 23)
            let after = try png(channels: 3, bits: 8, code: 87, width: 258)
            try before.write(to: image)
            try NativeSourceFileTags.update(image, role: "input", classLabel: "Wood", classifierIdentity: classifierIdentity,
                verifiedSHA256: NativeMaterialTrainer.checksum(before), reviewStatus: "approved",
                verifiedMetadata: "{\"width\":256}", pairingName: "oak_planks")
            let previous = try NativeSourceFileTags.identity(image), attribute = try tagBytes(image)
            try after.write(to: image)
            let currentIdentity = try NativeSourceFileTags.identity(image)
            XCTAssertNotEqual(currentIdentity, previous)
            let tags = try XCTUnwrap(NativeSourceFileTags.read(image))
            XCTAssertEqual(tags.identity, currentIdentity)
            XCTAssertEqual(tags.role, "input")
            XCTAssertEqual(tags.classLabel, "Wood")
            XCTAssertEqual(tags.classifierIdentity, classifierIdentity)
            XCTAssertEqual(tags.classOrigin, "user")
            XCTAssertTrue(tags.hasManualClass)
            XCTAssertEqual(tags.reviewStatus, "approved")
            XCTAssertEqual(tags.pairingName, "oak_planks")
            XCTAssertNil(tags.verifiedSHA256)
            XCTAssertNil(tags.verifiedMetadata)
            XCTAssertEqual(try tagBytes(image), attribute, "Reading changed pixels does not rewrite semantic teaching")
            XCTAssertEqual(try Data(contentsOf: image), after)

            try NativeSourceFileTags.update(image, reviewStatus: "excluded")
            let saved = try JSONDecoder().decode(NativeSourceFileTags.Tags.self, from: tagBytes(image))
            XCTAssertEqual(saved.identity, currentIdentity)
            XCTAssertEqual(saved.classLabel, "Wood")
            XCTAssertEqual(saved.classOrigin, "user")
            XCTAssertEqual(saved.role, "input")
            XCTAssertEqual(saved.pairingName, "oak_planks")
            XCTAssertEqual(saved.reviewStatus, "excluded")
            XCTAssertNil(saved.verifiedSHA256)
            XCTAssertNil(saved.verifiedMetadata)
        }
    }

    func testChangedPixelsKeepAssignedClassAndExpirePredictionEvidence() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("Original.png")
        let before = try png(channels: 3, bits: 8, code: 23)
        try before.write(to: image)
        try NativeSourceFileTags.update(image, role: "input", classLabel: "Wood", classifierIdentity: "classifier-sha256",
            verifiedSHA256: NativeMaterialTrainer.checksum(before), reviewStatus: "approved",
            verifiedMetadata: "{\"width\":256}", pairingName: "oak_planks")
        XCTAssertFalse(try XCTUnwrap(NativeSourceFileTags.read(image)).hasManualClass)
        try png(channels: 3, bits: 8, code: 87, width: 258).write(to: image)
        let tags = try XCTUnwrap(NativeSourceFileTags.read(image))
        XCTAssertEqual(tags.role, "input")
        XCTAssertEqual(tags.reviewStatus, "approved")
        XCTAssertEqual(tags.pairingName, "oak_planks")
        XCTAssertEqual(tags.classLabel, "Wood", "An unchanged assigned class remains semantic teaching after its cached prediction expires")
        XCTAssertEqual(tags.classOrigin, "classifier")
        XCTAssertFalse(tags.hasManualClass, "Expiring prediction evidence does not convert an assigned class into user teaching")
        XCTAssertNil(tags.classifierIdentity)
        XCTAssertNil(tags.verifiedSHA256)
        XCTAssertNil(tags.verifiedMetadata)
    }

    func testCopiedExtendedAttributesKeepTeachingWithoutReusingVerification() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("Original.png"), copy = root.appendingPathComponent("Copy.png")
        let bytes = try png(channels: 3, bits: 8)
        try bytes.write(to: original)
        try NativeSourceFileTags.update(original, role: "input", classLabel: "Fabric", verifiedSHA256: NativeMaterialTrainer.checksum(bytes),
            reviewStatus: "approved", verifiedMetadata: "{\"width\":256}", pairingName: "woven_fabric")
        try FileManager.default.copyItem(at: original, to: copy)
        XCTAssertEqual(try Data(contentsOf: copy), bytes)
        XCTAssertNotEqual(try NativeSourceFileTags.identity(original).inode, try NativeSourceFileTags.identity(copy).inode)
        XCTAssertGreaterThan(getxattr(copy.path, NativeSourceFileTags.attribute, nil, 0, 0, 0), 0, "The test copy retains the stale attribute to exercise its inode guard")
        let tags = try XCTUnwrap(NativeSourceFileTags.read(copy))
        XCTAssertEqual(tags.identity, try NativeSourceFileTags.identity(copy))
        XCTAssertEqual(tags.role, "input")
        XCTAssertEqual(tags.classLabel, "Fabric")
        XCTAssertEqual(tags.classOrigin, "user")
        XCTAssertEqual(tags.reviewStatus, "approved")
        XCTAssertEqual(tags.pairingName, "woven_fabric")
        XCTAssertNil(tags.verifiedSHA256)
        XCTAssertNil(tags.verifiedMetadata)
        XCTAssertEqual(NativeSourceFileTags.read(original)?.classLabel, "Fabric")
        XCTAssertEqual(NativeSourceFileTags.read(original)?.verifiedSHA256, NativeMaterialTrainer.checksum(bytes))
    }

    func testFilenameRenameKeepsTeachingPairingAndUnchangedPixelVerification() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("oak_diffuse.png"), renamed = root.appendingPathComponent("photo.png")
        let bytes = try png(channels: 3, bits: 8), sha = NativeMaterialTrainer.checksum(bytes)
        try bytes.write(to: original)
        try NativeSourceFileTags.update(original, role: "input", classLabel: "Wood", classifierIdentity: "user",
            verifiedSHA256: sha, reviewStatus: "approved", verifiedMetadata: "{\"width\":256}", pairingName: "oak")
        let identity = try NativeSourceFileTags.identity(original), attribute = try tagBytes(original)
        try FileManager.default.moveItem(at: original, to: renamed)
        let tags = try XCTUnwrap(NativeSourceFileTags.read(renamed))
        XCTAssertEqual(tags.identity, identity)
        XCTAssertEqual(tags.role, "input")
        XCTAssertEqual(tags.classLabel, "Wood")
        XCTAssertEqual(tags.classifierIdentity, "user")
        XCTAssertEqual(tags.classOrigin, "user")
        XCTAssertEqual(tags.pairingName, "oak")
        XCTAssertEqual(tags.reviewStatus, "approved")
        XCTAssertEqual(tags.verifiedSHA256, sha)
        XCTAssertEqual(tags.verifiedMetadata, "{\"width\":256}")
        XCTAssertEqual(try tagBytes(renamed), attribute)
        XCTAssertEqual(try Data(contentsOf: renamed), bytes)
        XCTAssertNil(NativeSourceFileTags.read(original))
    }

    func testExplicitLabelWithoutClassifierIdentityReplacesMachinePredictionWithManualTeaching() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("Original.png")
        try png(channels: 3, bits: 8).write(to: image)
        try NativeSourceFileTags.update(image, classLabel: "Fabric", classifierIdentity: "classifier-sha256")
        try NativeSourceFileTags.update(image, classLabel: "Handmade Finish")
        XCTAssertNil(NativeSourceFileTags.read(image)?.classifierIdentity)
        XCTAssertEqual(NativeSourceFileTags.read(image)?.classOrigin, "user")
        try png(channels: 3, bits: 8, width: 258).write(to: image)
        XCTAssertEqual(NativeSourceFileTags.read(image)?.classLabel, "Handmade Finish")
    }

    func testLegacyClassTagsInferOriginWithoutLosingAssignedLabels() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for classifierIdentity in [nil, "user", "classifier-sha256"] as [String?] {
            let image = root.appendingPathComponent("\(classifierIdentity ?? "legacy")-Original.png")
            try png(channels: 3, bits: 8).write(to: image)
            let legacy = NativeSourceFileTags.Tags(identity: try NativeSourceFileTags.identity(image),
                role: "input", classLabel: "Wood", classifierIdentity: classifierIdentity)
            let data = try JSONEncoder().encode(legacy)
            XCTAssertNil((try JSONSerialization.jsonObject(with: data) as? [String: Any])?["classOrigin"])
            let result = data.withUnsafeBytes {
                setxattr(image.path, NativeSourceFileTags.attribute, $0.baseAddress, data.count, 0, 0)
            }
            guard result == 0 else { throw POSIXError(.EIO) }
            let isManual = classifierIdentity == nil || classifierIdentity == "user"
            let unchanged = try XCTUnwrap(NativeSourceFileTags.read(image))
            XCTAssertEqual(unchanged.hasManualClass, isManual)
            XCTAssertNil(unchanged.classOrigin)
            XCTAssertNil(unchanged.pairingName, "Legacy tag data decodes without a pairing field")

            try png(channels: 3, bits: 8, width: 258).write(to: image)
            let changed = try XCTUnwrap(NativeSourceFileTags.read(image))
            XCTAssertEqual(changed.classLabel, "Wood")
            XCTAssertEqual(changed.classOrigin, isManual ? "user" : "classifier")
            XCTAssertEqual(changed.hasManualClass, isManual)
            XCTAssertEqual(changed.classifierIdentity, classifierIdentity == "user" ? "user" : nil)
        }
    }

    func testRestoringAssignedClassAfterCopyPreservesItsSemanticOrigin() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("Original.png"), copy = root.appendingPathComponent("Copy.png")
        let bytes = try png(channels: 3, bits: 8)
        try bytes.write(to: original)
        try NativeSourceFileTags.update(original, role: "input", classLabel: "Wood", classifierIdentity: "classifier-sha256",
            reviewStatus: "approved", pairingName: "oak")
        try FileManager.default.copyItem(at: original, to: copy)
        let copied = try XCTUnwrap(NativeSourceFileTags.read(copy))
        XCTAssertEqual(copied.classLabel, "Wood")
        XCTAssertEqual(copied.classOrigin, "classifier")
        XCTAssertNil(copied.classifierIdentity)
        XCTAssertFalse(copied.hasManualClass)

        let restored = root.appendingPathComponent("Restored.png")
        try bytes.write(to: restored)
        try NativeSourceFileTags.update(restored, role: copied.role, classLabel: copied.classLabel,
            classifierIdentity: copied.classifierIdentity, reviewStatus: copied.reviewStatus,
            pairingName: copied.pairingName, classOrigin: copied.classOrigin)
        let tags = try XCTUnwrap(NativeSourceFileTags.read(restored))
        XCTAssertEqual(tags.role, "input")
        XCTAssertEqual(tags.classLabel, "Wood")
        XCTAssertEqual(tags.classOrigin, "classifier")
        XCTAssertEqual(tags.reviewStatus, "approved")
        XCTAssertEqual(tags.pairingName, "oak")
        XCTAssertFalse(tags.hasManualClass)
    }

    func testExplicitSameSizePairsKeepUniqueFamiliesAfterFinderRenamesAndMoves() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        let inputA = source.appendingPathComponent("Image101.png"), inputB = source.appendingPathComponent("Image201.png")
        let heightA = source.appendingPathComponent("Image102.png"), heightB = source.appendingPathComponent("Image202.png")
        let normalA = source.appendingPathComponent("Image103.png")
        let files = [(inputA, try png(channels: 3, bits: 8, code: 23)),
                     (heightA, try png(channels: 1, bits: 16)),
                     (normalA, try png(channels: 3, bits: 8, code: 127)),
                     (inputB, try png(channels: 3, bits: 8, code: 87)),
                     (heightB, try png(channels: 1, bits: 16))]
        for (file, bytes) in files { try bytes.write(to: file) }
        _ = try await command(["add-material", "--dataset", source.path, "--name", "Manual Pair",
            "--input", inputA.path, "--height", heightA.path, "--normal", normalA.path, "--normal-convention", "directx"])
        _ = try await command(["add-material", "--dataset", source.path, "--name", "Manual Pair",
            "--input", inputB.path, "--height", heightB.path])
        try NativeSourceFileTags.update(inputA, classLabel: "Wood", classifierIdentity: "user", reviewStatus: "approved")
        try NativeSourceFileTags.update(inputB, classLabel: "Wood", classifierIdentity: "user", reviewStatus: "approved")
        let familyA = try XCTUnwrap(NativeMaterialDatasetService.sourceMapFilename(try XCTUnwrap(NativeSourceFileTags.read(inputA)?.pairingName))?.0)
        let familyB = try XCTUnwrap(NativeMaterialDatasetService.sourceMapFilename(try XCTUnwrap(NativeSourceFileTags.read(inputB)?.pairingName))?.0)
        XCTAssertNotEqual(familyA, familyB, "Equal display names and pixel sizes cannot merge distinct explicit pairs")
        XCTAssertEqual(NativeMaterialDatasetService.sourceMapFilename(try XCTUnwrap(NativeSourceFileTags.read(normalA)?.pairingName))?.1, "nor_dx")

        var renamed: [URL: URL] = [:]
        for (index, item) in files.enumerated() {
            let destination = source.appendingPathComponent("Renamed \(index).png")
            try FileManager.default.moveItem(at: item.0, to: destination)
            renamed[item.0] = destination
        }
        let records = try NativeMaterialDatasetService.sourceFolderRecords(source)
        XCTAssertEqual(records.count, 2)
        let samples = Dictionary(records.map { ($0.sample["source_family_id"] as! String, $0.sample) }, uniquingKeysWith: { first, _ in first })
        XCTAssertEqual(Set(samples.keys), [familyA, familyB])
        let mapsA = try XCTUnwrap(samples[familyA]?["maps"] as? [String: String])
        let mapsB = try XCTUnwrap(samples[familyB]?["maps"] as? [String: String])
        XCTAssertEqual(mapsA["input"], renamed[inputA]?.path)
        XCTAssertEqual(mapsA["height"], renamed[heightA]?.path)
        XCTAssertEqual(mapsA["normal"], renamed[normalA]?.path)
        XCTAssertEqual(mapsB["input"], renamed[inputB]?.path)
        XCTAssertEqual(mapsB["height"], renamed[heightB]?.path)
        XCTAssertEqual(((samples[familyA]?["map_metadata"] as? [String: [String: Any]])?["normal"]?["source"] as? [String: Any])?["suffix"] as? String, "nor_dx")
        for (file, bytes) in files {
            let current = try XCTUnwrap(renamed[file])
            XCTAssertEqual(try Data(contentsOf: current), bytes)
            XCTAssertNil(((samples[file == inputB || file == heightB ? familyB : familyA]?["map_metadata"] as? [String: [String: Any]])?[file == inputA || file == inputB ? "input" : file == normalA ? "normal" : "height"]?["source"] as? [String: Any])?["provider"],
                "A manual pairing alias is not a claim of provider provenance")
        }
        XCTAssertEqual(NativeSourceFileTags.read(try XCTUnwrap(renamed[inputA]))?.classLabel, "Wood")
        XCTAssertEqual(NativeSourceFileTags.read(try XCTUnwrap(renamed[inputA]))?.reviewStatus, "approved")
        try FileManager.default.moveItem(at: try XCTUnwrap(renamed[inputA]), to: root.appendingPathComponent("Moved Photo.png"))
        let moved = try NativeMaterialDatasetService.sourceFolderRecords(source)
        XCTAssertEqual(moved.count, 1)
        XCTAssertEqual(moved.first?.sample["source_family_id"] as? String, familyB)
        try FileManager.default.removeItem(at: try XCTUnwrap(renamed[heightB]))
        XCTAssertTrue(try NativeMaterialDatasetService.sourceFolderRecords(source).isEmpty,
            "Remembered pairing tags cannot manufacture maps moved or deleted from the folder")
    }

    func testAddRenameAndMoveMaterialFoldersChangeMembershipWithoutChangingImages() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), outside = root.appendingPathComponent("Original material"), moved = root.appendingPathComponent("Moved")
        for folder in [source, outside, moved] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false) }
        let inputBytes = try png(channels: 3, bits: 8), depthBytes = try png(channels: 1, bits: 16)
        try inputBytes.write(to: outside.appendingPathComponent("BaseColor.png"))
        try depthBytes.write(to: outside.appendingPathComponent("Height.png"))
        let added = try await command(["add-source", "--dataset", source.path, "--source", outside.path])
        XCTAssertEqual((added["original_materials"] as? [Any])?.count, 1)
        let copied = source.appendingPathComponent(outside.lastPathComponent)
        XCTAssertEqual(try Data(contentsOf: copied.appendingPathComponent("BaseColor.png")), inputBytes)
        XCTAssertEqual(try Data(contentsOf: copied.appendingPathComponent("Height.png")), depthBytes)
        XCTAssertTrue(FileManager.default.fileExists(atPath: outside.path), "Add copies by default and preserves the external original")
        _ = try await command(["rename-source", "--dataset", source.path, "--source", copied.path, "--name", "Carpet"])
        let renamed = source.appendingPathComponent("Carpet")
        XCTAssertFalse(FileManager.default.fileExists(atPath: copied.path))
        XCTAssertEqual(try Data(contentsOf: renamed.appendingPathComponent("Height.png")), depthBytes)
        let removed = try await command(["move-source", "--dataset", source.path, "--source", renamed.path, "--destination", moved.path])
        XCTAssertEqual((removed["original_materials"] as? [Any])?.count, 0)
        XCTAssertEqual((removed["source_inventory"] as? [Any])?.count, 0)
        XCTAssertEqual(try Data(contentsOf: moved.appendingPathComponent("Carpet/Height.png")), depthBytes)
        XCTAssertEqual(try Data(contentsOf: outside.appendingPathComponent("BaseColor.png")), inputBytes)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: source.path), [])
    }

    func testAddingCopiedFolderRetainsTeachingWithNewIdentityAndNoVerificationCache() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), original = root.appendingPathComponent("Tagged Material")
        let nested = original.appendingPathComponent("Nested")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        let input = nested.appendingPathComponent("Image123.png"), height = nested.appendingPathComponent("Image124.png")
        let inputBytes = try png(channels: 3, bits: 8), heightBytes = try png(channels: 1, bits: 16)
        try inputBytes.write(to: input); try heightBytes.write(to: height)
        for (file, role, bytes) in [(input, "input", inputBytes), (height, "height", heightBytes)] {
            let sha = NativeMaterialTrainer.checksum(bytes)
            try NativeSourceFileTags.update(file, role: role, classLabel: "Fabric", classifierIdentity: "user",
                verifiedSHA256: sha, reviewStatus: "approved", verifiedMetadata: "{\"file_sha256\":\"" + sha + "\"}",
                pairingName: "woven_fabric")
        }
        let inputIdentity = try NativeSourceFileTags.identity(input), heightIdentity = try NativeSourceFileTags.identity(height)
        _ = try await command(["add-source", "--dataset", source.path, "--source", original.path, "--defer-refresh", "true"])
        let copied = source.appendingPathComponent("Tagged Material/Nested")
        for (file, copiedFile, role, bytes) in [(input, copied.appendingPathComponent(input.lastPathComponent), "input", inputBytes),
                                               (height, copied.appendingPathComponent(height.lastPathComponent), "height", heightBytes)] {
            let tags = try XCTUnwrap(NativeSourceFileTags.read(copiedFile))
            XCTAssertEqual(tags.identity, try NativeSourceFileTags.identity(copiedFile))
            XCTAssertNotEqual(tags.identity.inode, try NativeSourceFileTags.identity(file).inode)
            XCTAssertEqual(tags.role, role)
            XCTAssertEqual(tags.classLabel, "Fabric")
            XCTAssertEqual(tags.classifierIdentity, "user")
            XCTAssertEqual(tags.reviewStatus, "approved")
            XCTAssertEqual(tags.pairingName, "woven_fabric")
            XCTAssertNil(tags.verifiedSHA256)
            XCTAssertNil(tags.verifiedMetadata)
            XCTAssertEqual(try Data(contentsOf: copiedFile), bytes)
            XCTAssertEqual(try Data(contentsOf: file), bytes)
            XCTAssertEqual(NativeSourceFileTags.read(file)?.verifiedSHA256, NativeMaterialTrainer.checksum(bytes))
        }
        XCTAssertEqual(try NativeSourceFileTags.identity(input), inputIdentity)
        XCTAssertEqual(try NativeSourceFileTags.identity(height), heightIdentity)
    }

    func testAddPhotoRejectsSymlinkDestinationWithoutWritingOutsideSourceFolder() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), outside = root.appendingPathComponent("Outside")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: false)
        try FileManager.default.createSymbolicLink(at: source.appendingPathComponent("Carpet"), withDestinationURL: outside)
        let original = root.appendingPathComponent("photo.png"), bytes = try png(channels: 3, bits: 8)
        try bytes.write(to: original)
        do {
            _ = try await command(["add-source", "--dataset", source.path, "--source", original.path, "--folder-name", "Carpet"])
            XCTFail("A symbolic-link destination must not copy images outside the open folder")
        } catch { XCTAssertTrue(error.localizedDescription.contains("inside the open source folder")) }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: outside.path), [])
        XCTAssertEqual(try Data(contentsOf: original), bytes)
        XCTAssertNil(NativeSourceFileTags.read(original))
    }

    func testNestedLegacyManifestDoesNotHideItsOriginalImagesFromFolderMembership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), nested = source.appendingPathComponent("Legacy")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        let input = nested.appendingPathComponent("BaseColor.png"), height = nested.appendingPathComponent("Height.png")
        let inputBytes = try png(channels: 3, bits: 8), heightBytes = try png(channels: 1, bits: 16)
        let manifest = nested.appendingPathComponent("dataset.json"), manifestBytes = Data("{}".utf8)
        try inputBytes.write(to: input); try heightBytes.write(to: height); try manifestBytes.write(to: manifest)
        let result = try await command(["dataset", "--dataset", source.path])
        let inventory = try XCTUnwrap(result["source_inventory"] as? [[String: Any]])
        XCTAssertEqual(Set(inventory.compactMap { $0["path"] as? String }), [input.path, height.path])
        XCTAssertEqual((result["original_materials"] as? [Any])?.count, 1)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
        XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        XCTAssertEqual(try Data(contentsOf: manifest), manifestBytes)
    }

    func testManualAssignmentRefreshesReplacedDimensionsAndHonorsChangedRole() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), material = source.appendingPathComponent("Material")
        try FileManager.default.createDirectory(at: material, withIntermediateDirectories: true)
        let input = material.appendingPathComponent("Image123.png"), height = material.appendingPathComponent("Image124.png")
        try png(channels: 3, bits: 8).write(to: input)
        try png(channels: 1, bits: 16).write(to: height)
        _ = try await command(["add-material", "--dataset", source.path, "--name", "Manual Pair", "--input", input.path, "--height", height.path])
        let replacement = try png(channels: 3, bits: 8, width: 258, height: 256)
        try replacement.write(to: input, options: .atomic)
        let updated = try await command(["dataset", "--dataset", source.path])
        let updatedInventory = try XCTUnwrap(updated["source_inventory"] as? [[String: Any]])
        XCTAssertEqual(updatedInventory.first { $0["path"] as? String == input.path }?["width"] as? Int, 258)
        XCTAssertEqual((updated["original_materials"] as? [Any])?.count, 1)
        _ = try await command(["tag-source", "--dataset", source.path, "--source", input.path, "--role", "roughness", "--defer-refresh", "true"])
        let relabeled = try await command(["dataset", "--dataset", source.path])
        let inventory = try XCTUnwrap(relabeled["source_inventory"] as? [[String: Any]])
        XCTAssertEqual(inventory.first { $0["path"] as? String == input.path }?["role"] as? String, "roughness")
        XCTAssertEqual((relabeled["original_materials"] as? [Any])?.count, 0)
        XCTAssertTrue((relabeled["warnings"] as? [String] ?? []).contains { $0.contains("Manual map assignment needs review") })
        XCTAssertEqual(try Data(contentsOf: input), replacement)
    }

    func testSourceMutationsRejectRootRenamesMovesAndExternalTags() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), destination = root.appendingPathComponent("Outside")
        for folder in [source, destination] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false) }
        let external = destination.appendingPathComponent("photo.png")
        let bytes = try png(channels: 3, bits: 8)
        try bytes.write(to: external)
        for arguments in [
            ["rename-source", "--dataset", source.path, "--source", source.path, "--name", "Renamed"],
            ["move-source", "--dataset", source.path, "--source", source.path, "--destination", destination.path],
            ["tag-source", "--dataset", source.path, "--source", external.path, "--role", "input", "--class-label", "Wood"]
        ] {
            do { _ = try await command(arguments); XCTFail("Out-of-scope source mutation unexpectedly succeeded") }
            catch { XCTAssertTrue(error.localizedDescription.contains("inside the open source folder")) }
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
        XCTAssertEqual(try Data(contentsOf: external), bytes)
        XCTAssertNil(NativeSourceFileTags.read(external))
    }

    func testExplicitImageRolesPairArbitraryFilenamesAndTagChangesRemainTrainable() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), material = source.appendingPathComponent("Material")
        try FileManager.default.createDirectory(at: material, withIntermediateDirectories: true)
        let input = material.appendingPathComponent("Image123.png"), depth = material.appendingPathComponent("Image124.png")
        let inputBytes = try png(channels: 3, bits: 8), depthBytes = try png(channels: 1, bits: 16)
        try inputBytes.write(to: input); try depthBytes.write(to: depth)
        let before = try await command(["dataset", "--dataset", source.path, "--review-size", "256"])
        XCTAssertEqual((before["original_materials"] as? [Any])?.count, 0)
        _ = try await command(["tag-source", "--dataset", source.path, "--source", input.path, "--role", "input", "--class-label", "Fabric"])
        let paired = try await command(["tag-source", "--dataset", source.path, "--source", depth.path, "--role", "height", "--class-label", "Fabric"])
        XCTAssertEqual((paired["original_materials"] as? [Any])?.count, 1)
        let inventory = try XCTUnwrap(paired["source_inventory"] as? [[String: Any]])
        XCTAssertTrue(inventory.allSatisfy { $0["source_verification"] as? String == "header_inspected" })
        XCTAssertTrue(inventory.allSatisfy { $0["class_label"] as? String == "Fabric" })
        let inputIdentity = try NativeSourceFileTags.identity(input), depthIdentity = try NativeSourceFileTags.identity(depth)
        _ = try await command(["tag-source", "--dataset", source.path, "--source", input.path, "--role", "input", "--class-label", "Carpet"])
        XCTAssertEqual(try NativeSourceFileTags.identity(input), inputIdentity)
        XCTAssertEqual(try NativeSourceFileTags.identity(depth), depthIdentity)
        let preparedInfo = try await command(["prepare-size", "--dataset", source.path, "--size", "256", "--target", "height"])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(preparedInfo["dataset_path"] as? String))
        defer { _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertFalse(samples.isEmpty, "Extended-attribute changes cannot invalidate unchanged pending original pixels")
        XCTAssertEqual(NativeSourceFileTags.read(input)?.verifiedSHA256, NativeMaterialTrainer.checksum(inputBytes))
        XCTAssertEqual(NativeSourceFileTags.read(depth)?.verifiedSHA256, NativeMaterialTrainer.checksum(depthBytes))
        for sample in samples {
            let png = try NativePNG.decode(Data(contentsOf: sample.targetURL))
            XCTAssertEqual(png.header.bits, 16)
            XCTAssertEqual(png.pixels.prefix(4), Data([0, 0, 0, 1]))
        }
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
        XCTAssertEqual(try Data(contentsOf: depth), depthBytes)
    }

    func testLiveReviewTagsOverrideSavedFolderTeachingAndPreparedCache() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), material = source.appendingPathComponent("Material")
        try FileManager.default.createDirectory(at: material, withIntermediateDirectories: true)
        let input = material.appendingPathComponent("BaseColor.png"), depth = material.appendingPathComponent("Height.png")
        try png(channels: 3, bits: 8).write(to: input); try png(channels: 1, bits: 16).write(to: depth)
        let opened = try await command(["dataset", "--dataset", source.path, "--review-size", "256"])
        let originals = (opened["original_materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        let sampleID = try XCTUnwrap(originals.first?["sample_id"] as? String)
        _ = try await command(["curate", "--dataset", source.path, "--sample", sampleID,
            "--original-source", "true", "--status", "excluded"])
        try NativeSourceFileTags.update(input, reviewStatus: "approved")
        try NativeSourceFileTags.update(depth, reviewStatus: "approved")
        let preparedInfo = try await command(["prepare-size", "--dataset", source.path, "--size", "256", "--target", "height"])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(preparedInfo["dataset_path"] as? String))
        defer { _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        XCTAssertFalse(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").isEmpty)
        try NativeSourceFileTags.update(input, reviewStatus: "excluded")
        let refreshed = try await command(["dataset", "--dataset", source.path, "--review-size", "256"])
        let current = (refreshed["original_materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertEqual(current.first?["status"] as? String, "excluded")
        try NativeSourceFileTags.update(input, reviewStatus: "approved")
        let included = try await command(["dataset", "--dataset", source.path, "--review-size", "256"])
        let currentIncluded = (included["original_materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertEqual(currentIncluded.first?["status"] as? String, "approved", "Live file teaching remains authoritative after cached verification")
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("source-file-tags-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root.resolvingSymlinksInPath().standardizedFileURL
    }
    private func tagBytes(_ image: URL) throws -> Data {
        let size = getxattr(image.path, NativeSourceFileTags.attribute, nil, 0, 0, 0)
        guard size > 0 else { throw POSIXError(.ENOATTR) }
        var data = Data(count: size)
        let count = data.withUnsafeMutableBytes {
            getxattr(image.path, NativeSourceFileTags.attribute, $0.baseAddress, size, 0, 0)
        }
        guard count == size else { throw POSIXError(.EIO) }
        return data
    }
    private func png(channels: Int, bits: Int, code: UInt8 = 97, width: Int = 256, height: Int = 256) throws -> Data {
        var pixels = Data(repeating: code, count: width * height * channels * bits / 8)
        if bits == 16 {
            for value in 0..<(width * height) {
                pixels[value * 2] = UInt8((value >> 8) & 255)
                pixels[value * 2 + 1] = UInt8(value & 255)
            }
        }
        return try NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels,
                                           color: channels == 3 ? 2 : 0, interlace: 0), pixels: pixels, colorChunks: []).encoded()
    }
    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let text = try await NativeMaterialDatasetService.run(arguments: arguments + ["--skip-source-recovery"])
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(text).utf8)) as? [String: Any])
    }
}
