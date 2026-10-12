import CryptoKit
import CreateML
import Foundation
import XCTest
@testable import TextureStudio

final class NativeDiffuseClassifierDatasetTests: XCTestCase {
    func testCreateMLLoadsPopulatedTrainingViewWith120ImagesAndEmptyReservedCurationFolders() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Curated")
        let families = ["red_brick", "concrete", "pine_bark", "wood_planks", "snow", "forest_leaves", "gravel",
            "metal_plate", "marble_rock", "floor_tiles", "denim_fabric", "white_stucco", "rubber"]
        var sources: [NativeDiffuseClassifierDatasetService.Source] = []
        for (familyIndex, family) in families.enumerated() {
            for imageIndex in 0..<(familyIndex == families.count - 1 ? 12 : 9) {
                let name = family + "_" + String(imageIndex), url = root.appendingPathComponent(name + "_diff_1k.png")
                try png(width: 12, height: 13, value: UInt8(familyIndex * 10 + imageIndex)).write(to: url)
                sources.append(.init(family: name, variant: "color_1", url: url, width: 12, height: 13, metadata: nil))
            }
        }
        let result = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination))
        XCTAssertEqual(result["total"] as? Int, 120); XCTAssertEqual(result["training_count"] as? Int, 120)
        XCTAssertEqual(result["training_class_count"] as? Int, 13); XCTAssertEqual(result["training_ready"] as? Bool, true)
        let training = URL(fileURLWithPath: try XCTUnwrap(result["training_path"] as? String))
        XCTAssertEqual(training.path, destination.path + "-createml")
        for name in ["Other Diffuse Textures", "Other Fallback"] {
            XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: destination.appendingPathComponent(name).path).isEmpty)
            XCTAssertFalse(FileManager.default.fileExists(atPath: training.appendingPathComponent(name).path))
        }
        XCTAssertThrowsError(try MLImageClassifier.DataSource.labeledDirectories(at: destination).labeledImages(),
                             "The curation folder intentionally retains empty reserved labels and cannot be imported directly")
        let labels = try MLImageClassifier.DataSource.labeledDirectories(at: training).labeledImages()
        XCTAssertEqual(labels.count, 13); XCTAssertEqual(labels.values.reduce(0) { $0 + $1.count }, 120)
        let viewManifest = try trainingManifest(training)
        for file in viewManifest.files {
            let output = training.appendingPathComponent(file.relativePath), source = URL(fileURLWithPath: file.sourcePath)
            XCTAssertEqual(try Data(contentsOf: output), try Data(contentsOf: source))
            XCTAssertEqual(try NativePNG.inspect(output), try NativePNG.inspect(source))
            XCTAssertFalse(try output.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink ?? true)
        }
        // Subsequent curation adds a formerly empty class and removes every
        // example from Wood. The atomically refreshed view follows both edits.
        let custom = destination.appendingPathComponent("Other Diffuse Textures/user-original.png")
        let customBytes = try png(width: 17, height: 19, value: 201); try customBytes.write(to: custom)
        for file in try FileManager.default.contentsOfDirectory(at: destination.appendingPathComponent("Wood"), includingPropertiesForKeys: nil) {
            try FileManager.default.removeItem(at: file)
        }
        let updated = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination))
        XCTAssertEqual(updated["training_count"] as? Int, 112)
        let updatedLabels = try MLImageClassifier.DataSource.labeledDirectories(at: training).labeledImages()
        XCTAssertNil(updatedLabels["Wood"]); XCTAssertEqual(updatedLabels["Other Diffuse Textures"]?.count, 1)
        XCTAssertEqual(try Data(contentsOf: training.appendingPathComponent("Other Diffuse Textures/user-original.png")), customBytes)
        XCTAssertEqual(try Data(contentsOf: custom), customBytes)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: destination.appendingPathComponent("Other Fallback").path).isEmpty)
        let html = try String(contentsOf: root.appendingPathComponent("Curated-review.html"), encoding: .utf8)
        XCTAssertTrue(html.contains("Choose <b>" + training.path + "</b>"))
    }

    func testTrainingViewNeverClobbersUnownedFoldersOrEditedGeneratedImages() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Curated"), training = root.appendingPathComponent("Curated-createml")
        let original = root.appendingPathComponent("wood_planks_diff_1k.png")
        try png(width: 12, height: 13, value: 39).write(to: original)
        let source = NativeDiffuseClassifierDatasetService.Source(family: "wood_planks", variant: "color_1", url: original, width: 12, height: 13, metadata: nil)
        try FileManager.default.createDirectory(at: training, withIntermediateDirectories: false)
        let userFile = training.appendingPathComponent("my-image.png"), userBytes = try png(width: 21, height: 17, value: 212)
        try userBytes.write(to: userFile)
        let unowned = try object(NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination))
        XCTAssertEqual(unowned["total"] as? Int, 1); XCTAssertEqual(unowned["training_ready"] as? Bool, false)
        XCTAssertFalse(try XCTUnwrap(unowned["warnings"] as? [String]).isEmpty)
        XCTAssertEqual(try Data(contentsOf: userFile), userBytes)
        XCTAssertEqual(try manifest(destination).entries.count, 1, "The completed curation dataset remains saved on view conflicts")
        try FileManager.default.removeItem(at: training)
        let generated = try object(NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination))
        XCTAssertEqual(generated["training_ready"] as? Bool, true)
        let inventory = try trainingManifest(training)
        let output = training.appendingPathComponent(try XCTUnwrap(inventory.files.first?.relativePath))
        let editedBytes = try png(width: 23, height: 25, value: 92); try editedBytes.write(to: output)
        let additionalURL = root.appendingPathComponent("red_brick_diff_1k.png")
        try png(width: 12, height: 13, value: 75).write(to: additionalURL)
        let additional = NativeDiffuseClassifierDatasetService.Source(family: "red_brick", variant: "color_1", url: additionalURL, width: 12, height: 13, metadata: nil)
        let modified = try object(NativeDiffuseClassifierDatasetService.refresh(sources: [source, additional], destination: destination))
        XCTAssertEqual(modified["total"] as? Int, 2); XCTAssertEqual(modified["added"] as? Int, 1)
        XCTAssertEqual(modified["training_ready"] as? Bool, false)
        XCTAssertEqual(try Data(contentsOf: output), editedBytes)
        XCTAssertEqual(try trainingManifest(training).files.count, 1, "An edited prior view stays intact while new curation completes")
        XCTAssertEqual(try manifest(destination).entries.count, 2)
    }

    func testTrainingViewPreservesUnexpectedUserFilesAndSymlinkDestinations() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Curated"), training = root.appendingPathComponent("Curated-createml")
        let original = root.appendingPathComponent("metal_plate_diff_1k.png")
        try png(width: 12, height: 13, value: 87).write(to: original)
        let source = NativeDiffuseClassifierDatasetService.Source(family: "metal_plate", variant: "color_1", url: original, width: 12, height: 13, metadata: nil)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination)
        let userFile = training.appendingPathComponent(".my-notes"), bytes = Data("Keep my review notes".utf8)
        try bytes.write(to: userFile)
        let result = try object(NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination))
        XCTAssertEqual(result["training_ready"] as? Bool, false); XCTAssertEqual(try Data(contentsOf: userFile), bytes)
        try FileManager.default.removeItem(at: training)
        let unrelated = root.appendingPathComponent("Unrelated")
        try FileManager.default.createDirectory(at: unrelated, withIntermediateDirectories: false)
        try bytes.write(to: unrelated.appendingPathComponent("important.txt"))
        try FileManager.default.createSymbolicLink(at: training, withDestinationURL: unrelated)
        let symbolic = try object(NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination))
        XCTAssertEqual(symbolic["training_ready"] as? Bool, false)
        XCTAssertEqual(try Data(contentsOf: unrelated.appendingPathComponent("important.txt")), bytes)
        XCTAssertEqual(try training.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink, true)
    }
    func testSpecificMaterialNamesPrecedeBroadProviderTags() throws {
        let metadata = NativeDiffuseClassifierDatasetService.ProviderMetadata(name: "Wood Chip Path",
            tags: ["wood", "ground", "terrain"], categories: ["wood"], pageURL: "https://polyhaven.com/a/wood_chip_path")
        XCTAssertEqual(NativeDiffuseClassifierDatasetService.classify(family: "wood_chip_path", filename: "wood_chip_path_diff_1k.png", metadata: metadata).classID, "medium_particles")
        let cases = ["painted_worn_brick": "bricks", "marble_tiles": "tiles", "concrete_pavement_02": "concrete",
            "Fabric085": "fabric", "snow013": "small_particles", "leaves_forest_ground": "ground",
            "pine_bark": "bark", "white_stucco": "wall_finish", "old_linoleum_flooring_01": "plastic"]
        for (family, expected) in cases {
            XCTAssertEqual(NativeDiffuseClassifierDatasetService.classify(family: family, filename: family + "_diff_1k.png", metadata: nil).classID, expected, family)
        }
        let opaque = NativeDiffuseClassifierDatasetService.ProviderMetadata(name: "Surface 007", tags: ["woven"], categories: ["fabric"], pageURL: nil)
        let assignment = NativeDiffuseClassifierDatasetService.classify(family: "surface_007", filename: "surface_007_diff_1k.png", metadata: opaque)
        XCTAssertEqual(assignment.classID, "fabric"); XCTAssertEqual(assignment.confidence, "provider_tags_review")
        XCTAssertNil(NativeDiffuseClassifierDatasetService.classify(family: "unknown_009", filename: "unknown_009_diff_1k.png", metadata: nil).classID)
        let barkMetadata = NativeDiffuseClassifierDatasetService.ProviderMetadata(name: "Pine Bark", tags: ["wood"], categories: ["wood"], pageURL: nil)
        XCTAssertEqual(NativeDiffuseClassifierDatasetService.classify(family: "pine_bark", filename: "pine_bark_diff_1k.png", metadata: barkMetadata).classID, "bark")
        let taggedBark = NativeDiffuseClassifierDatasetService.ProviderMetadata(name: "Surface 009", tags: ["bark"], categories: ["wood"], pageURL: nil)
        XCTAssertEqual(NativeDiffuseClassifierDatasetService.classify(family: "surface_009", filename: "surface_009_diff_1k.png", metadata: taggedBark).classID, "bark")
    }

    func testSemanticSubtypeDistinguishesNamedWoodFormsWithoutGuessing() {
        let cases = ["wood_planks": "planks", "wood_boards": "boards", "cedar_beams": "beams",
            "dark_paneled_wood": "panels", "oriented_strand_board": "oriented_strand_board"]
        for (family, expected) in cases {
            XCTAssertEqual(NativeDiffuseClassifierDatasetService.subtype(family: family, filename: family + "_diff_1k.png", metadata: nil, classID: "wood"), expected)
        }
        let generic = NativeDiffuseClassifierDatasetService.ProviderMetadata(name: "Wood Surface", tags: ["natural"], categories: ["wood"], pageURL: nil)
        XCTAssertNil(NativeDiffuseClassifierDatasetService.subtype(family: "wood_007", filename: "wood_007_diff_1k.png", metadata: generic, classID: "wood"))
        let tagged = NativeDiffuseClassifierDatasetService.ProviderMetadata(name: "Wood Surface", tags: ["boards"], categories: ["wood"], pageURL: nil)
        XCTAssertEqual(NativeDiffuseClassifierDatasetService.subtype(family: "wood_007", filename: "wood_007_diff_1k.png", metadata: tagged, classID: "wood"), "boards")
        XCTAssertEqual(NativeDiffuseClassifierDatasetService.subtype(family: "pine_bark", filename: "pine_bark_diff_1k.png", metadata: nil, classID: "bark"), "pine")
        XCTAssertNil(NativeDiffuseClassifierDatasetService.subtype(family: "bark_brown_02", filename: "bark_brown_02_diff_1k.png", metadata: nil, classID: "bark"))
    }

    func testLegacyWoodTaxonomySplitsUntouchedBarkAndPreservesManualCuration() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("CreateML")
        let families = ["pine_bark", "maple_bark", "birch_bark", "oak_bark", "cedar_bark", "bark_brown_02"]
        let sources = try families.enumerated().map { offset, family in
            let url = root.appendingPathComponent(family + "_diff_1k.png")
            try png(width: 12, height: 13, value: UInt8(30 + offset)).write(to: url)
            return NativeDiffuseClassifierDatasetService.Source(family: family, variant: "color_1", url: url, width: 12, height: 13, metadata: nil)
        }
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination)
        // Reconstruct the old schema: Bark was suggested into Wood and the
        // optional subtype field did not exist yet.
        let manifestURL = destination.appendingPathComponent(".classifier-manifest.json")
        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
        var oldEntries = try XCTUnwrap(legacy["entries"] as? [[String: Any]])
        for index in oldEntries.indices {
            let from = URL(fileURLWithPath: try XCTUnwrap(oldEntries[index]["currentPath"] as? String))
            let target = destination.appendingPathComponent("Wood/" + from.lastPathComponent)
            try FileManager.default.moveItem(at: from, to: target)
            oldEntries[index]["currentPath"] = target.path; oldEntries[index]["currentClass"] = "wood"
            oldEntries[index]["suggestedClass"] = "wood"; oldEntries[index]["status"] = "preserved"
            oldEntries[index]["subtype"] = nil
            if oldEntries[index]["family"] as? String == "bark_brown_02" {
                oldEntries[index]["status"] = "moved_by_user"; oldEntries[index]["needsReview"] = false
            }
        }
        legacy["classes"] = (legacy["classes"] as? [[String: Any]])?.filter { $0["id"] as? String != "bark" }
        legacy["entries"] = oldEntries; try writeJSON(legacy, to: manifestURL)
        XCTAssertTrue(try manifest(destination).entries.allSatisfy { $0.subtype == nil }, "Older manifests decode without subtype")
        func oldPath(_ family: String) throws -> URL {
            URL(fileURLWithPath: try XCTUnwrap(oldEntries.first { $0["family"] as? String == family }?["currentPath"] as? String))
        }
        let untouchedPath = try oldPath("pine_bark"), renamed = destination.appendingPathComponent("Wood/user-cedar.png")
        let moved = destination.appendingPathComponent("Metal/user-maple.png"), edited = try oldPath("birch_bark")
        let editedBytes = try png(width: 14, height: 15, value: 90)
        try FileManager.default.moveItem(at: oldPath("maple_bark"), to: moved)
        try editedBytes.write(to: edited)
        try FileManager.default.removeItem(at: oldPath("oak_bark"))
        try FileManager.default.moveItem(at: oldPath("cedar_bark"), to: renamed)
        let result = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination))
        XCTAssertEqual(result["reclassified"] as? Int, 1); XCTAssertEqual(result["added"] as? Int, 0)
        let refreshed = try manifest(destination)
        func refreshedEntry(_ family: String) throws -> NativeDiffuseClassifierDatasetService.Entry {
            try XCTUnwrap(refreshed.entries.first { $0.family == family })
        }
        let bark = try refreshedEntry("pine_bark")
        XCTAssertEqual(bark.currentClass, "bark"); XCTAssertEqual(bark.suggestedClass, "bark")
        XCTAssertEqual(bark.status, "taxonomy_refreshed"); XCTAssertEqual(bark.subtype, "pine")
        XCTAssertEqual(URL(fileURLWithPath: try XCTUnwrap(bark.currentPath)).lastPathComponent, untouchedPath.lastPathComponent)
        XCTAssertFalse(FileManager.default.fileExists(atPath: untouchedPath.path))
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(bark.currentPath))), try Data(contentsOf: sources[0].url))
        XCTAssertEqual(try refreshedEntry("maple_bark").currentPath, moved.path)
        XCTAssertEqual(try refreshedEntry("maple_bark").currentClass, "metal")
        XCTAssertEqual(try refreshedEntry("birch_bark").status, "modified_by_user")
        XCTAssertEqual(try Data(contentsOf: edited), editedBytes)
        XCTAssertNil(try refreshedEntry("oak_bark").currentPath)
        XCTAssertEqual(try refreshedEntry("oak_bark").status, "removed_by_user")
        XCTAssertEqual(try refreshedEntry("cedar_bark").currentPath, renamed.path)
        XCTAssertEqual(try refreshedEntry("cedar_bark").currentClass, "wood")
        XCTAssertEqual(try refreshedEntry("cedar_bark").status, "moved_by_user")
        XCTAssertEqual(try refreshedEntry("bark_brown_02").currentClass, "wood", "An explicit prior curation status blocks automatic migration")
        let second = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination))
        XCTAssertEqual(second["reclassified"] as? Int, 0)
        let repeated = try manifest(destination)
        XCTAssertEqual(repeated.entries.first { $0.family == "cedar_bark" }?.currentPath, renamed.path)
        XCTAssertNil(repeated.entries.first { $0.family == "oak_bark" }?.currentPath)
        let html = try String(contentsOf: root.appendingPathComponent("CreateML-review.html"), encoding: .utf8)
        XCTAssertTrue(html.contains("Subtype: <b>pine</b>"))
    }

    func testTaxonomyMigrationNeverOverwritesAnAddedImageAtTheDestination() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("CreateML"), original = root.appendingPathComponent("pine_bark_diff_1k.png")
        try png(width: 12, height: 13, value: 48).write(to: original)
        let source = NativeDiffuseClassifierDatasetService.Source(family: "pine_bark", variant: "color_1", url: original, width: 12, height: 13, metadata: nil)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination)
        let manifestURL = destination.appendingPathComponent(".classifier-manifest.json")
        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
        var entries = try XCTUnwrap(legacy["entries"] as? [[String: Any]])
        let barkPath = URL(fileURLWithPath: try XCTUnwrap(entries[0]["currentPath"] as? String))
        let woodPath = destination.appendingPathComponent("Wood/" + barkPath.lastPathComponent)
        try FileManager.default.moveItem(at: barkPath, to: woodPath)
        entries[0]["currentPath"] = woodPath.path; entries[0]["currentClass"] = "wood"; entries[0]["suggestedClass"] = "wood"
        legacy["entries"] = entries; try writeJSON(legacy, to: manifestURL)
        let userBytes = try png(width: 19, height: 20, value: 74); try userBytes.write(to: barkPath)
        let conflict = try object(NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination))
        XCTAssertEqual(conflict["reclassified"] as? Int, 0)
        XCTAssertFalse(try XCTUnwrap(conflict["warnings"] as? [String]).isEmpty)
        XCTAssertEqual(try manifest(destination).entries.first?.status, "taxonomy_conflict_review")
        XCTAssertEqual(try Data(contentsOf: barkPath), userBytes)
        XCTAssertEqual(try Data(contentsOf: woodPath), try Data(contentsOf: original))
        try FileManager.default.removeItem(at: barkPath)
        let resolved = try object(NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination))
        XCTAssertEqual(resolved["reclassified"] as? Int, 1)
        XCTAssertEqual(try Data(contentsOf: barkPath), try Data(contentsOf: original))
    }

    func testSmallestOriginalTierAndEveryColorVariantAreCopiedWithoutResizingOrGammaChanges() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let corpus = root.appendingPathComponent("Originals"), destination = root.appendingPathComponent("CreateML")
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        let fixtures = [("wood_planks_diff_1k.png", 13, 11, 37), ("wood_planks_diff_4k.png", 26, 22, 39),
            ("wood_planks_col_2_1k.png", 13, 11, 40), ("wood_planks_nor_gl_1k.png", 13, 11, 128),
            ("Fabric085_1K-PNG_Color.png", 12, 17, 54)]
        var originals: [URL: Data] = [:]
        for (name, width, height, code) in fixtures {
            let file = corpus.appendingPathComponent(name), bytes = try png(width: width, height: height, value: UInt8(code))
            try bytes.write(to: file); originals[file] = bytes
        }
        let sources = try NativeDiffuseClassifierDatasetService.discover(dataset: nil, sourceFolder: corpus)
        XCTAssertEqual(sources.count, 3)
        XCTAssertEqual(Set(sources.map(\.variant)), ["color_1", "color_2"])
        XCTAssertFalse(sources.contains { $0.url.lastPathComponent.contains("4k") || $0.url.lastPathComponent.contains("nor_gl") })
        let result = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination))
        XCTAssertEqual(result["added"] as? Int, 3); XCTAssertEqual(result["total"] as? Int, 3)
        let manifest = try JSONDecoder().decode(NativeDiffuseClassifierDatasetService.Manifest.self,
            from: Data(contentsOf: destination.appendingPathComponent(".classifier-manifest.json")))
        for entry in manifest.entries {
            let source = URL(fileURLWithPath: entry.sourcePath), copied = URL(fileURLWithPath: try XCTUnwrap(entry.currentPath))
            XCTAssertEqual(try Data(contentsOf: copied), originals[source])
            XCTAssertEqual(entry.sourceSHA256, sha256(try XCTUnwrap(originals[source])))
            XCTAssertEqual(try NativePNG.inspect(copied), try NativePNG.inspect(source))
            XCTAssertFalse(try copied.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink ?? true)
        }
        for (source, bytes) in originals { XCTAssertEqual(try Data(contentsOf: source), bytes) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent("Other Diffuse Textures").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent("Other Fallback").path))
        let second = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination))
        XCTAssertEqual(second["added"] as? Int, 0); XCTAssertEqual(second["preserved"] as? Int, 3)
    }

    func testRefreshPreservesManualMovesRenamesEditsDeletesAndUserAddedExamples() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("CreateML")
        let families = ["red_brick", "wood_planks", "metal_plate", "white_stucco"]
        let sources = try families.enumerated().map { offset, family in
            let url = root.appendingPathComponent(family + "_diff_1k.png")
            try png(width: 12, height: 13, value: UInt8(40 + offset)).write(to: url)
            return NativeDiffuseClassifierDatasetService.Source(family: family, variant: "color_1", url: url, width: 12, height: 13, metadata: nil)
        }
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination)
        let before = try manifest(destination)
        func entry(_ family: String) throws -> NativeDiffuseClassifierDatasetService.Entry { try XCTUnwrap(before.entries.first { $0.family == family }) }
        let moved = destination.appendingPathComponent("Tiles and Pavers/manually-renamed.png")
        try FileManager.default.moveItem(at: URL(fileURLWithPath: try XCTUnwrap(entry("red_brick").currentPath)), to: moved)
        let edited = URL(fileURLWithPath: try XCTUnwrap(entry("wood_planks").currentPath)), editedBytes = try png(width: 14, height: 15, value: 67)
        try editedBytes.write(to: edited)
        try FileManager.default.removeItem(at: URL(fileURLWithPath: try XCTUnwrap(entry("metal_plate").currentPath)))
        let custom = destination.appendingPathComponent("Other Diffuse Textures/my-material.png"), customBytes = try png(width: 9, height: 10, value: 81)
        try customBytes.write(to: custom)
        let output = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination))
        XCTAssertEqual(output["added"] as? Int, 0); XCTAssertEqual(output["total"] as? Int, 4)
        let after = try manifest(destination)
        let movedEntry = try XCTUnwrap(after.entries.first { $0.family == "red_brick" })
        XCTAssertEqual(movedEntry.currentPath, moved.path); XCTAssertEqual(movedEntry.currentClass, "tiles")
        XCTAssertEqual(movedEntry.status, "moved_by_user"); XCTAssertFalse(movedEntry.needsReview)
        XCTAssertEqual(after.entries.first { $0.family == "wood_planks" }?.status, "modified_by_user")
        XCTAssertEqual(after.entries.first { $0.family == "metal_plate" }?.status, "removed_by_user")
        XCTAssertEqual(try Data(contentsOf: edited), editedBytes); XCTAssertEqual(try Data(contentsOf: custom), customBytes)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination)
        XCTAssertNil(try manifest(destination).entries.first { $0.family == "metal_plate" }?.currentPath,
                     "Removing a suggested example remains an explicit curation decision across refreshes")
    }

    func testUnknownDiffuseStaysOutsideTheCreateMLClassFolders() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("opaque_diff_1k.png"), destination = root.appendingPathComponent("CreateML")
        try png(width: 13, height: 13, value: 111).write(to: sourceURL)
        let source = NativeDiffuseClassifierDatasetService.Source(family: "opaque", variant: "color_1", url: sourceURL, width: 13, height: 13, metadata: nil)
        let output = try object(NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination))
        XCTAssertEqual(output["total"] as? Int, 0); XCTAssertEqual(output["unclassified"] as? Int, 1)
        let entry = try XCTUnwrap(try manifest(destination).entries.first)
        XCTAssertNil(entry.currentClass); XCTAssertTrue(entry.needsReview)
        XCTAssertTrue(try XCTUnwrap(entry.currentPath).hasPrefix(destination.path + "-pending-review/"))
        for folder in ["Other Diffuse Textures", "Other Fallback"] {
            XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: destination.appendingPathComponent(folder).path).isEmpty)
        }
        let moved = destination.appendingPathComponent("Other Diffuse Textures/reviewed.png")
        try FileManager.default.moveItem(at: URL(fileURLWithPath: try XCTUnwrap(entry.currentPath)), to: moved)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination)
        XCTAssertEqual(try manifest(destination).entries.first?.currentClass, "other_diffuse")
        XCTAssertEqual(try manifest(destination).entries.first?.status, "moved_by_user")
    }

    func testDatasetDiscoveryUsesOriginalSourcesAndIgnoresExcludedSamples() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), original = root.appendingPathComponent("wood_planks_diff_1k.png")
        try png(width: 18, height: 14, value: 84).write(to: original)
        try FileManager.default.createDirectory(at: dataset.appendingPathComponent("samples/wood"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dataset.appendingPathComponent("samples/excluded"), withIntermediateDirectories: true)
        let map: [String: Any] = ["variant_id": "color_1", "source": ["path": original.path]]
        for (path, status) in [("wood", "unreviewed"), ("excluded", "excluded")] {
            try writeJSON(["source_family_id": "wood_planks", "status": status, "input_variants": [map]],
                to: dataset.appendingPathComponent("samples/\(path)/sample.json"))
        }
        try writeJSON(["samples": [["path": "samples/wood"], ["path": "samples/excluded"]]], to: dataset.appendingPathComponent("dataset.json"))
        let sources = try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: nil)
        XCTAssertEqual(sources.count, 1); XCTAssertEqual(sources.first?.url, original)
        XCTAssertEqual(sources.first?.width, 18); XCTAssertEqual(sources.first?.height, 14)
        try writeJSON(["samples": [["path": "../outside"]]], to: dataset.appendingPathComponent("dataset.json"))
        XCTAssertThrowsError(try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: nil))
    }

    func testRefreshRejectsSymlinkClassAndOriginalMapDestination() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("wood_diff_1k.png"), destination = root.appendingPathComponent("CreateML")
        try png(width: 12, height: 13, value: 56).write(to: original)
        let source = NativeDiffuseClassifierDatasetService.Source(family: "wood", variant: "color_1", url: original, width: 12, height: 13, metadata: nil)
        XCTAssertThrowsError(try NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: root))
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: destination.appendingPathComponent("Bricks"), withDestinationURL: root)
        XCTAssertThrowsError(try NativeDiffuseClassifierDatasetService.refresh(sources: [source], destination: destination))
        XCTAssertTrue(FileManager.default.fileExists(atPath: original.path))
    }

    private func manifest(_ root: URL) throws -> NativeDiffuseClassifierDatasetService.Manifest {
        try JSONDecoder().decode(NativeDiffuseClassifierDatasetService.Manifest.self, from: Data(contentsOf: root.appendingPathComponent(".classifier-manifest.json")))
    }
    private func trainingManifest(_ root: URL) throws -> NativeDiffuseClassifierDatasetService.TrainingViewManifest {
        try JSONDecoder().decode(NativeDiffuseClassifierDatasetService.TrainingViewManifest.self,
            from: Data(contentsOf: root.appendingPathComponent(".classifier-training-view.json")))
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("classifier-dataset-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true); return root.resolvingSymlinksInPath().standardizedFileURL
    }
    private func png(width: Int, height: Int, value: UInt8) throws -> Data {
        try NativePNG(header: .init(width: width, height: height, bits: 8, channels: 3, color: 2, interlace: 0),
            pixels: Data(repeating: value, count: width * height * 3), colorChunks: [("gAMA", Data([0, 0, 177, 143]))]).encoded()
    }
    private func object(_ text: String) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]) }
    private func writeJSON(_ value: [String: Any], to url: URL) throws { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: url) }
    private func sha256(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
}
