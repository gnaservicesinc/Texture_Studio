import CryptoKit
import CreateML
import Foundation
import ImageIO
import XCTest
@testable import TextureStudio

final class NativeDiffuseClassifierDatasetTests: XCTestCase {
    func testSelectedGridUsesEligibleNativeTierAndReportsEveryUnreadableOrSmallerOriginal() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let corpus = root.appendingPathComponent("Originals")
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        for (name, width, height) in [("wood_planks_diff_1k.png", 8, 8), ("wood_planks_diff_2k.png", 17, 16),
                                     ("wood_planks_diff_4k.png", 32, 32), ("wood_planks_col_2_2k.png", 16, 17),
                                     ("linen_diff_2k.png", 16, 8)] {
            try png(width: width, height: height, value: 73).write(to: corpus.appendingPathComponent(name))
        }
        try Data("malformed PNG".utf8).write(to: corpus.appendingPathComponent("brick_diff_2k.png"))
        try Data("ancillary".utf8).write(to: corpus.appendingPathComponent("wood_planks_ambientOcclusion_2k.png"))
        let result = try NativeDiffuseClassifierDatasetService.discoverReport(dataset: nil, sourceFolder: corpus, minimumSize: 16)
        XCTAssertEqual(result.minimumSize, 16)
        XCTAssertEqual(result.discoveredCount, 6)
        XCTAssertEqual(result.eligibleCount, 3)
        XCTAssertEqual(result.undersizedCount, 2)
        XCTAssertEqual(result.unreadableCount, 1)
        XCTAssertEqual(result.sources.count, 2)
        XCTAssertEqual(Set(result.sources.map(\.variant)), ["color_1", "color_2"])
        XCTAssertTrue(result.sources.allSatisfy { $0.width >= 16 && $0.height >= 16 && $0.id.contains("|native_") })
        XCTAssertEqual(result.sources.first { $0.variant == "color_1" }?.url.lastPathComponent, "wood_planks_diff_2k.png")
        XCTAssertTrue(result.warnings.contains { $0.contains("brick_diff_2k.png") })
    }

    func testDistinctFoldersWithSameNormalizedFamilyKeepTheirOwnTiersAndCuration() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let corpus = root.appendingPathComponent("Originals"), destination = root.appendingPathComponent("Curated")
        for (offset, folderName) in ["Pine Bark", "pine_bark"].enumerated() {
            let folder = corpus.appendingPathComponent(folderName)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for (tier, edge) in [("1k", 8), ("2k", 16), ("4k", 32)] {
                try png(width: edge, height: edge, value: UInt8(25 + offset)).write(to: folder.appendingPathComponent("pine_bark_diff_" + tier + ".png"))
            }
        }
        let selected = try NativeDiffuseClassifierDatasetService.discover(dataset: nil, sourceFolder: corpus, minimumSize: 16)
        XCTAssertEqual(selected.count, 2)
        XCTAssertEqual(Set(selected.map(\.id)).count, 2)
        XCTAssertTrue(selected.allSatisfy { $0.width == 16 && $0.id.contains("|folder_") })
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: selected, destination: destination, minimumSize: 16)
        let initial = try manifest(destination)
        let first = try XCTUnwrap(initial.entries.first), second = try XCTUnwrap(initial.entries.last)
        let moved = destination.appendingPathComponent("Metal/first-folder-only.png")
        try FileManager.default.moveItem(at: URL(fileURLWithPath: try XCTUnwrap(first.currentPath)), to: moved)
        let larger = try NativeDiffuseClassifierDatasetService.discover(dataset: nil, sourceFolder: corpus, minimumSize: 32)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: larger, destination: destination, minimumSize: 32)
        let updated = try manifest(destination)
        let firstDirectory = URL(fileURLWithPath: first.sourcePath).deletingLastPathComponent().path
        let current = updated.entries.filter { $0.width == 32 }
        XCTAssertEqual(current.count, 2)
        XCTAssertEqual(current.first { URL(fileURLWithPath: $0.sourcePath).deletingLastPathComponent().path == firstDirectory }?.currentClass, "metal")
        XCTAssertEqual(current.first { URL(fileURLWithPath: $0.sourcePath).deletingLastPathComponent().path != firstDirectory }?.currentClass, "bark")
        XCTAssertEqual(try Data(contentsOf: moved), try Data(contentsOf: URL(fileURLWithPath: first.sourcePath)))
        XCTAssertTrue(FileManager.default.fileExists(atPath: try XCTUnwrap(second.currentPath)))
    }

    func testManualOriginalFileTagOverridesFilenameSuggestionAndSurvivesRefresh() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("wood_planks_diff_2k.png"), destination = root.appendingPathComponent("Curated")
        let bytes = try png(width: 16, height: 16, value: 26)
        try bytes.write(to: original)
        try NativeSourceFileTags.update(original, classLabel: "Metal", classifierIdentity: "user", reviewStatus: "reviewed")
        let sources = try NativeDiffuseClassifierDatasetService.discover(dataset: nil, sourceFolder: root, minimumSize: 16)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination, minimumSize: 16)
        let entry = try XCTUnwrap(try manifest(destination).entries.first)
        XCTAssertEqual(entry.currentClass, "metal")
        XCTAssertEqual(entry.suggestedClass, "metal")
        XCTAssertEqual(entry.confidence, "user_file_tag")
        XCTAssertEqual(NativeSourceFileTags.read(original)?.classifierIdentity, "user")
        XCTAssertEqual(NativeSourceFileTags.read(original)?.classLabel, "Metal")
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination, minimumSize: 16)
        XCTAssertEqual(try manifest(destination).entries.first?.currentClass, "metal")
        XCTAssertEqual(try Data(contentsOf: original), bytes)
    }

    func testOpenPlainSourceFolderAndUserAssignedDiffuseRoleNeedNoDatasetManifest() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("my-original.png"), bytes = try png(width: 16, height: 16, value: 49)
        try bytes.write(to: original)
        try NativeSourceFileTags.update(original, role: "input", classLabel: "Fabric", classifierIdentity: "user")
        let report = try NativeDiffuseClassifierDatasetService.discoverReport(dataset: root, sourceFolder: nil, minimumSize: 16)
        XCTAssertEqual(report.sources.count, 1)
        XCTAssertEqual(report.sources.first?.url, original)
        XCTAssertEqual(report.minimumSize, 16)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("dataset.json").path))
        XCTAssertEqual(try Data(contentsOf: original), bytes)
    }

    func testJPEGAndTaggedTIFFOriginalsKeepDimensionsExtensionsAndExactBytes() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let corpus = root.appendingPathComponent("Originals"), destination = root.appendingPathComponent("Curated")
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        let jpeg = corpus.appendingPathComponent("cotton_basecolor_2k.jpg"), tiff = corpus.appendingPathComponent("my-photo.tiff")
        let jpegBytes = try imageBytes(width: 256, height: 257, type: "public.jpeg")
        let tiffBytes = try imageBytes(width: 258, height: 256, type: "public.tiff")
        try jpegBytes.write(to: jpeg); try tiffBytes.write(to: tiff)
        try NativeSourceFileTags.update(tiff, role: "input", classLabel: "Metal", classifierIdentity: "user")
        let sources = try NativeDiffuseClassifierDatasetService.discover(dataset: nil, sourceFolder: corpus, minimumSize: 256)
        XCTAssertEqual(sources.count, 2)
        XCTAssertEqual(sources.first { $0.url == jpeg }?.width, 256)
        XCTAssertEqual(sources.first { $0.url == jpeg }?.height, 257)
        XCTAssertEqual(sources.first { $0.url == tiff }?.width, 258)
        let result = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination, minimumSize: 256))
        let entries = try manifest(destination).entries
        for entry in entries {
            let original = URL(fileURLWithPath: entry.sourcePath), copied = URL(fileURLWithPath: try XCTUnwrap(entry.currentPath))
            XCTAssertEqual(copied.pathExtension, original.pathExtension)
            XCTAssertEqual(try Data(contentsOf: copied), try Data(contentsOf: original))
        }
        XCTAssertEqual(result["training_count"] as? Int, 2)
        let training = try trainingManifest(URL(fileURLWithPath: try XCTUnwrap(result["training_path"] as? String)))
        XCTAssertEqual(Set(training.files.map { URL(fileURLWithPath: $0.relativePath).pathExtension }), ["jpg", "tiff"])
        XCTAssertEqual(try Data(contentsOf: jpeg), jpegBytes)
        XCTAssertEqual(try Data(contentsOf: tiff), tiffBytes)
    }

    func testManualUnlabeledAndNonDiffuseRolesOverrideFilenameTaxonomy() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let corpus = root.appendingPathComponent("Originals"), destination = root.appendingPathComponent("Curated")
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        let diffuse = corpus.appendingPathComponent("wood_planks_diff_2k.png")
        let ancillary = corpus.appendingPathComponent("brick_basecolor_2k.png")
        let bytes = try png(width: 16, height: 16, value: 61)
        try bytes.write(to: diffuse); try bytes.write(to: ancillary)
        try NativeSourceFileTags.update(ancillary, role: "ancillary", classLabel: "Metal", classifierIdentity: "user")
        let sources = try NativeDiffuseClassifierDatasetService.discover(dataset: nil, sourceFolder: corpus, minimumSize: 16)
        XCTAssertEqual(sources.map(\.url), [diffuse])
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination, minimumSize: 16)
        XCTAssertEqual(try manifest(destination).entries.first?.currentClass, "wood")
        try NativeSourceFileTags.update(diffuse, role: "input", classLabel: "", classifierIdentity: "user")
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination, minimumSize: 16)
        let unlabeled = try XCTUnwrap(try manifest(destination).entries.first)
        XCTAssertNil(unlabeled.currentClass)
        XCTAssertNil(unlabeled.suggestedClass)
        XCTAssertEqual(unlabeled.confidence, "user_file_tag")
        XCTAssertTrue(try XCTUnwrap(unlabeled.currentPath).contains("-pending-review/"))
        XCTAssertEqual(NativeSourceFileTags.read(diffuse)?.classLabel, "")
        XCTAssertEqual(NativeSourceFileTags.read(diffuse)?.classifierIdentity, "user")
        XCTAssertEqual(try Data(contentsOf: diffuse), bytes)
        XCTAssertEqual(try Data(contentsOf: ancillary), bytes)
    }

    func testManualFileLabelAgreesWithUserMoveWithoutMovingOrConflicting() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("wood_diff_2k.png"), destination = root.appendingPathComponent("Curated")
        let bytes = try png(width: 16, height: 16, value: 39)
        try bytes.write(to: original)
        let sources = try NativeDiffuseClassifierDatasetService.discover(dataset: nil, sourceFolder: root, minimumSize: 16)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination, minimumSize: 16)
        let first = try XCTUnwrap(try manifest(destination).entries.first)
        let moved = destination.appendingPathComponent("Metal/" + first.filename)
        try FileManager.default.moveItem(at: URL(fileURLWithPath: try XCTUnwrap(first.currentPath)), to: moved)
        try NativeSourceFileTags.update(original, classLabel: "Metal", classifierIdentity: "user")
        let result = try object(NativeDiffuseClassifierDatasetService.refresh(sources: sources, destination: destination, minimumSize: 16))
        let updated = try XCTUnwrap(try manifest(destination).entries.first)
        XCTAssertEqual(updated.currentPath, moved.path)
        XCTAssertEqual(updated.currentClass, "metal")
        XCTAssertEqual(updated.suggestedClass, "metal")
        XCTAssertFalse((result["warnings"] as? [String] ?? []).contains { $0.contains("destination already") })
        XCTAssertEqual(try Data(contentsOf: moved), bytes)
    }

    func testPhysicalFolderAndPreparedDatasetDiscoveryUseTheSelectedNativeGrid() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        let prepared = dataset.appendingPathComponent(".training-data/prepared")
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: false)
        for (index, edge) in [8, 16, 32].enumerated() {
            let file = dataset.appendingPathComponent("wood_diff_\(index + 1)k.png")
            try png(width: edge, height: edge, value: UInt8(edge)).write(to: file)
        }
        // Old ordinary manifests cannot choose the classifier grid or hide physical originals.
        try writeJSON(["training_size": 8, "samples": []], to: dataset.appendingPathComponent("dataset.json"))
        try FileManager.default.createDirectory(at: prepared, withIntermediateDirectories: true)
        try writeJSON(["crop_size": 16, "source_folder_dataset_path": dataset.path,
            "native_size_preparation": ["schema": "texture-studio-native-crops-v1", "source_dataset_path": dataset.path, "target_resized": false]],
            to: prepared.appendingPathComponent("dataset.json"))
        XCTAssertEqual(try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: nil, minimumSize: 8).first?.width, 8)
        XCTAssertEqual(try NativeDiffuseClassifierDatasetService.discover(dataset: prepared, sourceFolder: nil).first?.width, 16)
        XCTAssertEqual(try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: nil, minimumSize: 32).first?.width, 32)
        XCTAssertTrue(try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: nil).isEmpty,
            "A stale ordinary manifest does not override the default classifier resolution")
    }

    func testHigherNativeTierPreservesLegacyCurationAndFiltersAllSmallCuratedImages() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Curated")
        let families = ["wood_planks", "wood_boards", "cedar_beams"]
        var smallSources: [NativeDiffuseClassifierDatasetService.Source] = [], largeSources: [NativeDiffuseClassifierDatasetService.Source] = []
        for family in families {
            let small = root.appendingPathComponent(family + "_diff_1k.png"), large = root.appendingPathComponent(family + "_diff_2k.png")
            try png(width: 8, height: 8, value: 71).write(to: small)
            try png(width: 16, height: 16, value: 93).write(to: large)
            smallSources.append(.init(family: family, variant: "color_1", url: small, width: 8, height: 8, metadata: nil))
            largeSources.append(.init(family: family, variant: "color_1", url: large, width: 16, height: 16, metadata: nil, resolutionAware: true))
        }
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: smallSources, destination: destination)
        let old = try manifest(destination)
        func oldPath(_ family: String) throws -> URL {
            URL(fileURLWithPath: try XCTUnwrap(old.entries.first { $0.family == family }?.currentPath))
        }
        let moved = destination.appendingPathComponent("Metal/my-reviewed-planks.png"), edited = try oldPath("wood_boards")
        try FileManager.default.moveItem(at: oldPath("wood_planks"), to: moved)
        let editedBytes = try png(width: 9, height: 9, value: 116)
        try editedBytes.write(to: edited)
        try FileManager.default.removeItem(at: oldPath("cedar_beams"))
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: smallSources, destination: destination)
        let addedSmall = destination.appendingPathComponent("Bricks/my-small-image.png")
        let addedLarge = destination.appendingPathComponent("Bricks/my-large-image.png")
        let smallBytes = try png(width: 4, height: 16, value: 88), largeBytes = try png(width: 32, height: 16, value: 90)
        try smallBytes.write(to: addedSmall); try largeBytes.write(to: addedLarge)
        let output = try object(NativeDiffuseClassifierDatasetService.refresh(sources: largeSources, destination: destination, minimumSize: 16))
        let updated = try manifest(destination)
        XCTAssertEqual(updated.minimumNativeSize, 16)
        XCTAssertTrue(updated.selectionPolicy.contains("at_least_16"))
        XCTAssertEqual(try Data(contentsOf: moved), try Data(contentsOf: smallSources[0].url))
        XCTAssertEqual(try Data(contentsOf: edited), editedBytes)
        XCTAssertEqual(try Data(contentsOf: addedSmall), smallBytes)
        XCTAssertEqual(try Data(contentsOf: addedLarge), largeBytes)
        let newPlanks = try XCTUnwrap(updated.entries.first { $0.id == largeSources[0].id })
        XCTAssertEqual(newPlanks.currentClass, "metal")
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(newPlanks.currentPath))), try Data(contentsOf: largeSources[0].url))
        XCTAssertEqual(updated.entries.first { $0.id == largeSources[2].id }?.status, "removed_by_user")
        XCTAssertNil(updated.entries.first { $0.id == largeSources[2].id }?.currentPath)
        XCTAssertEqual(output["training_count"] as? Int, 3)
        XCTAssertEqual(output["training_undersized_count"] as? Int, 3)
        let view = try trainingManifest(URL(fileURLWithPath: try XCTUnwrap(output["training_path"] as? String)))
        XCTAssertTrue(view.files.allSatisfy { file in
            guard let header = try? NativePNG.inspect(URL(fileURLWithPath: file.sourcePath)) else { return false }
            return header.width >= 16 && header.height >= 16
        })
        let repeated = try object(NativeDiffuseClassifierDatasetService.refresh(sources: largeSources, destination: destination, minimumSize: 16))
        XCTAssertEqual(repeated["added"] as? Int, 0)
        XCTAssertEqual(repeated["training_count"] as? Int, 3)
        XCTAssertEqual(try Data(contentsOf: moved), try Data(contentsOf: smallSources[0].url))
    }

    func testFolderCommandDefaultsTo2048AndDoesNotFetchProviderCatalog() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let corpus = root.appendingPathComponent("Originals"), destination = root.appendingPathComponent("Curated")
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        let small = corpus.appendingPathComponent("wood_diff_1k.png"), bytes = try png(width: 1024, height: 1024, value: 17)
        try bytes.write(to: small)
        ClassifierNetworkProbe.reset()
        URLProtocol.registerClass(ClassifierNetworkProbe.self)
        defer { URLProtocol.unregisterClass(ClassifierNetworkProbe.self) }
        let result = try await NativeDiffuseClassifierDatasetService.run(arguments: [
            "refresh-classifier-dataset", "--source-folder", corpus.path, "--destination", destination.path
        ])
        let output = try object(try XCTUnwrap(result))
        XCTAssertEqual(output["minimum_native_size"] as? Int, 2048)
        XCTAssertEqual(output["discovered_count"] as? Int, 1)
        XCTAssertEqual(output["undersized_count"] as? Int, 1)
        XCTAssertEqual(output["training_count"] as? Int, 0)
        XCTAssertEqual(ClassifierNetworkProbe.requests, 0)
        XCTAssertEqual(try Data(contentsOf: small), bytes)
        let html = try String(contentsOf: root.appendingPathComponent("Curated-review.html"), encoding: .utf8)
        XCTAssertTrue(html.contains("at least 2048 × 2048"))
        XCTAssertTrue(html.contains("1 original diffuse images are smaller"))
    }

    func testLocalRefreshPreservesUnclassifiedOriginalAndContinuesPastAReadableHeaderWithCorruptBody() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let corpus = root.appendingPathComponent("Originals"), destination = root.appendingPathComponent("Curated")
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        let unclassified = corpus.appendingPathComponent("surface_009_diff_2k.png")
        let bytes = try png(width: 16, height: 16, value: 17)
        try bytes.write(to: unclassified)
        var corrupt = bytes; corrupt[45] ^= 0x40
        let damaged = corpus.appendingPathComponent("wood_diff_2k.png")
        try corrupt.write(to: damaged)
        let result = try await NativeDiffuseClassifierDatasetService.run(arguments: [
            "refresh-classifier-dataset", "--source-folder", corpus.path, "--destination", destination.path, "--size", "16"
        ])
        let output = try object(try XCTUnwrap(result))
        XCTAssertEqual(output["unclassified"] as? Int, 1)
        XCTAssertEqual(output["unreadable_count"] as? Int, 1)
        XCTAssertEqual(output["training_count"] as? Int, 0)
        XCTAssertTrue(try XCTUnwrap(output["warnings"] as? [String]).contains { $0.contains(damaged.path) })
        let entry = try XCTUnwrap(try manifest(destination).entries.first)
        XCTAssertNil(entry.currentClass)
        XCTAssertTrue(try XCTUnwrap(entry.currentPath).contains("pending-review"))
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(entry.currentPath))), bytes)
        XCTAssertEqual(try Data(contentsOf: unclassified), bytes)
        XCTAssertEqual(try Data(contentsOf: damaged), corrupt)
    }

    func testDeveloperClassifierGridDiscoveryUsesNative2048OriginalsWithoutChangingUserFolders() throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_CLASSIFIER_GRID_PROOF"] == "1" else {
            throw XCTSkip("Opt-in read-only classifier source discovery against the developer corpus.")
        }
        let corpus = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let started = ProcessInfo.processInfo.systemUptime
        let report = try NativeDiffuseClassifierDatasetService.discoverReport(dataset: nil, sourceFolder: corpus, minimumSize: 2048)
        let seconds = ProcessInfo.processInfo.systemUptime - started
        XCTAssertTrue(report.sources.allSatisfy { $0.width >= 2048 && $0.height >= 2048 })
        XCTAssertGreaterThan(report.sources.count, 100)
        let evidence: [String: Any] = ["source_folder": corpus.path, "minimum_native_size": report.minimumSize,
            "discovered_count": report.discoveredCount, "eligible_count": report.eligibleCount,
            "undersized_count": report.undersizedCount, "unreadable_count": report.unreadableCount,
            "selected_count": report.sources.count, "discovery_seconds": seconds, "user_folders_modified": false,
            "selected_sources": report.sources.map { ["path": $0.url.path, "width": $0.width, "height": $0.height] as [String: Any] }]
        try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-classifier-native-grid-proof.json"))
        print("Classifier local header scan: \(report.sources.count) native 2048+ originals; \(report.undersizedCount) smaller images preserved; \(seconds) seconds.")
    }

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

    func testSmallestEligibleOriginalTierAndEveryColorVariantAreCopiedWithoutResizingOrGammaChanges() throws {
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
        let sources = try NativeDiffuseClassifierDatasetService.discover(dataset: nil, sourceFolder: corpus, minimumSize: 11)
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

    func testDatasetDiscoveryUsesCurrentContainedOriginalsAndIgnoresStaleReferenceLists() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: false)
        let original = dataset.appendingPathComponent("wood_planks_diff_1k.png")
        let outside = root.appendingPathComponent("outside_diff_1k.png")
        let bytes = try png(width: 18, height: 14, value: 84)
        try bytes.write(to: original); try bytes.write(to: outside)
        let stale: [String: Any] = ["training_size": 12,
            "source_folders": [root.path], "samples": [["path": "../outside"], ["path": "samples/excluded"]]]
        try writeJSON(stale, to: dataset.appendingPathComponent("dataset.json"))
        let sources = try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: nil, minimumSize: 12)
        XCTAssertEqual(sources.count, 1); XCTAssertEqual(sources.first?.url, original)
        XCTAssertEqual(sources.first?.width, 18); XCTAssertEqual(sources.first?.height, 14)
        XCTAssertEqual(try Data(contentsOf: original), bytes)
        try FileManager.default.removeItem(at: original)
        XCTAssertTrue(try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: nil, minimumSize: 12).isEmpty,
            "A reference list cannot retain deleted originals or include files outside the open folder")
        XCTAssertEqual(try Data(contentsOf: outside), bytes)
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
    private func imageBytes(width: Int, height: Int, type: String) throws -> Data {
        let pngBytes = try png(width: width, height: height, value: 71)
        let source = try XCTUnwrap(CGImageSourceCreateWithData(pngBytes as CFData, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let bytes = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(bytes, type as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return bytes as Data
    }
    private func object(_ text: String) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]) }
    private func writeJSON(_ value: [String: Any], to url: URL) throws { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: url) }
    private func sha256(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
}

private final class ClassifierNetworkProbe: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var requestCount = 0
    static var requests: Int { lock.lock(); defer { lock.unlock() }; return requestCount }
    static func reset() { lock.lock(); requestCount = 0; lock.unlock() }
    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "api.polyhaven.com" && request.url?.path == "/assets"
    }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.lock.lock(); Self.requestCount += 1; Self.lock.unlock()
        client?.urlProtocol(self, didFailWithError: StudioError("Classifier refresh must use local data."))
    }
    override func stopLoading() {}
}
