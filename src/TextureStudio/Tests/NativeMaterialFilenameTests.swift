import CreateML
import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialFilenameTests: XCTestCase {
    func testSharedExplicitRoleParserSupportsProviderAndAuthoredNamesWithoutGuessingPhotos() throws {
        let cases = [
            ("Paper003_1K-PNG_Color.png", "paper003", "diff", "1k"),
            ("Paper005_4K-PNG_Displacement.png", "paper005", "disp", "4k"),
            ("Cardboard_01_color_1K.png", "cardboard_01", "color", "1k"),
            ("Cardboard_01_height_2K.png", "cardboard_01", "height", "2k"),
            ("Cardboard_01_normal_1K.png", "cardboard_01", "normal", "1k"),
            ("Cardboard_01_normalDX_2K.png", "cardboard_01", "nor_dx", "2k"),
            ("Cardboard_01_1K_NormalGL.png", "cardboard_01", "nor_gl", "1k"),
            ("Cardboard_01_BaseColor.png", "cardboard_01", "basecolor", "native"),
            ("stone_col_2_4k.png", "stone", "col_2", "4k"),
            ("wood_2k_Height.png", "wood", "height", "2k"),
            ("pine_bark_diff_2k.png", "pine_bark", "diff", "2k"),
            ("pine_bark_disp_4k.png", "pine_bark", "disp", "4k"),
            ("pine_bark_nor_gl_1k.png", "pine_bark", "nor_gl", "1k"),
            ("FabricSubstance003_COMPILED_baseColor_515.png", "fabricsubstance003_compiled", "basecolor", "515"),
            ("FabricSubstance003_COMPILED_normal_515.png", "fabricsubstance003_compiled", "normal", "515"),
            ("My Surface-Base-Colour-2048px.png", "my surface", "base_colour", "2048px"),
            ("My Surface_260x262_Normal_OpenGL.png", "my surface", "nor_gl", "260x262"),
            ("leather_DiffuseColour_1024.png", "leather", "diffusecolour", "1024"),
            ("leather_DisplacementMap_1024.png", "leather", "displacementmap", "1024"),
            ("leather_NRM_DX_1024.png", "leather", "nor_dx", "1024"),
            ("leather_RoughnessMap_1024.png", "leather", "roughnessmap", "1024"),
            ("stone_diff_2.png", "stone", "diff_2", "native")
        ]
        for (name, family, suffix, resolution) in cases {
            let parsed = try XCTUnwrap(NativeMaterialDatasetService.sourceMapFilename(name), name)
            XCTAssertEqual(parsed.0, family, name); XCTAssertEqual(parsed.1, suffix, name); XCTAssertEqual(parsed.2, resolution, name)
        }
        for name in ["IMG_6678.png", "my_room.png", "person_1k.png", "Paper003.png", "height_of_room_1k.png", "wood_gloss_2k.png", "wood_ORM_2k.png"] {
            XCTAssertNil(NativeMaterialDatasetService.sourceMapFilename(name), name)
            XCTAssertNil(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Surface/" + name)), name)
        }
        XCTAssertEqual(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Surface/Color.png"))?.0, "surface")
        XCTAssertEqual(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Surface/Base_Colour.png"))?.0, "surface")
        XCTAssertNil(NativeMaterialDatasetService.sourceMapFilename("Diffuse_Color.png"), "A standalone alias belongs to its enclosing material folder")
        XCTAssertNil(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Surface/wood_bump_2k.png")))
        XCTAssertEqual(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Surface/Cardboard_01_color_1K.png"))?.0, "cardboard_01")
        XCTAssertEqual(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Surface/My Surface_col01_1K.png"))?.0, "my_surface")
        XCTAssertEqual(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Surface/My Surface_col01_1K.png"))?.1, "color_1")
        XCTAssertNil(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Surface/Cardboard_01_height_1K.png")))
    }

    func testAmbientOcclusionAndPackedPBRMapsNeverBecomeDiffuseInput() throws {
        for name in ["ambientOcclusion.png", "Ambient_Occlusion.png", "AO.png", "ORM.png", "ARM.png",
                     "CarpetSubstance001_COMPILED_ambientOcclusion_8k.png",
                     "CarpetSubstance001_COMPILED_ambient_occlusion_2k.png",
                     "CarpetSubstance001_COMPILED_4K_AO.png", "carpet_Ambient-Occlusion-2048px.png",
                     "carpet_ao_2k.png", "carpet_rough_ao_2k.png", "carpet_ORM_2k.png",
                     "Paper003_2K-PNG_AmbientOcclusion.png"] {
            XCTAssertTrue(NativeMaterialDatasetService.isAncillarySourceFilename(name), name)
            XCTAssertNil(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Carpet/" + name)), name)
        }
        for name in ["CarpetSubstance001_COMPILED_baseColor_8k.png", "BaseColor.png", "height.png",
                     "carpet_roughness_2k.png", "IMG_6678.png"] {
            XCTAssertFalse(NativeMaterialDatasetService.isAncillarySourceFilename(name), name)
        }
        XCTAssertEqual(NativeMaterialDatasetService.diffuseSourceFilename(URL(fileURLWithPath: "/Sources/Carpet/CarpetSubstance001_COMPILED_baseColor_8k.png"))?.0,
                       "carpetsubstance001_compiled")
    }

    func testAuthoredAndAmbientImportsKeepEveryTargetExactOriginalIdentityAndPairedCrops() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Originals"), dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        var originals: [URL: Data] = [:], cardboard: [URL] = []
        for (family, dimension, format) in [("Cardboard_01", 512, "generic"), ("Paper003", 512, "ambient"), ("My Surface", 513, "native")] {
            let folder = sources.appendingPathComponent(family)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            for (generic, ambient, bits, channels) in [("color", "Color", 8, 3), ("height", "Displacement", 16, 1), ("normal", "NormalGL", 16, 3), ("roughness", "Roughness", 8, 1)] {
                let name = format == "ambient" ? family + "_1K-PNG_" + ambient + ".png"
                    : format == "generic" ? family + "_" + generic + "_1K.png" : family + "_" + (generic == "color" ? "BaseColor" : generic) + ".png"
                let nativeChannels = family == "Paper003" && generic == "color" ? 1 : channels
                let file = folder.appendingPathComponent(name), bytes = try image(width: dimension, height: dimension, bits: bits, channels: nativeChannels).encoded()
                try bytes.write(to: file); originals[file] = bytes
                if family == "Cardboard_01" { cardboard.append(file) }
            }
        }
        XCTAssertTrue(NativeMaterialDatasetService.croppingRecoveryRequests(files: cardboard, size: 256, target: "height").isEmpty,
                      "Generic authored filenames do not authorize a guessed provider download")
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "New source formats", "--training-size", "256", "--subject-validation", "disabled"])
        let imported = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--skip-source-recovery"])
        XCTAssertEqual(imported["source_set_count"] as? Int, 3)
        let classifier = try NativeDiffuseClassifierDatasetService.discover(dataset: dataset, sourceFolder: sources)
        XCTAssertEqual(Set(classifier.map(\.family)), ["cardboard_01", "paper003", "my_surface"])
        XCTAssertEqual(classifier.count, 3, "Dataset and folder discovery share each original diffuse identity")
        for target in ["height", "normal", "roughness"] {
            let prepared = try await command(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target, "--skip-source-recovery"])
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: target)
            XCTAssertEqual(samples.count, 3, "Every explicit map format supplies \(target)")
            for sample in samples {
                let inputSource = try XCTUnwrap(sample.augmentationSourceInputURL), targetSource = try XCTUnwrap(sample.augmentationSourceTargetURL)
                let rectangle = try XCTUnwrap(sample.cropRectangle)
                let diffuse = try NativePNG.decode(Data(contentsOf: sample.inputURL)), output = try NativePNG.decode(Data(contentsOf: sample.targetURL))
                XCTAssertEqual(diffuse.pixels, try NativePNG.crop(inputSource, rectangle: rectangle).pixels)
                XCTAssertEqual(output.pixels, try NativePNG.crop(targetSource, rectangle: rectangle).pixels)
                XCTAssertEqual(output.header.bits, target == "roughness" ? 8 : 16)
                XCTAssertEqual(diffuse.header.width, 256); XCTAssertEqual(output.header.height, 256)
                if inputSource.lastPathComponent == "Paper003_1K-PNG_Color.png" {
                    XCTAssertEqual(diffuse.header.channels, 1, "Grayscale diffuse crops preserve original native channels")
                }
                XCTAssertEqual(sample.inputSHA256, NativeMaterialTrainer.checksum(try Data(contentsOf: sample.inputURL)))
                XCTAssertEqual(sample.targetSHA256, NativeMaterialTrainer.checksum(try Data(contentsOf: sample.targetURL)))
                XCTAssertFalse(try XCTUnwrap(sample.materialName).isEmpty, "Training descriptors retain the registered material name for model cards")
            }
            _ = try await command(["cleanup-size", "--dataset", preparedURL.path])
        }
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes, "Original source bytes and paths are immutable") }
    }

    func testExplicitGrayAndGrayAlphaSourcesPreserveNativeCropCodesAtBothPrecisions() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Grayscale sources", "--training-size", "256", "--subject-validation", "disabled"])
        var originals: [URL: Data] = [:]
        for bits in [8, 16] {
            for channels in [1, 2] {
                let name = "Gray \(bits) Channels \(channels)", folder = root.appendingPathComponent(name)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
                let diffuse = folder.appendingPathComponent("diffuse.png"), height = folder.appendingPathComponent("height.png")
                for (path, precision, count) in [(diffuse, bits, channels), (height, 16, 1)] {
                    let bytes = try image(width: 260, height: 262, bits: precision, channels: count).encoded()
                    try bytes.write(to: path); originals[path] = bytes
                }
                _ = try await command(["add-material", "--dataset", dataset.path, "--name", name, "--input", diffuse.path, "--height", height.path, "--skip-source-recovery"])
            }
        }
        let result = try await command(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--skip-source-recovery"])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertEqual(samples.count, 4)
        for sample in samples {
            let source = try XCTUnwrap(sample.augmentationSourceInputURL), rectangle = try XCTUnwrap(sample.cropRectangle)
            let actual = try NativePNG.decode(Data(contentsOf: sample.inputURL)), expected = try NativePNG.crop(source, rectangle: rectangle)
            XCTAssertEqual(actual.header, expected.header); XCTAssertEqual(actual.pixels, expected.pixels)
            XCTAssertTrue([1, 2].contains(actual.header.channels))
        }
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
    }

    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments)
        let result = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(result.utf8)) as? [String: Any])
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-filenames-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); return root
    }
    private func image(width: Int, height: Int, bits: Int, channels: Int) -> NativePNG {
        let pixels = Data((0..<width * height * channels * bits / 8).map { UInt8(truncatingIfNeeded: $0 * 97) })
        return NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels, color: [1: UInt8(0), 2: 4, 3: 2, 4: 6][channels]!, interlace: 0),
                         pixels: pixels, colorChunks: [])
    }
}

final class NativeDiffuseClassifierDynamicClassTests: XCTestCase {
    func testCustomClassFoldersAndManualExtrasEnterCreateMLWithoutDuplicatingNewOriginals() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("classifier-custom-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Curated"), original = root.appendingPathComponent("wood_planks_diff_1k.png")
        let woodBytes = try png(51).encoded(); try woodBytes.write(to: original)
        let wood = NativeDiffuseClassifierDatasetService.Source(family: "wood_planks", variant: "color_1", url: original, width: 7, height: 9, metadata: nil)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: [wood], destination: destination)
        let old = try manifest(destination), leather = destination.appendingPathComponent("Leather")
        try FileManager.default.createDirectory(at: leather, withIntermediateDirectories: false)
        let moved = leather.appendingPathComponent(try XCTUnwrap(old.entries.first?.filename))
        try FileManager.default.moveItem(at: URL(fileURLWithPath: try XCTUnwrap(old.entries.first?.currentPath)), to: moved)
        try FileManager.default.removeItem(at: destination.appendingPathComponent("Plastic Rubber Clay"))
        let paperSource = root.appendingPathComponent("Paper005_1K-PNG_Color.png"), paperBytes = try png(93).encoded()
        try paperBytes.write(to: paperSource)
        let paper = destination.appendingPathComponent("Paper/" + paperSource.lastPathComponent)
        try paperBytes.write(to: paper)
        let drawing = destination.appendingPathComponent("Paper/Drawing_1.png"), drawingBytes = try png(135).encoded()
        try drawingBytes.write(to: drawing)
        let source = NativeDiffuseClassifierDatasetService.Source(family: "paper005", variant: "color_1", url: paperSource, width: 7, height: 9, metadata: nil)
        let text = try NativeDiffuseClassifierDatasetService.refresh(sources: [wood, source], destination: destination)
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["total"] as? Int, 3); XCTAssertEqual(result["added"] as? Int, 0)
        let updated = try manifest(destination)
        let userClass = try XCTUnwrap(updated.classes.first { $0.name == "Leather" })
        XCTAssertTrue(userClass.id.hasPrefix("user_"))
        XCTAssertEqual(updated.entries.first { $0.family == "wood_planks" }?.currentClass, userClass.id)
        XCTAssertEqual(updated.entries.first { $0.family == "paper005" }?.currentPath, paper.path)
        XCTAssertEqual(updated.entries.first { $0.family == "paper005" }?.status, "adopted_user_class")
        XCTAssertFalse(updated.classes.contains { $0.name == "Plastic Rubber Clay" })
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.appendingPathComponent("Plastic Rubber Clay").path))
        let training = URL(fileURLWithPath: try XCTUnwrap(result["training_path"] as? String))
        let labels = try MLImageClassifier.DataSource.labeledDirectories(at: training).labeledImages()
        XCTAssertEqual(labels["Paper"]?.count, 2); XCTAssertEqual(labels["Leather"]?.count, 1)
        XCTAssertEqual(try Data(contentsOf: drawing), drawingBytes); XCTAssertEqual(try Data(contentsOf: paper), paperBytes)
        XCTAssertEqual(try Data(contentsOf: moved), woodBytes)
        _ = try NativeDiffuseClassifierDatasetService.refresh(sources: [wood, source], destination: destination)
        XCTAssertEqual(try manifest(destination).classes.first { $0.name == "Leather" }?.id, userClass.id)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: destination.appendingPathComponent("Paper").path).count, 2)
    }

    private func manifest(_ root: URL) throws -> NativeDiffuseClassifierDatasetService.Manifest {
        try JSONDecoder().decode(NativeDiffuseClassifierDatasetService.Manifest.self, from: Data(contentsOf: root.appendingPathComponent(".classifier-manifest.json")))
    }
    private func png(_ value: UInt8) -> NativePNG {
        NativePNG(header: .init(width: 7, height: 9, bits: 8, channels: 3, color: 2, interlace: 0),
                  pixels: Data(repeating: value, count: 7 * 9 * 3), colorChunks: [])
    }
}
