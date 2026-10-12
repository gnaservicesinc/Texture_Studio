import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialCropAugmentationTests: XCTestCase {
    func testUnchangedCropCopiesExactBytesAndReusesOriginalDepthProIdentity() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appendingPathComponent("input.png"), target = root.appendingPathComponent("height.png")
        let original = png(width: 4, height: 4, channels: 3, codes: (0..<16).flatMap { [$0 * 1000, $0 * 1000, $0 * 1000] })
        let diffuse = NativePNG(header: original.header, pixels: original.pixels,
            colorChunks: [("tEXt", Data("note\0keep original compressed crop".utf8))])
        let inputBytes = try diffuse.encoded()
        let targetBytes = try png(width: 4, height: 4, channels: 1, codes: (0..<16).map { $0 * 1000 }).encoded()
        try inputBytes.write(to: input); try targetBytes.write(to: target)
        let sample = Sample(id: "paired", split: "train", width: 4, height: 4,
            inputURL: input, targetURL: target, inputSHA256: NativeMaterialTrainer.checksum(inputBytes),
            targetSHA256: NativeMaterialTrainer.checksum(targetBytes), inputEncoding: "srgb", targetConvention: "opengl")
        let seed = try XCTUnwrap((UInt64(0)..<100).first { seed in
            var random = NativeMaterialRandom(seed: seed ^ 0x63726f70732d7632)
            return random.next() % 4 == 0
        })
        var predictions = 0
        let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "a", count: 64), directory: root.appendingPathComponent("depth")) { _ in
            predictions += 1
            return .init(width: 4, height: 4, values: (1...16).map { Float($0) }, outputName: "depthMeters", interpretation: "Camera depth in meters")
        }
        _ = try cache.input(imageURL: input, expectedSHA256: sample.inputSHA256, width: 4, height: 4)
        let augmentation = NativeMaterialCropAugmentation(directory: root.appendingPathComponent("crop-cache"), target: "height", seed: seed)
        var completed: [Int] = []
        let refreshed = try augmentation.refresh([sample], epoch: 1, flip: true, spin: false, depthProCache: cache, check: {},
            onProgress: { done, total, id in
                completed.append(done); XCTAssertEqual(total, 1); XCTAssertEqual(id, sample.id)
            })
        XCTAssertEqual(completed, [1])
        XCTAssertEqual(try Data(contentsOf: refreshed[0].inputURL), inputBytes)
        XCTAssertEqual(try Data(contentsOf: refreshed[0].targetURL), targetBytes)
        XCTAssertEqual(refreshed[0].inputSHA256, sample.inputSHA256)
        _ = try cache.input(imageURL: refreshed[0].inputURL, expectedSHA256: refreshed[0].inputSHA256, width: 4, height: 4)
        XCTAssertEqual(predictions, 1, "Copying unchanged bytes must keep the existing prediction identity")
    }

    func testOnePairAtATimeKeepsColorVariantsAlignedWithinEachEpoch() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appendingPathComponent("input.png"), target = root.appendingPathComponent("height.png")
        let inputBytes = try png(width: 4, height: 4, channels: 3, codes: (0..<16).flatMap { [$0 * 1000, $0 * 1000, $0 * 1000] }).encoded()
        let targetBytes = try png(width: 4, height: 4, channels: 1, codes: (0..<16).map { $0 * 1000 }).encoded()
        try inputBytes.write(to: input); try targetBytes.write(to: target)
        var samples = ["a", "b"].map { variant in
            Sample(id: "paired|" + variant, split: "train", width: 4, height: 4,
                inputURL: input, targetURL: target, inputSHA256: NativeMaterialTrainer.checksum(inputBytes),
                targetSHA256: NativeMaterialTrainer.checksum(targetBytes), inputEncoding: "srgb", targetConvention: "opengl")
        }
        let augmentation = NativeMaterialCropAugmentation(directory: root.appendingPathComponent("crop-cache"), target: "height", seed: 17)
        for epoch in 1...3 {
            for index in samples.indices {
                samples[index] = try augmentation.refresh([samples[index]], epoch: epoch, flip: true, spin: false, depthProCache: nil, check: {})[0]
            }
            XCTAssertEqual(samples[0].targetSHA256, samples[1].targetSHA256)
            XCTAssertEqual(samples[0].targetURL, samples[1].targetURL)
            for sample in samples {
                let rgb = try NativePNG.decode(Data(contentsOf: sample.inputURL))
                let scalar = try NativePNG.decode(Data(contentsOf: sample.targetURL))
                XCTAssertEqual(stride(from: 0, to: 48, by: 3).map { codes(rgb)[$0] }, codes(scalar))
            }
        }
        XCTAssertEqual(augmentation.history.count, 6)
        let journal = try String(contentsOf: augmentation.directory.appendingPathComponent("crop-history.jsonl"), encoding: .utf8)
        XCTAssertEqual(journal.split(separator: "\n").count, 6)
        XCTAssertFalse(FileManager.default.fileExists(atPath: augmentation.directory.appendingPathComponent("crop-history.json").path))
        try augmentation.saveHistory()
        let saved = try NativeMaterialTransfer.object(augmentation.directory.appendingPathComponent("crop-history.json"))
        XCTAssertEqual((saved["history"] as? [[String: Any]])?.count, 6)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
        XCTAssertEqual(try Data(contentsOf: target), targetBytes)
    }

    private typealias Sample = NativeMaterialDatasetService.TrainingSample

    func testIntegerFlipsPreserveCodesAndReflectNormalAxes() throws {
        let image = png(width: 3, height: 2, channels: 1, codes: [1, 2, 65535, 4, 32768, 6])
        let result = try NativeMaterialCropAugmentation.flipped(image, horizontal: true, vertical: true)
        XCTAssertEqual(codes(result), [6, 32768, 4, 65535, 2, 1])
        XCTAssertEqual(try NativeMaterialCropAugmentation.flipped(result, horizontal: true, vertical: true).pixels, image.pixels)
        let normal = png(width: 1, height: 1, channels: 3, codes: [65535, 32768, 50000])
        XCTAssertEqual(codes(try NativeMaterialCropAugmentation.flipped(normal, horizontal: true, vertical: true, normal: true)), [0, 32767, 50000])
    }

    func testCenterSpinSamplesOriginalPairedPixelsWithoutChangingSources() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("height.png")
        let image = png(width: 8, height: 8, channels: 1, codes: (0..<64).map { $0 * 1000 })
        let bytes = try image.encoded(); try bytes.write(to: source)
        let sha = NativeMaterialTrainer.checksum(bytes)
        let spun = try NativeMaterialCropAugmentation.spunCenter(source: source, expectedSHA256: sha, size: 4, degrees: 90)
        XCTAssertEqual(codes(spun), [42000, 34000, 26000, 18000, 43000, 35000, 27000, 19000, 44000, 36000, 28000, 20000, 45000, 37000, 29000, 21000])
        XCTAssertEqual(try Data(contentsOf: source), bytes)
        XCTAssertEqual(spun.header.bits, 16)
        XCTAssertThrowsError(try NativeMaterialCropAugmentation.spunCenter(source: source, expectedSHA256: sha, size: 8, degrees: 45), "No introduced border pixels or padding")
    }

    func testQuarterTurnsKeepExactPairedCodesAtBothPrecisionsAndOddSourceDimensions() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for bits in [8, 16] {
            let ramp = (0..<35).map { $0 * (bits == 8 ? 7 : 1700) }
            let height = png(width: 5, height: 7, channels: 1, codes: ramp, bits: bits)
            let diffuse = png(width: 5, height: 7, channels: 3, codes: ramp.flatMap { [$0, $0, $0] }, bits: bits)
            let heightURL = root.appendingPathComponent("height-\(bits).png"), diffuseURL = root.appendingPathComponent("diffuse-\(bits).png")
            let heightBytes = try height.encoded(), diffuseBytes = try diffuse.encoded()
            try heightBytes.write(to: heightURL); try diffuseBytes.write(to: diffuseURL)
            let center = try height.crop([0, 1, 4, 4])
            for degrees in [90, 180, 270] {
                let rotated = try NativeMaterialCropAugmentation.spunCenter(source: heightURL,
                    expectedSHA256: NativeMaterialTrainer.checksum(heightBytes), size: 4, degrees: degrees)
                let paired = try NativeMaterialCropAugmentation.spunCenter(source: diffuseURL,
                    expectedSHA256: NativeMaterialTrainer.checksum(diffuseBytes), size: 4, degrees: degrees)
                XCTAssertEqual(codes(rotated).sorted(), codes(center).sorted(), "Rotations introduce no interpolated codes, even on odd source grids")
                XCTAssertEqual(stride(from: 0, to: paired.pixels.count / (bits / 8), by: 3).map { codes(paired)[$0] }, codes(rotated))
                XCTAssertEqual(try NativeMaterialCropAugmentation.quarterTurn(rotated, degrees: 360 - degrees).pixels, center.pixels)
                XCTAssertEqual(rotated.header.bits, bits); XCTAssertEqual(paired.header.bits, bits)
            }
            XCTAssertEqual(try Data(contentsOf: heightURL), heightBytes)
            XCTAssertEqual(try Data(contentsOf: diffuseURL), diffuseBytes)
        }
    }

    func testRotationsRejectInterpolatedAnglesAndPreserveHighContrastEdges() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("edge.png")
        let image = png(width: 12, height: 12, channels: 1, codes: (0..<144).map { $0 % 12 < 6 ? 0 : 65535 })
        let bytes = try image.encoded(); try bytes.write(to: source)
        let sha = NativeMaterialTrainer.checksum(bytes)
        for degrees in [15, 33, 45, 174, 360] {
            XCTAssertThrowsError(try NativeMaterialCropAugmentation.spunCenter(source: source,
                expectedSHA256: sha, size: 6, degrees: degrees))
        }
        for degrees in [90, 180, 270] {
            let rotated = try NativeMaterialCropAugmentation.spunCenter(source: source,
                expectedSHA256: sha, size: 6, degrees: degrees)
            XCTAssertEqual(Set(codes(rotated)), [0, 65535], "Sharp boundaries must contain only exact source codes")
        }
        let native = try NativeMaterialCropAugmentation.spunCenter(source: source,
            expectedSHA256: sha, size: 12, degrees: 90)
        XCTAssertEqual(codes(native).sorted(), codes(image).sorted(), "A native-sized field needs no surrounding padding for quarter turns")
    }

    func testNormalQuarterTurnsOnlyPermuteAndReflectAxesWithoutRenormalizingCodes() throws {
        for bits in [8, 16] {
            let maximum = bits == 8 ? 255 : 65535
            let nx = maximum / 3, ny = maximum / 5, nz = maximum / 7
            let image = png(width: 1, height: 1, channels: 3, codes: [nx, ny, nz], bits: bits)
            XCTAssertEqual(codes(try NativeMaterialCropAugmentation.quarterTurn(image, degrees: 90, normal: true)), [ny, maximum - nx, nz])
            XCTAssertEqual(codes(try NativeMaterialCropAugmentation.quarterTurn(image, degrees: 180, normal: true)), [maximum - nx, maximum - ny, nz])
            XCTAssertEqual(codes(try NativeMaterialCropAugmentation.quarterTurn(image, degrees: 270, normal: true)), [maximum - ny, nx, nz])
            var roundtrip = image
            for _ in 0..<4 { roundtrip = try NativeMaterialCropAugmentation.quarterTurn(roundtrip, degrees: 90, normal: true) }
            XCTAssertEqual(roundtrip.pixels, image.pixels, "Normal-axis transforms are exactly reversible at source precision")
        }
    }

    func testRetainedCedarRotationExportsCorrected1024WithoutChangingSourcesWhenRequested() throws {
        guard let outputPath = ProcessInfo.processInfo.environment["TEXTURE_STUDIO_ROTATION_CHECK_OUTPUT"], !outputPath.isEmpty else {
            throw XCTSkip("Set TEXTURE_STUDIO_ROTATION_CHECK_OUTPUT to export the retained cedar rotation spot check.")
        }
        let folder = URL(fileURLWithPath: "/opt/ipde/sources_mats/japanese_cedar_planks")
        let heightSource = folder.appendingPathComponent("japanese_cedar_planks_disp_2k.png")
        let diffuseSource = folder.appendingPathComponent("japanese_cedar_planks_diff_2k.png")
        guard FileManager.default.fileExists(atPath: heightSource.path), FileManager.default.fileExists(atPath: diffuseSource.path) else {
            throw XCTSkip("The optional retained public cedar fixture is unavailable.")
        }
        let output = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let heightBytes = try Data(contentsOf: heightSource), diffuseBytes = try Data(contentsOf: diffuseSource)
        let heightSHA = NativeMaterialTrainer.checksum(heightBytes), diffuseSHA = NativeMaterialTrainer.checksum(diffuseBytes)
        let corrected = try NativeMaterialCropAugmentation.spunCenter(source: heightSource,
            expectedSHA256: heightSHA, size: 1024, degrees: 270)
        let paired = try NativeMaterialCropAugmentation.spunCenter(source: diffuseSource,
            expectedSHA256: diffuseSHA, size: 1024, degrees: 270)
        XCTAssertEqual(corrected.header.width, 1024); XCTAssertEqual(corrected.header.height, 1024)
        XCTAssertEqual(corrected.header.bits, 16); XCTAssertEqual(corrected.header.channels, 1)
        XCTAssertEqual(paired.header.width, 1024); XCTAssertEqual(paired.header.height, 1024)
        let values = codes(corrected), minimum = try XCTUnwrap(values.min()), maximum = try XCTUnwrap(values.max())
        XCTAssertGreaterThan(maximum, minimum)
        XCTAssertGreaterThanOrEqual(minimum, 0); XCTAssertLessThanOrEqual(maximum, 65535)
        let heightOutput = output.appendingPathComponent("cedar-height-exact-270deg-1024.png")
        let diffuseOutput = output.appendingPathComponent("cedar-diffuse-exact-270deg-1024.png")
        try corrected.encoded().write(to: heightOutput, options: .atomic)
        try paired.encoded().write(to: diffuseOutput, options: .atomic)
        let report: [String: Any] = ["center_spin_sampling": NativeMaterialCropAugmentation.centerSpinSampling,
            "degrees": 270, "width": 1024, "height": 1024, "height_bits": corrected.header.bits,
            "height_source": heightSource.path, "height_source_sha256": heightSHA,
            "diffuse_source": diffuseSource.path, "diffuse_source_sha256": diffuseSHA,
            "height_output": heightOutput.path, "diffuse_output": diffuseOutput.path,
            "minimum_height_code": minimum, "maximum_height_code": maximum, "original_sources_modified": false]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("cedar-rotation-check.json"), options: .atomic)
        XCTAssertEqual(NativeMaterialTrainer.checksum(try Data(contentsOf: heightSource)), heightSHA)
        XCTAssertEqual(NativeMaterialTrainer.checksum(try Data(contentsOf: diffuseSource)), diffuseSHA)
    }

    func testProgressiveFlipsUseCachedPairsAndReuseDepthProPrediction() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appendingPathComponent("input.png"), target = root.appendingPathComponent("height.png")
        let diffuse = png(width: 4, height: 4, channels: 3, codes: (0..<16).flatMap { [$0 * 1000, $0 * 1000, $0 * 1000] })
        let height = png(width: 4, height: 4, channels: 1, codes: (0..<16).map { $0 * 1000 })
        let inputBytes = try diffuse.encoded(), targetBytes = try height.encoded()
        try inputBytes.write(to: input); try targetBytes.write(to: target)
        let sample = NativeMaterialDatasetService.TrainingSample(id: "paired", split: "train", width: 4, height: 4,
            inputURL: input, targetURL: target, inputSHA256: NativeMaterialTrainer.checksum(inputBytes),
            targetSHA256: NativeMaterialTrainer.checksum(targetBytes), inputEncoding: "srgb", targetConvention: "opengl")
        var predictions = 0
        let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "a", count: 64), directory: root.appendingPathComponent("depth")) { _ in
            predictions += 1
            return .init(width: 4, height: 4, values: (1...16).map { Float($0) }, outputName: "depthMeters", interpretation: "Camera depth in meters")
        }
        let augmentation = NativeMaterialCropAugmentation(directory: root.appendingPathComponent("crop-cache"), target: "height", seed: 17)
        var current = [sample]
        for epoch in 1...4 {
            current = try augmentation.refresh(current, epoch: epoch, flip: true, spin: false, depthProCache: cache, check: {})
            let rgb = try NativePNG.decode(Data(contentsOf: current[0].inputURL))
            let scalar = try NativePNG.decode(Data(contentsOf: current[0].targetURL))
            XCTAssertEqual(stride(from: 0, to: 48, by: 3).map { codes(rgb)[$0] }, codes(scalar))
            _ = try cache.input(imageURL: current[0].inputURL, expectedSHA256: current[0].inputSHA256, width: 4, height: 4)
        }
        XCTAssertEqual(predictions, 1)
        XCTAssertEqual(augmentation.history.count, 4)
        XCTAssertTrue(augmentation.history.allSatisfy { $0["progressive_cached_flip"] as? Bool == true })
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
        XCTAssertEqual(try Data(contentsOf: target), targetBytes)
    }

    func testNativeCropOptionsDisableRememberedFlagsAndRejectInvalidTiming() throws {
        let base = ["train", "--dataset", "/dataset", "--output", "/output", "--model-family", "compact-scalar"]
        let never = try NativeMaterialTrainer.Options(base + ["--randomize-crops", "never", "--random-flip", "--random-spin-centercut"])
        XCTAssertFalse(never.randomFlip); XCTAssertFalse(never.randomSpinCentercut)
        let epoch = try NativeMaterialTrainer.Options(base + ["--randomize-crops", "epoch", "--random-flip", "--random-spin-centercut"])
        XCTAssertTrue(epoch.randomFlip); XCTAssertTrue(epoch.randomSpinCentercut)
        XCTAssertEqual(epoch.trainingConfiguration["randomize_crops"] as? String, "epoch")
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(base + ["--randomize-crops", "sometimes"]))
    }

    func testMarginalCenterSourceRotatesExactPairedPixelsWithoutPadding() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let originalInput = root.appendingPathComponent("original-diffuse.png"), originalTarget = root.appendingPathComponent("original-height.png")
        let diffuse = png(width: 4, height: 5, channels: 3, codes: (0..<20).flatMap { [$0 * 1000, $0 * 1000, $0 * 1000] })
        let height = png(width: 4, height: 5, channels: 1, codes: (0..<20).map { $0 * 1000 })
        let originalRGB = try diffuse.encoded(), originalHeight = try height.encoded()
        try originalRGB.write(to: originalInput); try originalHeight.write(to: originalTarget)
        let input = root.appendingPathComponent("crop-diffuse.png"), target = root.appendingPathComponent("crop-height.png")
        let inputBytes = try diffuse.crop([0, 0, 4, 4]).encoded(), targetBytes = try height.crop([0, 0, 4, 4]).encoded()
        try inputBytes.write(to: input); try targetBytes.write(to: target)
        let sample = NativeMaterialDatasetService.TrainingSample(id: "marginal-center", split: "train", width: 4, height: 4,
            inputURL: input, targetURL: target, inputSHA256: NativeMaterialTrainer.checksum(inputBytes),
            targetSHA256: NativeMaterialTrainer.checksum(targetBytes), inputEncoding: "srgb", targetConvention: "opengl",
            augmentationSourceInputURL: originalInput, augmentationSourceInputSHA256: NativeMaterialTrainer.checksum(originalRGB),
            augmentationSourceTargetURL: originalTarget, augmentationSourceTargetSHA256: NativeMaterialTrainer.checksum(originalHeight), cropKind: "center")
        var predictions = 0
        let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "c", count: 64), directory: root.appendingPathComponent("depth")) { _ in
            predictions += 1
            return .init(width: 4, height: 4, values: (1...16).map { Float($0) }, outputName: "depthMeters", interpretation: "Camera depth in meters")
        }
        let augmentation = NativeMaterialCropAugmentation(directory: root.appendingPathComponent("crop-cache"), target: "height", seed: 17)
        var current = [sample]
        for epoch in 1...4 {
            current = try augmentation.refresh(current, epoch: epoch, flip: true, spin: true, depthProCache: cache, check: {})
            let rgb = try NativePNG.decode(Data(contentsOf: current[0].inputURL)), scalar = try NativePNG.decode(Data(contentsOf: current[0].targetURL))
            XCTAssertEqual(stride(from: 0, to: 48, by: 3).map { codes(rgb)[$0] }, codes(scalar))
            XCTAssertEqual(codes(scalar).sorted(), (0..<16).map { $0 * 1000 })
            XCTAssertTrue([90, 180, 270].contains(augmentation.history.last?["source_center_spin_degrees"] as? Int ?? -1))
            XCTAssertEqual(augmentation.history.last?["center_spin_skipped_insufficient_support"] as? Bool, false)
        }
        XCTAssertGreaterThanOrEqual(predictions, 1)
        XCTAssertLessThanOrEqual(predictions, Set(augmentation.history.compactMap { $0["source_center_spin_degrees"] as? Int }).count,
                                 "An exact flip may already have cached an equivalent quarter-turn orientation")
        XCTAssertEqual(try Data(contentsOf: originalInput), originalRGB)
        XCTAssertEqual(try Data(contentsOf: originalTarget), originalHeight)
    }

    func testCenterSpinsReturnToOriginalAndRegenerateDepthBeforeAlignedFlip() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let originalInput = root.appendingPathComponent("original-diffuse.png"), originalTarget = root.appendingPathComponent("original-height.png")
        let sourceInput = png(width: 8, height: 8, channels: 3, codes: (0..<64).flatMap { [$0 * 1000, $0 * 1000, $0 * 1000] })
        let sourceTarget = png(width: 8, height: 8, channels: 1, codes: (0..<64).map { $0 * 1000 })
        let inputOriginal = try sourceInput.encoded(), targetOriginal = try sourceTarget.encoded()
        try inputOriginal.write(to: originalInput); try targetOriginal.write(to: originalTarget)
        let input = root.appendingPathComponent("crop-diffuse.png"), target = root.appendingPathComponent("crop-height.png")
        let inputBytes = try sourceInput.crop([2, 2, 4, 4]).encoded(), targetBytes = try sourceTarget.crop([2, 2, 4, 4]).encoded()
        try inputBytes.write(to: input); try targetBytes.write(to: target)
        let sample = NativeMaterialDatasetService.TrainingSample(id: "center", split: "train", width: 4, height: 4,
            inputURL: input, targetURL: target, inputSHA256: NativeMaterialTrainer.checksum(inputBytes),
            targetSHA256: NativeMaterialTrainer.checksum(targetBytes), inputEncoding: "srgb", targetConvention: "opengl",
            augmentationSourceInputURL: originalInput, augmentationSourceInputSHA256: NativeMaterialTrainer.checksum(inputOriginal),
            augmentationSourceTargetURL: originalTarget, augmentationSourceTargetSHA256: NativeMaterialTrainer.checksum(targetOriginal),
            cropKind: "center")
        var predictions = 0
        let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "b", count: 64), directory: root.appendingPathComponent("depth")) { _ in
            predictions += 1; return .init(width: 4, height: 4, values: (1...16).map { Float($0) }, outputName: "depthMeters", interpretation: "Camera depth in meters")
        }
        let augmentation = NativeMaterialCropAugmentation(directory: root.appendingPathComponent("crop-cache"), target: "height", seed: 31)
        let incomplete = NativeMaterialDatasetService.TrainingSample(id: "incomplete", split: "train", width: 4, height: 4,
            inputURL: input, targetURL: target, inputSHA256: sample.inputSHA256, targetSHA256: sample.targetSHA256,
            inputEncoding: "srgb", targetConvention: "opengl", augmentationSourceInputURL: originalInput,
            augmentationSourceInputSHA256: sample.augmentationSourceInputSHA256, cropKind: "center")
        XCTAssertThrowsError(try augmentation.refresh([incomplete], epoch: 1, flip: false, spin: true, depthProCache: cache, check: {}),
            "A center spin must never rotate only one half of a pair")
        var current = [sample]
        for epoch in 1...2 {
            current = try augmentation.refresh(current, epoch: epoch, flip: true, spin: true, depthProCache: cache, check: {})
            let receipt = try XCTUnwrap(augmentation.history.last)
            let angle = try XCTUnwrap(receipt["source_center_spin_degrees"] as? Int)
            XCTAssertTrue([90, 180, 270].contains(angle))
            XCTAssertEqual(receipt["center_spin_sampling"] as? String, "exact-quarter-turn-pixel-permutation-v1")
            XCTAssertEqual(receipt["source_pixels_resized"] as? Bool, false)
            XCTAssertEqual(receipt["source_pixels_interpolated"] as? Bool, false)
            let fresh = try NativeMaterialCropAugmentation.spunCenter(source: originalTarget,
                expectedSHA256: NativeMaterialTrainer.checksum(targetOriginal), size: 4, degrees: angle)
            let expected = try NativeMaterialCropAugmentation.flipped(fresh,
                horizontal: receipt["horizontal_flip"] as? Bool ?? false, vertical: receipt["vertical_flip"] as? Bool ?? false)
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: current[0].targetURL)).pixels, expected.pixels)
        }
        XCTAssertGreaterThanOrEqual(predictions, 1)
        XCTAssertLessThanOrEqual(predictions, Set(augmentation.history.compactMap { $0["source_center_spin_degrees"] as? Int }).count,
                                 "An exact flip may already have cached an equivalent quarter-turn orientation")
        XCTAssertEqual(try Data(contentsOf: originalInput), inputOriginal)
        XCTAssertEqual(try Data(contentsOf: originalTarget), targetOriginal)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
        XCTAssertEqual(try Data(contentsOf: target), targetBytes)
    }

    private func temporary() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("crop-augmentation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true); return url
    }
    private func png(width: Int, height: Int, channels: Int, codes: [Int], bits: Int = 16) -> NativePNG {
        var pixels = Data()
        for value in codes {
            if bits == 16 { pixels.append(UInt8(value >> 8)) }
            pixels.append(UInt8(value & 255))
        }
        return .init(header: .init(width: width, height: height, bits: bits, channels: channels, color: channels == 1 ? 0 : 2, interlace: 0), pixels: pixels, colorChunks: [])
    }
    private func codes(_ image: NativePNG) -> [Int] {
        if image.header.bits == 8 { return image.pixels.map(Int.init) }
        return stride(from: 0, to: image.pixels.count, by: 2).map { Int(image.pixels[$0]) * 256 + Int(image.pixels[$0 + 1]) }
    }
}
