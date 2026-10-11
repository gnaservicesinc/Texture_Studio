import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialCropAugmentationTests: XCTestCase {
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

    func testMarginalCenterSourceKeepsPairedCachedPixelsAndFlipsWithoutPadding() throws {
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
            XCTAssertEqual(augmentation.history.last?["source_center_spin_degrees"] as? Int, 0)
            XCTAssertEqual(augmentation.history.last?["center_spin_skipped_insufficient_support"] as? Bool, true)
        }
        XCTAssertEqual(predictions, 1)
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
            XCTAssertTrue((1...179).contains(angle))
            let fresh = try NativeMaterialCropAugmentation.spunCenter(source: originalTarget,
                expectedSHA256: NativeMaterialTrainer.checksum(targetOriginal), size: 4, degrees: angle)
            let expected = try NativeMaterialCropAugmentation.flipped(fresh,
                horizontal: receipt["horizontal_flip"] as? Bool ?? false, vertical: receipt["vertical_flip"] as? Bool ?? false)
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: current[0].targetURL)).pixels, expected.pixels)
        }
        XCTAssertEqual(predictions, 2)
        XCTAssertEqual(try Data(contentsOf: originalInput), inputOriginal)
        XCTAssertEqual(try Data(contentsOf: originalTarget), targetOriginal)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
        XCTAssertEqual(try Data(contentsOf: target), targetBytes)
    }

    private func temporary() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("crop-augmentation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true); return url
    }
    private func png(width: Int, height: Int, channels: Int, codes: [Int]) -> NativePNG {
        var pixels = Data()
        for value in codes { pixels.append(UInt8(value >> 8)); pixels.append(UInt8(value & 255)) }
        return .init(header: .init(width: width, height: height, bits: 16, channels: channels, color: channels == 1 ? 0 : 2, interlace: 0), pixels: pixels, colorChunks: [])
    }
    private func codes(_ image: NativePNG) -> [Int] {
        stride(from: 0, to: image.pixels.count, by: 2).map { Int(image.pixels[$0]) * 256 + Int(image.pixels[$0 + 1]) }
    }
}
