import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeCompactMaterialModelTests: XCTestCase {
    func testTrainingSkipsPredictionReadbackAndReusesOnlyExactParameterFeeds() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 31)
            let program = try model.program(width: 64, height: 64, target: "height")
            let input = Self.rgb(width: 64, height: 64)
            let reference = (0..<64 * 64).map { Float(($0 * 11) % 251) / 251 }
            XCTAssertThrowsError(try program.execute(rgb: input, weights: [:], reference: reference,
                gradientsOnly: true, includePrediction: false))
            XCTAssertEqual(program.executionStatistics.parameterFeedUploads, 0)
            XCTAssertEqual(program.executionStatistics.parameterFeedCacheHits, 0)
            XCTAssertEqual(program.executionStatistics.compilations, 0)
            let withPrediction = try program.execute(rgb: input, weights: model.weights, reference: reference, gradientsOnly: true)
            let training = try program.execute(rgb: input, weights: model.weights, reference: reference,
                gradientsOnly: true, includePrediction: false)
            XCTAssertEqual(withPrediction.output.count, 64 * 64)
            XCTAssertTrue(training.output.isEmpty)
            XCTAssertEqual(try XCTUnwrap(training.loss), try XCTUnwrap(withPrediction.loss), accuracy: 1e-6)
            XCTAssertEqual(try XCTUnwrap(training.valueLoss), try XCTUnwrap(withPrediction.valueLoss), accuracy: 1e-6)
            XCTAssertEqual(try XCTUnwrap(training.gradientLoss), try XCTUnwrap(withPrediction.gradientLoss), accuracy: 1e-6)
            for name in model.weights.keys {
                for (expected, actual) in zip(try withPrediction.gradients[name]!.floatValues(), try training.gradients[name]!.floatValues()) {
                    XCTAssertEqual(actual, expected, accuracy: 1e-5, name)
                }
            }
            XCTAssertEqual(program.executionStatistics.parameterFeedUploads, 1)
            XCTAssertEqual(program.executionStatistics.parameterFeedCacheHits, 1)
            var changed = model.weights
            let originalBias = try model.weights["output.bias"]!.floatValues()[0]
            changed["output.bias"] = .floats([originalBias + 0.25], shape: model.weights["output.bias"]!.shape)
            let next = try program.execute(rgb: input, weights: changed, reference: reference,
                gradientsOnly: true, includePrediction: false)
            XCTAssertNotEqual(next.loss, training.loss, "An optimizer update must replace the cached parameter feeds.")
            XCTAssertEqual(program.executionStatistics.parameterFeedUploads, 2)
            var malformed = changed
            malformed["output.bias"] = NativeTensor(dtype: "F32", shape: [1], bytes: changed["output.bias"]!.bytes)
            XCTAssertThrowsError(try program.execute(rgb: input, weights: malformed, reference: reference,
                gradientsOnly: true, includePrediction: false))
            _ = try program.execute(rgb: input, weights: changed, reference: reference,
                gradientsOnly: true, includePrediction: false)
            XCTAssertEqual(program.executionStatistics.parameterFeedUploads, 2, "Rejected weights must not replace a valid feed cache.")
            XCTAssertEqual(program.executionStatistics.parameterFeedCacheHits, 2)
            XCTAssertEqual(program.executionStatistics.compilations, 2, "Repeated frozen microbatches retain the training executable.")
        }.value
    }

    /// A bounded, opt-in engine comparison. It uses procedural native samples
    /// and cannot mutate a user dataset or claim learned material quality.
    func testCompactTrainingThroughput() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["TEXTURE_STUDIO_COMPACT_THROUGHPUT_BENCHMARK"] == "1")
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let reportPath = try XCTUnwrap(environment["TEXTURE_STUDIO_COMPACT_THROUGHPUT_REPORT"])
        let size = Int(environment["TEXTURE_STUDIO_COMPACT_THROUGHPUT_SIZE"] ?? "512") ?? 512
        let width = Int(environment["TEXTURE_STUDIO_COMPACT_THROUGHPUT_WIDTH"] ?? "32") ?? 32
        guard size >= 64, size <= 2048, size % 64 == 0 else { XCTFail("Choose a bounded native benchmark grid divisible by 64."); return }
        try await Task.detached {
            let input = Self.rgb(width: size, height: size)
            let reference = (0..<size * size).map { Float(($0 * 11) % 251) / 251 }
            var runs: [[String: Any]] = []
            for includePrediction in [true, false] {
                for accumulation in [1, 2] {
                    let model = try NativeCompactMaterialModel(target: "height", width: width, seed: 31)
                    let program = try model.program(width: size, height: size, target: "height")
                    var state: [String: NativeTensor] = [:], durations: [Double] = [], optimizerSeconds: [Double] = []
                    for step in 1...7 {
                        let started = ProcessInfo.processInfo.systemUptime
                        var gradients = NativeMaterialGradientAccumulator()
                        for _ in 0..<accumulation {
                            let execution = try program.execute(rgb: input, weights: model.weights, reference: reference,
                                gradientsOnly: true, includePrediction: includePrediction)
                            try gradients.add(execution.gradients)
                        }
                        let optimizerStarted = ProcessInfo.processInfo.systemUptime
                        let update = try NativeMaterialOptimizer.apply(gradients: gradients.averaged(), weights: model.weights,
                            state: state, learningRate: 1e-3, step: step, configuration: .init())
                        optimizerSeconds.append(ProcessInfo.processInfo.systemUptime - optimizerStarted)
                        try model.updateWeights(update.weights); state = update.state
                        durations.append(ProcessInfo.processInfo.systemUptime - started)
                    }
                    let warm = Array(durations.dropFirst()).sorted()
                    let warmMedian = (warm[2] + warm[3]) / 2
                    let statistics = try JSONSerialization.jsonObject(with: JSONEncoder().encode(program.executionStatistics))
                    runs.append(["prediction_readback": includePrediction, "accumulation_steps": accumulation,
                        "cold_update_seconds": durations[0], "warm_update_median_seconds": warmMedian,
                        "warm_sample_median_seconds": warmMedian / Double(accumulation),
                        "update_seconds": durations, "optimizer_seconds": optimizerSeconds, "statistics": statistics])
                    XCTAssertEqual(program.executionStatistics.compilations, 1)
                    XCTAssertEqual(program.executionStatistics.parameterFeedUploads, 7)
                    XCTAssertEqual(program.executionStatistics.parameterFeedCacheHits, accumulation == 2 ? 7 : 0)
                }
            }
            let report: [String: Any] = ["fixture": "deterministic native RGB/Float32 scalar arrays; seven optimizer updates",
                "size": size, "network_width": width, "quality_qualification": false,
                "comparison": "prediction readback only; all runs include exact parameter feed reuse and bulk Float32 decoding",
                "runs": runs]
            let url = URL(fileURLWithPath: reportPath)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: url)
        }.value
    }

    func testLinearDiffuseContractPersistsAndLegacyCheckpointsRetainSRGB() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 29)
        XCTAssertEqual(try NativeTrainableMaterialModel(model).diffuseInputTransfer, .linear)
        let current = try model.checkpointConfiguration(size: 64, step: 4)
        XCTAssertEqual(current["input_tensor_transfer"] as? String, NativeMaterialInputTransfer.linear.rawValue)
        XCTAssertEqual(current["target_transfer"] as? String, "linear numeric source codes / code maximum")
        for legacy in [false, true] {
            var configuration = current
            if legacy {
                configuration.removeValue(forKey: "input_tensor_transfer")
                configuration["input_transfer"] = NativeMaterialInputTransfer.srgb.description
                configuration.removeValue(forKey: "convolution_padding")
            }
            let checkpoint = directory.appendingPathComponent(legacy ? "legacy.safetensors" : "linear.safetensors")
            try NativeSafetensors.write(tensors: model.weights,
                metadata: ["configuration": NativeMaterialTransfer.json(configuration)], to: checkpoint)
            let original = try Data(contentsOf: checkpoint)
            let loaded = try NativeCompactMaterialModel.load(checkpointURL: checkpoint, target: "height", width: 16)
            let expected: NativeMaterialInputTransfer = legacy ? .srgb : .linear
            XCTAssertEqual(try NativeTrainableMaterialModel(loaded).diffuseInputTransfer, expected)
            XCTAssertEqual(try loaded.checkpointConfiguration(size: 64, step: 5)["input_tensor_transfer"] as? String, expected.rawValue)
            let options = try NativeMaterialTrainer.Options(["infer", "--baseline", "--checkpoint", checkpoint.path,
                "--image", "/unused.png", "--output", "/unused-output"])
            let baseline = try NativeTrainableMaterialModel.load(options, training: false)
            XCTAssertEqual(try baseline.diffuseInputTransfer, expected,
                "The fresh baseline comparison must use the same input convention as the trained checkpoint")
            XCTAssertEqual(baseline.configuration["convolution_padding"] as? String,
                try NativeCompactMaterialModel.convolutionPadding(in: configuration),
                "The fresh baseline must retain the checkpoint graph's original border behavior")
            XCTAssertEqual(try Data(contentsOf: checkpoint), original)
        }
        var invalid = current; invalid["input_tensor_transfer"] = "unsupported"
        let snapshot = try NativeSafetensors(bytes: NativeSafetensors.encoded(tensors: model.weights, metadata: [:]))
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: invalid))
    }

    func testDepthProRefinerUsesDistinctSingleChannelArchitectureAndCheckpointContract() throws {
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 19, family: .depthProHeightRefiner)
        let repeated = try NativeCompactMaterialModel(target: "height", width: 16, seed: 19, family: .depthProHeightRefiner)
        let rgb = try NativeCompactMaterialModel(target: "height", width: 16, seed: 19)
        XCTAssertEqual(model.architectureID, NativeCompactMaterialModel.depthProRefinerArchitecture)
        XCTAssertEqual(model.modelFamily, .depthProHeightRefiner)
        XCTAssertEqual(model.configuration["convolution_padding"] as? String, NativeCompactMaterialModel.reflectionConvolutionPadding)
        XCTAssertEqual(model.inputChannels, 1); XCTAssertEqual(model.outputChannels, 1)
        XCTAssertEqual(model.weights["input.weight"]?.shape, [16, 1, 3, 3])
        XCTAssertEqual(model.baseSHA256, repeated.baseSHA256)
        XCTAssertNotEqual(model.baseSHA256, rgb.baseSHA256)
        XCTAssertThrowsError(try model.updateWeights(rgb.weights))
        XCTAssertThrowsError(try NativeCompactMaterialModel(target: "roughness", family: .depthProHeightRefiner))
        XCTAssertThrowsError(try NativeCompactMaterialModel(target: "normal", family: .depthProHeightRefiner))
        model.configuration["depthpro_model_sha256"] = String(repeating: "a", count: 64)
        let configuration = try model.checkpointConfiguration(size: 256, step: 1)
        let snapshot = try NativeSafetensors(bytes: NativeSafetensors.encoded(tensors: model.weights,
            metadata: ["configuration": NativeMaterialTransfer.json(configuration)]))
        XCTAssertNoThrow(try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: configuration))
        for field in ["depthpro_model_sha256", "depthpro_preparation_contract", "input_kind", "input_transfer", "target_transfer"] {
            var missing = configuration; missing.removeValue(forKey: field)
            XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: missing), field)
        }
        for (field, value) in [("model_family", "compact-scalar" as Any), ("input_channels", 3),
                               ("depthpro_model_sha256", "wrong"), ("depthpro_preparation_contract", "display-grayscale"),
                               ("input_transfer", "sRGB diffuse codes / code maximum")] {
            var wrong = configuration; wrong[field] = value
            XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: wrong), field)
        }
        let rgbSnapshot = try NativeSafetensors(bytes: NativeSafetensors.encoded(tensors: rgb.weights, metadata: [:]))
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(rgbSnapshot, configuration: configuration))
        var unsupported = configuration; unsupported["convolution_padding"] = "unknown"
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: unsupported))
        unsupported["convolution_padding"] = 1
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: unsupported))
        var legacy = configuration; legacy.removeValue(forKey: "convolution_padding")
        XCTAssertNoThrow(try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: legacy))
    }

    func testReflectionKeepsConstantFieldFreeOfFrameAndLegacyCheckpointRetainsZeroBorders() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 23, family: .depthProHeightRefiner)
            let input = [Float](repeating: 0.5, count: 64 * 64)
            let reflected = try model.predict(rgb: input, width: 64, height: 64, target: "height")
            XCTAssertLessThan(try XCTUnwrap(reflected.values.max()) - XCTUnwrap(reflected.values.min()), 0.00002,
                "A spatially constant field must stay constant through reflection convolutions at every scale")
            model.configuration.removeValue(forKey: "convolution_padding")
            let implicitZero = try model.predict(rgb: input, width: 64, height: 64, target: "height")
            XCTAssertGreaterThan(try XCTUnwrap(implicitZero.values.max()) - XCTUnwrap(implicitZero.values.min()), 0.0001)
            model.configuration["convolution_padding"] = NativeCompactMaterialModel.legacyConvolutionPadding
            let explicitZero = try model.predict(rgb: input, width: 64, height: 64, target: "height")
            XCTAssertEqual(implicitZero.values.map(\.bitPattern), explicitZero.values.map(\.bitPattern))
            model.configuration["depthpro_model_sha256"] = String(repeating: "a", count: 64)
            var legacy = try model.checkpointConfiguration(size: 64, step: 1)
            legacy.removeValue(forKey: "convolution_padding")
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let checkpoint = directory.appendingPathComponent("legacy-model.safetensors")
            try NativeSafetensors.write(tensors: model.weights,
                metadata: ["configuration": NativeMaterialTransfer.json(legacy)], to: checkpoint)
            let original = try Data(contentsOf: checkpoint)
            let loaded = try NativeCompactMaterialModel.load(checkpointURL: checkpoint, target: "height", width: 16, family: .depthProHeightRefiner)
            let reloaded = try loaded.predict(rgb: input, width: 64, height: 64, target: "height")
            XCTAssertEqual(reloaded.values.map(\.bitPattern), implicitZero.values.map(\.bitPattern))
            XCTAssertEqual(try loaded.checkpointConfiguration(size: 64, step: 2)["convolution_padding"] as? String,
                NativeCompactMaterialModel.legacyConvolutionPadding)
            XCTAssertEqual(try Data(contentsOf: checkpoint), original)
        }.value
    }

    func testDepthProRefinerPredictsAndDifferentiatesOneScalarChannel() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let depth = (0..<64 * 64).map { Float(($0 * 7) % 251) / 250 }
            let target = (0..<64 * 64).map { Float(($0 * 3) % 251) / 250 }
            let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 23, family: .depthProHeightRefiner)
            let prediction = try model.predict(rgb: depth, width: 64, height: 64, target: "height")
            XCTAssertEqual(prediction.channels, 1); XCTAssertEqual(prediction.values.count, depth.count)
            XCTAssertTrue(prediction.values.allSatisfy(\.isFinite))
            let program = try model.program(width: 64, height: 64, target: "height")
            let execution = try program.execute(rgb: depth, weights: model.weights, reference: target, gradientsOnly: true)
            XCTAssertEqual(Set(execution.gradients.keys), Set(model.weights.keys))
            XCTAssertTrue(try XCTUnwrap(execution.loss).isFinite)
            for tensor in execution.gradients.values { XCTAssertTrue(try tensor.floatValues().allSatisfy(\.isFinite)) }
            XCTAssertThrowsError(try model.predict(rgb: depth + depth + depth, width: 64, height: 64, target: "height"))
            var invalid = depth; invalid[0] = -.infinity
            XCTAssertThrowsError(try model.predict(rgb: invalid, width: 64, height: 64, target: "height"))
        }.value
    }

    func testScratchInitializationAndOutputFamiliesHaveDeterministicSeparateIdentities() throws {
        let height = try NativeCompactMaterialModel(target: "height", width: 16, seed: 19)
        let repeated = try NativeCompactMaterialModel(target: "height", width: 16, seed: 19)
        let otherSeed = try NativeCompactMaterialModel(target: "height", width: 16, seed: 20)
        let normal = try NativeCompactMaterialModel(target: "normal", width: 16, seed: 19)
        let roughness = try NativeCompactMaterialModel(target: "roughness", width: 16, seed: 19)
        XCTAssertEqual(height.baseSHA256, repeated.baseSHA256)
        XCTAssertNotEqual(height.baseSHA256, otherSeed.baseSHA256)
        XCTAssertNotEqual(height.baseSHA256, normal.baseSHA256)
        XCTAssertEqual(height.architectureID, NativeCompactMaterialModel.scalarArchitecture)
        XCTAssertEqual(roughness.architectureID, height.architectureID)
        XCTAssertEqual(normal.architectureID, NativeCompactMaterialModel.normalArchitecture)
        XCTAssertEqual(height.outputChannels, 1); XCTAssertEqual(roughness.outputChannels, 1); XCTAssertEqual(normal.outputChannels, 3)
        XCTAssertEqual(height.weights["output.weight"]!.shape, [1, 16, 3, 3])
        XCTAssertEqual(normal.weights["output.weight"]!.shape, [3, 16, 3, 3])
        XCTAssertEqual(Set(height.weights.keys), Set(repeated.weights.keys))
        for name in height.weights.keys { XCTAssertEqual(height.weights[name]!.bytes, repeated.weights[name]!.bytes) }
        XCTAssertNil(height.configuration["base"]); XCTAssertNil(height.configuration["layers"])
        XCTAssertEqual(height.configuration["from_scratch"] as? Bool, true)
        XCTAssertThrowsError(try NativeCompactMaterialModel(target: "albedo"))
        XCTAssertThrowsError(try NativeCompactMaterialModel(target: "height", width: 96))
    }

    func testFullCheckpointRejectsWrongFamilyMissingWeightAndMalformedValues() throws {
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 17)
        let configuration = try model.checkpointConfiguration(size: 256, step: 1)
        func snapshot(_ weights: [String: NativeTensor]) throws -> NativeSafetensors {
            let metadata = String(decoding: try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys]), as: UTF8.self)
            return try NativeSafetensors(bytes: NativeSafetensors.encoded(tensors: weights, metadata: ["configuration": metadata]))
        }
        let original = try snapshot(model.weights)
        XCTAssertNoThrow(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: configuration))
        for field in ["step", "training_size", "image_padding", "image_resizing", "scope", "native_runtime", "initial_weights_sha256", "target", "network_width"] {
            var missingField = configuration; missingField.removeValue(forKey: field)
            XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: missingField), field)
        }
        for (field, invalid) in [("step", -1 as Any), ("training_size", 255 as Any), ("image_padding", true as Any),
                                 ("image_resizing", 0 as Any), ("native_runtime", "unknown" as Any), ("scope", "final-map" as Any)] {
            var wrongField = configuration; wrongField[field] = invalid
            XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: wrongField), field)
        }
        var wrongFamily = configuration
        wrongFamily["architecture"] = NativeCompactMaterialModel.normalArchitecture
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: wrongFamily))
        var wrongTarget = configuration; wrongTarget["target"] = "normal"
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: wrongTarget))
        var wrongSeed = configuration; wrongSeed["initialization_seed"] = UInt64(18)
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: wrongSeed))
        var noninteger = configuration; noninteger["network_width"] = 16.5
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(original, configuration: noninteger))
        var missing = model.weights; missing.removeValue(forKey: "input.weight")
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot(missing), configuration: configuration))
        var malformed = model.weights
        malformed["output.weight"] = .floats([Float](repeating: 0, count: 16 * 9), shape: [16, 1, 3, 3])
        XCTAssertThrowsError(try model.updateWeights(malformed))
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(snapshot(malformed), configuration: configuration))
        var bytes = original.bytes, infinity = Float.infinity.bitPattern.littleEndian
        let offset = original.payloadStart + original.tensors["output.bias"]!.dataOffsets[0]
        withUnsafeBytes(of: &infinity) { bytes.replaceSubrange(offset..<(offset + 4), with: $0) }
        let nonfinite = try NativeSafetensors(bytes: bytes)
        XCTAssertThrowsError(try NativeCompactMaterialModel.validateCheckpoint(nonfinite, configuration: configuration)) { error in
            XCTAssertFalse(error is NativeMaterialSampleError, "Malformed stored model weights are fatal model failures.")
        }
    }

    func testNativeScalarAndNormalPredictionPreserveInputGridAndHeadDimensions() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let rgb = Self.rgb(width: 128, height: 64)
            for target in ["height", "roughness", "normal"] {
                let model = try NativeCompactMaterialModel(target: target, width: 16)
                let result = try model.predict(rgb: rgb, width: 128, height: 64, target: target)
                XCTAssertEqual(result.width, 128); XCTAssertEqual(result.height, 64)
                XCTAssertEqual(result.channels, target == "normal" ? 3 : 1)
                XCTAssertEqual(result.values.count, 128 * 64 * result.channels)
                XCTAssertTrue(result.values.allSatisfy(\.isFinite))
                let program = try model.program(width: 128, height: 64, target: target)
                XCTAssertEqual(program.executionStatistics.compilations, 1)
                let repeated = try model.predict(rgb: rgb, width: 128, height: 64, target: target)
                XCTAssertEqual(program.executionStatistics.compilations, 1, "Prediction reuses the executable and never builds autodiff.")
                XCTAssertEqual(result.values.map(\.bitPattern), repeated.values.map(\.bitPattern))
                XCTAssertThrowsError(try model.predict(rgb: rgb, width: 128, height: 64, target: target == "normal" ? "height" : "normal"))
                XCTAssertThrowsError(try model.predict(rgb: rgb, width: 127, height: 64, target: target))
            }
        }.value
    }

    func testEveryWeightHasFiniteAutodiffAndAdamUpdatesEarlyMiddleAndOutputWeights() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 23)
            XCTAssertEqual(model.configuration["convolution_padding"] as? String, NativeCompactMaterialModel.reflectionConvolutionPadding)
            let before = model.weights, rgb = Self.rgb(width: 64, height: 64)
            let reference = (0..<64 * 64).map { Float(($0 * 11) % 251) / 251 }
            let program = try model.program(width: 64, height: 64, target: "height")
            let evaluated = try program.execute(rgb: rgb, weights: before, reference: reference, gradientsOnly: true)
            XCTAssertEqual(Set(evaluated.gradients.keys), Set(before.keys))
            XCTAssertTrue(evaluated.updated.isEmpty); XCTAssertTrue(evaluated.optimizerState.isEmpty)
            XCTAssertTrue(try XCTUnwrap(evaluated.loss).isFinite)
            for name in before.keys {
                XCTAssertEqual(before[name]!.bytes, model.weights[name]!.bytes, "Gradient evaluation cannot mutate weights.")
                XCTAssertEqual(before[name]!.shape, evaluated.gradients[name]!.shape)
                XCTAssertTrue(try evaluated.gradients[name]!.floatValues().allSatisfy(\.isFinite))
            }
            // Compare one graph gradient with an independently perturbed loss.
            let epsilon: Float = 1e-3
            func loss(shift: Float) throws -> Float {
                var shifted = before
                let bias = try before["output.bias"]!.floatValues()[0]
                shifted["output.bias"] = .floats([bias + shift], shape: [1, 1, 1, 1])
                return try XCTUnwrap(program.execute(rgb: rgb, weights: shifted, reference: reference).loss)
            }
            let finiteDifference = try (loss(shift: epsilon) - loss(shift: -epsilon)) / (2 * epsilon)
            XCTAssertEqual(try evaluated.gradients["output.bias"]!.floatValues()[0], finiteDifference, accuracy: 0.005)
            func inputWeightLoss(shift: Float) throws -> Float {
                var shifted = before
                var values = try before["input.weight"]!.floatValues()
                values[0] += shift
                shifted["input.weight"] = .floats(values, shape: before["input.weight"]!.shape)
                return try XCTUnwrap(program.execute(rgb: rgb, weights: shifted, reference: reference).loss)
            }
            let inputFiniteDifference = try (inputWeightLoss(shift: epsilon) - inputWeightLoss(shift: -epsilon)) / (2 * epsilon)
            XCTAssertEqual(try evaluated.gradients["input.weight"]!.floatValues()[0], inputFiniteDifference, accuracy: 0.005,
                "Autodiff must accumulate gradients through the reflected borders in downstream convolutions")
            let update = try NativeMaterialOptimizer.apply(gradients: evaluated.gradients, weights: before, state: [:],
                learningRate: 1e-3, step: 1, configuration: .init())
            XCTAssertEqual(Set(update.weights.keys), Set(before.keys))
            XCTAssertEqual(update.state.count, before.count * 2)
            for name in ["input.weight", "encoder.0.expand.weight", "middle.expand.weight", "up.2.weight", "output.weight", "output.bias"] {
                XCTAssertNotEqual(before[name]!.bytes, update.weights[name]!.bytes, "Full-backbone training must update \(name).")
            }
            try model.updateWeights(update.weights)
            let second = try program.execute(rgb: rgb, weights: model.weights, reference: reference, gradientsOnly: true)
            XCTAssertTrue(try XCTUnwrap(second.loss).isFinite)
            // Validation and training retain their own executables.
            XCTAssertEqual(program.executionStatistics.compilations, 2)
        }.value
    }

    func testSavedFullWeightsReloadPredictExactlyAndResumeWithoutPBRnxt() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let model = try NativeCompactMaterialModel(target: "normal", width: 16, seed: 29)
            let rgb = Self.rgb(width: 64, height: 64), before = model.weights
            let reference = (0..<3 * 64 * 64).map { Float(($0 * 7) % 251) / 251 }
            let program = try model.program(width: 64, height: 64, target: "normal")
            let derivative = try program.execute(rgb: rgb, weights: before, reference: reference, gradientsOnly: true)
            let update = try NativeMaterialOptimizer.apply(gradients: derivative.gradients, weights: before, state: [:],
                learningRate: 1e-4, step: 1, configuration: .init())
            try model.updateWeights(update.weights)
            let prediction = try model.predict(rgb: rgb, width: 64, height: 64, target: "normal")
            let checkpoint = directory.appendingPathComponent("model.safetensors")
            let configuration = try model.checkpointConfiguration(size: 256, step: 1)
            let metadata = String(decoding: try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys]), as: UTF8.self)
            try NativeSafetensors.write(tensors: model.weights, metadata: ["configuration": metadata], to: checkpoint)
            let hash = try NativeMaterialTransfer.hash(checkpoint)
            let reloaded = try NativeCompactMaterialModel.load(checkpointURL: checkpoint, expectedSHA256: hash,
                target: "normal", width: 16, seed: 99)
            XCTAssertEqual(reloaded.networkWidth, 16); XCTAssertEqual(reloaded.initializationSeed, 29)
            XCTAssertEqual(reloaded.configuration["step"] as? Int, 1)
            XCTAssertEqual(reloaded.baseSHA256, model.baseSHA256)
            for name in model.weights.keys { XCTAssertEqual(reloaded.weights[name]!.bytes, model.weights[name]!.bytes) }
            let repeated = try reloaded.predict(rgb: rgb, width: 64, height: 64, target: "normal")
            XCTAssertEqual(prediction.values.map(\.bitPattern), repeated.values.map(\.bitPattern))
            XCTAssertThrowsError(try NativeCompactMaterialModel.load(checkpointURL: checkpoint,
                expectedSHA256: String(repeating: "0", count: 64), target: "normal"))
            XCTAssertThrowsError(try NativeCompactMaterialModel.load(checkpointURL: checkpoint, target: "height"))
            XCTAssertThrowsError(try NativeCompactMaterialModel.load(checkpointURL: checkpoint, target: "normal", width: 32))
        }.value
    }

    func testFiniteSampleOverflowSkipsWithoutCorruptingFollowingEvaluation() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let model = try NativeCompactMaterialModel(target: "height", width: 16)
            let before = model.weights, rgb = Self.rgb(width: 64, height: 64)
            let program = try model.program(width: 64, height: 64, target: "height")
            let extreme = (0..<64 * 64).map { $0 % 2 == 0 ? Float.greatestFiniteMagnitude : -Float.greatestFiniteMagnitude }
            XCTAssertThrowsError(try program.execute(rgb: rgb, weights: before, reference: extreme, gradientsOnly: true)) { error in
                XCTAssertTrue(error is NativeMaterialSampleError)
            }
            for name in before.keys { XCTAssertEqual(before[name]!.bytes, model.weights[name]!.bytes) }
            let next = try program.execute(rgb: rgb, weights: before,
                reference: [Float](repeating: 0.25, count: 64 * 64), gradientsOnly: true)
            XCTAssertTrue(try XCTUnwrap(next.loss).isFinite)
            XCTAssertEqual(Set(next.gradients.keys), Set(before.keys))
        }.value
    }

    private static func rgb(width: Int, height: Int) -> [Float] {
        (0..<width * height * 3).map { Float(($0 * 37 + $0 / width) % 251) / 251 }
    }
}
