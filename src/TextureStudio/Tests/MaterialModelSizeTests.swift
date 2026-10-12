import Foundation
import Darwin
import Metal
import XCTest
@testable import TextureStudio

final class MaterialModelSizeTests: XCTestCase {
    func testPresetsReportExactCountsForEachOutputAndInputFamily() throws {
        let scalar = [256_257, 478_017, 999_937, 2_050_497, 3_949_569]
        for (index, preset) in MaterialTrainingModelSize.allCases.enumerated() {
            let width = preset.rawValue
            XCTAssertEqual(preset.parameterCount(target: "height", family: .compactScalar), scalar[index])
            XCTAssertEqual(preset.parameterCount(target: "normal", family: .compactNormal), scalar[index] + 18 * width + 2)
            XCTAssertEqual(preset.parameterCount(target: "height", family: .depthProHeightRefiner), scalar[index] - 18 * width)
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", "/unused", "--output", "/unused",
                "--model-family", "compact-scalar", "--compact-width", String(width)])
            XCTAssertEqual(options.compactWidth, width)
        }
        let saved = try JSONDecoder().decode(MaterialTrainingOptions.self, from: Data("{}".utf8))
        XCTAssertEqual(saved.compactWidth, 32, "Existing preferences retain the original width")
        var options = saved; options.modelFamily = .compactScalar; options.compactWidth = 46
        XCTAssertEqual(options, try JSONDecoder().decode(MaterialTrainingOptions.self, from: JSONEncoder().encode(options)))
        XCTAssertNil(options.configurationIssue)
    }

    func testLargerCheckpointKeepsItsSizeAndWeightIdentityWhenReopened() throws {
        let model = try NativeCompactMaterialModel(target: "height", width: 46, seed: 7)
        XCTAssertEqual(model.configuration["parameter_count"] as? Int, 2_050_497)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("model.safetensors")
        try NativeSafetensors.write(tensors: model.weights,
            metadata: ["configuration": NativeMaterialTransfer.json(model.checkpointConfiguration(size: 64, step: 2))], to: file)
        let original = try Data(contentsOf: file)
        let options = try NativeMaterialTrainer.Options(["refine", "--dataset", "/unused", "--output", "/unused", "--checkpoint", file.path])
        XCTAssertEqual(options.compactWidth, 46)
        let reopened = try NativeTrainableMaterialModel.load(options, training: true)
        XCTAssertEqual(reopened.parameterCount, 2_050_497)
        XCTAssertEqual(reopened.baseSHA256, model.baseSHA256)
        XCTAssertEqual(try Data(contentsOf: file), original)
    }

    func testExpandedWidthsRunNativeBackwardAndLionUpdateWithoutChangingGrid() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let input = (0..<64 * 64 * 3).map { Float(($0 * 7) % 251) / 250 }
            let target = (0..<64 * 64).map { Float(($0 * 3) % 251) / 250 }
            for width in [22, 46, 64] {
                let model = try NativeCompactMaterialModel(target: "height", width: width, seed: 3)
                let program = try model.program(width: 64, height: 64, target: "height")
                let execution = try program.execute(rgb: input, weights: model.weights, reference: target, gradientsOnly: true)
                XCTAssertTrue(try XCTUnwrap(execution.loss).isFinite)
                let update = try NativeMaterialOptimizer.apply(gradients: execution.gradients, weights: model.weights,
                    state: [:], learningRate: 0.0001, step: 1,
                    configuration: .init(algorithm: "lion", beta2: 0.99))
                try model.updateWeights(update.weights)
                let prediction = try model.predict(rgb: input, width: 64, height: 64, target: "height")
                XCTAssertEqual(prediction.width, 64); XCTAssertEqual(prediction.height, 64)
                XCTAssertEqual(prediction.values.count, target.count)
                XCTAssertTrue(prediction.values.allSatisfy(\.isFinite))
            }
        }.value
    }

    func testHalfAndTwoMillionPresetsCompleteOneUpdateAt256() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let size = 256
            let input = (0..<size * size * 3).map { Float(($0 * 7) % 251) / 250 }
            let target = (0..<size * size).map { Float(($0 * 3) % 251) / 250 }
            for width in [22, 46] {
                let model = try NativeCompactMaterialModel(target: "height", width: width, seed: 3)
                let execution = try model.program(width: size, height: size, target: "height")
                    .execute(rgb: input, weights: model.weights, reference: target, gradientsOnly: true)
                XCTAssertTrue(try XCTUnwrap(execution.loss).isFinite)
                let update = try NativeMaterialOptimizer.apply(gradients: execution.gradients, weights: model.weights,
                    state: [:], learningRate: 0.001, step: 1, configuration: .init())
                XCTAssertTrue(model.weights.contains { name, tensor in update.weights[name]?.bytes != tensor.bytes },
                    "The native backward pass and optimizer must change actual learned weights")
                try model.updateWeights(update.weights)
                let prediction = try model.predict(rgb: input, width: size, height: size, target: "height")
                XCTAssertEqual(prediction.width, size); XCTAssertEqual(prediction.height, size)
                XCTAssertEqual(prediction.values.count, size * size)
                XCTAssertTrue(prediction.values.allSatisfy(\.isFinite))
            }
        }.value
    }

    /// One update per capacity on the real native 2K grid. This tests execution,
    /// Float32 numerics and dimensions; it does not qualify learned map quality.
    func testOptInHalfMillionPresetCompletesOneNative2048Update() async throws {
        try await probeNativeGrid(width: 22, size: 2048)
    }

    func testOptInTwoMillionPresetCompletesOneNative2048Update() async throws {
        try await probeNativeGrid(width: 46, size: 2048)
    }

    func testOptInTwoMillionPresetCompletesOneNative1024Update() async throws {
        try await probeNativeGrid(width: 46, size: 1024)
    }

    private func probeNativeGrid(width requestedWidth: Int, size: Int) async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_COMPACT_2048_PROBE"] == "1" else {
            throw XCTSkip("Opt-in native 2048 model-capacity execution probe.")
        }
        let reportURL = URL(fileURLWithPath: environment["TEXTURE_STUDIO_COMPACT_2048_PROBE_REPORT"] ?? "/tmp/texture-compact-2048-probe.json")
        try await Task.detached {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
            let started = ProcessInfo.processInfo.systemUptime
            let input = (0..<size * size * 3).map { Float(($0 * 7) % 251) / 250 }
            let target = (0..<size * size).map { Float(($0 * 3) % 251) / 250 }
            var runs: [[String: Any]] = []
            for width in [requestedWidth] {
                let run = try autoreleasepool { () throws -> [String: Any] in
                    print("Native \(size) execution probe starting width \(width); one update only.")
                    let runStarted = ProcessInfo.processInfo.systemUptime
                    let model = try NativeCompactMaterialModel(target: "height", width: width, seed: 3)
                    let program = try model.program(width: size, height: size, target: "height")
                    let backwardStarted = ProcessInfo.processInfo.systemUptime
                    let execution = try program.execute(rgb: input, weights: model.weights, reference: target, gradientsOnly: true)
                    let backwardSeconds = ProcessInfo.processInfo.systemUptime - backwardStarted
                    let loss = try XCTUnwrap(execution.loss)
                    XCTAssertTrue(loss.isFinite)
                    XCTAssertEqual(Set(execution.gradients.keys), Set(model.weights.keys))
                    let finiteGradients = try execution.gradients.values.allSatisfy { try $0.floatValues().allSatisfy(\.isFinite) }
                    XCTAssertTrue(finiteGradients)
                    let optimizerStarted = ProcessInfo.processInfo.systemUptime
                    let update = try NativeMaterialOptimizer.apply(gradients: execution.gradients, weights: model.weights,
                        state: [:], learningRate: 0.001, step: 1, configuration: .init())
                    let optimizerSeconds = ProcessInfo.processInfo.systemUptime - optimizerStarted
                    let changed = model.weights.contains { name, tensor in update.weights[name]?.bytes != tensor.bytes }
                    XCTAssertTrue(changed)
                    try model.updateWeights(update.weights)
                    let forwardStarted = ProcessInfo.processInfo.systemUptime
                    let prediction = try model.predict(rgb: input, width: size, height: size, target: "height")
                    let forwardSeconds = ProcessInfo.processInfo.systemUptime - forwardStarted
                    XCTAssertEqual(prediction.width, size); XCTAssertEqual(prediction.height, size)
                    XCTAssertEqual(prediction.channels, 1); XCTAssertEqual(prediction.values.count, size * size)
                    let finiteOutput = prediction.values.allSatisfy(\.isFinite)
                    XCTAssertTrue(finiteOutput)
                    var resources = rusage(); getrusage(RUSAGE_SELF, &resources)
                    return ["network_width": width, "parameter_count": model.configuration["parameter_count"]!,
                        "native_dimensions": [prediction.width, prediction.height], "completed_updates": 1,
                        "loss": loss, "finite_gradients": finiteGradients, "finite_output": finiteOutput, "weights_changed": changed,
                        "backward_seconds": backwardSeconds, "optimizer_seconds": optimizerSeconds,
                        "post_update_forward_seconds": forwardSeconds,
                        "total_seconds": ProcessInfo.processInfo.systemUptime - runStarted,
                        "process_cumulative_peak_resident_bytes": resources.ru_maxrss,
                        "metal_allocated_bytes_at_forward_completion": device.currentAllocatedSize]
                }
                runs.append(run)
                let report: [String: Any] = ["purpose": "Bounded native Float32 \(size) execution only; learned quality is not assessed",
                    "device": device.name, "training_size": size, "precision": "Float32", "optimizer": "adamw",
                    "reduced_precision_fast_math": false, "runs": runs,
                    "total_wall_seconds": ProcessInfo.processInfo.systemUptime - started]
                try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: reportURL, options: .atomic)
                print("Native \(size) execution probe completed width \(width): \(run)")
            }
        }.value
    }

    func testOptInTwoMillionPresetNative2048InferenceWithoutTrainingGraph() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_COMPACT_2048_PROBE"] == "1" else {
            throw XCTSkip("Opt-in native 2048 model-capacity inference probe.")
        }
        let reportURL = URL(fileURLWithPath: environment["TEXTURE_STUDIO_COMPACT_2048_PROBE_REPORT"] ?? "/tmp/texture-compact-2048-inference-probe.json")
        try await Task.detached {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
            let size = 2048, started = ProcessInfo.processInfo.systemUptime
            let input = (0..<size * size * 3).map { Float(($0 * 7) % 251) / 250 }
            let model = try NativeCompactMaterialModel(target: "height", width: 46, seed: 3)
            let forwardStarted = ProcessInfo.processInfo.systemUptime
            let prediction = try model.predict(rgb: input, width: size, height: size, target: "height")
            let forwardSeconds = ProcessInfo.processInfo.systemUptime - forwardStarted
            let finite = prediction.values.allSatisfy(\.isFinite)
            XCTAssertEqual(prediction.width, size); XCTAssertEqual(prediction.height, size)
            XCTAssertEqual(prediction.channels, 1); XCTAssertEqual(prediction.values.count, size * size)
            XCTAssertTrue(finite)
            var resources = rusage(); getrusage(RUSAGE_SELF, &resources)
            let report: [String: Any] = ["purpose": "Native 2048 inference only; no optimizer or backward graph and no quality assessment",
                "device": device.name, "network_width": 46, "parameter_count": model.configuration["parameter_count"]!,
                "native_dimensions": [prediction.width, prediction.height], "precision": "Float32",
                "completed_updates": 0, "finite_output": finite, "forward_seconds": forwardSeconds,
                "total_seconds": ProcessInfo.processInfo.systemUptime - started,
                "process_cumulative_peak_resident_bytes": resources.ru_maxrss,
                "metal_allocated_bytes_at_forward_completion": device.currentAllocatedSize]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: reportURL, options: .atomic)
            print("Native 2048 inference-only probe completed: \(report)")
        }.value
    }
}
