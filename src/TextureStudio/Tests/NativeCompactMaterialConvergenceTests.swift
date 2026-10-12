import Foundation
import Metal
import XCTest
@testable import TextureStudio

/// Repeated updates must recover a known nonconstant target. A finite forward
/// pass and changed parameter bytes alone do not establish that training learns.
final class NativeCompactMaterialConvergenceTests: XCTestCase {
    func testRGBScalarLearnsDeterministicNonconstantTarget() async throws {
        try await learnSynthetic(family: .compactScalar)
    }

    func testSingleChannelRefinerLearnsDeterministicNonconstantTarget() async throws {
        try await learnSynthetic(family: .depthProHeightRefiner)
    }

    private func learnSynthetic(family: MaterialTrainingModelFamily) async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let size = 64, count = size * size
            var field = [Float](repeating: 0, count: count)
            for y in 0..<size {
                for x in 0..<size {
                    field[y * size + x] = 0.5 + 0.18 * sin(Float(x) * 0.31) +
                        0.16 * cos(Float(y) * 0.43) + 0.08 * sin(Float(x + y) * 0.19)
                }
            }
            let reference = field.map { 0.15 + 0.7 * $0 }
            let input = family == .depthProHeightRefiner ? field :
                field + field.map { 0.1 + 0.8 * $0 } + field.map { 0.9 - 0.8 * $0 }
            let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 23, family: family)
            let directory = try Self.reportDirectory(name: "synthetic-" + family.rawValue)
            let report = try Self.train(model: model, input: input, reference: reference, size: size,
                updates: 150, learningRate: 0.001, directory: directory,
                provenance: ["fixture": "deterministic spatial scalar; target = 0.15 + 0.7 * input channel 0",
                    "quality_scope": "known learnable fixed pair; no claim about novel materials"])
            let final = try XCTUnwrap(report.last)
            XCTAssertLessThan(final.mae, final.constantMAE * 0.45,
                "Training must beat the optimal constant prediction by a substantial margin")
            XCTAssertGreaterThan(final.correlation, 0.9, "Recovered structure must align with the target")
            XCTAssertGreaterThan(final.standardDeviationRatio, 0.6, "A gray prediction cannot satisfy convergence")
            XCTAssertLessThan(final.standardDeviationRatio, 1.4, "Contrast must be close to the target rather than amplified")
        }.value
    }

    /// Read-only source probe. Provide a diffuse PNG and its original 16-bit
    /// height PNG; use the same exact native crop in both files, without resizing.
    func testRealBarkFixedNativeCropConvergence() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE"] == "1" else {
            throw XCTSkip("Opt-in real material convergence probe.")
        }
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let inputPath = try XCTUnwrap(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_INPUT"], "Set the original diffuse PNG path")
        let targetPath = try XCTUnwrap(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_HEIGHT"], "Set the paired original 16-bit height PNG path")
        try await Task.detached {
            let sourceInput = URL(fileURLWithPath: inputPath), sourceTarget = URL(fileURLWithPath: targetPath)
            let inputBytes = try Data(contentsOf: sourceInput), targetBytes = try Data(contentsOf: sourceTarget)
            let inputPNG = try NativePNG.decode(inputBytes), targetPNG = try NativePNG.decode(targetBytes)
            XCTAssertEqual(inputPNG.header.width, targetPNG.header.width)
            XCTAssertEqual(inputPNG.header.height, targetPNG.header.height)
            let size = Int(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_SIZE"] ?? "256") ?? 0
            let x = Int(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_X"] ?? "0") ?? -1
            let y = Int(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_Y"] ?? "0") ?? -1
            let updates = Int(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_UPDATES"] ?? "200") ?? 0
            let width = Int(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_WIDTH"] ?? "16") ?? 0
            let learningRate = Float(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_RATE"] ?? "0.001") ?? .nan
            guard updates > 0, size >= 64, size % 64 == 0 else { throw StudioError("Invalid convergence probe dimensions or update count.") }
            let crop = [x, y, size, size]
            let inputCrop = try inputPNG.crop(crop), targetCrop = try targetPNG.crop(crop)
            let checkpoint = environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_CHECKPOINT"].map { URL(fileURLWithPath: $0) }
            let model = try NativeCompactMaterialModel.load(checkpointURL: checkpoint, target: "height", width: width, seed: 23)
            let transfer = try NativeMaterialInputTransfer.recorded(in: model.configuration)
            let input = try inputCrop.modelFloatSamples(role: "input", encoding: "srgb", inputTransfer: transfer)
            let reference = try targetCrop.modelFloatSamples(role: "height")
            let directory = try Self.reportDirectory(name: "real-bark")
            try inputCrop.encoded().write(to: directory.appendingPathComponent("input-crop.png"))
            try targetCrop.encoded().write(to: directory.appendingPathComponent("height-crop.png"))
            var provenance: [String: Any] = ["source_input": sourceInput.path, "source_height": sourceTarget.path,
                "source_input_sha256": NativeMaterialTrainer.checksum(inputBytes),
                "source_height_sha256": NativeMaterialTrainer.checksum(targetBytes),
                "native_crop": crop, "input_tensor_transfer": transfer.rawValue,
                "source_input_bits": inputPNG.header.bits, "source_height_bits": targetPNG.header.bits,
                "quality_scope": "repeated updates on one known native crop; no claim about novel materials"]
            if let checkpoint {
                provenance["starting_checkpoint"] = checkpoint.path
                provenance["starting_checkpoint_sha256"] = try NativeMaterialTransfer.hash(checkpoint)
            }
            let report = try Self.train(model: model, input: input, reference: reference, size: size,
                updates: updates, learningRate: learningRate, directory: directory, provenance: provenance)
            let initialStep = (model.configuration["step"] as? NSNumber)?.intValue ?? 0
            var configuration = try model.checkpointConfiguration(size: size, step: initialStep + updates)
            configuration["model_name"] = "Bark known-crop recovery probe"
            configuration["quality_qualification"] = false
            configuration["training_configuration"] = provenance.merging([
                "validation_probe": true, "training_size": size, "initial_step": initialStep,
                "requested_updates": updates, "learning_rate": learningRate,
                "optimizer_state_restored": false, "model_family": model.modelFamily.rawValue,
                "scope": "full-model", "trainable_parameters": "complete compact network"
            ].merging(NativeMaterialOptimizerConfiguration().metadata) { _, current in current }) { _, current in current }
            configuration["validation"] = ["status": "known-training-pair-diagnostic",
                "sample_count": 1, "scope": "fixed native crop used for training",
                "mae": report.last!.mae, "quality_qualification": false]
            try NativeSafetensors.write(tensors: model.weights,
                metadata: ["configuration": NativeMaterialTransfer.json(configuration)],
                to: directory.appendingPathComponent("known-crop-probe.safetensors"))
            let prediction = try model.predict(rgb: input, width: size, height: size, target: "height")
            try NativeMaterialNumericExporter.writeEXR(prediction,
                to: directory.appendingPathComponent("known-crop-prediction.float32.exr"))
            XCTAssertEqual(try Data(contentsOf: sourceInput), inputBytes, "The source diffuse must remain byte-identical")
            XCTAssertEqual(try Data(contentsOf: sourceTarget), targetBytes, "The source height must remain byte-identical")
            if let checkpoint {
                XCTAssertEqual(try NativeMaterialTransfer.hash(checkpoint), provenance["starting_checkpoint_sha256"] as? String,
                    "The convergence probe must retain the original checkpoint")
            }
            let final = try XCTUnwrap(report.last)
            XCTAssertLessThan(final.mae, final.constantMAE * 0.8,
                "Known native bark must improve on the optimal constant prediction")
            XCTAssertGreaterThan(final.correlation, 0.5, "Known native structure must be recovered")
            XCTAssertGreaterThan(final.standardDeviationRatio, 0.35, "Known material training cannot converge to gray")
        }.value
    }

    /// Measure the recovered or starting RGB checkpoint on the complete original
    /// diffuse/height pair through the same inference path used by comparison.
    /// This test records quality without assigning a default pass threshold.
    func testRecoveredCheckpointFullSourceAgreement() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE"] == "1" else {
            throw XCTSkip("Opt-in full original source agreement probe.")
        }
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let inputPath = try XCTUnwrap(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_INPUT"], "Set the original diffuse PNG path")
        let targetPath = try XCTUnwrap(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_HEIGHT"], "Set the paired original 16-bit height PNG path")
        let checkpointPath = try XCTUnwrap(environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_CHECKPOINT"], "Set the exact RGB checkpoint path")
        try await Task.detached {
            let sourceInput = URL(fileURLWithPath: inputPath), sourceTarget = URL(fileURLWithPath: targetPath)
            let checkpoint = NativeMaterialModel.resolvedCheckpoint(URL(fileURLWithPath: checkpointPath))
            let inputBytes = try Data(contentsOf: sourceInput), targetBytes = try Data(contentsOf: sourceTarget)
            let inputHash = NativeMaterialTrainer.checksum(inputBytes), targetHash = NativeMaterialTrainer.checksum(targetBytes)
            let checkpointHash = try NativeMaterialTransfer.hash(checkpoint)
            let inputPNG = try NativePNG.decode(inputBytes), targetPNG = try NativePNG.decode(targetBytes)
            let nativeDimensions = [inputPNG.header.width, inputPNG.header.height]
            guard nativeDimensions == [targetPNG.header.width, targetPNG.header.height] else {
                throw StudioError("Full source agreement requires the original diffuse and height to share their native grid.")
            }
            let directory = try Self.reportDirectory(name: "full-source")
            let inferenceDirectory = directory.appendingPathComponent("inference")
            let encoding = environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_INPUT_ENCODING"] ?? "srgb"
            let arguments = ["infer", "--checkpoint", checkpoint.path, "--expected-sha256", checkpointHash,
                "--image", sourceInput.path, "--input-encoding", encoding, "--output", inferenceDirectory.path]
            let options = try NativeMaterialTrainer.Options(arguments)
            guard options.modelFamily == .compactScalar, options.target == "height" else {
                throw StudioError("This full source probe requires an RGB compact height checkpoint.")
            }
            let model = try NativeCompactMaterialModel.load(checkpointURL: checkpoint, expectedSHA256: checkpointHash,
                target: "height", width: options.compactWidth, family: .compactScalar)
            let transfer = try NativeMaterialInputTransfer.recorded(in: model.configuration)
            try autoreleasepool {
                let input = try inputPNG.modelFloatSamples(role: "input", encoding: encoding, inputTransfer: transfer)
                let reference = try targetPNG.modelFloatSamples(role: "height")
                XCTAssertEqual(input.count, nativeDimensions[0] * nativeDimensions[1] * 3)
                XCTAssertEqual(reference.count, nativeDimensions[0] * nativeDimensions[1])
                XCTAssertTrue(input.allSatisfy(\.isFinite)); XCTAssertTrue(reference.allSatisfy(\.isFinite))
            }
            let forbiddenTransfer = ForbiddenDepthProTransfer()
            let missingDepthProModel = directory.appendingPathComponent("unused-depthpro/DepthPro.mlpackage")
            let resources = NativeDepthProResources(managedURL: missingDepthProModel, transfer: forbiddenTransfer)
            let started = ProcessInfo.processInfo.systemUptime
            let inferenceText = try await NativeMaterialTrainer.run(arguments: arguments, depthProResources: resources)
            let inference = try NativeMaterialTransfer.object(Data(inferenceText.utf8))
            XCTAssertEqual(inference["model_family"] as? String, MaterialTrainingModelFamily.compactScalar.rawValue)
            XCTAssertEqual(inference["input_kind"] as? String, "diffuse")
            XCTAssertEqual(inference["input_tensor_transfer"] as? String, transfer.rawValue)
            XCTAssertEqual(inference["checkpoint_sha256"] as? String, checkpointHash)
            XCTAssertEqual(inference["checkpoint_step"] as? Int, model.configuration["step"] as? Int)
            XCTAssertEqual(inference["image_sha256"] as? String, inputHash)
            XCTAssertEqual(inference["native_dimensions"] as? [Int], nativeDimensions)
            XCTAssertEqual(inference["output_channels"] as? Int, 1)
            XCTAssertEqual(inference["source_bytes_modified"] as? Bool, false)
            let generation = try XCTUnwrap(inference["generation"] as? [String: Any])
            XCTAssertEqual(generation["model_input_dimensions"] as? [Int], nativeDimensions)
            XCTAssertEqual(generation["source_pixels_resized"] as? Bool, false)
            XCTAssertEqual(generation["source_pixels_discarded"] as? Bool, false)
            XCTAssertNil(inference["prepared_input_path"]); XCTAssertNil(inference["depthpro_model_sha256"])
            XCTAssertEqual(forbiddenTransfer.count, 0, "The RGB full-source path cannot acquire or run DepthPro")
            XCTAssertFalse(FileManager.default.fileExists(atPath: missingDepthProModel.deletingLastPathComponent().path))
            let outputs = try XCTUnwrap(inference["outputs"] as? [String: [String: Any]])
            let output = try XCTUnwrap(outputs["height"])
            let predictionURL = URL(fileURLWithPath: try XCTUnwrap(output["path"] as? String))
            XCTAssertEqual(output["storage"] as? String, "FLOAT32")
            let agreement = try MaterialScalarAgreement.evaluate(referenceURL: sourceTarget, predictionURL: predictionURL, mapType: "height")
            XCTAssertEqual([agreement.width, agreement.height], nativeDimensions)
            XCTAssertEqual(agreement.referenceSHA256, targetHash)
            XCTAssertEqual(agreement.predictionSHA256, output["sha256"] as? String)
            XCTAssertEqual(try Data(contentsOf: sourceInput), inputBytes)
            XCTAssertEqual(try Data(contentsOf: sourceTarget), targetBytes)
            XCTAssertEqual(try NativeMaterialTransfer.hash(checkpoint), checkpointHash)
            var packageConfiguration = model.configuration
            packageConfiguration["model_name"] = "Bark known-material \(nativeDimensions[0]) recovery probe"
            packageConfiguration["quality_qualification"] = false
            packageConfiguration["production_agreement"] = Self.agreementObject(agreement).merging([
                "scope": "full original known material; used for training", "depthpro_used": false,
                "source_input_sha256": inputHash, "quality_qualification": false
            ]) { _, recorded in recorded }
            let packageURL = directory.appendingPathComponent("model-package")
            _ = try NativeMaterialPackage.export(model: model, configuration: packageConfiguration,
                to: packageURL, developer: false)
            let (_, packageHashes) = try NativeMaterialPackage.verify(packageURL)
            var report: [String: Any] = ["source_input": sourceInput.path, "source_height": sourceTarget.path,
                "source_input_sha256": inputHash, "source_height_sha256": targetHash,
                "checkpoint": checkpoint.path, "checkpoint_sha256": checkpointHash,
                "native_dimensions": nativeDimensions, "source_input_bits": inputPNG.header.bits,
                "source_height_bits": targetPNG.header.bits, "source_input_encoding": encoding,
                "input_tensor_transfer": transfer.rawValue, "inference": inference,
                "agreement": Self.agreementObject(agreement),
                "wall_seconds": ProcessInfo.processInfo.systemUptime - started,
                "quality_scope": "full original known material pair; no claim about novel materials",
                "source_pixels_resized": false, "source_bytes_modified": false,
                "checkpoint_bytes_modified": false, "depthpro_used": false]
            report["model_package"] = packageURL.path
            report["model_package_sha256"] = packageHashes["model.safetensors"]
            var thresholds: [String: Double] = [:]
            for name in ["MAX_CONSTANT_RATIO", "MIN_CORRELATION", "MIN_STD_RATIO"] {
                if let text = environment["TEXTURE_STUDIO_COMPACT_FULL_SOURCE_" + name] {
                    guard let value = Double(text), value.isFinite else { throw StudioError("Invalid full source agreement threshold: \(name).") }
                    thresholds[name] = value
                }
            }
            report["explicit_quality_thresholds"] = thresholds
            let reportURL = directory.appendingPathComponent("report.json")
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: reportURL)
            print("compact-full-source agreement: \(Self.agreementObject(agreement))")
            print("compact-full-source report: \(reportURL.path)")
            if let ratio = thresholds["MAX_CONSTANT_RATIO"] {
                XCTAssertLessThanOrEqual(agreement.mae, agreement.medianConstantMAE * ratio)
            }
            if let minimum = thresholds["MIN_CORRELATION"] {
                XCTAssertGreaterThanOrEqual(try XCTUnwrap(agreement.pearsonCorrelation), minimum)
            }
            if let minimum = thresholds["MIN_STD_RATIO"] {
                XCTAssertGreaterThanOrEqual(try XCTUnwrap(agreement.standardDeviationRatio), minimum)
            }
        }.value
    }

    private static func agreementObject(_ agreement: MaterialScalarAgreement.Result) -> [String: Any] {
        var result: [String: Any] = ["width": agreement.width, "height": agreement.height,
            "pixel_count": agreement.pixelCount, "reference_sha256": agreement.referenceSHA256,
            "prediction_sha256": agreement.predictionSHA256, "reference_mean": agreement.referenceMean,
            "reference_median": agreement.referenceMedian, "reference_std": agreement.referenceStandardDeviation,
            "prediction_mean": agreement.predictionMean, "prediction_std": agreement.predictionStandardDeviation,
            "reference_min": agreement.referenceMinimum, "reference_max": agreement.referenceMaximum,
            "prediction_min": agreement.predictionMinimum, "prediction_max": agreement.predictionMaximum,
            "mae": agreement.mae, "mean_constant_mae": agreement.meanConstantMAE,
            "median_constant_mae": agreement.medianConstantMAE,
            "reference_adjacent_difference_std": agreement.sourceAdjacentDifferenceStandardDeviation,
            "prediction_adjacent_difference_std": agreement.predictionAdjacentDifferenceStandardDeviation,
            "nearly_flat": agreement.nearlyFlat]
        result["correlation"] = agreement.pearsonCorrelation
        result["std_ratio"] = agreement.standardDeviationRatio
        result["adjacent_difference_std_ratio"] = agreement.adjacentDifferenceStandardDeviationRatio
        return result
    }

    private final class ForbiddenDepthProTransfer: ModelArtifactTransferring, @unchecked Sendable {
        private let lock = NSLock()
        private var requests = 0
        var count: Int { lock.withLock { requests } }
        func fetch(_ artifact: ModelDownloadArtifact, to destination: URL,
                   progress: @escaping @Sendable (Double) -> Void) async throws {
            lock.withLock { requests += 1 }
            throw CancellationError()
        }
    }

    private struct Metrics {
        let step: Int
        let mae: Double, constantMAE: Double, mean: Double, standardDeviation: Double
        let referenceStandardDeviation: Double, standardDeviationRatio: Double, correlation: Double
        let valueLoss: Double, detailLoss: Double, totalLoss: Double
        var object: [String: Any] {
            ["step": step, "mae": mae, "best_constant_mae": constantMAE,
                "prediction_mean": mean, "prediction_std": standardDeviation,
                "reference_std": referenceStandardDeviation, "std_ratio": standardDeviationRatio,
                "correlation": correlation, "value_l1": valueLoss, "detail_l1": detailLoss, "total": totalLoss]
        }
    }

    private static func reportDirectory(name: String) throws -> URL {
        let supplied = ProcessInfo.processInfo.environment["TEXTURE_STUDIO_COMPACT_CONVERGENCE_OUTPUT"]
        let directory = supplied.map { URL(fileURLWithPath: $0).appendingPathComponent(name) } ??
            FileManager.default.temporaryDirectory.appendingPathComponent("compact-convergence-" + name + "-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func train(model: NativeCompactMaterialModel, input: [Float], reference: [Float], size: Int,
                              updates: Int, learningRate: Float, directory: URL,
                              provenance: [String: Any]) throws -> [Metrics] {
        let program = try model.program(width: size, height: size, target: "height")
        var moments: [String: NativeTensor] = [:], snapshots: [Metrics] = [], gradientNorms: [Double] = []
        let started = ProcessInfo.processInfo.systemUptime
        func measure(step: Int) throws {
            let evaluated = try program.execute(rgb: input, weights: model.weights, reference: reference)
            let metrics = try metrics(step: step, prediction: evaluated.output, reference: reference, execution: evaluated)
            snapshots.append(metrics)
            print("compact-convergence \(model.modelFamily.rawValue) \(metrics.object)")
            try NativeTensor.floats(evaluated.output, shape: [1, 1, size, size]).bytes.write(
                to: directory.appendingPathComponent(String(format: "prediction-step-%05d.float32", step)))
        }
        try NativeTensor.floats(input, shape: [1, model.inputChannels, size, size]).bytes.write(to: directory.appendingPathComponent("input.float32"))
        try NativeTensor.floats(reference, shape: [1, 1, size, size]).bytes.write(to: directory.appendingPathComponent("reference.float32"))
        try measure(step: 0)
        for step in 1...updates {
            let derivative = try autoreleasepool {
                try program.execute(rgb: input, weights: model.weights, reference: reference, gradientsOnly: true)
            }
            let update = try NativeMaterialOptimizer.apply(gradients: derivative.gradients, weights: model.weights,
                state: moments, learningRate: learningRate, step: step, configuration: .init())
            try model.updateWeights(update.weights); moments = update.state
            gradientNorms.append(update.gradientNorm)
            if step % 25 == 0 || step == updates { try measure(step: step) }
        }
        var report = provenance
        report["model_family"] = model.modelFamily.rawValue; report["network_width"] = model.networkWidth
        report["initialization_seed"] = model.initializationSeed; report["native_dimensions"] = [size, size]
        report["updates"] = updates; report["learning_rate"] = learningRate
        report["optimizer"] = NativeMaterialOptimizerConfiguration().metadata
        report["gradient_norms"] = gradientNorms; report["snapshots"] = snapshots.map(\.object)
        report["wall_seconds"] = ProcessInfo.processInfo.systemUptime - started
        report["array_format"] = "little-endian Float32 planar CHW; raw model values, no display contrast"
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("report.json"))
        print("compact-convergence report: \(directory.appendingPathComponent("report.json").path)")
        return snapshots
    }

    private static func metrics(step: Int, prediction: [Float], reference: [Float],
                                execution: NativeMaterialModel.Program.Execution) throws -> Metrics {
        guard prediction.count == reference.count, !reference.isEmpty else { throw StudioError("Convergence measurement grids differ.") }
        let count = Double(reference.count), p = prediction.map(Double.init), r = reference.map(Double.init)
        let sorted = r.sorted(), median = sorted[sorted.count / 2]
        let pMean = p.reduce(0, +) / count, rMean = r.reduce(0, +) / count
        var mae = 0.0, constant = 0.0, pVariance = 0.0, rVariance = 0.0, covariance = 0.0
        for index in p.indices {
            let pd = p[index] - pMean, rd = r[index] - rMean
            mae += abs(p[index] - r[index]); constant += abs(r[index] - median)
            pVariance += pd * pd; rVariance += rd * rd; covariance += pd * rd
        }
        let pStd = sqrt(pVariance / count), rStd = sqrt(rVariance / count)
        return Metrics(step: step, mae: mae / count, constantMAE: constant / count, mean: pMean,
            standardDeviation: pStd, referenceStandardDeviation: rStd,
            standardDeviationRatio: rStd > 0 ? pStd / rStd : 0,
            correlation: pVariance > 0 && rVariance > 0 ? covariance / sqrt(pVariance * rVariance) : 0,
            valueLoss: Double(try XCTUnwrap(execution.valueLoss)),
            detailLoss: Double(try XCTUnwrap(execution.gradientLoss)), totalLoss: Double(try XCTUnwrap(execution.loss)))
    }
}
