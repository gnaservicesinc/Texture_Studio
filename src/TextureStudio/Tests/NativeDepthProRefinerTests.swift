import CoreGraphics
import Darwin
import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeDepthProRefinerTests: XCTestCase {
    func testPreparationPreservesAsymmetricRowsAndMakesNearerDepthHigher() throws {
        let values: [Float] = [1, 2, 4, 8, 16, 32]
        let prepared = try NativeDepthProContract.prepare(Self.depth(width: 3, height: 2, values: values), width: 3, height: 2)
        XCTAssertEqual(prepared.count, 6)
        for index in values.indices {
            let expected = Float((1 / Double(values[index]) - 1 / 32.0) / (1 - 1 / 32.0))
            XCTAssertEqual(prepared[index], expected, accuracy: 1e-7, "Top-left row-major position \(index)")
        }
        XCTAssertEqual(prepared[0], 1); XCTAssertEqual(prepared[5], 0)
        XCTAssertGreaterThan(prepared[1], prepared[3], "Rows and columns must not be transposed or flipped.")

        let resized = try NativeDepthProContract.prepare(Self.depth(width: 2, height: 2, values: [1, 4, 8, 16]), width: 4, height: 2)
        let meters: [Double] = [1, 1.75, 3.25, 4, 8, 10, 14, 16]
        for index in meters.indices {
            let expected = Float((1 / meters[index] - 1 / 16.0) / (1 - 1 / 16.0))
            XCTAssertEqual(resized[index], expected, accuracy: 1e-7,
                "Interpolate camera meters at pixel centers before reciprocal scaling.")
        }
    }

    func testPreparationDefinesConstantFieldsAndRejectsInvalidMetricDepth() throws {
        XCTAssertEqual(try NativeDepthProContract.prepare(Self.depth(width: 2, height: 2,
            values: [3, 3, 3, 3]), width: 3, height: 2), [Float](repeating: 0, count: 6))
        for invalid: Float in [0, -1, .infinity, -.infinity, .nan] {
            XCTAssertThrowsError(try NativeDepthProContract.prepare(Self.depth(width: 2, height: 2,
                values: [1, 2, invalid, 4]), width: 2, height: 2))
        }
        XCTAssertThrowsError(try NativeDepthProContract.prepare(Self.depth(width: 2, height: 2,
            values: [1, 2, 3]), width: 2, height: 2))
        XCTAssertThrowsError(try NativeDepthProContract.prepare(Self.depth(width: 1, height: 1,
            values: [1]), width: 0, height: 2))
    }

    func testCacheReusesNumericBitsAndInvalidatesSourceAndModelIdentity() throws {
        let root = try Self.temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let imageURL = root.appendingPathComponent("source.png"), directory = root.appendingPathComponent("cache")
        let original = try Self.photo(width: 4, height: 3, variant: 0)
        try original.write(to: imageURL)
        let sourceHash = NativeMaterialTrainer.checksum(original), counter = Counter()
        let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "a", count: 64), directory: directory) { image in
            counter.increment()
            return Self.analyticDepth(image)
        }
        let first = try cache.input(imageURL: imageURL, expectedSHA256: sourceHash, width: 4, height: 3)
        let repeated = try cache.input(imageURL: imageURL, expectedSHA256: sourceHash, width: 4, height: 3)
        XCTAssertEqual(first.map(\.bitPattern), repeated.map(\.bitPattern))
        XCTAssertEqual(counter.value, 1); XCTAssertEqual(cache.generatedCount, 1); XCTAssertEqual(cache.reusedCount, 1)
        let restarted = NativeDepthProInputCache(modelSHA256: cache.modelSHA256, directory: directory) { _ in
            XCTFail("Persistent numeric cache must avoid another upstream prediction.")
            throw StudioError("Unexpected DepthPro prediction")
        }
        let reopened = try restarted.input(imageURL: imageURL, expectedSHA256: sourceHash, width: 4, height: 3)
        XCTAssertEqual(first.map(\.bitPattern), reopened.map(\.bitPattern))
        XCTAssertEqual(restarted.generatedCount, 0); XCTAssertEqual(restarted.reusedCount, 1)

        let changed = try Self.photo(width: 4, height: 3, variant: 1)
        try changed.write(to: imageURL)
        XCTAssertThrowsError(try cache.input(imageURL: imageURL, expectedSHA256: sourceHash, width: 4, height: 3))
        let changedHash = NativeMaterialTrainer.checksum(changed)
        _ = try cache.input(imageURL: imageURL, expectedSHA256: changedHash, width: 4, height: 3)
        XCTAssertEqual(counter.value, 2); XCTAssertEqual(cache.generatedCount, 2)
        let otherModel = NativeDepthProInputCache(modelSHA256: String(repeating: "b", count: 64), directory: directory) { image in
            counter.increment()
            return Self.analyticDepth(image)
        }
        _ = try otherModel.input(imageURL: imageURL, expectedSHA256: changedHash, width: 4, height: 3)
        XCTAssertEqual(counter.value, 3)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).count, 3)
    }

    func testCacheRejectsCorruptNormalizedInputAndSourceMutationDuringPrediction() throws {
        let root = try Self.temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let imageURL = root.appendingPathComponent("source.png"), directory = root.appendingPathComponent("cache")
        let original = try Self.photo(width: 4, height: 3, variant: 0)
        try original.write(to: imageURL)
        let hash = NativeMaterialTrainer.checksum(original), counter = Counter()
        let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "a", count: 64), directory: directory) { image in
            counter.increment(); return Self.analyticDepth(image)
        }
        _ = try cache.input(imageURL: imageURL, expectedSHA256: hash, width: 4, height: 3)
        let entry = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).first)
        let snapshot = try NativeSafetensors(contentsOf: entry)
        var tensors = try snapshot.nativeTensors()
        var changed = try XCTUnwrap(tensors["input"]).floatValues(); changed[0] = 0.25
        tensors["input"] = .floats(changed, shape: [1, 1, 3, 4])
        try NativeSafetensors.write(tensors: tensors, metadata: snapshot.metadata, to: entry)
        XCTAssertThrowsError(try cache.input(imageURL: imageURL, expectedSHA256: hash, width: 4, height: 3))
        XCTAssertEqual(counter.value, 1, "Corrupted numeric input cannot be silently reused.")

        let mutated = NativeDepthProInputCache(modelSHA256: String(repeating: "c", count: 64),
            directory: root.appendingPathComponent("mutated-cache")) { image in
                try Self.photo(width: 4, height: 3, variant: 2).write(to: imageURL)
                return Self.analyticDepth(image)
            }
        XCTAssertThrowsError(try mutated.input(imageURL: imageURL, expectedSHA256: hash, width: 4, height: 3))
        XCTAssertEqual(mutated.generatedCount, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: mutated.directory.path))
    }

    func testTwoEpochTrainingCachesDepthProOnceAndInferenceUsesIdenticalScalarInput() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let root = try Self.temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let dataset = root.appendingPathComponent(".training-data/dataset")
            let fixture = try Self.datasetFixture(dataset)
            let counter = Counter()
            let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "d", count: 64),
                directory: root.appendingPathComponent("cache")) { image in
                    counter.increment(); return Self.analyticDepth(image)
                }
            _ = try Self.roundTrip(root: root, dataset: dataset, fixture: fixture, cache: cache, epochs: 2)
            XCTAssertEqual(counter.value, 1)
            XCTAssertEqual(cache.generatedCount, 1)
            XCTAssertGreaterThanOrEqual(cache.reusedCount, 2, "Epochs reuse cached Float32 depth without running DepthPro.")
        }.value
    }

    func testTwoEpochProgressiveFlipsTrainAlignedPairsWithoutRepeatingDepthProOrChangingDatasets() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let root = try Self.temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let originalRoot = root.appendingPathComponent("original"), dataset = root.appendingPathComponent("prepared/.training-data/dataset")
            let original = try Self.datasetFixture(originalRoot.appendingPathComponent(".training-data/dataset"))
            let fixture = try Self.datasetFixture(dataset, input: Data(contentsOf: original.inputURL),
                height: Data(contentsOf: original.heightURL))
            let protectedURLs = [original.inputURL, original.heightURL, fixture.inputURL, fixture.heightURL,
                originalRoot.appendingPathComponent("dataset.json"), dataset.appendingPathComponent("dataset.json"),
                original.inputURL.deletingLastPathComponent().appendingPathComponent("sample.json"),
                fixture.inputURL.deletingLastPathComponent().appendingPathComponent("sample.json")]
            let protectedBytes = try Dictionary(uniqueKeysWithValues: protectedURLs.map { ($0, try Data(contentsOf: $0)) })
            let originalPhoto = try NativePNG.decode(Data(contentsOf: original.inputURL))
            let originalHeight = try NativePNG.decode(Data(contentsOf: original.heightURL))
            let depth = Self.depth(width: 256, height: 256, values: (0..<256 * 256).map {
                1 + Float($0 % 256) / 256 + Float($0 / 256) * 2 / 256
            })
            let originalInput = try NativeDepthProContract.prepare(depth, width: 256, height: 256)
            let counter = Counter(), cacheDirectory = root.appendingPathComponent("depthpro-cache")
            let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "d", count: 64), directory: cacheDirectory) { image in
                XCTAssertEqual(image.width, 256); XCTAssertEqual(image.height, 256)
                counter.increment(); return depth
            }
            let output = root.appendingPathComponent("trained")
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
                "--model-family", "depthpro-height-refiner", "--target", "height", "--size", "256",
                "--compact-width", "16", "--updates-per-map", "2", "--seed", "4", "--learning-rate", "0.001",
                "--randomize-crops", "epoch", "--random-flip", "--checkpoint-every", "1", "--checkpoint-unit", "epoch"])
            let cropDirectory = options.cropCacheURL
            let recorder = Recorder(), crops = CropEpochRecorder(cropDirectory: cropDirectory,
                depthDirectory: cacheDirectory, modelSHA256: cache.modelSHA256)
            let trained = try NativeMaterialTransfer.object(Data(NativeMaterialTrainer.train(options,
                onEvent: { recorder.record($0); crops.record($0) }, control: .init(), depthProCache: cache).utf8))
            XCTAssertEqual(trained["completed_updates"] as? Int, 2)
            XCTAssertEqual(trained["completed_epochs"] as? Int, 2)
            XCTAssertEqual(trained["sample_evaluations"] as? Int, 2)
            XCTAssertEqual(trained["skipped_sample_count"] as? Int, 0)
            XCTAssertEqual(recorder.events.filter { $0["event"] as? String == "update" }.count, 2)
            let firstUpdate = try XCTUnwrap(recorder.events.firstIndex { $0["event"] as? String == "update_started" })
            let firstCrop = try XCTUnwrap(recorder.events.firstIndex { $0["event"] as? String == "crop_preparation" })
            XCTAssertLessThan(firstUpdate, firstCrop, "DepthPro crop work begins only after its optimizer update is scheduled")
            XCTAssertEqual(counter.value, 1)
            XCTAssertEqual(cache.generatedCount, 1)
            XCTAssertEqual(cache.remappedCount, 2)
            XCTAssertTrue(crops.errors.isEmpty, crops.errors.joined(separator: "; "))
            XCTAssertEqual(crops.snapshots.count, 2)
            let historyDocument = try NativeMaterialTransfer.object(cropDirectory.appendingPathComponent("crop-history.json"))
            let history = try XCTUnwrap(historyDocument["history"] as? [[String: Any]])
            XCTAssertEqual(history.count, 2)
            let cachePointer = try NativeMaterialTransfer.object(output.appendingPathComponent("crop-cache.json"))
            XCTAssertEqual(cachePointer["crop_cache_path"] as? String, cropDirectory.path)
            XCTAssertEqual(cachePointer["prepared_dataset_path"] as? String, dataset.path)
            XCTAssertEqual(history.compactMap { $0["epoch"] as? Int }, [1, 2])
            XCTAssertEqual(history.compactMap { $0["progressive_cached_flip"] as? Bool }, [true, true])
            XCTAssertEqual(history.compactMap { $0["source_center_spin_degrees"] as? Int }, [0, 0])
            XCTAssertEqual(history.compactMap { $0["horizontal_flip"] as? Bool }, [true, false])
            XCTAssertEqual(history.compactMap { $0["vertical_flip"] as? Bool }, [false, true])
            var indices = Array(0..<256 * 256)
            for (position, snapshot) in crops.snapshots.enumerated() {
                let horizontal = try XCTUnwrap(history[position]["horizontal_flip"] as? Bool)
                let vertical = try XCTUnwrap(history[position]["vertical_flip"] as? Bool)
                let preceding = indices
                for y in 0..<256 { for x in 0..<256 {
                    indices[y * 256 + x] = preceding[(vertical ? 255 - y : y) * 256 + (horizontal ? 255 - x : x)]
                } }
                let photo = try NativePNG.decode(snapshot.photo), height = try NativePNG.decode(snapshot.height)
                XCTAssertEqual(photo.pixels, Data(indices.flatMap { index in
                    (0..<3).map { originalPhoto.pixels[index * 3 + $0] }
                }), "Epoch \(position + 1) diffuse must flip the preceding cached crop")
                XCTAssertEqual(height.pixels, Data(indices.flatMap { index in
                    (0..<2).map { originalHeight.pixels[index * 2 + $0] }
                }), "Epoch \(position + 1) original 16-bit height codes must follow the identical permutation")
                let normalized = try NativeTensor(dtype: "F32", shape: [1, 1, 256, 256], bytes: snapshot.depthInput).floatValues()
                let meters = try NativeTensor(dtype: "F32", shape: [1, 1, 256, 256], bytes: snapshot.depthMeters).floatValues()
                XCTAssertEqual(normalized.map(\.bitPattern), indices.map { originalInput[$0].bitPattern })
                XCTAssertEqual(meters.map(\.bitPattern), indices.map { depth.values[$0].bitPattern })
                XCTAssertEqual(NativeMaterialTrainer.checksum(snapshot.photo), history[position]["input_sha256"] as? String)
                XCTAssertEqual(NativeMaterialTrainer.checksum(snapshot.height), history[position]["target_sha256"] as? String)
            }
            let checkpoint = URL(fileURLWithPath: try XCTUnwrap(trained["checkpoint_path"] as? String))
            let (configuration, _) = try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent())
            let training = try XCTUnwrap(configuration["training_configuration"] as? [String: Any])
            XCTAssertEqual(training["randomize_crops"] as? String, "epoch")
            XCTAssertEqual(training["random_flip"] as? Bool, true)
            XCTAssertEqual(training["random_spin_centercut"] as? Bool, false)
            XCTAssertEqual(training["crop_preparation"] as? String, "per-map-on-demand-v1")
            XCTAssertEqual(training["depthpro_predictor_lifetime"] as? String, "current_run")
            XCTAssertEqual(training["depthpro_generated_inputs"] as? Int, cache.generatedCount)
            XCTAssertEqual(training["depthpro_reused_inputs"] as? Int, cache.reusedCount)
            XCTAssertEqual(training["depthpro_remapped_inputs"] as? Int, cache.remappedCount)
            let receipt = try XCTUnwrap(trained["training_configuration"] as? [String: Any])
            XCTAssertEqual(receipt["depthpro_reused_inputs"] as? Int, cache.reusedCount)
            for step in 1...2 {
                let saved = try Self.trainingConfiguration(output.appendingPathComponent(String(format: "checkpoint-step-%08d.safetensors", step)))
                XCTAssertEqual(saved["depthpro_generated_inputs"] as? Int, 1)
                XCTAssertEqual(saved["depthpro_reused_inputs"] as? Int, step * 2 - 1,
                    "Only the requested crop's pre-flip field and active training input access the DepthPro cache")
                XCTAssertEqual(saved["depthpro_remapped_inputs"] as? Int, step)
            }
            for (url, bytes) in protectedBytes { XCTAssertEqual(try Data(contentsOf: url), bytes, url.path) }
        }.value
    }

    func testEpochCenterSpinsRecordPredictionCountsAtEachCheckpointAndFinalExport() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let root = try Self.temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let sourceInput = root.appendingPathComponent("original-diffuse.png")
            let sourceTarget = root.appendingPathComponent("original-height.png")
            let inputBytes = try Self.photo(width: 512, height: 512, variant: 0)
            let targetBytes = try NativePNG(header: .init(width: 512, height: 512, bits: 16, channels: 1, color: 0, interlace: 0),
                pixels: Data(repeating: 97, count: 512 * 512 * 2), colorChunks: []).encoded()
            try inputBytes.write(to: sourceInput); try targetBytes.write(to: sourceTarget)
            let dataset = root.appendingPathComponent(".training-data/dataset")
            let fixture = try Self.datasetFixture(dataset,
                input: NativePNG.crop(inputBytes, rectangle: [128, 128, 256, 256]).encoded(),
                height: NativePNG.crop(targetBytes, rectangle: [128, 128, 256, 256]).encoded())
            let sampleURL = fixture.inputURL.deletingLastPathComponent().appendingPathComponent("sample.json")
            var sample = try NativeMaterialTransfer.object(sampleURL)
            sample["crop_kind"] = "center"
            sample["crop_rectangle_top_left_xywh"] = [128, 128, 256, 256]
            sample["source_square_rectangle_top_left_xywh"] = [0, 0, 512, 512]
            var maps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
            maps["input"]?["source"] = ["path": sourceInput.path, "file_sha256": NativeMaterialTrainer.checksum(inputBytes), "width": 512, "height": 512]
            maps["height"]?["source"] = ["path": sourceTarget.path, "file_sha256": NativeMaterialTrainer.checksum(targetBytes), "width": 512, "height": 512]
            sample["map_metadata"] = maps
            try JSONSerialization.data(withJSONObject: sample).write(to: sampleURL)
            let counter = Counter()
            let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "e", count: 64), directory: root.appendingPathComponent("depthpro-cache")) { image in
                counter.increment(); return Self.analyticDepth(image)
            }
            let output = root.appendingPathComponent("trained")
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
                "--model-family", "depthpro-height-refiner", "--target", "height", "--size", "256", "--compact-width", "16",
                "--updates-per-map", "2", "--seed", "4", "--randomize-crops", "epoch", "--random-spin-centercut",
                "--checkpoint-every", "1", "--checkpoint-unit", "epoch"])
            let trained = try NativeMaterialTransfer.object(Data(NativeMaterialTrainer.train(options,
                onEvent: { _ in }, control: .init(), depthProCache: cache).utf8))
            XCTAssertEqual(trained["completed_updates"] as? Int, 2)
            XCTAssertEqual(counter.value, 2)
            XCTAssertEqual(cache.generatedCount, 2)
            for step in 1...2 {
                let saved = try Self.trainingConfiguration(output.appendingPathComponent(String(format: "checkpoint-step-%08d.safetensors", step)))
                XCTAssertEqual(saved["depthpro_generated_inputs"] as? Int, step)
                XCTAssertEqual(saved["depthpro_reused_inputs"] as? Int, 0,
                    "A spun crop generates its first DepthPro input directly in its active optimizer step")
                XCTAssertEqual(saved["depthpro_remapped_inputs"] as? Int, 0)
            }
            let checkpoint = URL(fileURLWithPath: try XCTUnwrap(trained["checkpoint_path"] as? String))
            let (configuration, _) = try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent())
            for recipe in [try XCTUnwrap(configuration["training_configuration"] as? [String: Any]),
                           try XCTUnwrap(trained["training_configuration"] as? [String: Any])] {
                XCTAssertEqual(recipe["depthpro_generated_inputs"] as? Int, cache.generatedCount)
                XCTAssertEqual(recipe["depthpro_reused_inputs"] as? Int, cache.reusedCount)
                XCTAssertEqual(recipe["depthpro_remapped_inputs"] as? Int, 0)
            }
        }.value
    }

    func testUpstreamPredictionOrCorruptCacheFailureAbortsTrainingWithoutExport() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let root = try Self.temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let dataset = root.appendingPathComponent(".training-data/dataset"), fixture = try Self.datasetFixture(dataset)
            for failure in ["prediction", "cache"] {
                let cache = NativeDepthProInputCache(modelSHA256: String(repeating: "d", count: 64),
                    directory: root.appendingPathComponent("cache-" + failure)) { image in
                        if failure == "prediction" { throw StudioError("Upstream inference failed") }
                        return Self.analyticDepth(image)
                    }
                if failure == "cache" {
                    _ = try cache.input(imageURL: fixture.inputURL, expectedSHA256: fixture.inputSHA256, width: 256, height: 256)
                    let entry = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: cache.directory,
                        includingPropertiesForKeys: nil).first)
                    try Data("damaged cache".utf8).write(to: entry)
                }
                let output = root.appendingPathComponent("failed-" + failure), recorder = Recorder()
                let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path,
                    "--output", output.path, "--model-family", "depthpro-height-refiner",
                    "--size", "256", "--compact-width", "16", "--updates-per-map", "1"])
                XCTAssertThrowsError(try NativeMaterialTrainer.train(options, onEvent: { recorder.record($0) },
                    control: .init(), depthProCache: cache)) { error in
                        XCTAssertTrue(error is NativeDepthProRuntimeError, failure)
                    }
                XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("export").path))
                let receipt = try NativeMaterialTransfer.object(output.appendingPathComponent("run.json"))
                XCTAssertEqual(receipt["status"] as? String, "failed")
                XCTAssertEqual(receipt["completed_updates"] as? Int, 0)
                XCTAssertNil(receipt["checkpoint_path"])
                XCTAssertFalse(recorder.events.contains { ["training_completed", "sample_skipped"].contains($0["event"] as? String ?? "") },
                    "Upstream runtime failures must abort rather than publish successful zero-update models.")
            }
        }.value
    }

    /// A short execution check with real upstream weights. It does not qualify
    /// trained quality or novel-material generalization.
    func testActualDepthProModelTrainingAndInferenceSmoke() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["TEXTURE_STUDIO_DEPTHPRO_MODEL"], !modelPath.isEmpty else {
            throw XCTSkip("Opt-in native DepthPro smoke: set TEXTURE_STUDIO_DEPTHPRO_MODEL to the local .mlpackage.")
        }
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        try await Task.detached {
            let root = try Self.temporary(); defer { try? FileManager.default.removeItem(at: root) }
            let started = ProcessInfo.processInfo.systemUptime, dataset = root.appendingPathComponent("dataset")
            let modelURL = URL(fileURLWithPath: modelPath)
            let material = try Self.realMaterial(environment: environment)
            let fixture = try Self.datasetFixture(dataset, input: material.input, height: material.height)
            let cache = try NativeDepthProInputCache(modelURL: modelURL, directory: root.appendingPathComponent("cache"))
            var receipt = try Self.roundTrip(root: root, dataset: dataset, fixture: fixture, cache: cache, epochs: 1)
            XCTAssertEqual(cache.generatedCount, 1)
            receipt["upstream_model_path"] = modelURL.path
            receipt["fixture"] = material.description
            receipt["actual_depthpro_prediction"] = true
            receipt["wall_seconds"] = ProcessInfo.processInfo.systemUptime - started
            receipt["metal_device"] = MTLCreateSystemDefaultDevice()?.name ?? "unavailable"
            var resources = rusage(); getrusage(RUSAGE_SELF, &resources)
            receipt["peak_resident_bytes"] = resources.ru_maxrss
            receipt["quality_qualification"] = false
            receipt["experiment_scope"] = "one-step 256 native execution verification"
            receipt["native_dimensions"] = [256, 256]
            if let trialPath = environment["TEXTURE_STUDIO_DEPTHPRO_TRIAL_DIRECTORY"], !trialPath.isEmpty {
                let destination = URL(fileURLWithPath: trialPath)
                guard !FileManager.default.fileExists(atPath: destination.path) else {
                    throw StudioError("Choose a new directory for the DepthPro smoke artifacts.")
                }
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: root, to: destination)
                try Self.relocateTrialManifests(from: root, to: destination)
                receipt["trial_directory"] = destination.path
                receipt["trial_checkpoint_path"] = destination.appendingPathComponent("trained/export/model.safetensors").path
                receipt["trial_diffuse_path"] = destination.appendingPathComponent("dataset/samples/train/diffuse.png").path
                receipt["trial_original_height_path"] = destination.appendingPathComponent("dataset/samples/train/height.png").path
                receipt["trial_depthpro_input_path"] = destination.appendingPathComponent("inference/depthpro-input.float32.exr").path
                receipt["trial_height_prediction_path"] = destination.appendingPathComponent("inference/height.float32.exr").path
            }
            if let report = environment["TEXTURE_STUDIO_DEPTHPRO_REPORT"], !report.isEmpty {
                let destination = URL(fileURLWithPath: report)
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: destination)
            }
        }.value
    }

    private static func roundTrip(root: URL, dataset suppliedDataset: URL, fixture: PairFixture,
                                  cache: NativeDepthProInputCache, epochs: Int) throws -> [String: Any] {
        let dataset = fixture.preparedDatasetURL
        let recorder = Recorder()
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path,
            "--output", root.appendingPathComponent("trained").path, "--model-family", "depthpro-height-refiner",
            "--target", "height", "--size", "256", "--compact-width", "16", "--updates-per-map", String(epochs),
            "--seed", "31", "--learning-rate", "0.001"])
        let trained = try NativeMaterialTransfer.object(Data(NativeMaterialTrainer.train(options,
            onEvent: { recorder.record($0) }, control: .init(), depthProCache: cache).utf8))
        XCTAssertEqual(trained["completed_updates"] as? Int, epochs)
        XCTAssertEqual(trained["completed_epochs"] as? Int, epochs)
        XCTAssertEqual(trained["skipped_sample_count"] as? Int, 0)
        XCTAssertEqual(trained["model_family"] as? String, "depthpro-height-refiner")
        let checkpoint = URL(fileURLWithPath: try XCTUnwrap(trained["checkpoint_path"] as? String))
        let (configuration, files) = try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent())
        XCTAssertEqual(configuration["input_channels"] as? Int, 1)
        XCTAssertEqual(configuration["output_channels"] as? Int, 1)
        XCTAssertEqual(configuration["depthpro_model_sha256"] as? String, cache.modelSHA256)
        XCTAssertEqual(configuration["depthpro_preparation_contract"] as? String, NativeDepthProContract.identifier)
        XCTAssertNil(files["adapter.safetensors"])
        let initial = try NativeCompactMaterialModel(target: "height", width: 16, seed: 31, family: .depthProHeightRefiner)
        let weights = try NativeSafetensors(contentsOf: checkpoint).nativeTensors()
        for name in ["input.weight", "middle.expand.weight", "output.weight"] {
            XCTAssertNotEqual(initial.weights[name]?.bytes, weights[name]?.bytes, "The complete refiner learns from the paired height.")
        }
        let prepared = try cache.input(imageURL: fixture.inputURL, expectedSHA256: fixture.inputSHA256, width: 256, height: 256)
        let inferenceCache = NativeDepthProInputCache(modelSHA256: cache.modelSHA256, directory: cache.directory) { _ in
            XCTFail("Inference of the same source must reuse the training input.")
            throw StudioError("Unexpected upstream prediction")
        }
        let inferenceOptions = try NativeMaterialTrainer.Options(["infer", "--checkpoint", checkpoint.path,
            "--image", fixture.inputURL.path, "--output", root.appendingPathComponent("inference").path])
        let inferred = try NativeMaterialTransfer.object(Data(NativeMaterialTrainer.infer(inferenceOptions,
            control: .init(), depthProCache: inferenceCache).utf8))
        XCTAssertEqual(inferred["input_kind"] as? String, "depthpro")
        XCTAssertEqual(inferred["input_channels"] as? Int, 1)
        XCTAssertEqual(inferred["output_channels"] as? Int, 1)
        XCTAssertEqual(inferred["native_dimensions"] as? [Int], [256, 256])
        let generation = try XCTUnwrap(inferred["generation"] as? [String: Any])
        XCTAssertEqual(generation["model_input_dimensions"] as? [Int], [256, 256])
        XCTAssertEqual(generation["depthpro_input_dimensions"] as? [Int], [1536, 1536])
        XCTAssertEqual(generation["depthpro_original_width"] as? Int, 256)
        XCTAssertEqual(generation["depthpro_full_source_field_of_view"] as? Bool, true)
        XCTAssertEqual(generation["source_pixels_resized"] as? Bool, true)
        XCTAssertEqual(generation["source_pixels_discarded"] as? Bool, false)
        XCTAssertEqual(generation["refiner_input_resized"] as? Bool, false)
        XCTAssertEqual(generation["refiner_input_channels"] as? Int, 1)
        let inputEXR = URL(fileURLWithPath: try XCTUnwrap(inferred["prepared_input_path"] as? String))
        let inferredInput = try scalarEXR(inputEXR, width: 256, height: 256)
        XCTAssertEqual(prepared.map(\.bitPattern), inferredInput.map(\.bitPattern),
            "Training and inference must share exact prepared depth bits and top-left coordinates.")
        XCTAssertEqual(inferenceCache.generatedCount, 0); XCTAssertEqual(inferenceCache.reusedCount, 1)
        let output = try XCTUnwrap((inferred["outputs"] as? [String: [String: Any]])?["height"])
        let outputURL = URL(fileURLWithPath: try XCTUnwrap(output["path"] as? String))
        let values = try scalarEXR(outputURL, width: 256, height: 256)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        XCTAssertEqual(output["storage"] as? String, "FLOAT32")
        XCTAssertEqual(try NativeMaterialTransfer.hash(fixture.inputURL), fixture.inputSHA256)
        XCTAssertEqual(try NativeMaterialTransfer.hash(fixture.heightURL), fixture.heightSHA256)
        XCTAssertEqual(recorder.events.filter { $0["event"] as? String == "update" }.count, epochs)

        let mismatched = NativeDepthProInputCache(modelSHA256: String(repeating: "f", count: 64), directory: cache.directory) { _ in
            XCTFail("Upstream identity mismatch must fail before a prediction.")
            throw StudioError("Unexpected upstream prediction")
        }
        XCTAssertThrowsError(try NativeMaterialTrainer.infer(inferenceOptions, control: .init(), depthProCache: mismatched)) { error in
            XCTAssertTrue(error.localizedDescription.contains("exact DepthPro model"))
        }
        let resume = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--checkpoint", checkpoint.path,
            "--size", "256", "--updates-per-map", "1", "--output", root.appendingPathComponent("mismatch").path])
        XCTAssertThrowsError(try NativeMaterialTrainer.train(resume, onEvent: { _ in }, control: .init(), depthProCache: mismatched))
        return ["depthpro_model_sha256": cache.modelSHA256, "preparation_contract": NativeDepthProContract.identifier,
            "model_family": "depthpro-height-refiner", "input_channels": 1, "output_channels": 1,
            "completed_updates": epochs, "generated_inputs": cache.generatedCount,
            "cached_inputs_reused": cache.reusedCount + inferenceCache.reusedCount,
            "input_numeric_roundtrip_bitwise_equal": true, "source_and_target_unchanged": true,
            "source_sha256": fixture.inputSHA256, "height_sha256": fixture.heightSHA256,
            "output_sha256": try NativeMaterialTransfer.hash(outputURL), "output_minimum": values.min()!,
            "output_maximum": values.max()!, "output_storage": "FLOAT32", "upstream_mismatch_rejected": true]
    }

    private struct PairFixture {
        let preparedDatasetURL: URL
        let inputURL: URL, heightURL: URL
        let inputSHA256: String, heightSHA256: String
    }
    private static func trainingConfiguration(_ checkpoint: URL) throws -> [String: Any] {
        let snapshot = try NativeSafetensors(contentsOf: checkpoint)
        let text = try XCTUnwrap(snapshot.metadata["configuration"])
        let configuration = try NativeMaterialTransfer.object(Data(text.utf8))
        return try XCTUnwrap(configuration["training_configuration"] as? [String: Any])
    }
    private static func datasetFixture(_ requestedRoot: URL, input: Data? = nil, height: Data? = nil) throws -> PairFixture {
        let root = requestedRoot.deletingLastPathComponent().lastPathComponent == ".training-data"
            ? requestedRoot : requestedRoot.appendingPathComponent(".training-data/dataset")
        let original = root.deletingLastPathComponent().deletingLastPathComponent()
        let folder = root.appendingPathComponent("samples/train")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let inputBytes = try input ?? photo(width: 256, height: 256, variant: 0)
        let heightBytes: Data
        if let height { heightBytes = height }
        else {
            var pixels = Data(count: 256 * 256 * 2)
            pixels.withUnsafeMutableBytes { storage in
                let bytes = storage.bindMemory(to: UInt8.self)
                for y in 0..<256 { for x in 0..<256 {
                    let code = UInt16((x * 191 + y * 37 + (x / 17 % 2) * 8000) % 65536)
                    let offset = (y * 256 + x) * 2
                    bytes[offset] = UInt8(code >> 8); bytes[offset + 1] = UInt8(code & 255)
                } }
            }
            heightBytes = try NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0),
                pixels: pixels, colorChunks: []).encoded()
        }
        let inputURL = folder.appendingPathComponent("diffuse.png"), heightURL = folder.appendingPathComponent("height.png")
        try inputBytes.write(to: inputURL); try heightBytes.write(to: heightURL)
        let fixture = PairFixture(preparedDatasetURL: root, inputURL: inputURL, heightURL: heightURL,
            inputSHA256: NativeMaterialTrainer.checksum(inputBytes), heightSHA256: NativeMaterialTrainer.checksum(heightBytes))
        let originals = original.appendingPathComponent("originals/material-a")
        try FileManager.default.createDirectory(at: originals, withIntermediateDirectories: true)
        try inputBytes.write(to: originals.appendingPathComponent("diffuse.png"))
        try heightBytes.write(to: originals.appendingPathComponent("height.png"))
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "name": "Depth refiner originals", "samples": [[String: Any]]()])
            .write(to: original.appendingPathComponent("dataset.json"))
        let transform: [String: Any] = ["type": "native_crop", "algorithm": "exact_native_integer_codes", "range_normalization": false, "gamma_applied": false]
        let sample: [String: Any] = ["sample_id": "train", "material_id": "material-a", "status": "approved", "split": "train",
            "sample_pixel_dimensions": [256, 256], "crop_rectangle_top_left_xywh": [0, 0, 256, 256],
            "source_square_rectangle_top_left_xywh": [0, 0, 256, 256], "maps": ["input": "diffuse.png", "height": "height.png"],
            "map_metadata": ["input": ["filename": "diffuse.png", "encoding": "srgb", "sample_sha256": fixture.inputSHA256,
                "storage": "prepared_crop", "exact_source_crop": true, "transforms": [transform],
                "source": ["path": originals.appendingPathComponent("diffuse.png").path, "file_sha256": fixture.inputSHA256, "width": 256, "height": 256]],
                "height": ["filename": "height.png", "sample_sha256": fixture.heightSHA256,
                "storage": "prepared_crop", "exact_source_crop": true, "transforms": [transform],
                "source": ["path": originals.appendingPathComponent("height.png").path, "file_sha256": fixture.heightSHA256, "width": 256, "height": 256]]]]
        try JSONSerialization.data(withJSONObject: sample).write(to: folder.appendingPathComponent("sample.json"))
        try JSONSerialization.data(withJSONObject: ["schema_version": 2,
            "native_size_preparation": ["schema": "texture-studio-native-crops-v1", "source_dataset_path": original.path, "target_resized": false],
            "samples": [["sample_id": "train", "material_id": "material-a", "status": "approved", "split": "train", "path": "samples/train"]],
            "validation": ["enabled": false, "quick_count": 0]]).write(to: root.appendingPathComponent("dataset.json"))
        return fixture
    }

    private static func realMaterial(environment: [String: String]) throws -> (input: Data?, height: Data?, description: String) {
        let defaultPhoto = "/opt/ipde/sources_mats/white_stucco_02/white_stucco_02_diff_2k.png"
        let defaultHeight = "/opt/ipde/sources_mats/white_stucco_02/white_stucco_02_disp_2k.png"
        let requestedPhoto = environment["TEXTURE_STUDIO_DEPTHPRO_INPUT"], requestedHeight = environment["TEXTURE_STUDIO_DEPTHPRO_HEIGHT"]
        guard (requestedPhoto == nil) == (requestedHeight == nil) else { throw StudioError("Provide both DepthPro smoke input and original height paths.") }
        let photoPath = requestedPhoto ?? defaultPhoto, heightPath = requestedHeight ?? defaultHeight
        if requestedPhoto == nil, !FileManager.default.fileExists(atPath: photoPath) || !FileManager.default.fileExists(atPath: heightPath) {
            return (nil, nil, "procedural 256 RGB plus original 16-bit scalar fixture")
        }
        let photo = try NativePNG.decode(Data(contentsOf: URL(fileURLWithPath: photoPath)))
        let height = try NativePNG.decode(Data(contentsOf: URL(fileURLWithPath: heightPath)))
        guard photo.header.width == height.header.width, photo.header.height == height.header.height,
              photo.header.width >= 256, photo.header.height >= 256 else { throw StudioError("Smoke material pair must share a grid of at least 256 pixels.") }
        let crop = [(photo.header.width - 256) / 2, (photo.header.height - 256) / 2, 256, 256]
        return (try photo.crop(crop).encoded(), try height.crop(crop).encoded(),
            "exact aligned 256 center crop from original material: \(photoPath); height: \(heightPath)")
    }

    private static func photo(width: Int, height: Int, variant: Int) throws -> Data {
        var pixels = Data(count: width * height * 3)
        pixels.withUnsafeMutableBytes { storage in
            let bytes = storage.bindMemory(to: UInt8.self)
            for y in 0..<height { for x in 0..<width {
                let offset = (y * width + x) * 3
                bytes[offset] = UInt8((x * 3 + y + variant * 31) % 251)
                bytes[offset + 1] = UInt8((x + y * 5 + variant * 17) % 251)
                bytes[offset + 2] = UInt8((x * 7 + y * 2 + variant * 13) % 251)
            } }
        }
        return try NativePNG(header: .init(width: width, height: height, bits: 8, channels: 3, color: 2, interlace: 0),
            pixels: pixels, colorChunks: []).encoded()
    }
    private static func analyticDepth(_ image: CGImage) -> ModelDepthResult {
        depth(width: image.width, height: image.height, values: (0..<image.width * image.height).map {
            1 + Float($0 % image.width) / Float(image.width) + Float($0 / image.width) * 2 / Float(image.height)
        })
    }
    private static func depth(width: Int, height: Int, values: [Float]) -> ModelDepthResult {
        .init(width: width, height: height, values: values, outputName: "depthMeters", interpretation: "Camera depth in meters")
    }
    private static func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("depthpro-refiner-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private static func relocateTrialManifests(from original: URL, to retained: URL) throws {
        func relocated(_ value: Any) -> Any {
            if let path = value as? String, path == original.path || path.hasPrefix(original.path + "/") {
                return retained.path + String(path.dropFirst(original.path.count))
            }
            if let object = value as? [String: Any] { return object.mapValues(relocated) }
            if let array = value as? [Any] { return array.map(relocated) }
            return value
        }
        // These operation manifests are outside the checksummed export. Keep
        // its model, configuration, indexes and hashes exactly as published.
        for name in ["trained/run.json", "inference/inference.json"] {
            let url = retained.appendingPathComponent(name)
            let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            try JSONSerialization.data(withJSONObject: relocated(object), options: [.prettyPrinted, .sortedKeys])
                .write(to: url, options: .atomic)
        }
    }

    /// Read the generated uncompressed scalar EXR directly. This keeps numeric
    /// storage and top-left row order independent of preview/color conversion.
    private static func scalarEXR(_ url: URL, width: Int, height: Int) throws -> [Float] {
        let bytes = try Data(contentsOf: url)
        func u32(_ offset: Int) throws -> UInt32 {
            guard offset >= 0, offset <= bytes.count - 4 else { throw StudioError("Truncated scalar EXR word.") }
            return bytes.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }
        func u64(_ offset: Int) throws -> UInt64 {
            guard offset >= 0, offset <= bytes.count - 8 else { throw StudioError("Truncated scalar EXR offset.") }
            return bytes.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) }
        }
        XCTAssertEqual(try u32(0), 20_000_630)
        var cursor = 8, attributes: [String: Data] = [:]
        func string() throws -> String {
            let start = cursor
            while cursor < bytes.count, bytes[cursor] != 0 { cursor += 1 }
            guard cursor < bytes.count else { throw StudioError("Truncated scalar EXR attribute.") }
            defer { cursor += 1 }
            return String(decoding: bytes[start..<cursor], as: UTF8.self)
        }
        while cursor < bytes.count, bytes[cursor] != 0 {
            let name = try string(); _ = try string()
            let count = Int(try u32(cursor)); cursor += 4
            guard count <= bytes.count - cursor else { throw StudioError("Invalid scalar EXR attribute length.") }
            attributes[name] = bytes.subdata(in: cursor..<cursor + count); cursor += count
        }
        cursor += 1
        XCTAssertEqual(attributes["compression"], Data([0]))
        XCTAssertEqual(attributes["lineOrder"], Data([0]))
        let channels = try XCTUnwrap(attributes["channels"])
        XCTAssertEqual(channels.prefix(2), Data([89, 0]))
        XCTAssertEqual(channels.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 2, as: UInt32.self)) }, 2)
        var values = [Float](repeating: 0, count: width * height)
        for row in 0..<height {
            let offset = Int(try u64(cursor + row * 8))
            XCTAssertEqual(try u32(offset), UInt32(row))
            XCTAssertEqual(try u32(offset + 4), UInt32(width * 4))
            for x in 0..<width { values[row * width + x] = Float(bitPattern: try u32(offset + 8 + x * 4)) }
        }
        return values
    }

    private final class Counter {
        private(set) var value = 0
        func increment() { value += 1 }
    }
    private final class CropEpochRecorder: @unchecked Sendable {
        struct Snapshot { let photo: Data, height: Data, depthInput: Data, depthMeters: Data }
        private let lock = NSLock()
        private let cropDirectory: URL, depthDirectory: URL, modelSHA256: String
        private(set) var snapshots: [Snapshot] = []
        private(set) var errors: [String] = []
        init(cropDirectory: URL, depthDirectory: URL, modelSHA256: String) {
            self.cropDirectory = cropDirectory; self.depthDirectory = depthDirectory; self.modelSHA256 = modelSHA256
        }
        func record(_ line: String) {
            guard let event = try? NativeMaterialTransfer.object(Data(line.utf8)),
                  event["event"] as? String == "crop_preparation", event["operation"] as? String == "Cached paired crop",
                  let id = event["sample_id"] as? String else { return }
            do {
                let stem = NativeMaterialTrainer.checksum(Data(id.utf8))
                let geometry = String(id.split(separator: "|", maxSplits: 1)[0])
                let targetStem = NativeMaterialTrainer.checksum(Data(geometry.utf8))
                let photo = try Data(contentsOf: cropDirectory.appendingPathComponent(stem + "-diffuse.png"))
                let height = try Data(contentsOf: cropDirectory.appendingPathComponent(targetStem + "-height.png"))
                let binding = [NativeDepthProContract.identifier, NativeMaterialTrainer.checksum(photo), modelSHA256, "256", "256"].joined(separator: "|")
                let key = NativeMaterialTrainer.checksum(Data(binding.utf8))
                let entry = try NativeSafetensors(contentsOf: depthDirectory.appendingPathComponent(key + ".safetensors"))
                let snapshot = Snapshot(photo: photo, height: height, depthInput: try entry.tensorBytes(named: "input"),
                    depthMeters: try entry.tensorBytes(named: "depth_meters"))
                lock.withLock { snapshots.append(snapshot) }
            } catch { lock.withLock { errors.append(error.localizedDescription) } }
        }
    }
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var events: [[String: Any]] = []
        func record(_ line: String) {
            if let event = try? NativeMaterialTransfer.object(Data(line.utf8)) { lock.withLock { events.append(event) } }
        }
    }
}
