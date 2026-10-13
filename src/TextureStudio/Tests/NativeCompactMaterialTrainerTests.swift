import Darwin
import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeCompactMaterialTrainerTests: XCTestCase {
    func testOptionsChooseCompatibleFamiliesAndKeepLegacyDefaults() throws {
        let arguments = ["train", "--dataset", "/unused", "--output", "/new"]
        let legacy = try NativeMaterialTrainer.Options(arguments)
        XCTAssertEqual(legacy.modelFamily, .pbrnxt); XCTAssertEqual(legacy.scope, "final-map")
        let scalar = try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar"])
        XCTAssertEqual(scalar.size, 512); XCTAssertEqual(scalar.compactWidth, 32)
        XCTAssertEqual(scalar.scope, "full-model"); XCTAssertEqual(scalar.learningRate, 0.001)
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar", "--target", "normal"]))
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-normal"]))
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar", "--scope", "final-map"]))
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar", "--compact-width", "24"]))
    }

    func testPublicTrainingRejectsGlobalFoldersPreparedGlobalLineageAndMaterialSubsets() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let prepared = root.appendingPathComponent(".training-data/fixture")
        try fixture(prepared, target: "height")
        for command in ["train", "refine"] {
            for path in [root, prepared] {
                let output = root.appendingPathComponent("rejected-" + command + "-" + path.lastPathComponent)
                do {
                    _ = try await NativeMaterialTrainer.run(arguments: [command, "--dataset", path.path,
                        "--output", output.path, "--model-family", "compact-scalar", "--size", "256",
                        "--developer-mode", "--crop-selection-prepared"])
                    XCTFail("Product training requires a material category even for developer mode or prepared data")
                } catch {
                    XCTAssertTrue(error.localizedDescription.contains("Choose a material category"))
                    XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
                }
            }
        }
        let category = try await categoryFixture(root.appendingPathComponent("Sources"), target: "height", trainingCount: 2)
        do {
            _ = try await NativeMaterialTrainer.run(arguments: ["train", "--dataset", category.path,
                "--output", root.appendingPathComponent("subset-rejected").path, "--model-family", "compact-scalar",
                "--size", "256", "--material", "train-0"])
            XCTFail("A selected folder cannot narrow a category's tag-defined membership")
        } catch { XCTAssertTrue(error.localizedDescription.contains("current file tags")) }
    }

    func testCheckpointOptionsPinWeightDigestAndRejectLegacyFamilyConfusion() throws {
        let fixture = try NativeMaterialPackageFixture(); defer { fixture.remove() }
        let checkpoint = fixture.root.appendingPathComponent("adapter.safetensors")
        let original = try fixture.adapter().bytes
        try original.write(to: checkpoint)
        let arguments = ["infer", "--baseline", "--checkpoint", checkpoint.path,
            "--image", "/unused.png", "--output", fixture.root.appendingPathComponent("output").path]
        let options = try NativeMaterialTrainer.Options(arguments)
        XCTAssertEqual(options.expectedSHA256, NativeMaterialTrainer.checksum(original))
        let confused = try NativeMaterialTrainer.Options(arguments + ["--model-family", "compact-scalar", "--scope", "full-model"])
        XCTAssertThrowsError(try NativeTrainableMaterialModel.load(confused, training: false))
        try fixture.adapter(a: [9, 10]).bytes.write(to: checkpoint)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: checkpoint, expectedSHA256: options.expectedSHA256))
    }

    func testAllFamiliesTrainExportAndInferWithoutDownloadedBase() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for target in ["height", "roughness", "normal"] {
            let source = root.appendingPathComponent("Sources-" + target)
            let dataset = try await categoryFixture(source, target: target)
            let sourceInput = source.appendingPathComponent("train-0/diffuse.png")
            let sourceTarget = source.appendingPathComponent("train-0/" + target + ".png")
            let originalTarget = try Data(contentsOf: sourceTarget)
            let output = root.appendingPathComponent("trained-" + target)
            let family = target == "normal" ? "compact-normal" : "compact-scalar"
            var arguments = ["train", "--crop-selection-prepared", "--randomize-crops", "never", "--dataset", dataset.path, "--output", output.path,
                "--model-family", family, "--target", target, "--size", "256", "--compact-width", "16",
                "--updates-per-map", "1", "--seed", "31", "--model-directory", root.appendingPathComponent("missing-base").path]
            if target == "height" { arguments += ["--exaggerate-features"] }
            let recorder = Recorder()
            let result = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: arguments,
                onEvent: { recorder.record($0) }).utf8))
            XCTAssertEqual(result["completed_updates"] as? Int, 1)
            XCTAssertEqual(result["model_family"] as? String, family)
            XCTAssertEqual(result["output_channels"] as? Int, target == "normal" ? 3 : 1)
            let checkpoint = URL(fileURLWithPath: try XCTUnwrap(result["checkpoint_path"] as? String))
            XCTAssertEqual(checkpoint.lastPathComponent, "model.safetensors")
            let (configuration, inventory) = try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent())
            XCTAssertEqual(configuration["target"] as? String, target)
            let trainingConfiguration = try XCTUnwrap(configuration["training_configuration"] as? [String: Any])
            XCTAssertEqual(trainingConfiguration["exaggerate_features"] as? Bool, target == "height")
            XCTAssertEqual(trainingConfiguration["depth_target_transform"] as? String,
                target == "height" ? "native-range-anchored-tanh-contrast-strength-2" : "original-native-values")
            XCTAssertEqual(try Data(contentsOf: sourceTarget), originalTarget)
            XCTAssertEqual(configuration["input_tensor_transfer"] as? String, NativeMaterialInputTransfer.linear.rawValue)
            XCTAssertNil(configuration["base"]); XCTAssertNil(inventory["adapter.safetensors"])
            let initial = try NativeCompactMaterialModel(target: target, width: 16, seed: 31)
            let trained = try NativeSafetensors(contentsOf: checkpoint).nativeTensors()
            XCTAssertNotEqual(initial.weights["input.weight"]!.bytes, trained["input.weight"]!.bytes)
            XCTAssertNotEqual(initial.weights["output.weight"]!.bytes, trained["output.weight"]!.bytes)
            let hash = try NativeMaterialTransfer.hash(checkpoint)
            let inferred = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["infer",
                "--checkpoint", checkpoint.path, "--expected-sha256", hash,
                "--image", sourceInput.path,
                "--output", root.appendingPathComponent("inferred-" + target).path]).utf8))
            XCTAssertEqual(inferred["native_dimensions"] as? [Int], [256, 256])
            XCTAssertEqual(inferred["output_channels"] as? Int, target == "normal" ? 3 : 1)
            XCTAssertEqual(inferred["model_family"] as? String, family)
            XCTAssertEqual(inferred["checkpoint_sha256"] as? String, hash)
            XCTAssertEqual(inferred["input_tensor_transfer"] as? String, NativeMaterialInputTransfer.linear.rawValue)
            if target == "height" {
                let baseline = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["infer", "--baseline",
                    "--checkpoint", checkpoint.path, "--image", sourceInput.path,
                    "--output", root.appendingPathComponent("untrained").path]).utf8))
                XCTAssertEqual(baseline["untrained_initialization"] as? Bool, true)
                XCTAssertEqual(baseline["checkpoint_step"] as? Int, 0)
                for override in [["--target", "roughness"], ["--compact-width", "32"], ["--seed", "32"]] {
                    let changed = try NativeMaterialTrainer.Options(["infer", "--baseline", "--checkpoint", checkpoint.path,
                        "--image", sourceInput.path,
                        "--output", root.appendingPathComponent("invalid-baseline").path] + override)
                    XCTAssertThrowsError(try NativeTrainableMaterialModel.load(changed, training: false))
                }
                let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: dataset,
                    size: 256, target: "height", automaticValidation: true)
                var changed = try NativePNG.decode(originalTarget)
                changed.pixels[0] ^= 1
                let replacement = try changed.encoded()
                try replacement.write(to: sourceTarget)
                let resumed = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["train", "--crop-selection-prepared",
                    "--dataset", prepared.path, "--checkpoint", checkpoint.path, "--expected-sha256", hash,
                    "--size", "256", "--updates-per-map", "1", "--output", root.appendingPathComponent("resumed").path]).utf8))
                XCTAssertEqual(resumed["completed_updates"] as? Int, 1)
                XCTAssertEqual(resumed["skipped_sample_count"] as? Int, 0,
                               "Rewritten originals must be freshly prepared before a public run")
                XCTAssertEqual(try Data(contentsOf: sourceTarget), replacement)
                XCTAssertEqual(NativeSourceFileTags.read(sourceInput)?.classLabel, "Wood")
                let inspected = try NativeMaterialCheckpoint.inspect(at: URL(fileURLWithPath: try XCTUnwrap(resumed["checkpoint_path"] as? String)))
                let info = try NativeMaterialTransfer.object(Data(inspected.utf8))
                XCTAssertEqual(info["step"] as? Int, 2)
                XCTAssertEqual(info["network_width"] as? Int, 16)
                XCTAssertEqual(info["initialization_seed"] as? UInt64, 31)
            }
            XCTAssertEqual(recorder.events.filter { $0["event"] as? String == "update" }.count, 1)
        }
    }

    func testFinalValidationFailureStillPublishesCompactWeights() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent(".training-data/dataset"), output = root.appendingPathComponent("trained")
        try fixture(dataset, target: "height", validation: true)
        let recorder = Recorder()
        let validationMap = dataset.appendingPathComponent("samples/validation/height.png")
        let options = try NativeMaterialTrainer.Options(["train", "--randomize-crops", "never",
            "--dataset", dataset.path, "--output", output.path, "--model-family", "compact-scalar", "--size", "256",
            "--compact-width", "16", "--updates-per-map", "1"])
        let result = try NativeMaterialTransfer.object(Data(try NativeMaterialTrainer.train(options, onEvent: { line in
                recorder.record(line)
                if let event = try? NativeMaterialTransfer.object(Data(line.utf8)), event["event"] as? String == "update" {
                    try? Data("corrupt validation sample".utf8).write(to: validationMap)
                }
            }, control: .init()).utf8))
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
        XCTAssertEqual(result["skipped_sample_count"] as? Int, 1)
        let checkpoint = URL(fileURLWithPath: try XCTUnwrap(result["checkpoint_path"] as? String))
        XCTAssertNoThrow(try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent()))
        XCTAssertTrue(recorder.events.contains { $0["event"] as? String == "sample_skipped" && $0["phase"] as? String == "validation" })
    }

    func testPreparedCategoryTrainingExportKeepsCategoryNameSubjectAnd16BitTarget() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("Artist Bark Collection")
        _ = try await NativeMaterialDatasetService.run(arguments: ["create-dataset", "--dataset", original.path,
            "--source-folder", "true", "--name", "Artist Bark Collection", "--training-size", "256", "--validation-enabled", "no"], onEvent: { _ in })
        var sourceBytes: [URL: Data] = [:]
        for (variant, name) in ["Pine Bark", "Fabric"].enumerated() {
            let folder = original.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let diffuse = folder.appendingPathComponent("diffuse.png"), height = folder.appendingPathComponent("height.png")
            let input = try NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 3, color: 2, interlace: 0),
                pixels: Data((0..<256 * 256 * 6).map { UInt8(($0 * 7 + variant * 31) % 251) }), colorChunks: []).encoded()
            let target = try NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0),
                pixels: Data((0..<256 * 256 * 2).map { UInt8(($0 * 3 + variant * 13) % 251) }), colorChunks: []).encoded()
            try input.write(to: diffuse); try target.write(to: height)
            sourceBytes[diffuse] = input; sourceBytes[height] = target
            try NativeSourceFileTags.update(diffuse, role: "input", classLabel: variant == 0 ? "Bark" : "Fabric", classifierIdentity: "user")
            try NativeSourceFileTags.update(height, role: "height")
        }
        let category = try await category(from: original, label: "Bark")
        let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: category, size: 256,
            target: "height", automaticValidation: false)
        let preparedIndex = try NativeMaterialTransfer.object(Data(contentsOf: prepared.appendingPathComponent("dataset.json")))
        XCTAssertNil(preparedIndex["name"], "This regression covers a prepared manifest with no copied display name")
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertEqual(descriptors.filter { $0.split == "train" }.count, 1)
        XCTAssertEqual(Set(descriptors.compactMap(\.materialName)), ["Pine Bark"])
        XCTAssertEqual(try NativePNG.inspect(XCTUnwrap(descriptors.first?.targetURL)).bits, 16)
        let result = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["train", "--crop-selection-prepared",
            "--dataset", prepared.path, "--output", root.appendingPathComponent("trained").path,
            "--model-family", "compact-scalar", "--target", "height", "--size", "256", "--compact-width", "16",
            "--model-name", "Pine Bark Surface Height", "--updates-per-map", "1", "--max-minutes", "1",
            "--randomize-crops", "never", "--crop-selection-prepared"]).utf8))
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
        let checkpoint = URL(fileURLWithPath: try XCTUnwrap(result["checkpoint_path"] as? String))
        let (configuration, _) = try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent())
        XCTAssertEqual(configuration["texture_pipeline"] as? String, "MatterLoom")
        XCTAssertEqual(configuration["training_dataset_name"] as? String, "Bark")
        XCTAssertEqual(configuration["training_subjects"] as? [String], ["Pine Bark"])
        XCTAssertEqual(configuration["model_name"] as? String, "Pine Bark Surface Height")
        let readme = try String(contentsOf: checkpoint.deletingLastPathComponent().appendingPathComponent("README.md"), encoding: .utf8)
        XCTAssertTrue(readme.contains("MatterLoom texture pipeline"))
        XCTAssertTrue(readme.contains("surface height / displacement specialist for Pine Bark"))
        for (url, bytes) in sourceBytes { XCTAssertEqual(try Data(contentsOf: url), bytes) }
    }

    /// Opt-in short throughput probe. It deliberately does not assess trained quality.
    func testCompact512TrainingBenchmark() async throws {
        try await benchmark(size: 512)
    }

    func testCompact1024TrainingBenchmark() async throws {
        try await benchmark(size: 1024)
    }

    private func benchmark(size: Int) async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_COMPACT_BENCHMARK"] == "1" else { throw XCTSkip("Opt-in compact training throughput probe.") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await categoryFixture(root.appendingPathComponent("Sources"), target: "height", size: size, trainingCount: 3)
        let report = URL(fileURLWithPath: environment["TEXTURE_STUDIO_COMPACT_BENCHMARK_OUTPUT"] ?? "/opt/ipde/ipde/out/compact-native-benchmark.json")
        let warmStart = environment["TEXTURE_STUDIO_COMPACT_BENCHMARK_CHECKPOINT"].map { URL(fileURLWithPath: $0) }
        let originalDigest = try warmStart.map { try NativeMaterialTransfer.hash($0) }
        let recorder = Recorder(), start = ProcessInfo.processInfo.systemUptime
        var arguments = ["train", "--crop-selection-prepared",
            "--dataset", dataset.path, "--output", root.appendingPathComponent("trained").path,
            "--model-family", "compact-scalar", "--size", String(size), "--updates-per-map", "2"]
        if let warmStart, let originalDigest { arguments += ["--checkpoint", warmStart.path, "--expected-sha256", originalDigest] }
        let result = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: arguments,
            onEvent: { recorder.record($0) }).utf8))
        let updates = recorder.events.filter { $0["event"] as? String == "update" }
        XCTAssertEqual(result["completed_updates"] as? Int, 6)
        XCTAssertEqual(updates.count, 6)
        XCTAssertTrue(updates.allSatisfy { ($0["total"] as? Double)?.isFinite == true })
        if let warmStart, let originalDigest {
            XCTAssertEqual(try NativeMaterialTransfer.hash(warmStart), originalDigest)
            let final = try NativeMaterialTransfer.object(Data(NativeMaterialCheckpoint.inspect(at:
                URL(fileURLWithPath: try XCTUnwrap(result["checkpoint_path"] as? String))).utf8))
            XCTAssertEqual(final["training_size"] as? Int, size)
            XCTAssertEqual(final["step"] as? Int, (updates.first?["initial_step"] as? Int ?? 0) + 6)
        }
        var resources = rusage(); getrusage(RUSAGE_SELF, &resources)
        var receipt: [String: Any] = ["model_family": "compact-scalar", "size": size, "network_width": 32,
            "parameters": 999_937, "fixture": "three distinct procedural native RGB/16-bit scalar pairs; two epochs",
            "device": device.name, "total_wall_seconds": ProcessInfo.processInfo.systemUptime - start,
            "peak_resident_bytes": resources.ru_maxrss, "metal_allocated_bytes_at_completion": device.currentAllocatedSize,
            "updates": updates, "completed_updates": 6,
            "quality_qualification": false, "cold_step_includes_graph_compilation": true]
        if let originalDigest { receipt["warm_start_sha256"] = originalDigest; receipt["initial_step"] = updates.first?["initial_step"] }
        try FileManager.default.createDirectory(at: report.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: report)
    }

    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [[String: Any]] = []
        func record(_ text: String) {
            guard let event = try? NativeMaterialTransfer.object(Data(text.utf8)) else { return }
            lock.withLock { recorded.append(event) }
        }
        var events: [[String: Any]] { lock.withLock { recorded } }
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("compact-training-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true); return root
    }

    private func categoryFixture(_ source: URL, target: String, size: Int = 256, trainingCount: Int = 1) async throws -> URL {
        for variant in 0..<trainingCount {
            let folder = source.appendingPathComponent("train-\(variant)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let input = try NativePNG(header: .init(width: size, height: size, bits: 8, channels: 3, color: 2, interlace: 0),
                pixels: Data((0..<size * size * 3).map { UInt8(($0 * 7 + $0 / size + variant * 31) % 251) }), colorChunks: []).encoded()
            let normal = target == "normal", channels = normal ? 3 : 1, bits = normal ? 8 : 16
            let reference = try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels, color: normal ? 2 : 0, interlace: 0),
                pixels: Data((0..<size * size * channels * (bits / 8)).map { UInt8(($0 * 3 + variant * 13) % 251) }), colorChunks: []).encoded()
            let photo = folder.appendingPathComponent("diffuse.png"), map = folder.appendingPathComponent(target + ".png")
            try input.write(to: photo); try reference.write(to: map)
            try NativeSourceFileTags.update(photo, role: "input", classLabel: "Wood", classifierIdentity: "user")
            try NativeSourceFileTags.update(map, role: target)
        }
        return try await category(from: source, label: "Wood")
    }

    private func category(from source: URL, label: String) async throws -> URL {
        let classifier = NativeMaterialSpecialistService.Classifier(path: "/unused-manual-category.mlmodelc",
            sha256: String(repeating: "a", count: 64), labels: [label, "Fabric"], minimumImageSize: 0) { _ in
                throw StudioError("Manually taught fixtures must not invoke a classifier.")
            }
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: source,
            workspaceURL: source.deletingLastPathComponent().appendingPathComponent("Categories-" + source.lastPathComponent),
            classifier: classifier)
        return URL(fileURLWithPath: try XCTUnwrap(registry.specialists.first { $0.classLabel == label }?.datasetPath))
    }

    private func fixture(_ root: URL, target: String, size: Int = 256, trainingCount: Int = 1, validation: Bool = false) throws {
        let original = root.deletingLastPathComponent().deletingLastPathComponent()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var entries: [[String: Any]] = []
        let identities = (0..<trainingCount).map { ("train-\($0)", "train", $0) } + (validation ? [("validation", "validation", trainingCount)] : [])
        for (id, split, variant) in identities {
            let folder = root.appendingPathComponent("samples/" + id)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let input = try NativePNG(header: .init(width: size, height: size, bits: 8, channels: 3, color: 2, interlace: 0),
                pixels: Data((0..<size * size * 3).map { UInt8(($0 * 7 + $0 / size + variant * 31) % 251) }), colorChunks: []).encoded()
            let normal = target == "normal", channels = normal ? 3 : 1, bits = normal ? 8 : 16
            let reference = try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels, color: normal ? 2 : 0, interlace: 0),
                pixels: Data((0..<size * size * channels * (bits / 8)).map { UInt8(($0 * 3 + variant * 13) % 251) }), colorChunks: []).encoded()
            try input.write(to: folder.appendingPathComponent("diffuse.png")); try reference.write(to: folder.appendingPathComponent(target + ".png"))
            entries.append(["sample_id": id, "material_id": id, "status": "approved", "split": split, "path": "samples/" + id])
            let originals = original.appendingPathComponent("originals/" + id)
            try FileManager.default.createDirectory(at: originals, withIntermediateDirectories: true)
            try input.write(to: originals.appendingPathComponent("diffuse.png"))
            try reference.write(to: originals.appendingPathComponent(target + ".png"))
            let transform: [String: Any] = ["type": "native_crop", "algorithm": "exact_native_integer_codes", "range_normalization": false, "gamma_applied": false]
            let sample: [String: Any] = ["sample_id": id, "material_id": id, "status": "approved", "split": split,
                "sample_pixel_dimensions": [size, size], "maps": ["input": "diffuse.png", target: target + ".png"],
                "map_metadata": ["input": ["filename": "diffuse.png", "encoding": "srgb", "sample_sha256": NativeMaterialTrainer.checksum(input), "storage": "prepared_crop", "exact_source_crop": true, "transforms": [transform], "source": ["path": originals.appendingPathComponent("diffuse.png").path, "file_sha256": NativeMaterialTrainer.checksum(input), "width": size, "height": size]],
                    target: ["filename": target + ".png", "sample_sha256": NativeMaterialTrainer.checksum(reference), "storage": "prepared_crop", "exact_source_crop": true, "transforms": [transform], "source": ["path": originals.appendingPathComponent(target + ".png").path, "file_sha256": NativeMaterialTrainer.checksum(reference), "width": size, "height": size]]]]
            try JSONSerialization.data(withJSONObject: sample).write(to: folder.appendingPathComponent("sample.json"))
        }
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "native_size_preparation": ["schema": "texture-studio-native-crops-v1", "source_dataset_path": original.path, "target_resized": false], "samples": entries,
            "validation": ["enabled": validation, "quick_count": 1]]).write(to: root.appendingPathComponent("dataset.json"))
    }
}
