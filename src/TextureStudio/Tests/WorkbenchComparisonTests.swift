import Foundation
import Metal
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchComparisonTests: XCTestCase {
    func testImportedPhotosPrepareAtCompactGridRegardlessOfSourceDimensions() async throws {
        for (side, width, height) in [(512, 64, 48), (512, 777, 529), (512, 2048, 1536), (256, 777, 529)] {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let suite = "imported-comparison-\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            defer { defaults.removePersistentDomain(forName: suite) }
            let image = root.appendingPathComponent("new-material-\(width)-\(height).png")
            let original = try NativePNG(header: .init(width: width, height: height, bits: 8, channels: 3, color: 2, interlace: 0),
                pixels: Data((0..<width * height * 3).map { UInt8(($0 * 7 + $0 / width) % 251) }),
                colorChunks: [("sRGB", Data([0]))]).encoded()
            try original.write(to: image)
            var calls: [[String]] = []
            var inputs: [String] = []
            let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
                calls.append(arguments)
                let input = arguments[try XCTUnwrap(arguments.firstIndex(of: "--image")) + 1]
                inputs.append(input)
                XCTAssertEqual(arguments[try XCTUnwrap(arguments.firstIndex(of: "--input-encoding")) + 1], "srgb")
                XCTAssertNotEqual(input, image.path, "Only the prepared model input follows the training grid")
                let png = try NativePNG.inspect(URL(fileURLWithPath: input))
                XCTAssertEqual(png.width, side)
                XCTAssertEqual(png.height, side)
                XCTAssertEqual(png.bits, 16, "Preparation retains color precision rather than writing an 8-bit display preview")
                let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
                return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"compact-sha\"}"
            })
            store.sourceImageURL = image
            store.training.modelFamily = .compactScalar
            store.training.size = side
            let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
            {"checkpoint_path":"/models/compact/model.safetensors","sha256":"compact-sha",
             "schema":"texture-studio-compact-material-v1","architecture":"texture-studio-compact-scalar-native-v1",
             "target":"height","step":42,"compatible":true,"variant":"full","supports_training_warm_start":true}
            """)
            store.checkpoints = [checkpoint]
            store.comparisonCheckpointIds = [checkpoint.id]
            store.compare()
            try await waitForOperation(store, timeout: 20)
            XCTAssertNil(store.error)
            XCTAssertEqual(calls.count, 2)
            XCTAssertEqual(Set(inputs).count, 1, "Baseline and trained model must receive the same prepared pixels")
            XCTAssertEqual(store.comparisonCandidates.map(\.role), ["diffuse", "base", "checkpoint"])
            XCTAssertEqual(store.comparisonCandidates.first?.sampleLabel, image.lastPathComponent)
            XCTAssertEqual(try Data(contentsOf: image), original, "Import and comparison never resize or rewrite the source")
        }
    }

    func testComparisonPreservesSelectedDiffuseVariantEncodingForBothModels() async throws {
        for encoding in ["linear_rgb", "source_srgb_assumed", nil] as [String?] {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let suite = "comparison-transfer-\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            defer { defaults.removePersistentDomain(forName: suite) }
            let input = root.appendingPathComponent("selected-variant.png")
            let original = try NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0),
                pixels: Data(repeating: 128, count: 256 * 256 * 3), colorChunks: []).encoded()
            try original.write(to: input)
            let expectedEncoding = encoding ?? "srgb"
            let normalized = Float(128) / 255
            let expectedTensorValue = encoding == "linear_rgb" ? normalized : pow((normalized + 0.055) / 1.055, 2.4)
            var inferenceCount = 0
            let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
                XCTAssertEqual(arguments.first, "infer")
                inferenceCount += 1
                let path = arguments[try XCTUnwrap(arguments.firstIndex(of: "--image")) + 1]
                XCTAssertEqual(path, input.path, "The selected variant supplies both model inputs")
                let suppliedEncoding = arguments[try XCTUnwrap(arguments.firstIndex(of: "--input-encoding")) + 1]
                XCTAssertEqual(suppliedEncoding, expectedEncoding)
                let tensor = try NativePNG.decode(Data(contentsOf: URL(fileURLWithPath: path)))
                    .modelFloatSamples(role: "input", encoding: suppliedEncoding, inputTransfer: .linear)
                XCTAssertEqual(try XCTUnwrap(tensor.first), expectedTensorValue, accuracy: 0.000001,
                    "Comparison must repeat the dataset transfer instead of decoding linear source values as sRGB")
                let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
                return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"exact-sha\"}"
            })
            var variant: [String: Any] = ["path": input.path, "width": 256, "height": 256, "variant_id": "selected"]
            variant["encoding"] = encoding
            let document: [String: Any] = ["dataset_path": root.path, "index_sha256": "index", "materials": [
                ["material_id": "soil", "samples": [["sample_id": "soil", "status": "approved", "split": "train",
                    "width": 256, "height": 256, "maps": ["input": ["path": "/unselected-srgb.png", "encoding": "srgb"]],
                    "input_variants": [variant]]]]]]
            store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self,
                output: String(decoding: JSONSerialization.data(withJSONObject: document), as: UTF8.self))
            store.selectedSampleId = "soil"
            store.selectedInputVariantId = "selected"
            store.training.size = 256
            store.checkpoints = [try checkpoint(sha: "exact-sha", target: "height")]
            store.comparisonCheckpointIds = ["exact-sha"]
            store.compare()
            try await waitForOperation(store)
            XCTAssertNil(store.error)
            XCTAssertEqual(inferenceCount, 2)
            let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf:
                XCTUnwrap(store.lastOutputURL).appendingPathComponent("review-manifest.json"))) as? [String: Any]
            let material = (manifest?["materials"] as? [[String: Any]])?.first
            XCTAssertEqual(material?["diffuse_source_encoding"] as? String, expectedEncoding)
            XCTAssertEqual(try Data(contentsOf: input), original)
        }
    }

    func testCompactComparisonRunsRecordedUntrainedInitializationWithoutPBRDependency() async throws {
        for target in ["height", "roughness", "normal"] {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let suite = "compact-comparison-\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            defer { defaults.removePersistentDomain(forName: suite) }
            var calls: [[String]] = []
            let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
                calls.append(arguments)
                let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
                return "{\"outputs\":{\"\(target)\":{\"path\":\"\(output)/\(target).exr\"}},\"checkpoint_sha256\":\"exact-compact-sha\"}"
            })
            store.modelDirectory = root.appendingPathComponent("missing-pbr-base").path
            store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
            {"dataset_path":"/dataset","index_sha256":"dataset-sha","materials":[{"material_id":"soil","samples":[
              {"sample_id":"soil_crop","status":"approved","split":"train","width":1024,"height":1024,
               "maps":{"input":{"path":"/dataset/soil/photo.png"},"\(target)":{"path":"/dataset/soil/\(target).png"}}}]}]}
            """)
            store.selectedSampleId = "soil_crop"
            let family: MaterialTrainingModelFamily = target == "normal" ? .compactNormal : .compactScalar
            let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
            {"checkpoint_path":"/models/compact/model.safetensors","sha256":"exact-compact-sha",
             "schema":"texture-studio-compact-material-v1","architecture":"\(family.architecture!)",
             "target":"\(target)","step":42,"compatible":true,"variant":"full","supports_training_warm_start":true}
            """)
            store.checkpoints = [checkpoint]
            store.comparisonCheckpointIds = [checkpoint.id]
            XCTAssertEqual(store.comparisonBaselineLabel, "Include untrained compact initialization")
            XCTAssertNil(store.comparisonConfigurationIssue)
            store.compare()
            try await waitForOperation(store)
            XCTAssertNil(store.error)
            XCTAssertEqual(calls.count, 2)
            XCTAssertTrue(calls[0].contains("--baseline"))
            XCTAssertFalse(calls[1].contains("--baseline"))
            for call in calls {
                XCTAssertFalse(call.contains("--model-directory"))
                XCTAssertTrue(call.contains("exact-compact-sha"))
                XCTAssertTrue(call.contains(checkpoint.checkpointPath))
            }
            let baseline = try XCTUnwrap(store.comparisonCandidates.first { $0.role == "base" })
            XCTAssertTrue(baseline.label.contains("Untrained initialization"))
            XCTAssertTrue(baseline.detail?.contains("recorded random initialization") == true)
            XCTAssertEqual(baseline.modelIdentity?.architecture, family.architecture)
            XCTAssertEqual(baseline.modelIdentity?.mapType, target)
            XCTAssertNil(baseline.modelIdentity?.checkpointStep)
        }
    }

    func testComparisonReconstructsTrainingGridAndRetainsOnlyOriginalReferences() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "comparison-grid-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.path, forKey: "workspace")
        let diffuse = root.appendingPathComponent("source-diffuse.png")
        let height = root.appendingPathComponent("source-height.png")
        try Data("original diffuse values".utf8).write(to: diffuse)
        try Data("original height values".utf8).write(to: height)
        let diffuseHash = ReviewImageLoader.hash(try Data(contentsOf: diffuse))
        let heightHash = ReviewImageLoader.hash(try Data(contentsOf: height))
        var calls: [[String]] = []
        var temporaryInput: URL?
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
            calls.append(arguments)
            let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
            if arguments.first == "review-source" {
                XCTAssertEqual(arguments[try XCTUnwrap(arguments.firstIndex(of: "--image")) + 1], diffuse.path)
                XCTAssertEqual(arguments[try XCTUnwrap(arguments.firstIndex(of: "--expected-sha256")) + 1], diffuseHash)
                XCTAssertEqual(arguments[try XCTUnwrap(arguments.firstIndex(of: "--size")) + 1], "1024")
                temporaryInput = URL(fileURLWithPath: output)
                try Data("exact training resize".utf8).write(to: temporaryInput!)
                return "{}"
            }
            let input = URL(fileURLWithPath: arguments[try XCTUnwrap(arguments.firstIndex(of: "--image")) + 1])
            XCTAssertEqual(input, temporaryInput)
            XCTAssertEqual(try Data(contentsOf: input), Data("exact training resize".utf8))
            return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"exact-sha\"}"
        })
        let maps: [String: Any] = ["input": ["path": "/purged-stage/diffuse.png", "width": 1024, "height": 1024,
            "original_source_path": diffuse.path, "original_source_sha256": diffuseHash,
            "original_source_width": 4096, "original_source_height": 4096],
            "height": ["path": "/purged-stage/height.png", "width": 1024, "height": 1024,
            "source_bits": 16, "original_source_path": height.path, "original_source_sha256": heightHash,
            "original_source_width": 4096, "original_source_height": 4096]]
        let document: [String: Any] = ["dataset_path": root.path, "index_sha256": "index-sha", "materials": [
            ["material_id": "soil", "samples": [["sample_id": "soil_full", "status": "approved", "split": "train",
                "width": 1024, "height": 1024, "maps": maps]]]]]
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        store.dataset = try decoder.decode(WorkbenchDataset.self, from: JSONSerialization.data(withJSONObject: document))
        store.selectedSampleId = "soil_full"
        store.checkpoints = [try checkpoint(sha: "exact-sha", target: "height")]
        store.comparisonCheckpointIds = ["exact-sha"]
        store.compare()
        try await waitForOperation(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["review-source", "infer", "infer"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: try XCTUnwrap(temporaryInput).deletingLastPathComponent().path))
        XCTAssertEqual(store.comparisonCandidates.first?.mapURL, diffuse)
        XCTAssertEqual(store.comparisonCandidates.first?.displayTransform?.size, 1024)
        XCTAssertEqual(store.comparisonCandidates.first { $0.role == "target" }?.mapURL, height)
        let review = ReviewSessionStore(preferences: defaults)
        review.load(try XCTUnwrap(store.lastOutputURL).appendingPathComponent("review-manifest.json"))
        XCTAssertNil(review.error)
        XCTAssertEqual(review.groups.first?.candidates.first?.mapURL, diffuse)
        XCTAssertEqual(review.groups.first?.candidates.first?.displayTransform?.sourceSHA256, diffuseHash)
        XCTAssertEqual(review.groups.first?.candidates.first { $0.role == "target" }?.displayTransform?.sourceSHA256, heightHash)
        XCTAssertEqual(try Data(contentsOf: diffuse), Data("original diffuse values".utf8))
    }

    func testOneCheckpointComparesWithHonestBaseAndLabelsSampleAndReference() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "comparison-test-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.path, forKey: "workspace")
        var calls: [[String]] = []
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
            calls.append(arguments)
            let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
            return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"exact-sha\"}"
        })
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
        {"dataset_path":"/dataset","index_sha256":"dataset-sha","materials":[{"material_id":"soil","samples":[{"sample_id":"soil_crop_002","status":"approved","split":"train","width":2048,"height":2048,"maps":{"input":{"path":"/dataset/soil/photo.png"},"height":{"path":"/dataset/soil/displacement.png","source_bits":16}}}]}]}
        """)
        store.selectedSampleId = "soil_crop_002"
        store.checkpoints = [try checkpoint(sha: "exact-sha", target: "height")]
        store.comparisonCheckpointIds = ["exact-sha"]
        store.compare()
        try await waitForOperation(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.count, 2, "One baseline and one selected checkpoint run")
        XCTAssertTrue(calls[0].contains("--baseline"))
        XCTAssertFalse(calls[1].contains("--baseline"))
        for call in calls {
            XCTAssertEqual(call[try XCTUnwrap(call.firstIndex(of: "--image")) + 1], "/dataset/soil/photo.png")
            XCTAssertEqual(call[try XCTUnwrap(call.firstIndex(of: "--expected-sha256")) + 1], "exact-sha")
        }
        XCTAssertEqual(store.comparisonCandidates.map(\.role), ["diffuse", "target", "base", "checkpoint"])
        XCTAssertTrue(store.comparisonCandidates.allSatisfy { $0.sampleLabel == "soil_crop_002" })
        let base = try XCTUnwrap(store.comparisonCandidates.first { $0.role == "base" })
        XCTAssertTrue(base.label.contains("Base"))
        XCTAssertTrue(base.detail?.contains("before refinement") == true)
        XCTAssertEqual(base.modelIdentity?.architecture, "PBRnxt material model")
        XCTAssertNil(base.modelIdentity?.checkpointStep, "The untrained baseline does not inherit a trained checkpoint's step")
        let trained = try XCTUnwrap(store.comparisonCandidates.first { $0.role == "checkpoint" })
        XCTAssertTrue(trained.detail?.contains("step 42") == true)
        XCTAssertEqual(trained.modelIdentity?.checkpointPath, "/runs/material-2k/model.safetensors")
        XCTAssertEqual(trained.modelIdentity?.checkpointSHA256, "exact-sha")
        XCTAssertEqual(trained.modelIdentity?.checkpointStep, 42)
        XCTAssertEqual(trained.modelIdentity?.architecture, "PBRnxt material model")
        XCTAssertEqual(trained.modelIdentity?.mapType, "height")
        XCTAssertTrue(trained.accessibleLabel.contains("soil_crop_002"))
        XCTAssertTrue(trained.exportFilename.contains("soil_crop_002"))
        XCTAssertTrue(trained.exportFilename.contains("exact-sha"))
        XCTAssertFalse(trained.exportFilename.contains("/"))
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates(store.comparisonCandidates).map(\.role), ["target", "base", "checkpoint"])
        let manifest = try XCTUnwrap(store.lastOutputURL).appendingPathComponent("review-manifest.json")
        let review = ReviewSessionStore()
        let savedPreference = UserDefaults(suiteName: "org.ipde.material-tools")!.object(forKey: "reviewManifest")
        defer { UserDefaults(suiteName: "org.ipde.material-tools")!.set(savedPreference, forKey: "reviewManifest") }
        review.load(manifest)
        XCTAssertNil(review.error)
        XCTAssertEqual(review.groups.first?.id, "soil_crop_002")
        XCTAssertEqual(review.groups.first?.candidates.map(\.role), ["diffuse", "target", "base", "checkpoint"])
        XCTAssertEqual(review.groups.first?.candidates.last?.detail, trained.detail)
        XCTAssertEqual(review.groups.first?.candidates.last?.modelIdentity, trained.modelIdentity)
        XCTAssertEqual(review.groups.first?.candidates.first { $0.role == "base" }?.modelIdentity, base.modelIdentity)
        let reviewed = root.appendingPathComponent("decisions.json")
        try review.writeReview(to: reviewed)
        review.load(reviewed)
        XCTAssertEqual(review.groups.first?.candidates.last?.modelIdentity, trained.modelIdentity,
            "Saving decisions must retain the inspected live model identity")
    }

    func testEverySelectedCheckpointIsInitiallyVisibleAndSourceCanBeEnabled() {
        let roles = ["source", "base", "checkpoint", "checkpoint", "checkpoint"]
        let candidates = roles.enumerated().map { index, role in
            MapReviewCandidate(id: "\(index)", label: "Candidate \(index)", mapURL: URL(fileURLWithPath: "/map-\(index).exr"),
                numeric: role != "source", sampleLabel: "sand_crop_003", detail: "Step \(index * 100)", role: role)
        }
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates(candidates).map(\.id), ["1", "2", "3", "4"])
        XCTAssertTrue(candidates[0].accessibleLabel.contains("sand_crop_003"))
    }

    func testSingleCheckpointWithoutBaselineRunsWithOrWithoutSourceHeight() async throws {
        for includeReference in [true, false] {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let suite = "single-output-comparison-\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            defer { defaults.removePersistentDomain(forName: suite) }
            var calls: [[String]] = []
            let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
                calls.append(arguments)
                let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
                return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"exact-sha\"}"
            })
            let reference = includeReference ? ",\"height\":{\"path\":\"/dataset/soil/height.png\"}" : ""
            store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
            {"dataset_path":"/dataset","index_sha256":"index","materials":[{"material_id":"soil","samples":[{"sample_id":"soil","status":"approved","split":"train","width":1024,"height":1024,"maps":{"input":{"path":"/dataset/soil/photo.png"}\(reference)}}]}]}
            """)
            store.selectedSampleId = "soil"
            store.checkpoints = [try checkpoint(sha: "exact-sha", target: "height")]
            store.comparisonCheckpointIds = ["exact-sha"]
            store.comparisonIncludesBase = false
            XCTAssertNil(store.comparisonConfigurationIssue)
            store.compare()
            try await waitForOperation(store)
            XCTAssertNil(store.error)
            XCTAssertEqual(calls.count, 1, "Disabling the baseline runs only the selected checkpoint")
            XCTAssertFalse(try XCTUnwrap(calls.first).contains("--baseline"))
            let roles = includeReference ? ["diffuse", "target", "checkpoint"] : ["diffuse", "checkpoint"]
            XCTAssertEqual(store.comparisonCandidates.map(\.role), roles)
            XCTAssertEqual(ReviewWorkbenchView.initialCandidates(store.comparisonCandidates).map(\.role),
                includeReference ? ["target", "checkpoint"] : ["checkpoint"])
            let review = ReviewSessionStore(preferences: defaults)
            review.load(try XCTUnwrap(store.lastOutputURL).appendingPathComponent("review-manifest.json"))
            XCTAssertNil(review.error)
            XCTAssertEqual(review.groups.first?.candidates.map(\.role), roles)
        }
    }

    func testRGBCheckpointComparisonUsesOriginalSourceHeightWithoutDepthProSetupOrInputPane() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "rgb-source-height-comparison-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let size = 64, diffuse = root.appendingPathComponent("diffuse.png"), height = root.appendingPathComponent("original-height.png")
        let diffuseBytes = try NativePNG(header: .init(width: size, height: size, bits: 8, channels: 3, color: 2, interlace: 0),
            pixels: Data((0..<size * size * 3).map { UInt8(($0 * 7 + $0 / size) % 251) }), colorChunks: []).encoded()
        let heightBytes = try NativePNG(header: .init(width: size, height: size, bits: 16, channels: 1, color: 0, interlace: 0),
            pixels: Data((0..<size * size * 2).map { UInt8(($0 * 3) % 251) }), colorChunks: []).encoded()
        try diffuseBytes.write(to: diffuse); try heightBytes.write(to: height)
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 23)
        let checkpointURL = root.appendingPathComponent("rgb-model.safetensors")
        try NativeSafetensors.write(tensors: model.weights,
            metadata: ["configuration": NativeMaterialTransfer.json(model.checkpointConfiguration(size: size, step: 0))], to: checkpointURL)
        let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: NativeMaterialCheckpoint.inspect(at: checkpointURL))
        let forbiddenTransfer = ForbiddenDepthProTransfer()
        let missingResources = root.appendingPathComponent("unused-depthpro/DepthPro.mlpackage")
        let resources = NativeDepthProResources(managedURL: missingResources, transfer: forbiddenTransfer)
        var calls: [[String]] = []
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
            calls.append(arguments)
            XCTAssertEqual(arguments.first, "infer")
            XCTAssertFalse(arguments.contains("--depthpro-model"), "RGB checkpoints have no DepthPro prerequisite")
            XCTAssertFalse(arguments.contains("--baseline"), "Only the selected RGB checkpoint should execute")
            let result = try await NativeMaterialTrainer.run(arguments: arguments, depthProResources: resources)
            let object = try NativeMaterialTransfer.object(Data(result.utf8))
            XCTAssertEqual(object["input_kind"] as? String, "diffuse")
            XCTAssertNil(object["prepared_input_path"])
            XCTAssertNil(object["depthpro_model_sha256"])
            return result
        })
        store.depthProModelPath = missingResources.path
        let document: [String: Any] = ["dataset_path": root.path, "index_sha256": "fixture-index", "materials": [
            ["material_id": "bark", "samples": [["sample_id": "bark-native", "status": "approved", "split": "train",
                "width": size, "height": size, "maps": [
                    "input": ["path": diffuse.path, "encoding": "srgb", "width": size, "height": size],
                    "height": ["path": height.path, "sha256": NativeMaterialTrainer.checksum(heightBytes),
                        "source_bits": 16, "width": size, "height": size]]]]]]]
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self,
            output: String(decoding: JSONSerialization.data(withJSONObject: document), as: UTF8.self))
        store.selectedSampleId = "bark-native"
        store.training.size = size
        // An unrelated saved training family must not choose inference dependencies.
        store.training.modelFamily = .depthProHeightRefiner
        store.checkpoints = [checkpoint]
        store.comparisonCheckpointIds = [checkpoint.id]
        store.comparisonIncludesBase = false
        store.compare()
        try await waitForOperation(store, timeout: 30)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(forbiddenTransfer.count, 0, "RGB inference cannot download or run DepthPro")
        XCTAssertFalse(FileManager.default.fileExists(atPath: missingResources.deletingLastPathComponent().path))
        XCTAssertEqual(store.comparisonCandidates.map(\.role), ["diffuse", "target", "checkpoint"])
        XCTAssertFalse(store.comparisonCandidates.contains { $0.role == "model-input" || $0.label.contains("DepthPro") })
        let reference = try XCTUnwrap(store.comparisonCandidates.first { $0.isReference })
        XCTAssertEqual(reference.mapURL, height)
        XCTAssertEqual(reference.sourceIdentity?.sha256, NativeMaterialTrainer.checksum(heightBytes))
        XCTAssertEqual(reference.sourceIdentity?.bits, 16)
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates(store.comparisonCandidates).map(\.role), ["target", "checkpoint"])
        let review = ReviewSessionStore(preferences: defaults)
        review.load(try XCTUnwrap(store.lastOutputURL).appendingPathComponent("review-manifest.json"))
        XCTAssertNil(review.error)
        XCTAssertEqual(review.groups.first?.candidates.map(\.role), ["diffuse", "target", "checkpoint"])
        XCTAssertEqual(try Data(contentsOf: diffuse), diffuseBytes)
        XCTAssertEqual(try Data(contentsOf: height), heightBytes)
    }

    func testBaselineOnlyRunsWithRecordedCheckpointOrConfiguredCompactInitialization() async throws {
        for useRecordedCheckpoint in [true, false] {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let suite = "baseline-only-comparison-\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            defer { defaults.removePersistentDomain(forName: suite) }
            var calls: [[String]] = []
            let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
                calls.append(arguments)
                let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
                return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"compact-sha\"}"
            })
            store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
            {"dataset_path":"/dataset","index_sha256":"index","materials":[{"material_id":"soil","samples":[{"sample_id":"soil","status":"approved","split":"train","width":1024,"height":1024,"maps":{"input":{"path":"/dataset/photo.png"},"height":{"path":"/dataset/height.png"}}}]}]}
            """)
            store.selectedSampleId = "soil"
            store.training.modelFamily = .compactScalar
            store.training.seed = 123
            if useRecordedCheckpoint {
                let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
                {"checkpoint_path":"/models/compact/model.safetensors","sha256":"compact-sha","schema":"texture-studio-compact-material-v1","architecture":"texture-studio-compact-scalar-native-v1","target":"height","step":42,"compatible":true}
                """)
                store.checkpoints = [checkpoint]
                store.selectedCheckpointId = checkpoint.id
            }
            store.comparisonCheckpointIds = []
            store.comparisonIncludesBase = true
            XCTAssertEqual(store.comparisonBaselineLabel, "Include untrained compact initialization")
            XCTAssertNil(store.comparisonConfigurationIssue)
            store.compare()
            try await waitForOperation(store)
            XCTAssertNil(store.error)
            XCTAssertEqual(calls.count, 1)
            let call = try XCTUnwrap(calls.first)
            XCTAssertTrue(call.contains("--baseline"))
            XCTAssertFalse(call.contains("--model-directory"))
            XCTAssertEqual(call.contains("--checkpoint"), useRecordedCheckpoint)
            if useRecordedCheckpoint {
                XCTAssertTrue(call.contains("compact-sha"))
                XCTAssertFalse(call.contains("--seed"), "A checkpoint baseline uses the recorded seed")
            } else {
                XCTAssertEqual(call[try XCTUnwrap(call.firstIndex(of: "--model-family")) + 1], "compact-scalar")
                XCTAssertEqual(call[try XCTUnwrap(call.firstIndex(of: "--target")) + 1], "height")
                XCTAssertEqual(call[try XCTUnwrap(call.firstIndex(of: "--seed")) + 1], "123")
            }
            XCTAssertEqual(store.comparisonCandidates.map(\.role), ["diffuse", "target", "base"])
            let baseline = try XCTUnwrap(store.comparisonCandidates.last)
            XCTAssertEqual(baseline.modelIdentity?.architecture, MaterialTrainingModelFamily.compactScalar.architecture)
            XCTAssertNil(baseline.modelIdentity?.checkpointStep)
        }
    }

    func testComparisonModelSelectionRequiresCheckpointOrBaseline() throws {
        let suite = "comparison-selection-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.sourceImageURL = URL(fileURLWithPath: "/photo.png")
        store.checkpoints = [try checkpoint(sha: "height", target: "height")]
        for includeCheckpoint in [true, false] {
            for includeBase in [true, false] {
                store.comparisonCheckpointIds = includeCheckpoint ? ["height"] : []
                store.comparisonIncludesBase = includeBase
                XCTAssertEqual(store.comparisonConfigurationIssue == nil, includeCheckpoint || includeBase)
            }
        }
        store.comparisonCheckpointIds = []
        store.comparisonIncludesBase = false
        store.compare()
        XCTAssertNotNil(store.error)
        XCTAssertFalse(store.isBusy)
    }

    func testComparisonStillRejectsDifferentMapTypesBeforeLaunchingWorkers() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "comparison-test-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.path, forKey: "workspace")
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { _, _ in
            XCTFail("Invalid comparison must not launch workers")
            return "{}"
        })
        store.sourceImageURL = URL(fileURLWithPath: "/photo.png")
        store.checkpoints = [try checkpoint(sha: "height", target: "height"), try checkpoint(sha: "normal", target: "normal")]
        store.comparisonIncludesBase = true
        store.comparisonCheckpointIds = ["height", "normal"]
        store.compare()
        XCTAssertNotNil(store.error)
        XCTAssertFalse(store.isBusy)
    }

    func testComparisonUsesSelectedSamplesVerifiedFullSourceAndKeepsCropCoordinates() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "selected-reference-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.path, forKey: "workspace")
        var samples: [[String: Any]] = []
        var fullSources: [URL] = []
        var crops: [URL] = []
        for index in 0..<2 {
            let folder = root.appendingPathComponent("sample-\(index)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            let map = folder.appendingPathComponent("displacement.png")
            let bytes = Data("Exact cropped height bytes \(index)".utf8)
            try bytes.write(to: map)
            let original = root.appendingPathComponent("original-4k-height-\(index).png")
            try Data("Full original height bytes \(index)".utf8).write(to: original)
            let hash = ReviewImageLoader.hash(bytes)
            let metadata: [String: Any] = ["sample_id": "sample-\(index)", "maps": ["height": "displacement.png"],
                "crop_rectangle_top_left_xywh": [index * 2048, 0, 2048, 2048],
                "map_metadata": ["height": ["sample_sha256": hash, "source": ["path": original.path,
                    "file_sha256": "original-source-sha-\(index)", "sample_bits": 16, "width": 4096, "height": 4096,
                    "published_url": "https://example.com/material-\(index)-disp.png"]]]]
            try JSONSerialization.data(withJSONObject: metadata).write(to: folder.appendingPathComponent("sample.json"))
            samples.append(["sample_id": "sample-\(index)", "status": "approved", "split": "train", "width": 2048, "height": 2048,
                "maps": ["input": ["path": folder.appendingPathComponent("photo.png").path],
                    "height": ["path": map.path, "sha256": hash, "source_bits": 16, "width": 2048, "height": 2048]]])
            fullSources.append(original); crops.append(map)
        }
        let document: [String: Any] = ["dataset_path": root.path, "index_sha256": "fixture-index-sha",
            "materials": [["material_id": "brick", "samples": samples]]]
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
            let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
            return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"exact-sha\"}"
        })
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        store.dataset = try decoder.decode(WorkbenchDataset.self, from: JSONSerialization.data(withJSONObject: document))
        store.selectedSampleId = "sample-1"
        store.checkpoints = [try checkpoint(sha: "exact-sha", target: "height")]
        store.comparisonCheckpointIds = ["exact-sha"]
        store.compare()
        // Selection may change while a comparison is running. The captured
        // sample, its source and the model input must remain the same pair.
        store.selectedSampleId = "sample-0"
        try await waitForOperation(store)
        XCTAssertNil(store.error)
        let reference = try XCTUnwrap(store.comparisonCandidates.first { $0.isReference })
        XCTAssertEqual(reference.sampleLabel, "sample-1")
        XCTAssertEqual(reference.mapURL, crops[1])
        XCTAssertEqual(reference.sourceIdentity?.path, fullSources[1].path)
        XCTAssertEqual(reference.sourceIdentity?.cropRectangle, [2048, 0, 2048, 2048])
        XCTAssertEqual(reference.sourceIdentity?.pixelDimensions, [4096, 4096])
        XCTAssertEqual(reference.fullSourceReference?.mapURL, fullSources[1])
        XCTAssertNotEqual(reference.fullSourceReference?.mapURL, fullSources[0])
        XCTAssertEqual(try Data(contentsOf: crops[1]), Data("Exact cropped height bytes 1".utf8))
    }

    private func checkpoint(sha: String, target: String) throws -> WorkbenchCheckpoint {
        try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
        {"checkpoint_path":"/runs/material-2k/model.safetensors","sha256":"\(sha)","schema":"texture-studio-material-checkpoint-v1","target":"\(target)","step":42,"compatible":true}
        """)
    }

    private final class ForbiddenDepthProTransfer: ModelArtifactTransferring, @unchecked Sendable {
        private let lock = NSLock()
        private var requests = 0
        var count: Int { lock.withLock { requests } }
        func fetch(_ artifact: ModelDownloadArtifact, to destination: URL,
                   progress: @escaping @Sendable (Double) -> Void) async throws {
            lock.withLock { requests += 1 }
            // Cancellation avoids downloader retries if family routing regresses.
            throw CancellationError()
        }
    }

    private func waitForOperation(_ store: WorkbenchStore, timeout: TimeInterval = 5) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while store.isBusy, Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isBusy)
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("comparison-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
