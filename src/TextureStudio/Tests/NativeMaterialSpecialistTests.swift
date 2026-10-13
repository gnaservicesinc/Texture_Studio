import CryptoKit
import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeMaterialSpecialistTests: XCTestCase {
    func testFileTagsReusePredictionsAndManualTeachingSurvivesPixelReplacement() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        let originals = try await add("cedar", size: 256, value: 31, to: dataset, under: root)
        let input = try XCTUnwrap(originals.first { $0.0.lastPathComponent == "input.png" }?.0)
        let probe = PredictionProbe()
        let classifier = fake(labels: ["Wood", "Paper"], probe: probe, choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: root.appendingPathComponent("FirstWorkspace"), classifier: classifier)
        XCTAssertEqual(probe.paths, [input.path])
        XCTAssertEqual(try specialist("Wood", initial).materialIds.count, 1)
        let automatic = try XCTUnwrap(NativeSourceFileTags.read(input))
        XCTAssertEqual(automatic.classLabel, "Wood")
        XCTAssertEqual(automatic.classifierIdentity, classifier.sha256)
        XCTAssertEqual(automatic.verifiedSHA256?.count, 64)

        // A fresh workspace proves reuse comes from the file, independently
        // of any prior specialist assignment or loaded Core ML prediction.
        let secondWorkspace = root.appendingPathComponent("SecondWorkspace")
        let tagged = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: secondWorkspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1)
        XCTAssertEqual(try specialist("Wood", tagged).materialIds.count, 1)
        try NativeSourceFileTags.update(input, classLabel: "Paper", classifierIdentity: "user")
        let taught = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: secondWorkspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1, "Teaching a file must override cached classifier assignments without another prediction")
        XCTAssertEqual(try specialist("Paper", taught).assignments.first?.source, "user_file_tag")
        XCTAssertEqual(try specialist("Paper", taught).materialIds.count, 1)
        XCTAssertEqual(NativeSourceFileTags.read(input)?.classifierIdentity, "user")
        XCTAssertEqual(NativeSourceFileTags.read(input)?.classLabel, "Paper")
        XCTAssertEqual(try Data(contentsOf: input), try XCTUnwrap(originals.first { $0.0 == input }?.1))
        try NativeSourceFileTags.update(input, classLabel: "Handmade Finish", classifierIdentity: "user")
        let custom = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: root.appendingPathComponent("CustomTaughtWorkspace"), classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1)
        XCTAssertEqual(try specialist("Handmade Finish", custom).materialIds.count, 1,
                       "User teaching may define a class beyond the classifier's original taxonomy")

        try png(size: 256, value: 103).write(to: input)
        let retained = try XCTUnwrap(NativeSourceFileTags.read(input))
        XCTAssertEqual(retained.classLabel, "Handmade Finish")
        XCTAssertEqual(retained.classifierIdentity, "user")
        XCTAssertNil(retained.verifiedSHA256, "Pixel replacement expires verification, while explicit teaching belongs to the physical file")
        let rebuilt = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: secondWorkspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1)
        XCTAssertEqual(try specialist("Handmade Finish", rebuilt).materialIds.count, 1)
    }

    func testBareSourceFolderClassificationChoosesTwoKOriginalAndPreservesSixteenBitTargets() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("SourceFolders"), woodFolder = source.appendingPathComponent("Wood")
        let smallFolder = source.appendingPathComponent("SmallOnly")
        try FileManager.default.createDirectory(at: woodFolder, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: smallFolder, withIntermediateDirectories: true)
        var originals: [URL: Data] = [:]
        for (folder, family, dimension, label) in [(woodFolder, "wood", 1024, "1k"),
                                                  (woodFolder, "wood", 2048, "2k"),
                                                  (smallFolder, "small", 1024, "1k")] {
            for (suffix, bits, channels) in [("diff", 8, 3), ("disp", 16, 1)] {
                let path = folder.appendingPathComponent("\(family)_\(suffix)_\(label).png")
                let bytes = try png(size: dimension, value: suffix == "disp" ? 171 : 83, bits: bits, channels: channels)
                try bytes.write(to: path); originals[path] = bytes
            }
        }
        let beforeFolders = Set(try FileManager.default.contentsOfDirectory(atPath: source.path))
        let probe = PredictionProbe(), workspace = root.appendingPathComponent("Specialists")
        let classifier = fake(labels: ["Wood", "Paper"], probe: probe, minimumImageSize: 2048, choose: { _ in "Wood" })
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace,
            classifier: classifier)
        XCTAssertEqual(probe.paths, [woodFolder.appendingPathComponent("wood_diff_2k.png").path],
                       "A new source-folder dataset must not classify one-K originals when the classifier grid is two K")
        let woodSpecialist = try specialist("Wood", registry)
        XCTAssertEqual(Set(woodSpecialist.materialIds), ["wood_1k", "wood_2k"])
        XCTAssertEqual(woodSpecialist.assignments.first?.inputPath, probe.paths.first)
        XCTAssertTrue(registry.warnings.contains { $0.contains("small") && $0.contains("2048") })
        let specialistDataset = URL(fileURLWithPath: woodSpecialist.datasetPath)
        let index = try object(specialistDataset.appendingPathComponent("dataset.json"))
        let entries = try XCTUnwrap(index["samples"] as? [[String: Any]])
        XCTAssertEqual(entries.count, 2)
        for entry in entries {
            let sample = try object(specialistDataset.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json"))
            let sourceMetadata = try XCTUnwrap((sample["map_metadata"] as? [String: [String: Any]])?["height"]?["source"] as? [String: Any])
            XCTAssertEqual(sourceMetadata["sample_bits"] as? Int, 16)
            let originalURL = URL(fileURLWithPath: try XCTUnwrap(sourceMetadata["path"] as? String))
            XCTAssertEqual(try NativePNG.inspect(originalURL).bits, 16)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: source.appendingPathComponent("dataset.json").path))
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: source.path)), beforeFolders)
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
        let oneK = woodFolder.appendingPathComponent("wood_diff_1k.png")
        try NativeSourceFileTags.update(oneK, classLabel: "Paper", classifierIdentity: "user")
        let taughtFamily = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace,
            classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1, "A user's label on another native tier overrides the cached family prediction")
        XCTAssertEqual(Set(try specialist("Paper", taughtFamily).materialIds), ["wood_1k", "wood_2k"])
        try NativeSourceFileTags.update(oneK, classLabel: "", classifierIdentity: "user")
        let unlabeled = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace,
            classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1, "Explicitly clearing a label must not trigger another automatic prediction")
        XCTAssertTrue(unlabeled.specialists.flatMap(\.assignments).isEmpty)
        XCTAssertTrue(unlabeled.warnings.contains { $0.contains("unlabeled") })
    }

    func testTaggedNativeOneKFolderParticipatesInItsSpecialistWithoutClassifierResizing() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("Sources"), folder = source.appendingPathComponent("Native Bricks")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let input = folder.appendingPathComponent("brick_diff_1k.png"), target = folder.appendingPathComponent("brick_disp_1k.png")
        let inputBytes = try png(size: 1024, value: 79), targetBytes = try png(size: 1024, value: 152, bits: 16, channels: 1)
        try inputBytes.write(to: input); try targetBytes.write(to: target)
        try NativeSourceFileTags.update(input, role: "input", classLabel: "Bricks", classifierIdentity: "user")
        let probe = PredictionProbe(), classifier = fake(labels: ["Bricks", "Wood"], probe: probe, choose: { _ in "Wood" })
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: source,
            workspaceURL: root.appendingPathComponent("Specialists"), classifier: classifier)
        let selected = try specialist("Bricks", registry)
        XCTAssertEqual(selected.materialIds, ["brick_1k"])
        XCTAssertEqual(selected.assignments.first?.source, "user_file_tag")
        XCTAssertTrue(probe.paths.isEmpty, "A manually tagged class must not require a 2048 classifier crop")
        let preparedText = try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", selected.datasetPath,
            "--size", "1024", "--target", "height", "--skip-source-recovery"])
        let preparedInfo = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(preparedText).utf8)) as? [String: Any])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(preparedInfo["dataset_path"] as? String))
        defer { _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 1024, target: "height")
        XCTAssertEqual(samples.filter { $0.split == "train" }.count, 1)
        XCTAssertTrue(samples.allSatisfy { $0.width == 1024 && $0.height == 1024 })
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
        XCTAssertEqual(try Data(contentsOf: target), targetBytes)
    }

    func testFirstHeaderOnlyCategoryTrainingRegistersWithUnpreparedSmallerTier() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try await dataset(root)
        let small = try await add("wood_planks", size: 128, value: 52, to: source, under: root)
        let large = try await add("wood_planks", size: 256, value: 77, to: source, under: root)
        for (file, _) in small + large where file.lastPathComponent == "input.png" {
            try NativeSourceFileTags.update(file, classLabel: "Wood", classifierIdentity: "user")
        }
        let workspace = root.appendingPathComponent("Categories")
        let classifier = fake(labels: ["Wood"], choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", initial)
        let before = try XCTUnwrap(wood.targets.first { $0.target == "height" })
        XCTAssertEqual(before.materialCount, 2)
        XCTAssertNil(NativeSourceFileTags.read(small[0].0)?.verifiedSHA256)
        let output = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["train",
            "--dataset", wood.datasetPath, "--output", root.appendingPathComponent("Trained").path,
            "--model-family", "compact-scalar", "--target", "height", "--size", "256", "--compact-width", "16",
            "--updates-per-map", "1", "--randomize-crops", "never"]).utf8))
        XCTAssertEqual(output["completed_updates"] as? Int, 1)
        let current = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace)
        XCTAssertEqual(try specialist("Wood", current).targets.first { $0.target == "height" }?.membershipSha256, before.membershipSha256,
                       "Preparing only the eligible tier cannot change the current-content proof")
        let checkpoint = URL(fileURLWithPath: try XCTUnwrap(output["checkpoint_path"] as? String))
        let registered = try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 256,
            completedUpdates: 1, expectedMembershipSHA256: before.membershipSha256)
        let saved = try XCTUnwrap(specialist("Wood", registered).models.first)
        XCTAssertEqual(saved.membershipSchema, "texture-studio-material-content-proof-v1")
        XCTAssertEqual(saved.membershipSha256, before.membershipSha256)
        XCTAssertNil(NativeSourceFileTags.read(small.first { $0.0.lastPathComponent == "height.png" }!.0)?.verifiedSHA256,
                     "A smaller retained original needs a content checksum, not a fabricated prepared-grid verification")
        for (file, bytes) in small + large { XCTAssertEqual(try Data(contentsOf: file), bytes) }
    }

    func testApprovedCategoryModelSurvivesFileAndFolderRenamesAndSafelyMigratesLegacyProof() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try await dataset(root)
        let originals = try await add("wood_planks", size: 256, value: 54, to: source, under: root)
        for (file, _) in originals where file.lastPathComponent == "input.png" {
            try NativeSourceFileTags.update(file, classLabel: "Wood", classifierIdentity: "user")
        }
        let workspace = root.appendingPathComponent("Categories"), classifier = fake(labels: ["Wood"], choose: { _ in "Wood" })
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", registry), category = URL(fileURLWithPath: wood.datasetPath)
        _ = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: category, size: 256, target: "height", automaticValidation: false)
        let checkpoint = try savedCheckpoint(root, dataset: category, step: 2)
        let recorded = try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 64, completedUpdates: 2)
        let model = try XCTUnwrap(specialist("Wood", recorded).models.first)
        _ = try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id)
        // Replace only the persisted model proof with the actual prior-format
        // proof of these owned historical samples, simulating an existing user.
        let entries = try XCTUnwrap(object(category.appendingPathComponent("dataset.json"))["samples"] as? [[String: Any]])
        let samples = try entries.map { entry in try object(category.appendingPathComponent(entry["path"] as! String).appendingPathComponent("sample.json")) }
        let identities = samples.sorted { ($0["material_id"] as? String ?? "") < ($1["material_id"] as? String ?? "") }.map { sample -> [String] in
            let maps = sample["map_metadata"] as! [String: [String: Any]], target = maps["height"]!["source"] as! [String: Any]
            let variants = (sample["input_variants"] as? [[String: Any]] ?? []) + [maps["input"]!]
            let hashes = Array(Set(variants.compactMap { ($0["source"] as? [String: Any])?["file_sha256"] as? String })).sorted()
            return [sample["material_id"] as! String, sample["source_family_id"] as? String ?? sample["asset_family_id"] as? String ?? sample["material_id"] as! String,
                target["path"] as! String, target["file_sha256"] as! String] + hashes
        }
        let legacy = NativeMaterialTrainer.checksum(try JSONSerialization.data(withJSONObject: identities, options: [.sortedKeys]))
        let registryPath = workspace.appendingPathComponent("specialists.json")
        var saved = try object(registryPath), specialists = saved["specialists"] as! [[String: Any]]
        let offset = specialists.firstIndex { $0["id"] as? String == wood.id }!
        var models = specialists[offset]["models"] as! [[String: Any]]
        models[0]["membership_sha256"] = legacy; models[0].removeValue(forKey: "membership_schema")
        specialists[offset]["models"] = models; saved["specialists"] = specialists
        try JSONSerialization.data(withJSONObject: saved).write(to: registryPath)
        let originalTarget = try XCTUnwrap(originals.first { $0.0.lastPathComponent == "height.png" })
        try png(size: 256, value: 102, bits: 16, channels: 1).write(to: originalTarget.0)
        let changedLegacy = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Wood")
        XCTAssertEqual(try specialist("Wood", changedLegacy).targets.first { $0.target == "height" }?.status, "needs_update")
        XCTAssertNil(try specialist("Wood", changedLegacy).models.first?.membershipSchema,
                     "Historical proof cannot approve changed current pixels")
        try originalTarget.1.write(to: originalTarget.0)
        let oldFolder = originals[0].0.deletingLastPathComponent(), renamedFolder = oldFolder.deletingLastPathComponent().appendingPathComponent("Renamed Surface")
        try FileManager.default.moveItem(at: oldFolder, to: renamedFolder)
        let renamedInput = renamedFolder.appendingPathComponent("Arbitrary Photo Name.png")
        try FileManager.default.moveItem(at: renamedFolder.appendingPathComponent("input.png"), to: renamedInput)
        let live = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Wood")
        let current = try specialist("Wood", live)
        XCTAssertEqual(current.targets.first { $0.target == "height" }?.materialCount, 1)
        XCTAssertEqual(current.targets.first { $0.target == "height" }?.status, "approved")
        XCTAssertEqual(current.models.first?.membershipSchema, "texture-studio-material-content-proof-v1")
        XCTAssertEqual(current.models.first?.membershipSha256, model.membershipSha256)
        let prediction = NativeDiffuseTextureClassifier.Prediction(label: "Wood", confidence: 0.94,
            probabilities: ["Wood": 0.94], modelSHA256: classifier.sha256)
        let routed = try NativeMaterialSpecialistRouter.resolve(prediction: prediction, registry: live,
            workspaceURL: workspace, modelDirectory: root.appendingPathComponent("UnusedBase").path)
        XCTAssertEqual(routed.checkpoints["height"]?.checkpointPath, checkpoint.path)
        let refreshed = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(try specialist("Wood", refreshed).models.first?.membershipSchema, "texture-studio-material-content-proof-v1")
        for (file, bytes) in originals {
            let currentFile = file.lastPathComponent == "input.png" ? renamedInput : renamedFolder.appendingPathComponent(file.lastPathComponent)
            XCTAssertEqual(try Data(contentsOf: currentFile), bytes)
        }
        try png(size: 256, value: 101, bits: 16, channels: 1).write(to: renamedFolder.appendingPathComponent("height.png"))
        let edited = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Wood")
        XCTAssertEqual(try specialist("Wood", edited).targets.first { $0.target == "height" }?.status, "needs_update")
    }

    func testRealCompactSaveRegistersApprovesAndRoutesOnlyTheIntendedSpecialist() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        let originals = try await add("wood_planks", size: 256, value: 83, to: dataset, under: root)
        let paperOriginals = try await add("paper", size: 256, value: 48, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists")
        let classifier = fake(labels: ["Wood", "Paper"], choose: { $0.path.contains("/paper/") ? "Paper" : "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", initial)
        _ = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: URL(fileURLWithPath: wood.datasetPath), size: 256, target: "height", automaticValidation: false)
        let verified = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace)
        let membership = try XCTUnwrap(specialist("Wood", verified).targets.first { $0.target == "height" }?.membershipSha256)
        let output = root.appendingPathComponent("Trained")
        let text = try await NativeMaterialTrainer.run(arguments: ["train", "--dataset", wood.datasetPath,
            "--output", output.path, "--model-family", "compact-scalar", "--target", "height", "--size", "256",
            "--compact-width", "16", "--updates-per-map", "1", "--randomize-crops", "never", "--seed", "19"])
        let trained = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        let updates = try XCTUnwrap(trained["completed_updates"] as? Int)
        XCTAssertEqual(updates, 1); XCTAssertEqual(trained["training_performed"] as? Bool, true)
        let checkpoint = URL(fileURLWithPath: try XCTUnwrap(trained["checkpoint_path"] as? String))
        let recorded = try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 256,
            completedUpdates: updates, expectedMembershipSHA256: membership)
        let saved = try XCTUnwrap(specialist("Wood", recorded).models.first)
        XCTAssertFalse(saved.approved)
        let prediction = NativeDiffuseTextureClassifier.Prediction(label: "Wood", confidence: 0.94,
            probabilities: ["Wood": 0.94, "Paper": 0.06], modelSHA256: classifier.sha256)
        let awaiting = try NativeMaterialSpecialistRouter.resolve(prediction: prediction, registry: recorded,
            workspaceURL: workspace, modelDirectory: root.appendingPathComponent("UnusedBase").path)
        XCTAssertTrue(awaiting.checkpoints.isEmpty)
        _ = try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: saved.id)
        let unrelatedTarget = try XCTUnwrap(paperOriginals.first { $0.0.lastPathComponent == "height.png" }?.0)
        try png(size: 256, value: 118, bits: 16, channels: 1).write(to: unrelatedTarget)
        let live = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Wood")
        let untrainedPaper = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Paper")
        XCTAssertTrue(try specialist("Paper", untrainedPaper).models.isEmpty,
                      "An untrained category can be inspected after source changes without borrowing the Wood model")
        let route = try NativeMaterialSpecialistRouter.resolve(prediction: prediction, registry: live,
            workspaceURL: workspace, modelDirectory: root.appendingPathComponent("UnusedBase").path)
        XCTAssertEqual(route.checkpoints["height"]?.checkpointPath, checkpoint.path)
        XCTAssertEqual(route.checkpoints["height"]?.sha256, saved.checkpointSha256)
        XCTAssertNil(route.checkpoints["normal"]); XCTAssertNil(route.checkpoints["roughness"])
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
        // Edited target pixels keep the material in its tagged category while
        // marking the saved model's training membership out of date.
        let target = try XCTUnwrap(originals.first { $0.0.lastPathComponent == "height.png" }?.0)
        try png(size: 256, value: 91, bits: 16, channels: 1).write(to: target)
        let changed = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Wood")
        let changedWood = try specialist("Wood", changed)
        XCTAssertEqual(changedWood.targets.first { $0.target == "height" }?.materialCount, 1)
        XCTAssertEqual(changedWood.targets.first { $0.target == "height" }?.status, "needs_update")
        let changedRoute = try NativeMaterialSpecialistRouter.resolve(prediction: prediction, registry: changed,
            workspaceURL: workspace, modelDirectory: root.appendingPathComponent("UnusedBase").path)
        XCTAssertTrue(changedRoute.checkpoints.isEmpty)
    }
    func testDynamicFamilyClassificationCreatesOriginalReferenceDatasetsWithoutScaling() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        let originals = try await add("wood_planks", size: 32, value: 51, to: dataset, under: root)
            + add("wood_planks", size: 64, value: 73, to: dataset, under: root)
        let probe = PredictionProbe()
        let classifier = fake(labels: ["Wood", "Paper", "Leather"], probe: probe) { _ in "Wood" }
        let workspace = root.appendingPathComponent("Specialists")
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(registry.materialCount, 2); XCTAssertEqual(registry.familyCount, 1)
        XCTAssertEqual(probe.paths.count, 1, "Resolution tiers share one class prediction to avoid family leakage")
        XCTAssertEqual(try NativePNG.inspect(URL(fileURLWithPath: XCTUnwrap(probe.paths.first))).width, 32)
        let wood = try specialist("Wood", registry)
        XCTAssertEqual(wood.materialIds.count, 2); XCTAssertEqual(wood.assignments.count, 1)
        for target in ["height", "roughness", "normal"] {
            XCTAssertEqual(wood.targets.first { $0.target == target }?.materialCount, 2)
            XCTAssertEqual(wood.targets.first { $0.target == target }?.status, "ready")
            let records = try XCTUnwrap(NativeMaterialDatasetService.folderProjectionRecords(URL(fileURLWithPath: wood.datasetPath)))
            XCTAssertEqual(records.count, 2)
            for record in records {
                let maps = try XCTUnwrap(record.sample["maps"] as? [String: String])
                XCTAssertTrue(try XCTUnwrap(maps["input"]).hasPrefix(dataset.path + "/originals/"))
                XCTAssertTrue(try XCTUnwrap(maps[target]).hasPrefix(dataset.path + "/originals/"))
            }
        }
        XCTAssertEqual(try specialist("Paper", registry).targets.first?.materialCount, 0)
        XCTAssertEqual(try specialist("Paper", registry).targets.first?.status, "no_targets")
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: wood.datasetPath), includingPropertiesForKeys: nil)
            .allSatisfy { $0.pathExtension.lowercased() != "png" })
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }
        let cached = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1); XCTAssertEqual(try specialist("Wood", cached).materialIds, wood.materialIds)
        XCTAssertEqual(try NativeMaterialSpecialistService.load(workspaceURL: workspace).classifierLabels, ["Wood", "Paper", "Leather"])
    }

    func testCuratedClassAndClassifierOnlyPhotosRemainSeparateFromDepthTargets() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root), originals = try await add("cardboard", size: 32, value: 45, to: dataset, under: root)
        let curated = root.appendingPathComponent("Curated")
        for label in ["Paper", "Leather", "Other Diffuse Textures"] {
            try FileManager.default.createDirectory(at: curated.appendingPathComponent(label), withIntermediateDirectories: true)
        }
        let copied = curated.appendingPathComponent("Paper/cardboard_color.png")
        try XCTUnwrap(originals.first { $0.0.lastPathComponent == "input.png" }?.1).write(to: copied)
        let userFile = curated.appendingPathComponent("Leather/my-leather.png"), userBytes = try png(size: 29, value: 201)
        try userBytes.write(to: userFile)
        let workspace = root.appendingPathComponent("Specialists")
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, classifierDatasetURL: curated,
            workspaceURL: workspace, classifier: fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" }))
        XCTAssertEqual(registry.classifierOnlyCount, 1); XCTAssertEqual(registry.classifierOnlyAssets.first?.classLabel, "Leather")
        XCTAssertEqual(registry.classifierOnlyAssets.first?.status, "classifier_only_no_targets")
        XCTAssertEqual(registry.classifierOnlyAssets.first?.width, 29)
        let paper = try specialist("Paper", registry), leather = try specialist("Leather", registry)
        XCTAssertEqual(paper.materialIds.count, 1); XCTAssertEqual(paper.assignments.first?.source, "curated_classifier_dataset")
        XCTAssertEqual(paper.assignments.first?.predictedLabel, "Wood")
        XCTAssertTrue(leather.materialIds.isEmpty); XCTAssertTrue(leather.targets.allSatisfy { $0.materialCount == 0 })
        let another = curated.appendingPathComponent("Paper/user-paper.png"), anotherBytes = try png(size: 27, value: 169)
        try anotherBytes.write(to: another)
        let refreshed = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" }))
        XCTAssertEqual(refreshed.classifierOnlyCount, 2); XCTAssertEqual(refreshed.materialCount, 1)
        XCTAssertEqual(try Data(contentsOf: userFile), userBytes); XCTAssertEqual(try Data(contentsOf: another), anotherBytes)
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }
    }

    func testSourceRemovalAndReviewTeachingDetermineLiveSpecialistTrainingMembership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        let planks = try await add("wood_planks", size: 256, value: 50, to: dataset, under: root)
        let boards = try await add("wood_boards", size: 256, value: 60, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists"), classifier = fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" })
        let first = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let category = URL(fileURLWithPath: try specialist("Wood", first).datasetPath)
        let input = try XCTUnwrap(planks.first { $0.0.lastPathComponent == "input.png" }?.0)
        try NativeSourceFileTags.update(input, reviewStatus: "excluded")
        let removed = dataset.appendingPathComponent("originals/wood_boards")
        try FileManager.default.moveItem(at: removed, to: root.appendingPathComponent("Removed"))
        let excluded = try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: category, size: 256)
        XCTAssertEqual(excluded["source_set_count"] as? Int, 1, "The existing excluded family remains inspectable")
        XCTAssertThrowsError(try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: category, size: 256, target: "height", automaticValidation: false))
        try NativeSourceFileTags.update(input, reviewStatus: "approved")
        let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: category, size: 256, target: "height", automaticValidation: false)
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").count, 1)
        let updated = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(try specialist("Wood", updated).materialIds.count, 1)
        for (url, bytes) in planks { XCTAssertEqual(try Data(contentsOf: url), bytes) }
        for (url, bytes) in boards { XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("Removed/256/" + url.lastPathComponent)), bytes) }
    }

    func testCustomSubtypeAndAdditionalSourceInventoryUseOriginalTargetsWithoutMutatingMainMembership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 43, to: dataset, under: root)
        _ = try await add("wood_boards", size: 32, value: 66, to: dataset, under: root)
        let additional = root.appendingPathComponent("Additional/Paper003")
        try FileManager.default.createDirectory(at: additional, withIntermediateDirectories: true)
        let paperBytes = try png(size: 31, value: 191), heightBytes = try png(size: 31, value: 99, bits: 16, channels: 1)
        try paperBytes.write(to: additional.appendingPathComponent("input.png")); try heightBytes.write(to: additional.appendingPathComponent("height.png"))
        let workspace = root.appendingPathComponent("Specialists")
        _ = try await NativeMaterialDatasetService.run(arguments: ["add-source", "--dataset", dataset.path, "--source", additional.path])
        let first = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: workspace, classifier: fake(labels: ["Wood", "Paper"], choose: { $0.path.contains("Paper003") ? "Paper" : "Wood" }))
        XCTAssertEqual(first.materialCount, 3); XCTAssertEqual(try specialist("Paper", first).materialIds.count, 1)
        let main = try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: dataset, size: 256)
        XCTAssertEqual(main["source_set_count"] as? Int, 3, "Only the physically copied originals enter this folder")
        let result = try await NativeMaterialSpecialistService.addSpecialist(workspaceURL: workspace,
            name: "Wood Planks", classLabel: "Wood", subtype: "planks")
        let custom = try XCTUnwrap(result.specialists.first { $0.isCustom })
        XCTAssertEqual(custom.materialIds, ["wood_planks_32x32"]); XCTAssertEqual(custom.subtype, "planks")
        XCTAssertEqual(custom.targets.first { $0.target == "height" }?.materialCount, 1)
        XCTAssertEqual(try Data(contentsOf: additional.appendingPathComponent("input.png")), paperBytes)
        XCTAssertEqual(try Data(contentsOf: additional.appendingPathComponent("height.png")), heightBytes)
    }

    func testSavedModelsRequireTrainedSpecialistProvenanceReviewAndCurrentMembership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 256, value: 54, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists"), classifier = fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", initial), native = URL(fileURLWithPath: wood.datasetPath)
        _ = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: native, size: 256, target: "height", automaticValidation: false)
        let verified = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace)
        let membership = try XCTUnwrap(specialist("Wood", verified).targets.first { $0.target == "height" }?.membershipSha256)
        let checkpoint = try savedCheckpoint(root, dataset: native, step: 2)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 64, completedUpdates: 0))
        let foreign = try savedCheckpoint(root.appendingPathComponent("foreign"), dataset: dataset, step: 2)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: foreign, trainingSize: 64, completedUpdates: 2))
        let recorded = try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 64,
            completedUpdates: 2, expectedMembershipSHA256: membership)
        let model = try XCTUnwrap(specialist("Wood", recorded).models.first)
        XCTAssertFalse(model.approved); XCTAssertEqual(try specialist("Wood", recorded).targets.first?.status, "awaiting_review")
        let approved = try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id)
        XCTAssertTrue(try XCTUnwrap(specialist("Wood", approved).models.first?.approved))
        XCTAssertEqual(try specialist("Wood", approved).targets.first?.status, "approved")
        _ = try await add("manual_wood", size: 256, value: 89, to: dataset, under: root)
        try NativeSourceFileTags.update(dataset.appendingPathComponent("originals/manual_wood/256/input.png"), classLabel: "Wood", classifierIdentity: "user")
        let live = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace)
        XCTAssertEqual(try specialist("Wood", live).targets.first?.status, "needs_update")
        XCTAssertNotEqual(try specialist("Wood", live).targets.first?.membershipSha256, model.membershipSha256)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id))
        let refreshed = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(try specialist("Wood", refreshed).models.first?.checkpointSha256, model.checkpointSha256)
        XCTAssertEqual(try specialist("Wood", refreshed).targets.first?.status, "needs_update")
        try Data("changed checkpoint".utf8).write(to: checkpoint)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id))
        let revoked = try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id, approved: false)
        XCTAssertFalse(try XCTUnwrap(specialist("Wood", revoked).models.first?.approved))
    }

    func testUnownedWorkspaceAndSpecialistFolderArePreserved() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 44, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists")
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let userFile = workspace.appendingPathComponent("my-record.txt"), bytes = Data("Keep this".utf8)
        try bytes.write(to: userFile)
        do {
            _ = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
                classifier: fake(labels: ["Wood"], choose: { _ in "Wood" }))
            XCTFail("Unowned workspace must be rejected")
        } catch { XCTAssertEqual(try Data(contentsOf: userFile), bytes) }
    }

    func testCategoryReadsFollowPhysicalMembershipAndCurrentTagsWithoutRegistryRefresh() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try await dataset(root)
        let first = try await add("first", size: 256, value: 82, to: source, under: root)
        let input = try XCTUnwrap(first.first { $0.0.lastPathComponent == "input.png" }?.0)
        try NativeSourceFileTags.update(input, classLabel: "Bricks", classifierIdentity: "user")
        let workspace = root.appendingPathComponent("Specialists")
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace,
            classifier: fake(labels: ["Bricks", "Wood"], choose: { _ in "Wood" }))
        let category = URL(fileURLWithPath: try specialist("Bricks", registry).datasetPath)
        func read() async throws -> [String: Any] {
            let text = try await NativeMaterialDatasetService.run(arguments: ["dataset", "--dataset", category.path,
                "--review-size", "256", "--skip-source-recovery"])
            return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(text).utf8)) as? [String: Any])
        }
        let initial = try await read()
        XCTAssertEqual(initial["is_source_folder"] as? Bool, false)
        XCTAssertEqual(initial["source_folder_path"] as? String, source.path)
        XCTAssertEqual(initial["source_set_count"] as? Int, 1)
        let second = try await add("second", size: 256, value: 43, to: source, under: root)
        let nextInput = try XCTUnwrap(second.first { $0.0.lastPathComponent == "input.png" }?.0)
        try NativeSourceFileTags.update(nextInput, classLabel: "Bricks", classifierIdentity: "user")
        let added = try await read()
        XCTAssertEqual(added["source_set_count"] as? Int, 2, "A new tagged folder appears immediately without a registry refresh")
        try NativeSourceFileTags.update(input, classLabel: "Wood", classifierIdentity: "user")
        let retagged = try await read()
        XCTAssertEqual(retagged["source_set_count"] as? Int, 1, "Retagging immediately changes category membership")
        try FileManager.default.moveItem(at: nextInput.deletingLastPathComponent(), to: root.appendingPathComponent("Moved Out"))
        let removed = try await read()
        XCTAssertEqual(removed["source_set_count"] as? Int, 0, "A saved category cannot keep a removed last material")
        var ghost = try object(category.appendingPathComponent("dataset.json"))
        ghost["samples"] = [["material_id": "ghost", "sample_id": "ghost_full", "path": "samples/missing"]]
        try JSONSerialization.data(withJSONObject: ghost).write(to: category.appendingPathComponent("dataset.json"), options: .atomic)
        let stale = try await read()
        XCTAssertEqual(stale["source_set_count"] as? Int, 0, "Malformed saved member lists cannot create phantom materials")
    }

    func testWorkspaceMigrationRejectsPartialContainmentAndConflictingRegistryWithoutChangingOwnership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let legacy = try await dataset(root)
        _ = try await add("wood", size: 32, value: 48, to: legacy, under: root)
        _ = try await add("paper", size: 32, value: 62, to: legacy, under: root)
        let workspace = root.appendingPathComponent("Loom Room"), classifier = fake(labels: ["Wood"], choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: legacy, workspaceURL: workspace, classifier: classifier)
        let markerURL = workspace.appendingPathComponent(".specialist-workspace.json")
        let registryURL = workspace.appendingPathComponent(NativeMaterialSpecialistService.registryFilename)
        let markerBytes = try Data(contentsOf: markerURL), registryBytes = try Data(contentsOf: registryURL)
        let onlyOneFamily = legacy.appendingPathComponent("originals/wood")
        XCTAssertFalse(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: onlyOneFamily))
        do {
            _ = try await NativeMaterialSpecialistService.refresh(datasetURL: onlyOneFamily, workspaceURL: workspace, classifier: classifier)
            XCTFail("One overlapping family cannot claim a workspace for another corpus")
        } catch { XCTAssertTrue(error.localizedDescription.contains("another source")) }
        XCTAssertEqual(try Data(contentsOf: markerURL), markerBytes); XCTAssertEqual(try Data(contentsOf: registryURL), registryBytes)
        let allOriginals = legacy
        XCTAssertTrue(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: allOriginals))
        let owned = try specialist("Wood", initial)
        let bindingURL = URL(fileURLWithPath: owned.datasetPath).appendingPathComponent(".specialist-binding.json")
        let bindingBytes = try Data(contentsOf: bindingURL)
        var colliding = try object(bindingURL); colliding["workspace_path"] = root.appendingPathComponent("Unrelated Workspace").path
        try JSONSerialization.data(withJSONObject: colliding).write(to: bindingURL, options: .atomic)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: allOriginals))
        do {
            _ = try await NativeMaterialSpecialistService.refresh(datasetURL: allOriginals, workspaceURL: workspace, classifier: classifier)
            XCTFail("A colliding category folder must retain its existing binding")
        } catch { XCTAssertFalse(error.localizedDescription.isEmpty) }
        XCTAssertEqual(try Data(contentsOf: markerURL), markerBytes); XCTAssertEqual(try Data(contentsOf: registryURL), registryBytes)
        try bindingBytes.write(to: bindingURL, options: .atomic)
        var mismatched = try object(registryURL)
        mismatched["source_dataset_path"] = root.appendingPathComponent("Another Legacy Dataset").path
        try JSONSerialization.data(withJSONObject: mismatched, options: [.sortedKeys]).write(to: registryURL, options: .atomic)
        let conflictingBytes = try Data(contentsOf: registryURL)
        XCTAssertFalse(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: allOriginals))
        do {
            _ = try await NativeMaterialSpecialistService.refresh(datasetURL: allOriginals, workspaceURL: workspace, classifier: classifier)
            XCTFail("Conflicting ownership cannot be repaired by claiming it")
        } catch { XCTAssertTrue(error.localizedDescription.contains("another source")) }
        XCTAssertEqual(try Data(contentsOf: markerURL), markerBytes)
        XCTAssertEqual(try Data(contentsOf: registryURL), conflictingBytes)
        let partial = root.appendingPathComponent("Partial Workspace")
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
        var partialMarker = try object(markerURL); partialMarker["workspace_path"] = partial.path
        try JSONSerialization.data(withJSONObject: partialMarker).write(to: partial.appendingPathComponent(".specialist-workspace.json"))
        XCTAssertTrue(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: partial, datasetURL: allOriginals), "An owned interrupted cache refresh can rebuild from the same current physical source folder")
        let absent = root.appendingPathComponent("New Workspace")
        XCTAssertTrue(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: absent, datasetURL: allOriginals))
        XCTAssertFalse(FileManager.default.fileExists(atPath: absent.path), "Eligibility must remain read-only")
    }

    func testCategoryRenameUsesNewPhysicalPathsAndKeepsSourceTeachingTrainable() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try await dataset(root)
        let originals = try await add("wood", size: 256, value: 83, to: source, under: root)
        let input = try XCTUnwrap(originals.first { $0.0.lastPathComponent == "input.png" }?.0)
        try NativeSourceFileTags.update(input, classLabel: "Wood", classifierIdentity: "user", reviewStatus: "approved")
        let workspace = root.appendingPathComponent("Specialists")
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace,
            classifier: fake(labels: ["Wood"], choose: { _ in "Wood" }))
        let category = URL(fileURLWithPath: try specialist("Wood", registry).datasetPath)
        let oldFolder = input.deletingLastPathComponent(), newFolder = oldFolder.deletingLastPathComponent().appendingPathComponent("Renamed")
        try FileManager.default.moveItem(at: oldFolder, to: newFolder)
        let text = try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", category.path,
            "--size", "256", "--target", "height", "--skip-source-recovery"])
        let info = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(text).utf8)) as? [String: Any])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(info["dataset_path"] as? String))
        defer { _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertFalse(samples.isEmpty)
        XCTAssertTrue(samples.allSatisfy { $0.augmentationSourceInputURL?.path.hasPrefix(newFolder.path + "/") == true })
        XCTAssertEqual(NativeSourceFileTags.read(newFolder.appendingPathComponent("input.png"))?.reviewStatus, "approved")
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: newFolder.appendingPathComponent(path.lastPathComponent)), bytes) }
    }

    func testCurrentDirectoriesReplaceStaleManifestTaxonomyAndRetireOnlyEmptyAutomaticRecords() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 34, to: dataset, under: root)
        let curated = root.appendingPathComponent("Curated"), workspace = root.appendingPathComponent("Specialists")
        try FileManager.default.createDirectory(at: curated.appendingPathComponent("Paper"), withIntermediateDirectories: true)
        let obsolete = NativeDiffuseClassifierDatasetService.TextureClass(id: "obsolete", name: "Obsolete Finish", description: "Old class")
        let manifest = NativeDiffuseClassifierDatasetService.Manifest(schema: NativeDiffuseClassifierDatasetService.schema,
            updatedAt: "fixture", datasetPath: curated.path, selectionPolicy: "fixture", classes: [obsolete], entries: [])
        try JSONEncoder().encode(manifest).write(to: curated.appendingPathComponent(".classifier-manifest.json"))
        let old = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, classifierDatasetURL: curated,
            workspaceURL: workspace, classifier: fake(labels: ["Wood", "Obsolete Finish"], choose: { _ in "Wood" }))
        XCTAssertEqual(old.specialists.count, 3)
        let obsoleteRecord = try specialist("Obsolete Finish", old)
        let next = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood"], choose: { _ in "Wood" }))
        XCTAssertEqual(Set(next.specialists.map(\.classLabel)), ["Wood", "Paper"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: obsoleteRecord.datasetPath), "Retiring an empty registry entry never deletes its folder")
        // User-added paired data prevents retirement even though no model or
        // current classifier label mentions this old class.
        let returned = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood", "Obsolete Finish"], choose: { _ in "Wood" }))
        _ = try specialist("Obsolete Finish", returned)
        _ = try await add("manual_finish", size: 32, value: 94, to: dataset, under: root)
        try NativeSourceFileTags.update(dataset.appendingPathComponent("originals/manual_finish/32/input.png"), classLabel: "Obsolete Finish", classifierIdentity: "user")
        let preserved = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood"], choose: { _ in "Wood" }))
        XCTAssertEqual(try specialist("Obsolete Finish", preserved).materialIds, ["manual_finish_32x32"])
    }

    private func fake(labels: [String], hash: String = String(repeating: "a", count: 64), probe: PredictionProbe? = nil, minimumImageSize: Int = 0,
                      choose: @escaping @Sendable (URL) -> String) -> NativeMaterialSpecialistService.Classifier {
        .init(path: "/fixture.mlmodel", sha256: hash, labels: labels, minimumImageSize: minimumImageSize) { url in
            probe?.append(url.path); let label = choose(url)
            return .init(label: label, confidence: 0.94, probabilities: Dictionary(uniqueKeysWithValues: labels.map { ($0, $0 == label ? 0.94 : 0.01) }), modelSHA256: hash)
        }
    }
    private func specialist(_ label: String, _ registry: NativeMaterialSpecialistService.Registry) throws -> NativeMaterialSpecialistService.Specialist {
        try XCTUnwrap(registry.specialists.first { !$0.isCustom && $0.classLabel == label })
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("specialist-tests-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.resolvingSymlinksInPath().standardizedFileURL
    }
    private func dataset(_ root: URL) async throws -> URL {
        let url = root.appendingPathComponent("Main")
        _ = try await NativeMaterialDatasetService.run(arguments: ["create-dataset", "--dataset", url.path,
            "--name", "Specialist fixture", "--source-folder", "true"])
        return url
    }
    private func add(_ family: String, size: Int, value: UInt8, to dataset: URL, under root: URL) async throws -> [(URL, Data)] {
        let folder = dataset.appendingPathComponent("originals/" + family + "/" + String(size))
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var arguments = ["add-material", "--dataset", dataset.path, "--name", family], originals: [(URL, Data)] = []
        for (role, bits, channels) in [("input", 8, 3), ("height", 16, 1), ("roughness", 8, 1), ("normal", 8, 3)] {
            let file = folder.appendingPathComponent(role + ".png"), bytes = try png(size: size, value: value, bits: bits, channels: channels)
            try bytes.write(to: file); originals.append((file, bytes)); arguments += ["--" + role, file.path]
        }
        _ = try await NativeMaterialDatasetService.run(arguments: arguments)
        return originals
    }
    private func png(size: Int, value: UInt8, bits: Int = 8, channels: Int = 3) throws -> Data {
        try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels, color: channels == 1 ? 0 : 2, interlace: 0),
            pixels: Data(repeating: value, count: size * size * channels * bits / 8), colorChunks: []).encoded()
    }
    private func savedCheckpoint(_ root: URL, dataset: URL, step: Int) throws -> URL {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 13)
        var configuration = try model.checkpointConfiguration(size: 64, step: step)
        configuration["training_configuration"] = ["training_cache_root": dataset.appendingPathComponent(".training-data").path,
            "initial_step": 0, "dataset_manifest_sha256": String(repeating: "f", count: 64)]
        let file = root.appendingPathComponent("trained.safetensors")
        try NativeSafetensors.write(tensors: model.weights, metadata: ["configuration": NativeMaterialTransfer.json(configuration)], to: file)
        return file
    }
    private func object(_ url: URL) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]) }
}

private final class PredictionProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [String] = []
    var paths: [String] { lock.lock(); defer { lock.unlock() }; return stored }
    func append(_ path: String) { lock.lock(); stored.append(path); lock.unlock() }
}
