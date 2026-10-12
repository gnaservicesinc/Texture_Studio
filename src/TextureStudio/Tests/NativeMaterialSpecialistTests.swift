import CryptoKit
import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeMaterialSpecialistTests: XCTestCase {
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
        let membership = try XCTUnwrap(wood.targets.first { $0.target == "height" }?.membershipSha256)
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
        XCTAssertThrowsError(try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Paper"))
        let route = try NativeMaterialSpecialistRouter.resolve(prediction: prediction, registry: live,
            workspaceURL: workspace, modelDirectory: root.appendingPathComponent("UnusedBase").path)
        XCTAssertEqual(route.checkpoints["height"]?.checkpointPath, checkpoint.path)
        XCTAssertEqual(route.checkpoints["height"]?.sha256, saved.checkpointSha256)
        XCTAssertNil(route.checkpoints["normal"]); XCTAssertNil(route.checkpoints["roughness"])
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
        // A changed original target invalidates live routing even if the saved
        // registry membership has not yet refreshed.
        let target = try XCTUnwrap(originals.first { $0.0.lastPathComponent == "height.png" }?.0)
        try png(size: 256, value: 91, bits: 16, channels: 1).write(to: target)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Wood"))
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
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: URL(fileURLWithPath: wood.datasetPath), size: 32, target: target)
            XCTAssertEqual(samples.count, 1); XCTAssertEqual(samples.first?.width, 32)
            XCTAssertTrue(try XCTUnwrap(samples.first?.inputURL.path).hasPrefix(root.path + "/originals/"))
            XCTAssertTrue(try XCTUnwrap(samples.first?.targetURL.path).hasPrefix(root.path + "/originals/"))
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

    func testRefreshPreservesManualExclusionRemovalAndAddedPairedMaterials() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 50, to: dataset, under: root)
        _ = try await add("wood_boards", size: 32, value: 60, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists"), classifier = fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" })
        let first = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", first), specialistDataset = URL(fileURLWithPath: wood.datasetPath)
        let removedID = try XCTUnwrap(wood.materialIds.first { $0.hasPrefix("wood_boards") })
        _ = try await NativeMaterialDatasetService.run(arguments: ["remove-material", "--dataset", specialistDataset.path, "--material", removedID])
        let planksID = try XCTUnwrap(wood.materialIds.first { $0.hasPrefix("wood_planks") })
        _ = try await NativeMaterialDatasetService.run(arguments: ["curate", "--dataset", specialistDataset.path,
            "--sample", planksID + "_full", "--status", "excluded", "--note", "User reviewed this pair"])
        let extras = try await add("manual_wood", size: 32, value: 80, to: specialistDataset, under: root)
        let updated = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let newWood = try specialist("Wood", updated)
        XCTAssertFalse(newWood.materialIds.contains(removedID)); XCTAssertTrue(newWood.materialIds.contains("manual_wood_32x32"))
        XCTAssertEqual(newWood.targets.first { $0.target == "height" }?.materialCount, 1, "Excluded originals stay excluded after refresh")
        let sample = try object(specialistDataset.appendingPathComponent("samples/" + planksID + "_full/sample.json"))
        XCTAssertEqual(sample["status"] as? String, "excluded"); XCTAssertEqual(sample["curation_note"] as? String, "User reviewed this pair")
        for (url, bytes) in extras { XCTAssertEqual(try Data(contentsOf: url), bytes) }
        let moved = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood", "Paper"], hash: String(repeating: "b", count: 64), choose: { _ in "Paper" }))
        XCTAssertTrue(try specialist("Wood", moved).materialIds.contains(planksID), "Curated references survive classifier changes")
        XCTAssertEqual(try specialist("Paper", moved).materialIds.count, 2)
        XCTAssertFalse(moved.warnings.isEmpty)
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
        let first = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, sourceFolderURL: additional.deletingLastPathComponent(),
            workspaceURL: workspace, classifier: fake(labels: ["Wood", "Paper"], choose: { $0.path.contains("Paper003") ? "Paper" : "Wood" }))
        XCTAssertEqual(first.materialCount, 3); XCTAssertEqual(try specialist("Paper", first).materialIds.count, 1)
        let main = try object(dataset.appendingPathComponent("dataset.json"))
        XCTAssertEqual((main["samples"] as? [[String: Any]])?.count, 2, "Additional source pairs never enter the main database during classification")
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
        _ = try await add("wood_planks", size: 32, value: 54, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists"), classifier = fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", initial), native = URL(fileURLWithPath: wood.datasetPath)
        let membership = try XCTUnwrap(wood.targets.first { $0.target == "height" }?.membershipSha256)
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
        _ = try await add("manual_wood", size: 32, value: 89, to: native, under: root)
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
        let restored = try specialist("Obsolete Finish", returned)
        _ = try await add("manual_finish", size: 32, value: 94, to: URL(fileURLWithPath: restored.datasetPath), under: root)
        let preserved = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood"], choose: { _ in "Wood" }))
        XCTAssertEqual(try specialist("Obsolete Finish", preserved).materialIds, ["manual_finish_32x32"])
    }

    private func fake(labels: [String], hash: String = String(repeating: "a", count: 64), probe: PredictionProbe? = nil,
                      choose: @escaping @Sendable (URL) -> String) -> NativeMaterialSpecialistService.Classifier {
        .init(path: "/fixture.mlmodel", sha256: hash, labels: labels) { url in
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
            "--name", "Specialist fixture", "--training-size", "256", "--subject-validation", "disabled"])
        return url
    }
    private func add(_ family: String, size: Int, value: UInt8, to dataset: URL, under root: URL) async throws -> [(URL, Data)] {
        let folder = root.appendingPathComponent("originals/" + family + "/" + String(size))
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
