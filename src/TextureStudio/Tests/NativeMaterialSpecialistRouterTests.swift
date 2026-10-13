import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialSpecialistRouterTests: XCTestCase {
    func testApprovedCurrentFullModelsRouteIndependentlyWithoutChangingCheckpointBytes() throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        var models: [NativeMaterialSpecialistService.ModelRecord] = []
        for target in ["height", "normal", "roughness"] { models.append(try checkpoint(root, target: target)) }
        let original = try models.map { try Data(contentsOf: URL(fileURLWithPath: $0.checkpointPath)) }
        let specialist = makeSpecialist(root, models: models)
        let registry = makeRegistry(root, specialists: [specialist])
        let result = try resolve(registry, root: root)
        XCTAssertEqual(Set(result.checkpoints.keys), ["height", "normal", "roughness"])
        XCTAssertEqual(result.classLabel, "Bricks")
        XCTAssertEqual(result.probabilities["Bricks"], 0.98)
        for (index, record) in models.enumerated() {
            let selected = try XCTUnwrap(result.checkpoints[record.target])
            XCTAssertEqual(selected.sha256, record.checkpointSha256)
            XCTAssertEqual(selected.checkpointPath, record.checkpointPath)
            XCTAssertEqual(selected.workspacePath, root.path)
            XCTAssertEqual(selected.modelDirectory, root.appendingPathComponent("models").path)
            XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: record.checkpointPath)), original[index])
        }
    }

    func testUnreviewedUntrainedStaleAndParentClassCustomModelsUseFallback() throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let approved = try checkpoint(root, target: "height")
        var unreviewed = approved; unreviewed.approved = false; unreviewed.approvedAt = nil
        let untrained = record(approved, updates: 0)
        let stale = record(approved, membership: "previous-membership")
        for candidate in [unreviewed, untrained, stale] {
            let registry = makeRegistry(root, specialists: [makeSpecialist(root, models: [candidate])])
            XCTAssertTrue(try resolve(registry, root: root).checkpoints.isEmpty)
        }
        let custom = makeSpecialist(root, models: [approved], custom: true, subtype: "Weathered")
        XCTAssertTrue(try resolve(makeRegistry(root, specialists: [custom]), root: root).checkpoints.isEmpty,
            "A broad Bricks prediction must not select a manually created Weathered Bricks subset")
        let zeroMaterials = makeSpecialist(root, models: [approved], materialCount: 0)
        XCTAssertTrue(try resolve(makeRegistry(root, specialists: [zeroMaterials]), root: root).checkpoints.isEmpty)
    }

    func testChangedCheckpointIsSkippedAndEarlierApprovedCurrentModelCanRoute() throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let good = try checkpoint(root, target: "height", name: "earlier", completedAt: "2026-10-11T00:00:00Z")
        let changed = try checkpoint(root, target: "height", name: "changed", completedAt: "2026-10-12T00:00:00Z")
        var bytes = try Data(contentsOf: URL(fileURLWithPath: changed.checkpointPath))
        bytes[bytes.count - 1] ^= 1
        try bytes.write(to: URL(fileURLWithPath: changed.checkpointPath))
        let registry = makeRegistry(root, specialists: [makeSpecialist(root, models: [good, changed])])
        let result = try resolve(registry, root: root)
        XCTAssertEqual(result.checkpoints["height"]?.sha256, good.checkpointSha256)
        XCTAssertTrue(result.notices.contains(where: { $0.contains("Skipped") }))
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: changed.checkpointPath)), bytes)
    }

    func testInvalidOrWrongTargetCheckpointCannotRouteEvenWhenRegistrySaysApproved() throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let normal = try checkpoint(root, target: "normal", name: "wrong-target")
        let wrongTarget = NativeMaterialSpecialistService.ModelRecord(id: normal.id, target: "height",
            modelFamily: "compact-scalar", checkpointPath: normal.checkpointPath, checkpointSha256: normal.checkpointSha256,
            trainingSize: normal.trainingSize, completedUpdates: normal.completedUpdates, completedAt: normal.completedAt,
            membershipSha256: membership("height"), approved: true, approvedAt: normal.approvedAt)
        let stepZero = try checkpoint(root, target: "height", name: "zero-step", step: 0)
        let missing = record(stepZero, path: root.appendingPathComponent("missing.safetensors").path)
        for candidate in [wrongTarget, stepZero, missing] {
            let result = try resolve(makeRegistry(root, specialists: [makeSpecialist(root, models: [candidate])]), root: root)
            XCTAssertTrue(result.checkpoints.isEmpty)
            XCTAssertTrue(result.notices.contains(where: { $0.contains("Skipped") }))
        }
    }

    func testThresholdAndClassifierIdentityPreventStaleAutomaticRouting() throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let approved = try checkpoint(root, target: "height")
        let registry = makeRegistry(root, specialists: [makeSpecialist(root, models: [approved])])
        XCTAssertTrue(try resolve(registry, root: root, confidence: 0.699).checkpoints.isEmpty)
        XCTAssertNotNil(try resolve(registry, root: root, confidence: 0.7).checkpoints["height"])
        var stale = registry; stale.classifierSha256 = String(repeating: "b", count: 64)
        XCTAssertTrue(try resolve(stale, root: root).checkpoints.isEmpty)
        stale = registry; stale.classifierLabels = ["Wood"]
        XCTAssertTrue(try resolve(stale, root: root).checkpoints.isEmpty)
        for threshold in [-0.1, 1.1, Double.nan] {
            XCTAssertThrowsError(try NativeMaterialSpecialistRouter.resolve(prediction: prediction(), registry: registry,
                workspaceURL: root, modelDirectory: "", minimumConfidence: threshold))
        }
        XCTAssertThrowsError(try resolve(registry, root: root, confidence: 1.1))
    }

    func testPreparedDiffuseUsesConfiguredClassifierAndPersistedApprovedRegistry() async throws {
        let source = URL(fileURLWithPath: "/opt/ipde/classifer_ds/Bricks/red_brick--color_1--3f6d0fdd452e.png")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: source.path), "Optional known corpus image is unavailable")
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let saved = try checkpoint(root, target: "height")
        let originalInput = try sourceCategoryDataset(root, diffuse: source)
        var registry = makeRegistry(root, specialists: [makeSpecialist(root, models: [saved])])
        let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase
        let registryURL = root.appendingPathComponent(NativeMaterialSpecialistService.registryFilename)
        try encoder.encode(registry).write(to: registryURL)
        let current = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: root)
        let target = try XCTUnwrap(current.specialists.first?.targets.first { $0.target == "height" })
        XCTAssertEqual(target.materialCount, 1, "The actual copied maps and their Bricks tags determine current membership")
        XCTAssertEqual(try NativeMaterialTransfer.hash(originalInput), try NativeMaterialTransfer.hash(source))
        let currentMembership = target.membershipSha256
        let approved = record(saved, membership: currentMembership)
        registry.specialists[0].models = [approved]
        try encoder.encode(registry).write(to: registryURL)
        let registryBytes = try Data(contentsOf: registryURL)
        let color = try XCTUnwrap(CIImage(contentsOf: source))
        let linear = try LinearColorFrame(color, context: CIContext()).color
        let custom = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Resources/DiffuseTextureTypeClassifier.mlmodel")
        let result = try await NativeMaterialSpecialistRouter.route(diffuse: linear, workspaceURL: root,
            modelDirectory: "", classifierModelURL: custom)
        XCTAssertEqual(result.classLabel, "Bricks")
        XCTAssertEqual(result.prediction.modelSHA256, NativeDiffuseTextureClassifier.bundledModelSHA256)
        XCTAssertEqual(result.checkpoints["height"]?.sha256, approved.checkpointSha256)
        XCTAssertNil(result.checkpoints["normal"])
        XCTAssertNil(result.checkpoints["roughness"])
        try NativeSourceFileTags.update(originalInput, reviewStatus: "excluded")
        let afterExclusion = try await NativeMaterialSpecialistRouter.route(diffuse: linear, workspaceURL: root,
            modelDirectory: "", classifierModelURL: custom)
        XCTAssertTrue(afterExclusion.checkpoints.isEmpty,
            "Current manual curation invalidates membership immediately, even before registry refresh")
        XCTAssertEqual(try Data(contentsOf: registryURL), registryBytes, "Routing reads current membership without modifying the registry")
    }

    private func resolve(_ registry: NativeMaterialSpecialistService.Registry, root: URL,
                         confidence: Double = 0.98) throws -> NativeMaterialSpecialistRouter.RouteResult {
        try NativeMaterialSpecialistRouter.resolve(prediction: prediction(confidence: confidence), registry: registry,
            workspaceURL: root, modelDirectory: root.appendingPathComponent("models").path)
    }
    private func prediction(confidence: Double = 0.98) -> NativeDiffuseTextureClassifier.Prediction {
        .init(label: "Bricks", confidence: confidence, probabilities: ["Bricks": confidence, "Wood": 1 - confidence],
            modelSHA256: NativeDiffuseTextureClassifier.bundledModelSHA256)
    }
    private func membership(_ target: String) -> String { "current-membership-" + target }
    private func makeRegistry(_ root: URL, specialists: [NativeMaterialSpecialistService.Specialist]) -> NativeMaterialSpecialistService.Registry {
        .init(schema: NativeMaterialSpecialistService.schema, workspacePath: root.path, sourceDatasetPath: root.appendingPathComponent("source").path,
            classifierDatasetPath: nil, classifierPath: "", classifierSha256: NativeDiffuseTextureClassifier.bundledModelSHA256,
            classifierLabels: ["Bricks", "Wood"], updatedAt: "2026-10-11T00:00:00Z", specialists: specialists,
            classifierOnlyAssets: [], classifierOnlyCount: 0, materialCount: 1, familyCount: 1, warnings: [])
    }
    private func makeSpecialist(_ root: URL, models: [NativeMaterialSpecialistService.ModelRecord],
                                custom: Bool = false, subtype: String? = nil, materialCount: Int = 1) -> NativeMaterialSpecialistService.Specialist {
        .init(id: "bricks", name: "Bricks", classLabel: "Bricks", subtype: subtype, isCustom: custom,
            datasetPath: root.appendingPathComponent("datasets/bricks").path, materialIds: ["brick-original"], familyIds: ["brick-family"],
            assignments: [], targets: ["height", "normal", "roughness"].map {
                .init(target: $0, materialCount: materialCount, status: "approved", membershipSha256: membership($0))
            }, models: models, automaticallyManagedMaterialIds: [], automaticallyRemovedMaterialIds: [])
    }
    private func checkpoint(_ root: URL, target: String, name: String? = nil, step: Int = 5,
                            completedAt: String = "2026-10-11T00:00:00Z") throws -> NativeMaterialSpecialistService.ModelRecord {
        let file = root.appendingPathComponent((name ?? target) + ".safetensors")
        let model = try NativeCompactMaterialModel(target: target, width: 16, seed: 29)
        let metadata = String(decoding: try JSONSerialization.data(withJSONObject: model.checkpointConfiguration(size: 256, step: step)), as: UTF8.self)
        try NativeSafetensors.write(tensors: model.weights, metadata: ["configuration": metadata], to: file)
        return .init(id: name ?? target, target: target, modelFamily: target == "normal" ? "compact-normal" : "compact-scalar",
            checkpointPath: file.path, checkpointSha256: try NativeMaterialTransfer.hash(file), trainingSize: 256,
            completedUpdates: 5, completedAt: completedAt, membershipSha256: membership(target), approved: true,
            approvedAt: "2026-10-11T01:00:00Z")
    }
    private func record(_ original: NativeMaterialSpecialistService.ModelRecord, updates: Int? = nil,
                        membership: String? = nil, path: String? = nil) -> NativeMaterialSpecialistService.ModelRecord {
        .init(id: original.id, target: original.target, modelFamily: original.modelFamily,
            checkpointPath: path ?? original.checkpointPath, checkpointSha256: original.checkpointSha256,
            trainingSize: original.trainingSize, completedUpdates: updates ?? original.completedUpdates,
            completedAt: original.completedAt, membershipSha256: membership ?? original.membershipSha256,
            approved: original.approved, approvedAt: original.approvedAt)
    }
    private func sourceCategoryDataset(_ root: URL, diffuse: URL) throws -> URL {
        let dataset = root.appendingPathComponent("datasets/bricks"), originals = root.appendingPathComponent("source/Brick")
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: originals, withIntermediateDirectories: true)
        let header = try NativePNG.inspect(diffuse), input = originals.appendingPathComponent("Diffuse.png")
        let height = originals.appendingPathComponent("Height.png")
        try FileManager.default.copyItem(at: diffuse, to: input)
        try NativePNG(header: .init(width: header.width, height: header.height, bits: 16, channels: 1, color: 0, interlace: 0),
            pixels: Data(repeating: 0, count: header.width * header.height * 2), colorChunks: []).encoded().write(to: height)
        try NativeSourceFileTags.update(input, role: "input", classLabel: "Bricks", classifierIdentity: "user", reviewStatus: "approved")
        try NativeSourceFileTags.update(height, role: "height", reviewStatus: "approved")
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "name": "Bricks", "storage_policy": "original-source-references-v1",
            "managed_source_folders": true, "samples": []]).write(to: dataset.appendingPathComponent("dataset.json"))
        try JSONSerialization.data(withJSONObject: ["schema": NativeMaterialSpecialistService.schema, "specialist_id": "bricks",
            "workspace_path": root.path, "source_folder_path": root.appendingPathComponent("source").path,
            "class_label": "Bricks"]).write(to: dataset.appendingPathComponent(".specialist-binding.json"))
        try NativeMaterialDatasetService.refreshFolderProjection(dataset)
        return input
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("specialist-router-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
