import Darwin
import Foundation
import XCTest
@testable import TextureStudio

/// Explicit opt-in maintenance for the user's verified bark originals. This
/// publishes source metadata and retires old outputs; it never starts training.
final class BarkFreshPreparationProductionTests: XCTestCase {
    func testPrepareAuthorizedFreshBarkDatasetWithoutOldWeightsOrCrops() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["TEXTURE_STUDIO_BARK_FRESH_SETUP"] == "1" else {
            throw XCTSkip("Opt-in user-authorized fresh bark preparation.")
        }
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Texture Studio")
        let evidence = support.appendingPathComponent("Validation/bark-fresh-start-2026-10-12")
        let inventoryURL = evidence.appendingPathComponent("source-inventory.json")
        guard try NativeMaterialTransfer.hash(inventoryURL) == "513227a8c8ac7974c1a83acd59d7cc0ca5a1bacad70e1e852d83b9d27f9b357a" else {
            throw StudioError("The verified bark inventory changed before preparation.")
        }
        let inventory = try object(inventoryURL)
        let folders = try XCTUnwrap(inventory["fresh_membership_source_folders"] as? [String])
        guard folders.count == 18, Set(folders).count == 18 else { throw StudioError("The authorized bark folder inventory differs.") }
        let originals = try XCTUnwrap(inventory["source_folders"] as? [[String: Any]])
            .flatMap { $0["maps"] as? [[String: Any]] ?? [] }
        guard originals.count == 221 else { throw StudioError("The authorized bark map inventory differs.") }
        for folder in folders {
            let url = URL(fileURLWithPath: folder)
            let state = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard state.isDirectory == true, state.isSymbolicLink != true,
                  url.resolvingSymlinksInPath().deletingLastPathComponent().path == "/opt/ipde/sources_mats" else {
                throw StudioError("An authorized original bark folder changed.")
            }
        }
        try verifyOriginals(originals, phase: "before")
        let workspace = support.appendingPathComponent("Material Workspace/Specialist Models - Wolf")
        let destination = workspace.appendingPathComponent("datasets/class-bark-3ebaec6ba84c")
        let archive = evidence.appendingPathComponent("Archived old training")
        guard !FileManager.default.fileExists(atPath: archive.path) else {
            throw StudioError("Fresh bark preparation already has a retained archive; do not repeat its mutations.")
        }
        var registry = try NativeMaterialSpecialistService.load(workspaceURL: workspace)
        let barkOffset = try XCTUnwrap(registry.specialists.firstIndex { $0.id == "class-bark-3ebaec6ba84c" })
        guard registry.specialists[barkOffset].datasetPath == destination.path else {
            throw StudioError("The Bark specialist dataset binding changed.")
        }
        let stage = workspace.appendingPathComponent("datasets/.fresh-bark-" + UUID().uuidString)
        func command(_ arguments: [String]) async throws -> [String: Any] {
            let result = try await NativeMaterialDatasetService.run(arguments: arguments)
            let output = try XCTUnwrap(result)
            return try NativeMaterialTransfer.object(Data(output.utf8))
        }
        _ = try await command(["create-dataset", "--dataset", stage.path, "--name", "Bark · fresh originals",
            "--description", "18 user-selected bark families; original native maps; fresh RGB height training.",
            "--training-size", "1024", "--validation-enabled", "yes", "--validation-percent", "10",
            "--validation-max-crops", "2", "--validation-quick-count", "2"])
        for (offset, folder) in folders.enumerated() {
            print("BARK_FRESH_IMPORT \(offset + 1)/\(folders.count) \(folder)")
            _ = try await command(["import-folder", "--dataset", stage.path, "--folder", folder,
                "--skip-source-recovery"])
        }
        let info = try await command(["dataset", "--dataset", stage.path, "--size", "1024", "--target", "height",
            "--skip-source-recovery"])
        XCTAssertEqual(info["source_set_count"] as? Int, 51)
        let materials = try XCTUnwrap(info["materials"] as? [[String: Any]])
        let samples = materials.flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        let families = Set(samples.compactMap { $0["source_family_id"] as? String })
        guard families.count == 18, info["source_set_count"] as? Int == 51 else {
            throw StudioError("Fresh bark import did not retain all 18 families and 51 original resolution sets.")
        }
        let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: stage,
            size: 1024, target: "height", cropSeed: 1729, automaticValidation: true,
            onEvent: { print("BARK_FRESH_PREPARE " + $0) })
        let pairs = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 1024, target: "height")
        let trainingCount = pairs.filter { $0.split == "train" }.count
        let validationCount = pairs.filter { $0.split == "validation" }.count
        guard trainingCount == 51, validationCount == 1 else {
            throw StudioError("The fresh crop membership differs from the verified plan.")
        }
        var cropReceipts: [[String: Any]] = []
        for (offset, pair) in pairs.enumerated() {
            print("BARK_FRESH_VERIFY_CROP \(offset + 1)/\(pairs.count) \(pair.id)")
            try autoreleasepool {
                let sourceInput = try XCTUnwrap(pair.augmentationSourceInputURL)
                let sourceHeight = try XCTUnwrap(pair.augmentationSourceTargetURL)
                let rectangle = try XCTUnwrap(pair.cropRectangle)
                let input = try NativePNG.decode(Data(contentsOf: pair.inputURL))
                let height = try NativePNG.decode(Data(contentsOf: pair.targetURL))
                let expectedInput = try NativePNG.crop(sourceInput, rectangle: rectangle)
                let expectedHeight = try NativePNG.crop(sourceHeight, rectangle: rectangle)
                guard input.pixels == expectedInput.pixels, height.pixels == expectedHeight.pixels,
                      height.header.bits == 16,
                      [input.header.width, input.header.height] == [1024, 1024],
                      [height.header.width, height.header.height] == [1024, 1024] else {
                    throw StudioError("A fresh training crop differs from its native source codes: \(pair.id)")
                }
                let values = try height.modelFloatSamples(role: "height")
                guard !values.isEmpty, values.allSatisfy(\.isFinite) else {
                    throw StudioError("A fresh height crop contains invalid model values: \(pair.id)")
                }
                cropReceipts.append(["sample_id": pair.id, "split": pair.split, "crop_rectangle": rectangle,
                    "source_input": sourceInput.path, "source_height": sourceHeight.path,
                    "input_sha256": pair.inputSHA256, "height_sha256": pair.targetSHA256,
                    "height_bits": height.header.bits, "source_codes_bit_exact": true,
                    "height_minimum": values.min()!, "height_maximum": values.max()!])
            }
        }
        _ = try NativeMaterialDatasetService.cleanup(prepared)
        let caches = stage.appendingPathComponent(".training-data")
        if FileManager.default.fileExists(atPath: caches.path), try FileManager.default.contentsOfDirectory(atPath: caches.path).isEmpty {
            try FileManager.default.removeItem(at: caches)
        }
        guard !FileManager.default.fileExists(atPath: caches.path) else {
            throw StudioError("The validation preparation left unexpected cache contents; preserve the stage for inspection.")
        }
        try verifyOriginals(originals, phase: "after")
        // The concrete replacement is verified before touching the old dataset.
        let descriptor = Darwin.open(destination.appendingPathComponent(".material-workbench.lock").path,
            O_RDWR | O_NOFOLLOW)
        guard descriptor >= 0 else { throw StudioError("The old Bark dataset cannot be locked before replacement.") }
        defer { flock(descriptor, LOCK_UN); Darwin.close(descriptor) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            throw StudioError("Finish the active Bark operation before publishing the fresh dataset.")
        }
        try FileManager.default.createDirectory(at: archive, withIntermediateDirectories: false)
        let registryURL = workspace.appendingPathComponent("specialists.json")
        try FileManager.default.copyItem(at: registryURL, to: archive.appendingPathComponent("specialists-before.json"))
        let prefs = try XCTUnwrap(UserDefaults(suiteName: "org.ipde.material-tools"))
        if let settings = prefs.data(forKey: WorkbenchPreferences.key) {
            try settings.write(to: archive.appendingPathComponent("workbench-settings-before.json"))
        }
        try FileManager.default.moveItem(at: destination, to: archive.appendingPathComponent("old-bark-dataset"))
        do { try FileManager.default.moveItem(at: stage, to: destination) }
        catch {
            try? FileManager.default.moveItem(at: archive.appendingPathComponent("old-bark-dataset"), to: destination)
            throw error
        }
        let binding: [String: Any] = ["schema": NativeMaterialSpecialistService.schema,
            "specialist_id": registry.specialists[barkOffset].id, "workspace_path": workspace.path]
        try write(binding, to: destination.appendingPathComponent(".specialist-binding.json"))
        let outputs = support.appendingPathComponent("Material Workspace/out/material-training")
        if FileManager.default.fileExists(atPath: outputs.path) {
            try FileManager.default.moveItem(at: outputs, to: archive.appendingPathComponent("old-material-training"))
            try FileManager.default.createDirectory(at: outputs, withIntermediateDirectories: false)
        }
        for target in ["height", "normal", "roughness"] {
            let selected = SelectedMaterialCheckpoint.registryURL(for: target)
            if FileManager.default.fileExists(atPath: selected.path) {
                try FileManager.default.moveItem(at: selected, to: archive.appendingPathComponent(selected.lastPathComponent))
            }
        }
        registry.specialists[barkOffset].models = []
        registry.updatedAt = ISO8601DateFormatter().string(from: Date())
        let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(registry).write(to: registryURL, options: .atomic)
        // Recompute membership from exact current sources, without fabricating
        // classifier predictions or retaining old model approvals.
        registry = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Bark")
        try encoder.encode(registry).write(to: registryURL, options: .atomic)
        guard registry.specialists[barkOffset].familyIds.count == 18,
              registry.specialists[barkOffset].materialIds.count == 51,
              registry.specialists[barkOffset].models.isEmpty,
              registry.specialists[barkOffset].targets.allSatisfy({ $0.status == "ready" && $0.materialCount == 51 }) else {
            throw StudioError("The fresh Bark specialist registry did not retain its verified membership.")
        }
        var settings = WorkbenchPreferences.load(from: prefs)
        var options = settings.training ?? MaterialTrainingOptions()
        options.modelFamily = .compactScalar; options.target = "height"; options.scope = "full-model"
        options.modelName = "Bark Displacement"; options.size = 1024; options.learningRate = 0.0001
        options.useWarmStart = false; options.useSelectedMaterialOnly = false
        options.validationEvery = 1; options.validationUnit = .epoch
        options.checkpointEvery = 1; options.checkpointUnit = .epoch
        options.maxMinutes = 20000
        settings = WorkbenchPreferences(training: options, comparisonCheckpointIds: [], comparisonIncludesBase: false)
        settings.save(to: prefs); prefs.set([String](), forKey: "checkpoints"); prefs.set(destination.path, forKey: "dataset")
        prefs.set(registry.specialists[barkOffset].id, forKey: "selectedSpecialist"); prefs.set("height", forKey: "specialistTarget")
        prefs.synchronize()
        let report: [String: Any] = ["dataset_path": destination.path, "archive_path": archive.path,
            "source_inventory_sha256": try NativeMaterialTransfer.hash(inventoryURL),
            "source_originals_unchanged": true, "family_count": 18, "source_resolution_set_count": 51,
            "training_crop_count": trainingCount, "validation_crop_count": validationCount,
            "validation_scope": "extra crop from a known subject; no unseen-material accuracy claim",
            "crop_policy": NativeMaterialDatasetService.cropSelectionPolicy,
            "native_training_size": 1024, "learning_rate": 0.0001, "warm_start": false,
            "old_checkpoint_selection_cleared": true, "old_training_outputs_archived": true,
            "old_bark_dataset_and_caches_archived": true, "fresh_dataset_has_training_cache": false,
            "depthpro_used": false, "training_started": false, "crops": cropReceipts]
        try write(report, to: evidence.appendingPathComponent("fresh-preparation.json"))
        print("BARK_FRESH_READY " + destination.path)
    }

    private func verifyOriginals(_ maps: [[String: Any]], phase: String) throws {
        for (offset, map) in maps.enumerated() {
            let path = try XCTUnwrap(map["path"] as? String), expected = try XCTUnwrap(map["file_sha256"] as? String)
            guard path.hasPrefix("/opt/ipde/sources_mats/") else { throw StudioError("The original-map verification escaped the source corpus.") }
            print("BARK_FRESH_SOURCE_HASH \(phase) \(offset + 1)/\(maps.count)")
            guard try NativeMaterialTransfer.hash(URL(fileURLWithPath: path)) == expected else {
                throw StudioError("An original bark map changed: \(path)")
            }
        }
    }
    private func object(_ url: URL) throws -> [String: Any] { try NativeMaterialTransfer.object(Data(contentsOf: url)) }
    private func write(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    }
}
