import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchTrainingCompletionTests: XCTestCase {
    func testNeverRebuildsFromCurrentTaggedCategoryWithZeroSeedBeforeTraining() async throws {
        let outcome = try await run(failing: "none", cropTiming: .never)
        XCTAssertEqual(outcome.progress?.state, .completed)
        XCTAssertNil(outcome.error)
        let preparation = try XCTUnwrap(outcome.arguments.first { $0.first == "prepare-size" })
        let category = preparation[try XCTUnwrap(preparation.firstIndex(of: "--dataset")) + 1]
        XCTAssertTrue(category.hasSuffix("/categories/wicker"), "Refresh must replace an old randomized cache with its current category")
        XCTAssertEqual(preparation[try XCTUnwrap(preparation.firstIndex(of: "--crop-seed")) + 1], "0")
        let training = try XCTUnwrap(outcome.arguments.first { $0.first == "train" })
        let trainingDataset = training[try XCTUnwrap(training.firstIndex(of: "--dataset")) + 1]
        XCTAssertNotEqual(trainingDataset, category, "Never must use the returned deterministic preparation")
        XCTAssertTrue(trainingDataset.hasSuffix("/training-cache/512"))
        XCTAssertEqual(Array(outcome.commands.prefix(3)), ["refresh-specialists", "dataset", "prepare-size"])
        let registration = try XCTUnwrap(outcome.arguments.first { $0.first == "record-specialist-model" })
        XCTAssertEqual(registration[try XCTUnwrap(registration.firstIndex(of: "--membership-sha256")) + 1], "current-wicker-members",
            "Completion must bind the saved model to refreshed membership, not the obsolete browser snapshot")
    }

    func testUnexpectedCleanupFilesKeepTrainingCompletedAndFinalModelAvailable() async throws {
        let outcome = try await run(failing: "cleanup-size")
        XCTAssertEqual(outcome.progress?.state, .completed)
        XCTAssertEqual(outcome.progress?.completedUpdates, 6425)
        XCTAssertTrue(outcome.activity.contains("could not finish training-file cleanup"))
        XCTAssertTrue(outcome.activity.contains("Unexpected files"))
        XCTAssertNotNil(outcome.checkpoint)
        XCTAssertNotNil(outcome.package)
        XCTAssertNil(outcome.error)
        XCTAssertTrue(outcome.savedBytesUnchanged)
        XCTAssertTrue(outcome.commands.contains("dataset"), "Cleanup failure must still reopen the source dataset")
        XCTAssertTrue(outcome.datasetPath?.hasSuffix("/categories/wicker") == true)
        XCTAssertFalse(outcome.activity.contains("could not reopen the source dataset"))
    }

    func testSourceReopenFailureIsReportedSeparatelyAfterSavedTraining() async throws {
        let outcome = try await run(failing: "dataset")
        XCTAssertEqual(outcome.progress?.state, .completed)
        XCTAssertTrue(outcome.commands.contains("cleanup-size"))
        XCTAssertTrue(outcome.activity.contains("could not reopen the source dataset"))
        XCTAssertFalse(outcome.activity.contains("could not finish training-file cleanup"))
        XCTAssertNotNil(outcome.package)
        XCTAssertNil(outcome.error)
        XCTAssertTrue(outcome.savedBytesUnchanged)
    }

    func testAccountAndUploadFailuresCannotTurnASavedRunIntoTrainingFailure() async throws {
        for command in ["hub-account", "upload-selected"] {
            let outcome = try await run(failing: command, publish: true)
            XCTAssertEqual(outcome.progress?.state, .completed)
            XCTAssertTrue(outcome.activity.contains("Hugging Face upload did not finish"))
            XCTAssertNil(outcome.error)
            XCTAssertNotNil(outcome.checkpoint)
            XCTAssertTrue(outcome.savedBytesUnchanged)
        }
    }

    func testLibraryRefreshFailureKeepsFinalPackageAndDoesNotUploadPreviousSelection() async throws {
        let outcome = try await run(failing: "checkpoint", publish: true)
        XCTAssertEqual(outcome.progress?.state, .completed)
        XCTAssertTrue(outcome.activity.contains("Could not refresh the model library"))
        XCTAssertNil(outcome.error)
        XCTAssertNotNil(outcome.package)
        XCTAssertEqual(outcome.checkpoint?.url.lastPathComponent, "previous.safetensors")
        XCTAssertFalse(outcome.commands.contains("upload-selected"))
        XCTAssertTrue(outcome.savedBytesUnchanged)
    }

    func testCatalogRefreshFailureAfterCommitStillReportsSuccessfulUpload() async throws {
        let outcome = try await run(failing: "hub-models", publish: true)
        XCTAssertEqual(outcome.progress?.state, .completed)
        XCTAssertNotNil(outcome.upload)
        XCTAssertTrue(outcome.activity.contains("Uploaded"))
        XCTAssertTrue(outcome.activity.contains("Could not refresh the Hub model list"))
        XCTAssertFalse(outcome.activity.contains("upload did not finish"))
        XCTAssertNil(outcome.error)
    }

    func testCancellationAfterFinalSavePreservesCompletedTraining() async throws {
        let outcome = try await run(failing: "none", cancelAfterSave: true)
        XCTAssertEqual(outcome.progress?.state, .completed)
        XCTAssertEqual(outcome.progress?.completedUpdates, 6425)
        XCTAssertNotNil(outcome.package)
        XCTAssertNil(outcome.error)
        XCTAssertTrue(outcome.savedBytesUnchanged)
        XCTAssertFalse(outcome.commands.contains("upload-selected"))
        XCTAssertTrue(outcome.datasetPath?.hasSuffix("/categories/wicker") == true,
            "Post-save cancellation must still restore the current category")
    }

    func testTrainingRequiresCategoryWithoutStartingAnyWorker() throws {
        let suite = "training-no-category-" + UUID().uuidString
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        var calls = 0
        let store = WorkbenchStore(preferences: preferences, workerOverride: { _, _ in
            calls += 1
            XCTFail("Choosing a category is required before any worker command")
            return "{}"
        })
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
            {"dataset_path":"/originals","index_sha256":"cached","is_source_folder":true,
             "materials":[{"material_id":"wicker","samples":[{"sample_id":"old-id","status":"approved",
             "split":"train","width":2048,"height":2048,"maps":{"input":{"path":"/originals/color.png"},
             "height":{"path":"/originals/height.png","source_bits":16}}}]}]}
            """)
        store.selectTrainingModelFamily(.compactScalar)
        store.startTraining()
        XCTAssertEqual(store.error, "Choose a material category in Loom Room.")
        XCTAssertFalse(store.isBusy)
        XCTAssertEqual(calls, 0)
    }

    func testOrderAndFeatureControlsDispatchOnlySupportedTargetFlags() async throws {
        for (target, shuffle, exaggerate) in [("height", true, false), ("height", false, true), ("roughness", false, true)] {
            let outcome = try await run(failing: "none", target: target, shuffle: shuffle, exaggerate: exaggerate)
            XCTAssertNil(outcome.error)
            let training = try XCTUnwrap(outcome.arguments.first { $0.first == "train" })
            XCTAssertEqual(training[try XCTUnwrap(training.firstIndex(of: "--shuffle-training-order")) + 1], shuffle ? "true" : "false")
            XCTAssertEqual(training.contains("--exaggerate-features"), target == "height" && exaggerate)
        }
    }

    private struct Outcome {
        let progress: WorkbenchTrainingProgress?
        let activity: String
        let error: String?
        let checkpoint: WorkbenchCheckpoint?
        let package: URL?
        let upload: URL?
        let commands: [String]
        let arguments: [[String]]
        let savedBytesUnchanged: Bool
        let datasetPath: String?
    }

    private func run(failing command: String, publish: Bool = false, cancelAfterSave: Bool = false,
                     cropTiming: MaterialTrainingCropRandomization = .run, target: String = "height",
                     shuffle: Bool = true, exaggerate: Bool = false) async throws -> Outcome {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("training-completion-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "training-completion-" + UUID().uuidString
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let global = StudioPreferences.defaults
        let previousDeveloper = global.object(forKey: StudioPreferences.developerModeKey)
        global.set(publish, forKey: StudioPreferences.developerModeKey)
        defer { global.set(previousDeveloper, forKey: StudioPreferences.developerModeKey) }
        let package = root.appendingPathComponent("export")
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        let file = package.appendingPathComponent("model.safetensors")
        let saved = Data("previously published final model".utf8)
        try saved.write(to: file)
        let hash = String(repeating: "a", count: 64)
        let source = root.appendingPathComponent("source").path
        let category = root.appendingPathComponent("categories/wicker").path
        let modelWorkspace = root.appendingPathComponent("categories").path
        let prepared = root.appendingPathComponent("training-cache/512").path
        try FileManager.default.createDirectory(atPath: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: category, withIntermediateDirectories: true)
        func json(_ object: [String: Any]) throws -> String {
            String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        }
        func dataset(path: String, prepared: Bool) throws -> String {
            var object: [String: Any] = ["dataset_path": path, "source_folder_path": source,
                "index_sha256": "dataset-sha", "materials": [
                ["material_id": "wicker", "samples": [["sample_id": "wicker", "status": "approved", "split": "train",
                    "width": 512, "height": 512, "maps": [
                        "input": ["path": source + "/diffuse.png", "width": 512, "height": 512],
                        target: ["path": source + "/" + target + ".png", "width": 512, "height": 512, "source_bits": 16]]]]]]]
            if prepared {
                object["automatic_validation"] = ["policy": "subject-extra-crops-v2", "target": target, "material_ids": ["wicker"]]
                object["preparation"] = ["source_dataset_path": category, "source_index_sha256": "source-sha",
                    "prepared_dataset_path": path, "crop_size": 512, "reused": false, "target_resized": false,
                    "original_dataset_modified": false]
            }
            return try json(object)
        }
        func registry() throws -> String {
            try json(["workspace_path": modelWorkspace, "source_dataset_path": source,
                "classifier_sha256": "", "classifier_labels": ["Wicker"], "updated_at": "2026-10-12",
                "classifier_only_count": 0, "material_count": 1, "family_count": 1, "warnings": [],
                "specialists": [["id": "wicker", "name": "Wicker", "class_label": "Wicker", "is_custom": false,
                    "dataset_path": category, "material_ids": ["wicker"], "family_ids": ["wicker"], "models": [],
                    "targets": [["target": target, "material_count": 1, "status": "ready", "membership_sha256": "current-wicker-members"]]]]])
        }
        var commands: [String] = []
        var capturedArguments: [[String]] = []
        weak var activeStore: WorkbenchStore?
        let store = WorkbenchStore(preferences: preferences, managedWorkspaceURL: root, workerOverride: { arguments, operation in
            commands.append(operation)
            capturedArguments.append(arguments)
            if operation == command, operation != "dataset" || commands.contains("train") {
                throw StudioError(operation == "cleanup-size" ? "Unexpected files in temporary training folder." : "Simulated " + operation + " failure")
            }
            switch operation {
            case "refresh-specialists", "record-specialist-model": return try registry()
            case "prepare-size": return try dataset(path: prepared, prepared: true)
            case "train":
                if cancelAfterSave { activeStore?.abort() }
                return try json(["status": "completed", "completed_updates": 6425, "requested_updates": 6425,
                    "checkpoint_path": file.path, "package_path": package.path])
            case "checkpoint":
                return try json(["checkpoint_path": file.path, "sha256": hash, "schema": "texture-studio-compact-material-v1",
                    "architecture": "texture-studio-compact-scalar-native-v1", "target": target, "step": 6425,
                    "compatible": true, "scope": "full-model", "training_size": 512])
            case "cleanup-size": return try json(["dataset_path": prepared, "source_dataset_path": category, "removed": true])
            case "dataset":
                XCTAssertEqual(arguments[try XCTUnwrap(arguments.firstIndex(of: "--dataset")) + 1], category)
                return try dataset(path: category, prepared: false)
            case "hub-account": return try json(["authenticated": true, "username": "test-account", "message": "Signed in"])
            case "upload-selected":
                let repository = arguments[try XCTUnwrap(arguments.firstIndex(of: "--repo")) + 1]
                return try json(["repository": repository, "private": true, "url": "https://huggingface.co/" + repository,
                    "commit_url": "https://huggingface.co/" + repository + "/commit/test", "source_checkpoint_sha256": hash,
                    "package_path": package.path])
            case "hub-models": return "{\"models\":[]}"
            default: throw StudioError("Unexpected test command: " + operation)
            }
        })
        activeStore = store
        let initialPath = cropTiming == .never ? root.appendingPathComponent("training-cache/randomized").path : prepared
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: dataset(path: initialPath, prepared: true))
        store.specialistSourceDatasetPath = source
        store.specialistTrainingContext = .init(workspacePath: modelWorkspace, specialistID: "wicker", name: "Wicker",
            target: target, datasetPath: category, membershipSHA256: "old-wicker-members")
        store.training.size = 512
        store.training.randomizeCrops = cropTiming
        store.selectTrainingModelFamily(.compactScalar)
        store.training.target = target
        store.training.shuffleTrainingOrder = shuffle
        store.training.exaggerateFeatures = exaggerate
        if command == "checkpoint" {
            let previous = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: json([
                "checkpoint_path": root.appendingPathComponent("previous.safetensors").path, "sha256": String(repeating: "b", count: 64),
                "schema": "texture-studio-compact-material-v1", "architecture": "texture-studio-compact-scalar-native-v1",
                "target": "height", "step": 100, "compatible": true]))
            store.checkpoints = [previous]
            store.selectedCheckpointId = previous.id
        }
        store.startTraining()
        for _ in 0..<500 where store.isBusy { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isBusy)
        return Outcome(progress: store.trainingProgress, activity: store.activity, error: store.error,
            checkpoint: store.selectedCheckpoint, package: store.lastPackageURL, upload: store.lastUploadURL,
            commands: commands, arguments: capturedArguments, savedBytesUnchanged: try Data(contentsOf: file) == saved,
            datasetPath: store.dataset?.datasetPath)
    }
}
