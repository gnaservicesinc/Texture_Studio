import Foundation
import Metal
import XCTest
@testable import TextureStudio

/// Runs only when explicitly requested, using copies of current originals.
/// The receipt establishes execution and native dimensions, not model quality.
final class FolderTrainingProductionTests: XCTestCase {
    func testTaggedRealTwoKCategoryCompletesNativeTrainingWithoutChangingOriginals() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_FOLDER_TRAINING_PROOF"] == "1" else {
            throw XCTSkip("Opt-in current-folder 2048 training proof")
        }
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let outputRoot = URL(fileURLWithPath: try XCTUnwrap(environment["TEXTURE_STUDIO_FOLDER_TRAINING_OUTPUT"]))
        let root = outputRoot.appendingPathComponent("real-two-k-" + UUID().uuidString)
        let source = root.appendingPathComponent("Sources")
        let originals = [
            URL(fileURLWithPath: "/opt/ipde/sources_mats/brick_wall_005/brick_wall_005_diff_2k.png"),
            URL(fileURLWithPath: "/opt/ipde/sources_mats/brick_wall_005/brick_wall_005_disp_2k.png")
        ]
        let originalHashes = try originals.map(NativeMaterialTransfer.hash)
        for (name, label) in [("Bricks", "Bricks"), ("Other Category", "Wood")] {
            let folder = source.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for (original, role, filename) in [(originals[0], "input", "diffuse.png"), (originals[1], "height", "height.png")] {
                let copy = folder.appendingPathComponent(filename)
                try FileManager.default.copyItem(at: original, to: copy)
                try NativeSourceFileTags.update(copy, role: role, classLabel: label, classifierIdentity: "user")
            }
        }
        // An obsolete list cannot turn this folder into a one-K dataset or
        // hide one of its physical materials.
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "training_size": 1024, "samples": []])
            .write(to: source.appendingPathComponent("dataset.json"))
        _ = try await NativeMaterialDatasetService.run(arguments: ["edit-dataset", "--dataset", source.path,
            "--training-size", "2048", "--validation-enabled", "no"])
        let classifier = NativeMaterialSpecialistService.Classifier(path: "/unused-tagged-only", sha256: String(repeating: "a", count: 64),
            labels: ["Bricks", "Wood"], predict: { _ in throw StudioError("Manual tags must avoid classifier prediction") })
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: source,
            workspaceURL: root.appendingPathComponent("Categories"), classifier: classifier)
        let bricks = try XCTUnwrap(registry.specialists.first { $0.classLabel == "Bricks" })
        XCTAssertEqual(bricks.materialIds.count, 1)
        let recorder = Recorder()
        let result = try NativeMaterialTransfer.object(Data(try await NativeMaterialTrainer.run(arguments: ["train",
            "--dataset", bricks.datasetPath, "--output", root.appendingPathComponent("Trained Bricks").path,
            "--model-family", "compact-scalar", "--target", "height", "--size", "2048", "--compact-width", "22",
            "--updates-per-map", "1", "--randomize-crops", "epoch", "--model-name", "Bricks 2048 Folder Proof",
            "--max-minutes", "5", "--shuffle-training-order", "false", "--exaggerate-features"], onEvent: recorder.record).utf8))
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
        let started = try XCTUnwrap(recorder.events.first { $0["event"] as? String == "training_started" })
        XCTAssertEqual(started["training_size"] as? Int, 2048)
        XCTAssertEqual(started["training_map_count"] as? Int, 1, "Only the current Bricks tag may train")
        let checkpoint = URL(fileURLWithPath: try XCTUnwrap(result["checkpoint_path"] as? String))
        let (configuration, _) = try NativeMaterialPackage.verify(checkpoint.deletingLastPathComponent())
        XCTAssertEqual(configuration["training_size"] as? Int, 2048)
        XCTAssertEqual(configuration["model_name"] as? String, "Bricks 2048 Folder Proof")
        let settings = try XCTUnwrap(configuration["training_configuration"] as? [String: Any])
        XCTAssertEqual(settings["exaggerate_features"] as? Bool, true)
        XCTAssertEqual(settings["shuffle_training_order"] as? Bool, false)
        XCTAssertEqual(try originals.map(NativeMaterialTransfer.hash), originalHashes)
        let receipt: [String: Any] = ["source_originals": originals.map(\.path), "source_sha256": originalHashes,
            "originals_unchanged": true, "training_size": 2048, "training_map_count": 1, "completed_updates": 1,
            "tag_scope": "Bricks", "excluded_tag_scope": "Wood", "result": result, "events": recorder.events,
            "quality_qualification": false]
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
            .write(to: outputRoot.appendingPathComponent("real-two-k-training-proof.json"))
    }

    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [[String: Any]] = []
        func record(_ text: String) {
            guard let value = try? NativeMaterialTransfer.object(Data(text.utf8)) else { return }
            lock.withLock { lines.append(value) }
        }
        var events: [[String: Any]] { lock.withLock { lines } }
    }
}
