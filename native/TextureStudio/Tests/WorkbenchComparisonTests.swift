import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchComparisonTests: XCTestCase {
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
        store.dataset = try WorkbenchProcess.decode(WorkbenchDataset.self, output: """
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
        XCTAssertEqual(store.comparisonCandidates.map(\.role), ["source", "target", "base", "checkpoint"])
        XCTAssertTrue(store.comparisonCandidates.allSatisfy { $0.sampleLabel == "soil_crop_002" })
        let base = try XCTUnwrap(store.comparisonCandidates.first { $0.role == "base" })
        XCTAssertTrue(base.label.contains("untrained"))
        XCTAssertTrue(base.detail?.contains("not a pretrained depth estimator") == true)
        XCTAssertEqual(base.modelIdentity?.architecture, "DINOv2 Base + untrained material head")
        XCTAssertNil(base.modelIdentity?.checkpointStep, "The untrained baseline does not inherit a trained checkpoint's step")
        let trained = try XCTUnwrap(store.comparisonCandidates.first { $0.role == "checkpoint" })
        XCTAssertTrue(trained.detail?.contains("step 42") == true)
        XCTAssertEqual(trained.modelIdentity?.checkpointPath, "/runs/material-2k/checkpoint.selected.pt")
        XCTAssertEqual(trained.modelIdentity?.checkpointSHA256, "exact-sha")
        XCTAssertEqual(trained.modelIdentity?.checkpointStep, 42)
        XCTAssertEqual(trained.modelIdentity?.architecture, "DINOv2 Base + trained material head")
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
        XCTAssertEqual(review.groups.first?.candidates.map(\.role), ["source", "target", "base", "checkpoint"])
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

    func testDisablingBaseStillRequiresTwoMatchingCheckpoints() async throws {
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
        store.comparisonCheckpointIds = ["height"]
        store.comparisonIncludesBase = false
        store.compare()
        XCTAssertNotNil(store.error)
        XCTAssertFalse(store.isBusy)
        store.error = nil
        store.comparisonIncludesBase = true
        store.comparisonCheckpointIds = ["height", "normal"]
        store.compare()
        XCTAssertNotNil(store.error)
        XCTAssertFalse(store.isBusy)
    }

    private func checkpoint(sha: String, target: String) throws -> WorkbenchCheckpoint {
        try WorkbenchProcess.decode(WorkbenchCheckpoint.self, output: """
        {"checkpoint_path":"/runs/material-2k/checkpoint.selected.pt","sha256":"\(sha)","schema":"texture-studio-material-training-cycle-v1","target":"\(target)","step":42,"compatible":true}
        """)
    }

    private func waitForOperation(_ store: WorkbenchStore) async throws {
        let deadline = Date().addingTimeInterval(5)
        while store.isBusy, Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isBusy)
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("comparison-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
