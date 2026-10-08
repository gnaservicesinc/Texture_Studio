import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class ReviewIdentityTests: XCTestCase {
    func testLegacyReviewNamesIdentifyExactTrainingRunsAndNonModelReferences() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "review-identity-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let manifest = root.appendingPathComponent("legacy.json")
        let variants: [[String: Any]] = [
            ["name": "flat", "height": "flat.exr"],
            ["name": "target", "height": "displacement.png"],
            ["name": "starting_head", "height": "starting.exr", "checkpoint": "/runs/four-material-adaptation-01/frozen/checkpoint.final.pt",
             "checkpoint_sha256": "2bec143e0c5f-exact-starting", "checkpoint_step": 1200],
            ["name": "trained_2k", "height": "trained.exr", "checkpoint": "/runs/native-2k-material-cycle-01/checkpoint.final.pt",
             "checkpoint_sha256": "1a3b86ed7008-exact-trained", "checkpoint_step": 800]
        ]
        try writeManifest(variants, to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        XCTAssertNil(store.error)
        let candidates = try XCTUnwrap(store.groups.first?.candidates)
        XCTAssertEqual(candidates.map(\.role), ["source", "base", "target", "checkpoint", "checkpoint"])
        XCTAssertEqual(candidates[1].label, "Flat baseline · no model")
        XCTAssertTrue(candidates[1].detail?.contains("no model inference") == true)
        XCTAssertEqual(candidates[2].label, "Reference displacement")
        XCTAssertTrue(candidates[2].detail?.contains("not a model output") == true)
        XCTAssertTrue(candidates[2].detail?.contains("16-bit") == true)
        XCTAssertTrue(candidates[3].label.contains("four-material-adaptation-01/frozen"))
        XCTAssertTrue(candidates[3].detail?.contains("1,200") == true)
        XCTAssertTrue(candidates[4].label.contains("native-2k-material-cycle-01"))
        XCTAssertTrue(candidates[4].detail?.contains("800") == true)
        XCTAssertTrue(candidates[3].accessibleLabel.contains("brick_crop_003"))
        XCTAssertTrue(candidates[3].exportFilename.contains("2bec143e0c5f"))
        XCTAssertTrue(candidates[4].exportFilename.contains("1a3b86ed7008"))
        XCTAssertNotEqual(candidates[3].exportFilename, candidates[4].exportFilename)
    }

    func testSavedReviewRetainsStructuredProvenanceAndOriginalBytes() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "review-identity-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let map = root.appendingPathComponent("signed-float.exr")
        let original = Data("signed float source samples must stay untouched".utf8)
        try original.write(to: map)
        let manifest = root.appendingPathComponent("comparison.json")
        try writeManifest([["name": "trained_2k", "height": map.path,
            "checkpoint": "runs/refined/checkpoint.selected.pt", "checkpoint_sha256": "unique-verified-checkpoint-sha",
            "checkpoint_step": 7600, "model_architecture": "DINOv2 Base features + material height head"]], to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        let candidate = try XCTUnwrap(store.groups.first?.candidates.last)
        store.decisions[candidate.id] = "needs_work"
        store.notes[candidate.id] = "Inspect mortar relief"
        let saved = root.appendingPathComponent("review.json")
        try store.writeReview(to: saved)
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: saved)) as? [String: Any])
        let materials = try XCTUnwrap(raw["materials"] as? [[String: Any]])
        let variants = try XCTUnwrap(materials.first?["variants"] as? [[String: Any]])
        let recorded = try XCTUnwrap(variants.last)
        XCTAssertEqual(recorded["checkpoint"] as? String, root.appendingPathComponent("runs/refined/checkpoint.selected.pt").path)
        XCTAssertEqual(recorded["checkpoint_sha256"] as? String, "unique-verified-checkpoint-sha")
        XCTAssertEqual(recorded["checkpoint_step"] as? Int, 7600)
        XCTAssertEqual(recorded["model_architecture"] as? String, "DINOv2 Base features + material height head")
        let reopened = ReviewSessionStore(preferences: preferences)
        reopened.load(saved)
        XCTAssertNil(reopened.error)
        let restored = try XCTUnwrap(reopened.groups.first?.candidates.last)
        XCTAssertEqual(restored, candidate)
        XCTAssertEqual(restored.exportFilename, candidate.exportFilename)
        XCTAssertEqual(reopened.decisions[candidate.id], "needs_work")
        XCTAssertEqual(reopened.notes[candidate.id], "Inspect mortar relief")
        XCTAssertEqual(try Data(contentsOf: map), original)
    }

    func testDeepBumpAndMissingArchitectureRemainHonest() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "review-identity-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let manifest = root.appendingPathComponent("comparison.json")
        try writeManifest([
            ["name": "deepbump_height", "height": "deepbump.exr"],
            ["name": "trained_2k", "height": "unknown.exr", "checkpoint": "/runs/da3-in-filename/checkpoint.pt"]
        ], to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        let candidates = try XCTUnwrap(store.groups.first?.candidates)
        XCTAssertEqual(candidates[1].role, "model")
        XCTAssertEqual(candidates[1].label, "DeepBump · displacement")
        XCTAssertTrue(candidates[1].detail?.contains("not the dataset reference") == true)
        XCTAssertNil(candidates[2].modelIdentity?.architecture)
        XCTAssertTrue(candidates[2].detail?.contains("Architecture not recorded") == true)
        XCTAssertFalse(candidates[2].detail?.contains("Depth Anything 3") == true)
    }

    func testSavedCheckpointRunWithUnderscoresDoesNotAccumulateDuplicateLabels() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "review-identity-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let initial = root.appendingPathComponent("comparison.json")
        let name = "native_2k_refinement · Displacement"
        try writeManifest([["name": name, "height": "height.exr", "role": "checkpoint", "candidate_id": "exact-candidate",
            "checkpoint": "/runs/native_2k_refinement/checkpoint.selected.pt", "checkpoint_sha256": "exact-model-sha",
            "checkpoint_step": 123, "model_architecture": "DINOv2 Base + trained material head"]], to: initial)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(initial)
        let candidate = try XCTUnwrap(store.groups.first?.candidates.last)
        XCTAssertEqual(candidate.label, name)
        for cycle in 1...3 {
            let saved = root.appendingPathComponent("review-\(cycle).json")
            try store.writeReview(to: saved)
            store.load(saved)
            XCTAssertNil(store.error)
            let restored = try XCTUnwrap(store.groups.first?.candidates.last)
            XCTAssertEqual(restored.label, candidate.label)
            XCTAssertEqual(restored.modelIdentity, candidate.modelIdentity)
            XCTAssertEqual(restored.exportFilename, candidate.exportFilename)
        }
    }

    func testLongExportNamesKeepCheckpointIdentitySuffix() {
        let sample = String(repeating: "long-sample-", count: 30)
        let candidates = ["first-exact-sha", "second-exact-sha"].map { checksum in
            MapReviewCandidate(id: "same-leading-sample/\(checksum)", label: "Same material run", mapURL: URL(fileURLWithPath: "/map.exr"),
                numeric: true, sampleLabel: sample, role: "checkpoint",
                modelIdentity: MapReviewModelIdentity(checkpointSHA256: checksum))
        }
        XCTAssertNotEqual(candidates[0].exportFilename, candidates[1].exportFilename)
        XCTAssertTrue(candidates[0].exportFilename.hasSuffix("first-exact-.exr"))
        XCTAssertTrue(candidates[1].exportFilename.hasSuffix("second-exact.exr"))
        XCTAssertLessThanOrEqual(candidates[0].exportFilename.count, 184)
    }

    private func writeManifest(_ variants: [[String: Any]], to url: URL) throws {
        let manifest: [String: Any] = ["materials": [["material_id": "brick_crop_003", "diffuse": "diffuse.png",
            "target_original_bits": 16, "variants": variants]]]
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]).write(to: url)
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("review-identity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
