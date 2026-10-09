import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class ReviewIdentityTests: XCTestCase {
    func testTrainingReviewRestoresExactGridAndSavesSourceTransforms() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "review-transform-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let manifest = root.appendingPathComponent("training-review.json")
        let diffuseHash = String(repeating: "a", count: 64), targetHash = String(repeating: "b", count: 64)
        let material: [String: Any] = ["material_id": "stone", "diffuse": "/original/diffuse.png",
            "diffuse_source_sha256": diffuseHash, "diffuse_native_size": 2048, "diffuse_resize_algorithm": "exact_native_integer_codes", "diffuse_source_crop_rectangle": [1024, 1024, 2048, 2048],
            "variants": [["name": "target", "role": "target", "height": "/original/height.png",
                "source_sha256": targetHash, "native_dimensions": [2048, 2048],
                "source_resize_algorithm": MapReviewDisplayTransform.exactCrop],
                ["name": "refined", "role": "checkpoint", "height": "/prediction/height.exr", "map_type": "height"]]]
        try JSONSerialization.data(withJSONObject: ["materials": [material], "comparison_target": "height"]).write(to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        XCTAssertNil(store.error)
        let candidates = try XCTUnwrap(store.selected?.candidates)
        let diffuse = try XCTUnwrap(candidates.first { $0.role == "diffuse" })
        XCTAssertEqual(diffuse.label, "Diffuse · model input")
        XCTAssertEqual(diffuse.mapURL.path, "/original/diffuse.png")
        XCTAssertEqual(diffuse.displayTransform?.sourceSHA256, diffuseHash)
        XCTAssertEqual(diffuse.displayTransform?.size, 2048)
        let reference = try XCTUnwrap(candidates.first(where: \.isReference))
        XCTAssertEqual(reference.displayTransform?.sourceSHA256, targetHash)
        XCTAssertNil(try XCTUnwrap(reference.fullSourceReference).displayTransform, "Full source opens without reconstructing a training grid")
        let saved = root.appendingPathComponent("saved.json")
        try store.writeReview(to: saved)
        let reopened = ReviewSessionStore(preferences: preferences)
        reopened.load(saved)
        XCTAssertNil(reopened.error)
        XCTAssertEqual(reopened.selected?.candidates.first { $0.role == "diffuse" }?.displayTransform, diffuse.displayTransform)
    }
    func testNormalCropConventionSurvivesSavedReviewWithoutGlobalTarget() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "normal-crop-review-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let manifest = root.appendingPathComponent("normal-review.json")
        let variant: [String: Any] = ["name": "source normal", "role": "target", "normal": "/source/normal.png",
            "map_type": "normal", "source_sha256": String(repeating: "a", count: 64), "native_dimensions": [2048, 2048],
            "source_resize_algorithm": MapReviewDisplayTransform.exactCrop, "source_normal_convention": "directx",
            "source_crop_rectangle": [1024, 1024, 2048, 2048]]
        let document: [String: Any] = ["comparison_target": "normal", "materials": [["material_id": "stone", "variants": [variant]]]]
        try JSONSerialization.data(withJSONObject: document).write(to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        XCTAssertNil(store.error)
        let original = try XCTUnwrap(store.groups.first?.candidates.first)
        let saved = root.appendingPathComponent("saved.json")
        try store.writeReview(to: saved)
        store.load(saved)
        XCTAssertNil(store.error)
        let restored = try XCTUnwrap(store.groups.first?.candidates.first)
        XCTAssertEqual(restored.displayTransform, original.displayTransform)
        XCTAssertEqual(restored.displayTransform?.mapType, "normal")
        XCTAssertEqual(restored.displayTransform?.normalConvention, "directx")
        XCTAssertEqual(restored.displayTransform?.cropRectangle, [1024, 1024, 2048, 2048])
    }

    func testDeclaredReviewRolesIdentifyExactTrainingRunsAndRealReferences() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "review-identity-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let manifest = root.appendingPathComponent("review.json")
        let variants: [[String: Any]] = [
            ["name": "Pretrained base", "role": "base", "height": "base.exr"],
            ["name": "target", "role": "target", "height": "displacement.png"],
            ["name": "Starting model", "role": "checkpoint", "height": "starting.exr", "checkpoint": "/runs/four-material-adaptation-01/frozen/checkpoint.final.pt",
             "checkpoint_sha256": "2bec143e0c5f-exact-starting", "checkpoint_step": 1200],
            ["name": "Trained model", "role": "checkpoint", "height": "trained.exr", "checkpoint": "/runs/native-2k-material-cycle-01/checkpoint.final.pt",
             "checkpoint_sha256": "1a3b86ed7008-exact-trained", "checkpoint_step": 800]
        ]
        try writeManifest(variants, to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        XCTAssertNil(store.error)
        let candidates = try XCTUnwrap(store.groups.first?.candidates)
        XCTAssertEqual(candidates.map(\.role), ["diffuse", "base", "target", "checkpoint", "checkpoint"])
        XCTAssertEqual(candidates[1].label, "Pretrained base")
        XCTAssertTrue(candidates[1].detail?.contains("Base model prediction") == true)
        XCTAssertEqual(candidates[2].label, "Source displacement · reference")
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
        try writeManifest([["name": "Trained model", "role": "checkpoint", "height": map.path,
            "checkpoint": "runs/refined/checkpoint.selected.pt", "checkpoint_sha256": "unique-verified-checkpoint-sha",
            "checkpoint_step": 7600, "model_architecture": "PBRnxt material height"]], to: manifest)
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
        XCTAssertEqual(recorded["model_architecture"] as? String, "PBRnxt material height")
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

    func testUnknownNamesNeverInferAnArchitectureOrPredictionRole() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "review-identity-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let manifest = root.appendingPathComponent("comparison.json")
        try writeManifest([
            ["name": "unknown_height", "height": "unknown-height.exr"],
            ["name": "Trained model", "role": "checkpoint", "height": "unknown.exr", "checkpoint": "/runs/misleading-model-name/checkpoint.safetensors"]
        ], to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        let candidates = try XCTUnwrap(store.groups.first?.candidates)
        XCTAssertEqual(candidates[1].role, "map")
        XCTAssertEqual(candidates[1].label, "unknown_height")
        XCTAssertNil(candidates[1].modelIdentity?.architecture)
        XCTAssertNil(candidates[2].modelIdentity?.architecture)
        XCTAssertTrue(candidates[2].detail?.contains("Architecture not recorded") == true)
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
            "checkpoint_step": 123, "model_architecture": "PBRnxt material height"]], to: initial)
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

    func testSourceReferenceProvenanceAndFullOriginalActionSurviveSavingWithoutChangingFiles() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "reference-provenance-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let original = root.appendingPathComponent("full-original-4k-height.png")
        let cropped = root.appendingPathComponent("reference-height.exr")
        let originalBytes = Data("Original numeric source bytes".utf8)
        let cropBytes = Data("Exact Float32 crop bytes".utf8)
        try originalBytes.write(to: original); try cropBytes.write(to: cropped)
        let manifest = root.appendingPathComponent("review-manifest.json")
        try writeManifest([
            ["name": "raw reference", "role": "target", "height": cropped.lastPathComponent,
             "source_path": original.lastPathComponent, "source_sha256": "verified-full-source-sha256",
             "source_url": "https://polyhaven.com/a/white_stucco_02", "source_bits": 16,
             "source_pixel_dimensions": [4096, 4096], "source_crop_rectangle": [120, 256, 1024, 1024],
             "detail": "Only an old freeform description"],
            ["name": "PBRnxt base", "role": "base", "height": "base-height.exr",
             "model_architecture": "Complete PBRnxt", "detail": "This description cannot erase its identity"]], to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        XCTAssertNil(store.error)
        let reference = try XCTUnwrap(store.groups.first?.candidates.first { $0.isReference })
        XCTAssertEqual(reference.label, "Source displacement · reference")
        XCTAssertEqual(reference.mapURL, cropped)
        XCTAssertTrue(reference.detail?.contains("Real source map · not a model output") == true)
        XCTAssertTrue(reference.detail?.contains("16-bit original source") == true)
        XCTAssertTrue(reference.detail?.contains("Full source 4096 × 4096") == true)
        XCTAssertTrue(reference.detail?.contains("x120, y256, 1024 × 1024") == true)
        let full = try XCTUnwrap(reference.fullSourceReference)
        XCTAssertEqual(full.mapURL, original, "The full-source action must use explicit provenance, not a guessed crop or model path")
        XCTAssertEqual(full.role, "target")
        XCTAssertNil(full.fullSourceReference, "An already open full source must not offer itself as a different full image")
        let base = try XCTUnwrap(store.groups.first?.candidates.first { $0.role == "base" })
        XCTAssertTrue(base.detail?.contains("Base model prediction · not the real source map") == true)
        XCTAssertTrue(base.detail?.contains("Complete PBRnxt") == true)
        let saved = root.appendingPathComponent("decisions.json")
        for _ in 0..<3 {
            try store.writeReview(to: saved)
            store.load(saved)
            XCTAssertNil(store.error)
            let restored = try XCTUnwrap(store.groups.first?.candidates.first { $0.isReference })
            XCTAssertEqual(restored, reference, "Saving must preserve provenance without accumulating duplicate labels")
            XCTAssertEqual(restored.fullSourceReference, full)
        }
        XCTAssertEqual(try Data(contentsOf: original), originalBytes)
        XCTAssertEqual(try Data(contentsOf: cropped), cropBytes)
    }

    func testReferenceOmittedFromVariantsIsRecoveredOnlyFromAnExplicitMaterialPath() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "reference-fallback-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let manifest = root.appendingPathComponent("review-manifest.json")
        let document: [String: Any] = ["comparison_target": "height", "materials": [["material_id": "brick",
            "reference_height": "real-reference.exr", "source_path": "original-displacement.png", "source_bits": 16,
            "variants": [["name": "PBRnxt base", "role": "base", "height": "base.exr"]]]]]
        try JSONSerialization.data(withJSONObject: document).write(to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        XCTAssertNil(store.error)
        let candidates = try XCTUnwrap(store.groups.first?.candidates)
        XCTAssertEqual(candidates.map(\.role), ["target", "base"])
        XCTAssertEqual(candidates[0].mapURL, root.appendingPathComponent("real-reference.exr"))
        XCTAssertEqual(candidates[0].fullSourceReference?.mapURL, root.appendingPathComponent("original-displacement.png"))
        try writeManifest([["name": "PBRnxt base", "role": "base", "height": "base.exr"]], to: manifest)
        store.load(manifest)
        XCTAssertFalse(store.groups.flatMap(\.candidates).contains(where: \.isReference), "No explicit target means no invented reference")
    }

    func testExplicitRawReferencePathWinsOverAnEightBitPreview() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "raw-reference-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let manifest = root.appendingPathComponent("review-manifest.json")
        try writeManifest([["name": "target", "role": "target", "height": "display-preview.png", "reference_exr": "raw-reference.exr"]], to: manifest)
        let store = ReviewSessionStore(preferences: preferences)
        store.load(manifest)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.groups.first?.candidates.first { $0.isReference }?.mapURL, root.appendingPathComponent("raw-reference.exr"))
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
