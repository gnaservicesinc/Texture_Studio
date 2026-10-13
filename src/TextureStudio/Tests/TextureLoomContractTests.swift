import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class TextureLoomContractTests: XCTestCase {
    func testPackageRejectsOutsidePathsAndExternalLinks() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for path in ["../model.bin", "/tmp/model.bin", "models//weights", "models/./weights", "models\\weights"] {
            XCTAssertThrowsError(try TextureLoomPackage.resolve(path, root: root), path)
        }
        try FileManager.default.createSymbolicLink(atPath: root.appendingPathComponent("outside").path, withDestinationPath: "/tmp")
        XCTAssertThrowsError(try TextureLoomPackage.resolve("outside/model.bin", root: root))
    }
    func testValidBuildIdentityReadsOnlyItsManifestAndInventory() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let package = try TextureLoomPackage(root: root)
        XCTAssertEqual(package.manifest.categories.map(\.name), ["Bricks"])
        XCTAssertEqual(package.manifest.categories[0].models.map(\.map), ["depth"])
    }
    func testChangedWeightsCannotEnterBuiltInPackage() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        try Data("changed".utf8).write(to: root.appendingPathComponent("weights.safetensors"))
        XCTAssertThrowsError(try TextureLoomPackage(root: root))
    }
    func testUnregisteredRuntimeWeightsCannotHideBehindAPartialInventory() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        try Data("unregistered weights".utf8).write(to: root.appendingPathComponent("depth.mlmodelc/extra.weights"))
        XCTAssertThrowsError(try TextureLoomPackage(root: root))
    }
    func testClassifierCannotReferToAnAbsentDataset() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        try changeManifest(root) { document in
            var classifier = document["classifier"] as! [String: Any]
            classifier["labels"] = [["id": "wood", "name": "Wood"]]; document["classifier"] = classifier
        }
        XCTAssertThrowsError(try TextureLoomPackage(root: root))
    }
    func testClassifierWeightsAreOptionalButMaterialWeightsAreRequired() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        try changeManifest(root) { document in
            var classifier = document["classifier"] as! [String: Any]
            classifier.removeValue(forKey: "weightsPath"); document["classifier"] = classifier
        }
        XCTAssertNil(try TextureLoomPackage(root: root).manifest.classifier.weightsPath)
        try changeManifest(root) { document in
            var categories = document["categories"] as! [[String: Any]]
            var models = categories[0]["models"] as! [[String: Any]]
            models[0].removeValue(forKey: "weightsPath"); categories[0]["models"] = models; document["categories"] = categories
        }
        XCTAssertThrowsError(try TextureLoomPackage(root: root))
    }
    func testOldExternalSelectionsCannotReplaceTheBuiltInBase() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let suite = "texture-loom-contract-\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        preferences.set("/old/material/checkpoint.safetensors", forKey: "selectedCheckpoint")
        preferences.set("/old/specialists.json", forKey: "specialistWorkspace")
        let store = TextureLoomStore(bundleRoot: root, storage: root.appendingPathComponent("LoRAs"), preferences: preferences)
        XCTAssertEqual(store.package?.manifest.name, "Texture Loom Fixture")
        XCTAssertTrue(store.adapters.isEmpty)
    }
    func testWeightsOnlyAdapterIsRejectedBeforeInstallation() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let package = try TextureLoomPackage(root: root)
        let adapter = try temporary(); defer { try? FileManager.default.removeItem(at: adapter) }
        let weight = Data("adapter".utf8); try weight.write(to: adapter.appendingPathComponent("lora.safetensors"))
        let document: [String: Any] = ["schema": "texture-loom.adapter.v1", "id": UUID().uuidString, "name": "Bricks LoRA", "version": "1",
            "baseModel": ["id": package.manifest.id, "version": package.manifest.version],
            "categories": [["id": "bricks", "version": "1", "adapters": [["map": "depth", "format": "safetensors", "path": "lora.safetensors", "weightsPath": "lora.safetensors", "baseSHA256": try package.modelHash(package.manifest.categories[0].models[0])]]]],
            "files": [["path": "lora.safetensors", "sha256": TextureLoomPackage.hash(weight)]]]
        try JSONSerialization.data(withJSONObject: document).write(to: adapter.appendingPathComponent("manifest.json"))
        let store = TextureLoomStore(bundleRoot: root, storage: root.appendingPathComponent("LoRAs"))
        XCTAssertThrowsError(try store.importAdapter(adapter))
        XCTAssertTrue(store.adapters.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("LoRAs").path))
    }
    func testMissingBuildPayloadDoesNotAdoptAnExternalModel() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let store = TextureLoomStore(bundleRoot: root, storage: root.appendingPathComponent("LoRAs"))
        XCTAssertNil(store.package)
        XCTAssertTrue(store.statusDescription.contains("Loom Factory"))
        XCTAssertThrowsError(try store.activeOverrides())
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-contract-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true); return root
    }
    private func fixture() throws -> URL {
        let root = try temporary()
        let contents = ["classifier.mlmodelc/fixture": Data("classifier identity fixture".utf8),
            "depth.mlmodelc/fixture": Data("model identity fixture".utf8), "weights.safetensors": Data("base identity fixture".utf8)]
        for (name, data) in contents {
            let path = root.appendingPathComponent(name); try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: path)
        }
        // Contract-only fixture: no inference or learned-quality claim is made by these bytes.
        let document: [String: Any] = ["schema": "texture-loom.package.v1", "id": UUID().uuidString, "name": "Texture Loom Fixture", "version": "1",
            "classifier": ["format": "coreml", "path": "classifier.mlmodelc", "weightsPath": "weights.safetensors", "architecture": "fixture", "labels": [["id": "bricks", "name": "Bricks"]]],
            "categories": [["id": "bricks", "name": "Bricks", "version": "1", "models": [["map": "depth", "format": "coreml", "path": "depth.mlmodelc", "weightsPath": "weights.safetensors", "architecture": "fixture"]]]],
            "files": contents.map { ["path": $0.key, "sha256": TextureLoomPackage.hash($0.value)] }]
        try JSONSerialization.data(withJSONObject: document).write(to: root.appendingPathComponent("manifest.json")); return root
    }
    private func changeManifest(_ root: URL, _ change: (inout [String: Any]) -> Void) throws {
        let url = root.appendingPathComponent("manifest.json")
        var document = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        change(&document); try JSONSerialization.data(withJSONObject: document).write(to: url)
    }
}
