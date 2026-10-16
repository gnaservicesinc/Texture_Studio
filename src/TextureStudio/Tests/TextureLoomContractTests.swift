import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class TextureLoomContractTests: XCTestCase {
    func testPackageHashesOriginalBytesAcrossChunkBoundariesAndRepeatedReads() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("hash-boundaries.bin")
        let bytes = Data((0..<(3 * 1_048_576 + 17)).map { UInt8($0 % 251) })
        try bytes.write(to: file)
        let expected = TextureLoomPackage.hash(bytes)
        for _ in 0..<4 { XCTAssertEqual(try TextureLoomPackage.sha256(file), expected) }
        XCTAssertEqual(try Data(contentsOf: file), bytes)
        try Data().write(to: file)
        XCTAssertEqual(try TextureLoomPackage.sha256(file), TextureLoomPackage.hash(Data()))
    }

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
    func testRecipeSelectionRejectsMissingAdaptersAndAvoidsInvalidatingAnUnchangedSelection() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let suite = "loom-selection-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let store = TextureLoomStore(bundleRoot: root, storage: root.appendingPathComponent("LoRAs"), preferences: preferences)
        let revision = store.revision
        try store.setEnabledAdapters([])
        XCTAssertEqual(store.revision, revision)
        XCTAssertThrowsError(try store.setEnabledAdapters([UUID().uuidString]))
        XCTAssertEqual(store.revision, revision)
        XCTAssertNil(preferences.object(forKey: "textureLoom.enabledAdapters.v1"))
        XCTAssertTrue(store.adapters.isEmpty)
    }
    func testRecipeAndMultiAdapterFailuresPreserveTheEnabledSelection() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let suite = "loom-selection-transaction-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let store = TextureLoomStore(bundleRoot: root, storage: root.appendingPathComponent("LoRAs"), preferences: preferences)
        let first = try adapterFixture(base: root, name: "First")
        let second = try adapterFixture(base: root, name: "Second")
        let firstID = try store.importAdapter(first), secondID = try store.importAdapter(second)
        try store.setEnabled(firstID, true)
        let revision = store.revision
        XCTAssertThrowsError(try store.setEnabledAdapters([firstID, secondID]))
        XCTAssertEqual(store.adapters.filter(\.enabled).map(\.id), [firstID])
        XCTAssertEqual(preferences.stringArray(forKey: "textureLoom.enabledAdapters.v1"), [firstID])
        XCTAssertEqual(store.revision, revision)

        let workspace = TextureWorkspace(textureLoom: store)
        let recipe = TextureRecipe(photoPath: root.appendingPathComponent("missing-photo.png").path,
            depthChoice: .textureLoom, settings: TextureSettings(), textureLoomID: store.package?.manifest.id,
            textureLoomVersion: store.package?.manifest.version, enabledLoRAs: [secondID])
        let recipeURL = root.appendingPathComponent("missing-photo.texture.json")
        try JSONEncoder().encode(recipe).write(to: recipeURL)
        workspace.openRecipe(recipeURL)
        for _ in 0..<1000 where workspace.isBusy { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(workspace.isBusy)
        XCTAssertNotNil(workspace.notice)
        XCTAssertEqual(store.adapters.filter(\.enabled).map(\.id), [firstID])
        XCTAssertEqual(store.revision, revision)

        // A later validation failure must also leave the prior selection intact.
        let weights = root.appendingPathComponent("LoRAs/\(secondID)/lora.safetensors")
        try Data("corrupt".utf8).write(to: weights)
        XCTAssertThrowsError(try store.setEnabledAdapters([secondID]))
        XCTAssertEqual(store.adapters.filter(\.enabled).map(\.id), [firstID])
        XCTAssertEqual(preferences.stringArray(forKey: "textureLoom.enabledAdapters.v1"), [firstID])
        XCTAssertEqual(store.revision, revision)
    }
    private func adapterFixture(base: URL, name: String) throws -> URL {
        let folder = base.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        // Core ML spec v7: one ReLU from Float32 NCHW rgb to material_map.
        // Only the interface is used; this is not trained material-model evidence.
        let model = try XCTUnwrap(Data(base64Encoded: "CAcSMwoTCgNyZ2IaDCoKCgQBAyAgEKCABFIcCgxtYXRlcmlhbF9tYXAaDCoKCgQBAyAgEKCABKIfLAomCgxmaXh0dXJlX3JlbHUSA3JnYhoMbWF0ZXJpYWxfbWFwkggCUgAoATAB"))
        let weights = Data("adapter weights fixture".utf8)
        try model.write(to: folder.appendingPathComponent("adapter.mlmodel"))
        try weights.write(to: folder.appendingPathComponent("lora.safetensors"))
        let package = try TextureLoomPackage(root: base)
        let document: [String: Any] = ["schema": "texture-loom.adapter.v1", "id": UUID().uuidString,
            "name": name, "version": "1", "baseModel": ["id": package.manifest.id, "version": package.manifest.version],
            "categories": [["id": "bricks", "version": "1", "adapters": [["map": "depth", "format": "coreml",
                "path": "adapter.mlmodel", "weightsPath": "lora.safetensors",
                "baseSHA256": try package.modelHash(package.manifest.categories[0].models[0])]]]],
            "files": [["path": "adapter.mlmodel", "sha256": TextureLoomPackage.hash(model)],
                      ["path": "lora.safetensors", "sha256": TextureLoomPackage.hash(weights)]]]
        try JSONSerialization.data(withJSONObject: document).write(to: folder.appendingPathComponent("manifest.json"))
        return folder
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
