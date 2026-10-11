import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialCheckpointProvenanceTests: XCTestCase {
    func testPBRCheckpointRetainsItsEstablishedExportMetadata() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let file = fixture.root.appendingPathComponent("pbr.safetensors")
        let configuration: [String: Any] = ["schema": "texture-studio-material-checkpoint-v1",
            "architecture": "pbrnxt-native-v1", "target": "height", "step": 12, "training_size": 1024,
            "image_padding": false, "image_resizing": false, "base": ["sha256": String(repeating: "a", count: 64)]]
        try NativeSafetensors.write(tensors: ["weight": .floats([1], shape: [1])],
            metadata: ["configuration": NativeMaterialTransfer.json(configuration)], to: file)
        let selected = try fixture.select(file)
        let metadata = try fixture.metadata()
        XCTAssertEqual(metadata.count, 8, "PBRnxt retains its established provenance fields")
        XCTAssertEqual(metadata["backend"] as? String, "Apple MPSGraph / Metal")
        XCTAssertEqual(metadata["model_type"] as? String, "PBRnxt material refinement")
        XCTAssertEqual(metadata["input"] as? String, "Prepared diffuse map shared with material rendering")
        XCTAssertEqual(metadata["checkpoint_sha256"] as? String, selected.sha256)
        XCTAssertEqual(metadata["height_interpretation"] as? String, "surface height; no camera-depth normalization")
    }

    func testRGBCompactExportUsesItsValidatedFamilyAndChannelLabels() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let file = try fixture.compactCheckpoint(family: .compactScalar)
        _ = try fixture.select(file)
        let metadata = try fixture.metadata()
        XCTAssertEqual(metadata["model_type"] as? String, MaterialTrainingModelFamily.compactScalar.label)
        XCTAssertEqual(metadata["model_family"] as? String, "compact-scalar")
        XCTAssertEqual(metadata["architecture"] as? String, NativeCompactMaterialModel.scalarArchitecture)
        XCTAssertEqual(metadata["input_channels"] as? Int, 3)
        XCTAssertEqual(metadata["output_channels"] as? Int, 1)
        XCTAssertEqual(metadata["backend"] as? String, "Apple MPSGraph / Metal")
        XCTAssertNil(metadata["depthpro_model_sha256"])
    }

    func testRefinerExportRecordsTwoStagesAndDepthProIdentityWithoutFalseNoResize() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let file = try fixture.compactCheckpoint(family: .depthProHeightRefiner)
        let selected = try fixture.select(file)
        let workspace = fixture.workspace()
        workspace.settings.outputSize = 256
        let metadata = try NativeMaterialTransfer.object(workspace.materialCheckpointProvenance())
        XCTAssertEqual(metadata["model_type"] as? String, "DepthPro Height Refiner")
        XCTAssertEqual(metadata["backend"] as? String, "Core ML DepthPro followed by Apple MPSGraph / Metal")
        XCTAssertEqual(metadata["checkpoint_sha256"] as? String, selected.sha256)
        XCTAssertEqual(metadata["input_channels"] as? Int, 1)
        XCTAssertEqual(metadata["output_channels"] as? Int, 1)
        XCTAssertEqual(metadata["depthpro_model_sha256"] as? String, fixture.depthProHash)
        XCTAssertEqual(metadata["depthpro_preparation_contract"] as? String, NativeDepthProContract.identifier)
        XCTAssertEqual(metadata["prepared_diffuse_dimensions"] as? [Int], [256, 256])
        XCTAssertEqual(metadata["depthpro_input_dimensions"] as? [Int], [1536, 1536])
        XCTAssertEqual(metadata["depthpro_output_dimensions"] as? [Int], [1536, 1536])
        XCTAssertEqual(metadata["refiner_input_dimensions"] as? [Int], [256, 256])
        XCTAssertEqual(metadata["refiner_output_dimensions"] as? [Int], [256, 256])
        XCTAssertEqual(metadata["depthpro_original_width"] as? Int, 256)
        XCTAssertEqual(metadata["source_pixels_resized"] as? Bool, true)
        XCTAssertEqual(metadata["refiner_input_resized"] as? Bool, false)
        XCTAssertEqual(metadata["source_pixels_discarded"] as? Bool, false)
        XCTAssertTrue((metadata["input"] as? String)?.contains("normalized inverse DepthPro") == true)
    }

    func testProvenanceRejectsChangedCheckpointAndMismatchedSelectedTarget() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let file = try fixture.compactCheckpoint(family: .depthProHeightRefiner)
        let selected = try fixture.select(file)
        let bytes = try Data(contentsOf: file)
        var changed = bytes
        changed[changed.count - 1] ^= 1
        try changed.write(to: file)
        XCTAssertThrowsError(try fixture.metadata(), "Metadata must validate the selected exact checkpoint hash")
        try bytes.write(to: file)
        let invalidSelection = SelectedMaterialCheckpoint(checkpointPath: selected.checkpointPath,
            sha256: selected.sha256, target: "roughness", workspacePath: selected.workspacePath,
            modelDirectory: selected.modelDirectory)
        try invalidSelection.save(to: fixture.registry)
        XCTAssertThrowsError(try fixture.metadata(), "The selected target must match the validated checkpoint")
    }

    @MainActor private struct Fixture {
        let root: URL
        let registry: URL
        let depthProHash = String(repeating: "b", count: 64)

        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("material-provenance-\(UUID().uuidString)")
            registry = root.appendingPathComponent("selected.json")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }

        func compactCheckpoint(family: MaterialTrainingModelFamily) throws -> URL {
            let model = try NativeCompactMaterialModel(target: "height", width: 16, family: family)
            if family == .depthProHeightRefiner { model.configuration["depthpro_model_sha256"] = depthProHash }
            let file = root.appendingPathComponent(family.rawValue + ".safetensors")
            let configuration = try model.checkpointConfiguration(size: 256, step: 3)
            try NativeSafetensors.write(tensors: model.weights,
                metadata: ["configuration": NativeMaterialTransfer.json(configuration)], to: file)
            return file
        }

        func select(_ file: URL) throws -> SelectedMaterialCheckpoint {
            let selected = SelectedMaterialCheckpoint(checkpointPath: file.path,
                sha256: try NativeMaterialTransfer.hash(file), target: "height", workspacePath: root.path,
                modelDirectory: root.appendingPathComponent("missing-pbr-base").path)
            try selected.save(to: registry)
            return selected
        }

        func workspace() -> TextureWorkspace { TextureWorkspace(checkpointRegistryURL: registry) }
        func metadata() throws -> [String: Any] {
            try NativeMaterialTransfer.object(workspace().materialCheckpointProvenance())
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
    }
}
