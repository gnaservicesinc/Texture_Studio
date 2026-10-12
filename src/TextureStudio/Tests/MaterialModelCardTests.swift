import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialModelCardTests: XCTestCase {
    func testExportDescribesNamedBarkSpecialistAndKeepsRecordedWeightConfiguration() throws {
        let fixture = try NativeMaterialPackageFixture(); defer { fixture.remove() }
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 19)
        model.configuration["model_name"] = "Pine Bark Height"
        model.configuration["texture_pipeline"] = MaterialModelCard.pipelineName
        model.configuration["training_dataset_name"] = "Bark"
        model.configuration["training_subjects"] = ["Pine Bark"]
        model.configuration["training_configuration"] = ["optimizer": "lion", "learning_rate": 0.0001,
            "optimizer_beta1": 0.9, "optimizer_beta2": 0.99, "weight_decay": 0.01,
            "learning_rate_schedule": "cosine", "gradient_accumulation_steps": 4,
            "randomize_crops": "epoch", "training_map_count": 30, "validation_map_count": 4]
        let configuration = try model.checkpointConfiguration(size: 2048, step: 25)
        let package = fixture.root.appendingPathComponent("Named Bark")
        _ = try NativeMaterialPackage.export(model: model, configuration: configuration, to: package, developer: false)
        let (saved, _) = try NativeMaterialPackage.verify(package)
        XCTAssertEqual(saved["model_name"] as? String, "Pine Bark Height")
        XCTAssertEqual(saved["training_subjects"] as? [String], ["Pine Bark"])
        let readme = try String(contentsOf: package.appendingPathComponent("README.md"), encoding: .utf8)
        XCTAssertTrue(readme.contains("# Pine Bark Height"))
        XCTAssertTrue(readme.contains("surface height / displacement specialist for Pine Bark"))
        XCTAssertTrue(readme.contains("MatterLoom texture pipeline"))
        XCTAssertTrue(readme.contains("| Native training grid | 2048 × 2048 pixels |"))
        XCTAssertTrue(readme.contains("| Optimizer | lion |"))
        XCTAssertTrue(readme.contains("| Gradient accumulation | 4 |"))
        XCTAssertFalse(readme.contains("run-2026"))
    }

    func testLegacyExportUsesOnlyRecordedSettingsAndDepthProDependencyIsClear() throws {
        let legacy = MaterialModelCard.readme(configuration: ["target": "roughness"], familyDescription: "Legacy notices.")
        XCTAssertTrue(legacy.contains("Surface roughness"))
        XCTAssertFalse(legacy.contains("| Optimizer |"))
        XCTAssertFalse(legacy.contains("| Native training grid |"))
        let depth = MaterialModelCard.readme(configuration: ["target": "height", "input_kind": "depthpro",
            "model_name": "Bark | Height\n# injected", "training_subjects": ["<b>Pine</b>"]],
            familyDescription: "Matching upstream weights are not included.")
        XCTAssertTrue(depth.contains("Cached normalized inverse DepthPro depth"))
        XCTAssertTrue(depth.contains("upstream weights are not included"))
        XCTAssertFalse(depth.contains("\n# injected"))
        XCTAssertFalse(depth.contains("<b>"))
    }
}
