import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class DepthProTrainingSelectionTests: XCTestCase {
    func testRefinerSelectionPersistsAndTargetChangesRespectItsHeightContract() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.selectTrainingTarget("roughness")
        store.selectTrainingModelFamily(.depthProHeightRefiner)
        XCTAssertEqual(store.training.modelFamily, .depthProHeightRefiner)
        XCTAssertEqual(store.training.target, "height")
        XCTAssertEqual(store.training.effectiveScope, "full-model")
        XCTAssertEqual(store.training.learningRate, 0.001)
        store.selectTrainingTarget("height")
        XCTAssertEqual(store.training.modelFamily, .depthProHeightRefiner,
            "Choosing height must preserve the depth-to-height input contract")

        let reopened = fixture.store()
        XCTAssertEqual(reopened.training, store.training)
        XCTAssertEqual(reopened.training.modelFamily, .depthProHeightRefiner)
        reopened.selectTrainingTarget("roughness")
        XCTAssertEqual(reopened.training.modelFamily, .compactScalar)
        XCTAssertEqual(reopened.training.target, "roughness")
        XCTAssertEqual(fixture.store().training.modelFamily, .compactScalar)
    }

    func testSuggestedNameDistinguishesDepthRefinerFromRGBHeightModel() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
        {"dataset_path":"/dataset","index_sha256":"index","name":"Stone holdouts","materials":[]}
        """)
        store.selectTrainingModelFamily(.depthProHeightRefiner)
        XCTAssertEqual(store.suggestedTrainingModelName, "Stone holdouts DepthPro Height Refiner")
        XCTAssertEqual(store.effectiveTrainingModelName, store.suggestedTrainingModelName)
        store.selectTrainingModelFamily(.compactScalar)
        XCTAssertEqual(store.suggestedTrainingModelName, "Stone holdouts Displacement")
    }

    func testDepthProModelLocationPersistsAndSuppliesRefinerDependencies() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let model = fixture.root.appendingPathComponent("My DepthPro/DepthPro.mlpackage").path
        store.depthProModelPath = model
        let reopened = fixture.store()
        XCTAssertEqual(reopened.depthProModelPath, model)
        let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
        {"checkpoint_path":"/models/refiner/model.safetensors","sha256":"refiner-sha","schema":"texture-studio-compact-material-v1","architecture":"texture-studio-depthpro-height-refiner-native-v1","target":"height","step":42,"compatible":true}
        """)
        XCTAssertEqual(checkpoint.modelFamily, .depthProHeightRefiner)
        XCTAssertEqual(reopened.dependencyArguments(for: checkpoint), ["--depthpro-model", model])
    }

    func testNativeCapabilitiesAndOptionsRejectNonHeightRefinerTargets() async throws {
        let output = try await NativeMaterialCommands.run(arguments: ["capabilities", "--model-family", "depthpro-height-refiner"],
            onEvent: { _ in }, control: .init())
        let capabilities = try NativeMaterialTransfer.object(Data(output.utf8))
        XCTAssertEqual(capabilities["targets"] as? [String], ["height"])
        XCTAssertEqual(capabilities["model_family"] as? String, "depthpro-height-refiner")
        XCTAssertEqual(capabilities["scope"] as? String, "full-model")
        let families = try XCTUnwrap(capabilities["model_families"] as? [[String: Any]])
        let refiner = try XCTUnwrap(families.first { $0["id"] as? String == "depthpro-height-refiner" })
        XCTAssertEqual(refiner["targets"] as? [String], ["height"])
        XCTAssertEqual(refiner["input_channels"] as? Int, 1)
        XCTAssertEqual(refiner["output_channels"] as? Int, 1)
        XCTAssertEqual(refiner["depthpro_required"] as? Bool, true)

        var options = MaterialTrainingOptions()
        options.modelFamily = .depthProHeightRefiner
        XCTAssertNil(options.configurationIssue)
        for target in ["normal", "roughness"] {
            options.target = target
            XCTAssertNotNil(options.configurationIssue,
                "The refiner must reject \(target) instead of accepting an incompatible head")
        }
    }

    @MainActor private struct PreferencesFixture {
        let suite: String
        let defaults: UserDefaults
        let root: URL

        init() throws {
            suite = "depthpro-training-selection-\(UUID().uuidString)"
            defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            root = FileManager.default.temporaryDirectory.appendingPathComponent(suite)
        }

        func store() -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, managedWorkspaceURL: root,
                selectedCheckpointRegistryURL: root.appendingPathComponent("selection.json"))
        }

        func remove() {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }
    }
}
