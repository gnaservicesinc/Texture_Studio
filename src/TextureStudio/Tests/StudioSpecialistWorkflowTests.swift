import CoreImage
import XCTest
@testable import TextureStudio

@MainActor final class StudioSpecialistWorkflowTests: XCTestCase {
    func testClassifierRoutesAllMapsWithSharedPreparedDiffuseAndPinsRecipe() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        var routed = 0, predicted = 0
        var mapTargets: [String] = []
        let selections = Dictionary(uniqueKeysWithValues: ["height", "normal", "roughness"].map { target in
            (target, SelectedMaterialCheckpoint(checkpointPath: root.appendingPathComponent(target + ".safetensors").path,
                sha256: target + "-sha", target: target, workspacePath: root.path, modelDirectory: "", displayName: "Wood " + target))
        })
        let classification = NativeDiffuseTextureClassifier.Prediction(label: "Wood", confidence: 0.95,
            probabilities: ["Wood": 0.95, "Bark": 0.05], modelSHA256: "classifier-sha")
        let workspace = TextureWorkspace(checkpointRegistryURL: root.appendingPathComponent("absent.json"),
            checkpointPredictor: { input, checkpoint, size in
                predicted += 1
                XCTAssertEqual(input.orientedImage.extent.size, CGSize(width: 256, height: 256))
                XCTAssertEqual(checkpoint.target, "height")
                XCTAssertEqual(size, 256)
                return try TextureDepth(width: 2, height: 2, values: [0.4,0.5,0.6,0.7], sourceLabel: "Wood height", interpretation: .surfaceHeight)
            }, materialProcessor: { source, settings, _ in Self.material(source, settings: settings) },
            mapPredictor: { image, checkpoint, size in
                XCTAssertEqual(image.extent.size, CGSize(width: 256, height: 256))
                XCTAssertEqual(size, 256)
                mapTargets.append(checkpoint.target)
                return MaterialModelMap(target: checkpoint.target, image: image, sourceLabel: checkpoint.title)
            }, specialistRouter: { image in
                XCTAssertEqual(image.extent.size, CGSize(width: 256, height: 256))
                routed += 1
                return .init(prediction: classification, minimumConfidence: 0.7, checkpoints: selections, notices: [])
            })
        workspace.source = source(root.appendingPathComponent("diffuse.png"))
        workspace.depthChoice = .specialists
        workspace.settings.outputSize = 256
        let models = ModelManager()
        workspace.updatePreview(models: models)
        try await wait(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(routed, 1)
        XCTAssertEqual(predicted, 1)
        XCTAssertEqual(mapTargets, ["roughness", "normal"])
        XCTAssertEqual(workspace.activeMaterialModels.count, 3)
        XCTAssertFalse(workspace.materialNeedsUpdate)
        workspace.updatePreview(models: models)
        try await wait(workspace)
        XCTAssertEqual(routed, 1, "Preview reuses the exact generated material and classifier decision")
        let recipe = try workspace.makeRecipe()
        XCTAssertEqual(recipe.specialistModels?["height"]?.sha256, "height-sha")
        XCTAssertEqual(recipe.specialistClassification?.label, "Wood")
        let decoded = try JSONDecoder().decode(TextureRecipe.self, from: JSONEncoder().encode(recipe))
        XCTAssertEqual(decoded.depthChoice, .specialists)
        XCTAssertEqual(decoded.specialistModels?["normal"]?.sha256, "normal-sha")
    }

    func testMissingSpecialistUsesExplicitFlatFallbackAndReportsDecision() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        var depthWasNil = false
        let workspace = TextureWorkspace(checkpointRegistryURL: root.appendingPathComponent("absent.json"),
            materialProcessor: { source, settings, depth in
                depthWasNil = depth == nil
                return Self.material(source, settings: settings)
            }, specialistRouter: { _ in
                .init(prediction: .init(label: "Paper", confidence: 0.9, probabilities: ["Paper": 0.9], modelSHA256: "classifier-sha"),
                    minimumConfidence: 0.7, checkpoints: [:], notices: ["No approved Paper specialist."])
            })
        let oldFallback = workspace.specialistFallback
        defer { workspace.specialistFallback = oldFallback }
        workspace.source = source(root.appendingPathComponent("paper.png"))
        workspace.settings.outputSize = 256
        workspace.depthChoice = .specialists
        workspace.specialistFallback = "flat"
        workspace.updatePreview(models: ModelManager())
        try await wait(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertTrue(depthWasNil)
        XCTAssertEqual(workspace.heightSourceNotice, "No approved Paper specialist.")
        XCTAssertEqual(workspace.specialistClassification?.label, "Paper")
        XCTAssertTrue(workspace.activeMaterialModels.isEmpty)
    }

    private static func material(_ source: TextureSource, settings: TextureSettings) -> MaterialResult {
        let extent = CGRect(x: 0, y: 0, width: settings.outputSize, height: settings.outputSize)
        let gray = CIImage(color: .gray).cropped(to: extent)
        return MaterialResult(diffuse: gray, roughness: gray,
            normal: CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 1)).cropped(to: extent), height: gray,
            crop: source.orientedImage.extent, warnings: [], outputSize: settings.outputSize, depthOrigin: "Fixture",
            settings: settings, sourceURL: source.url, camera: source.camera)
    }
    private func source(_ url: URL) -> TextureSource {
        .init(url: url, orientedImage: CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64)),
            camera: CameraMetadata(), pixelWidth: 64, pixelHeight: 64)
    }
    private func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("studio-specialists-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    private func wait(_ workspace: TextureWorkspace) async throws {
        for _ in 0..<3000 where workspace.isBusy { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(workspace.isBusy)
    }
}
