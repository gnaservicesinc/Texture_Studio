import CoreImage
import Metal
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialModelMapsTests: XCTestCase {
    func testTargetRegistriesKeepThreeIndependentSelectedModels() throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        for target in ["height", "roughness", "normal"] {
            let checkpoint = selected(target: target, root: root)
            try checkpoint.save(to: SelectedMaterialCheckpoint.registryURL(for: target, heightRegistryURL: registry))
        }
        let models = SelectedMaterialCheckpoint.readAll(heightRegistryURL: registry)
        XCTAssertEqual(Set(models.keys), Set(["height", "roughness", "normal"]))
        XCTAssertEqual(models["height"]?.checkpointPath, root.appendingPathComponent("height.safetensors").path)
        XCTAssertEqual(models["normal"]?.target, "normal")
        XCTAssertEqual(models["roughness"]?.target, "roughness")
    }

    func testSelectedRoughnessAndNormalRunSequentiallyOnIdenticalPreparedDiffuseAndCacheAllModels() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        for target in ["roughness", "normal"] {
            let checkpoint = selected(target: target, root: root)
            try Data(target.utf8).write(to: URL(fileURLWithPath: checkpoint.checkpointPath))
            try checkpoint.save(to: SelectedMaterialCheckpoint.registryURL(for: target, heightRegistryURL: registry))
        }
        var targets: [String] = []
        var inputs: [ObjectIdentifier] = []
        let workspace = TextureWorkspace(checkpointRegistryURL: registry, mapPredictor: { diffuse, checkpoint, size in
            targets.append(checkpoint.target); inputs.append(ObjectIdentifier(diffuse))
            return Self.prediction(checkpoint.target, size: size)
        })
        workspace.source = TextureSource(url: root.appendingPathComponent("photo.png"),
            orientedImage: CIImage(color: CIColor(red: 0.24, green: 0.31, blue: 0.19)).cropped(to: CGRect(x: 0, y: 0, width: 1024, height: 1024)),
            camera: CameraMetadata(), pixelWidth: 1024, pixelHeight: 1024)
        workspace.settings.outputSize = 1024
        workspace.settings.useSupportingViews = false
        workspace.settings.lightingStrength = 0
        workspace.settings.roughnessBase = 0.1
        workspace.settings.roughnessDetail = 1
        let manager = ModelManager(storageDirectory: root.appendingPathComponent("models"))
        workspace.updatePreview(models: manager)
        try await finish(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(targets, ["roughness", "normal"])
        XCTAssertEqual(inputs.first, inputs.last, "All selected material models consume the very same prepared diffuse image")
        let material = try XCTUnwrap(workspace.result)
        XCTAssertEqual(Self.sample(material.roughness, format: .Rf, components: 1), [Float(0.7312349)])
        XCTAssertEqual(Self.sample(material.normal, format: .RGBAf, components: 4), [Float(0.13), Float(0.62), Float(0.9), Float(1)])
        XCTAssertFalse(workspace.materialNeedsUpdate)
        workspace.updatePreview(models: manager)
        try await finish(workspace)
        XCTAssertEqual(targets.count, 2, "Unchanged maps are reused")
        let normal = selected(target: "normal", root: root, hash: String(repeating: "d", count: 64))
        try normal.save(to: SelectedMaterialCheckpoint.registryURL(for: "normal", heightRegistryURL: registry))
        XCTAssertTrue(workspace.materialNeedsUpdate, "Changing any map model invalidates the material cache")
        let recipe = try workspace.makeRecipe()
        XCTAssertEqual(Set(recipe.materialMapCheckpoints?.keys ?? Dictionary<String, MaterialCheckpointIdentity>().keys), Set(["roughness", "normal"]))
        XCTAssertEqual(recipe.materialMapCheckpoints?["normal"]?.sha256, normal.sha256)
    }

    func testModelGridMismatchFailsInsteadOfResizingPrediction() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal unavailable") }
        let engine = TextureEngine()
        let source = TextureSource(url: URL(fileURLWithPath: "/photo.png"),
            orientedImage: CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 1024, height: 1024)),
            camera: CameraMetadata(), pixelWidth: 1024, pixelHeight: 1024)
        var settings = TextureSettings(); settings.outputSize = 1024; settings.useSupportingViews = false
        do {
            _ = try await engine.process(source: source, settings: settings, modelMaps: ["normal": Self.prediction("normal", size: 512)])
            XCTFail("A mismatched prediction was silently resized")
        } catch TextureError.processing { }
    }

    private func selected(target: String, root: URL, hash: String = String(repeating: "c", count: 64)) -> SelectedMaterialCheckpoint {
        SelectedMaterialCheckpoint(checkpointPath: root.appendingPathComponent(target + ".safetensors").path,
            sha256: hash, target: target, pythonPath: "/bin/sh", workspacePath: root.path,
            modelDirectory: root.appendingPathComponent("base").path, codeDirectory: root.appendingPathComponent("code").path)
    }
    private static func prediction(_ target: String, size: Int) -> MaterialModelMap {
        let color = target == "normal" ? CIColor(red: 0.13, green: 0.62, blue: 0.9, alpha: 1)
            : CIColor(red: 0.7312349, green: 0.7312349, blue: 0.7312349, alpha: 1)
        return MaterialModelMap(target: target, image: CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: size, height: size)), sourceLabel: "Fixture \(target)")
    }
    private static func sample(_ image: CIImage, format: CIFormat, components: Int) -> [Float] {
        var samples = [Float](repeating: 0, count: components)
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        samples.withUnsafeMutableBytes { context.render(image, toBitmap: $0.baseAddress!, rowBytes: components * 4,
            bounds: CGRect(x: 20, y: 20, width: 1, height: 1), format: format, colorSpace: nil) }
        return samples
    }
    private func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-maps-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    private func finish(_ workspace: TextureWorkspace) async throws {
        for _ in 0..<2000 {
            if !workspace.isBusy { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Material generation did not finish")
    }
}
