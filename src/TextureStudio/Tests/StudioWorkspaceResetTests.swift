import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class StudioWorkspaceResetTests: XCTestCase {
    func testOpeningAdapterWithoutABaseReportsTheModelRequirementRatherThanImportingAPhoto() {
        let workspace = TextureWorkspace(textureLoom: TextureLoomStore(bundleRoot: nil))
        workspace.openDocument(URL(fileURLWithPath: "/fixture/MyTexture.loomlora"))
        XCTAssertTrue(workspace.showModels)
        XCTAssertEqual(workspace.notice?.title, "Import LoRA")
        XCTAssertTrue(workspace.notice?.message.contains("Loom Factory") == true)
        XCTAssertNil(workspace.source)
        XCTAssertFalse(workspace.isBusy)
    }
    func testGeneratedMapsAreReusedAndChosenResolutionStays2048() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("studio-workspace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = TextureSource(url: root.appendingPathComponent("photo.png"),
            orientedImage: CIImage(color: CIColor(red: 0.4, green: 0.5, blue: 0.6)).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16)),
            camera: CameraMetadata(), pixelWidth: 16, pixelHeight: 16)
        var operations = 0
        let workspace = TextureWorkspace(textureLoom: TextureLoomStore(bundleRoot: nil, storage: root.appendingPathComponent("LoRAs")), materialProcessor: { source, settings, _ in
            operations += 1
            let extent = CGRect(x: 0, y: 0, width: settings.outputSize, height: settings.outputSize)
            let scalar = CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5)).cropped(to: extent)
            return MaterialResult(diffuse: source.orientedImage.transformed(by: CGAffineTransform(scaleX: 128, y: 128)), roughness: scalar,
                normal: CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 1)).cropped(to: extent), height: scalar,
                crop: source.orientedImage.extent, warnings: [], outputSize: settings.outputSize, depthOrigin: "Flat surface fixture",
                settings: settings, sourceURL: source.url, camera: source.camera)
        })
        workspace.source = source; workspace.depthChoice = .photoDetail; workspace.settings.outputSize = 2048
        workspace.updatePreview(); try await finish(workspace)
        XCTAssertNil(workspace.notice); XCTAssertEqual(operations, 1); XCTAssertFalse(workspace.materialNeedsUpdate)
        let map = try XCTUnwrap(workspace.cachedMaterialMapURL(.height))
        XCTAssertEqual(try XCTUnwrap(CIImage(contentsOf: map)).extent.size, CGSize(width: 2048, height: 2048))
        workspace.settings.exrPrecision = .float16
        XCTAssertFalse(workspace.materialNeedsUpdate)
        workspace.export(to: root.appendingPathComponent("Export"), reveal: false); try await finish(workspace)
        XCTAssertNil(workspace.notice); XCTAssertEqual(operations, 1); XCTAssertEqual(workspace.settings.outputSize, 2048)
        let recipe = try workspace.makeRecipe()
        XCTAssertEqual(recipe.version, 4); XCTAssertNil(recipe.textureLoomID); XCTAssertTrue(recipe.enabledLoRAs.isEmpty)
        let document = try JSONSerialization.jsonObject(with: JSONEncoder().encode(recipe)) as! [String: Any]
        XCTAssertNil(document["materialCheckpoint"]); XCTAssertNil(document["specialistModels"]); XCTAssertNil(document["modelID"])
    }
    func testMissingBundledModelProducesNoFabricatedMapsOrExternalFallback() async throws {
        let source = TextureSource(url: URL(fileURLWithPath: "/fixture/photo.png"),
            orientedImage: CIImage(color: CIColor(red: 0.4, green: 0.5, blue: 0.6)).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16)),
            camera: CameraMetadata(), pixelWidth: 16, pixelHeight: 16)
        let workspace = TextureWorkspace(textureLoom: TextureLoomStore(bundleRoot: nil))
        workspace.source = source; workspace.updatePreview(); try await finish(workspace)
        XCTAssertNil(workspace.result); XCTAssertTrue(workspace.materialNeedsUpdate)
        XCTAssertTrue(workspace.notice?.message.contains("Loom Factory") == true)
        XCTAssertEqual(workspace.depthChoice, .textureLoom)
    }
    func testOpeningRecipeClearsThePreviousPhotosReviewAndExport() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("studio-recipe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let photo = root.appendingPathComponent("new-photo.png")
        let image = CIImage(color: CIColor(red: 0.4, green: 0.5, blue: 0.6)).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16))
        let context = CIContext(options: [.useSoftwareRenderer: true])
        try context.writePNGRepresentation(of: image, to: photo, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let workspace = TextureWorkspace(textureLoom: TextureLoomStore(bundleRoot: nil, storage: root.appendingPathComponent("LoRAs")))
        workspace.decision = previousDecision()
        workspace.exportURL = root.appendingPathComponent("previous-export")
        workspace.warnings = ["Previous photo warning"]
        workspace.hasEdits = true
        let recipe = TextureRecipe(photoPath: photo.path, depthChoice: .photoDetail, settings: TextureSettings(), enabledLoRAs: [])
        let recipeURL = root.appendingPathComponent("new.texture.json")
        try JSONEncoder().encode(recipe).write(to: recipeURL)

        workspace.openRecipe(recipeURL); try await finish(workspace)

        XCTAssertNil(workspace.notice)
        XCTAssertEqual(workspace.source?.url, photo)
        XCTAssertEqual(workspace.recipeURL, recipeURL)
        XCTAssertNil(workspace.decision, "A review of the previous photo must never be exported with the new photo")
        XCTAssertNil(workspace.exportURL)
        XCTAssertTrue(workspace.warnings.isEmpty)
        XCTAssertFalse(workspace.hasEdits)
    }
    func testFailedRecipeLoadPreservesTheCurrentDocumentAndPreferences() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("studio-recipe-failure-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "studio-recipe-failure-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let store = TextureLoomStore(bundleRoot: nil, storage: root.appendingPathComponent("LoRAs"), preferences: preferences)
        let workspace = TextureWorkspace(preferences: preferences, textureLoom: store)
        workspace.source = TextureSource(url: root.appendingPathComponent("previous.png"),
            orientedImage: CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16)),
            camera: CameraMetadata(), pixelWidth: 16, pixelHeight: 16)
        workspace.depthChoice = .photoDetail
        workspace.settings.roughnessBase = 0.75
        workspace.decision = previousDecision()
        workspace.exportURL = root.appendingPathComponent("previous-export")
        workspace.recipeURL = root.appendingPathComponent("previous.texture.json")
        let saved = preferences.data(forKey: StudioPreferences.key)
        let decision = workspace.decision
        let revision = store.revision
        let recipe = TextureRecipe(photoPath: root.appendingPathComponent("missing.png").path,
            depthChoice: .textureLoom, settings: TextureSettings(), enabledLoRAs: [])
        let recipeURL = root.appendingPathComponent("broken.texture.json")
        try JSONEncoder().encode(recipe).write(to: recipeURL)

        workspace.openRecipe(recipeURL); try await finish(workspace)

        XCTAssertNotNil(workspace.notice)
        XCTAssertEqual(workspace.source?.url.lastPathComponent, "previous.png")
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.settings.roughnessBase, 0.75)
        XCTAssertEqual(workspace.recipeURL?.lastPathComponent, "previous.texture.json")
        XCTAssertEqual(workspace.exportURL?.lastPathComponent, "previous-export")
        XCTAssertEqual(workspace.decision, decision)
        XCTAssertEqual(store.revision, revision)
        XCTAssertEqual(preferences.data(forKey: StudioPreferences.key), saved)
    }
    private func previousDecision() -> MaterialDecision {
        MaterialDecision(id: UUID(), model: MaterialDecision.exactModel, lighting: .strong, noise: .none,
            relief: .strong, roughness: .glossy, confidence: .high, rationale: "Previous photo only")
    }
    private func finish(_ workspace: TextureWorkspace) async throws {
        for _ in 0..<1000 {
            if !workspace.isBusy { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Material operation did not finish.")
    }
}
