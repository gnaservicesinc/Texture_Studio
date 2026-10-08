import XCTest
import CoreImage
@testable import TextureStudio

@MainActor
final class WorkspaceTests: XCTestCase {
    func testDefaultUsesDA3AndExcludesPortraitDepth() {
        let workspace = TextureWorkspace(checkpointRegistryURL: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
        XCTAssertEqual(workspace.depthChoice, .model)
        XCTAssertEqual(workspace.modelID, LocalModelDescriptor.da3GiantID)
        XCTAssertEqual(workspace.settings.heightDetail, 0)
        XCTAssertFalse(DepthChoice.allCases.map(\.rawValue).contains("Embedded depth"))
        XCTAssertFalse(LocalModelDescriptor.catalog.contains { $0.id == "depth-anything-v2-small" })
    }

    func testAdviserCannotEnableBrightnessGeometry() {
        let workspace = TextureWorkspace()
        workspace.decision = MaterialDecision(id: UUID(), model: MaterialDecision.exactModel,
            lighting: .strong, noise: .moderate, relief: .strong, roughness: .matte,
            confidence: .high, rationale: "Fixed-choice suggestion")
        workspace.applyDecision()
        XCTAssertEqual(workspace.settings.heightDetail, 0)
        XCTAssertEqual(workspace.settings.heightStrength, 1)
    }

    func testLegacyPortraitRecipeMigratesToNeutralReliefWithoutSelectingAnotherModel() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let photo = directory.appendingPathComponent("photo.png")
        try CIContext().writePNGRepresentation(of: fixtureSource(photo).orientedImage, to: photo,
            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let oldAuxiliary = directory.appendingPathComponent("old-portrait-depth.bin")
        try Data("This obsolete auxiliary is deliberately unreadable as a material map".utf8).write(to: oldAuxiliary)
        let recipeURL = directory.appendingPathComponent("legacy.json")
        let object: [String: Any] = ["version": 1, "photoPath": photo.path,
            "depthChoice": "Embedded depth", "depthPath": oldAuxiliary.path, "modelID": "depth-anything-v2-small",
            "customInverseDepth": true, "settings": ["heightDetail": 0.35, "useEmbeddedDepth": true]]
        try JSONSerialization.data(withJSONObject: object).write(to: recipeURL)
        let workspace = TextureWorkspace()
        workspace.openRecipe(recipeURL)
        try await waitForOperation(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.modelID, LocalModelDescriptor.da3GiantID)
        XCTAssertEqual(workspace.settings.heightDetail, 0)
        XCTAssertEqual(workspace.settings.surfacePlaneRemoval, 1)
        XCTAssertFalse(workspace.warnings.isEmpty)
        XCTAssertNil(workspace.attachedDepth, "A legacy portrait map must not be decoded as an attached material map")
        XCTAssertNil(workspace.depthURL)
        XCTAssertEqual(try Data(contentsOf: oldAuxiliary), Data("This obsolete auxiliary is deliberately unreadable as a material map".utf8))
    }

    func testLegacyPortraitPreferencesMigrateToNeutralAndKeepOtherUserSettings() throws {
        let suite = "portrait-migration-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let object: [String: Any] = ["depthChoice": "Embedded depth", "modelID": "custom-depth",
            "settings": ["useEmbeddedDepth": true, "outputSize": 2048, "rotationX": 7.93, "useHDRGainMap": false],
            "selectedPreview": "Normal", "showInspector": false]
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: StudioPreferences.key)
        let workspace = TextureWorkspace(checkpointRegistryURL: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString),
            preferences: defaults)
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.modelID, "custom-depth")
        XCTAssertEqual(workspace.settings.outputSize, 2048)
        XCTAssertEqual(workspace.settings.rotationX, 7.93)
        XCTAssertFalse(workspace.settings.useHDRGainMap)
        XCTAssertEqual(workspace.selectedPreview, .normal)
        XCTAssertFalse(workspace.showInspector)
        let rewritten = try XCTUnwrap(JSONSerialization.jsonObject(with: try XCTUnwrap(defaults.data(forKey: StudioPreferences.key))) as? [String: Any])
        XCTAssertEqual(rewritten["depthChoice"] as? String, DepthChoice.photoDetail.rawValue)
        XCTAssertNil((rewritten["settings"] as? [String: Any])?["useEmbeddedDepth"])
        let reopened = TextureWorkspace(checkpointRegistryURL: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString),
            preferences: defaults)
        XCTAssertEqual(reopened.depthChoice, .photoDetail)
        XCTAssertEqual(reopened.settings, workspace.settings)
    }

    func testCancelledImportPreservesPreviousDocumentWithoutError() async throws {
        let workspace = TextureWorkspace()
        let source = fixtureSource(URL(fileURLWithPath: "/previous.png"))
        workspace.source = source
        workspace.warnings = ["Previous photo warning"]
        workspace.importPhoto(URL(fileURLWithPath: "/missing-new-photo.png"))
        workspace.cancel()
        try await waitForOperation(workspace)
        XCTAssertEqual(workspace.source?.url, source.url)
        XCTAssertEqual(workspace.warnings, ["Previous photo warning"])
        XCTAssertNil(workspace.notice)
    }

    func testOpeningRecipeClearsOtherPhotosReviewAndExportState() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let photo = directory.appendingPathComponent("photo.png")
        let source = fixtureSource(photo)
        try CIContext().writePNGRepresentation(of: source.orientedImage, to: photo, format: .RGBA8,
                                               colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let recipe = TextureRecipe(photoPath: photo.path, depthPath: nil, depthChoice: .photoDetail,
            modelID: LocalModelDescriptor.da3GiantID, customInverseDepth: true, settings: TextureSettings())
        let recipeURL = directory.appendingPathComponent("recipe.json")
        try JSONEncoder().encode(recipe).write(to: recipeURL)
        let workspace = TextureWorkspace()
        workspace.decision = MaterialDecision(id: UUID(), model: MaterialDecision.exactModel,
            lighting: .strong, noise: .moderate, relief: .strong, roughness: .glossy,
            confidence: .high, rationale: "Review of a different photograph")
        workspace.exportURL = directory.appendingPathComponent("old-export")
        workspace.warnings = ["Old warning"]
        workspace.generatedDepth = try TextureDepth(width: 2, height: 2, values: [0,1,2,3], sourceLabel: "Old depth")
        workspace.openRecipe(recipeURL)
        try await waitForOperation(workspace)
        XCTAssertNil(workspace.notice)
        XCTAssertEqual(workspace.source?.url, photo)
        XCTAssertNil(workspace.decision)
        XCTAssertNil(workspace.exportURL)
        XCTAssertNil(workspace.generatedDepth)
        XCTAssertEqual(workspace.warnings, [])
    }

    private func fixtureSource(_ url: URL) -> TextureSource {
        TextureSource(url: url,
            orientedImage: CIImage(color: CIColor(red: 0.3,green: 0.4,blue: 0.5))
                .cropped(to: CGRect(x: 0,y: 0,width: 32,height: 32)),
            camera: CameraMetadata(), pixelWidth: 32, pixelHeight: 32)
    }

    private func waitForOperation(_ workspace: TextureWorkspace) async throws {
        for _ in 0..<1000 {
            if !workspace.isBusy { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("Workspace operation did not finish")
    }
}
