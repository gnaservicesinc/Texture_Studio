import XCTest
import CoreImage
@testable import TextureStudio

@MainActor
final class WorkspaceTests: XCTestCase {
    func testDefaultUsesFlatSurfaceAndCustomCoreMLIsOptional() {
        let workspace = TextureWorkspace(checkpointRegistryURL: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
        XCTAssertEqual(workspace.depthChoice, .photoDetail)
        XCTAssertEqual(workspace.modelID, LocalModelDescriptor.customDepthID)
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
            modelID: LocalModelDescriptor.customDepthID, customInverseDepth: true, settings: TextureSettings())
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
