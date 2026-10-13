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
    private func finish(_ workspace: TextureWorkspace) async throws {
        for _ in 0..<1000 {
            if !workspace.isBusy { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Material operation did not finish.")
    }
}
