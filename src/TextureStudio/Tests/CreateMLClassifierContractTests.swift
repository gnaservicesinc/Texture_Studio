import CoreImage
import CoreML
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class CreateMLClassifierContractTests: XCTestCase {
    func testNativeImageInputRoutesSourceLabelToItsFixedDataset() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let package = try TextureLoomPackage(root: root)
        XCTAssertNil(package.manifest.classifier.weightsPath)
        XCTAssertEqual(package.manifest.classifier.inputName, "image")
        XCTAssertEqual(package.manifest.classifier.outputName, "target")
        let image = CIImage(color: CIColor(red: 0.36, green: 0.21, blue: 0.11))
            .cropped(to: CGRect(x: 17, y: 29, width: 960, height: 384))
        let result = try await TextureLoomService(computeUnits: .cpuOnly).classify(diffuse: image, package: package)
        let label = try XCTUnwrap(package.manifest.classifier.labels.first { $0.id == result.categoryID })
        XCTAssertTrue(result.categoryID.hasPrefix("dataset-native-"))
        XCTAssertNotEqual(result.categoryID, label.sourceLabel)
        XCTAssertEqual(result.categoryName, label.name)
        XCTAssertTrue(result.confidence.isFinite)
        XCTAssertTrue((0...1).contains(result.confidence))
        // This exercises classification and routing only, never the map fixture.
    }

    func testActualClassifierLabelsMustExactlyMatchCatalog() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        try change(root) { document in
            var classifier = document["classifier"] as! [String: Any]
            var labels = classifier["labels"] as! [[String: String]]
            labels[0]["sourceLabel"] = "A label absent from this model"
            classifier["labels"] = labels; document["classifier"] = classifier
        }
        XCTAssertThrowsError(try TextureLoomPackage(root: root))
    }

    func testDeclaredImageDimensionsCannotDifferFromNativeConstraint() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        try change(root) { document in
            var classifier = document["classifier"] as! [String: Any]
            classifier["inputWidth"] = 8; document["classifier"] = classifier
        }
        XCTAssertThrowsError(try TextureLoomPackage(root: root))
    }

    private func fixture() throws -> URL {
        let workspace = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let asset = ProcessInfo.processInfo.environment["LOOM_CREATE_ML_TEST_MODEL"].map { URL(fileURLWithPath: $0) }
            ?? workspace.appendingPathComponent("LoomFactory/Resources/DTTC3.mlmodel")
        guard FileManager.default.fileExists(atPath: asset.path) else { throw XCTSkip("Provide the trained image-classifier fixture with LOOM_CREATE_ML_TEST_MODEL.") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-native-classifier-\(UUID().uuidString)")
            .standardizedFileURL.resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let compiled = try MLModel.compileModel(at: asset)
        defer { try? FileManager.default.removeItem(at: compiled) }
        try FileManager.default.copyItem(at: compiled, to: root.appendingPathComponent("classifier.mlmodelc"))
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let description = try MLModel(contentsOf: compiled, configuration: configuration).modelDescription
        let input = try XCTUnwrap(description.inputDescriptionsByName.first { $0.value.type == .image })
        let output = try XCTUnwrap(description.predictedFeatureName)
        let constraint = try XCTUnwrap(input.value.imageConstraint)
        let labels = try XCTUnwrap(description.classLabels as? [String])
        let catalog = labels.enumerated().map { ["id": "dataset-native-\($0.offset)", "name": $0.element, "sourceLabel": $0.element] }
        // Map bytes test only the package contract; inference uses the real classifier.
        try FileManager.default.createDirectory(at: root.appendingPathComponent("map.mlmodelc"), withIntermediateDirectories: true)
        try Data("contract map fixture".utf8).write(to: root.appendingPathComponent("map.mlmodelc/fixture"))
        try Data("contract map weights".utf8).write(to: root.appendingPathComponent("map.safetensors"))
        let categories = catalog.map { label in ["id": label["id"]!, "name": label["name"]!, "version": "1",
            "models": [["map": "depth", "format": "coreml", "path": "map.mlmodelc", "weightsPath": "map.safetensors", "architecture": "fixture"]]] as [String: Any] }
        var files: [[String: String]] = []
        let entries = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey])!
        for case let file as URL in entries where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            let relative = String(file.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(root.path.count + 1))
            files.append(["path": relative, "sha256": try TextureLoomPackage.sha256(file)])
        }
        let document: [String: Any] = ["schema": "texture-loom.package.v1", "id": UUID().uuidString,
            "name": "Native classifier contract fixture", "version": "1",
            "classifier": ["format": "coreml", "path": "classifier.mlmodelc", "architecture": "create-ml-image-classifier-v1",
                "inputName": input.key, "outputName": output, "inputWidth": constraint.pixelsWide,
                "inputHeight": constraint.pixelsHigh, "inputColorSpace": "RGB", "labels": catalog],
            "categories": categories, "files": files]
        try JSONSerialization.data(withJSONObject: document).write(to: root.appendingPathComponent("manifest.json"))
        return root
    }

    private func change(_ root: URL, _ action: (inout [String: Any]) -> Void) throws {
        let file = root.appendingPathComponent("manifest.json")
        var document = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
        action(&document); try JSONSerialization.data(withJSONObject: document).write(to: file)
    }
}
