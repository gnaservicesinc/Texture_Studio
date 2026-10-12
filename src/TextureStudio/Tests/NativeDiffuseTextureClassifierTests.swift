import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

final class NativeDiffuseTextureClassifierTests: XCTestCase {
    func testBundledExportLoadsDeclaredImageInterfaceAndAllTrainedLabels() throws {
        let classifier = try NativeDiffuseTextureClassifier()
        XCTAssertEqual(classifier.modelSHA256, NativeDiffuseTextureClassifier.bundledModelSHA256)
        XCTAssertEqual(classifier.inputName, "image")
        XCTAssertEqual(classifier.inputWidth, 360); XCTAssertEqual(classifier.inputHeight, 360)
        XCTAssertEqual(classifier.predictedFeatureName, "target")
        XCTAssertEqual(classifier.predictedProbabilitiesName, "targetProbability")
        XCTAssertEqual(Set(classifier.labels), ["Bark", "Bricks", "Cement Concrete Pavement", "Drywall", "Fabric", "Ground", "Leather",
            "Medium Particles", "Metal", "Other Diffuse Textures", "Paper", "Rock Cliff Marble", "Small Particles", "Tiles and Pavers", "Wood"])
        XCTAssertEqual(Set(classifier.labels).count, classifier.labels.count)
        XCTAssertTrue(classifier.preprocessing.contains("scaleFill"))
        let custom = try NativeDiffuseTextureClassifier(modelURL: exportedModel)
        XCTAssertEqual(custom.labels, classifier.labels, "Runtime labels come from the selected model, including classes added after the original dataset")
        XCTAssertEqual(custom.modelSHA256, classifier.modelSHA256, "The bundled identity is the exact original exported specification")
        XCTAssertEqual(try NativeDiffuseTextureClassifier.modelIdentity(at: exportedModel), classifier.modelSHA256)
    }

    func testBundledDTTC3SpecificationIsTheExactProvidedExport() throws {
        let provided = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Classifiers/diffuse_texture_type/DTTC3.mlmodel")
        XCTAssertEqual(try Data(contentsOf: provided), try Data(contentsOf: exportedModel))
        XCTAssertEqual(try NativeDiffuseTextureClassifier.modelIdentity(at: provided), NativeDiffuseTextureClassifier.bundledModelSHA256)
        XCTAssertNotEqual(NativeDiffuseTextureClassifier.bundledModelSHA256,
            "e2936fcef0781fb4f85eb7e879cc2f7d865528415180d894afd74eddc1e048be", "Automatic predictions cached for IFPv2 cannot be reused for DTTC3")
        XCTAssertEqual(NativeDiffuseTextureClassifier.bundledDisplayName, "DTTC3")
    }

    func testRealModelClassifiesKnownCorpusAndIgnoresPNGFilenameWithoutChangingImages() throws {
        let fixtures: [(String, String)] = [
            ("Bricks/red_brick--color_1--3f6d0fdd452e.png", "Bricks"),
            ("Wood/japanese_cedar_planks--color_1--9733be0af1c3.png", "Wood"),
            ("Metal/metal_plate--color_1--42fb1a7ed34d.png", "Metal"),
            ("Paper/Paper003_1K-PNG_Color.png", "Paper"),
            ("Paper/Cardboard_01_color_1K.png", "Paper")
        ]
        let corpus = URL(fileURLWithPath: "/opt/ipde/classifer_ds", isDirectory: true)
        try XCTSkipUnless(fixtures.allSatisfy { FileManager.default.fileExists(atPath: corpus.appendingPathComponent($0.0).path) },
            "Optional known-material corpus is unavailable; this check is not held-out accuracy")
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let classifier = try NativeDiffuseTextureClassifier()
        for (index, fixture) in fixtures.enumerated() {
            let source = corpus.appendingPathComponent(fixture.0)
            let original = try Data(contentsOf: source)
            if fixture.0.contains("Paper003") {
                let header = try NativePNG.inspect(source)
                XCTAssertEqual(header.channels, 1); XCTAssertEqual(header.bits, 8)
            } else if fixture.0.contains("Cardboard") {
                let header = try NativePNG.inspect(source)
                XCTAssertEqual(header.channels, 4); XCTAssertEqual(header.bits, 16)
            }
            let arbitrary = root.appendingPathComponent("arbitrary-\(index)-unknown-photo.png")
            try original.write(to: arbitrary)
            let sourcePrediction = try classifier.predict(url: source)
            let renamedPrediction = try classifier.predict(url: arbitrary)
            let color = try XCTUnwrap(CIImage(contentsOf: source))
            let linear = try LinearColorFrame(color, context: CIContext()).color
            let preparedPrediction = try classifier.predict(image: linear)
            XCTAssertEqual(sourcePrediction.label, fixture.1, "Known corpus runtime regression only; not held-out model quality")
            XCTAssertEqual(renamedPrediction.label, sourcePrediction.label)
            XCTAssertEqual(preparedPrediction.label, sourcePrediction.label,
                "Prepared linear diffuse color is returned to classifier color encoding without changing the original image")
            XCTAssertEqual(Set(sourcePrediction.probabilities.keys), Set(classifier.labels))
            XCTAssertEqual(renamedPrediction.modelSHA256, classifier.modelSHA256)
            XCTAssertEqual(sourcePrediction.probabilities.values.reduce(0, +), 1, accuracy: 0.00001)
            for label in classifier.labels {
                XCTAssertEqual(try XCTUnwrap(renamedPrediction.probabilities[label]),
                    try XCTUnwrap(sourcePrediction.probabilities[label]), accuracy: 0.000001,
                    "The image contents determine the probability, regardless of its filename")
            }
            XCTAssertEqual(try Data(contentsOf: source), original)
            XCTAssertEqual(try Data(contentsOf: arbitrary), original)
        }
        XCTAssertThrowsError(try classifier.predict(url: root.appendingPathComponent("missing.png")))
    }

    func testCustomPackageIdentitySurvivesRelocationAndDetectsChangedWeights() throws {
        let root = try temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first.mlpackage"), second = root.appendingPathComponent("second.mlpackage")
        try FileManager.default.createDirectory(at: first.appendingPathComponent("Data"), withIntermediateDirectories: true)
        try Data("specification".utf8).write(to: first.appendingPathComponent("Manifest.json"))
        try Data([0, 1, 2, 3]).write(to: first.appendingPathComponent("Data/weights.bin"))
        try FileManager.default.copyItem(at: first, to: second)
        let digest = try NativeDiffuseTextureClassifier.modelIdentity(at: first)
        XCTAssertEqual(try NativeDiffuseTextureClassifier.modelIdentity(at: second), digest)
        try Data([0, 1, 2, 4]).write(to: second.appendingPathComponent("Data/weights.bin"))
        XCTAssertNotEqual(try NativeDiffuseTextureClassifier.modelIdentity(at: second), digest)
        XCTAssertThrowsError(try NativeDiffuseTextureClassifier(modelURL: root.appendingPathComponent("missing.mlmodel")))
        let invalid = root.appendingPathComponent("invalid.mlmodel")
        try Data("not a Core ML classifier".utf8).write(to: invalid)
        XCTAssertThrowsError(try NativeDiffuseTextureClassifier(modelURL: invalid))
    }

    private var exportedModel: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Resources/DiffuseTextureTypeClassifier.mlmodel")
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("diffuse-classifier-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
