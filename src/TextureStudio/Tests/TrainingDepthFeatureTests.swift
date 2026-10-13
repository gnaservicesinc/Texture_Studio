import Foundation
import XCTest
@testable import TextureStudio

final class TrainingDepthFeatureTests: XCTestCase {
    func testDepthFeatureCurveStrengthensInteriorContrastWithoutClippingOrMovingEndpoints() {
        let depth: [Float] = [0.2, 0.35, 0.425, 0.5, 0.575, 0.65, 0.8]
        let result = NativeMaterialTrainer.exaggeratedDepth(depth)
        XCTAssertEqual(result.first, depth.first)
        XCTAssertEqual(result.last, depth.last)
        XCTAssertEqual(result[3], 0.5, accuracy: 0.000001)
        XCTAssertGreaterThan(result[4] - result[2], depth[4] - depth[2])
        XCTAssertTrue(zip(result, result.dropFirst()).allSatisfy { $0 < $1 })
        XCTAssertTrue(result.allSatisfy { $0.isFinite && (0.2...0.8).contains($0) })
        XCTAssertEqual(depth, [0.2, 0.35, 0.425, 0.5, 0.575, 0.65, 0.8])
    }

    func testFlatDepthAndEmptyMapsRemainUnchanged() {
        XCTAssertEqual(NativeMaterialTrainer.exaggeratedDepth([]), [])
        XCTAssertEqual(NativeMaterialTrainer.exaggeratedDepth([0.4, 0.4, 0.4]), [0.4, 0.4, 0.4])
    }

    func testDepthExaggerationIsExplicitAndRejectedForOtherMapTypes() throws {
        let arguments = ["train", "--dataset", "/dataset", "--output", "/output", "--model-family", "compact-scalar"]
        XCTAssertFalse(try NativeMaterialTrainer.Options(arguments).exaggerateFeatures)
        XCTAssertTrue(try NativeMaterialTrainer.Options(arguments + ["--exaggerate-features"]).exaggerateFeatures)
        XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + ["--target", "roughness", "--exaggerate-features"]))
        let configuration = try NativeMaterialTrainer.Options(arguments + ["--exaggerate-features"]).trainingConfiguration
        XCTAssertEqual(configuration["exaggerate_features"] as? Bool, true)
    }
}
