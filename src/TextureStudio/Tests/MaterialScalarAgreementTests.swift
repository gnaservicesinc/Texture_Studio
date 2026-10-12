import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

final class MaterialScalarAgreementTests: XCTestCase {
    func testExportedProductionMapsReportRawSourceAgreement() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["TEXTURE_STUDIO_SCALAR_AGREEMENT_EXPORTS"] else {
            throw XCTSkip("Set TEXTURE_STUDIO_SCALAR_AGREEMENT_EXPORTS to audit exported source/prediction maps.")
        }
        let folder = URL(fileURLWithPath: path)
        let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
        let source = try XCTUnwrap(files.first { $0.lastPathComponent.hasSuffix("-Source-displacement-reference.png") })
        let predictions = files.filter { $0.pathExtension.lowercased() == "exr" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertFalse(predictions.isEmpty)
        var reports: [[String: Any]] = []
        for prediction in predictions {
            let result = try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: prediction, mapType: "height")
            XCTAssertEqual(result.referenceSHA256, "e02b952d984561ca89e5b33e788914d9bda286a4a1914323d0557e910892be98")
            XCTAssertEqual(result.width, 1024); XCTAssertEqual(result.height, 1024)
            XCTAssertEqual(result.referenceMean, 0.2610715210369562, accuracy: 1e-7)
            XCTAssertEqual(result.referenceStandardDeviation, 0.10307846849652004, accuracy: 1e-7)
            XCTAssertEqual(result.medianConstantMAE, 0.07750456114332635, accuracy: 1e-7)
            reports.append(["prediction_path": prediction.path, "reference_path": source.path,
                "reference_sha256": result.referenceSHA256, "prediction_sha256": result.predictionSHA256,
                "width": result.width, "height": result.height, "mae": result.mae,
                "reference_mean": result.referenceMean, "reference_median": result.referenceMedian,
                "reference_std": result.referenceStandardDeviation, "prediction_mean": result.predictionMean,
                "prediction_std": result.predictionStandardDeviation, "prediction_min": result.predictionMinimum,
                "prediction_max": result.predictionMaximum, "mean_constant_mae": result.meanConstantMAE,
                "median_constant_mae": result.medianConstantMAE,
                "pearson_correlation": result.pearsonCorrelation as Any? ?? NSNull(),
                "std_ratio": result.standardDeviationRatio as Any? ?? NSNull(),
                "adjacent_difference_std_ratio": result.adjacentDifferenceStandardDeviationRatio as Any? ?? NSNull(),
                "nearly_flat_heuristic": result.nearlyFlat])
        }
        if let reportPath = environment["TEXTURE_STUDIO_SCALAR_AGREEMENT_REPORT"] {
            let report = try JSONSerialization.data(withJSONObject: ["schema": "texture-studio-scalar-agreement-audit-v1", "maps": reports], options: [.prettyPrinted, .sortedKeys])
            try report.write(to: URL(fileURLWithPath: reportPath), options: .atomic)
        }
        print("SCALAR_AGREEMENT_AUDIT " + String(decoding: try JSONSerialization.data(withJSONObject: reports, options: [.sortedKeys]), as: UTF8.self))
    }

    func testIdenticalNativePNGAndFloat32EXRKeepTopLeftOrientationAndSourceCodes() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.png"), output = root.appendingPathComponent("prediction.exr")
        // Unequal rows detect a vertical flip; gamma and alpha must not alter
        // the immutable height codes entering numeric source agreement.
        let codes: [UInt16] = [0, 8192, 16384, 24576, 49152, 65535]
        let bytes = try png(codes, width: 3, height: 2, alpha: true)
        try bytes.write(to: source)
        let values = codes.map { Float($0) / 65535 }
        try NativeMaterialNumericExporter.writeEXR(.init(width: 3, height: 2, channels: 1, values: values), to: output)
        let measured = try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height")
        XCTAssertEqual(measured.width, 3); XCTAssertEqual(measured.height, 2); XCTAssertEqual(measured.pixelCount, 6)
        XCTAssertEqual(measured.mae, 0)
        XCTAssertEqual(try XCTUnwrap(measured.pearsonCorrelation), 1, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(measured.standardDeviationRatio), 1, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(measured.adjacentDifferenceStandardDeviationRatio), 1, accuracy: 1e-12)
        XCTAssertEqual(measured.referenceMinimum, 0); XCTAssertEqual(measured.referenceMaximum, 1)
        XCTAssertEqual(measured.referenceSHA256, hash(bytes))
        XCTAssertEqual(measured.predictionSHA256, hash(try Data(contentsOf: output)))
        XCTAssertFalse(measured.nearlyFlat)
    }

    func testConstantPredictionIsMeasuredAgainstTheBestSourceConstantBaseline() throws {
        let source: [Float] = [0, 1, 0, 1]
        let result = try MaterialScalarAgreement.statistics(reference: source, prediction: [0.5, 0.5, 0.5, 0.5], width: 2, height: 2)
        XCTAssertEqual(result.referenceMean, 0.5); XCTAssertEqual(result.referenceMedian, 0.5)
        XCTAssertEqual(result.referenceStandardDeviation, 0.5)
        XCTAssertEqual(result.predictionStandardDeviation, 0)
        XCTAssertEqual(result.mae, 0.5); XCTAssertEqual(result.meanConstantMAE, 0.5); XCTAssertEqual(result.medianConstantMAE, 0.5)
        XCTAssertNil(result.pearsonCorrelation)
        XCTAssertEqual(result.standardDeviationRatio, 0); XCTAssertEqual(result.adjacentDifferenceStandardDeviationRatio, 0)
        XCTAssertTrue(result.nearlyFlat)
        let unequal = try MaterialScalarAgreement.statistics(reference: [0, 0, 0, 1], prediction: [0, 0, 0, 0], width: 2, height: 2)
        XCTAssertEqual(unequal.meanConstantMAE, 0.375); XCTAssertEqual(unequal.medianConstantMAE, 0.25)
        XCTAssertEqual(unequal.mae, unequal.medianConstantMAE)
    }

    func testOffsetAndScaledRampsKeepSignedAndOutOfRangePredictions() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.png"), output = root.appendingPathComponent("prediction.exr")
        let codes: [UInt16] = [0, 8192, 16384, 24576, 49152, 65535]
        try png(codes, width: 3, height: 2).write(to: source)
        let values = codes.map { 3 * (Float($0) / 65535) - 0.5 }
        try NativeMaterialNumericExporter.writeEXR(.init(width: 3, height: 2, channels: 1, values: values), to: output)
        let result = try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "displacement")
        XCTAssertEqual(result.predictionMinimum, -0.5); XCTAssertEqual(result.predictionMaximum, 2.5)
        XCTAssertEqual(try XCTUnwrap(result.standardDeviationRatio), 3, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(result.adjacentDifferenceStandardDeviationRatio), 3, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(result.pearsonCorrelation), 1, accuracy: 1e-12)
        let expected = zip(codes.map { Float($0) / 65535 }, values).reduce(0.0) { $0 + abs(Double($1.0) - Double($1.1)) } / 6
        XCTAssertEqual(result.mae, expected, accuracy: 1e-12)
    }

    func testDeclaredCropUsesItsExactCoordinatesAndChecksOriginalHash() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.png"), output = root.appendingPathComponent("prediction.exr")
        let codes: [UInt16] = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        let bytes = try png(codes, width: 3, height: 3); try bytes.write(to: source)
        let selected: [UInt16] = [500, 600, 800, 900]
        try NativeMaterialNumericExporter.writeEXR(.init(width: 2, height: 2, channels: 1, values: selected.map { Float($0) / 65535 }), to: output)
        var transform = MapReviewDisplayTransform(size: 2, sourceSHA256: hash(bytes), algorithm: MapReviewDisplayTransform.exactCrop,
            mapType: "height", cropRectangle: [1, 1, 2, 2])
        let result = try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height", referenceTransform: transform)
        XCTAssertEqual(result.mae, 0); XCTAssertEqual(result.referenceSHA256, hash(bytes))
        transform.cropRectangle = [2, 1, 2, 2]
        XCTAssertThrowsError(try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height", referenceTransform: transform))
        transform.cropRectangle = nil
        XCTAssertThrowsError(try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height", referenceTransform: transform))
        let changed = MapReviewDisplayTransform(size: 2, sourceSHA256: "changed", algorithm: MapReviewDisplayTransform.exactCrop,
            mapType: "height", cropRectangle: [1, 1, 2, 2])
        XCTAssertThrowsError(try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height", referenceTransform: changed))
    }

    func testDimensionMismatchNonfiniteSamplesAndColorEXRAreRejected() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.png"), output = root.appendingPathComponent("prediction.exr")
        try png([0, 8192, 16384, 24576, 49152, 65535], width: 3, height: 2).write(to: source)
        try NativeMaterialNumericExporter.writeEXR(.init(width: 2, height: 3, channels: 1, values: Array(repeating: 0.5, count: 6)), to: output)
        XCTAssertThrowsError(try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height"))
        try NativeMaterialNumericExporter.writeEXR(.init(width: 3, height: 2, channels: 3, values: Array(repeating: 0.5, count: 18)), to: output)
        XCTAssertThrowsError(try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height"))
        try NativeMaterialNumericExporter.writeEXR(.init(width: 3, height: 2, channels: 1, values: Array(repeating: 0.5, count: 6)), to: output)
        var nonfinite = try Data(contentsOf: output)
        var nan = Float.nan.bitPattern.littleEndian
        withUnsafeBytes(of: &nan) { nonfinite.replaceSubrange(nonfinite.count - 4..<nonfinite.count, with: $0) }
        try nonfinite.write(to: output)
        XCTAssertThrowsError(try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height"))
        XCTAssertThrowsError(try MaterialScalarAgreement.statistics(reference: [0, .nan], prediction: [0, 0], width: 2, height: 1))
        XCTAssertThrowsError(try MaterialScalarAgreement.statistics(reference: [0, 1], prediction: [0, .infinity], width: 2, height: 1))
    }

    func testConstantSourceHasNoCorrelationOrFlatnessJudgment() throws {
        let result = try MaterialScalarAgreement.statistics(reference: [0.5, 0.5], prediction: [0.5, 0.5], width: 2, height: 1)
        XCTAssertEqual(result.mae, 0); XCTAssertEqual(result.medianConstantMAE, 0)
        XCTAssertNil(result.pearsonCorrelation); XCTAssertNil(result.standardDeviationRatio); XCTAssertNil(result.adjacentDifferenceStandardDeviationRatio)
        XCTAssertFalse(result.nearlyFlat)
    }

    func testRoughnessAcceptsEightBitCodesWithoutColorTransferWhileHeightRequiresSixteenBits() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.png"), output = root.appendingPathComponent("prediction.exr")
        let image = NativePNG(header: .init(width: 2, height: 1, bits: 8, channels: 1, color: 0, interlace: 0),
            pixels: Data([64, 192]), colorChunks: [("gAMA", Data([0, 0, 177, 143]))])
        try image.encoded().write(to: source)
        try NativeMaterialNumericExporter.writeEXR(.init(width: 2, height: 1, channels: 1, values: [64.0 / 255, 192.0 / 255]), to: output)
        XCTAssertEqual(try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "roughness").mae, 0)
        XCTAssertThrowsError(try MaterialScalarAgreement.evaluate(referenceURL: source, predictionURL: output, mapType: "height"))
    }

    private func temporary() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("scalar-agreement-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func png(_ codes: [UInt16], width: Int, height: Int, alpha: Bool = false) throws -> Data {
        var samples = Data()
        for (index, code) in codes.enumerated() {
            samples.append(UInt8(code >> 8)); samples.append(UInt8(code & 255))
            if alpha { samples.append(0); samples.append(UInt8(index)) }
        }
        return try NativePNG(header: .init(width: width, height: height, bits: 16, channels: alpha ? 2 : 1, color: alpha ? 4 : 0, interlace: 0),
            pixels: samples, colorChunks: [("gAMA", Data([0, 0, 177, 143]))]).encoded()
    }

    private func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}
