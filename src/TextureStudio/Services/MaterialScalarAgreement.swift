import CryptoKit
import Foundation

/// Agreement with an exact source map, measured before preview color conversion.
/// These diagnostics describe the compared pixels; they do not qualify novel
/// material accuracy, normalize either map, or decide whether training may run.
enum MaterialScalarAgreement {
    struct Result: Sendable {
        let width: Int, height: Int, pixelCount: Int
        let referenceSHA256: String, predictionSHA256: String
        let referenceMean: Double, referenceMedian: Double
        let referenceStandardDeviation: Double, predictionMean: Double, predictionStandardDeviation: Double
        let referenceMinimum: Double, referenceMaximum: Double, predictionMinimum: Double, predictionMaximum: Double
        let mae: Double, meanConstantMAE: Double, medianConstantMAE: Double
        let pearsonCorrelation: Double?, standardDeviationRatio: Double?
        let sourceAdjacentDifferenceStandardDeviation: Double, predictionAdjacentDifferenceStandardDeviation: Double
        let adjacentDifferenceStandardDeviationRatio: Double?
        /// Heuristic only: less than 10% of a nonconstant source's variation.
        let nearlyFlat: Bool
    }

    static func evaluate(referenceURL: URL, predictionURL: URL, mapType: String,
                         referenceTransform: MapReviewDisplayTransform? = nil) throws -> Result {
        try Task.checkCancellation()
        let role = mapType.lowercased() == "displacement" ? "height" : mapType.lowercased()
        guard ["height", "roughness"].contains(role) else {
            throw StudioError("Numeric source agreement supports height and roughness maps.")
        }
        let referenceBytes = try Data(contentsOf: referenceURL)
        let referenceHash = hash(referenceBytes)
        let source: NativePNG
        if let transform = referenceTransform {
            guard transform.sourceSHA256 == referenceHash else {
                throw StudioError("The original source changed since its training crop was recorded.")
            }
            guard transform.algorithm == MapReviewDisplayTransform.exactCrop,
                  transform.mapType == role, (1...16384).contains(transform.size) else {
                throw StudioError("Numeric source agreement requires an exact recorded scalar crop.")
            }
            let rectangle = transform.cropRectangle ?? [0, 0, transform.size, transform.size]
            guard rectangle.count == 4, rectangle[2] == transform.size, rectangle[3] == transform.size else {
                throw StudioError("A larger original needs its exact recorded source crop for numeric agreement.")
            }
            if transform.cropRectangle == nil {
                let decoded = try NativePNG.decode(referenceBytes)
                guard decoded.header.width == transform.size, decoded.header.height == transform.size else {
                    throw StudioError("A larger original needs its exact recorded source crop for numeric agreement.")
                }
                source = decoded
            } else {
                // Decode only the declared crop instead of allocating another
                // full-resolution copy of a large source's unfiltered pixels.
                source = try NativePNG.crop(referenceBytes, rectangle: rectangle)
            }
        } else { source = try NativePNG.decode(referenceBytes) }
        let reference = try source.modelFloatSamples(role: role)
        try Task.checkCancellation()
        let predictionBytes = try Data(contentsOf: predictionURL)
        let prediction = try scalarEXR(predictionBytes)
        guard source.header.width == prediction.width, source.header.height == prediction.height else {
            throw StudioError("Numeric source agreement needs matching pixel dimensions and an exact source crop. Neither map is resized.")
        }
        return try statistics(reference: reference, prediction: prediction.values,
            width: prediction.width, height: prediction.height,
            referenceSHA256: referenceHash, predictionSHA256: hash(predictionBytes))
    }

    /// Separate numeric entry point keeps analytic diagnostics independent of
    /// image encoding. Arrays use the same top-left row order as NativePNG and
    /// NativeMaterialNumericExporter.
    static func statistics(reference: [Float], prediction: [Float], width: Int, height: Int,
                           referenceSHA256: String = "", predictionSHA256: String = "") throws -> Result {
        guard width > 0, height > 0, width <= 16384, height <= 16384,
              width * height <= 150_000_000, reference.count == width * height,
              prediction.count == reference.count else { throw StudioError("Numeric source agreement has invalid pixel dimensions.") }
        var sourceMoments = Moments(), predictionMoments = Moments()
        var covariance = 0.0, error = 0.0
        var sourceMinimum = Double.infinity, sourceMaximum = -Double.infinity
        var predictionMinimum = Double.infinity, predictionMaximum = -Double.infinity
        for index in reference.indices {
            if index & 4095 == 0 { try Task.checkCancellation() }
            let source = Double(reference[index]), output = Double(prediction[index])
            guard source.isFinite, output.isFinite else { throw StudioError("Numeric source agreement requires finite samples in both maps.") }
            let sourceDelta = source - sourceMoments.mean
            sourceMoments.add(source); predictionMoments.add(output)
            covariance += sourceDelta * (output - predictionMoments.mean)
            error += abs(source - output)
            sourceMinimum = min(sourceMinimum, source); sourceMaximum = max(sourceMaximum, source)
            predictionMinimum = min(predictionMinimum, output); predictionMaximum = max(predictionMaximum, output)
        }
        try Task.checkCancellation()
        let ordered = reference.sorted()
        let middle = ordered.count / 2
        let median = ordered.count % 2 == 0 ? (Double(ordered[middle - 1]) + Double(ordered[middle])) / 2 : Double(ordered[middle])
        var meanError = 0.0, medianError = 0.0
        var sourceDifferences = Moments(), predictionDifferences = Moments()
        for y in 0..<height {
            try Task.checkCancellation()
            for x in 0..<width {
                let index = y * width + x, source = Double(reference[index])
                meanError += abs(source - sourceMoments.mean); medianError += abs(source - median)
                if x + 1 < width {
                    sourceDifferences.add(Double(reference[index + 1]) - source)
                    predictionDifferences.add(Double(prediction[index + 1]) - Double(prediction[index]))
                }
                if y + 1 < height {
                    sourceDifferences.add(Double(reference[index + width]) - source)
                    predictionDifferences.add(Double(prediction[index + width]) - Double(prediction[index]))
                }
            }
        }
        let sourceStd = sourceMoments.standardDeviation, outputStd = predictionMoments.standardDeviation
        let ratio = sourceStd > 0 ? outputStd / sourceStd : nil
        let differenceRatio = sourceDifferences.standardDeviation > 0 ? predictionDifferences.standardDeviation / sourceDifferences.standardDeviation : nil
        let correlation = sourceMoments.squaredDeviation > 0 && predictionMoments.squaredDeviation > 0 ?
            max(-1, min(1, covariance / sqrt(sourceMoments.squaredDeviation * predictionMoments.squaredDeviation))) : nil
        return .init(width: width, height: height, pixelCount: reference.count,
            referenceSHA256: referenceSHA256, predictionSHA256: predictionSHA256,
            referenceMean: sourceMoments.mean, referenceMedian: median,
            referenceStandardDeviation: sourceStd, predictionMean: predictionMoments.mean, predictionStandardDeviation: outputStd,
            referenceMinimum: sourceMinimum, referenceMaximum: sourceMaximum,
            predictionMinimum: predictionMinimum, predictionMaximum: predictionMaximum,
            mae: error / Double(reference.count), meanConstantMAE: meanError / Double(reference.count), medianConstantMAE: medianError / Double(reference.count),
            pearsonCorrelation: correlation, standardDeviationRatio: ratio,
            sourceAdjacentDifferenceStandardDeviation: sourceDifferences.standardDeviation,
            predictionAdjacentDifferenceStandardDeviation: predictionDifferences.standardDeviation,
            adjacentDifferenceStandardDeviationRatio: differenceRatio, nearlyFlat: ratio.map { $0 < 0.1 } ?? false)
    }

    private struct Moments {
        var count = 0, mean = 0.0, squaredDeviation = 0.0
        mutating func add(_ value: Double) {
            count += 1
            let delta = value - mean
            mean += delta / Double(count)
            squaredDeviation += delta * (value - mean)
        }
        var standardDeviation: Double { count > 0 ? sqrt(max(0, squaredDeviation / Double(count))) : 0 }
    }

    private static func hash(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }

    /// Read the native writers' uncompressed, single FLOAT32 channel directly.
    /// EXR scanline y coordinates give top-left row order without Core Image's
    /// coordinate transform, transfer function, premultiplication or clipping.
    private static func scalarEXR(_ bytes: Data) throws -> (width: Int, height: Int, values: [Float]) {
        var reader = EXRReader(bytes: bytes)
        guard try reader.word() == 20_000_630, try reader.word() == 2 else {
            throw StudioError("Numeric source agreement requires a standard scanline EXR.")
        }
        var attributes: [String: (String, Data)] = [:]
        while true {
            let name = try reader.string()
            if name.isEmpty { break }
            let type = try reader.string(), count = Int(try reader.word())
            guard attributes[name] == nil else { throw StudioError("The scalar EXR contains duplicate attributes.") }
            attributes[name] = (type, try reader.take(count))
        }
        guard attributes["compression"]?.1 == Data([0]),
              let channel = attributes["channels"], channel.0 == "chlist",
              let window = attributes["dataWindow"], window.0 == "box2i", window.1.count == 16 else {
            throw StudioError("Numeric source agreement requires an uncompressed scalar FLOAT32 EXR.")
        }
        var channelReader = EXRReader(bytes: channel.1)
        let name = try channelReader.string()
        guard ["Y", "R"].contains(name), try channelReader.word() == 2 else {
            throw StudioError("Numeric source agreement requires one scalar FLOAT32 EXR channel.")
        }
        _ = try channelReader.take(4)
        guard try channelReader.word() == 1, try channelReader.word() == 1,
              try channelReader.string().isEmpty, channelReader.offset == channel.1.count else {
            throw StudioError("Numeric source agreement does not accept subsampled or multichannel EXRs.")
        }
        var windowReader = EXRReader(bytes: window.1)
        let xMin = Int(Int32(bitPattern: try windowReader.word())), yMin = Int(Int32(bitPattern: try windowReader.word()))
        let xMax = Int(Int32(bitPattern: try windowReader.word())), yMax = Int(Int32(bitPattern: try windowReader.word()))
        let width = xMax - xMin + 1, height = yMax - yMin + 1
        guard width > 0, height > 0, width <= 16384, height <= 16384, width * height <= 150_000_000 else {
            throw StudioError("The scalar EXR has invalid pixel dimensions.")
        }
        let tableEnd = reader.offset + height * 8
        guard tableEnd <= bytes.count else { throw StudioError("The scalar EXR offset table is truncated.") }
        var values = [Float](repeating: 0, count: width * height), seen = Set<Int>()
        for _ in 0..<height {
            try Task.checkCancellation()
            let offset = try reader.longWord()
            guard offset <= UInt64(Int.max), offset >= UInt64(tableEnd) else { throw StudioError("The scalar EXR scanline offset is invalid.") }
            var scanline = EXRReader(bytes: bytes, offset: Int(offset))
            let row = Int(Int32(bitPattern: try scanline.word())) - yMin
            guard row >= 0, row < height, seen.insert(row).inserted,
                  try scanline.word() == UInt32(width * 4) else { throw StudioError("The scalar EXR scanline grid is invalid.") }
            let samples = try scanline.take(width * 4)
            samples.withUnsafeBytes { storage in
                for x in 0..<width {
                    values[row * width + x] = Float(bitPattern: UInt32(littleEndian: storage.loadUnaligned(fromByteOffset: x * 4, as: UInt32.self)))
                }
            }
        }
        return (width, height, values)
    }

    private struct EXRReader {
        let bytes: Data
        var offset = 0
        mutating func take(_ count: Int) throws -> Data {
            guard count >= 0, offset >= 0, offset <= bytes.count, count <= bytes.count - offset else {
                throw StudioError("The scalar EXR is truncated.")
            }
            defer { offset += count }
            return bytes.subdata(in: offset..<offset + count)
        }
        mutating func word() throws -> UInt32 { try take(4).withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) } }
        mutating func longWord() throws -> UInt64 { try take(8).withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) } }
        mutating func string() throws -> String {
            let start = offset
            guard start >= 0, start < bytes.count else { throw StudioError("The scalar EXR string is truncated.") }
            while offset < bytes.count, bytes[offset] != 0 { offset += 1 }
            guard offset < bytes.count, offset - start <= 255 else { throw StudioError("The scalar EXR string is invalid.") }
            defer { offset += 1 }
            return String(decoding: bytes[start..<offset], as: UTF8.self)
        }
    }
}
