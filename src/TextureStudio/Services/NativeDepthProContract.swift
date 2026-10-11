import CoreGraphics
import CryptoKit
import Foundation
import ImageIO

/// This version identifies numeric direction, scaling, coordinates and interpolation.
/// It is shared by cached training inputs and photograph inference.
enum NativeDepthProContract {
    static let identifier = "texture-studio-depthpro-inverse-depth-v1"
    static let modelPathKey = "depthProModelPath"
    static var managedModelURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/DepthPro/DepthPro.mlpackage")
    }
    static var configuredModelURL: URL {
        configuredModelURL(in: StudioPreferences.defaults)
    }
    static func configuredModelURL(in preferences: UserDefaults, migrating legacyPreferences: UserDefaults? = nil) -> URL {
        var path = preferences.string(forKey: modelPathKey) ?? ""
        if path.isEmpty, let legacyPath = legacyPreferences?.string(forKey: modelPathKey), !legacyPath.isEmpty {
            path = legacyPath
            preferences.set(path, forKey: modelPathKey)
        }
        if !path.isEmpty { return URL(fileURLWithPath: path) }
        return managedModelURL
    }
    static var cacheURL: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/DepthPro Inputs")
    }

    /// DepthPro camera meters are positive with nearer surfaces having lower values.
    /// Restore its complete field to the paired grid, then invert and normalize once.
    /// Neither target pixels nor source maps are modified by this preparation.
    static func prepare(_ depth: ModelDepthResult, width: Int, height: Int) throws -> [Float] {
        guard depth.width > 0, depth.height > 0, depth.width <= 4096, depth.height <= 4096, width > 0, height > 0,
              width <= 8192, height <= 8192, depth.values.count == depth.width * depth.height,
              depth.values.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw StudioError("DepthPro must return finite positive camera depth in meters.")
        }
        var inverse = [Double](repeating: 0, count: width * height)
        for y in 0..<height {
            try Task.checkCancellation()
            let fy = max(0, min(Double(depth.height - 1), (Double(y) + 0.5) * Double(depth.height) / Double(height) - 0.5))
            let y0 = Int(fy), y1 = min(y0 + 1, depth.height - 1), ay = fy - Double(y0)
            for x in 0..<width {
                let fx = max(0, min(Double(depth.width - 1), (Double(x) + 0.5) * Double(depth.width) / Double(width) - 0.5))
                let x0 = Int(fx), x1 = min(x0 + 1, depth.width - 1), ax = fx - Double(x0)
                let a = Double(depth.values[y0 * depth.width + x0]), b = Double(depth.values[y0 * depth.width + x1])
                let c = Double(depth.values[y1 * depth.width + x0]), d = Double(depth.values[y1 * depth.width + x1])
                inverse[y * width + x] = 1 / ((a + (b - a) * ax) * (1 - ay) + (c + (d - c) * ax) * ay)
            }
        }
        let low = inverse.min()!, high = inverse.max()!
        guard low.isFinite, high.isFinite else { throw StudioError("DepthPro scale cannot be represented.") }
        if high == low { return .init(repeating: 0, count: width * height) }
        return inverse.map { Float(($0 - low) / (high - low)) }
    }

    /// Paired flips only permute stored samples. They preserve extrema and every
    /// Float32 bit, so they must not interpolate or normalize the field again.
    static func flipped(_ values: [Float], width: Int, height: Int, horizontal: Bool, vertical: Bool,
                        checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> [Float] {
        guard width > 0, height > 0, width <= 8192, height <= 8192, values.count == width * height else {
            throw StudioError("DepthPro flip must preserve the exact numeric grid.")
        }
        try checkCancellation()
        guard horizontal || vertical else { return values }
        var result = values
        for y in 0..<height {
            try checkCancellation()
            let sourceY = vertical ? height - 1 - y : y
            for x in 0..<width {
                let sourceX = horizontal ? width - 1 - x : x
                result[y * width + x] = values[sourceY * width + sourceX]
            }
        }
        return result
    }
}

/// Stores numeric tensors, never display grayscale. Cache identity binds source
/// bytes, model bytes, dimensions and the complete preparation version.
struct NativeDepthProRuntimeError: LocalizedError {
    let underlying: Error
    var errorDescription: String? { "DepthPro could not prepare its input: \(underlying.localizedDescription)" }
}

final class NativeDepthProInputCache {
    let modelSHA256: String
    let directory: URL
    let fallbackDirectories: [URL]
    private let modelURL: URL?
    private var service: NativeDepthProService?
    private let suppliedPrediction: ((CGImage) throws -> ModelDepthResult)?
    private(set) var generatedCount = 0
    private(set) var reusedCount = 0
    private(set) var remappedCount = 0
    private(set) var legacyReusedCount = 0

    init(modelURL: URL, directory: URL = NativeDepthProContract.cacheURL, fallbackDirectories: [URL] = []) throws {
        self.modelURL = modelURL; self.directory = directory
        self.fallbackDirectories = fallbackDirectories.filter { $0.standardizedFileURL != directory.standardizedFileURL }
        modelSHA256 = try NativeDepthProService.modelIdentity(at: modelURL)
        suppliedPrediction = nil
    }
    init(modelSHA256: String, directory: URL, fallbackDirectories: [URL] = [], predict: @escaping (CGImage) throws -> ModelDepthResult) {
        self.modelSHA256 = modelSHA256; self.directory = directory
        self.fallbackDirectories = fallbackDirectories.filter { $0.standardizedFileURL != directory.standardizedFileURL }
        modelURL = nil; suppliedPrediction = predict
    }

    func releasePredictor() { service = nil }

    /// Carry an existing prediction through the same paired image flips. A
    /// rotated/re-cropped source must instead use input() with its new hash.
    /// Missing pre-flip entries return false; damaged entries remain fatal.
    @discardableResult
    func remapFlips(fromSourceSHA256: String, toSourceSHA256: String, width: Int, height: Int,
                    horizontal: Bool, vertical: Bool,
                    checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> Bool {
        try checkCancellation()
        let original = location(sourceSHA256: fromSourceSHA256, width: width, height: height)
        do {
            guard let cached = try reusableEntry(at: original.url, binding: original.binding, width: width, height: height,
                                                 checkCancellation: checkCancellation) else { return false }
            if !horizontal, !vertical, fromSourceSHA256 == toSourceSHA256 { return true }
            let input = NativeTensor.floats(try NativeDepthProContract.flipped(cached.inputValues,
                width: width, height: height, horizontal: horizontal, vertical: vertical,
                checkCancellation: checkCancellation), shape: [1, 1, height, width])
            let metric = NativeTensor.floats(try NativeDepthProContract.flipped(cached.metricValues,
                width: cached.metricWidth, height: cached.metricHeight, horizontal: horizontal, vertical: vertical,
                checkCancellation: checkCancellation), shape: [1, 1, cached.metricHeight, cached.metricWidth])
            let destination = location(sourceSHA256: toSourceSHA256, width: width, height: height)
            var metadata = cached.metadata
            metadata["binding"] = destination.binding
            metadata["source_sha256"] = toSourceSHA256
            metadata["input_sha256"] = NativeMaterialTrainer.checksum(input.bytes)
            metadata["metric_depth_sha256"] = NativeMaterialTrainer.checksum(metric.bytes)
            metadata["remapped_from_source_sha256"] = fromSourceSHA256
            metadata["paired_flip_horizontal"] = String(horizontal)
            metadata["paired_flip_vertical"] = String(vertical)
            let encoded = try NativeSafetensors.encoded(tensors: ["input": input, "depth_meters": metric], metadata: metadata)
            try checkCancellation()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try encoded.write(to: destination.url, options: .atomic)
            remappedCount += 1
            return true
        } catch {
            try checkCancellation()
            if error is CancellationError { throw error }
            throw NativeDepthProRuntimeError(underlying: error)
        }
    }

    private func location(sourceSHA256: String, width: Int, height: Int) -> (binding: String, url: URL) {
        let binding = [NativeDepthProContract.identifier, sourceSHA256, modelSHA256, String(width), String(height)].joined(separator: "|")
        let key = NativeMaterialTrainer.checksum(Data(binding.utf8))
        return (binding, directory.appendingPathComponent(key + ".safetensors"))
    }

    private struct CachedEntry {
        let inputValues: [Float], metricValues: [Float]
        let metricWidth: Int, metricHeight: Int
        let metadata: [String: String]
    }

    /// Older runs used a shared cache. Validate a requested entry before reuse,
    /// then link it into this dataset when possible without copying the corpus.
    /// Across volumes it remains a read-only fallback; new predictions are local.
    private func reusableEntry(at destination: URL, binding: String, width: Int, height: Int,
                               checkCancellation: () throws -> Void) throws -> CachedEntry? {
        if FileManager.default.fileExists(atPath: destination.path) {
            return try cachedEntry(at: destination, binding: binding, width: width, height: height)
        }
        for fallback in fallbackDirectories {
            let source = fallback.appendingPathComponent(destination.lastPathComponent)
            guard FileManager.default.fileExists(atPath: source.path) else { continue }
            let cached = try cachedEntry(at: source, binding: binding, width: width, height: height)
            try checkCancellation()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            do { try FileManager.default.linkItem(at: source, to: destination) }
            catch {
                // Another run may already have published this identity.
                if FileManager.default.fileExists(atPath: destination.path) {
                    return try cachedEntry(at: destination, binding: binding, width: width, height: height)
                }
                // A hard link may be unavailable across volumes or on the
                // dataset filesystem. The verified legacy entry is still usable.
            }
            legacyReusedCount += 1
            return cached
        }
        return nil
    }

    private func cachedEntry(at url: URL, binding: String, width: Int, height: Int) throws -> CachedEntry {
        let cached = try NativeSafetensors(contentsOf: url)
        guard cached.metadata["binding"] == binding, cached.metadata["preparation_contract"] == NativeDepthProContract.identifier,
              cached.metadata["depthpro_model_sha256"] == modelSHA256, let tensor = cached.tensors["input"],
              tensor.dtype == "F32", tensor.shape == [1, 1, height, width] else {
            throw StudioError("DepthPro cache identity or grid is invalid. Remove the damaged cache entry: \(url.path)")
        }
        let input = NativeTensor(dtype: "F32", shape: tensor.shape, bytes: try cached.tensorBytes(named: "input"))
        let values = try input.floatValues()
        guard NativeMaterialTrainer.checksum(input.bytes) == cached.metadata["input_sha256"],
              values.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw StudioError("DepthPro cached input failed its numeric checksum.")
        }
        guard let metric = cached.tensors["depth_meters"], metric.dtype == "F32", metric.shape.count == 4,
              metric.shape[0...1] == [1, 1], metric.shape[2] > 0, metric.shape[3] > 0,
              metric.shape[2] <= 4096, metric.shape[3] <= 4096 else {
            throw StudioError("DepthPro cached metric-depth tensor is invalid.")
        }
        let metricBytes = try cached.tensorBytes(named: "depth_meters")
        let metricValues = try NativeTensor(dtype: "F32", shape: metric.shape, bytes: metricBytes).floatValues()
        guard NativeMaterialTrainer.checksum(metricBytes) == cached.metadata["metric_depth_sha256"],
              metricValues.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw StudioError("DepthPro cached meters failed their numeric checksum.")
        }
        return CachedEntry(inputValues: values, metricValues: metricValues,
            metricWidth: metric.shape[3], metricHeight: metric.shape[2], metadata: cached.metadata)
    }

    func input(imageURL: URL, expectedSHA256: String, width: Int, height: Int,
               checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> [Float] {
        try checkCancellation()
        let bytes = try Data(contentsOf: imageURL)
        guard NativeMaterialTrainer.checksum(bytes) == expectedSHA256 else {
            throw StudioError("The diffuse source changed before DepthPro preparation.")
        }
        let entry = location(sourceSHA256: expectedSHA256, width: width, height: height)
        let binding = entry.binding, destination = entry.url
        do {
            if let cached = try reusableEntry(at: destination, binding: binding, width: width, height: height,
                                              checkCancellation: checkCancellation) {
            reusedCount += 1
            return cached.inputValues
            }
        } catch {
            try checkCancellation()
            if error is CancellationError { throw error }
            throw NativeDepthProRuntimeError(underlying: error)
        }
        guard let source = CGImageSourceCreateWithData(bytes as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              image.width == width, image.height == height else {
            throw StudioError("DepthPro input must match the exact prepared diffuse grid.")
        }
        let depth: ModelDepthResult
        do {
            if let suppliedPrediction { depth = try suppliedPrediction(image) }
            else {
                if service == nil { service = try NativeDepthProService(modelURL: modelURL!) }
                guard service!.modelSHA256 == modelSHA256 else { throw StudioError("DepthPro model changed during preparation.") }
                depth = try service!.predict(image: image)
            }
        } catch {
            try checkCancellation()
            if error is CancellationError { throw error }
            throw NativeDepthProRuntimeError(underlying: error)
        }
        try checkCancellation()
        let values: [Float]
        do { values = try NativeDepthProContract.prepare(depth, width: width, height: height) }
        catch { if error is CancellationError { throw error }; throw NativeDepthProRuntimeError(underlying: error) }
        let input = NativeTensor.floats(values, shape: [1, 1, height, width])
        let raw = NativeTensor.floats(depth.values, shape: [1, 1, depth.height, depth.width])
        let encoded = try NativeSafetensors.encoded(tensors: ["input": input, "depth_meters": raw], metadata: [
            "binding": binding, "preparation_contract": NativeDepthProContract.identifier,
            "source_sha256": expectedSHA256, "depthpro_model_sha256": modelSHA256,
            "input_sha256": NativeMaterialTrainer.checksum(input.bytes),
            "metric_depth_sha256": NativeMaterialTrainer.checksum(raw.bytes),
            "coordinates": "top-left row-major; complete field; pixel-center bilinear meters before inverse min-max",
            "constant_field": "zero", "target": "original linear height codes / code maximum"])
        try checkCancellation()
        guard NativeMaterialTrainer.checksum(try Data(contentsOf: imageURL)) == expectedSHA256 else {
            throw StudioError("The diffuse source changed during DepthPro prediction.")
        }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try encoded.write(to: destination, options: .atomic)
        } catch { throw NativeDepthProRuntimeError(underlying: error) }
        generatedCount += 1
        return values
    }
}
