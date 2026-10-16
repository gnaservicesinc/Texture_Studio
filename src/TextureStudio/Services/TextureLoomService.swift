import CoreImage
import CoreML
import CoreVideo
import Foundation
import Metal

struct TextureLoomPrediction: @unchecked Sendable {
    let categoryID: String
    let categoryName: String
    let confidence: Double
    let maps: [String: MaterialModelMap]
}

struct TextureLoomClassification: Sendable {
    let categoryID: String
    let categoryName: String
    let confidence: Double
}

actor TextureLoomService {
    private var models: [String: MLModel] = [:]
    private var temporaryCompilations: [URL] = []
    private let computeUnits: MLComputeUnits
    private let context = CIContext(options: [.workingFormat: CIFormat.RGBAf,
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, .cacheIntermediates: false])

    init(computeUnits: MLComputeUnits = .all) { self.computeUnits = computeUnits }

    isolated deinit {
        // Loaded models can retain mappings into compiled storage. Release them
        // before removing only the compiler outputs owned by this service.
        models.removeAll()
        for url in temporaryCompilations { try? FileManager.default.removeItem(at: url) }
    }

    func predict(diffuse: CIImage, package: TextureLoomPackage, overrides: [String: URL]) throws -> TextureLoomPrediction {
        try Task.checkCancellation()
        let width = Int(diffuse.extent.width), height = Int(diffuse.extent.height)
        guard width > 0, height > 0, width <= 8192, height <= 8192 else { throw StudioError("Texture Loom needs a supported prepared image size.") }
        let classification = try classify(diffuse: diffuse, package: package)
        guard let category = package.manifest.categories.first(where: { $0.id == classification.categoryID }) else {
            throw StudioError("The fixed Texture Loom classifier returned a dataset outside its model catalog.")
        }
        let compactModels = category.models.filter { $0.architecture == TextureLoomPackage.compactArchitecture }
        if !compactModels.isEmpty {
            try TextureLoomPackage.validateGrid(architecture: TextureLoomPackage.compactArchitecture, width: width, height: height)
            let metal = MTLCreateSystemDefaultDevice()
            let budget = TextureLoomPackage.compactMemoryBudget(physical: ProcessInfo.processInfo.physicalMemory,
                recommendedMetal: metal?.recommendedMaxWorkingSetSize, allocatedMetal: UInt64(metal?.currentAllocatedSize ?? 0),
                availableMemory: TextureEngine.availableMemoryBytes())
            let required = try compactModels.reduce(UInt64(0)) { sum, model in
                let forecast = TextureLoomPackage.compactMemoryForecast(width: width, height: height,
                    featureWidth: try package.compactFeatureWidth(model), map: model.map)
                let total = sum.addingReportingOverflow(forecast)
                return total.overflow ? .max : total.partialValue
            }
            guard required <= budget else {
                let gib = Double(1 << 30)
                throw StudioError(String(format: "Whole-field compact inference has a conservative memory forecast of %.1f GiB; the current budget is %.1f GiB. Use a compatible smaller native image or free memory before processing.", Double(required) / gib, Double(budget) / gib))
            }
        }
        let input = try imageTensor(diffuse, width: width, height: height)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: input)])
        var output: [String: MaterialModelMap] = [:]
        for entry in category.models {
            try Task.checkCancellation()
            try TextureLoomPackage.validateGrid(architecture: entry.architecture, width: width, height: height)
            let key = category.id + ":" + entry.map
            let url = try overrides[key] ?? TextureLoomPackage.resolve(entry.path, root: package.root)
            let model = try load(url)
            let prediction = try model.prediction(from: provider)
            guard let tensor = prediction.featureValue(for: "material_map")?.multiArrayValue else {
                throw StudioError("The Texture Loom \(entry.map) model did not return a material map.")
            }
            let target = entry.map == "depth" ? "height" : entry.map
            let image = try Self.materialImage(from: tensor, map: entry.map, architecture: entry.architecture,
                width: width, height: height)
            output[target] = MaterialModelMap(target: target, image: image, sourceLabel: "Texture Loom · \(category.name)")
        }
        return TextureLoomPrediction(categoryID: category.id, categoryName: category.name, confidence: classification.confidence, maps: output)
    }

    /// Compact exports have an immutable raw linear contract. Other installed
    /// model types retain their established bounded material-map interpretation.
    static func materialImage(from tensor: MLMultiArray, map: String, architecture: String,
                              width: Int, height: Int) throws -> CIImage {
        try TextureLoomPackage.validateGrid(architecture: architecture, width: width, height: height)
        let channels = map == "normal" ? 3 : 1
        guard tensor.shape.map(\.intValue) == [1, channels, height, width], tensor.dataType == .float32 else {
            throw StudioError("The Texture Loom \(map) map does not match the image grid.")
        }
        let components = channels == 3 ? 4 : 1
        let rawLinear = architecture == TextureLoomPackage.compactArchitecture
        var samples = [Float](repeating: 0, count: width * height * components)
        let strides = tensor.strides.map(\.intValue)
        let values = tensor.dataPointer.assumingMemoryBound(to: Float.self)
        for y in 0..<height {
            if y % 64 == 0 { try Task.checkCancellation() }
            for x in 0..<width {
                for channel in 0..<channels {
                    let value = values[channel * strides[1] + y * strides[2] + x * strides[3]]
                    guard value.isFinite else { throw StudioError("The Texture Loom \(map) model returned a nonfinite value.") }
                    samples[(y * width + x) * components + channel] = rawLinear ? value : min(1, max(0, value))
                }
                if components == 4 { samples[(y * width + x) * 4 + 3] = 1 }
            }
        }
        return CIImage(bitmapData: samples.withUnsafeBytes { Data($0) }, bytesPerRow: width * components * 4,
            size: CGSize(width: width, height: height), format: components == 4 ? .RGBAf : .Rf, colorSpace: nil)
    }

    func classify(diffuse: CIImage, package: TextureLoomPackage) throws -> TextureLoomClassification {
        try Task.checkCancellation()
        guard !diffuse.extent.isEmpty, !diffuse.extent.isInfinite else { throw StudioError("Texture Loom needs an image with finite dimensions.") }
        let entry = package.manifest.classifier
        let classifier = try load(TextureLoomPackage.resolve(entry.path, root: package.root))
        let description = classifier.modelDescription
        let inputName = entry.inputName ?? "rgb"
        let outputName = entry.outputName ?? "dataset"
        let provider: MLDictionaryFeatureProvider
        if let constraint = description.inputDescriptionsByName[inputName]?.imageConstraint {
            try TextureLoomPackage.validateImportedClassifier(description, classifier: entry)
            let buffer = try classifierPixels(diffuse, constraint: constraint)
            provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)])
        } else {
            guard let shape = description.inputDescriptionsByName[inputName]?.multiArrayConstraint?.shape.map(\.intValue),
                  shape.count == 4, shape[0] == 1, shape[1] == 3, shape[2] > 0, shape[3] > 0,
                  description.outputDescriptionsByName[outputName]?.type == .string else {
                throw StudioError("The fixed Texture Loom classifier has an unsupported image interface.")
            }
            let image = classifierThumbnail(diffuse, width: shape[3], height: shape[2])
            provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: imageTensor(image, width: shape[3], height: shape[2]))])
        }
        let result = try classifier.prediction(from: provider)
        guard let label = result.featureValue(for: outputName)?.stringValue,
              let catalogLabel = entry.labels.first(where: { $0.classifierValue == label }),
              let category = package.manifest.categories.first(where: { $0.id == catalogLabel.id }) else {
            throw StudioError("The fixed Texture Loom classifier returned a label outside its dataset catalog.")
        }
        let probabilityName = description.predictedProbabilitiesName ?? "probabilities"
        let confidence = result.featureValue(for: probabilityName)?.dictionaryValue[label]?.doubleValue ?? 0
        return TextureLoomClassification(categoryID: category.id, categoryName: category.name, confidence: confidence)
    }

    private func classifierThumbnail(_ image: CIImage, width: Int, height: Int) -> CIImage {
        image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(width) / image.extent.width, y: CGFloat(height) / image.extent.height))
    }

    private func classifierPixels(_ image: CIImage, constraint: MLImageConstraint) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, constraint.pixelsWide, constraint.pixelsHigh,
            constraint.pixelFormatType, [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else { throw StudioError("Could not prepare the fixed Texture Loom classifier's image input.") }
        let thumbnail = classifierThumbnail(image, width: constraint.pixelsWide, height: constraint.pixelsHigh)
        context.render(thumbnail, to: buffer, bounds: CGRect(x: 0, y: 0, width: constraint.pixelsWide, height: constraint.pixelsHigh),
                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return buffer
    }

    private func load(_ url: URL) throws -> MLModel {
        if let existing = models[url.path] { return existing }
        let compiled: URL
        if url.pathExtension == "mlmodelc" { compiled = url }
        else { compiled = try MLModel.compileModel(at: url) }
        let configuration = MLModelConfiguration(); configuration.computeUnits = computeUnits
        let model: MLModel
        do { model = try MLModel(contentsOf: compiled, configuration: configuration) }
        catch {
            if url.pathExtension != "mlmodelc" { try? FileManager.default.removeItem(at: compiled) }
            throw error
        }
        models[url.path] = model
        if url.pathExtension != "mlmodelc" { temporaryCompilations.append(compiled) }
        return model
    }
    func imageTensor(_ image: CIImage, width: Int, height: Int) throws -> sending MLMultiArray {
        var pixels = [Float](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBytes { context.render(image, toBitmap: $0.baseAddress!, rowBytes: width * 16,
            bounds: image.extent, format: .RGBAf, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) }
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let strides = array.strides.map(\.intValue)
        let values = array.dataPointer.assumingMemoryBound(to: Float.self)
        for channel in 0..<3 {
            for y in 0..<height {
                if y % 64 == 0 { try Task.checkCancellation() }
                for x in 0..<width { values[channel * strides[1] + y * strides[2] + x * strides[3]] = min(1, max(0, pixels[(y * width + x) * 4 + channel])) }
            }
        }
        return array
    }
}
