import CoreImage
import CoreML
import CoreVideo
import Foundation

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
    private let computeUnits: MLComputeUnits
    private let context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, .cacheIntermediates: false])

    init(computeUnits: MLComputeUnits = .all) { self.computeUnits = computeUnits }

    func predict(diffuse: CIImage, package: TextureLoomPackage, overrides: [String: URL]) throws -> TextureLoomPrediction {
        try Task.checkCancellation()
        let width = Int(diffuse.extent.width), height = Int(diffuse.extent.height)
        guard width > 0, height > 0, width <= 8192, height <= 8192 else { throw StudioError("Texture Loom needs a supported prepared image size.") }
        let input = try imageTensor(diffuse, width: width, height: height)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: input)])
        let classification = try classify(diffuse: diffuse, package: package)
        guard let category = package.manifest.categories.first(where: { $0.id == classification.categoryID }) else {
            throw StudioError("The fixed Texture Loom classifier returned a dataset outside its model catalog.")
        }
        var output: [String: MaterialModelMap] = [:]
        for entry in category.models {
            try Task.checkCancellation()
            let key = category.id + ":" + entry.map
            let url = try overrides[key] ?? TextureLoomPackage.resolve(entry.path, root: package.root)
            let model = try load(url)
            let prediction = try model.prediction(from: provider)
            guard let tensor = prediction.featureValue(for: "material_map")?.multiArrayValue else {
                throw StudioError("The Texture Loom \(entry.map) model did not return a material map.")
            }
            let channels = entry.map == "normal" ? 3 : 1
            let shape = tensor.shape.map(\.intValue)
            guard shape == [1, channels, height, width], tensor.dataType == .float32 else {
                throw StudioError("The Texture Loom \(entry.map) map does not match the image grid.")
            }
            let components = channels == 3 ? 4 : 1
            var samples = [Float](repeating: 0, count: width * height * components)
            let strides = tensor.strides.map(\.intValue)
            let values = tensor.dataPointer.assumingMemoryBound(to: Float.self)
            for y in 0..<height {
                if y % 64 == 0 { try Task.checkCancellation() }
                for x in 0..<width {
                    for channel in 0..<channels {
                        let value = values[channel * strides[1] + y * strides[2] + x * strides[3]]
                        guard value.isFinite else { throw StudioError("The Texture Loom \(entry.map) model returned a nonfinite value.") }
                        samples[(y * width + x) * components + channel] = min(1, max(0, value))
                    }
                    if components == 4 { samples[(y * width + x) * 4 + 3] = 1 }
                }
            }
            let target = entry.map == "depth" ? "height" : entry.map
            let image = CIImage(bitmapData: samples.withUnsafeBytes { Data($0) }, bytesPerRow: width * components * 4,
                size: CGSize(width: width, height: height), format: components == 4 ? .RGBAf : .Rf, colorSpace: nil)
            output[target] = MaterialModelMap(target: target, image: image, sourceLabel: "Texture Loom · \(category.name)")
        }
        return TextureLoomPrediction(categoryID: category.id, categoryName: category.name, confidence: classification.confidence, maps: output)
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
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        models[url.path] = model
        return model
    }
    private func imageTensor(_ image: CIImage, width: Int, height: Int) throws -> MLMultiArray {
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
