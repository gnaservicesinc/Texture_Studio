import Foundation
import CryptoKit
import CoreML

/// Every model path is rooted in an immutable build payload or an app-owned adapter.
/// There is no user preference that can replace the built-in base model.
struct TextureLoomPackage: Sendable {
    let root: URL
    let manifest: TextureLoomManifest
    init(root: URL) throws {
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        manifest = try JSONDecoder().decode(TextureLoomManifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
        guard manifest.schema == "texture-loom.package.v1", UUID(uuidString: manifest.id) != nil,
              !manifest.name.isEmpty, !manifest.version.isEmpty, !manifest.categories.isEmpty,
              Set(manifest.categories.map(\.id)).count == manifest.categories.count,
              Set(manifest.classifier.labels.map(\.id)).count == manifest.classifier.labels.count,
              Set(manifest.classifier.labels.map(\.classifierValue)).count == manifest.classifier.labels.count,
              Set(manifest.classifier.labels.map(\.id)) == Set(manifest.categories.map(\.id)),
              manifest.classifier.labels.allSatisfy({ label in
                  !label.classifierValue.isEmpty && manifest.categories.contains { $0.id == label.id && $0.name == label.name }
              }),
              manifest.classifier.format == "coreml" else {
            throw StudioError("The built-in Texture Loom package has an invalid identity or dataset catalog.")
        }
        try Self.verify(files: manifest.files, root: self.root)
        try requireArtifact(manifest.classifier.path)
        if let weights = manifest.classifier.weightsPath { try requireArtifact(weights) }
        if manifest.classifier.architecture == "create-ml-image-classifier-v1" {
            let classifierURL = try Self.resolve(manifest.classifier.path, root: self.root)
            let compiled = classifierURL.pathExtension == "mlmodelc" ? classifierURL : try MLModel.compileModel(at: classifierURL)
            let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
            let classifier = try MLModel(contentsOf: compiled, configuration: configuration)
            try Self.validateImportedClassifier(classifier.modelDescription, classifier: manifest.classifier)
        }
        for category in manifest.categories {
            guard !category.name.isEmpty, !category.version.isEmpty, !category.models.isEmpty,
                  Set(category.models.map(\.map)).count == category.models.count else {
                throw StudioError("The Texture Loom dataset \(category.name) has an invalid model catalog.")
            }
            for model in category.models {
                guard ["depth", "roughness", "normal"].contains(model.map), model.format == "coreml" else {
                    throw StudioError("The Texture Loom package contains an unsupported material model.")
                }
                try requireArtifact(model.path)
                try requireArtifact(model.weightsPath)
            }
        }
    }
    static func validateImportedClassifier(_ description: MLModelDescription, classifier: TextureLoomManifest.Classifier) throws {
        guard let input = classifier.inputName, let output = classifier.outputName,
              description.inputDescriptionsByName.count == 1,
              let feature = description.inputDescriptionsByName[input],
              description.outputDescriptionsByName[output]?.type == .string,
              let labels = description.classLabels as? [String], !labels.isEmpty,
              Set(labels).count == labels.count,
              Set(labels) == Set(classifier.labels.map(\.classifierValue)) else {
            throw StudioError("The fixed Texture Loom classifier's interface or labels do not match its dataset catalog. Rebuild Texture Studio with a complete Loom Factory export.")
        }
        if let image = feature.imageConstraint {
            guard image.pixelsWide > 0, image.pixelsHigh > 0, image.pixelsWide <= 8192, image.pixelsHigh <= 8192,
                  classifier.inputWidth.map({ $0 == image.pixelsWide }) ?? true,
                  classifier.inputHeight.map({ $0 == image.pixelsHigh }) ?? true,
                  classifier.inputType.map({ $0 == "image" }) ?? true,
                  classifier.inputColorSpace.map({ $0 == "RGB" }) ?? true else {
                throw StudioError("The fixed Texture Loom classifier's image dimensions do not match its native model.")
            }
        } else if let array = feature.multiArrayConstraint {
            let shape = array.shape.map(\.intValue)
            guard array.dataType == .float32, shape.count == 4, shape[0] == 1, shape[1] == 3,
                  shape[2] > 0, shape[3] > 0, shape[2] <= 8192, shape[3] <= 8192,
                  classifier.inputType.map({ $0 == "multiArray" }) ?? true,
                  classifier.inputShape.map({ $0 == shape }) ?? true else {
                throw StudioError("The fixed Texture Loom classifier requires a supported RGB tensor input.")
            }
        } else { throw StudioError("The fixed Texture Loom classifier requires an image or RGB tensor input.") }
    }
    func requireArtifact(_ path: String) throws {
        let url = try Self.resolve(path, root: root)
        guard FileManager.default.fileExists(atPath: url.path),
              manifest.files.contains(where: { $0.path == path || $0.path.hasPrefix(path + "/") }) else {
            throw StudioError("The built-in Texture Loom model is incomplete: \(path). Rebuild Texture Studio with a complete Loom Factory export.")
        }
        if (try url.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
            guard let entries = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else {
                throw StudioError("Could not inspect the Texture Loom artifact: \(path).")
            }
            let inventory = Set(manifest.files.map(\.path))
            for case let file as URL in entries {
                let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                guard values.isSymbolicLink != true else { throw StudioError("Model exports cannot contain links.") }
                if values.isRegularFile == true, file.lastPathComponent != ".DS_Store" {
                    let relative = String(file.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(root.standardizedFileURL.resolvingSymlinksInPath().path.count + 1))
                    guard inventory.contains(relative) else { throw StudioError("The Texture Loom artifact contains an unregistered file: \(relative).") }
                }
            }
        }
    }
    func modelHash(_ model: TextureLoomManifest.Model) throws -> String {
        try Self.sha256(Self.resolve(model.weightsPath, root: root))
    }
    static func resolve(_ path: String, root: URL) throws -> URL {
        let pieces = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, !path.contains("\\"), !path.hasPrefix("/"),
              pieces.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw StudioError("The model package contains a path outside its own folder.")
        }
        var cursor = root.standardizedFileURL.resolvingSymlinksInPath()
        for piece in pieces {
            cursor.appendPathComponent(String(piece))
            guard (try? FileManager.default.destinationOfSymbolicLink(atPath: cursor.path)) == nil else {
                throw StudioError("Model packages cannot contain symbolic links.")
            }
        }
        let result = cursor.standardizedFileURL
        guard result.path.hasPrefix(root.standardizedFileURL.resolvingSymlinksInPath().path + "/") else {
            throw StudioError("The model package contains a link outside its own folder.")
        }
        return result
    }
    static func verify(files: [TextureLoomManifest.File], root: URL) throws {
        guard !files.isEmpty, Set(files.map(\.path)).count == files.count else {
            throw StudioError("The model package file inventory is empty or repeated.")
        }
        for file in files {
            let url = try resolve(file.path, root: root)
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]))?.isRegularFile == true,
                  try sha256(url) == file.sha256.lowercased() else {
                throw StudioError("The model package file is missing or changed: \(file.path).")
            }
        }
    }
    static func sha256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty { digest.update(data: chunk) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}
