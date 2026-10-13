#!/usr/bin/env swift
import Foundation
import CoreML
import CryptoKit

struct Failure: Error, CustomStringConvertible { let description: String }
let fm = FileManager.default
func command(_ executable: String, _ args: [String]) throws -> String {
    let p = Process(); p.executableURL = URL(fileURLWithPath: executable); p.arguments = args
    let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
    try p.run(); let bytes = pipe.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit()
    guard p.terminationStatus == 0 else { throw Failure(description: String(decoding: bytes, as: UTF8.self)) }
    return String(decoding: bytes, as: UTF8.self)
}
func path(_ relative: String, _ root: URL) throws -> URL {
    guard !relative.isEmpty, !relative.hasPrefix("/"), !relative.contains("\\"),
          relative.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
        throw Failure(description: "Texture Loom package path escapes the package: \(relative)")
    }
    var target = root.standardizedFileURL.resolvingSymlinksInPath()
    for piece in relative.split(separator: "/") {
        target.appendPathComponent(String(piece))
        guard (try? fm.destinationOfSymbolicLink(atPath: target.path)) == nil else { throw Failure(description: "Texture Loom exports cannot contain links.") }
    }
    guard target.path.hasPrefix(root.standardizedFileURL.resolvingSymlinksInPath().path + "/") else { throw Failure(description: "Texture Loom package contains an external link.") }
    return target
}
func hash(_ file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
    var sha = SHA256()
    while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty { sha.update(data: chunk) }
    return sha.finalize().map { String(format: "%02x", $0) }.joined()
}
func verify(_ manifest: [String: Any], _ root: URL) throws {
    guard manifest["schema"] as? String == "texture-loom.package.v1", let id = manifest["id"] as? String, UUID(uuidString: id) != nil,
          let files = manifest["files"] as? [[String: String]], !files.isEmpty else { throw Failure(description: "Invalid Texture Loom package manifest.") }
    var inventory = Set<String>()
    for file in files {
        guard let relative = file["path"], let sha = file["sha256"], inventory.insert(relative).inserted else {
            throw Failure(description: "Texture Loom package inventory repeats a path or omits its identity.")
        }
        let actual = try hash(path(relative, root))
        guard actual == sha.lowercased() else { throw Failure(description: "Texture Loom package file changed: \(relative). Expected \(sha.prefix(12)), found \(actual.prefix(12)).") }
    }
    guard let categories = manifest["categories"] as? [[String: Any]], !categories.isEmpty,
          let classifier = manifest["classifier"] as? [String: Any], let labels = classifier["labels"] as? [[String: String]],
          Set(categories.compactMap { $0["id"] as? String }).count == categories.count,
          Set(labels.compactMap { $0["id"] }).count == labels.count,
          Set(labels.compactMap { $0["sourceLabel"] ?? $0["id"] }).count == labels.count,
          Set(categories.compactMap { $0["id"] as? String }) == Set(labels.compactMap { $0["id"] }),
          labels.allSatisfy({ label in categories.contains { ($0["id"] as? String) == label["id"] && ($0["name"] as? String) == label["name"] } }) else { throw Failure(description: "Texture Loom classifier does not match the datasets.") }
    let artifacts = [classifier] + categories.flatMap { $0["models"] as? [[String: Any]] ?? [] }
    for artifact in artifacts {
        guard let native = artifact["path"] as? String else { throw Failure(description: "Texture Loom is missing a native model artifact.") }
        var required = [native]
        if let weights = artifact["weightsPath"] as? String { required.append(weights) }
        else if artifact["labels"] == nil { throw Failure(description: "Texture Loom map models must include their training weights.") }
        for relative in required {
            let file = try path(relative, root)
            guard fm.fileExists(atPath: file.path), inventory.contains(where: { $0 == relative || $0.hasPrefix(relative + "/") }) else {
                throw Failure(description: "Texture Loom has an unregistered model artifact: \(relative).")
            }
        }
    }
}
func main() throws {
    guard CommandLine.arguments.count == 3 else { throw Failure(description: "usage: stage_texture_loom.swift APP PACKAGE") }
    let app = URL(fileURLWithPath: CommandLine.arguments[1])
    let resources = app.appendingPathComponent("Contents/Resources")
    let infoData = try Data(contentsOf: app.appendingPathComponent("Contents/Info.plist"))
    guard let info = try PropertyListSerialization.propertyList(from: infoData, format: nil) as? [String: Any], info["CFBundleIdentifier"] as? String == "org.ipde.texture-studio" else { throw Failure(description: "Unexpected application bundle.") }
    // Known generated products from the retired suite cannot survive an incremental build.
    for suffix in ["Contents/Applications", "Contents/PlugIns/TextureStudioTests.xctest", "Contents/Resources/DiffuseTextureTypeClassifier.mlmodelc", "Contents/Resources/MaterialReview.icns", "Contents/Resources/CheckpointCompare.icns", "Contents/Resources/MaterialDataset.icns", "Contents/Resources/MaterialTrainer.icns", "Contents/Resources/SCUNet_LICENSE", "Contents/Resources/NAFNet_LICENSE", "Contents/Resources/PBRnxt_LICENSE", "Contents/Resources/SwinTransformer_LICENSE", "Contents/Resources/ESRGANplus_LICENSE", "Contents/Resources/ModelLicenses/SCUNet_LICENSE", "Contents/Resources/ModelLicenses/NAFNet_LICENSE", "Contents/Resources/ModelLicenses/PBRnxt_LICENSE", "Contents/Resources/ModelLicenses/SwinTransformer_LICENSE", "Contents/Resources/ModelLicenses/ESRGANplus_LICENSE"] {
        let obsolete = app.appendingPathComponent(suffix)
        if fm.fileExists(atPath: obsolete.path) { try fm.removeItem(at: obsolete) }
    }
    let destination = resources.appendingPathComponent("TextureLoom", isDirectory: true)
    if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
    let provided = CommandLine.arguments[2]
    guard !provided.isEmpty else { print("Texture Studio shell built without a Texture Loom package. Set TEXTURE_LOOM_PACKAGE to a Loom Factory export for inference."); return }
    let source = URL(fileURLWithPath: provided)
    let scratch = fm.temporaryDirectory.appendingPathComponent("texture-loom-build-\(UUID().uuidString)")
    try fm.createDirectory(at: scratch, withIntermediateDirectories: true); defer { try? fm.removeItem(at: scratch) }
    let root: URL
    if (try source.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true { root = source }
    else {
        let listing = try command("/usr/bin/zipinfo", ["-1", source.path])
        let permissions = try command("/usr/bin/zipinfo", ["-l", source.path])
        guard !permissions.split(separator: "\n").contains(where: { $0.hasPrefix("l") }) else { throw Failure(description: "Texture Loom archives cannot contain links.") }
        let names = listing.split(separator: "\n").map(String.init)
        guard !names.isEmpty, names.count <= 10000 else { throw Failure(description: "Invalid Texture Loom archive inventory.") }
        for name in names { _ = try path(name.hasSuffix("/") ? String(name.dropLast()) : name, scratch) }
        _ = try command("/usr/bin/ditto", ["-x", "-k", source.path, scratch.path]); root = scratch
    }
    guard var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("manifest.json"))) as? [String: Any],
          let classifier = manifest["classifier"] as? [String: Any],
          let categories = manifest["categories"] as? [[String: Any]] else { throw Failure(description: "Invalid Texture Loom manifest structure.") }
    try verify(manifest, root)
    try fm.copyItem(at: root, to: destination)
    func compile(_ entry: [String: Any]) throws -> [String: Any] {
        var entry = entry
        guard entry["format"] as? String == "coreml", let relative = entry["path"] as? String else { throw Failure(description: "Every Texture Loom runtime model must export Core ML.") }
        let model = try path(relative, destination)
        let compiledURL: URL
        if model.pathExtension != "mlmodelc" {
            let compiled = try MLModel.compileModel(at: model)
            let newPath = String(relative.dropLast(model.pathExtension.count)) + "mlmodelc"
            let target = try path(newPath, destination)
            try fm.copyItem(at: compiled, to: target); try fm.removeItem(at: model); entry["path"] = newPath
            compiledURL = target
        } else { compiledURL = model }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let loaded = try MLModel(contentsOf: compiledURL, configuration: configuration)
        if entry["labels"] != nil {
            let inputName = entry["inputName"] as? String ?? "rgb"
            let outputName = entry["outputName"] as? String ?? "dataset"
            guard loaded.modelDescription.outputDescriptionsByName[outputName]?.type == .string else {
                throw Failure(description: "Texture Loom classifier requires its declared string label output.")
            }
            if loaded.modelDescription.inputDescriptionsByName[inputName]?.type == .image {
                guard loaded.modelDescription.inputDescriptionsByName.count == 1,
                      let constraint = loaded.modelDescription.inputDescriptionsByName[inputName]?.imageConstraint,
                      constraint.pixelsWide > 0, constraint.pixelsHigh > 0,
                      constraint.pixelsWide <= 8192, constraint.pixelsHigh <= 8192,
                      (entry["inputWidth"] as? Int).map({ $0 == constraint.pixelsWide }) ?? true,
                      (entry["inputHeight"] as? Int).map({ $0 == constraint.pixelsHigh }) ?? true,
                      (entry["inputType"] as? String).map({ $0 == "image" }) ?? true,
                      (entry["inputColorSpace"] as? String).map({ $0 == "RGB" }) ?? true,
                      let labels = entry["labels"] as? [[String: String]],
                      let actual = loaded.modelDescription.classLabels as? [String], !actual.isEmpty,
                      Set(actual).count == actual.count,
                      Set(actual) == Set(labels.compactMap { $0["sourceLabel"] ?? $0["id"] }) else {
                    throw Failure(description: "Texture Loom's fixed image classifier labels or interface do not match its dataset catalog.")
                }
                entry["inputWidth"] = constraint.pixelsWide
                entry["inputHeight"] = constraint.pixelsHigh
                entry["inputColorSpace"] = "RGB"
                entry["inputType"] = "image"
            } else {
                guard let constraint = loaded.modelDescription.inputDescriptionsByName[inputName]?.multiArrayConstraint,
                      constraint.dataType == .float32, constraint.shape.count == 4,
                      constraint.shape[0].intValue == 1, constraint.shape[1].intValue == 3,
                      constraint.shape[2].intValue > 0, constraint.shape[3].intValue > 0,
                      constraint.shape[2].intValue <= 8192, constraint.shape[3].intValue <= 8192,
                      (entry["inputType"] as? String).map({ $0 == "multiArray" }) ?? true,
                      (entry["inputShape"] as? [Int]).map({ $0 == constraint.shape.map(\.intValue) }) ?? true,
                      loaded.modelDescription.outputDescriptionsByName[loaded.modelDescription.predictedProbabilitiesName ?? "probabilities"]?.type == .dictionary else {
                    throw Failure(description: "Texture Loom tensor classifier requires its declared RGB input and probability output.")
                }
                if entry["architecture"] as? String == "create-ml-image-classifier-v1" {
                    guard loaded.modelDescription.inputDescriptionsByName.count == 1,
                          let actual = loaded.modelDescription.classLabels as? [String], !actual.isEmpty,
                          Set(actual).count == actual.count else {
                        throw Failure(description: "Texture Loom's imported classifier is missing its fixed native class catalog.")
                    }
                }
                if let labels = entry["labels"] as? [[String: String]],
                   let actual = loaded.modelDescription.classLabels as? [String] {
                    guard Set(actual) == Set(labels.compactMap { $0["sourceLabel"] ?? $0["id"] }) else {
                        throw Failure(description: "Texture Loom's tensor classifier labels do not match its dataset catalog.")
                    }
                }
                entry["inputType"] = "multiArray"
                entry["inputShape"] = constraint.shape.map(\.intValue)
            }
        } else {
            guard loaded.modelDescription.inputDescriptionsByName["rgb"]?.type == .multiArray,
                  loaded.modelDescription.outputDescriptionsByName["material_map"]?.type == .multiArray else { throw Failure(description: "Texture Loom material models require the rgb NCHW input and material_map tensor output.") }
        }
        return entry
    }
    manifest["classifier"] = try compile(classifier)
    manifest["categories"] = try categories.map { category in
        var result = category
        guard let models = category["models"] as? [[String: Any]], !models.isEmpty else { throw Failure(description: "Each classifier dataset needs material models.") }
        result["models"] = try models.map(compile)
        return result
    }
    var files: [[String: String]] = []
    guard let entries = fm.enumerator(at: destination, includingPropertiesForKeys: [.isRegularFileKey]) else { throw Failure(description: "Could not inspect the compiled Texture Loom payload.") }
    while let file = entries.nextObject() as? URL {
        if file.standardizedFileURL.resolvingSymlinksInPath().path == destination.appendingPathComponent("manifest.json").standardizedFileURL.resolvingSymlinksInPath().path { continue }
        if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            let relative = String(file.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(destination.standardizedFileURL.resolvingSymlinksInPath().path.count + 1)); files.append(["path": relative, "sha256": try hash(file)])
        }
    }
    manifest["files"] = files.sorted { ($0["path"] ?? "") < ($1["path"] ?? "") }
    try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]).write(to: destination.appendingPathComponent("manifest.json"), options: .atomic)
    try verify(manifest, destination)
    print("Bundled immutable Texture Loom \(manifest["name"] ?? "") · \(manifest["version"] ?? "")")
}
do { try main() } catch { fputs("Texture Loom build integration failed: \(error)\n", stderr); exit(1) }
