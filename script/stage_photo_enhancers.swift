import CoreML
import CryptoKit
import Foundation
import Darwin

// Independent, explicit build-time products. No fallback to material maps.
func fail(_ message: String) throws -> Never { throw NSError(domain: "PhotoEnhancers", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
func hash(_ url: URL) throws -> String { SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined() }
do {
    guard CommandLine.arguments.count == 2 else { try fail("Usage: stage_photo_enhancers.swift /path/to/Texture Studio.app") }
let app = URL(fileURLWithPath: CommandLine.arguments[1])
let base = app.appendingPathComponent("Contents/Resources/PhotoEnhancers")
let fm = FileManager.default
for name in ["PHOTO_REPAIR_MODEL", "PHOTO_LIGHTING_MODEL"] {
    if let path = ProcessInfo.processInfo.environment[name], !path.isEmpty {
        let source = URL(fileURLWithPath:path).resolvingSymlinksInPath().standardizedFileURL
        let destination = base.resolvingSymlinksInPath().standardizedFileURL
        guard source != destination, !source.path.hasPrefix(destination.path + "/") else { try fail("Photo enhancer exports must be outside the build bundle.") }
    }
}
// Never retain a model from a previous build when its explicit input is absent.
if fm.fileExists(atPath: base.path) { try fm.removeItem(at: base) }
for (variable, product) in [("PHOTO_REPAIR_MODEL", "photo-repair-v1"), ("PHOTO_LIGHTING_MODEL", "photo-lighting-v1")] {
    guard let path = ProcessInfo.processInfo.environment[variable], !path.isEmpty else { continue }
    let source = URL(fileURLWithPath: path), descriptor = source.appendingPathComponent("enhancer.json")
    guard let row = try JSONSerialization.jsonObject(with: Data(contentsOf: descriptor)) as? [String: Any],
          row["schema"] as? String == "loom.photo-enhancer.v1", row["product"] as? String == product,
          row["architecture"] as? String == "loom-factory-photo-compact-naf-v1",
          row["input"] as? String == "rgb", row["output"] as? String == "enhanced_rgb",
          row["pixelContract"] as? String == "opaque sRGB numeric Float32 NCHW; input [0,1]; raw residual RGB output; no vector normalization",
          [16,22,32,46].contains(row["width"] as? Int ?? 0), (row["updates"] as? Int ?? 0) > 0,
          row["modelFile"] as? String == "Enhancer.mlmodel", row["weights"] as? String == "weights.safetensors",
          try hash(source.appendingPathComponent("Enhancer.mlmodel")) == row["modelSHA256"] as? String,
          try hash(source.appendingPathComponent("weights.safetensors")) == row["weightsSHA256"] as? String else { try fail("Invalid or untrained \(product) export at \(path).") }
    let compiled = try MLModel.compileModel(at: source.appendingPathComponent("Enhancer.mlmodel"))
    defer { try? fm.removeItem(at: compiled) }
    let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
    let model = try MLModel(contentsOf: compiled, configuration: config)
    let attributes = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
    guard attributes?["loom.schema"] == "loom-factory-photo-model-v1", attributes?["loom.product"] == product,
          attributes?["loom.architecture"] == row["architecture"] as? String,
          attributes?["loom.width"] == String(row["width"] as! Int),
          model.modelDescription.inputDescriptionsByName["rgb"]?.multiArrayConstraint?.dataType == .float32,
          model.modelDescription.outputDescriptionsByName["enhanced_rgb"]?.multiArrayConstraint?.dataType == .float32 else { try fail("Compiled \(product) does not match its RGB contract.") }
    let destination = base.appendingPathComponent(product)
    try fm.createDirectory(at: destination, withIntermediateDirectories: true)
    for file in ["enhancer.json", "Enhancer.mlmodel", "weights.safetensors"] { try fm.copyItem(at: source.appendingPathComponent(file), to: destination.appendingPathComponent(file)) }
    try fm.copyItem(at: compiled, to: destination.appendingPathComponent("Enhancer.mlmodelc"))
    print("Included \(product) · \(row["name"] as? String ?? "candidate") · user-selected build input; not a quality certification")
}

} catch {
    FileHandle.standardError.write(Data("Photo enhancer staging failed: \(error.localizedDescription)\n".utf8))
    exit(1)
}
