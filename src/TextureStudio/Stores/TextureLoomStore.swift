import AppKit
import Foundation
import CoreML
import Observation

@MainActor @Observable
final class TextureLoomStore {
    struct Adapter: Identifiable {
        let manifest: TextureLoomAdapterManifest
        let folder: URL
        var enabled: Bool
        var id: String { manifest.id }
    }
    private(set) var package: TextureLoomPackage?
    private(set) var adapters: [Adapter] = []
    private(set) var error: String?
    private(set) var revision = UUID()
    let storage: URL
    private let preferences: UserDefaults
    @ObservationIgnored private var checkedRuntimePaths = Set<String>()
    var statusDescription: String {
        package.map { "\($0.manifest.name) · \($0.manifest.version) · \($0.manifest.categories.count) datasets" }
            ?? "This build has no Texture Loom model. Export a trained package from Loom Factory and include it when building Texture Studio."
    }
    init(bundleRoot: URL? = Bundle.main.resourceURL?.appendingPathComponent("TextureLoom"),
         storage: URL? = nil, preferences: UserDefaults = StudioPreferences.defaults) {
        self.preferences = preferences
        self.storage = storage ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/LoRAs", isDirectory: true)
        if let bundleRoot, FileManager.default.fileExists(atPath: bundleRoot.appendingPathComponent("manifest.json").path) {
            do { package = try TextureLoomPackage(root: bundleRoot) } catch { self.error = error.localizedDescription }
        }
        refreshAdapters()
    }
    @discardableResult
    func importAdapter(_ url: URL) throws -> String {
        guard let package else { throw StudioError(statusDescription) }
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("texture-loom-import-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let source: URL
        if url.hasDirectoryPath || (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true { source = url }
        else {
            try Self.extractArchive(url, to: scratch)
            source = scratch
        }
        let manifest = try validateAdapter(source, package: package)
        try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true)
        let destination = storage.appendingPathComponent(manifest.id)
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw StudioError("This LoRA is already installed.") }
        let staging = storage.appendingPathComponent(".import-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: staging) }
        try FileManager.default.copyItem(at: source, to: staging)
        _ = try validateAdapter(staging, package: package)
        try FileManager.default.moveItem(at: staging, to: destination)
        refreshAdapters()
        error = nil
        return manifest.id
    }
    func setEnabled(_ id: String, _ enabled: Bool) throws {
        guard let package, let index = adapters.firstIndex(where: { $0.id == id }) else { throw StudioError("This LoRA is no longer installed.") }
        _ = try validateAdapter(adapters[index].folder, package: package)
        // Only one override for a dataset/map can be effective at a time.
        if enabled {
            let requested = Set(adapters[index].manifest.categories.flatMap { category in category.adapters.map { category.id + ":" + $0.map } })
            for other in adapters.indices where other != index && adapters[other].enabled {
                let keys = Set(adapters[other].manifest.categories.flatMap { category in category.adapters.map { category.id + ":" + $0.map } })
                if !requested.isDisjoint(with: keys) { adapters[other].enabled = false }
            }
        }
        adapters[index].enabled = enabled
        preferences.set(adapters.filter(\.enabled).map(\.id), forKey: "textureLoom.enabledAdapters.v1")
        revision = UUID()
    }
    func activeOverrides() throws -> [String: URL] {
        guard let package else { throw StudioError(statusDescription) }
        var result: [String: URL] = [:]
        for adapter in adapters where adapter.enabled {
            _ = try validateAdapter(adapter.folder, package: package)
            for category in adapter.manifest.categories {
                for model in category.adapters { result[category.id + ":" + model.map] = try TextureLoomPackage.resolve(model.path, root: adapter.folder) }
            }
        }
        return result
    }
    private func refreshAdapters() {
        adapters = []
        guard let package, let folders = try? FileManager.default.contentsOfDirectory(at: storage, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        let enabled = Set(preferences.stringArray(forKey: "textureLoom.enabledAdapters.v1") ?? [])
        for folder in folders.sorted(by: { $0.path < $1.path }) where !folder.lastPathComponent.hasPrefix(".") {
            do {
                let manifest = try validateAdapter(folder, package: package)
                adapters.append(Adapter(manifest: manifest, folder: folder, enabled: enabled.contains(manifest.id)))
            } catch { self.error = "A saved LoRA could not be loaded: \(error.localizedDescription)" }
        }
    }
    private func validateAdapter(_ root: URL, package: TextureLoomPackage) throws -> TextureLoomAdapterManifest {
        let manifest = try JSONDecoder().decode(TextureLoomAdapterManifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
        guard manifest.schema == "texture-loom.adapter.v1", UUID(uuidString: manifest.id) != nil,
              !manifest.name.isEmpty, !manifest.categories.isEmpty,
              manifest.baseModel.id == package.manifest.id, manifest.baseModel.version == package.manifest.version else {
            throw StudioError("This LoRA was exported for a different Texture Loom model or version.")
        }
        try TextureLoomPackage.verify(files: manifest.files, root: root)
        guard Set(manifest.categories.map(\.id)).count == manifest.categories.count else { throw StudioError("This LoRA repeats a dataset.") }
        for category in manifest.categories {
            guard let base = package.manifest.categories.first(where: { $0.id == category.id && $0.version == category.version }),
                  !category.adapters.isEmpty, Set(category.adapters.map(\.map)).count == category.adapters.count else {
                throw StudioError("This LoRA targets a dataset version that is absent from this build.")
            }
            for model in category.adapters {
                guard model.format == "coreml", let baseMap = base.models.first(where: { $0.map == model.map }),
                      try package.modelHash(baseMap) == model.baseSHA256 else {
                    throw StudioError("This LoRA needs its matching compiled Core ML export and base weights.")
                }
                for path in [model.path, model.weightsPath] {
                    _ = try TextureLoomPackage.resolve(path, root: root)
                    guard manifest.files.contains(where: { $0.path == path || $0.path.hasPrefix(path + "/") }) else {
                        throw StudioError("The LoRA export is incomplete: \(path).")
                    }
                }
                let runtime = try TextureLoomPackage.resolve(model.path, root: root)
                if (try runtime.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
                    guard let entries = FileManager.default.enumerator(at: runtime, includingPropertiesForKeys: [.isRegularFileKey]) else { throw StudioError("The LoRA runtime folder could not be read.") }
                    let inventory = Set(manifest.files.map(\.path))
                    for case let file as URL in entries where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true && file.lastPathComponent != ".DS_Store" {
                        let relative = String(file.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(root.standardizedFileURL.resolvingSymlinksInPath().path.count + 1))
                        _ = try TextureLoomPackage.resolve(relative, root: root)
                        guard inventory.contains(relative) else { throw StudioError("The LoRA runtime contains an unregistered file: \(relative).") }
                    }
                }
                if !checkedRuntimePaths.contains(runtime.path) {
                    let compiled = runtime.pathExtension == "mlmodelc" ? runtime : try MLModel.compileModel(at: runtime)
                    let model = try MLModel(contentsOf: compiled)
                    guard model.modelDescription.inputDescriptionsByName["rgb"]?.type == .multiArray,
                          model.modelDescription.outputDescriptionsByName["material_map"]?.type == .multiArray else {
                        throw StudioError("This LoRA has an incompatible material inference interface.")
                    }
                    checkedRuntimePaths.insert(runtime.path)
                }
            }
        }
        return manifest
    }
    private static func extractArchive(_ archive: URL, to destination: URL) throws {
        let listing = try command("/usr/bin/zipinfo", ["-1", archive.path])
        let permissions = try command("/usr/bin/zipinfo", ["-l", archive.path])
        guard !permissions.split(separator: "\n").contains(where: { $0.hasPrefix("l") }) else { throw StudioError("LoRA archives cannot contain symbolic links.") }
        let names = listing.split(separator: "\n").map(String.init)
        guard !names.isEmpty, names.count <= 10000 else { throw StudioError("The LoRA archive has an invalid file inventory.") }
        for name in names {
            guard !name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else { throw StudioError("The LoRA archive contains an invalid filename.") }
            _ = try TextureLoomPackage.resolve(name.hasSuffix("/") ? String(name.dropLast()) : name, root: destination)
        }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        _ = try command("/usr/bin/ditto", ["-x", "-k", archive.path, destination.path])
    }
    private static func command(_ executable: String, _ arguments: [String]) throws -> String {
        let process = Process(); process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
        let output = Pipe(); process.standardOutput = output; process.standardError = output
        try process.run(); let bytes = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw StudioError("The LoRA archive could not be opened.") }
        return String(decoding: bytes, as: UTF8.self)
    }
}
