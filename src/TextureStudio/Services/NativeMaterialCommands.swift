import Foundation

enum NativeMaterialCommands {
    static func run(arguments: [String], onEvent: @escaping @Sendable (String) -> Void,
                    control: NativeMaterialTrainingControl) async throws -> String {
        try Task.checkCancellation()
        func value(_ flag: String) -> String? { arguments.firstIndex(of: flag).flatMap { arguments.indices.contains($0 + 1) ? arguments[$0 + 1] : nil } }
        func url(_ flag: String) throws -> URL {
            guard let path = value(flag), !path.isEmpty else { throw StudioError("Missing native material operation argument: \(flag)") }
            return URL(fileURLWithPath: path)
        }
        if let result = try await NativeMaterialDatasetService.run(arguments: arguments, onEvent: onEvent) { return result }
        switch arguments.first {
        case "checkpoint":
            let file = try url("--checkpoint"), expected = value("--expected-sha256")
            let job = Task.detached { try NativeMaterialCheckpoint.inspect(at: file, expectedSHA256: expected) }
            return try await withTaskCancellationHandler { try await job.value } onCancel: { job.cancel() }
        case "capabilities":
            let scope = value("--scope") ?? "final-map"
            guard ["final-map", "map-decoder"].contains(scope) else { throw StudioError("Unsupported material training scope.") }
            return try NativeMaterialTransfer.json(["training_sizes": [256,512,1024,2048,4096], "inference_sizes": [256,512,1024,2048,4096,8192], "targets": ["height","roughness","normal"], "scope": scope, "image_size_matches_training_size": true, "hidden_encoder_resize": false, "memory_admission_enabled": true])
        case "hub-account": return try await NativeHuggingFaceService().accountJSON()
        case "hub-models": return try await NativeHuggingFaceService().modelsJSON()
        case "train", "refine", "infer": return try await NativeMaterialTrainer.run(arguments: arguments, onEvent: onEvent, control: control)
        case "review-source":
            guard let result = try await ReviewImageLoader.runNativeSource(arguments: arguments) else { throw StudioError("Invalid native source review request.") }
            return result
        case "package": return try await NativeMaterialPackage.run(arguments: arguments)
        case "upload-selected":
            let package = try url("--output")
            let packageResult = try await NativeMaterialPackage.run(arguments: arguments)
            let packaged = try NativeMaterialTransfer.object(Data(packageResult.utf8))
            guard let repository = value("--repo") else { throw StudioError("Choose a Hub repository.") }
            let uploadResult = try await NativeMaterialTransfer().upload(package: package, repository: repository, isPublic: arguments.contains("--public"))
            var uploaded = try NativeMaterialTransfer.object(Data(uploadResult.utf8))
            uploaded["source_checkpoint_sha256"] = packaged["source_checkpoint_sha256"]
            uploaded["package_path"] = package.path
            return try NativeMaterialTransfer.json(uploaded)
        case "upload":
            guard let repository = value("--repo") else { throw StudioError("Choose a Hub repository.") }
            return try await NativeMaterialTransfer().upload(package: url("--package"), repository: repository, isPublic: arguments.contains("--public"))
        case "download-model":
            guard let repository = value("--repo"), let revision = value("--revision") else { throw StudioError("Choose a Hub model with an exact recorded revision.") }
            return try await NativeMaterialTransfer().download(repository: repository, revision: revision, to: url("--destination"))
        case "install-base": return try await NativeMaterialTransfer.installBase(at: url("--destination"))
        case "remove-base": return try NativeMaterialTransfer.removeBase(at: url("--directory"))
        default: throw StudioError("Unsupported native material operation: \(arguments.first ?? "missing")")
        }
    }
}

/// Native events are complete Swift strings, so logs need no pipe polling or
/// incremental UTF-8 decoding. Writes are serialized off the app's main actor.
final class NativeWorkbenchLog: @unchecked Sendable {
    private let lock = NSLock()
    private let url: URL
    private var contents = ""
    init(url: URL) { self.url = url }
    var text: String { lock.withLock { String(contents.suffix(100000)) } }
    func append(_ text: String) {
        lock.withLock {
            contents += text
            if contents.count > 100000 { contents = String(contents.suffix(100000)) }
            if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
            if let file = try? FileHandle(forWritingTo: url) {
                defer { try? file.close() }
                _ = try? file.seekToEnd(); try? file.write(contentsOf: Data(text.utf8))
            }
        }
    }
}
