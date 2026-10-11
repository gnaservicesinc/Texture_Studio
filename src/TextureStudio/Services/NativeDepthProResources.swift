import Darwin
import Foundation

/// Missing public prerequisites are installed before cache construction, for
/// training, comparison and production inference alike. No interpreter is used.
struct NativeDepthProResources: Sendable {
    static let artifacts: [ModelDownloadArtifact] = [
        artifact("Manifest.json", bytes: 617, sha256: "74452a1ff28db35a1239b318f373dd77dfbd7f7b7965615f7ca7adc3b6266adc"),
        artifact("Data/com.apple.CoreML/model.mlmodel", bytes: 958_432, sha256: "926dd96539fba643ea4cdbbea4f963a632d55da3a176a618586c6cd68a9d24c5"),
        artifact("Data/com.apple.CoreML/weights/weight.bin", bytes: 1_901_556_288, sha256: "8de676bd5dbc417085aa69f0e6264ab636d78c2af4015564c88ee4d14452447e")
    ]
    let managedURL: URL
    let artifacts: [ModelDownloadArtifact]
    let expectedIdentity: String
    let transfer: any ModelArtifactTransferring

    init(managedURL: URL = NativeDepthProContract.managedModelURL,
         artifacts: [ModelDownloadArtifact] = Self.artifacts,
         expectedIdentity: String = NativeDepthProService.pinnedPackageSHA256,
         transfer: any ModelArtifactTransferring = ModelDownloadService()) {
        self.managedURL = managedURL.standardizedFileURL
        self.artifacts = artifacts; self.expectedIdentity = expectedIdentity; self.transfer = transfer
    }

    private static func artifact(_ name: String, bytes: Int64, sha256: String) -> ModelDownloadArtifact {
        .init(relativePath: name,
            url: URL(string: "https://huggingface.co/\(NativeDepthProService.upstreamRepository)/resolve/\(NativeDepthProService.upstreamRevision)/DepthPro.mlpackage/\(name)")!,
            byteCount: bytes, sha256: sha256)
    }

    func ensureAvailable(requestedURL: URL, onEvent: @escaping NativeMaterialTrainer.Event = { _ in }) async throws -> URL {
        try Task.checkCancellation()
        let requested = requestedURL.standardizedFileURL
        // A present developer-selected model retains its own checkpoint identity.
        if requested.resolvingSymlinksInPath().path != managedURL.resolvingSymlinksInPath().path,
           FileManager.default.fileExists(atPath: requested.path) { return requested }
        let parent = managedURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let lock = try InstallationLock(parent.appendingPathComponent(".depthpro-install.lock"))
        while try !lock.acquire() {
            emit("Waiting for DepthPro model setup in another window", completed: 0, onEvent: onEvent)
            try await Task.sleep(for: .milliseconds(500))
        }
        defer { lock.close() }
        emit("Verifying DepthPro model", completed: 0, onEvent: onEvent)
        if (try? verify(managedURL)) != nil {
            emit("DepthPro model ready", completed: totalBytes, ready: true, onEvent: onEvent)
            return managedURL
        }

        // Keep already verified files across retries and cancellation. An
        // incomplete package never becomes the active model.
        let stage = parent.appendingPathComponent(".DepthPro-download.mlpackage")
        guard !isSymbolicLink(stage), !isSymbolicLink(managedURL) else {
            throw StudioError("DepthPro setup cannot replace a symbolic link in its managed model storage.")
        }
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
        try cleanStagingFiles(in: stage)
        var completed: Int64 = 0
        do {
            for artifact in artifacts {
                try Task.checkCancellation()
                let file = try containedFile(artifact.relativePath, in: stage)
                if (try? ModelFileValidation.validate(file, artifact: artifact)) == nil {
                    try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                    let progress = ResourceProgress(completed: completed, total: totalBytes, modelPath: managedURL.path, onEvent: onEvent)
                    emit("Downloading DepthPro model", completed: completed, onEvent: onEvent)
                    for attempt in 1...3 {
                        let partial = file.deletingLastPathComponent().appendingPathComponent(".depthpro-partial-\(UUID().uuidString)")
                        defer { try? FileManager.default.removeItem(at: partial) }
                        do {
                            try await transfer.fetch(artifact, to: partial) { progress.update($0, fileBytes: artifact.byteCount) }
                            try ModelFileValidation.validate(partial, artifact: artifact)
                            try Task.checkCancellation()
                            if FileManager.default.fileExists(atPath: file.path) { try FileManager.default.removeItem(at: file) }
                            try FileManager.default.moveItem(at: partial, to: file)
                            break
                        } catch {
                            try Task.checkCancellation()
                            if error is CancellationError { throw error }
                            guard attempt < 3 else { throw error }
                            emit("Retrying DepthPro download (\(attempt + 1)/3)", completed: completed, onEvent: onEvent)
                            try await Task.sleep(for: .seconds(attempt))
                        }
                    }
                }
                completed += artifact.byteCount
            }
            emit("Verifying downloaded DepthPro model", completed: totalBytes, onEvent: onEvent)
            try verify(stage)
            try Task.checkCancellation()
            let previous = parent.appendingPathComponent(".DepthPro-previous-\(UUID().uuidString).mlpackage")
            let hadPrevious = FileManager.default.fileExists(atPath: managedURL.path)
            if hadPrevious { try FileManager.default.moveItem(at: managedURL, to: previous) }
            do { try FileManager.default.moveItem(at: stage, to: managedURL) }
            catch {
                if hadPrevious { try? FileManager.default.moveItem(at: previous, to: managedURL) }
                throw error
            }
            if hadPrevious { try? FileManager.default.removeItem(at: previous) }
            // This receipt lives outside the package so its identity is stable.
            let receipt: [String: Any] = ["schema": "texture-studio-depthpro-install-v1",
                "repository": NativeDepthProService.upstreamRepository, "revision": NativeDepthProService.upstreamRevision,
                "package_sha256": expectedIdentity, "model_path": managedURL.path, "bytes": totalBytes,
                "artifacts": artifacts.map { ["path": $0.relativePath, "bytes": $0.byteCount, "sha256": $0.sha256] as [String: Any] }]
            if let bytes = try? JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]) {
                try? bytes.write(to: parent.appendingPathComponent("depthpro-install.json"), options: .atomic)
            }
            emit("DepthPro model ready", completed: totalBytes, ready: true, onEvent: onEvent)
            return managedURL
        } catch {
            try Task.checkCancellation()
            if error is CancellationError { throw error }
            throw StudioError("DepthPro setup could not finish after attempting its published model download: \(error.localizedDescription). Verified files are retained for the next attempt.")
        }
    }

    private var totalBytes: Int64 { artifacts.reduce(0) { $0 + $1.byteCount } }
    /// This is our private staging directory. Keep verified inventory members,
    /// and remove leftovers from interrupted transfers before hashing it.
    private func cleanStagingFiles(in stage: URL) throws {
        let allowed = Set(artifacts.map(\.relativePath))
        let root = stage.resolvingSymlinksInPath()
        guard let entries = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey]) else {
            throw StudioError("DepthPro setup cannot inspect its download staging directory.")
        }
        for case let file as URL in entries {
            let values = try file.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            let name = String(file.resolvingSymlinksInPath().path.dropFirst(root.path.count + 1))
            guard values.isSymbolicLink != true else { throw StudioError("DepthPro download staging contains a symbolic link.") }
            if values.isDirectory == true {
                if !allowed.contains(where: { $0.hasPrefix(name + "/") }) {
                    entries.skipDescendants()
                    try FileManager.default.removeItem(at: file)
                }
            } else if !allowed.contains(name) {
                try FileManager.default.removeItem(at: file)
            }
        }
    }
    private func verify(_ package: URL) throws {
        guard !isSymbolicLink(package) else { throw StudioError("DepthPro managed package must be a real directory.") }
        for artifact in artifacts { try ModelFileValidation.validate(containedFile(artifact.relativePath, in: package), artifact: artifact) }
        guard try NativeDepthProService.modelIdentity(at: package) == expectedIdentity else {
            throw StudioError("DepthPro package failed its pinned model identity check.")
        }
    }
    private func containedFile(_ name: String, in root: URL) throws -> URL {
        let parts = name.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.isEmpty, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }), !name.hasPrefix("/") else {
            throw StudioError("DepthPro model inventory contains an invalid package path.")
        }
        var path = root
        for part in parts {
            path.appendPathComponent(String(part))
            guard !isSymbolicLink(path) else { throw StudioError("DepthPro package contains a symbolic link.") }
        }
        return path
    }
    private func isSymbolicLink(_ path: URL) -> Bool {
        (try? path.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true
    }
    private func emit(_ operation: String, completed: Int64, ready: Bool = false, onEvent: NativeMaterialTrainer.Event) {
        ResourceProgress.emit(operation: operation, completed: completed, total: totalBytes,
            modelPath: managedURL.path, ready: ready, onEvent: onEvent)
    }

    private final class InstallationLock {
        private var descriptor: Int32
        init(_ url: URL) throws {
            descriptor = Darwin.open(url.path, O_RDWR | O_CREAT | O_NOFOLLOW, 0o600)
            guard descriptor >= 0 else { throw StudioError("DepthPro setup cannot open its model installation lock.") }
        }
        func acquire() throws -> Bool {
            if flock(descriptor, LOCK_EX | LOCK_NB) == 0 { return true }
            guard errno == EWOULDBLOCK || errno == EAGAIN else { throw StudioError("DepthPro setup cannot lock its model storage.") }
            return false
        }
        func close() { if descriptor >= 0 { flock(descriptor, LOCK_UN); Darwin.close(descriptor); descriptor = -1 } }
        deinit { close() }
    }

    private final class ResourceProgress: @unchecked Sendable {
        let completed: Int64, total: Int64, modelPath: String
        let onEvent: NativeMaterialTrainer.Event
        private let lock = NSLock()
        private var lastPercent = -1
        init(completed: Int64, total: Int64, modelPath: String, onEvent: @escaping NativeMaterialTrainer.Event) {
            self.completed = completed; self.total = total; self.modelPath = modelPath; self.onEvent = onEvent
        }
        func update(_ fraction: Double, fileBytes: Int64) {
            let bytes = completed + Int64(Double(fileBytes) * min(1, max(0, fraction)))
            let percent = Int(Double(bytes) / Double(max(1, total)) * 100)
            guard lock.withLock({ if percent == lastPercent { return false }; lastPercent = percent; return true }) else { return }
            Self.emit(operation: "Downloading DepthPro model", completed: bytes, total: total, modelPath: modelPath, onEvent: onEvent)
        }
        static func emit(operation: String, completed: Int64, total: Int64, modelPath: String, ready: Bool = false,
                         onEvent: NativeMaterialTrainer.Event) {
            let event: [String: Any] = ["event": "depthpro_model_setup", "operation": operation,
                "completed": completed, "total": total, "bytes_completed": completed, "bytes_total": total,
                "model_path": modelPath, "ready": ready, "repository": NativeDepthProService.upstreamRepository,
                "revision": NativeDepthProService.upstreamRevision]
            if let data = try? JSONSerialization.data(withJSONObject: event, options: [.sortedKeys]),
               let text = String(data: data, encoding: .utf8) { onEvent(text + "\n") }
        }
    }
}
