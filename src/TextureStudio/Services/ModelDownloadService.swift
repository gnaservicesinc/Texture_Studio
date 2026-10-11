import CryptoKit
import Foundation

protocol ModelArtifactTransferring: Sendable {
    func fetch(_ artifact: ModelDownloadArtifact, to destination: URL,
        progress: @escaping @Sendable (Double) -> Void) async throws
}

struct ModelDownloadService: ModelArtifactTransferring {
    func fetch(_ artifact: ModelDownloadArtifact, to destination: URL,
        progress: @escaping @Sendable (Double) -> Void) async throws {
        guard artifact.url.scheme == "https" else {
            throw LocalModelError.invalidDownload("downloads require HTTPS")
        }
        try Task.checkCancellation()
        let delegate = DownloadProgressDelegate(expectedBytes: artifact.byteCount, destination: destination, progress: progress)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 3600
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let task = session.downloadTask(with: artifact.url)
        // The async convenience API replaces the download delegate and does not
        // deliver intermediate didWriteData callbacks. Keep our delegate in
        // charge of the task, bridging only its completion into Swift async.
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                delegate.start(task, continuation: continuation)
            }
        } onCancel: { delegate.cancel() }
        try Task.checkCancellation()
        progress(1)
    }
}

private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let expectedBytes: Int64
    private let destination: URL
    private let progress: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var sizeViolation = false
    private var cancelled = false
    private var task: URLSessionDownloadTask?
    private var continuation: CheckedContinuation<Void, Error>?

    init(expectedBytes: Int64, destination: URL, progress: @escaping @Sendable (Double) -> Void) {
        self.expectedBytes = expectedBytes
        self.destination = destination
        self.progress = progress
    }
    func start(_ task: URLSessionDownloadTask, continuation: CheckedContinuation<Void, Error>) {
        let start = lock.withLock {
            guard !cancelled else { return false }
            self.task = task; self.continuation = continuation
            return true
        }
        if start { task.resume() }
        else { task.cancel(); continuation.resume(throwing: CancellationError()) }
    }
    func cancel() {
        let active = lock.withLock {
            cancelled = true
            return task
        }
        // Registered tasks finish on the serial delegate queue. Waiting for
        // that callback prevents caller cleanup racing the temporary-file move.
        // Cancellation before registration is remembered by start().
        active?.cancel()
    }
    private func finish(_ result: Result<Void, Error>) {
        let pending = lock.withLock {
            let pending = continuation
            continuation = nil; task = nil
            return (pending, cancelled)
        }
        pending.0?.resume(with: pending.1 ? .failure(CancellationError()) : result)
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard totalBytesWritten <= expectedBytes,
              totalBytesExpectedToWrite <= 0 || totalBytesExpectedToWrite <= expectedBytes else {
            lock.withLock { sizeViolation = true }
            downloadTask.cancel()
            return
        }
        progress(min(1, Double(totalBytesWritten) / Double(max(1, expectedBytes))))
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            guard let response = downloadTask.response as? HTTPURLResponse, response.statusCode == 200,
                  response.url?.scheme == "https" else {
                throw LocalModelError.invalidDownload("the server did not return a model artifact")
            }
            guard Int64(try location.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= expectedBytes else {
                throw LocalModelError.invalidDownload("artifact exceeded its published size")
            }
            guard !lock.withLock({ cancelled }) else { throw CancellationError() }
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            // URLSession removes location when this callback returns.
            try FileManager.default.moveItem(at: location, to: destination)
            finish(.success(()))
        } catch { finish(.failure(error)) }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let state = lock.withLock { (sizeViolation, cancelled) }
        if state.1 { finish(.failure(CancellationError())) }
        else if state.0 { finish(.failure(LocalModelError.invalidDownload("artifact exceeded its published size"))) }
        else if let error { finish(.failure(error)) }
        else { finish(.failure(LocalModelError.invalidDownload("the server returned no model artifact"))) }
    }
}

enum ModelFileValidation {
    static func validate(_ file: URL, artifact: ModelDownloadArtifact) throws {
        let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard size == Int(artifact.byteCount) else {
            throw LocalModelError.invalidDownload("\(artifact.relativePath) is incomplete or has the wrong size")
        }
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            try Task.checkCancellation()
            hash.update(data: data)
        }
        let actual = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual == artifact.sha256.lowercased() else {
            throw LocalModelError.invalidDownload("\(artifact.relativePath) failed its SHA-256 checksum")
        }
    }
}


struct LocalModelValidationService: ModelValidating {
    func inspect(at url: URL) async throws -> ModelInterface {
        try await ModelDepthService().inspect(at: url)
    }
}
