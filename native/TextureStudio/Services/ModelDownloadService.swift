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
        let delegate = DownloadProgressDelegate(expectedBytes: artifact.byteCount, progress: progress)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 3600
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let temporary: URL
        let response: URLResponse
        do { (temporary, response) = try await session.download(from: artifact.url, delegate: delegate) }
        catch {
            if delegate.exceededSize { throw LocalModelError.invalidDownload("artifact exceeded its published size") }
            throw error
        }
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
            throw LocalModelError.invalidDownload("the server did not return a model artifact")
        }
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: temporary, to: destination)
        progress(1)
    }
}

private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let expectedBytes: Int64
    private let progress: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var sizeViolation = false
    var exceededSize: Bool { lock.withLock { sizeViolation } }

    init(expectedBytes: Int64, progress: @escaping @Sendable (Double) -> Void) {
        self.expectedBytes = expectedBytes
        self.progress = progress
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
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
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
