import Darwin
import Foundation

/// Logs are files rather than pipes, so long training runs cannot deadlock.
/// Stop sends SIGINT: the trainer saves its latest optimizer state before exit.
final class WorkbenchProcess: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var stopping = false

    func stop() {
        lock.withLock {
            stopping = true
            if let process, process.isRunning { kill(process.processIdentifier, SIGINT) }
        }
    }

    func run(executable: URL, arguments: [String], directory: URL?, log: URL,
             onLog: @escaping @Sendable (String) -> Void) async throws -> String {
        try FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        defer { try? handle.close() }
        let task = Process()
        task.executableURL = executable
        task.arguments = arguments
        task.currentDirectoryURL = directory
        task.standardOutput = handle
        task.standardError = handle
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "PYTHONHOME")
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        task.environment = environment
        let drain = WorkbenchLogDrain()
        let monitor = Task.detached {
            guard let reader = try? FileHandle(forReadingFrom: log) else { return }
            defer { try? reader.close() }
            var decoder = WorkbenchLogDecoder()
            while true {
                let wasFinished = drain.isFinished
                if let bytes = try? reader.read(upToCount: 65536), !bytes.isEmpty {
                    let text = decoder.append(bytes)
                    if !text.isEmpty { onLog(text) }
                    // Drain large final bursts without a delay between chunks.
                    continue
                }
                // Observe completion before reading, so a write between EOF
                // and termination cannot be mistaken for a drained file.
                if wasFinished { break }
                try? await Task.sleep(for: .milliseconds(200))
            }
            let tail = decoder.finish()
            if !tail.isEmpty { onLog(tail) }
        }
        let status: Int32
        do {
            status = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Int32, Error>) in
                    task.terminationHandler = { continuation.resume(returning: $0.terminationStatus) }
                    do {
                        try lock.withLock {
                            guard !stopping else { throw CancellationError() }
                            process = task
                            try task.run()
                        }
                    } catch { continuation.resume(throwing: error) }
                }
            } onCancel: { self.stop() }
        } catch {
            drain.finish()
            await monitor.value
            lock.withLock { process = nil }
            throw error
        }
        // Exit guarantees that this worker has finished writing its log. Wait
        // for the reader's EOF, including checkpoint-save lines after SIGINT.
        drain.finish()
        await monitor.value
        lock.withLock { process = nil }
        let data = try Data(contentsOf: log)
        let output = String(decoding: data, as: UTF8.self)
        guard status == 0 else { throw StudioError(String(output.suffix(4000))) }
        return output
    }

    static func decode<T: Decodable>(_ type: T.Type, output: String) throws -> T {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        for line in output.split(separator: "\n").reversed() {
            if let decoded = try? decoder.decode(type, from: Data(line.utf8)) { return decoded }
        }
        throw StudioError("The material worker did not return a valid result. See the operation log.")
    }
}

private final class WorkbenchLogDrain: @unchecked Sendable {
    private let lock = NSLock()
    private var finished = false
    var isFinished: Bool { lock.withLock { finished } }
    func finish() { lock.withLock { finished = true } }
}

/// File reads may split a UTF-8 scalar, even though the file itself is valid.
/// Retain only the unfinished suffix rather than emitting replacement glyphs.
private struct WorkbenchLogDecoder {
    private var pending = Data()
    mutating func append(_ bytes: Data) -> String {
        pending.append(bytes)
        if let text = String(data: pending, encoding: .utf8) {
            pending.removeAll(keepingCapacity: true)
            return text
        }
        for suffixLength in 1...min(3, pending.count) {
            let boundary = pending.count - suffixLength
            if let text = String(data: pending.prefix(boundary), encoding: .utf8) {
                pending = Data(pending.suffix(suffixLength))
                return text
            }
        }
        // Invalid bytes already in a log must not make the buffer grow forever.
        let text = String(decoding: pending, as: UTF8.self)
        pending.removeAll(keepingCapacity: true)
        return text
    }
    mutating func finish() -> String {
        defer { pending.removeAll(keepingCapacity: true) }
        return String(decoding: pending, as: UTF8.self)
    }
}
