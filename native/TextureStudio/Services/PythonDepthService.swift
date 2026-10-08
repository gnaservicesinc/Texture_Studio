import CoreGraphics
import Darwin
import Foundation
import ImageIO
import Observation
import UniformTypeIdentifiers

enum LocalPythonRuntimeStatus: Equatable {
    case missing, checking, ready(String), failed(String)
    var isReady: Bool { if case .ready = self { true } else { false } }
    var message: String {
        switch self {
        case .missing: "Locate a Python environment with PyTorch/MPS, or install an app-managed runtime."
        case .checking: "Checking the local Metal runtime…"
        case .ready(let version): "Ready · \(version) · Apple Metal"
        case .failed(let message): message
        }
    }
}

@MainActor @Observable
final class PythonDepthService {
    private(set) var pythonURL: URL?
    private(set) var runtimeStatus: LocalPythonRuntimeStatus = .missing
    private(set) var isBusy = false
    private(set) var progress: Double?
    var lastError: String?
    let runtimeDirectory: URL
    let backendDirectory: URL
    @ObservationIgnored private var managed = false
    @ObservationIgnored private var runner: PythonProcessRunner?
    private var registryURL: URL { runtimeDirectory.deletingLastPathComponent().appendingPathComponent("python-runtime.json") }

    init(runtimeDirectory: URL? = nil, backendDirectory: URL? = nil) {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        self.runtimeDirectory = runtimeDirectory ?? support.appendingPathComponent("Texture Studio/Runtimes/da3-python", isDirectory: true)
        self.backendDirectory = backendDirectory ?? Bundle.main.resourceURL!.appendingPathComponent("DA3Backend", isDirectory: true)
        if let data = try? Data(contentsOf: registryURL), let record = try? JSONDecoder().decode(RuntimeRecord.self, from: data) {
            pythonURL = URL(fileURLWithPath: record.path)
            managed = record.managed
        }
        // A saved executable is probed before it can claim readiness.
        if let url = pythonURL {
            Task { [weak self] in
                do { try await self?.verifySavedRuntime(url) }
                catch { self?.lastError = error.localizedDescription }
            }
        }
    }

    func cancel() { runner?.cancel() }

    func locatePython(at url: URL) async throws {
        guard !isBusy else { throw LocalModelError.busy }
        guard !contains(url, in: runtimeDirectory.deletingLastPathComponent()) else { throw LocalModelError.unsafeRemoval }
        try begin()
        defer { finish() }
        do {
            let info = try await probe(url)
            try save(url: url, managed: false)
            runtimeStatus = .ready("Python \(info.python) · PyTorch \(info.torch)")
        } catch { fail(error); throw error }
    }

    func setupRuntime(using basePythonURL: URL) async throws {
        guard !isBusy else { throw LocalModelError.busy }
        try begin()
        defer { finish() }
        var setupStarted = false
        do {
            guard !FileManager.default.fileExists(atPath: runtimeDirectory.path) else {
                throw LocalModelError.unsupported("an app-managed runtime already exists; remove it before reinstalling")
            }
            try FileManager.default.createDirectory(at: runtimeDirectory.deletingLastPathComponent(), withIntermediateDirectories: true)
            let free = try runtimeDirectory.deletingLastPathComponent().resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage
            if let free, free < 5_000_000_000 { throw LocalModelError.unsupported("at least 5 GB of free disk space is needed for the Python runtime") }
            setupStarted = true
            let process = PythonProcessRunner(executable: basePythonURL)
            runner = process
            _ = try await process.run(arguments: ["-I", "-B", backendDirectory.appendingPathComponent("setup_runtime.py").path,
                "--destination", runtimeDirectory.path], timeout: 2400) { [weak self] event in
                Task { @MainActor in self?.progress = event.progress }
            }
            let python = runtimeDirectory.appendingPathComponent("bin/python")
            let info = try await probe(python)
            try save(url: python, managed: true)
            runtimeStatus = .ready("Python \(info.python) · PyTorch \(info.torch)")
        } catch {
            // Only this exact directory with the helper's ownership marker can
            // be removed on failed setup. The user's base Python is untouched.
            if setupStarted, ownsManagedDirectory() { try? FileManager.default.removeItem(at: runtimeDirectory) }
            fail(error); throw error
        }
    }

    /// External Python environments are unlinked; only our marked venv is deleted.
    func removeManagedRuntime() throws {
        guard !isBusy else { throw LocalModelError.busy }
        if managed {
            guard pythonURL?.standardizedFileURL.path == runtimeDirectory.appendingPathComponent("bin/python").standardizedFileURL.path,
                  ownsManagedDirectory() else { throw LocalModelError.unsafeRemoval }
            try FileManager.default.removeItem(at: runtimeDirectory)
        }
        if FileManager.default.fileExists(atPath: registryURL.path) { try FileManager.default.removeItem(at: registryURL) }
        pythonURL = nil; managed = false; runtimeStatus = .missing; lastError = nil
    }

    func predict(image: CGImage, modelURL: URL, processResolution: Int = 1036) async throws -> ModelDepthResult {
        guard !isBusy else { throw LocalModelError.busy }
        guard runtimeStatus.isReady, let pythonURL else {
            throw LocalModelError.unsupported("the PyTorch/MPS runtime is missing. Install or locate it in Models")
        }
        try Self.validateResolution(processResolution)
        guard Self.modelFilesExist(modelURL) else { throw LocalModelError.missingModel }
        let requiredGiB: UInt64 = processResolution <= 1036 ? 12 : (processResolution <= 1540 ? 24 : 40)
        let gib: UInt64 = 1 << 30
        guard ProcessInfo.processInfo.physicalMemory >= 24 * gib, Self.availableMemory() >= requiredGiB * gib else {
            throw LocalModelError.unsupported("GIANT at \(processResolution) pixels needs approximately \(requiredGiB) GB of available memory. Unload other models or choose a smaller inference size")
        }
        try begin(checking: false)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("texture-da3-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory); finish() }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let source = directory.appendingPathComponent("source.png")
            try Self.writeBoundedPNG(image, to: source, maxDimension: processResolution)
            let process = PythonProcessRunner(executable: pythonURL)
            runner = process
            _ = try await process.run(arguments: ["-I", "-B", backendDirectory.appendingPathComponent("worker.py").path,
                "--model", modelURL.path, "--image", source.path, "--output", directory.path,
                "--resolution", String(processResolution)], timeout: 1800) { [weak self] event in
                Task { @MainActor in self?.progress = event.progress }
            }
            let metadata = try JSONDecoder().decode(DepthMetadata.self, from: Data(contentsOf: directory.appendingPathComponent("metadata.json")))
            let data = try Data(contentsOf: directory.appendingPathComponent("depth.f32"))
            return try Self.decodeDepth(data, metadata: metadata)
        } catch {
            if !Self.modelFilesExist(modelURL) { lastError = LocalModelError.missingModel.localizedDescription; throw LocalModelError.missingModel }
            lastError = error is CancellationError ? nil : error.localizedDescription
            throw error
        }
    }

    private static func modelFilesExist(_ folder: URL) -> Bool {
        ["config.json", "model.safetensors"].allSatisfy {
            FileManager.default.fileExists(atPath: folder.appendingPathComponent($0).path)
        }
    }

    static func validateResolution(_ value: Int) throws {
        guard [1036, 1540, 2044].contains(value) else {
            throw LocalModelError.unsupported("choose an inference edge of 1036, 1540 or 2044 pixels; export size is independent")
        }
    }

    static func decodeDepth(_ data: Data, metadata: DepthMetadata) throws -> ModelDepthResult {
        guard metadata.width > 0, metadata.height > 0,
              metadata.width <= 2044, metadata.height <= 2044,
              data.count == metadata.width * metadata.height * 4,
              metadata.provenance.rowOrder == "top-down", metadata.provenance.fullSourceFieldOfView,
              metadata.provenance.checkpointSHA256 == LocalModelDescriptor.da3WeightsSHA256,
              metadata.provenance.revision == LocalModelDescriptor.da3Revision,
              metadata.provenance.device == "mps", metadata.provenance.precision == "Float32",
              metadata.provenance.modelID == "depth-anything/DA3-GIANT-1.1", metadata.provenance.backend == "PyTorch",
              metadata.provenance.upstreamRevision == "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              [1036, 1540, 2044].contains(metadata.provenance.processResolution),
              metadata.width <= metadata.provenance.processResolution, metadata.height <= metadata.provenance.processResolution else {
            throw LocalModelError.unsupported("the worker returned inconsistent depth dimensions or provenance")
        }
        let values = data.withUnsafeBytes { bytes in
            (0..<(data.count / 4)).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        guard values.allSatisfy(\.isFinite) else { throw LocalModelError.unsupported("the worker returned nonfinite depth") }
        return ModelDepthResult(width: metadata.width, height: metadata.height, values: values,
            outputName: metadata.outputName, interpretation: metadata.interpretation, provenance: metadata.provenance)
    }

    private func verifySavedRuntime(_ url: URL) async throws {
        guard !isBusy else { return }
        try begin(); defer { finish() }
        do {
            let info = try await probe(url)
            runtimeStatus = .ready("Python \(info.python) · PyTorch \(info.torch)")
        } catch { fail(error); throw error }
    }
    private func probe(_ url: URL) async throws -> RuntimeInfo {
        guard FileManager.default.isExecutableFile(atPath: url.path) else {
            throw LocalModelError.unsupported("the selected Python executable is missing or moved")
        }
        let process = PythonProcessRunner(executable: url); runner = process
        let output = try await process.run(arguments: ["-I", "-B", backendDirectory.appendingPathComponent("worker.py").path, "--probe"], timeout: 120)
        guard let info = output.split(separator: "\n").compactMap({ try? JSONDecoder().decode(RuntimeInfo.self, from: Data($0.utf8)) }).last,
              info.ready, info.device == "mps" else { throw LocalModelError.unsupported("Python did not report a ready Metal backend") }
        return info
    }
    private func begin(checking: Bool = true) throws {
        guard !isBusy else { throw LocalModelError.busy }
        guard FileManager.default.fileExists(atPath: backendDirectory.appendingPathComponent("worker.py").path) else {
            throw LocalModelError.unsupported("the app's bundled DA3 worker is missing")
        }
        isBusy = true; progress = 0; lastError = nil
        if checking { runtimeStatus = .checking }
    }
    private func finish() { runner = nil; isBusy = false; progress = nil }
    private func fail(_ error: Error) {
        lastError = error is CancellationError ? nil : error.localizedDescription
        runtimeStatus = error is CancellationError ? .missing : .failed(error.localizedDescription)
    }
    private func save(url: URL, managed: Bool) throws {
        try FileManager.default.createDirectory(at: registryURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(RuntimeRecord(path: url.standardizedFileURL.path, managed: managed)).write(to: registryURL, options: .atomic)
        self.pythonURL = url.standardizedFileURL; self.managed = managed
    }
    private func ownsManagedDirectory() -> Bool {
        let values = try? runtimeDirectory.resourceValues(forKeys: [.isSymbolicLinkKey])
        return values?.isSymbolicLink != true && runtimeDirectory.resolvingSymlinksInPath().path == runtimeDirectory.standardizedFileURL.path &&
            (try? String(contentsOf: runtimeDirectory.appendingPathComponent(".texture-studio-runtime"), encoding: .utf8)) == "Texture Studio DA3 runtime\n"
    }
    private func contains(_ url: URL, in root: URL) -> Bool { url.standardizedFileURL.path.hasPrefix(root.standardizedFileURL.path + "/") }
    private static func writeBoundedPNG(_ image: CGImage, to url: URL, maxDimension: Int) throws {
        let scale = min(1, Double(maxDimension) / Double(max(image.width, image.height)))
        let width = max(1, Int(Double(image.width) * scale)), height = max(1, Int(Double(image.height) * scale))
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw LocalModelError.unsupported("could not prepare the bounded RGB input")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let input = context.makeImage(), let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw LocalModelError.unsupported("could not encode the inference input")
        }
        CGImageDestinationAddImage(destination, input, nil)
        guard CGImageDestinationFinalize(destination) else { throw LocalModelError.unsupported("PNG input encoding failed") }
    }
    private static func availableMemory() -> UInt64 {
        var statistics = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &statistics) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return (UInt64(statistics.free_count) + UInt64(statistics.inactive_count) + UInt64(statistics.speculative_count)) * UInt64(max(1, sysconf(_SC_PAGESIZE)))
    }
    private struct RuntimeRecord: Codable { let path: String; let managed: Bool }
    private struct RuntimeInfo: Decodable { let ready: Bool; let python: String; let torch: String; let device: String }
    struct DepthMetadata: Codable, Sendable {
        let width: Int; let height: Int; let outputName: String; let interpretation: String; let provenance: ModelDepthProvenance
    }
}

/// Each inference is an isolated process. Termination releases its Metal memory.
/// stdout/stderr are files, avoiding pipe deadlocks during dependency installation.
private final class PythonProcessRunner: @unchecked Sendable {
    struct Event: Decodable, Sendable { let progress: Double?; let message: String? }
    private let executable: URL
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false
    private var timedOut = false
    init(executable: URL) { self.executable = executable }
    func cancel() {
        lock.withLock {
            cancelled = true
            if let process, process.isRunning { process.terminate() }
        }
        Task.detached { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            self?.lock.withLock {
                if let process = self?.process, process.isRunning { kill(process.processIdentifier, SIGKILL) }
            }
        }
    }
    func run(arguments: [String], timeout: Double, progress: (@Sendable (Event) -> Void)? = nil) async throws -> String {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("texture-python-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let stdoutURL = root.appendingPathComponent("stdout"), stderrURL = root.appendingPathComponent("stderr")
        FileManager.default.createFile(atPath: stdoutURL.path, contents: nil)
        FileManager.default.createFile(atPath: stderrURL.path, contents: nil)
        let stdout = try FileHandle(forWritingTo: stdoutURL), stderr = try FileHandle(forWritingTo: stderrURL)
        defer { try? stdout.close(); try? stderr.close() }
        let process = Process(); process.executableURL = executable; process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONPATH"); environment.removeValue(forKey: "PYTHONHOME")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"; environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"; environment["DA3_LOG_LEVEL"] = "ERROR"
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        process.environment = environment; process.standardOutput = stdout; process.standardError = stderr
        let monitor = Task.detached { [weak self] in
            let deadline = Date().addingTimeInterval(timeout)
            var delivered = 0
            while !Task.isCancelled {
                if Date() > deadline {
                    self?.lock.withLock { self?.timedOut = true }
                    self?.cancel(); return
                }
                if let data = try? Data(contentsOf: stdoutURL), let output = String(data: data, encoding: .utf8) {
                    let lines = output.split(separator: "\n")
                    if lines.count > delivered {
                        for line in lines.dropFirst(delivered) {
                            if let event = try? JSONDecoder().decode(Event.self, from: Data(line.utf8)) { progress?(event) }
                        }
                        delivered = lines.count
                    }
                }
                try? await Task.sleep(for: .milliseconds(300))
            }
        }
        defer { monitor.cancel() }
        let status = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Int32, Error>) in
                process.terminationHandler = { process in continuation.resume(returning: process.terminationStatus) }
                do {
                    try lock.withLock {
                        if cancelled { throw CancellationError() }
                        self.process = process
                        try process.run()
                    }
                } catch { continuation.resume(throwing: error) }
            }
        } onCancel: { self.cancel() }
        if lock.withLock({ timedOut }) { throw LocalModelError.unsupported("the Python operation exceeded its \(Int(timeout))-second time limit and was stopped. Reduce inference size or check the runtime") }
        if lock.withLock({ cancelled }) || Task.isCancelled { throw CancellationError() }
        let output = (try? String(contentsOf: stdoutURL, encoding: .utf8)) ?? ""
        guard status == 0 else {
            struct WorkerError: Decodable { let error: String }
            let error = output.split(separator: "\n").compactMap { try? JSONDecoder().decode(WorkerError.self, from: Data($0.utf8)).error }.first
            let diagnostics = ((try? String(contentsOf: stderrURL, encoding: .utf8)) ?? "").suffix(3000)
            throw LocalModelError.unsupported("\(error ?? "Python exited with status \(status)"). \(diagnostics)")
        }
        return output
    }
}
