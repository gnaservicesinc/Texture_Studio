import CoreImage
import Darwin
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchReviewTests: XCTestCase {
    func testRuntimeReconnectKeepsExactSelectedCheckpointAndAdditionalMetadata() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        let original = SelectedMaterialCheckpoint(checkpointPath: "/original/model.pt", sha256: "selected-sha256", target: "height",
            pythonPath: "/old/python", workspacePath: "/old/workspace", modelDirectory: "/old/weights", codeDirectory: "/old/code")
        try original.save(to: registry)
        var metadata = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: registry)) as? [String: Any])
        metadata["future_model_metadata"] = ["normal_convention": "OpenGL +Y"]
        try JSONSerialization.data(withJSONObject: metadata).write(to: registry)
        try SelectedMaterialCheckpoint.refreshRuntime(pythonPath: "/new/python", workspacePath: "/new/workspace",
            modelDirectory: "/managed/weights", codeDirectory: "/managed/code", at: registry)
        let bytes = try Data(contentsOf: registry)
        let current = try JSONDecoder().decode(SelectedMaterialCheckpoint.self, from: bytes)
        XCTAssertEqual(current.checkpointPath, original.checkpointPath)
        XCTAssertEqual(current.sha256, original.sha256)
        XCTAssertEqual(current.target, original.target)
        XCTAssertEqual(current.pythonPath, "/new/python")
        XCTAssertEqual(current.workspacePath, "/new/workspace")
        XCTAssertEqual(current.modelDirectory, "/managed/weights")
        XCTAssertEqual(current.codeDirectory, "/managed/code")
        let preserved = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        XCTAssertEqual((preserved["future_model_metadata"] as? [String: String])?["normal_convention"], "OpenGL +Y")
        try SelectedMaterialCheckpoint.refreshRuntime(pythonPath: current.pythonPath, workspacePath: current.workspacePath,
            modelDirectory: current.modelDirectory, codeDirectory: current.codeDirectory, at: registry)
        XCTAssertEqual(try Data(contentsOf: registry), bytes, "An unchanged runtime does not rewrite the selection")
    }

    func testRuntimeRefreshNeverCreatesOrRepairsAnUnselectedCorruptRegistry() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        try SelectedMaterialCheckpoint.refreshRuntime(pythonPath: "python", workspacePath: "workspace",
            modelDirectory: "weights", codeDirectory: "code", at: registry)
        XCTAssertFalse(FileManager.default.fileExists(atPath: registry.path))
        let invalid = Data("{\"checkpointPath\":\"/preserve/incomplete.pt\"}".utf8)
        try invalid.write(to: registry)
        XCTAssertThrowsError(try SelectedMaterialCheckpoint.refreshRuntime(pythonPath: "python", workspacePath: "workspace",
            modelDirectory: "weights", codeDirectory: "code", at: registry))
        XCTAssertEqual(try Data(contentsOf: registry), invalid)
    }

    func testQuickWorkerDrainsFinalBurstAndPreservesSplitUTF8() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        // The first four-byte scalar straddles the reader's 65,536-byte boundary.
        let expected = String(repeating: "x", count: 65535) + "🪵" + String(repeating: "y", count: 65533) + "\ncheckpoint saved ✓\n"
        let collector = WorkbenchLogCollector()
        let output = try await WorkbenchProcess().run(executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "printf '%s' \"$1\"", "worker", expected], directory: nil,
            log: root.appendingPathComponent("worker.log"), onLog: { collector.append($0) })
        XCTAssertEqual(output, expected)
        XCTAssertEqual(collector.text, expected, "Return waits for every final log byte to be delivered")
    }

    func testFailedWorkerAlsoDrainsDiagnosticTail() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let collector = WorkbenchLogCollector()
        do {
            _ = try await WorkbenchProcess().run(executable: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "printf 'final failure detail\\n'; exit 7"], directory: nil,
                log: root.appendingPathComponent("worker.log"), onLog: { collector.append($0) })
            XCTFail("A failed worker must report its failure")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("final failure detail"))
            XCTAssertEqual(collector.text, "final failure detail\n")
        }
    }

    func testStopBeforeLaunchNeverStartsWorker() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let marker = root.appendingPathComponent("should-not-exist")
        let process = WorkbenchProcess()
        process.stop()
        do {
            _ = try await process.run(executable: URL(fileURLWithPath: "/usr/bin/touch"),
                arguments: [marker.path], directory: nil, log: root.appendingPathComponent("worker.log"), onLog: { _ in })
            XCTFail("A stopped operation must not launch another worker")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))
    }

    func testStopAndSaveAllowsInterruptHandlerToSaveAndDrainsItsLog() async throws {
        // Exercise cooperative SIGINT/save handling without Apple's Python
        // developer-tool shim, whose startup depends on Xcode tool selection.
        let executable = URL(fileURLWithPath: "/bin/sh")
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let ready = root.appendingPathComponent("ready")
        let checkpoint = root.appendingPathComponent("saved.checkpoint")
        let program = """
        checkpoint="$2"
        trap 'printf "%s" "saved optimizer state" > "$checkpoint"; printf "checkpoint saved ✓\\n"; exit 0' INT
        printf "%s" "ready" > "$1"
        while :; do /bin/sleep .02; done
        """
        let process = WorkbenchProcess()
        let collector = WorkbenchLogCollector()
        let operation = Task {
            defer { collector.markFinished() }
            return try await process.run(executable: executable, arguments: ["-c", program, "sigint-fixture", ready.path, checkpoint.path],
                directory: nil, log: root.appendingPathComponent("worker.log"), onLog: { collector.append($0) })
        }
        defer { operation.cancel(); process.stop() }
        let deadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: ready.path), !collector.isFinished, Date() < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        guard FileManager.default.fileExists(atPath: ready.path) else {
            operation.cancel()
            process.stop()
            var reason = "Worker returned without publishing readiness"
            do { _ = try await operation.value }
            catch { reason = error.localizedDescription }
            let tail = (try? String(contentsOf: root.appendingPathComponent("worker.log"), encoding: .utf8)) ?? collector.text
            XCTFail("SIGINT fixture did not become ready. Executable: \(executable.path). \(reason). Log: \(tail.suffix(4000))")
            return
        }
        process.stopAndSave()
        let output = try await operation.value
        XCTAssertEqual(try String(contentsOf: checkpoint, encoding: .utf8), "saved optimizer state")
        XCTAssertEqual(output, "checkpoint saved ✓\n")
        XCTAssertEqual(collector.text, output)
    }

    func testQuitAbortsWorkerIgnoringInterruptAndTerminateWithinBoundedTime() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let ready = root.appendingPathComponent("ready")
        let program = """
        trap '' INT TERM
        printf '%s' "$$" > "$1"
        while :; do /bin/sleep .02; done
        """
        let process = WorkbenchProcess()
        let lifecycle = WorkbenchLifecycle()
        lifecycle.add(process)
        let collector = WorkbenchLogCollector()
        let operation = Task {
            defer { collector.markFinished(); lifecycle.remove(process) }
            return try await process.run(executable: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", program, "abort-fixture", ready.path], directory: nil,
                log: root.appendingPathComponent("worker.log"), onLog: { collector.append($0) })
        }
        defer { operation.cancel(); process.stop() }
        let readinessDeadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: ready.path), !collector.isFinished, Date() < readinessDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        let pid = try XCTUnwrap(Int32(String(contentsOf: ready, encoding: .utf8)))
        process.stopAndSave() // This worker ignores the cooperative save request.
        lifecycle.stopAll()   // Quitting must override a pending save and abort it.
        let deadline = Date().addingTimeInterval(4)
        while !collector.isFinished, Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        if !collector.isFinished {
            kill(pid, SIGKILL) // Clean up this test's worker even if the regression returns.
            XCTFail("Aborting a worker must not wait indefinitely for its signal handler")
        }
        do { _ = try await operation.value; XCTFail("An aborted worker cannot return a saved result") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertFalse(lifecycle.hasOperations)
    }

    func testLearnedHeightEXRMaterializationPreservesOrientationAndRawValues() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let values: [Float] = [-0.125, 0.25, 0.875, 1.25, 0.5, 0.0625]
        let depth = try TextureDepth(width: 3, height: 2, values: values, sourceLabel: "Asymmetric numeric fixture", interpretation: .surfaceHeight)
        let context = CIContext(options: [.useSoftwareRenderer: true, .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        let output = root.appendingPathComponent("height.exr")
        try FloatEXRWriter.write(depth.image, to: output, context: context, color: false)
        try FloatEXRWriter.verifyChannelPrecision(at: output, expected: .float32)
        // Python's worker exports a scalar Y channel, unlike the native writer's
        // scalar R channel. They have identical Float32 scanline storage.
        var bytes = try Data(contentsOf: output)
        let channel = try XCTUnwrap(bytes.range(of: Data([0x52, 0, 2, 0, 0, 0])))
        bytes[channel.lowerBound] = 0x59
        let workerOutput = root.appendingPathComponent("worker-height.exr")
        try bytes.write(to: workerOutput)
        let imported = try XCTUnwrap(CIImage(contentsOf: workerOutput, options: [.colorSpace: NSNull()]))
        var readback = [Float](repeating: 0, count: values.count)
        readback.withUnsafeMutableBytes {
            context.render(imported, toBitmap: $0.baseAddress!, rowBytes: 3 * 4, bounds: imported.extent, format: .Rf, colorSpace: nil)
        }
        XCTAssertEqual(readback, values, "EXR readback must keep native rows and signed/out-of-range height samples")
        let materialized = try TextureDepth(width: 3, height: 2, values: readback, sourceLabel: "Worker output", interpretation: .surfaceHeight)
        var final = [Float](repeating: 0, count: values.count)
        final.withUnsafeMutableBytes {
            context.render(materialized.image, toBitmap: $0.baseAddress!, rowBytes: 3 * 4, bounds: materialized.image.extent, format: .Rf, colorSpace: nil)
        }
        XCTAssertEqual(final, values)
        // Coordinates, not only round-trip arrays, establish the row convention:
        // Core Image uses bottom-left coordinates; bitmap row zero is the top.
        for (point, expected) in [(CGPoint(x: 0, y: 1), Float(-0.125)), (CGPoint(x: 0, y: 0), Float(1.25))] {
            var pixel = Float.zero
            withUnsafeMutableBytes(of: &pixel) {
                context.render(materialized.image, toBitmap: $0.baseAddress!, rowBytes: 4,
                    bounds: CGRect(origin: point, size: CGSize(width: 1, height: 1)), format: .Rf, colorSpace: nil)
            }
            XCTAssertEqual(pixel, expected)
        }
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("workbench-review-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}

private final class WorkbenchLogCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var chunks = ""
    private var finished = false
    var text: String { lock.withLock { chunks } }
    var isFinished: Bool { lock.withLock { finished } }
    func append(_ text: String) { lock.withLock { chunks += text } }
    func markFinished() { lock.withLock { finished = true } }
}
