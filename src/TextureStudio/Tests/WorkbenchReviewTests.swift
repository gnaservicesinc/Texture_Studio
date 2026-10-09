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
        let original = SelectedMaterialCheckpoint(checkpointPath: "/original/model.safetensors", sha256: "selected-sha256", target: "height",
            workspacePath: "/old/workspace", modelDirectory: "/old/weights")
        try original.save(to: registry)
        var metadata = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: registry)) as? [String: Any])
        metadata["future_model_metadata"] = ["normal_convention": "OpenGL +Y"]
        try JSONSerialization.data(withJSONObject: metadata).write(to: registry)
        try SelectedMaterialCheckpoint.refreshRuntime(workspacePath: "/new/workspace",
            modelDirectory: "/managed/weights", at: registry)
        let bytes = try Data(contentsOf: registry)
        let current = try JSONDecoder().decode(SelectedMaterialCheckpoint.self, from: bytes)
        XCTAssertEqual(current.checkpointPath, original.checkpointPath)
        XCTAssertEqual(current.sha256, original.sha256)
        XCTAssertEqual(current.target, original.target)
        XCTAssertEqual(current.workspacePath, "/new/workspace")
        XCTAssertEqual(current.modelDirectory, "/managed/weights")
        let preserved = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        XCTAssertEqual((preserved["future_model_metadata"] as? [String: String])?["normal_convention"], "OpenGL +Y")
        try SelectedMaterialCheckpoint.refreshRuntime(workspacePath: current.workspacePath,
            modelDirectory: current.modelDirectory, at: registry)
        XCTAssertEqual(try Data(contentsOf: registry), bytes, "An unchanged runtime does not rewrite the selection")
    }

    func testRuntimeRefreshNeverCreatesOrRepairsAnUnselectedCorruptRegistry() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("selected.json")
        try SelectedMaterialCheckpoint.refreshRuntime(workspacePath: "workspace",
            modelDirectory: "weights", at: registry)
        XCTAssertFalse(FileManager.default.fileExists(atPath: registry.path))
        let invalid = Data("{\"checkpointPath\":\"/preserve/incomplete.safetensors\"}".utf8)
        try invalid.write(to: registry)
        XCTAssertThrowsError(try SelectedMaterialCheckpoint.refreshRuntime(workspacePath: "workspace",
            modelDirectory: "weights", at: registry))
        XCTAssertEqual(try Data(contentsOf: registry), invalid)
    }

    func testNativeLifecycleCancelsPendingSaveAndReleasesOperation() async throws {
        let lifecycle = WorkbenchLifecycle()
        let control = NativeMaterialTrainingControl()
        let task = Task { try await Task.sleep(for: .seconds(30)) }
        lifecycle.add(control, cancel: { task.cancel() })
        XCTAssertTrue(lifecycle.hasOperations)
        control.stopAndSave()
        lifecycle.stopAll()
        do { try await task.value; XCTFail("Closing must cancel the native operation") }
        catch { XCTAssertTrue(error is CancellationError) }
        lifecycle.remove(control)
        XCTAssertFalse(lifecycle.hasOperations)
    }

    func testNativeEventLogKeepsUnicodeAndFinalCheckpointEvent() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("native.log")
        let log = NativeWorkbenchLog(url: url)
        let expected = String(repeating: "x", count: 65535) + "🪵" + "\ncheckpoint saved ✓\n"
        log.append(expected)
        XCTAssertEqual(log.text, expected)
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), expected)
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
        // Verify importing a scalar Y channel as well as the writer's R channel.
        // Both keep the same Float32 scanline storage.
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
