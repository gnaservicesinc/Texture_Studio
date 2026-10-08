import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class ModelPythonDepthTests: XCTestCase {
    private func fixture() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("PythonDepthTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
    private func backend(_ root: URL) throws -> URL {
        let directory = root.appendingPathComponent("backend")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("# fixture".utf8).write(to: directory.appendingPathComponent("worker.py"))
        return directory
    }
    private func executable(_ directory: URL, script: String) throws -> URL {
        let url = directory.appendingPathComponent("python-fixture")
        try Data(("#!/bin/sh\n" + script).utf8).write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }
    private func provenance(rowOrder: String = "top-down", revision: String = LocalModelDescriptor.da3Revision) -> ModelDepthProvenance {
        ModelDepthProvenance(modelID: "depth-anything/DA3-GIANT-1.1", revision: revision,
            checkpointSHA256: LocalModelDescriptor.da3WeightsSHA256, upstreamRevision: "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
            backend: "PyTorch", device: "mps", precision: "Float32", inputWidth: 2, inputHeight: 2,
            processResolution: 1036, fullSourceFieldOfView: true, rowOrder: rowOrder, elapsedSeconds: 1)
    }
    private func metadata(provenance: ModelDepthProvenance? = nil) -> PythonDepthService.DepthMetadata {
        PythonDepthService.DepthMetadata(width: 2, height: 2, outputName: "depth",
            interpretation: "relative_camera_z_depth", provenance: provenance ?? self.provenance())
    }
    private func bytes(_ values: [Float]) -> Data {
        var result = Data()
        for value in values { var bits = value.bitPattern.littleEndian; withUnsafeBytes(of: &bits) { result.append(contentsOf: $0) } }
        return result
    }

    func testRawFloatDepthPreservesValuesAndTopDownOrder() throws {
        let expected: [Float] = [-3, 0.125, 31.5, 0.000002]
        let result = try PythonDepthService.decodeDepth(bytes(expected), metadata: metadata())
        XCTAssertEqual(result.values.map(\.bitPattern), expected.map(\.bitPattern))
        XCTAssertEqual(result.width, 2); XCTAssertEqual(result.height, 2)
        XCTAssertEqual(result.provenance?.rowOrder, "top-down")
    }
    func testRawDepthRejectsWrongCheckpointShapeRowsAndNonfiniteValues() throws {
        let valid = bytes([1, 2, 3, 4])
        XCTAssertThrowsError(try PythonDepthService.decodeDepth(valid, metadata: metadata(provenance: provenance(revision: "old-GIANT"))))
        XCTAssertThrowsError(try PythonDepthService.decodeDepth(valid, metadata: metadata(provenance: provenance(rowOrder: "bottom-up"))))
        XCTAssertThrowsError(try PythonDepthService.decodeDepth(valid.dropLast(), metadata: metadata()))
        XCTAssertThrowsError(try PythonDepthService.decodeDepth(bytes([1, .infinity, 3, 4]), metadata: metadata()))
    }
    func testInferenceResolutionIsExplicitAndIndependentOfExport() throws {
        for edge in [1036, 1540, 2044] { XCTAssertNoThrow(try PythonDepthService.validateResolution(edge)) }
        for edge in [1024, 4098, 5120, 8192] { XCTAssertThrowsError(try PythonDepthService.validateResolution(edge)) }
    }
    func testExternalRuntimeIsUnlinkedWithoutDeletingItsExecutable() async throws {
        let root = try fixture(), external = try fixture()
        let python = try executable(external, script: "printf '%s\\n' '{\"ready\":true,\"python\":\"3.14\",\"torch\":\"2.14\",\"device\":\"mps\"}'\n")
        let runtime = PythonDepthService(runtimeDirectory: root.appendingPathComponent("managed"), backendDirectory: try backend(root))
        try await runtime.locatePython(at: python)
        XCTAssertTrue(runtime.runtimeStatus.isReady)
        try runtime.removeManagedRuntime()
        XCTAssertNil(runtime.pythonURL)
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: python.path))
    }
    func testExistingManagedRuntimeIsPreservedWhenSetupRejectsReplacement() async throws {
        let root = try fixture(), directory = root.appendingPathComponent("managed")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("Texture Studio DA3 runtime\n".utf8).write(to: directory.appendingPathComponent(".texture-studio-runtime"))
        let sentinel = directory.appendingPathComponent("keep")
        try Data("preserve".utf8).write(to: sentinel)
        let runtime = PythonDepthService(runtimeDirectory: directory, backendDirectory: try backend(root))
        do { try await runtime.setupRuntime(using: URL(fileURLWithPath: "/usr/bin/true")); XCTFail("Existing runtime overwritten") }
        catch { XCTAssertEqual(try Data(contentsOf: sentinel), Data("preserve".utf8)) }
    }
    func testCancelStopsRuntimeProbeAndDoesNotSaveExternalRecord() async throws {
        let root = try fixture(), external = try fixture()
        let python = try executable(external, script: "trap 'exit 143' TERM\nwhile true; do sleep 0.1; done\n")
        let runtime = PythonDepthService(runtimeDirectory: root.appendingPathComponent("managed"), backendDirectory: try backend(root))
        let task = Task { try await runtime.locatePython(at: python) }
        try await Task.sleep(for: .milliseconds(150))
        runtime.cancel()
        do { try await task.value; XCTFail("Cancelled probe succeeded") } catch is CancellationError { } catch { XCTFail("Unexpected \(error)") }
        XCTAssertFalse(runtime.isBusy)
        XCTAssertNil(runtime.pythonURL)
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: python.path))
    }
    func testAppBundlesIsolatedWorkerAndPinnedUpstreamSource() throws {
        let directory = try XCTUnwrap(Bundle.main.resourceURL).appendingPathComponent("DA3Backend")
        for file in ["worker.py", "setup_runtime.py", "requirements.txt", "UPSTREAM_LICENSE", "UPSTREAM_REVISION", "upstream/depth_anything_3/cfg.py"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path), "Missing \(file)")
        }
        let revision = try String(contentsOf: directory.appendingPathComponent("UPSTREAM_REVISION"), encoding: .utf8)
        XCTAssertTrue(revision.contains("3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"))
        let worker = try String(contentsOf: directory.appendingPathComponent("worker.py"), encoding: .utf8)
        XCTAssertTrue(worker.contains(LocalModelDescriptor.da3WeightsSHA256))
    }
}
