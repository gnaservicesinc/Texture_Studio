import CoreML
import CoreVideo
import CryptoKit
import XCTest
@testable import TextureStudio

@MainActor
final class ModelManagerTests: XCTestCase {
    private func fixture() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("TextureModelTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func descriptor(bytes: Data = Data("verified model".utf8)) -> LocalModelDescriptor {
        LocalModelDescriptor(id: "fixture", name: "Fixture", summary: "Test", license: "Test",
            licenseURL: URL(string: "https://example.test/license")!, sourceURL: URL(string: "https://example.test/model")!,
            packageName: "Fixture.mlpackage", artifacts: [ModelDownloadArtifact(relativePath: "weights.bin",
                url: URL(string: "https://example.test/model/weights.bin")!, byteCount: Int64(bytes.count),
                sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())])
    }

    private func waitForCompletion(_ manager: ModelManager, id: String = "fixture") async throws {
        for _ in 0..<500 {
            if !manager.status(for: id).isBusy { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Model operation did not complete")
    }

    func testManagedDownloadRemovalPreservesUnrelatedFiles() async throws {
        let root = try fixture()
        let sentinel = root.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: sentinel)
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(), validator: FixtureValidator())
        manager.download(id: "fixture")
        try await waitForCompletion(manager)
        XCTAssertEqual(manager.status(for: "fixture"), .ready)
        XCTAssertEqual(manager.records["fixture"]?.isManaged, true)
        let modelURL = try XCTUnwrap(manager.availableURL(for: "fixture"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: modelURL.path))
        try manager.remove(id: "fixture")
        XCTAssertEqual(manager.status(for: "fixture"), .missing)
        XCTAssertFalse(FileManager.default.fileExists(atPath: modelURL.path))
        XCTAssertEqual(try Data(contentsOf: sentinel), Data("keep".utf8))
    }

    func testExternalModelIsUnlinkedAndMovedPathCanBeReconnected() async throws {
        let root = try fixture(), externalRoot = try fixture()
        let original = externalRoot.appendingPathComponent("model.mlmodel")
        let moved = externalRoot.appendingPathComponent("moved.mlmodel")
        try Data("external".utf8).write(to: original)
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(), validator: FixtureValidator())
        try await manager.locate(id: "fixture", url: original)
        try FileManager.default.moveItem(at: original, to: moved)
        manager.refresh()
        XCTAssertEqual(manager.status(for: "fixture"), .missing)
        XCTAssertNil(manager.availableURL(for: "fixture"))
        try await manager.locate(id: "fixture", url: moved)
        try manager.remove(id: "fixture")
        XCTAssertEqual(try Data(contentsOf: moved), Data("external".utf8))
        XCTAssertNil(manager.records["fixture"])
    }

    func testMovedManagedModelCanReconnectWithoutRedownloadOrDeletingIt() async throws {
        let root = try fixture(), externalRoot = try fixture()
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(), validator: FixtureValidator())
        manager.download(id: "fixture")
        try await waitForCompletion(manager)
        let original = try XCTUnwrap(manager.availableURL(for: "fixture"))
        let moved = externalRoot.appendingPathComponent("Moved.mlpackage")
        try FileManager.default.moveItem(at: original, to: moved)
        manager.refresh()
        XCTAssertEqual(manager.status(for: "fixture"), .missing)
        try await manager.locate(id: "fixture", url: moved)
        XCTAssertEqual(manager.records["fixture"]?.isManaged, false)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("fixture").path))
        try manager.remove(id: "fixture")
        XCTAssertTrue(FileManager.default.fileExists(atPath: moved.path))
    }

    func testChecksumFailureDoesNotPublishAndRetrySucceeds() async throws {
        let root = try fixture()
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(corruptFirst: true), validator: FixtureValidator())
        manager.download(id: "fixture")
        try await waitForCompletion(manager)
        guard case .failed = manager.status(for: "fixture") else { return XCTFail("Checksum corruption was accepted") }
        XCTAssertNil(manager.records["fixture"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("fixture").path))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent(".staging").path).isEmpty)
        manager.download(id: "fixture")
        try await waitForCompletion(manager)
        XCTAssertEqual(manager.status(for: "fixture"), .ready)
    }

    func testCancellationLeavesNoManagedModelOrPartialArtifacts() async throws {
        let root = try fixture()
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(delay: .seconds(3)), validator: FixtureValidator())
        manager.download(id: "fixture")
        try await Task.sleep(for: .milliseconds(20))
        manager.cancelDownload(id: "fixture")
        try await waitForCompletion(manager)
        XCTAssertEqual(manager.status(for: "fixture"), .missing)
        XCTAssertNil(manager.lastError)
        XCTAssertNil(manager.records["fixture"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("fixture").path))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent(".staging").path).isEmpty)
    }

    func testCoreMLValidationFailureDoesNotPublish() async throws {
        let root = try fixture()
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(), validator: FixtureValidator(reject: true))
        manager.download(id: "fixture")
        try await waitForCompletion(manager)
        guard case .failed = manager.status(for: "fixture") else { return XCTFail("Invalid model was accepted") }
        XCTAssertNil(manager.availableURL(for: "fixture"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("fixture").path))
    }

    func testAmbiguousOutputRequiresChoiceAndPersistsChoice() async throws {
        let root = try fixture(), externalRoot = try fixture()
        let external = externalRoot.appendingPathComponent("model.mlmodel")
        try Data("external".utf8).write(to: external)
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(), validator: FixtureValidator(ambiguous: true))
        try await manager.locate(id: "fixture", url: external)
        XCTAssertNil(manager.records["fixture"]?.selectedOutput)
        XCTAssertThrowsError(try manager.chooseOutput(id: "fixture", name: "not-depth"))
        try manager.chooseOutput(id: "fixture", name: "depth-two")
        let reopened = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(), validator: FixtureValidator())
        XCTAssertEqual(reopened.records["fixture"]?.selectedOutput, "depth-two")
    }

    func testTamperedManagedRecordCannotDeleteExternalModel() throws {
        let root = try fixture(), externalRoot = try fixture()
        let external = externalRoot.appendingPathComponent("Fixture.mlpackage")
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        let record = InstalledLocalModel(path: external.path, isManaged: true, interface: FixtureValidator.interface,
            selectedOutput: "depth", installedAt: Date())
        try JSONEncoder().encode(["fixture": record]).write(to: root.appendingPathComponent("model-records.json"))
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(), validator: FixtureValidator())
        XCTAssertThrowsError(try manager.remove(id: "fixture"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: external.path))
    }

    func testManagedDirectorySymlinkCannotDeleteItsExternalTarget() throws {
        let root = try fixture(), externalRoot = try fixture()
        let external = externalRoot.appendingPathComponent("Fixture.mlpackage")
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        let symlink = root.appendingPathComponent("fixture")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: externalRoot)
        let record = InstalledLocalModel(path: symlink.appendingPathComponent("Fixture.mlpackage").path,
            isManaged: true, interface: FixtureValidator.interface, selectedOutput: "depth", installedAt: Date())
        try JSONEncoder().encode(["fixture": record]).write(to: root.appendingPathComponent("model-records.json"))
        let manager = ModelManager(storageDirectory: root, catalog: [descriptor()],
            transfer: FixtureTransfer(), validator: FixtureValidator())
        XCTAssertThrowsError(try manager.remove(id: "fixture"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: external.path))
    }






    func testCatalogContainsOnlyDeveloperCoreML() {
        XCTAssertEqual(LocalModelDescriptor.catalog.map(\.id), [LocalModelDescriptor.customDepthID])
        XCTAssertEqual(LocalModelDescriptor.catalog.first?.backend, .coreML)
        XCTAssertTrue(LocalModelDescriptor.catalog.allSatisfy { !$0.downloadable })
    }

    func testOpeningLibraryNeverDeletesUnknownModelFiles() throws {
        let root = try fixture()
        let package = root.appendingPathComponent("development-model/Model.mlpackage")
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        let sentinel = package.appendingPathComponent("weights.bin")
        try Data("keep existing weights".utf8).write(to: sentinel)
        let record = InstalledLocalModel(path: package.path, isManaged: true, interface: FixtureValidator.interface,
            selectedOutput: "depth", installedAt: Date())
        try JSONEncoder().encode(["development-model": record]).write(to: root.appendingPathComponent("model-records.json"))
        _ = ModelManager(storageDirectory: root)
        XCTAssertEqual(try Data(contentsOf: sentinel), Data("keep existing weights".utf8))
    }

    func testRawTensorDecodingHonorsStridesAndKeepsValues() throws {
        let memory = UnsafeMutablePointer<Float>.allocate(capacity: 10)
        memory.initialize(repeating: 1234, count: 10)
        memory[0] = -3; memory[1] = 0.5; memory[2] = 7
        memory[5] = 2; memory[6] = -0.125; memory[7] = 9
        let array = try MLMultiArray(dataPointer: memory, shape: [1, 2, 3], dataType: .float32,
            strides: [10, 5, 1], deallocator: { $0.assumingMemoryBound(to: Float.self).deallocate() })
        let (width, height, values) = try ModelDepthService.decode(array)
        XCTAssertEqual(width, 3); XCTAssertEqual(height, 2)
        XCTAssertEqual(values, [-3, 0.5, 7, 2, -0.125, 9])
    }

    func testRawFloat16ImageDecodingKeepsNativeValuesWithRowPadding() throws {
        var optional: CVPixelBuffer?
        XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 3, 2, kCVPixelFormatType_OneComponent16Half,
            [kCVPixelBufferBytesPerRowAlignmentKey as String: 64] as CFDictionary, &optional), kCVReturnSuccess)
        let buffer = try XCTUnwrap(optional)
        CVPixelBufferLockBaseAddress(buffer, [])
        let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer))
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let expected: [Float16] = [-3, 0.5, 7, 2, -0.125, 9]
        for y in 0..<2 { for x in 0..<3 { base.storeBytes(of: expected[y * 3 + x], toByteOffset: y * stride + x * 2, as: Float16.self) } }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let decoded = try ModelDepthService.decode(buffer)
        XCTAssertEqual(decoded.2, expected.map(Float.init))
    }
}

private actor FixtureTransfer: ModelArtifactTransferring {
    var corruptFirst: Bool
    let delay: Duration
    init(corruptFirst: Bool = false, delay: Duration = .zero) { self.corruptFirst = corruptFirst; self.delay = delay }
    func fetch(_ artifact: ModelDownloadArtifact, to destination: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        if delay > .zero { try await Task.sleep(for: delay) }
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        var data = Data("verified model".utf8)
        if corruptFirst { data[0] = 0; corruptFirst = false }
        try data.write(to: destination)
        progress(1)
    }
}

private struct FixtureValidator: ModelValidating {
    var reject = false
    var ambiguous = false
    static let interface = ModelInterface(inputName: "image", inputWidth: 8, inputHeight: 8,
        outputs: [ModelOutputDescriptor(name: "depth", width: 8, height: 8, storage: "Float32 tensor")], interpretation: "Fixture")
    func inspect(at url: URL) async throws -> ModelInterface {
        if reject { throw LocalModelError.unsupported("invalid fixture") }
        if ambiguous { return ModelInterface(inputName: "image", inputWidth: 8, inputHeight: 8,
            outputs: [ModelOutputDescriptor(name: "depth-one", width: 8, height: 8, storage: "Float32 tensor"),
                      ModelOutputDescriptor(name: "depth-two", width: 8, height: 8, storage: "Float32 tensor")], interpretation: "Fixture") }
        return Self.interface
    }
}
