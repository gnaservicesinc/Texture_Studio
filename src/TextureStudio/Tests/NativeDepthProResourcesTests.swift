import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

final class NativeDepthProResourcesTests: XCTestCase {
    func testMissingConfiguredModelInstallsManagedPackageAndReusesItOffline() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let events = Events()
        let requested = fixture.root.appendingPathComponent("checkout/out/models/DepthPro.mlpackage")
        let result = try await fixture.resources.ensureAvailable(requestedURL: requested) { events.append($0) }
        XCTAssertEqual(result, fixture.managed)
        XCTAssertFalse(FileManager.default.fileExists(atPath: requested.deletingLastPathComponent().path))
        XCTAssertEqual(fixture.transfer.count, 3)
        XCTAssertEqual(try NativeDepthProService.modelIdentity(at: result), fixture.identity)
        XCTAssertTrue(events.values.contains { $0["ready"] as? Bool == true && $0["model_path"] as? String == result.path })
        XCTAssertTrue(events.values.contains { $0["bytes_total"] as? Int == 38 })
        _ = try await fixture.resources.ensureAvailable(requestedURL: requested)
        XCTAssertEqual(fixture.transfer.count, 3, "A verified cached package must work without another download")
        let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.managed.deletingLastPathComponent().appendingPathComponent("depthpro-install.json"))) as! [String: Any]
        XCTAssertEqual(receipt["package_sha256"] as? String, fixture.identity)
    }

    func testPresentDeveloperModelIsPreservedWithoutNetworkOrManagedDirectoryCreation() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let custom = fixture.root.appendingPathComponent("custom.mlmodel"), bytes = Data("custom model".utf8)
        try bytes.write(to: custom)
        let result = try await fixture.resources.ensureAvailable(requestedURL: custom)
        XCTAssertEqual(result, custom); XCTAssertEqual(try Data(contentsOf: custom), bytes)
        XCTAssertEqual(fixture.transfer.count, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.managed.deletingLastPathComponent().path))
    }

    func testIncompleteManagedPackageIsRepairedAndVerifiedStagingFilesResume() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let stage = fixture.managed.deletingLastPathComponent().appendingPathComponent(".DepthPro-download.mlpackage")
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
        try fixture.contents["Manifest.json"]!.write(to: stage.appendingPathComponent("Manifest.json"))
        try Data("interrupted transfer".utf8).write(to: stage.appendingPathComponent(".depthpro-partial-abandoned"))
        try Data("stale package metadata".utf8).write(to: stage.appendingPathComponent("stale.json"))
        try FileManager.default.createDirectory(at: fixture.managed, withIntermediateDirectories: true)
        try Data("partial".utf8).write(to: fixture.managed.appendingPathComponent("Manifest.json"))
        let result = try await fixture.resources.ensureAvailable(requestedURL: fixture.managed)
        XCTAssertEqual(fixture.transfer.count, 2)
        XCTAssertEqual(try NativeDepthProService.modelIdentity(at: result), fixture.identity)
        XCTAssertFalse(FileManager.default.fileExists(atPath: stage.path))
    }

    func testConcurrentSetupDownloadsOnceAndBothOperationsReceiveCompleteModel() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        async let first = fixture.resources.ensureAvailable(requestedURL: fixture.managed)
        async let second = fixture.resources.ensureAvailable(requestedURL: fixture.managed)
        let (a, b) = try await (first, second)
        XCTAssertEqual(a, b); XCTAssertEqual(fixture.transfer.count, 3)
        XCTAssertEqual(try NativeDepthProService.modelIdentity(at: a), fixture.identity)
    }

    func testManagedPackageAliasStillRepairsIncompleteInstallation() async throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        try FileManager.default.createDirectory(at: fixture.managed, withIntermediateDirectories: true)
        try Data("incomplete".utf8).write(to: fixture.managed.appendingPathComponent("Manifest.json"))
        let alias = fixture.root.appendingPathComponent("selected-alias.mlpackage")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.managed)
        let result = try await fixture.resources.ensureAvailable(requestedURL: alias)
        XCTAssertEqual(result.path, fixture.managed.path)
        XCTAssertEqual(fixture.transfer.count, 3)
        XCTAssertEqual(try NativeDepthProService.modelIdentity(at: result), fixture.identity)
        XCTAssertTrue(try alias.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true)
    }

    func testBadDownloadCannotReplaceExistingModelAndReportsAttemptedRecovery() async throws {
        let fixture = try Fixture(corrupt: true); defer { fixture.remove() }
        try FileManager.default.createDirectory(at: fixture.managed, withIntermediateDirectories: true)
        let previous = Data("old incomplete model".utf8), marker = fixture.managed.appendingPathComponent("original.txt")
        try previous.write(to: marker)
        do {
            _ = try await fixture.resources.ensureAvailable(requestedURL: fixture.managed)
            XCTFail("Corrupt model bytes cannot be published")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("after attempting its published model download"))
            XCTAssertTrue(error.localizedDescription.contains("SHA-256"))
        }
        XCTAssertEqual(fixture.transfer.count, 3, "Recovery retries before reporting its actual remaining obstacle")
        XCTAssertEqual(try Data(contentsOf: marker), previous)
    }

    func testCancelledDownloadDoesNotPublishIncompleteModelAndKeepsCancellationType() async throws {
        let fixture = try Fixture(cancel: true); defer { fixture.remove() }
        do {
            _ = try await fixture.resources.ensureAvailable(requestedURL: fixture.managed)
            XCTFail("Cancelled setup cannot publish a model")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(fixture.transfer.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.managed.path))
    }

    func testPublishedManifestPinsEveryFileAndMatchesExistingAdapterIdentity() {
        XCTAssertEqual(NativeDepthProResources.artifacts.count, 3)
        XCTAssertEqual(NativeDepthProResources.artifacts.reduce(0) { $0 + $1.byteCount }, 1_902_515_337)
        var digest = SHA256()
        digest.update(data: Data("texture-studio-depthpro-package-v1\n".utf8))
        for item in NativeDepthProResources.artifacts.sorted(by: { $0.relativePath < $1.relativePath }) {
            XCTAssertTrue(item.url.absoluteString.contains("/resolve/" + NativeDepthProService.upstreamRevision + "/DepthPro.mlpackage/"))
            digest.update(data: Data("\(item.relativePath.utf8.count):\(item.relativePath):\(item.byteCount):\(item.sha256)\n".utf8))
        }
        XCTAssertEqual(digest.finalize().map { String(format: "%02x", $0) }.joined(), NativeDepthProService.pinnedPackageSHA256)
    }

    func testPublishedSmallArtifactReportsIntermediateNativeDownloadProgress() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_DEPTHPRO_ACQUISITION_VERIFY"] == "1" else {
            throw XCTSkip("Opt-in published small model artifact download progress check.")
        }
        let fixture = try Fixture(); defer { fixture.remove() }
        let artifact = NativeDepthProResources.artifacts[1], destination = fixture.root.appendingPathComponent("downloaded.mlmodel")
        let fractions = Fractions()
        try await ModelDownloadService().fetch(artifact, to: destination) { fractions.append($0) }
        try ModelFileValidation.validate(destination, artifact: artifact)
        XCTAssertTrue(fractions.values.contains { $0 > 0 && $0 < 1 }, "A native download must report bytes received before completion")
        XCTAssertEqual(fractions.values.last, 1)
    }

    func testCancellingNativeDownloadDuringProgressFinishesBeforeCallerCleanup() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_DEPTHPRO_ACQUISITION_VERIFY"] == "1" else {
            throw XCTSkip("Opt-in native download cancellation check.")
        }
        let fixture = try Fixture(); defer { fixture.remove() }
        let destination = fixture.root.appendingPathComponent("cancelled.mlmodel"), cancellation = DownloadCancellation()
        let job = Task {
            try await ModelDownloadService().fetch(NativeDepthProResources.artifacts[1], to: destination) { fraction in
                if fraction > 0 { cancellation.cancel() }
            }
        }
        cancellation.install(job)
        do { try await job.value; XCTFail("The cancelled native transfer must not succeed") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path), "A delegate must not recreate the partial after cancellation returns")
    }

    func testRealMissingModelIsAcquiredAndNativeTrainingAndInferenceContinue() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["TEXTURE_STUDIO_DEPTHPRO_ACQUISITION_VERIFY"] == "1" else {
            throw XCTSkip("Opt-in genuine model acquisition and one-step native training/inference.")
        }
        let image = URL(fileURLWithPath: try XCTUnwrap(env["TEXTURE_STUDIO_DEPTHPRO_ACQUISITION_IMAGE"]))
        let height = URL(fileURLWithPath: try XCTUnwrap(env["TEXTURE_STUDIO_DEPTHPRO_ACQUISITION_HEIGHT"]))
        let run = NativeDepthProContract.managedModelURL.deletingLastPathComponent()
            .appendingPathComponent("verification-\(UUID().uuidString)")
        let output = run.appendingPathComponent("inference"), events = Events()
        let started = Date()
        let originalPhoto = try NativePNG.decode(Data(contentsOf: image)), originalHeight = try NativePNG.decode(Data(contentsOf: height))
        XCTAssertEqual(originalPhoto.header.width, originalHeight.header.width)
        XCTAssertEqual(originalPhoto.header.height, originalHeight.header.height)
        let rectangle = [(originalPhoto.header.width - 256) / 2, (originalPhoto.header.height - 256) / 2, 256, 256]
        let photoBytes = try originalPhoto.crop(rectangle).encoded(), heightBytes = try originalHeight.crop(rectangle).encoded()
        let dataset = run.appendingPathComponent("Sources"), sample = dataset.appendingPathComponent("White Stucco")
        try FileManager.default.createDirectory(at: sample, withIntermediateDirectories: true)
        let croppedPhoto = sample.appendingPathComponent("diffuse.png"), croppedHeight = sample.appendingPathComponent("height.png")
        try photoBytes.write(to: croppedPhoto); try heightBytes.write(to: croppedHeight)
        try NativeSourceFileTags.update(croppedPhoto, role: "input", classLabel: "Stucco", classifierIdentity: "user", reviewStatus: "approved")
        try NativeSourceFileTags.update(croppedHeight, role: "height", classLabel: "Stucco", classifierIdentity: "user", reviewStatus: "approved")
        _ = try await NativeMaterialDatasetService.run(arguments: ["edit-dataset", "--dataset", dataset.path,
            "--training-size", "256", "--validation-enabled", "no"])
        let classifier = NativeMaterialSpecialistService.Classifier(path: "/unused-tagged-only", sha256: String(repeating: "a", count: 64),
            labels: ["Stucco"], predict: { _ in throw StudioError("Manual teaching must not invoke classifier prediction") })
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: run.appendingPathComponent("Categories"), classifier: classifier)
        let category = try XCTUnwrap(registry.specialists.first { $0.classLabel == "Stucco" })
        XCTAssertEqual(category.materialIds.count, 1)
        let missing = run.appendingPathComponent("missing/DepthPro.mlpackage")
        let trainedText = try await NativeMaterialTrainer.run(arguments: ["train", "--dataset", category.datasetPath,
            "--output", run.appendingPathComponent("trained").path, "--target", "height", "--size", "256",
            "--model-family", "depthpro-height-refiner", "--compact-width", "16", "--updates-per-map", "1",
            "--learning-rate", "0.001", "--depthpro-model", missing.path],
            onEvent: { event in events.append(event); FileHandle.standardOutput.write(Data(event.utf8)) })
        let trained = try JSONSerialization.jsonObject(with: Data(trainedText.utf8)) as! [String: Any]
        XCTAssertEqual(trained["completed_updates"] as? Int, 1)
        let checkpoint = try XCTUnwrap(trained["checkpoint_path"] as? String)
        let hash = try NativeMaterialTrainer.checksum(Data(contentsOf: URL(fileURLWithPath: checkpoint)))
        let text = try await NativeMaterialTrainer.run(arguments: ["infer", "--image", croppedPhoto.path,
            "--output", output.path, "--target", "height", "--size", "256", "--model-family", "depthpro-height-refiner",
            "--checkpoint", checkpoint, "--expected-sha256", hash, "--depthpro-model", missing.path],
            onEvent: { event in events.append(event); FileHandle.standardOutput.write(Data(event.utf8)) })
        let result = try JSONSerialization.jsonObject(with: Data(text.utf8)) as! [String: Any]
        XCTAssertEqual(result["depthpro_model_sha256"] as? String, NativeDepthProService.pinnedPackageSHA256)
        XCTAssertTrue(events.values.contains { $0["ready"] as? Bool == true })
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("height.float32.exr").path))
        let report: [String: Any] = ["model_path": NativeDepthProContract.managedModelURL.path,
            "package_sha256": NativeDepthProService.pinnedPackageSHA256,
            "model_bytes": NativeDepthProResources.artifacts.reduce(0) { $0 + $1.byteCount }, "missing_selection_recovered": true,
            "numeric_inference_result": result, "output_path": output.path, "wall_seconds": Date().timeIntervalSince(started),
            "training_result": trained, "training_started": true, "completed_optimizer_updates": 1,
            "quality_qualification": false, "source_photo": image.path, "original_height": height.path,
            "crop_rectangle": rectangle, "download_events": events.values.filter { $0["operation"] as? String == "Downloading DepthPro model" }.count]
        if let path = env["TEXTURE_STUDIO_DEPTHPRO_ACQUISITION_REPORT"] {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: path))
        }
    }

    private struct Fixture {
        let root: URL, managed: URL, identity: String
        let contents: [String: Data]
        let transfer: Transfer
        let resources: NativeDepthProResources
        init(corrupt: Bool = false, cancel: Bool = false) throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("depthpro-resource-test-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            managed = root.appendingPathComponent("user-storage/DepthPro.mlpackage")
            contents = ["Manifest.json": Data("manifest".utf8), "Data/com.apple.CoreML/model.mlmodel": Data("specification".utf8),
                "Data/com.apple.CoreML/weights/weight.bin": Data("published-weights".utf8)]
            let reference = root.appendingPathComponent("reference.mlpackage")
            for (name, bytes) in contents {
                let file = reference.appendingPathComponent(name)
                try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                try bytes.write(to: file)
            }
            identity = try NativeDepthProService.modelIdentity(at: reference)
            let payloads = contents
            let artifacts = payloads.keys.sorted().map { name in
                ModelDownloadArtifact(relativePath: name, url: URL(string: "https://fixture.invalid/" + name)!,
                    byteCount: Int64(payloads[name]!.count), sha256: NativeMaterialTrainer.checksum(payloads[name]!))
            }
            transfer = Transfer(contents: contents, corrupt: corrupt, cancel: cancel)
            resources = NativeDepthProResources(managedURL: managed, artifacts: artifacts, expectedIdentity: identity, transfer: transfer)
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
    }
    private final class Transfer: ModelArtifactTransferring, @unchecked Sendable {
        let contents: [String: Data], corrupt: Bool, cancel: Bool
        private let lock = NSLock()
        private var requests = 0
        var count: Int { lock.withLock { requests } }
        init(contents: [String: Data], corrupt: Bool, cancel: Bool) { self.contents = contents; self.corrupt = corrupt; self.cancel = cancel }
        func fetch(_ artifact: ModelDownloadArtifact, to destination: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
            lock.withLock { requests += 1 }
            if cancel { throw CancellationError() }
            var bytes = contents[artifact.relativePath]!
            if corrupt { bytes[0] ^= 1 }
            try bytes.write(to: destination); progress(1)
        }
    }
    private final class Events: @unchecked Sendable {
        private let lock = NSLock()
        private var collected: [[String: Any]] = []
        var values: [[String: Any]] { lock.withLock { collected } }
        func append(_ text: String) {
            if let value = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] {
                lock.withLock { collected.append(value) }
            }
        }
    }
    private final class Fractions: @unchecked Sendable {
        private let lock = NSLock()
        private var collected: [Double] = []
        var values: [Double] { lock.withLock { collected } }
        func append(_ fraction: Double) { lock.withLock { collected.append(fraction) } }
    }
    private final class DownloadCancellation: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<Void, Error>?
        private var requested = false
        func install(_ task: Task<Void, Error>) {
            let cancel = lock.withLock { self.task = task; return requested }
            if cancel { task.cancel() }
        }
        func cancel() {
            let task = lock.withLock { requested = true; return self.task }
            task?.cancel()
        }
    }
}
