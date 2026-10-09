import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class HuggingFaceUploadTests: XCTestCase {
    func testBlankDestinationUsesSavedAccountAndExactSelectedCheckpoint() async throws {
        let fixture = try UploadFixture()
        defer { fixture.remove() }
        let store = fixture.store { args, _ in
            if args.first == "hub-models" { return "{\"models\":[]}" }
            return "{\"authenticated\":true,\"username\":\"artist\",\"message\":\"Signed in as artist\"}"
        }
        let first = try fixture.checkpoint(hash: "abcdef0123456789", directory: "Native 2K Height")
        let second = try fixture.checkpoint(hash: "9876543210abcdef", directory: "Refined Height")
        store.checkpoints = [first, second]
        store.selectedCheckpointId = first.id
        XCTAssertEqual(store.effectiveUploadRepo, "")
        XCTAssertFalse(store.canUploadSelectedCheckpoint)
        store.refreshUploadAccount()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.uploadAccount, "artist")
        XCTAssertEqual(store.effectiveUploadRepo, "artist/texture-height-native-2k-height-abcdef01")
        XCTAssertTrue(store.canUploadSelectedCheckpoint)
        store.selectedCheckpointId = second.id
        XCTAssertEqual(store.effectiveUploadRepo, "artist/texture-height-refined-height-98765432")
        XCTAssertFalse(store.uploadPublic)
    }

    func testUploadPackagesTheSelectedModelDirectlyAndNeverUsesPreviousExport() async throws {
        let fixture = try UploadFixture()
        defer { fixture.remove() }
        var calls: [[String]] = []
        let selected = try fixture.checkpoint(hash: "selected-sha", directory: "Selected Run")
        let store = fixture.store { args, _ in
            calls.append(args)
            if args.first == "hub-models" { return "{\"models\":[]}" }
            if args.first == "hub-account" {
                return "{\"authenticated\":true,\"username\":\"artist\",\"message\":\"Signed in as artist\"}"
            }
            XCTAssertEqual(args.first, "upload-selected")
            XCTAssertEqual(self.value("--checkpoint", in: args), selected.checkpointPath)
            XCTAssertEqual(self.value("--expected-sha256", in: args), selected.sha256)
            XCTAssertFalse(args.contains("--package"))
            return try fixture.uploadResponse(repository: "artist/chosen-destination", checksum: selected.sha256,
                path: try XCTUnwrap(self.value("--output", in: args)), isPrivate: false)
        }
        store.checkpoints = [selected]
        store.selectedCheckpointId = selected.id
        store.lastPackageURL = fixture.root.appendingPathComponent("another-model-export")
        store.lastPackageCheckpointId = "another-model-sha"
        store.uploadRepo = " artist/chosen-destination "
        store.uploadPublic = true
        store.refreshUploadAccount()
        try await settled(store)
        store.uploadPackage()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.compactMap(\.first), ["hub-account", "hub-models", "upload-selected", "hub-models"])
        let upload = try XCTUnwrap(calls.first { $0.first == "upload-selected" })
        XCTAssertEqual(value("--repo", in: upload), "artist/chosen-destination")
        XCTAssertTrue(upload.contains("--public"))
        let staged = URL(fileURLWithPath: try XCTUnwrap(value("--output", in: upload)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.path), "Upload staging must be purged after publication")
        XCTAssertEqual(store.lastUploadURL?.absoluteString, "https://huggingface.co/artist/chosen-destination/commit/verified")
    }

    func testAccountSetupIsInlineAndUploadPreferencesPersist() async throws {
        let fixture = try UploadFixture()
        defer { fixture.remove() }
        var calls = 0
        let store = fixture.store { _, _ in
            calls += 1
            return "{\"authenticated\":false,\"username\":null,\"message\":\"Run hf auth login, then click Refresh Account.\"}"
        }
        let selected = try fixture.checkpoint(hash: "selected-sha", directory: "Run")
        store.checkpoints = [selected]
        store.selectedCheckpointId = selected.id
        store.refreshUploadAccount()
        try await settled(store)
        XCTAssertNil(store.error, "A missing saved login is setup guidance, not a failed upload")
        XCTAssertNil(store.uploadAccount)
        XCTAssertTrue(store.uploadAccountMessage.contains("hf auth login"))
        store.uploadPackage()
        XCTAssertEqual(calls, 1, "No upload is attempted without authentication")
        store.uploadRepo = "studio/shared-model"
        store.uploadPublic = true
        store.saveUploadConfiguration()
        let reopened = fixture.store { _, _ in XCTFail("Discovery should not start by constructing a store"); return "" }
        XCTAssertEqual(reopened.uploadRepo, "studio/shared-model")
        XCTAssertTrue(reopened.uploadPublic)
    }

    func testUploadResponseMustMatchTheCapturedSelectedCheckpoint() async throws {
        let fixture = try UploadFixture()
        defer { fixture.remove() }
        let store = fixture.store { args, _ in
            if args.first == "hub-models" { return "{\"models\":[]}" }
            if args.first == "hub-account" {
                return "{\"authenticated\":true,\"username\":\"artist\",\"message\":\"Signed in as artist\"}"
            }
            return try fixture.uploadResponse(repository: try XCTUnwrap(self.value("--repo", in: args)),
                checksum: "wrong-model", path: "/wrong-package", isPrivate: true)
        }
        let selected = try fixture.checkpoint(hash: "chosen-model", directory: "Run")
        store.checkpoints = [selected]
        store.selectedCheckpointId = selected.id
        store.refreshUploadAccount()
        try await settled(store)
        store.uploadPackage()
        try await settled(store)
        XCTAssertTrue(store.error?.contains("did not match") == true)
        XCTAssertNil(store.lastPackageURL)
        XCTAssertNil(store.lastUploadURL)
    }

    func testRepositoryValidationRejectsAmbiguousOrUnsafeDestinations() {
        for invalid in ["", "artist", "/model", "artist/", "artist/a/b", "artist/with space", "artist/.hidden", "artist/end-", "artist/a..b", "artist/a--b", "artist/" + String(repeating: "a", count: 97)] {
            XCTAssertFalse(HuggingFaceUpload.validRepository(invalid), invalid)
        }
        XCTAssertTrue(HuggingFaceUpload.validRepository("artist/material_height.v2"))
    }

    private func settled(_ store: WorkbenchStore) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while store.isBusy, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.isBusy, "Upload operation did not settle")
    }
    private func value(_ flag: String, in args: [String]) -> String? {
        guard let index = args.firstIndex(of: flag), args.indices.contains(index + 1) else { return nil }
        return args[index + 1]
    }
}

@MainActor private final class UploadFixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("hf-upload-tests-\(UUID().uuidString)")
    let suite = "org.ipde.upload-tests.\(UUID().uuidString)"
    let preferences: UserDefaults
    init() throws {
        preferences = UserDefaults(suiteName: suite)!
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        preferences.set(root.path, forKey: "workspace")
        preferences.set(false, forKey: "uploadPublic")
    }
    func remove() {
        preferences.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: root)
    }
    func store(worker: @escaping @MainActor ([String], String) async throws -> String) -> WorkbenchStore {
        WorkbenchStore(preferences: preferences, workerOverride: worker)
    }
    func checkpoint(hash: String, directory: String) throws -> WorkbenchCheckpoint {
        let data = try JSONSerialization.data(withJSONObject: ["checkpoint_path": root.appendingPathComponent(directory).appendingPathComponent("adapter.safetensors").path,
            "sha256": hash, "schema": "texture-studio-material-lora-v1", "target": "height", "step": 25,
            "compatible": true, "variant": "lora"])
        return try WorkbenchProcess.decode(WorkbenchCheckpoint.self, output: String(decoding: data, as: UTF8.self))
    }
    func uploadResponse(repository: String, checksum: String, path: String, isPrivate: Bool) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: ["repository": repository, "private": isPrivate,
            "url": "https://huggingface.co/\(repository)", "commit_url": "https://huggingface.co/\(repository)/commit/verified",
            "source_checkpoint_sha256": checksum, "package_path": path]), as: UTF8.self)
    }
}
