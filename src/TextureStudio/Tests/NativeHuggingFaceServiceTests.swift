import Foundation
import XCTest
@testable import TextureStudio

final class NativeHuggingFaceServiceTests: XCTestCase {
    func testAccountUsesDirectAuthenticatedAPIWithoutWorker() async throws {
        let service = NativeHuggingFaceService(token: "test-credential") { request in
            XCTAssertEqual(request.url?.absoluteString, "https://huggingface.co/api/whoami-v2")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-credential")
            return Data("{\"name\":\"artist\"}".utf8)
        }
        let result = try WorkbenchResult.decode(HuggingFaceAccountResponse.self, output: await service.accountJSON())
        XCTAssertTrue(result.authenticated)
        XCTAssertEqual(result.username, "artist")
    }

    func testMissingTokenNeverStartsNetworkDiscovery() async throws {
        let service = NativeHuggingFaceService(token: nil) { _ in
            XCTFail("No request is needed without a saved login")
            return Data()
        }
        let account = try await service.account()
        XCTAssertFalse(account.authenticated)
    }

    func testDiscoveryMergesSavedAndRemoteModelsAndRetainsCatalogOnFailure() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let catalog = root.appendingPathComponent("catalog.json")
        let original = Data("{\"models\":[{\"repository\":\"artist/saved\",\"target\":\"height\",\"revision\":\"old\"}]}".utf8)
        try original.write(to: catalog)
        let service = NativeHuggingFaceService(token: "test", catalogURL: catalog) { request in
            if request.url?.path == "/api/whoami-v2" { return Data("{\"name\":\"artist\"}".utf8) }
            let components = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
            XCTAssertEqual(components.queryItems?.first { $0.name == "author" }?.value, "artist")
            return Data("[{\"id\":\"artist/saved\",\"sha\":\"new\",\"siblings\":[{\"rfilename\":\"adapter.safetensors\"}]},{\"id\":\"artist/full\",\"sha\":\"full-sha\",\"siblings\":[{\"rfilename\":\"model.safetensors\"}]},{\"id\":\"artist/unrelated\",\"siblings\":[]}]".utf8)
        }
        let result = try WorkbenchResult.decode(WorkbenchHubModels.self, output: await service.modelsJSON())
        XCTAssertEqual(result.models.map(\.repository), ["artist/full", "artist/saved"])
        XCTAssertEqual(result.models.first { $0.repository == "artist/saved" }?.revision, "new")
        XCTAssertEqual(result.models.first { $0.repository == "artist/saved" }?.target, "height")
        XCTAssertEqual(try Data(contentsOf: catalog), original)
        let unavailable = NativeHuggingFaceService(token: "test", catalogURL: catalog) { request in
            if request.url?.path == "/api/whoami-v2" { return Data("{\"name\":\"artist\"}".utf8) }
            throw URLError(.notConnectedToInternet)
        }
        let retained = try WorkbenchResult.decode(WorkbenchHubModels.self, output: await unavailable.modelsJSON())
        XCTAssertEqual(retained.models.map(\.repository), ["artist/saved"])
    }

    func testCredentialDiscoveryUsesEnvironmentOrNativeKeychain() {
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: [:], keychain: { " test-token\n" }), "test-token")
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: ["HF_TOKEN": " override\n"], keychain: { "saved" }), "override")
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: ["HF_TOKEN": " \n"], keychain: { "saved" }), "saved")
        XCTAssertNil(NativeHuggingFaceService.savedToken(environment: [:], keychain: { nil }))
    }

    func testExplicitTransportCancellationIsNeverReportedAsMissingCredentials() async throws {
        let service = NativeHuggingFaceService(token: "test") { _ in throw CancellationError() }
        do {
            _ = try await service.account()
            XCTFail("Cancellation must propagate to the caller")
        } catch is CancellationError { }
    }

    func testCancelledAccountCannotPublishSuccessfulIgnoredCancellationResponse() async throws {
        let gate = ResponseGate()
        let service = NativeHuggingFaceService(token: "test") { _ in
            await gate.wait()
            return Data("{\"name\":\"artist\"}".utf8)
        }
        let request = Task { try await service.account() }
        await gate.waitUntilRequested()
        request.cancel()
        await gate.release()
        do {
            _ = try await request.value
            XCTFail("A cancelled account request must not publish success")
        } catch is CancellationError { }
    }

    func testCancelledInventoryCannotPublishIgnoredCancellationResponse() async throws {
        let gate = ResponseGate()
        let catalog = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        let service = NativeHuggingFaceService(token: "test", catalogURL: catalog) { request in
            if request.url?.path == "/api/whoami-v2" { return Data("{\"name\":\"artist\"}".utf8) }
            await gate.wait()
            return Data("[]".utf8)
        }
        let request = Task { try await service.modelsJSON() }
        await gate.waitUntilRequested()
        request.cancel()
        await gate.release()
        do {
            _ = try await request.value
            XCTFail("A cancelled model inventory must not publish success")
        } catch is CancellationError { }
    }

    private actor ResponseGate {
        var continuation: CheckedContinuation<Void, Never>?
        func wait() async { await withCheckedContinuation { continuation = $0 } }
        func waitUntilRequested() async {
            while continuation == nil { await Task.yield() }
        }
        func release() { continuation?.resume(); continuation = nil }
    }
}
