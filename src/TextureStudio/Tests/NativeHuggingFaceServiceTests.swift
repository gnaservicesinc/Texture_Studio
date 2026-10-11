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

    func testExistingNativeHubLoginWhenRequested() async throws {
        guard let username = ProcessInfo.processInfo.environment["TEXTURE_STUDIO_VERIFY_HF_ACCOUNT"], !username.isEmpty else {
            throw XCTSkip("Opt in with the expected account name to verify an existing login without uploading.")
        }
        let service = NativeHuggingFaceService()
        XCTAssertTrue(service.token != nil, "The native service should discover an existing credential.")
        let account = try await service.account()
        XCTAssertTrue(account.authenticated, account.message)
        XCTAssertEqual(account.username, username)
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

    func testCredentialDiscoveryUsesEnvironmentKeychainOrCLIActiveToken() {
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: [:], keychain: { " test-token\n" }), "test-token")
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: ["HF_TOKEN": " override\n"], keychain: { "saved" }), "override")
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: ["HF_TOKEN": " \n"], keychain: { "saved" }), "saved")
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: ["HUGGING_FACE_HUB_TOKEN": " legacy\n"], keychain: { "saved" }), "legacy")
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: [:], keychain: { "saved" }, readTokenFile: { _ in "cli" }), "saved")
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: [:], keychain: { nil }, readTokenFile: { _ in " cli\n" }), "cli")
        XCTAssertNil(NativeHuggingFaceService.savedToken(environment: [:], keychain: { nil }, readTokenFile: { _ in nil }))
    }

    func testCLIActiveTokenPathHonorsHubEnvironmentAndDefaultHome() {
        let home = URL(fileURLWithPath: "/test/home")
        func path(_ environment: [String: String]) -> String {
            NativeHubCredentials.tokenFileURL(environment: environment, homeDirectory: home).path
        }
        XCTAssertEqual(path([:]), "/test/home/.cache/huggingface/token")
        XCTAssertEqual(path(["XDG_CACHE_HOME": "/shared/cache"]), "/shared/cache/huggingface/token")
        XCTAssertEqual(path(["HF_HOME": "/shared/hub", "XDG_CACHE_HOME": "/shared/cache"]), "/shared/hub/token")
        XCTAssertEqual(path(["HF_TOKEN_PATH": "/shared/active", "HF_HOME": "/shared/hub"]), "/shared/active")
        XCTAssertEqual(path(["HF_HOME": "~/hf"]), "/test/home/hf/token")
        XCTAssertEqual(path(["HF_TOKEN_PATH": "${HOME}/auth/$ACCOUNT-token", "HOME": home.path, "ACCOUNT": "artist"]), "/test/home/auth/artist-token")
        XCTAssertEqual(path(["HF_HOME": "", "HF_TOKEN_PATH": " \n"]), "/test/home/.cache/huggingface/token")
    }

    func testCLIFileLoginIsReadNativelyAndFollowsAccountSwitches() throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: home) }
        let tokenURL = NativeHubCredentials.tokenFileURL(environment: [:], homeDirectory: home)
        try FileManager.default.createDirectory(at: tokenURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        XCTAssertNil(NativeHuggingFaceService.savedToken(environment: [:], keychain: { nil }, homeDirectory: home))
        try Data("first-test-login\n".utf8).write(to: tokenURL)
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: [:], keychain: { nil }, homeDirectory: home), "first-test-login")
        try Data("second-test-login\n".utf8).write(to: tokenURL, options: .atomic)
        XCTAssertEqual(NativeHuggingFaceService.savedToken(environment: [:], keychain: { nil }, homeDirectory: home), "second-test-login")
        // An explicit token path does not silently choose a different account
        // from the default cache when it is missing.
        XCTAssertNil(NativeHuggingFaceService.savedToken(environment: ["HF_TOKEN_PATH": home.appendingPathComponent("missing").path], keychain: { nil }, homeDirectory: home))
        try Data(" \n".utf8).write(to: tokenURL)
        XCTAssertNil(NativeHuggingFaceService.savedToken(environment: [:], keychain: { nil }, homeDirectory: home))
        try Data(repeating: 65, count: 16_385).write(to: tokenURL)
        XCTAssertNil(NativeHubCredentials.readTokenFile(tokenURL))
        try Data([0xff, 0xfe]).write(to: tokenURL)
        XCTAssertNil(NativeHubCredentials.readTokenFile(tokenURL))
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
