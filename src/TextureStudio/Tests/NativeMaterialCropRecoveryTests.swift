import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialCropRecoveryTests: XCTestCase {
    func testCroppingRequestsMissingOneTwoAndFourKTiersForRelevantTargetOnly() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let input = try fixture.write(role: "input", pixels: 1024)
        let height = try fixture.write(role: "height", pixels: 1024)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input, height], size: 1024, target: "roughness")
        XCTAssertEqual(Self.identities(requests), ["1k|roughness", "2k|input", "2k|roughness", "4k|input", "4k|roughness"])
        XCTAssertFalse(requests.contains { $0.target == "height" || $0.target == "normal" })
        for request in requests {
            let pixels = try XCTUnwrap(Int(request.resolution.dropLast())) * 1024
            XCTAssertEqual(request.width, pixels); XCTAssertEqual(request.height, pixels)
            XCTAssertEqual(request.asset, Fixture.asset)
            XCTAssertEqual(request.destination.deletingLastPathComponent().path, fixture.root.path)
            XCTAssertEqual(request.destination.lastPathComponent,
                "\(Fixture.asset)_\(request.target == "input" ? "diff" : "rough")_\(request.resolution).png")
        }
    }

    func testHeightWithoutEightKDiffuseDoesNotRequestHighTierInput() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let height = try fixture.write(role: "height", pixels: 8192)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [height], size: 1024, target: "height")
        XCTAssertEqual(Self.identities(requests), ["1k|height", "1k|input", "2k|height", "2k|input", "4k|height", "4k|input"])
        XCTAssertFalse(requests.contains { $0.resolution == "8k" },
                       "A high-resolution target alone must not cause a new high-resolution diffuse download")
    }

    func testExistingEightKDiffuseRequestsItsMissingTargetWithoutRefetchingInput() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let input = try fixture.write(role: "input", pixels: 8192)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 1024, target: "height")
        XCTAssertEqual(Self.identities(requests), ["1k|height", "1k|input", "2k|height", "2k|input", "4k|height", "4k|input", "8k|height"])
        let highTier = try XCTUnwrap(requests.first { $0.resolution == "8k" })
        XCTAssertEqual(highTier.width, 8192); XCTAssertEqual(highTier.height, 8192)
        XCTAssertFalse(requests.contains { $0.resolution == "8k" && $0.target == "input" })
    }

    func testCompleteTierPairsAreNotRefetchedAndSmallProviderLabelsDoNotTriggerRecovery() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        var files: [URL] = []
        for pixels in [1024, 2048, 4096] {
            files.append(try fixture.write(role: "input", pixels: pixels))
            files.append(try fixture.write(role: "height", pixels: pixels))
        }
        XCTAssertTrue(NativeMaterialDatasetService.croppingRecoveryRequests(files: files + files,
            size: 1024, target: "height").isEmpty)
        let small = fixture.root.appendingPathComponent("small_material_diff_1k.png")
        try Fixture.png(role: "input", pixels: 16).write(to: small)
        XCTAssertTrue(NativeMaterialDatasetService.croppingRecoveryRequests(files: [small],
            size: 1024, target: "height").isEmpty)
        XCTAssertTrue(NativeMaterialDatasetService.croppingRecoveryRequests(files: files,
            size: 1024, target: "unsupported").isEmpty)
    }

    func testBalancedTierRequestsRecoverThroughInjectedTransportWithExactPublishedBytes() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let input = try fixture.write(role: "input", pixels: 1024)
        let originalBytes = try Data(contentsOf: input)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 1024, target: "height")
        XCTAssertEqual(requests.count, 5)
        var catalog: [String: [String: [String: [String: Any]]]] = [:]
        var payloads: [URL: Data] = [:]
        for request in requests {
            let bytes = try Fixture.png(role: request.target, pixels: request.width)
            let publishedURL = URL(string: "https://dl.polyhaven.org/file/ph-assets/Textures/png/\(request.resolution)/\(request.asset)/\(request.destination.lastPathComponent)")!
            payloads[publishedURL] = bytes
            let role = request.target == "input" ? "Diffuse" : "Displacement"
            catalog[role, default: [:]][request.resolution] = ["png": ["url": publishedURL.absoluteString,
                "size": bytes.count, "md5": Insecure.MD5.hash(data: bytes).map { String(format: "%02x", $0) }.joined()]]
        }
        let metadata = try JSONSerialization.data(withJSONObject: catalog)
        let published = payloads
        let recovery = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            .init(data: metadata, statusCode: 200, url: url)
        }, download: { url, destination in
            let bytes = try XCTUnwrap(published[url], "Every download must belong to the injected catalog")
            try bytes.write(to: destination)
            return .init(statusCode: 200, url: url)
        }))
        for request in requests {
            let result = try await recovery.recover(request)
            guard case .recovered(let provenance) = result else { return XCTFail("Expected published recovery for \(request.destination.lastPathComponent)") }
            let stored = try Data(contentsOf: request.destination)
            XCTAssertEqual(stored, published[provenance.downloadURL])
            let header = try NativePNG.inspect(request.destination)
            XCTAssertEqual(header.width, request.width); XCTAssertEqual(header.height, request.height)
            XCTAssertEqual(header.bits, request.target == "height" ? 16 : 8)
        }
        XCTAssertEqual(try Data(contentsOf: input), originalBytes)
        let files = try FileManager.default.contentsOfDirectory(at: fixture.root, includingPropertiesForKeys: nil)
        XCTAssertTrue(NativeMaterialDatasetService.croppingRecoveryRequests(files: files, size: 1024, target: "height").isEmpty)
    }

    private static func identities(_ requests: [NativeMaterialSourceRecovery.Request]) -> [String] {
        requests.map { $0.resolution + "|" + $0.target }.sorted()
    }

    private struct Fixture {
        static let asset = "ribbed_corduroy"
        let root: URL
        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("material-crop-recovery-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
        func write(role: String, pixels: Int) throws -> URL {
            let suffix = role == "input" ? "diff" : "disp"
            let url = root.appendingPathComponent("\(Self.asset)_\(suffix)_\(pixels / 1024)k.png")
            try Self.png(role: role, pixels: pixels).write(to: url)
            return url
        }
        static func png(role: String, pixels: Int) throws -> Data {
            let bits = role == "height" ? 16 : 8, channels = role == "input" ? 3 : 1
            return try NativePNG(header: .init(width: pixels, height: pixels, bits: bits, channels: channels,
                color: channels == 3 ? 2 : 0, interlace: 0),
                pixels: Data(repeating: 97, count: pixels * pixels * channels * (bits / 8)), colorChunks: []).encoded()
        }
    }
}
