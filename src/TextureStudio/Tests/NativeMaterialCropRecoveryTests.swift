import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

final class NativeMaterialCropRecoveryTests: XCTestCase {
    func testSmallTrainingGridsRecoverTheirMissingTargetAtOneKWithDefaultTierLimit() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let input = try fixture.write(role: "input", pixels: 1024)
        for size in [256, 512] {
            let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: size, target: "height")
            XCTAssertEqual(Self.identities(requests), ["1k|height"])
            XCTAssertEqual(requests.first?.width, 1024)
            XCTAssertEqual(requests.first?.height, 1024)
        }
    }

    func testTwoKTrainingFetchLimitDefaultsToOnlyTwoKAndExplicitlyExpands() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let input = try fixture.write(role: "input", pixels: 1024)
        let original = try Data(contentsOf: input)
        for (count, expected) in [(1, Set(["2k"])), (2, Set(["2k", "4k"])), (3, Set(["2k", "4k", "8k"]))] {
            let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 2048, target: "height", tierCount: count)
            XCTAssertEqual(Set(requests.map(\.resolution)), expected)
            XCTAssertEqual(requests.count, expected.count * 2)
        }
        let defaults = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 2048, target: "height")
        XCTAssertEqual(Set(defaults.map(\.resolution)), ["2k"])
        for count in [0, 4, -1] {
            XCTAssertTrue(NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 2048, target: "height", tierCount: count).isEmpty)
        }
        XCTAssertEqual(try Data(contentsOf: input), original)
    }

    func testCroppingRequestsMissingOneTwoAndFourKTiersForRelevantTargetOnly() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let input = try fixture.write(role: "input", pixels: 1024)
        let height = try fixture.write(role: "height", pixels: 1024)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input, height], size: 1024, target: "roughness", tierCount: 3)
        XCTAssertEqual(Self.identities(requests), ["1k|roughness", "2k|input", "2k|roughness", "4k|input", "4k|roughness"])
        XCTAssertFalse(requests.contains { $0.target == "height" || $0.target == "normal" })
        for request in requests {
            let pixels = try XCTUnwrap(Int(request.resolution.dropLast())) * 1024
            XCTAssertEqual(request.width, pixels); XCTAssertEqual(request.height, pixels)
            XCTAssertEqual(request.asset, Fixture.asset)
            XCTAssertEqual(request.destination.deletingLastPathComponent().path, fixture.root.resolvingSymlinksInPath().standardizedFileURL.path)
            XCTAssertEqual(request.destination.lastPathComponent,
                "\(Fixture.asset)_\(request.target == "input" ? "diff" : "rough")_\(request.resolution).png")
        }
    }

    func testHeightWithoutEightKDiffuseDoesNotRequestHighTierInput() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let height = try fixture.write(role: "height", pixels: 8192)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [height], size: 1024, target: "height", tierCount: 3)
        XCTAssertEqual(Self.identities(requests), ["1k|height", "1k|input", "2k|height", "2k|input", "4k|height", "4k|input"])
        XCTAssertFalse(requests.contains { $0.resolution == "8k" },
                       "A high-resolution target alone must not cause a new high-resolution diffuse download")
    }

    func testExistingEightKDiffuseDoesNotExceedTheSelectedDownloadTierLimit() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let input = try fixture.write(role: "input", pixels: 8192)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 1024, target: "height", tierCount: 3)
        XCTAssertEqual(Self.identities(requests), ["1k|height", "1k|input", "2k|height", "2k|input", "4k|height", "4k|input"])
        XCTAssertFalse(requests.contains { $0.resolution == "8k" })
        XCTAssertTrue(FileManager.default.fileExists(atPath: input.path))
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

    func testCompleteRectangularBarkPairsStartAtTheirAvailableNativeTierWithoutAcquiringOneK() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        var nativeFiles: [URL] = []
        for pixels in [2048, 4096] {
            nativeFiles.append(try fixture.write(role: "input", pixels: pixels, height: pixels * 2))
            nativeFiles.append(try fixture.write(role: "height", pixels: pixels, height: pixels * 2))
        }
        for asset in ["japanese_sycamore", "chinese_hackberry_bark", "palm_tree_bark"] {
            var files: [URL] = []
            for original in nativeFiles {
                let file = fixture.root.appendingPathComponent(original.lastPathComponent.replacingOccurrences(of: Fixture.asset, with: asset))
                try FileManager.default.copyItem(at: original, to: file); files.append(file)
            }
            XCTAssertTrue(NativeMaterialDatasetService.croppingRecoveryRequests(files: files, size: 1024, target: "height").isEmpty,
                "A paired native rectangle supplies this grid without fabricating or downloading an absent 1K tier")
        }
    }

    func testRectangularTierRecoveryStillRepairsMissingRegisteredTargetsAndRequiresMatchingAdequatePixels() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let input2K = try fixture.write(role: "input", pixels: 2048, height: 4096)
        let input4K = try fixture.write(role: "input", pixels: 4096, height: 8192)
        let height4K = try fixture.write(role: "height", pixels: 4096, height: 8192)
        let repairs = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input2K, input4K, height4K], size: 1024, target: "height", tierCount: 3)
        XCTAssertEqual(Self.identities(repairs), ["2k|height"])
        XCTAssertEqual(repairs.first?.width, 2048); XCTAssertEqual(repairs.first?.height, 4096)
        let square4K = try fixture.write(role: "height", pixels: 4096)
        let mismatched = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input2K, input4K, square4K], size: 1024, target: "height")
        XCTAssertTrue(mismatched.contains { $0.resolution == "1k" }, "Mismatched maps do not prove an adequate rectangular pair")
        let thinInput = try fixture.write(role: "input", pixels: 2048, height: 512)
        let thinHeight = try fixture.write(role: "height", pixels: 2048, height: 512)
        let undersized = NativeMaterialDatasetService.croppingRecoveryRequests(files: [thinInput, thinHeight], size: 1024, target: "height")
        XCTAssertTrue(undersized.contains { $0.resolution == "1k" }, "A matching nominal edge cannot authorize an undersized short edge")
    }

    func testBalancedTierRequestsRecoverThroughInjectedTransportWithExactPublishedBytes() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let input = try fixture.write(role: "input", pixels: 1024)
        let originalBytes = try Data(contentsOf: input)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 1024, target: "height", tierCount: 3)
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
        func write(role: String, pixels: Int, height: Int? = nil) throws -> URL {
            let suffix = role == "input" ? "diff" : "disp"
            let url = root.appendingPathComponent("\(Self.asset)_\(suffix)_\(pixels / 1024)k.png")
            try Self.png(role: role, pixels: pixels, height: height).write(to: url)
            return url
        }
        static func png(role: String, pixels: Int, height: Int? = nil) throws -> Data {
            let bits = role == "height" ? 16 : 8, channels = role == "input" ? 3 : 1
            let rows = height ?? pixels
            return try NativePNG(header: .init(width: pixels, height: rows, bits: bits, channels: channels,
                color: channels == 3 ? 2 : 0, interlace: 0),
                pixels: Data(repeating: 97, count: pixels * rows * channels * (bits / 8)), colorChunks: []).encoded()
        }
    }
}
