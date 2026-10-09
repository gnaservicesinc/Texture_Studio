import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialTransferTests: XCTestCase {
    func testStreamingSizeGateAndRedirectProtectionWithoutStartingNetworkTask() throws {
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }
        var original = URLRequest(url: URL(string: "https://huggingface.co/fixture/object")!)
        original.setValue("fixture-token", forHTTPHeaderField: "Authorization")
        original.setValue("signed-secret", forHTTPHeaderField: "X-Signed-Request")
        original.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        let task = session.downloadTask(with: original) // Never resumed.
        let sizeGate = MaterialDownloadDelegate(maxBytes: 32)
        sizeGate.urlSession(session, downloadTask: task, didWriteData: 1, totalBytesWritten: 1, totalBytesExpectedToWrite: 33)
        XCTAssertTrue(sizeGate.exceededSize)
        let response = HTTPURLResponse(url: original.url!, statusCode: 302, httpVersion: nil, headerFields: nil)!
        var insecure = original; insecure.url = URL(string: "http://objects.fixture.test/object")!
        sizeGate.urlSession(session, task: task, willPerformHTTPRedirection: response, newRequest: insecure) { next in
            XCTAssertNil(next)
        }
        var secure = original; secure.url = URL(string: "https://objects.fixture.test/object")!
        sizeGate.urlSession(session, task: task, willPerformHTTPRedirection: response, newRequest: secure) { next in
            XCTAssertEqual(next?.url, secure.url)
            XCTAssertNil(next?.value(forHTTPHeaderField: "Authorization"))
            XCTAssertNil(next?.value(forHTTPHeaderField: "X-Signed-Request"))
            XCTAssertEqual(next?.value(forHTTPHeaderField: "Content-Type"), "application/octet-stream")
        }
    }

    func testHTTPSBasicLFSUploadPublishesOnlyVerifiedFilesAndRegistersExactCommit() async throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        let package = try fixture.package()
        let mock = MaterialHubMock(files: try fixture.files(package))
        let catalog = fixture.root.appendingPathComponent("catalog.json")
        let transfer = NativeMaterialTransfer(token: "fixture-token", send: { try await mock.send($0, file: $1) }, catalogURL: catalog)
        let result = try NativeMaterialTransfer.object(Data(try await transfer.upload(package: package, repository: "fixture/material", isPublic: true).utf8))
        XCTAssertEqual(result["revision"] as? String, MaterialHubMock.revision)
        XCTAssertEqual(result["source_photos_uploaded"] as? Bool, false)
        let stats = await mock.statistics()
        XCTAssertEqual(stats.uploaded, try Data(contentsOf: package.appendingPathComponent("adapter.safetensors")))
        XCTAssertEqual(stats.commits, 1); XCTAssertEqual(stats.verifications, 1)
        let catalogRecord = try XCTUnwrap((NativeMaterialTransfer.object(catalog)["models"] as? [[String: Any]])?.first)
        XCTAssertEqual(catalogRecord["revision"] as? String, MaterialHubMock.revision)
        XCTAssertEqual(catalogRecord["sha256"] as? String, try NativeMaterialTransfer.hash(package.appendingPathComponent("adapter.safetensors")))
    }

    func testMultipartLFSStreamsOriginalBytesAndRequiresAllReceipts() async throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        let package = try fixture.package()
        let mock = MaterialHubMock(files: try fixture.files(package), behavior: .multipart)
        let transfer = NativeMaterialTransfer(token: "fixture-token", send: { try await mock.send($0, file: $1) }, catalogURL: fixture.root.appendingPathComponent("catalog.json"))
        _ = try await transfer.upload(package: package, repository: "fixture/material", isPublic: true)
        let stats = await mock.statistics()
        XCTAssertEqual(stats.uploaded, try Data(contentsOf: package.appendingPathComponent("adapter.safetensors")))
        XCTAssertGreaterThan(stats.parts, 1); XCTAssertEqual(stats.commits, 1)
    }

    func testRejectsInsecureLFSURLDuplicateInventoryAndVisibilityMismatchBeforeCommit() async throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        let package = try fixture.package()
        for behavior in [MaterialHubMock.Behavior.insecure, .duplicateInventory, .visibilityMismatch, .missingReceipt, .wrongLFSSize, .invalidChunkSize, .mutatedPackage] {
            let mock = MaterialHubMock(files: try fixture.files(package), behavior: behavior, mutationRoot: package)
            let transfer = NativeMaterialTransfer(token: "fixture-token", send: { try await mock.send($0, file: $1) }, catalogURL: fixture.root.appendingPathComponent(UUID().uuidString))
            do { _ = try await transfer.upload(package: package, repository: "fixture/material", isPublic: true); XCTFail("Invalid Hub response was accepted") }
            catch { XCTAssertFalse(error is CancellationError) }
            let stats = await mock.statistics()
            XCTAssertEqual(stats.commits, 0)
        }
    }

    func testBoundedMockDownloadChecksEveryDigestBeforePublishingOrReusingPackage() async throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        let package = try fixture.package()
        let mock = MaterialHubMock(files: try fixture.files(package))
        let transfer = NativeMaterialTransfer(token: "fixture-token", fetch: { try await mock.fetch($0, to: $1, maxBytes: $2) }, catalogURL: fixture.root.appendingPathComponent("catalog.json"))
        let destination = fixture.root.appendingPathComponent("Downloaded")
        _ = try await transfer.download(repository: "fixture/material", revision: MaterialHubMock.revision, to: destination)
        _ = try NativeMaterialPackage.verify(destination)
        let before = await mock.statistics().downloads
        _ = try await transfer.download(repository: "fixture/material", revision: MaterialHubMock.revision, to: destination)
        let after = await mock.statistics().downloads
        XCTAssertEqual(before, after)
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("adapter.safetensors")), try Data(contentsOf: package.appendingPathComponent("adapter.safetensors")))
        let bad = MaterialHubMock(files: try fixture.files(package), behavior: .corruptDownload)
        let failed = fixture.root.appendingPathComponent("Failed")
        let failing = NativeMaterialTransfer(token: nil, fetch: { try await bad.fetch($0, to: $1, maxBytes: $2) }, catalogURL: fixture.root.appendingPathComponent("unused.json"))
        do { _ = try await failing.download(repository: "fixture/material", revision: MaterialHubMock.revision, to: failed); XCTFail("Corrupt download was accepted") }
        catch { XCTAssertFalse(FileManager.default.fileExists(atPath: failed.path)) }
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: fixture.root.path).contains { $0.hasPrefix(".material-download-") })
    }
}

/// A strict, fully local Hub/LFS peer. No request reaches URLSession and no
/// saved credentials or application-support catalog is read or written.
private actor MaterialHubMock {
    enum Behavior { case basic, multipart, insecure, duplicateInventory, visibilityMismatch, corruptDownload, missingReceipt, wrongLFSSize, invalidChunkSize, mutatedPackage }
    struct Statistics: Sendable { let uploaded: Data; let commits: Int; let verifications: Int; let parts: Int; let downloads: Int }
    static let revision = String(repeating: "c", count: 40)
    let files: [String: Data], behavior: Behavior, mutationRoot: URL?
    var uploaded = Data(), parts: [Int: Data] = [:], commits = 0, verifications = 0, downloads = 0
    init(files: [String: Data], behavior: Behavior = .basic, mutationRoot: URL? = nil) { self.files = files; self.behavior = behavior; self.mutationRoot = mutationRoot }
    func statistics() -> Statistics { .init(uploaded: uploaded, commits: commits, verifications: verifications, parts: parts.count, downloads: downloads) }
    func send(_ request: URLRequest, file: URL?) throws -> (Data, HTTPURLResponse) {
        let url = try require(request.url, "missing URL"), host = url.host ?? "", path = url.path
        guard url.scheme == "https" else { throw StudioError("Mock received an insecure request") }
        if host == "huggingface.co" {
            guard request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-token" else { throw StudioError("Missing fixture API authorization") }
        } else if request.value(forHTTPHeaderField: "Authorization") != nil { throw StudioError("Repository token leaked to object storage") }
        func response(_ object: [String: Any], headers: [String: String]? = nil) throws -> (Data, HTTPURLResponse) {
            (try JSONSerialization.data(withJSONObject: object), HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: headers)!)
        }
        if path == "/api/repos/create" { return try response([:]) }
        if path == "/api/models/fixture/material" { return try response(["private": behavior == .visibilityMismatch]) }
        if path.hasSuffix("/preupload/main") {
            let body = try NativeMaterialTransfer.object(try require(request.httpBody, "missing upload inventory"))
            let records = try require(body["files"] as? [[String: Any]], "invalid inventory")
            guard Set(records.compactMap { $0["path"] as? String }) == Set(files.keys) else { throw StudioError("Unexpected upload inventory") }
            var modes: [[String: Any]] = records.map { ["path": $0["path"]!, "uploadMode": $0["path"] as? String == "adapter.safetensors" ? "lfs" : "regular"] }
            if behavior == .duplicateInventory { modes.append(modes[0]) }
            return try response(["files": modes])
        }
        if path.hasSuffix("/objects/batch") {
            guard request.value(forHTTPHeaderField: "Content-Type") == "application/vnd.git-lfs+json" else { throw StudioError("Wrong LFS content type") }
            let body = try NativeMaterialTransfer.object(try require(request.httpBody, "missing LFS body"))
            let object = try require((body["objects"] as? [[String: Any]])?.first, "missing LFS object")
            guard object["size"] as? Int == files["adapter.safetensors"]?.count else { throw StudioError("Wrong LFS byte count") }
            var headers: [String: String] = ["X-Signed-Request": "fixture"]
            if behavior == .multipart || behavior == .missingReceipt {
                let size = files["adapter.safetensors"]!.count, chunk = 113
                headers = ["chunk_size": String(chunk)]
                for part in 1...(size / chunk + (size % chunk == 0 ? 0 : 1)) { headers[String(part)] = "https://objects.fixture.test/part/\(part)" }
            }
            if behavior == .invalidChunkSize { headers = ["chunk_size": "0"] }
            if behavior == .mutatedPackage, let mutationRoot {
                try Data("Replaced README during upload".utf8).write(to: mutationRoot.appendingPathComponent("README.md"), options: .atomic)
                var hashes = try NativeMaterialTransfer.object(mutationRoot.appendingPathComponent("SHA256SUMS.json"))
                hashes["README.md"] = try NativeMaterialTransfer.hash(mutationRoot.appendingPathComponent("README.md"))
                try NativeMaterialTransfer.writeJSON(hashes, to: mutationRoot.appendingPathComponent("SHA256SUMS.json"))
            }
            let href = behavior == .insecure ? "http://objects.fixture.test/object" : "https://objects.fixture.test/object"
            return try response(["objects": [["oid": object["oid"]!, "size": behavior == .wrongLFSSize ? -1 : object["size"]!, "actions": ["upload": ["href": href, "header": headers], "verify": ["href": "https://objects.fixture.test/verify"]]]]])
        }
        if host == "objects.fixture.test", request.httpMethod == "PUT" {
            let bytes = try Data(contentsOf: require(file, "missing upload file"))
            if path.hasPrefix("/part/"), let part = Int(url.lastPathComponent) { parts[part] = bytes }
            else { uploaded = bytes }
            return try response([:], headers: behavior == .missingReceipt ? [:] : ["ETag": "\"receipt-\(url.lastPathComponent)\""])
        }
        if path == "/object", request.httpMethod == "POST" {
            let body = try NativeMaterialTransfer.object(try require(request.httpBody, "missing multipart completion"))
            let receipts = try require(body["parts"] as? [[String: Any]], "missing multipart receipts")
            guard receipts.count == parts.count, receipts.enumerated().allSatisfy({ $0.element["partNumber"] as? Int == $0.offset + 1 && $0.element["etag"] as? String == "\"receipt-\($0.offset + 1)\"" }) else { throw StudioError("Wrong multipart receipts") }
            uploaded = parts.keys.sorted().reduce(into: Data()) { $0.append(parts[$1]!) }
            return try response([:])
        }
        if path == "/verify" {
            guard uploaded == files["adapter.safetensors"] else { throw StudioError("Uploaded weights differ from original") }
            verifications += 1; return try response([:])
        }
        if path.hasSuffix("/commit/main") {
            guard request.value(forHTTPHeaderField: "Content-Type") == "application/x-ndjson", uploaded == files["adapter.safetensors"] else { throw StudioError("Invalid commit payload") }
            let lines = String(decoding: try require(request.httpBody, "missing commit"), as: UTF8.self).split(separator: "\n")
            var paths = Set<String>()
            for line in lines {
                let record = try NativeMaterialTransfer.object(Data(line.utf8))
                guard record["key"] as? String != "header" else { continue }
                let value = try require(record["value"] as? [String: Any], "invalid commit record")
                let name = try require(value["path"] as? String, "missing path")
                guard files[name] != nil, paths.insert(name).inserted else { throw StudioError("Unexpected commit file") }
                if record["key"] as? String == "file" {
                    guard Data(base64Encoded: value["content"] as? String ?? "") == files[name] else { throw StudioError("Regular upload changed bytes") }
                } else {
                    guard value["size"] as? Int == files[name]?.count, NativeMaterialTransfer.isDigest(value["oid"] as? String ?? "") else { throw StudioError("Invalid LFS commit") }
                }
            }
            guard paths == Set(files.keys) else { throw StudioError("Commit omitted package files") }
            commits += 1; return try response(["commitOid": Self.revision])
        }
        throw StudioError("Unexpected fixture request: " + path)
    }
    func fetch(_ request: URLRequest, to destination: URL, maxBytes: Int64) throws {
        let url = try require(request.url, "missing download URL")
        guard url.scheme == "https", url.host == "huggingface.co", maxBytes == (url.path.hasSuffix(".safetensors") ? 4_294_967_296 : url.path.hasSuffix("SHA256SUMS.json") ? 1_048_576 : 4_194_304) else { throw StudioError("Unexpected download budget or destination") }
        let prefix = "/fixture/material/resolve/" + Self.revision + "/"
        guard url.path.hasPrefix(prefix) else { throw StudioError("Download did not pin the exact revision") }
        let name = String(url.path.dropFirst(prefix.count))
        var bytes = try require(files[name], "Unexpected downloaded file")
        if behavior == .corruptDownload, name == "adapter.safetensors" { bytes[bytes.count - 1] ^= 1 }
        guard bytes.count <= maxBytes else { throw StudioError("Fixture exceeded budget") }
        try bytes.write(to: destination); downloads += 1
    }
    private func require<T>(_ value: T?, _ reason: String) throws -> T {
        guard let value else { throw StudioError(reason) }; return value
    }
}
