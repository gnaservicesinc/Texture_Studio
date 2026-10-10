import CryptoKit
import Darwin
import Foundation

/// Repairs missing provider maps before the ordinary source scan. Only the
/// provider's verified PNG or an exact pixel crop is added; existing files stay.
actor NativeMaterialSourceRecovery {
    struct Request: Sendable {
        let asset: String
        let resolution: String
        let target: String
        let destination: URL
        let width: Int
        let height: Int
        let expectedSHA256: String?
        let publishedRole: String?

        init(asset: String, resolution: String, target: String, destination: URL,
             width: Int, height: Int, expectedSHA256: String? = nil, publishedRole: String? = nil) {
            self.asset = asset; self.resolution = resolution; self.target = target
            self.destination = destination; self.width = width; self.height = height
            self.expectedSHA256 = expectedSHA256
            self.publishedRole = publishedRole
        }
    }

    struct Provenance: Sendable {
        let destination: URL
        let downloadURL: URL
        let apiURL: URL
        let publishedMD5: String
        let publishedBytes: Int
        let sha256: String
        let sourceWidth: Int
        let sourceHeight: Int
        let cropRectangle: [Int]?
    }

    enum Result: Sendable {
        case recovered(Provenance)
        case alreadyPresent
        case notPublished
        case failed(reason: String)
    }

    struct HTTPData: Sendable {
        let data: Data
        let statusCode: Int
        let url: URL
    }

    struct HTTPResponse: Sendable {
        let statusCode: Int
        let url: URL
    }

    struct Transport: Sendable {
        let fetchMetadata: @Sendable (URL) async throws -> HTTPData
        let download: @Sendable (URL, URL) async throws -> HTTPResponse

        static let live = Transport(fetchMetadata: { url in
            let (data, response) = try await networkSession.data(for: networkRequest(url))
            guard let http = response as? HTTPURLResponse, let finalURL = response.url else {
                throw RecoveryFailure("Poly Haven did not return an HTTP response.")
            }
            return HTTPData(data: data, statusCode: http.statusCode, url: finalURL)
        }, download: { url, destination in
            let (temporary, response) = try await networkSession.download(for: networkRequest(url))
            defer { try? FileManager.default.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, let finalURL = response.url else {
                throw RecoveryFailure("Poly Haven did not return an HTTP response.")
            }
            try Task.checkCancellation()
            try FileManager.default.moveItem(at: temporary, to: destination)
            return HTTPResponse(statusCode: http.statusCode, url: finalURL)
        })
    }

    private struct PublishedFile: Sendable {
        let url: URL
        let md5: String
        let bytes: Int
    }
    private typealias Catalog = [String: [String: PublishedFile]]
    private let transport: Transport
    private var catalogs: [String: Catalog] = [:]
    private var unpublishedAssets: Set<String> = []
    private static let roles = ["input": "Diffuse", "height": "Displacement", "roughness": "Rough", "normal": "nor_gl", "normal_dx": "nor_dx"]

    init(transport: Transport = .live) { self.transport = transport }

    /// Unavailable assets and network failures leave import usable. Cancellation
    /// still stops the operation and removes any incomplete temporary download.
    func recover(_ request: Request,
                 onProgress: @escaping @Sendable (String) -> Void = { _ in }) async throws -> Result {
        try Task.checkCancellation()
        guard Self.validAsset(request.asset),
              request.resolution.range(of: #"^(?:1|2|4|8|16)k$"#, options: .regularExpression) != nil,
              let defaultRole = Self.roles[request.target],
              request.publishedRole.map({ request.target == "input" && Self.compatibleInputRole($0) }) ?? true,
              request.width > 0, request.height > 0, request.width <= 65536, request.height <= 65536,
              request.destination.isFileURL, request.destination.pathExtension.lowercased() == "png",
              request.expectedSHA256.map({ Self.hexadecimal($0, count: 64) }) ?? true else {
            return .failed(reason: "The source does not identify a supported Poly Haven map.")
        }
        let role = request.publishedRole ?? defaultRole
        let destination = request.destination.standardizedFileURL
        if Self.pathExists(destination) { return .alreadyPresent }
        let folder = destination.deletingLastPathComponent()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return .failed(reason: "The original source folder is unavailable: \(folder.path)")
        }
        let staged = folder.appendingPathComponent(".material-map-download-\(UUID().uuidString).tmp")
        defer { try? FileManager.default.removeItem(at: staged) }
        do {
            let apiURL = URL(string: "https://api.polyhaven.com/files/" + request.asset)!
            guard let catalog = try await catalog(for: request.asset, apiURL: apiURL),
                  let published = catalog[role]?[request.resolution] else { return .notPublished }
            let expectedFolder = "/file/ph-assets/Textures/png/\(request.resolution)/\(request.asset)/"
            guard Self.validProviderURL(published.url, host: "dl.polyhaven.org"),
                  published.url.deletingLastPathComponent().path + "/" == expectedFolder,
                  published.url.pathExtension.lowercased() == "png",
                  published.bytes > 0, published.bytes <= 2 * 1024 * 1024 * 1024,
                  Self.hexadecimal(published.md5, count: 32) else {
                return .failed(reason: "Poly Haven's published map information could not be verified.")
            }
            try Task.checkCancellation()
            if Self.pathExists(destination) { return .alreadyPresent }
            onProgress("Obtaining \(destination.lastPathComponent) from Poly Haven…")
            let response = try await transport.download(published.url, staged)
            try Task.checkCancellation()
            guard response.statusCode == 200,
                  Self.validProviderURL(response.url, host: "dl.polyhaven.org"), response.url.path == published.url.path else {
                throw RecoveryFailure("The published map download is unavailable (HTTP \(response.statusCode)).")
            }
            var state = stat()
            guard lstat(staged.path, &state) == 0, state.st_mode & S_IFMT == S_IFREG,
                  state.st_size == published.bytes else {
                throw RecoveryFailure("The download did not match Poly Haven's published file size.")
            }
            let verified = try Self.checksums(staged)
            guard verified.md5 == published.md5.lowercased() else {
                throw RecoveryFailure("The download did not match Poly Haven's published checksum.")
            }
            let bytes = try Data(contentsOf: staged, options: .mappedIfSafe)
            let metadata = try NativePNG.sourceMetadata(bytes)
            guard request.target != "height" || metadata["sample_bits"] as? Int == 16 else {
                throw RecoveryFailure("The published displacement PNG does not preserve 16-bit source precision.")
            }
            guard !["input", "normal", "normal_dx"].contains(request.target) || [3, 4].contains(metadata["channels"] as? Int ?? 0) else {
                throw RecoveryFailure("The published color or normal map does not contain native RGB or RGBA channels.")
            }
            let width = metadata["width"] as? Int ?? 0, height = metadata["height"] as? Int ?? 0
            var cropRectangle: [Int]?, savedSHA256 = verified.sha256
            if width != request.width || height != request.height {
                let horizontalExcess = width - request.width, verticalExcess = height - request.height
                guard horizontalExcess >= 0, verticalExcess >= 0,
                      horizontalExcess <= max(2, Int(ceil(Double(request.width) * 0.02))),
                      verticalExcess <= max(2, Int(ceil(Double(request.height) * 0.02))) else {
                    throw RecoveryFailure("The published map has different pixel dimensions from this source set.")
                }
                let rectangle = [horizontalExcess / 2, verticalExcess / 2, request.width, request.height]
                let cropped = try NativePNG.crop(staged, rectangle: rectangle)
                try Task.checkCancellation()
                try cropped.encoded().write(to: staged, options: .atomic)
                savedSHA256 = try Self.checksums(staged).sha256
                cropRectangle = rectangle
            } else {
                // A tiny retained crop still streams every compressed row and
                // rejects invalid PNG pixels without allocating the full image.
                _ = try NativePNG.crop(staged, rectangle: [0, 0, 1, 1])
            }
            if let expected = request.expectedSHA256, savedSHA256 != expected.lowercased() {
                throw RecoveryFailure("The recovered map differs from this dataset's recorded original.")
            }
            try Task.checkCancellation()
            // Both files are on the same filesystem. link creates the complete
            // destination atomically and fails if another scan already added it.
            guard link(staged.path, destination.path) == 0 else {
                if errno == EEXIST { return .alreadyPresent }
                throw RecoveryFailure("The recovered map could not be saved to \(folder.path).")
            }
            if cropRectangle != nil {
                onProgress("Cropped \(destination.lastPathComponent) from \(width) × \(height) to \(request.width) × \(request.height), preserving original pixels.")
            }
            onProgress("Saved \(destination.lastPathComponent) to the source folder.")
            return .recovered(Provenance(destination: destination, downloadURL: published.url,
                apiURL: apiURL, publishedMD5: published.md5.lowercased(),
                publishedBytes: published.bytes, sha256: savedSHA256,
                sourceWidth: width, sourceHeight: height, cropRectangle: cropRectangle))
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            try Task.checkCancellation()
            return .failed(reason: error.localizedDescription)
        }
    }

    private func catalog(for asset: String, apiURL: URL) async throws -> Catalog? {
        if let cached = catalogs[asset] { return cached }
        if unpublishedAssets.contains(asset) { return nil }
        let response = try await transport.fetchMetadata(apiURL)
        try Task.checkCancellation()
        guard Self.validProviderURL(response.url, host: "api.polyhaven.com"), response.url.path == apiURL.path else {
            throw RecoveryFailure("Poly Haven's file listing did not come from the expected provider.")
        }
        if response.statusCode == 404 { unpublishedAssets.insert(asset); return nil }
        guard response.statusCode == 200, response.data.count <= 2 * 1024 * 1024,
              let object = try JSONSerialization.jsonObject(with: response.data) as? [String: Any] else {
            throw RecoveryFailure("Poly Haven's file listing is unavailable (HTTP \(response.statusCode)).")
        }
        var catalog: Catalog = [:]
        for role in object.keys where Self.roles.values.contains(role) || Self.compatibleInputRole(role) {
            guard let resolutions = object[role] as? [String: Any] else { continue }
            for (resolution, formats) in resolutions {
                guard let png = (formats as? [String: Any])?["png"] as? [String: Any],
                      let rawURL = png["url"] as? String, let url = URL(string: rawURL),
                      let md5 = png["md5"] as? String, let bytes = png["size"] as? Int else { continue }
                catalog[role, default: [:]][resolution] = PublishedFile(url: url, md5: md5, bytes: bytes)
            }
        }
        catalogs[asset] = catalog
        return catalog
    }

    private static func validAsset(_ asset: String) -> Bool {
        !asset.isEmpty && asset.count <= 200 && asset.range(of: #"^[a-z0-9]+(?:_[a-z0-9]+)*$"#, options: .regularExpression) != nil
    }
    private static func compatibleInputRole(_ role: String) -> Bool {
        role.range(of: #"^(?:col|diffuse|diff|color|albedo)(?:_?\d+)?$"#, options: [.regularExpression, .caseInsensitive]) != nil
    }
    private static func validProviderURL(_ url: URL, host: String) -> Bool {
        url.scheme == "https" && url.host == host && url.port == nil && url.user == nil && url.password == nil && url.query == nil && url.fragment == nil
    }
    private static func hexadecimal(_ value: String, count: Int) -> Bool {
        value.count == count && value.range(of: "^[a-fA-F0-9]+$", options: .regularExpression) != nil
    }
    private static func pathExists(_ url: URL) -> Bool {
        var state = stat()
        return lstat(url.path, &state) == 0
    }
    private static func checksums(_ url: URL) throws -> (md5: String, sha256: String) {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var md5 = Insecure.MD5(), sha256 = SHA256()
        while let bytes = try file.read(upToCount: 1024 * 1024), !bytes.isEmpty {
            try Task.checkCancellation()
            md5.update(data: bytes); sha256.update(data: bytes)
        }
        return (md5.finalize().map { String(format: "%02x", $0) }.joined(),
                sha256.finalize().map { String(format: "%02x", $0) }.joined())
    }

    private struct RecoveryFailure: LocalizedError {
        let errorDescription: String?
        init(_ description: String) { errorDescription = description }
    }
    private final class ProviderRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
            let host = task.originalRequest?.url?.host
            guard let url = request.url, let host, ["api.polyhaven.com", "dl.polyhaven.org"].contains(host),
                  NativeMaterialSourceRecovery.validProviderURL(url, host: host) else {
                completionHandler(nil); return
            }
            completionHandler(request)
        }
    }
    private static let networkSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 180
        configuration.httpMaximumConnectionsPerHost = 2
        return URLSession(configuration: configuration, delegate: ProviderRedirects(), delegateQueue: nil)
    }()
    private static func networkRequest(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        request.setValue("IPDE/1.0 (material source recovery)", forHTTPHeaderField: "User-Agent")
        return request
    }
}
