import CryptoKit
import Darwin
import Foundation
import zlib

/// Repairs missing provider maps before the ordinary source scan. Only the
/// provider's verified PNG or an exact pixel crop is added; existing files stay.
actor NativeMaterialSourceRecovery {
    enum Provider: String, Sendable { case polyHaven, ambientCG }
    struct Request: Sendable {
        let asset: String
        let resolution: String
        let target: String
        let destination: URL
        let width: Int
        let height: Int
        let expectedSHA256: String?
        let publishedRole: String?
        let provider: Provider
        let preserveOriginalDimensions: Bool

        init(asset: String, resolution: String, target: String, destination: URL,
             width: Int, height: Int, expectedSHA256: String? = nil, publishedRole: String? = nil,
             provider: Provider = .polyHaven, preserveOriginalDimensions: Bool = false) {
            self.asset = asset; self.resolution = resolution; self.target = target
            self.destination = destination; self.width = width; self.height = height
            self.expectedSHA256 = expectedSHA256
            self.publishedRole = publishedRole
            self.provider = provider
            self.preserveOriginalDimensions = preserveOriginalDimensions
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
        var provider: Provider = .polyHaven
        var archiveSHA256: String? = nil
        var archiveSHA1: String? = nil
        var archiveMember: String? = nil
        var archiveMemberCRC32: String? = nil
        var archiveMemberBytes: Int? = nil
        var provenanceSidecarURL: URL? = nil
        var provenanceWarning: String? = nil
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
        let publishedSHA1: String?
        init(statusCode: Int, url: URL, publishedSHA1: String? = nil) {
            self.statusCode = statusCode; self.url = url; self.publishedSHA1 = publishedSHA1
        }
    }

    struct Transport: Sendable {
        let fetchMetadata: @Sendable (URL) async throws -> HTTPData
        let download: @Sendable (URL, URL) async throws -> HTTPResponse

        static let live = Transport(fetchMetadata: { url in
            let (data, response) = try await networkSession.data(for: networkRequest(url))
            guard let http = response as? HTTPURLResponse, let finalURL = response.url else {
                throw RecoveryFailure("The material provider did not return an HTTP response.")
            }
            return HTTPData(data: data, statusCode: http.statusCode, url: finalURL)
        }, download: { url, destination in
            let (temporary, response) = try await networkSession.download(for: networkRequest(url))
            defer { try? FileManager.default.removeItem(at: temporary) }
            guard let http = response as? HTTPURLResponse, let finalURL = response.url else {
                throw RecoveryFailure("The material provider did not return an HTTP response.")
            }
            try Task.checkCancellation()
            try FileManager.default.moveItem(at: temporary, to: destination)
            return HTTPResponse(statusCode: http.statusCode, url: finalURL,
                                publishedSHA1: http.value(forHTTPHeaderField: "x-bz-content-sha1"))
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
    private struct AmbientPublishedPackage: Sendable { let url: URL; let filename: String; let bytes: Int }
    private struct AmbientPackage: Sendable {
        let key: String; let published: AmbientPublishedPackage; let archive: AmbientZIP
        let sha256: String; let publishedSHA1: String?
    }
    private var ambientCatalogs: [String: [String: AmbientPublishedPackage]] = [:]
    private var cachedAmbientPackage: AmbientPackage?

    init(transport: Transport = .live) { self.transport = transport }

    /// Unavailable assets and network failures leave import usable. Cancellation
    /// still stops the operation and removes any incomplete temporary download.
    func recover(_ request: Request,
                 onProgress: @escaping @Sendable (String) -> Void = { _ in }) async throws -> Result {
        try Task.checkCancellation()
        if request.provider == .ambientCG { return try await recoverAmbientCG(request, onProgress: onProgress) }
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
            guard let catalog = try await catalog(for: request.asset, apiURL: apiURL) else { return .notPublished }
            let published: PublishedFile
            if request.target == "input" {
                let candidates = catalog.keys.sorted().filter { Self.canonicalInputRole($0) == Self.canonicalInputRole(role) }
                    .compactMap { catalog[$0]?[request.resolution] }
                guard let selected = candidates.first else { return .notPublished }
                guard candidates.allSatisfy({ $0.url == selected.url && $0.md5.lowercased() == selected.md5.lowercased() && $0.bytes == selected.bytes }) else {
                    throw RecoveryFailure("Poly Haven publishes competing files for this color variant.")
                }
                published = selected
            } else {
                guard let selected = catalog[role]?[request.resolution] else { return .notPublished }
                published = selected
            }
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
            if !request.preserveOriginalDimensions && (width != request.width || height != request.height) {
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
            let proof = Provenance(destination: destination, downloadURL: published.url,
                apiURL: apiURL, publishedMD5: published.md5.lowercased(),
                publishedBytes: published.bytes, sha256: savedSHA256,
                sourceWidth: width, sourceHeight: height, cropRectangle: cropRectangle)
            return .recovered(Self.persisted(proof, onProgress: onProgress))
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            try Task.checkCancellation()
            return .failed(reason: error.localizedDescription)
        }
    }

    private func recoverAmbientCG(_ request: Request,
                                 onProgress: @escaping @Sendable (String) -> Void) async throws -> Result {
        let roles = ["input": "Color", "height": "Displacement", "roughness": "Roughness",
                     "normal": "NormalGL", "normal_dx": "NormalDX"]
        guard Self.validAmbientAsset(request.asset),
              request.resolution.range(of: #"^(?:1|2|4|8|16)k$"#, options: .regularExpression) != nil,
              let role = roles[request.target], request.publishedRole == nil,
              request.width > 0, request.height > 0, request.width <= 65536, request.height <= 65536,
              request.destination.isFileURL, request.destination.pathExtension.lowercased() == "png",
              request.expectedSHA256.map({ Self.hexadecimal($0, count: 64) }) ?? true else {
            return .failed(reason: "The source does not identify a supported ambientCG map.")
        }
        let destination = request.destination.standardizedFileURL
        if Self.pathExists(destination) { return .alreadyPresent }
        let folder = destination.deletingLastPathComponent()
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &directory), directory.boolValue else {
            return .failed(reason: "The original source folder is unavailable: \(folder.path)")
        }
        let staged = folder.appendingPathComponent(".material-map-download-\(UUID().uuidString).tmp")
        defer { try? FileManager.default.removeItem(at: staged) }
        do {
            let apiURL = Self.ambientAPIURL(request.asset)
            let catalog = try await ambientCatalog(for: request.asset, apiURL: apiURL)
            guard let published = catalog[request.resolution] else { return .notPublished }
            let key = request.asset + "|" + request.resolution
            let package: AmbientPackage
            if let cached = cachedAmbientPackage, cached.key == key { package = cached }
            else {
                // Keep only the current tier. The caller groups paired requests
                // so a package is downloaded once, without retaining a dataset
                // worth of large archives during preparation.
                cachedAmbientPackage = nil
                let archiveStage = folder.appendingPathComponent(".material-package-download-\(UUID().uuidString).tmp")
                defer { try? FileManager.default.removeItem(at: archiveStage) }
                onProgress("Obtaining \(published.filename) from ambientCG…")
                let response = try await transport.download(published.url, archiveStage)
                try Task.checkCancellation()
                guard response.statusCode == 200, Self.validAmbientDownloadURL(response.url, filename: published.filename) else {
                    throw RecoveryFailure("The published ambientCG package download is unavailable (HTTP \(response.statusCode)).")
                }
                var state = stat()
                guard lstat(archiveStage.path, &state) == 0, state.st_mode & S_IFMT == S_IFREG,
                      state.st_size == published.bytes else {
                    throw RecoveryFailure("The download did not match ambientCG's published package size.")
                }
                let verified = try Self.checksums(archiveStage)
                if let expected = response.publishedSHA1 {
                    guard Self.hexadecimal(expected, count: 40), verified.sha1 == expected.lowercased() else {
                        throw RecoveryFailure("The download did not match ambientCG's published package SHA1.")
                    }
                }
                let archive = try AmbientZIP(Data(contentsOf: archiveStage, options: .mappedIfSafe))
                package = AmbientPackage(key: key, published: published, archive: archive,
                                         sha256: verified.sha256, publishedSHA1: response.publishedSHA1?.lowercased())
                cachedAmbientPackage = package
            }
            let member = String(published.filename.dropLast(4)) + "_" + role + ".png"
            guard let entry = package.archive.entries[member] else { return .notPublished }
            try package.archive.extract(member, to: staged)
            let metadata = try NativePNG.sourceMetadata(Data(contentsOf: staged, options: .mappedIfSafe))
            let width = metadata["width"] as? Int ?? 0, height = metadata["height"] as? Int ?? 0
            guard request.preserveOriginalDimensions || (width == request.width && height == request.height) else {
                throw RecoveryFailure("The published ambientCG map has different pixel dimensions from this source set.")
            }
            guard request.target != "height" || metadata["sample_bits"] as? Int == 16 else {
                throw RecoveryFailure("The published displacement PNG does not preserve 16-bit source precision.")
            }
            guard !["input", "normal", "normal_dx"].contains(request.target) || [3, 4].contains(metadata["channels"] as? Int ?? 0) else {
                throw RecoveryFailure("The published color or normal map does not contain native RGB or RGBA channels.")
            }
            _ = try NativePNG.crop(staged, rectangle: [0, 0, 1, 1])
            let verified = try Self.checksums(staged)
            if let expected = request.expectedSHA256, verified.sha256 != expected.lowercased() {
                throw RecoveryFailure("The recovered map differs from this dataset's recorded original.")
            }
            try Task.checkCancellation()
            guard link(staged.path, destination.path) == 0 else {
                if errno == EEXIST { return .alreadyPresent }
                throw RecoveryFailure("The recovered map could not be saved to \(folder.path).")
            }
            onProgress("Saved \(destination.lastPathComponent) to the source folder.")
            let proof = Provenance(destination: destination, downloadURL: published.url, apiURL: apiURL,
                publishedMD5: "", publishedBytes: published.bytes, sha256: verified.sha256,
                sourceWidth: width, sourceHeight: height, cropRectangle: nil, provider: .ambientCG,
                archiveSHA256: package.sha256, archiveSHA1: package.publishedSHA1,
                archiveMember: member, archiveMemberCRC32: String(format: "%08x", entry.crc),
                archiveMemberBytes: entry.size)
            return .recovered(Self.persisted(proof, onProgress: onProgress))
        } catch is CancellationError { throw CancellationError() }
        catch {
            try Task.checkCancellation()
            return .failed(reason: error.localizedDescription)
        }
    }

    static func provenanceSidecar(for destination: URL) -> URL {
        let identity = SHA256.hash(data: Data(destination.lastPathComponent.utf8)).map { String(format: "%02x", $0) }.joined()
        return destination.deletingLastPathComponent().appendingPathComponent(".texture-studio-source-proof-\(identity).json")
    }

    /// Attach evidence only while it remains bound to the inspected source.
    static func sourceProvenance(for source: URL, sha256: String, bytes: Int, width: Int, height: Int) -> [String: Any]? {
        let path = source.resolvingSymlinksInPath().standardizedFileURL, sidecar = provenanceSidecar(for: path)
        guard let state = try? sidecar.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
              state.isRegularFile == true, state.isSymbolicLink != true, (state.fileSize ?? Int.max) <= 128 * 1024,
              let data = try? Data(contentsOf: sidecar), let proof = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              proof["schema"] as? String == "texture-studio-source-recovery-v1",
              proof["source_path"] as? String == path.path, proof["source_sha256"] as? String == sha256,
              proof["source_bytes"] as? Int == bytes, proof["source_width"] as? Int == width,
              proof["source_height"] as? Int == height, hexadecimal(sha256, count: 64),
              let provider = (proof["provider"] as? String).flatMap(Provider.init(rawValue:)),
              let apiText = proof["api_url"] as? String, let api = URL(string: apiText),
              let downloadText = proof["download_url"] as? String, let download = URL(string: downloadText),
              (proof["published_bytes"] as? Int ?? 0) > 0 else { return nil }
        let asset: String
        if provider == .polyHaven {
            asset = api.lastPathComponent
            guard validAsset(asset), apiText == "https://api.polyhaven.com/files/" + asset,
                  validProviderURL(download, host: "dl.polyhaven.org"), download.pathExtension.lowercased() == "png",
                  hexadecimal(proof["published_md5"] as? String ?? "", count: 32) else { return nil }
        } else {
            guard let components = URLComponents(url: api, resolvingAgainstBaseURL: false),
                  let name = components.queryItems?.first(where: { $0.name == "id" })?.value,
                  validAmbientAsset(name), api == ambientAPIURL(name),
                  let archiveMember = proof["archive_member"] as? String,
                  let filename = URLComponents(url: download, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "file" })?.value,
                  validAmbientDownloadURL(download, filename: filename), filename.hasPrefix(name + "_"),
                  archiveMember.hasPrefix(String(filename.dropLast(4)) + "_"), archiveMember.hasSuffix(".png"),
                  !archiveMember.contains("/"), !archiveMember.contains("\\"),
                  hexadecimal(proof["archive_sha256"] as? String ?? "", count: 64),
                  hexadecimal(proof["archive_member_crc32"] as? String ?? "", count: 8),
                  proof["archive_member_bytes"] as? Int == bytes else { return nil }
            asset = name
        }
        return ["provider": provider == .ambientCG ? "ambientCG" : "Poly Haven", "asset_id": asset,
            "published_api_url": apiText, "published_recovery_proof_path": sidecar.path,
            "source_recovery_provenance": proof]
    }

    private static func persisted(_ supplied: Provenance,
                                  onProgress: @escaping @Sendable (String) -> Void) -> Provenance {
        var proof = supplied
        do {
            let destination = proof.destination.resolvingSymlinksInPath().standardizedFileURL
            let state = try destination.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
            guard state.isRegularFile == true, state.isSymbolicLink != true,
                  let count = state.fileSize, try checksums(destination).sha256 == proof.sha256 else {
                throw RecoveryFailure("The original changed before recovery proof could be saved.")
            }
            let object: [String: Any] = ["schema": "texture-studio-source-recovery-v1", "provider": proof.provider.rawValue,
                "source_path": destination.path, "source_sha256": proof.sha256, "source_bytes": count,
                "source_width": proof.cropRectangle?[2] ?? proof.sourceWidth,
                "source_height": proof.cropRectangle?[3] ?? proof.sourceHeight,
                "api_url": proof.apiURL.absoluteString, "download_url": proof.downloadURL.absoluteString,
                "published_md5": proof.publishedMD5.isEmpty ? NSNull() : proof.publishedMD5 as Any,
                "published_bytes": proof.publishedBytes, "download_width": proof.sourceWidth, "download_height": proof.sourceHeight,
                "crop_rectangle": proof.cropRectangle as Any? ?? NSNull(),
                "archive_sha256": proof.archiveSHA256 as Any? ?? NSNull(), "archive_sha1": proof.archiveSHA1 as Any? ?? NSNull(),
                "archive_member": proof.archiveMember as Any? ?? NSNull(), "archive_member_crc32": proof.archiveMemberCRC32 as Any? ?? NSNull(),
                "archive_member_bytes": proof.archiveMemberBytes as Any? ?? NSNull(),
                "recovered_at": ISO8601DateFormatter().string(from: Date())]
            let sidecar = provenanceSidecar(for: destination)
            try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: sidecar, options: .atomic)
            proof.provenanceSidecarURL = sidecar
        } catch {
            proof.provenanceWarning = "Saved the original map, but its recovery proof could not be saved: " + error.localizedDescription
            onProgress(proof.provenanceWarning!)
        }
        return proof
    }

    private func ambientCatalog(for asset: String, apiURL: URL) async throws -> [String: AmbientPublishedPackage] {
        if let cached = ambientCatalogs[asset] { return cached }
        let response = try await transport.fetchMetadata(apiURL)
        try Task.checkCancellation()
        guard response.url == apiURL else {
            throw RecoveryFailure("ambientCG's file listing did not come from the expected provider.")
        }
        if response.statusCode == 404 { ambientCatalogs[asset] = [:]; return [:] }
        guard response.statusCode == 200, response.data.count <= 2 * 1024 * 1024,
              let object = try JSONSerialization.jsonObject(with: response.data) as? [String: Any],
              let assets = object["foundAssets"] as? [[String: Any]] else {
            throw RecoveryFailure("ambientCG's file listing is unavailable (HTTP \(response.statusCode)).")
        }
        var catalog: [String: AmbientPublishedPackage] = [:]
        for item in assets where item["assetId"] as? String == asset {
            for folder in (item["downloadFolders"] as? [String: [String: Any]] ?? [:]).values {
                let categories = folder["downloadFiletypeCategories"] as? [String: [String: Any]] ?? [:]
                for download in categories["zip"]?["downloads"] as? [[String: Any]] ?? [] {
                    guard let attribute = download["attribute"] as? String,
                          attribute.range(of: #"^(?:1|2|4|8|16)K-PNG$"#, options: .regularExpression) != nil,
                          let filename = download["fileName"] as? String, filename == asset + "_" + attribute + ".zip",
                          let urlString = download["downloadLink"] as? String, let url = URL(string: urlString),
                          Self.validAmbientDownloadURL(url, filename: filename),
                          let size = download["size"] as? Int, size > 0, size <= 2 * 1024 * 1024 * 1024 else { continue }
                    let resolution = String(attribute.dropLast(4)).lowercased()
                    guard catalog[resolution] == nil else { throw RecoveryFailure("ambientCG published competing PNG packages for this tier.") }
                    catalog[resolution] = AmbientPublishedPackage(url: url, filename: filename, bytes: size)
                }
            }
        }
        ambientCatalogs[asset] = catalog
        return catalog
    }

    private static func ambientAPIURL(_ asset: String) -> URL {
        var components = URLComponents(string: "https://ambientcg.com/api/v2/full_json")!
        components.queryItems = [URLQueryItem(name: "include", value: "downloadData"), URLQueryItem(name: "id", value: asset)]
        return components.url!
    }
    private static func validAmbientAsset(_ asset: String) -> Bool {
        asset.count <= 200 && asset.range(of: #"^[A-Z][A-Za-z0-9]*[0-9][A-Za-z0-9]*$"#, options: .regularExpression) != nil
    }
    private static func validAmbientDownloadURL(_ url: URL, filename: String) -> Bool {
        guard url.scheme == "https", url.port == nil, url.user == nil, url.password == nil, url.fragment == nil else { return false }
        if url.host == "ambientcg.com", url.path == "/get", let components = URLComponents(url: url, resolvingAgainstBaseURL: false) {
            return components.queryItems?.count == 1 && components.queryItems?.first?.name == "file" && components.queryItems?.first?.value == filename
        }
        return url.host == "acg-download.struffelproductions.com" && url.query == nil &&
               url.path.hasPrefix("/file/ambientCG-Web/download/") && url.lastPathComponent == filename
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
        role.range(of: #"^(?:col|diffuse|diff|color|albedo)(?:_?\d+)?$|^coll\d+$"#, options: [.regularExpression, .caseInsensitive]) != nil
    }
    private static func canonicalInputRole(_ role: String) -> String {
        let value = role.lowercased()
        if value.range(of: #"^(?:coll|col_?)\d+$"#, options: .regularExpression) != nil {
            let digits = value.drop { !$0.isNumber }
            return "col_" + String(Int(digits) ?? 0)
        }
        return value
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
    private static func checksums(_ url: URL) throws -> (md5: String, sha256: String, sha1: String) {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var md5 = Insecure.MD5(), sha256 = SHA256(), sha1 = Insecure.SHA1()
        while let bytes = try file.read(upToCount: 1024 * 1024), !bytes.isEmpty {
            try Task.checkCancellation()
            md5.update(data: bytes); sha256.update(data: bytes); sha1.update(data: bytes)
        }
        return (md5.finalize().map { String(format: "%02x", $0) }.joined(),
                sha256.finalize().map { String(format: "%02x", $0) }.joined(),
                sha1.finalize().map { String(format: "%02x", $0) }.joined())
    }

    /// Reads only one named PNG member. ZIP paths never become output paths,
    /// and compressed data expands in bounded chunks with the declared CRC.
    private struct AmbientZIP: Sendable {
        struct Entry: Sendable { let method: UInt16; let crc: UInt32; let compressed: Int; let size: Int; let offset: Int }
        let bytes: Data
        let directoryStart: Int
        let entries: [String: Entry]
        init(_ bytes: Data) throws {
            self.bytes = bytes
            guard bytes.count >= 22 else { throw RecoveryFailure("The ambientCG package is not a ZIP archive.") }
            var end: Int?
            for offset in stride(from: bytes.count - 22, through: max(0, bytes.count - 65557), by: -1) {
                if Self.u32(bytes, offset) == 0x06054b50,
                   offset + 22 + Int(Self.u16(bytes, offset + 20)) == bytes.count { end = offset; break }
            }
            guard let end, Self.u16(bytes, end + 4) == 0, Self.u16(bytes, end + 6) == 0,
                  Self.u16(bytes, end + 8) == Self.u16(bytes, end + 10) else {
                throw RecoveryFailure("The ambientCG package has an invalid ZIP directory.")
            }
            let count = Int(Self.u16(bytes, end + 10)), size = Int(Self.u32(bytes, end + 12)), start = Int(Self.u32(bytes, end + 16))
            guard count > 0, count <= 1024, start >= 0, start <= end, size >= 0, size == end - start else {
                throw RecoveryFailure("The ambientCG ZIP directory exceeds package bounds.")
            }
            directoryStart = start
            var entries: [String: Entry] = [:], cursor = start
            for _ in 0..<count {
                guard cursor <= start + size - 46, Self.u32(bytes, cursor) == 0x02014b50 else {
                    throw RecoveryFailure("The ambientCG ZIP member descriptor is invalid.")
                }
                let flags = Self.u16(bytes, cursor + 8), method = Self.u16(bytes, cursor + 10)
                let nameSize = Int(Self.u16(bytes, cursor + 28)), extraSize = Int(Self.u16(bytes, cursor + 30)), commentSize = Int(Self.u16(bytes, cursor + 32))
                let recordSize = 46 + nameSize + extraSize + commentSize
                let mode = Self.u32(bytes, cursor + 38) >> 16, type = mode & 0xf000
                guard flags & 1 == 0, [UInt16(0), 8].contains(method), Self.u16(bytes, cursor + 34) == 0,
                      [UInt32(0), 0x8000, 0x4000].contains(type), nameSize > 0, nameSize <= 1024,
                      recordSize <= start + size - cursor,
                      let name = String(data: bytes.subdata(in: cursor + 46..<cursor + 46 + nameSize), encoding: .utf8),
                      !name.hasPrefix("/"), !name.contains("\\"), !name.contains("\0"),
                      !name.split(separator: "/").contains(".."), entries[name] == nil else {
                    throw RecoveryFailure("The ambientCG package has an unsafe or unsupported ZIP member.")
                }
                let compressed = Int(Self.u32(bytes, cursor + 20)), expanded = Int(Self.u32(bytes, cursor + 24)), offset = Int(Self.u32(bytes, cursor + 42))
                guard compressed <= 2 * 1024 * 1024 * 1024, expanded <= 2 * 1024 * 1024 * 1024,
                      offset <= start - 30, compressed <= start - offset - 30 else {
                    throw RecoveryFailure("The ambientCG ZIP member exceeds package bounds.")
                }
                entries[name] = Entry(method: method, crc: Self.u32(bytes, cursor + 16), compressed: compressed, size: expanded, offset: offset)
                cursor += recordSize
            }
            guard cursor == start + size else { throw RecoveryFailure("The ambientCG ZIP directory size differs from its entries.") }
            self.entries = entries
        }
        func extract(_ name: String, to destination: URL) throws {
            try Task.checkCancellation()
            guard let entry = entries[name] else { throw RecoveryFailure("The requested ambientCG PNG is absent from its package.") }
            let cursor = entry.offset
            guard Self.u32(bytes, cursor) == 0x04034b50, Self.u16(bytes, cursor + 8) == entry.method,
                  Self.u16(bytes, cursor + 6) & 1 == 0 else { throw RecoveryFailure("The ambientCG ZIP local member is invalid.") }
            let nameSize = Int(Self.u16(bytes, cursor + 26)), extraSize = Int(Self.u16(bytes, cursor + 28))
            let start = cursor + 30 + nameSize + extraSize
            guard start <= directoryStart, entry.compressed <= directoryStart - start,
                  String(data: bytes.subdata(in: cursor + 30..<cursor + 30 + nameSize), encoding: .utf8) == name else {
                throw RecoveryFailure("The ambientCG ZIP local member name or size differs.")
            }
            guard FileManager.default.createFile(atPath: destination.path, contents: nil) else {
                throw RecoveryFailure("The ambientCG PNG staging file could not be created.")
            }
            let output = try FileHandle(forWritingTo: destination)
            defer { try? output.close() }
            var crc: uLong = 0, produced = 0
            if entry.method == 0 {
                guard entry.compressed == entry.size else { throw RecoveryFailure("The stored ambientCG ZIP member size differs.") }
                while produced < entry.size {
                    try Task.checkCancellation()
                    let count = min(1024 * 1024, entry.size - produced)
                    let chunk = bytes.subdata(in: start + produced..<start + produced + count)
                    crc = chunk.withUnsafeBytes { crc32_z(crc, $0.bindMemory(to: Bytef.self).baseAddress, count) }
                    try output.write(contentsOf: chunk); produced += count
                }
            } else {
                var stream = z_stream()
                guard inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
                    throw RecoveryFailure("The ambientCG ZIP decoder could not be initialized.")
                }
                defer { inflateEnd(&stream) }
                var buffer = [UInt8](repeating: 0, count: 1024 * 1024)
                try bytes.withUnsafeBytes { input in
                    stream.next_in = UnsafeMutablePointer(mutating: input.bindMemory(to: Bytef.self).baseAddress!.advanced(by: start))
                    stream.avail_in = uInt(entry.compressed)
                    var status: Int32 = Z_OK
                    repeat {
                        try Task.checkCancellation()
                        let count = try buffer.withUnsafeMutableBytes { storage -> Int in
                            stream.next_out = storage.bindMemory(to: Bytef.self).baseAddress
                            stream.avail_out = uInt(storage.count)
                            status = inflate(&stream, Z_NO_FLUSH)
                            guard status == Z_OK || status == Z_STREAM_END else {
                                throw RecoveryFailure("The ambientCG ZIP compressed member is invalid.")
                            }
                            let count = storage.count - Int(stream.avail_out)
                            guard count <= entry.size - produced else { throw RecoveryFailure("The ambientCG ZIP member expanded beyond its declared size.") }
                            crc = crc32_z(crc, storage.bindMemory(to: Bytef.self).baseAddress, count)
                            try output.write(contentsOf: Data(storage.prefix(count)))
                            return count
                        }
                        produced += count
                    } while status != Z_STREAM_END
                    guard stream.total_in == entry.compressed else { throw RecoveryFailure("The ambientCG ZIP compressed size differs.") }
                }
            }
            guard produced == entry.size, UInt32(crc) == entry.crc else {
                throw RecoveryFailure("The ambientCG ZIP PNG size or CRC checksum differs.")
            }
        }
        private static func u16(_ bytes: Data, _ offset: Int) -> UInt16 {
            UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8
        }
        private static func u32(_ bytes: Data, _ offset: Int) -> UInt32 {
            UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8 | UInt32(bytes[offset + 2]) << 16 | UInt32(bytes[offset + 3]) << 24
        }
    }

    private struct RecoveryFailure: LocalizedError {
        let errorDescription: String?
        init(_ description: String) { errorDescription = description }
    }
    private final class ProviderRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
            let host = task.originalRequest?.url?.host
            if host == "ambientcg.com", let original = task.originalRequest?.url,
               let filename = URLComponents(url: original, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "file" })?.value,
               let url = request.url, NativeMaterialSourceRecovery.validAmbientDownloadURL(url, filename: filename) {
                completionHandler(request); return
            }
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
