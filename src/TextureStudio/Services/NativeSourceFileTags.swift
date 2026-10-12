import Darwin
import Foundation

/// Optional extended attributes travel with originals without changing image
/// bytes. Changed or replaced files lose their cached checks and predictions.
enum NativeSourceFileTags {
    static let attribute = "org.ipde.texture-studio.source"
    private static let lock = NSLock()
    private static let readLock = NSLock()
    private struct Cached { let identity: Identity; let changed: Int64; let tags: Tags? }
    nonisolated(unsafe) private static var cache: [String: Cached] = [:]
    struct Identity: Codable, Equatable, Sendable {
        let bytes: Int64
        let modified: Int64
        let inode: UInt64
        let device: Int32
    }
    struct Tags: Codable, Sendable {
        let identity: Identity
        var role: String?
        var classLabel: String?
        var classifierIdentity: String?
        var verifiedSHA256: String?
        var verifiedMetadata: String?
        var reviewStatus: String?
    }
    static func identity(_ url: URL) throws -> Identity {
        var value = stat()
        guard stat(url.path, &value) == 0 else { throw POSIXError(.ENOENT) }
        return Identity(bytes: Int64(value.st_size), modified: Int64(value.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(value.st_mtimespec.tv_nsec), inode: UInt64(value.st_ino), device: Int32(value.st_dev))
    }
    static func read(_ url: URL) -> Tags? {
        var value = stat()
        guard stat(url.path, &value) == 0 else { return nil }
        let identity = Identity(bytes: Int64(value.st_size), modified: Int64(value.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(value.st_mtimespec.tv_nsec), inode: UInt64(value.st_ino), device: Int32(value.st_dev))
        let changed = Int64(value.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(value.st_ctimespec.tv_nsec)
        if let cached = readLock.withLock({ cache[url.path] }), cached.identity == identity, cached.changed == changed { return cached.tags }
        let tags = readAttribute(url, identity: identity)
        readLock.withLock {
            if cache.count >= 8192 { cache.removeAll(keepingCapacity: true) }
            cache[url.path] = Cached(identity: identity, changed: changed, tags: tags)
        }
        return tags
    }
    private static func readAttribute(_ url: URL, identity: Identity) -> Tags? {
        let size = getxattr(url.path, attribute, nil, 0, 0, 0)
        guard size > 0, size < 16_384 else { return nil }
        var data = Data(count: size)
        let count = data.withUnsafeMutableBytes { getxattr(url.path, attribute, $0.baseAddress, size, 0, 0) }
        guard count == size, let tags = try? JSONDecoder().decode(Tags.self, from: data),
              tags.identity == identity else { return nil }
        return tags
    }
    static func update(_ url: URL, role: String? = nil, classLabel: String? = nil,
                       classifierIdentity: String? = nil, verifiedSHA256: String? = nil,
                       reviewStatus: String? = nil, verifiedMetadata: String? = nil) throws {
        try lock.withLock {
        let initial = try identity(url)
        var tags = read(url) ?? Tags(identity: initial)
        if let role { tags.role = role }
        if let classLabel { tags.classLabel = classLabel }
        if let classifierIdentity { tags.classifierIdentity = classifierIdentity }
        if let verifiedSHA256 { tags.verifiedSHA256 = verifiedSHA256 }
        if let verifiedMetadata { tags.verifiedMetadata = verifiedMetadata }
        if let reviewStatus { tags.reviewStatus = reviewStatus }
        let data = try JSONEncoder().encode(tags)
        guard try identity(url) == initial else { throw StudioError("Image changed while saving its tags.") }
        let result = data.withUnsafeBytes { setxattr(url.path, attribute, $0.baseAddress, data.count, 0, 0) }
        guard result == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        readLock.withLock { cache[url.path] = nil }
        }
    }
}
