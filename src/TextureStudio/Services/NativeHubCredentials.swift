import Foundation
import Security

enum NativeHubCredentials {
    private static let service = "org.ipde.texture-studio.huggingface"

    /// Uses the same active-token path as `hf auth login`, without launching a
    /// shell or copying its credential into Texture Studio's Keychain entry.
    static func tokenFileURL(environment: [String: String],
                             homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        func expanded(_ path: String) -> String {
            let expression = try! NSRegularExpression(pattern: #"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)"#)
            var result = path
            for match in expression.matches(in: path, range: NSRange(path.startIndex..., in: path)).reversed() {
                let keyRange = match.range(at: match.range(at: 1).location == NSNotFound ? 2 : 1)
                guard let range = Range(match.range, in: result), let key = Range(keyRange, in: path),
                      let value = environment[String(path[key])] else { continue }
                result.replaceSubrange(range, with: value)
            }
            if result == "~" { return homeDirectory.path }
            if result.hasPrefix("~/") { return homeDirectory.appendingPathComponent(String(result.dropFirst(2))).path }
            return result
        }
        func path(_ key: String) -> String? {
            guard let value = environment[key], !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            return expanded(value)
        }
        if let tokenPath = path("HF_TOKEN_PATH") { return URL(fileURLWithPath: tokenPath) }
        let hubHome: URL
        if let hfHome = path("HF_HOME") { hubHome = URL(fileURLWithPath: hfHome) }
        else {
            let cache = path("XDG_CACHE_HOME").map { URL(fileURLWithPath: $0) } ?? homeDirectory.appendingPathComponent(".cache")
            hubHome = cache.appendingPathComponent("huggingface")
        }
        return hubHome.appendingPathComponent("token")
    }

    static func readTokenFile(_ url: URL) -> String? {
        guard let file = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? file.close() }
        // Tokens are small. Bound the read so a mistaken path cannot load a
        // model or another large file while checking an account.
        guard let bytes = try? file.read(upToCount: 16_385), bytes.count <= 16_384 else { return nil }
        return String(data: bytes, encoding: .utf8)
    }

    static func read() -> String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service, kSecAttrAccount as String: "token",
            kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
    static func save(_ value: String) throws {
        let token = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service, kSecAttrAccount as String: "token"]
        if token.isEmpty {
            let status = SecItemDelete(query as CFDictionary)
            guard [errSecSuccess, errSecItemNotFound].contains(status) else { throw StudioError("Could not remove the saved Hub token.") }
            return
        }
        let attributes: [String: Any] = [kSecValueData as String: Data(token.utf8)]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var entry = query
            entry[kSecValueData as String] = Data(token.utf8)
            entry[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            guard SecItemAdd(entry as CFDictionary, nil) == errSecSuccess else { throw StudioError("Could not save the Hub token in Keychain.") }
        } else if status != errSecSuccess { throw StudioError("Could not update the saved Hub token.") }
    }
}
