import Foundation

enum HuggingFaceUpload {
    static func validRepository(_ value: String) -> Bool {
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts.allSatisfy { part in
            part.count <= 96 && !part.contains("--") && !part.contains("..")
                && part.range(of: "^[A-Za-z0-9][A-Za-z0-9_.-]*$", options: .regularExpression) != nil
                && !part.hasSuffix(".") && !part.hasSuffix("-")
        }
    }

    /// A blank custom destination follows the selected model and saved account.
    /// Including its SHA avoids silently replacing another checkpoint's repo.
    static func repository(account: String, checkpoint: WorkbenchCheckpoint) -> String {
        let displayName = checkpoint.modelName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let name = (displayName.isEmpty ? checkpoint.url.deletingLastPathComponent().lastPathComponent : displayName)
            .folding(options: [.diacriticInsensitive, .widthInsensitive], locale: Locale(identifier: "en_US_POSIX")).lowercased()
            .replacingOccurrences(of: "[^a-z0-9]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        let stem = name.isEmpty ? "material" : String(name.prefix(50)).trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return "\(account)/matterloom-\(checkpoint.target)-\(stem)-\(checkpoint.sha256.prefix(8))"
    }
}

struct HuggingFaceAccountResponse: Decodable, Sendable {
    let authenticated: Bool
    let username: String?
    let message: String
}

struct HuggingFaceUploadResponse: Decodable, Sendable {
    let repository: String
    let `private`: Bool
    let url: String
    let commitUrl: String?
    let catalogWarning: String?
    let sourceCheckpointSha256: String
    let packagePath: String
}
