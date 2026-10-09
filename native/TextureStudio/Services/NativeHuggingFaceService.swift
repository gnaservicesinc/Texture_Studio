import Foundation

/// Account and catalog discovery use the Hub HTTPS API directly. This does not
/// import a model runtime or execute a CLI. Credentials live in Apple Keychain.
struct NativeHuggingFaceService: Sendable {
    typealias Transport = @Sendable (URLRequest) async throws -> Data
    let token: String?
    let catalogURL: URL
    let transport: Transport

    init(token: String? = Self.savedToken(), catalogURL: URL? = nil,
         transport: @escaping Transport = Self.fetch) {
        self.token = token
        self.catalogURL = catalogURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/material-hub-catalog.json")
        self.transport = transport
    }

    func accountJSON() async throws -> String {
        let identity = try await account()
        try Task.checkCancellation()
        return try encode(identity)
    }

    func modelsJSON() async throws -> String {
        var models: [String: [String: Any]] = [:]
        if FileManager.default.fileExists(atPath: catalogURL.path) {
            let document = try JSONSerialization.jsonObject(with: Data(contentsOf: catalogURL)) as? [String: Any]
            guard let records = document?["models"] as? [[String: Any]] else {
                throw StudioError("The saved Hub model catalog is invalid.")
            }
            for record in records {
                guard let repository = record["repository"] as? String,
                      HuggingFaceUpload.validRepository(repository) else { continue }
                models[repository] = record
            }
        }
        let identity = try await account()
        var message: String?
        if identity.authenticated, let username = identity.username {
            var url = URLComponents(string: "https://huggingface.co/api/models")!
            url.queryItems = [URLQueryItem(name: "author", value: username),
                URLQueryItem(name: "filter", value: "texture-studio-material"),
                URLQueryItem(name: "full", value: "true"), URLQueryItem(name: "limit", value: "100")]
            do {
                let bytes = try await transport(request(url.url!))
                try Task.checkCancellation()
                guard let records = try JSONSerialization.jsonObject(with: bytes) as? [[String: Any]] else {
                    throw StudioError("The Hub returned an invalid model inventory.")
                }
                for record in records {
                    guard let repository = record["id"] as? String, HuggingFaceUpload.validRepository(repository),
                          let files = record["siblings"] as? [[String: Any]] else { continue }
                    let names = Set(files.compactMap { $0["rfilename"] as? String })
                    let checkpoint = names.contains("model.safetensors") ? "model.safetensors" :
                        names.contains("adapter.safetensors") ? "adapter.safetensors" : nil
                    guard let checkpoint else { continue }
                    var model = models[repository] ?? [:]
                    model["repository"] = repository
                    model["revision"] = record["sha"] ?? NSNull()
                    model["checkpoint_filename"] = checkpoint
                    models[repository] = model
                }
            } catch {
                try Task.checkCancellation()
                if error is CancellationError { throw error }
                message = "Hub model discovery is unavailable; your saved downloadable models remain listed"
            }
        }
        var result: [String: Any] = ["models": models.keys.sorted().compactMap { models[$0] },
            "authenticated": identity.authenticated, "username": identity.username as Any? ?? NSNull()]
        if let message { result["message"] = message }
        try Task.checkCancellation()
        return String(decoding: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), as: UTF8.self)
    }

    struct Account: Codable, Sendable {
        let authenticated: Bool
        let username: String?
        let message: String
    }

    func account() async throws -> Account {
        try Task.checkCancellation()
        guard let token, !token.isEmpty else {
            return Account(authenticated: false, username: nil,
                message: "Save a Hugging Face token in settings, then refresh the account")
        }
        do {
            let bytes = try await transport(request(URL(string: "https://huggingface.co/api/whoami-v2")!))
            try Task.checkCancellation()
            guard let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                  let username = object["name"] as? String, !username.isEmpty else {
                throw StudioError("The Hub did not return an account identity.")
            }
            return Account(authenticated: true, username: username, message: "Signed in as " + username)
        } catch {
            try Task.checkCancellation()
            if error is CancellationError { throw error }
            return Account(authenticated: false, username: nil,
                message: "Hugging Face login could not be verified. Check your connection or update the token in settings, then refresh the account")
        }
    }

    static func savedToken(environment: [String: String] = ProcessInfo.processInfo.environment,
                           keychain: () -> String? = NativeHubCredentials.read) -> String? {
        func cleaned(_ value: String?) -> String? {
            guard let value else { return nil }
            let text = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? nil : text
        }
        return cleaned(environment["HF_TOKEN"]) ?? cleaned(keychain())
    }

    private func request(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url, timeoutInterval: 20)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let token { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
        return request
    }

    static func fetch(_ request: URLRequest) async throws -> Data {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForResource = 30
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let (bytes, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
            throw StudioError("The Hub request failed.")
        }
        return bytes
    }

    private func encode(_ account: Account) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(account), as: UTF8.self)
    }
}
