import Foundation

struct StudioPreferences: Codable, Equatable {
    var settings: TextureSettings?
    var depthChoice: DepthChoice?
    var selectedPreview: String?
    var showInspector: Bool?
    var exportDirectory: String?
    static let key = "studioSettings.v2"
    static var defaults: UserDefaults {
        if Bundle.main.bundleIdentifier == "org.ipde.texture-studio" { return .standard }
        return UserDefaults(suiteName: "org.ipde.texture-studio") ?? .standard
    }
    static func load(from defaults: UserDefaults = Self.defaults) -> Self {
        guard let data = defaults.data(forKey: key), let value = try? JSONDecoder().decode(Self.self, from: data) else { return Self() }
        return value
    }
    func save(to defaults: UserDefaults = Self.defaults) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        defaults.set(data, forKey: Self.key)
    }
}
