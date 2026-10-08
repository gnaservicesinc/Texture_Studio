import AppKit

enum MaterialToolLauncher {
    /// Installed tools live inside the parent bundle. Development bundles and
    /// separately copied tools can still resolve a sibling suite.
    static func studioURL(containing bundle: URL) -> URL {
        if bundle.lastPathComponent == "Texture Studio.app" { return bundle }
        let applications = bundle.deletingLastPathComponent()
        if applications.lastPathComponent == "Applications",
           applications.deletingLastPathComponent().lastPathComponent == "Contents" {
            return applications.deletingLastPathComponent().deletingLastPathComponent()
        }
        return applications.appendingPathComponent("Texture Studio.app")
    }
    static func toolURL(_ role: MaterialTool, containing bundle: URL) -> URL {
        let embedded = studioURL(containing: bundle).appendingPathComponent("Contents/Applications/\(role.title).app")
        if FileManager.default.fileExists(atPath: embedded.path) { return embedded }
        return bundle.deletingLastPathComponent().appendingPathComponent(role.title + ".app")
    }
    static func open(_ role: MaterialTool) {
        let app = toolURL(role, containing: Bundle.main.bundleURL)
        launch(app, fallbackArguments: ["--tool", role.rawValue])
    }
    static func openStudio() {
        let app = studioURL(containing: Bundle.main.bundleURL)
        launch(app, fallbackArguments: ["--tool", "studio"])
    }
    private static func launch(_ app: URL, fallbackArguments: [String]) {
        let config = NSWorkspace.OpenConfiguration()
        config.activates = true
        if FileManager.default.fileExists(atPath: app.path) { NSWorkspace.shared.openApplication(at: app, configuration: config) }
        else {
            config.createsNewApplicationInstance = true; config.arguments = fallbackArguments
            NSWorkspace.shared.openApplication(at: Bundle.main.bundleURL, configuration: config)
        }
    }
}
