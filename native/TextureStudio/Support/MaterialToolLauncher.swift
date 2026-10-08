import AppKit

enum MaterialToolLauncher {
    static func open(_ role: MaterialTool) {
        let app = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent(role.title + ".app")
        launch(app, fallbackArguments: ["--tool", role.rawValue])
    }
    static func openStudio() {
        let app = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("Texture Studio.app")
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
