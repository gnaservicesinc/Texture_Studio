import SwiftUI
import AppKit

@main
struct TextureStudioApp: App {
    @NSApplicationDelegateAdaptor(StudioAppDelegate.self) private var delegate
    @State private var models = ModelManager()
    @State private var adviser = OllamaDecisionService()
    @State private var workbench = WorkbenchStore()
    @AppStorage(StudioPreferences.developerModeKey, store: StudioPreferences.defaults) private var developerMode = false

    var body: some Scene {
        WindowGroup(MaterialTool.launchRole?.title ?? "Texture Studio") {
            if let role = MaterialTool.launchRole {
                MaterialToolRootView(role: role, store: workbench)
            } else {
                ContentView(models: models, adviser: adviser)
                    .frame(minWidth: 980, minHeight: 680)
            }
        }
        .defaultSize(width: 1320, height: 900)
        .commands { StudioCommands(); MaterialToolCommands() }
        Window("Model Training", id: "model-training") {
            ModelTrainingHubView(store: workbench)
        }
        .defaultSize(width: 1440, height: 940)
        Settings {
            if MaterialTool.launchRole != nil {
                WorkbenchRuntimeView(store: workbench).frame(width: 710, height: 500)
            } else {
                VStack(alignment: .leading, spacing: 0) {
                    Toggle("Developer mode", isOn: $developerMode).padding(20)
                        .help("Expose advanced model controls. Export fused full checkpoints alongside the separate LoRA, with Hugging Face publishing controls.")
                    Divider()
                    ModelLibraryView(models: models, adviser: adviser)
                        .frame(width: 640, height: 540)
                }
            }
        }
    }
}

struct MaterialToolCommands: Commands {
    @Environment(\.openWindow) private var openWindow
    var body: some Commands {
        CommandMenu("Material Tools") {
            Button("Model Training…", systemImage: "graduationcap") { openWindow(id: "model-training") }
                .keyboardShortcut("t", modifiers: [.command, .shift])
            Divider()
            ForEach(MaterialTool.allCases) { role in
                Button(role.title, systemImage: role.symbol) { MaterialToolLauncher.open(role) }
            }
            Button("Texture Studio", systemImage: "square.3.layers.3d") { MaterialToolLauncher.openStudio() }
        }
    }
}

final class StudioAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard WorkbenchLifecycle.shared.hasOperations else { return .terminateNow }
        WorkbenchLifecycle.shared.stopAll()
        Task { @MainActor in
            while WorkbenchLifecycle.shared.hasOperations { try? await Task.sleep(for: .milliseconds(250)) }
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if CommandLine.arguments.contains("--smoke-test") {
            Task {
                do {
                    try await StudioSmoke.run()
                    print("Texture Studio native smoke passed")
                    NSApp.terminate(nil)
                } catch {
                    fputs("Texture Studio smoke failed: \(error)\n", stderr)
                    exit(1)
                }
            }
        }
    }
}

private struct WorkspaceFocusKey: FocusedValueKey { typealias Value = TextureWorkspace }
private struct StudioModelsFocusKey: FocusedValueKey { typealias Value = ModelManager }
extension FocusedValues {
    var textureWorkspace: TextureWorkspace? {
        get { self[WorkspaceFocusKey.self] }
        set { self[WorkspaceFocusKey.self] = newValue }
    }
    var textureModels: ModelManager? {
        get { self[StudioModelsFocusKey.self] }
        set { self[StudioModelsFocusKey.self] = newValue }
    }
}

struct StudioCommands: Commands {
    @FocusedValue(\.textureWorkspace) private var workspace
    @FocusedValue(\.textureModels) private var models
    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Import Photo…") { workspace?.choosePhoto() }
                .keyboardShortcut("o")
                .disabled(workspace?.isBusy ?? true)
            Button("Open Texture Recipe…") { workspace?.chooseRecipe() }
                .keyboardShortcut("o", modifiers: [.command, .shift])
                .disabled(workspace?.isBusy ?? true)
        }
        CommandGroup(replacing: .saveItem) {
            Button("Save Texture Recipe…") { workspace?.saveRecipe() }
                .keyboardShortcut("s")
                .disabled(workspace?.source == nil || workspace?.isBusy == true)
            Button("Export Material…") {
                if let workspace, let models { workspace.chooseExport(models: models) }
            }
                .keyboardShortcut("e")
                .disabled(workspace?.source == nil || workspace?.isBusy == true || models == nil)
        }
        CommandMenu("Material") {
            Button("Generate / Update Material") {
                if let workspace, let models { workspace.updatePreview(models: models) }
            }
                .keyboardShortcut("r")
                .disabled(workspace?.source == nil || workspace?.isBusy == true || workspace?.materialNeedsUpdate != true || models == nil)
            Button("Open Full Quality") {
                if let workspace, let models { workspace.inspectFullQuality(models: models) }
            }
                .keyboardShortcut("f", modifiers: [.command, .shift])
                .disabled(workspace?.fullQualityAvailable != true || workspace?.isBusy == true || models == nil)
            Divider()
            Button("Attach Height / Depth Map…") { workspace?.chooseDepth() }
                .disabled(workspace?.source == nil || workspace?.isBusy == true)
            Button("Manage Local Models…") { workspace?.showModels = true }
            Button("Show Inspector") { workspace?.showInspector.toggle() }
                .keyboardShortcut("i", modifiers: [.command, .option])
        }
    }
}
