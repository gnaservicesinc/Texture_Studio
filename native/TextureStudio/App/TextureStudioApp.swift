import SwiftUI
import AppKit

@main
struct TextureStudioApp: App {
    @NSApplicationDelegateAdaptor(StudioAppDelegate.self) private var delegate
    @State private var models = ModelManager()
    @State private var adviser = OllamaDecisionService()
    @State private var runtime = PythonDepthService()
    @State private var workbench = WorkbenchStore()

    var body: some Scene {
        WindowGroup(MaterialTool.launchRole?.title ?? "Texture Studio") {
            if let role = MaterialTool.launchRole {
                MaterialToolRootView(role: role, store: workbench)
            } else {
                ContentView(models: models, adviser: adviser, runtime: runtime)
                    .frame(minWidth: 980, minHeight: 680)
            }
        }
        .defaultSize(width: 1320, height: 900)
        .commands { StudioCommands(); MaterialToolCommands() }
        Settings {
            if MaterialTool.launchRole != nil {
                WorkbenchRuntimeView(store: workbench).frame(width: 710, height: 500)
            } else {
                ModelLibraryView(models: models, adviser: adviser, runtime: runtime)
                    .frame(width: 640, height: 540)
            }
        }
    }
}

struct MaterialToolCommands: Commands {
    var body: some Commands {
        CommandMenu("Material Tools") {
            ForEach(MaterialTool.allCases) { role in
                Button(role.title, systemImage: role.symbol) { MaterialToolLauncher.open(role) }
            }
            Button("Texture Studio", systemImage: "square.3.layers.3d") { MaterialToolLauncher.openStudio() }
        }
    }
}

final class StudioAppDelegate: NSObject, NSApplicationDelegate {
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
extension FocusedValues {
    var textureWorkspace: TextureWorkspace? {
        get { self[WorkspaceFocusKey.self] }
        set { self[WorkspaceFocusKey.self] = newValue }
    }
}

struct StudioCommands: Commands {
    @FocusedValue(\.textureWorkspace) private var workspace
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
        }
        CommandMenu("Material") {
            Button("Attach Depth Map…") { workspace?.chooseDepth() }
                .disabled(workspace?.source == nil || workspace?.isBusy == true)
            Button("Manage Local Models…") { workspace?.showModels = true }
            Button("Show Inspector") { workspace?.showInspector.toggle() }
                .keyboardShortcut("i", modifiers: [.command, .option])
        }
    }
}
