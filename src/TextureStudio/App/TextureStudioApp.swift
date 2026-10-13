import SwiftUI
import AppKit

@main
struct TextureStudioApp: App {
    @NSApplicationDelegateAdaptor(StudioAppDelegate.self) private var delegate
    @State private var adviser = OllamaDecisionService()
    @State private var textureLoom = TextureLoomStore()
    var body: some Scene {
        WindowGroup("Texture Studio") {
            ContentView(adviser: adviser, textureLoom: textureLoom)
                .frame(minWidth: 980, minHeight: 680)
        }
        .defaultSize(width: 1320, height: 900)
        .commands { StudioCommands() }
        Settings {
            ModelLibraryView(textureLoom: textureLoom, adviser: adviser)
                .frame(width: 640, height: 540)
        }
    }
}

final class StudioAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationDidFinishLaunching(_ notification: Notification) {
        if NSClassFromString("XCTestCase") != nil {
            NSApp.setActivationPolicy(.prohibited); NSApp.windows.forEach { $0.orderOut(nil) }; return
        }
        if CommandLine.arguments.contains("--smoke-test") {
            NSApp.setActivationPolicy(.prohibited)
            Task {
                do { try await StudioSmoke.run(); print("Texture Studio native smoke passed"); exit(0) }
                catch { fputs("Texture Studio smoke failed: \(error)\n", stderr); exit(1) }
            }
            return
        }
        // Retired trainer/data commands must fail explicitly rather than launch any old backend.
        if CommandLine.arguments.contains(where: { ["--material-dataset", "--refresh-classifier-dataset", "--refresh-specialists", "--specialists", "--material-tool", "--open-training"].contains($0) }) {
            fputs("Model development is provided by Loom Factory. Texture Studio has no trainer or dataset commands.\n", stderr); exit(2)
        }
        NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true)
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
            Button("Import Photo…") { workspace?.choosePhoto() }.keyboardShortcut("o").disabled(workspace?.isBusy ?? true)
            Button("Open Texture Recipe…") { workspace?.chooseRecipe() }.keyboardShortcut("o", modifiers: [.command, .shift]).disabled(workspace?.isBusy ?? true)
        }
        CommandGroup(replacing: .saveItem) {
            Button("Save Texture Recipe…") { workspace?.saveRecipe() }.keyboardShortcut("s").disabled(workspace?.source == nil || workspace?.isBusy == true)
            Button("Export Material…") { workspace?.chooseExport() }.keyboardShortcut("e").disabled(workspace?.source == nil || workspace?.isBusy == true)
            Button("Export Original Auxiliary Data…") { workspace?.chooseAuxiliaryExport() }.disabled(workspace?.source == nil || workspace?.isBusy == true)
        }
        CommandMenu("Material") {
            Button("Generate / Update Material") { workspace?.updatePreview() }.keyboardShortcut("r").disabled(workspace?.source == nil || workspace?.isBusy == true || workspace?.materialNeedsUpdate != true)
            Button("Open Full Quality") { workspace?.inspectFullQuality() }.keyboardShortcut("f", modifiers: [.command, .shift]).disabled(workspace?.fullQualityAvailable != true || workspace?.isBusy == true)
            Divider()
            Button("Attach Height / Depth Map…") { workspace?.chooseDepth() }.disabled(workspace?.source == nil || workspace?.isBusy == true)
            Button("Manage LoRAs…") { workspace?.showModels = true }
            Button("Show Inspector") { workspace?.showInspector.toggle() }.keyboardShortcut("i", modifiers: [.command, .option])
        }
    }
}
