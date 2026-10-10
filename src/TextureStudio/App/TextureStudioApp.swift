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
                    VStack(alignment: .leading, spacing: 14) {
                        PreparationSettingsView(store: workbench)
                        Toggle("Developer mode", isOn: $developerMode)
                            .help("Expose advanced model controls. Export fused full checkpoints alongside the separate LoRA, with Hugging Face publishing controls.")
                    }.padding(20)
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
        // App-hosted unit tests drive their own lifecycle. Bringing their host
        // forward can redirect a user's Quit command into an asynchronous test.
        if NSClassFromString("XCTestCase") != nil {
            if ProcessInfo.processInfo.environment["TEXTURE_STUDIO_CONTROL_UI_VERIFY"] != "1" {
                NSApp.setActivationPolicy(.prohibited)
                NSApp.windows.forEach { $0.orderOut(nil) }
            }
            return
        }
        if let index = CommandLine.arguments.firstIndex(of: "--export-auxiliary") {
            Task {
                do {
                    let arguments = CommandLine.arguments
                    guard arguments.indices.contains(index + 1), let outputIndex = arguments.firstIndex(of: "--output"),
                          arguments.indices.contains(outputIndex + 1) else {
                        throw StudioError("Usage: Texture Studio --export-auxiliary SOURCE --output NEW_FOLDER")
                    }
                    let report = try await NativeAuxiliaryExporter.export(sourceURL: URL(fileURLWithPath: arguments[index + 1]),
                        to: URL(fileURLWithPath: arguments[outputIndex + 1]))
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
                    FileHandle.standardOutput.write(try encoder.encode(report))
                    FileHandle.standardOutput.write(Data([10]))
                    exit(0)
                } catch {
                    fputs("Auxiliary export failed: \(error)\n", stderr)
                    exit(1)
                }
            }
            return
        }
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if CommandLine.arguments.contains("--smoke-test") {
            Task {
                do {
                    try await StudioSmoke.run()
                    print("Texture Studio native smoke passed")
                    // Exit the command-line check without waiting for AppKit's quit flow.
                    exit(0)
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
private struct MaterialDatasetFocusKey: FocusedValueKey { typealias Value = WorkbenchStore }
extension FocusedValues {
    var textureWorkspace: TextureWorkspace? {
        get { self[WorkspaceFocusKey.self] }
        set { self[WorkspaceFocusKey.self] = newValue }
    }
    var textureModels: ModelManager? {
        get { self[StudioModelsFocusKey.self] }
        set { self[StudioModelsFocusKey.self] = newValue }
    }
    var materialDatasetStore: WorkbenchStore? {
        get { self[MaterialDatasetFocusKey.self] }
        set { self[MaterialDatasetFocusKey.self] = newValue }
    }
}

struct StudioCommands: Commands {
    @FocusedValue(\.textureWorkspace) private var workspace
    @FocusedValue(\.textureModels) private var models
    @FocusedValue(\.materialDatasetStore) private var datasetStore
    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            if let datasetStore {
                Button("New Dataset…") { datasetStore.showNewDatasetSheet = true }
                    .keyboardShortcut("n")
                    .disabled(datasetStore.isBusy)
                Button("Open Dataset…") { datasetStore.chooseDataset() }
                    .keyboardShortcut("o")
                    .disabled(datasetStore.isBusy)
            } else {
                Button("Import Photo…") { workspace?.choosePhoto() }
                    .keyboardShortcut("o")
                    .disabled(workspace?.isBusy ?? true)
                Button("Open Texture Recipe…") { workspace?.chooseRecipe() }
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                    .disabled(workspace?.isBusy ?? true)
            }
        }
        CommandGroup(replacing: .saveItem) {
            if datasetStore == nil {
                Button("Save Texture Recipe…") { workspace?.saveRecipe() }
                    .keyboardShortcut("s")
                    .disabled(workspace?.source == nil || workspace?.isBusy == true)
                Button("Export Material…") {
                    if let workspace, let models { workspace.chooseExport(models: models) }
                }
                    .keyboardShortcut("e")
                    .disabled(workspace?.source == nil || workspace?.isBusy == true || models == nil)
                Button("Export Original Auxiliary Data…") { workspace?.chooseAuxiliaryExport() }
                    .disabled(workspace?.source == nil || workspace?.isBusy == true)
            }
        }
        CommandMenu("Dataset") {
            Button("Add Material…") { datasetStore?.showAddMaterialSheet = true }
                .keyboardShortcut("a", modifiers: [.command, .shift])
                .disabled(datasetStore?.dataset == nil || datasetStore?.isBusy == true)
            Button("Import Material Folder…") { datasetStore?.importMaterialFolder() }
                .disabled(datasetStore?.dataset == nil || datasetStore?.isBusy == true)
            Divider()
            Button("Rename / Edit Dataset Info…") { datasetStore?.showDatasetInfoSheet = true }
                .keyboardShortcut("i")
                .disabled(datasetStore?.dataset == nil || datasetStore?.isBusy == true)
            Button("Show Dataset in Finder") { datasetStore?.revealDataset() }
                .disabled(datasetStore?.dataset == nil)
            Button("Close Dataset") { datasetStore?.closeDataset() }
                .disabled(datasetStore?.dataset == nil || datasetStore?.isBusy == true)
            Divider()
            Button("Move Dataset to Trash…", role: .destructive) { datasetStore?.showTrashDatasetConfirmation = true }
                .disabled(datasetStore?.dataset == nil || datasetStore?.isBusy == true)
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
