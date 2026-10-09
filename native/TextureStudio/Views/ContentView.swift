import SwiftUI

struct ContentView: View {
    @Environment(\.openWindow) private var openWindow
    let models: ModelManager
    let adviser: OllamaDecisionService
    @State private var workspace: TextureWorkspace

    init(models: ModelManager, adviser: OllamaDecisionService) {
        self.models = models
        self.adviser = adviser
        self._workspace = State(initialValue: TextureWorkspace(preferences: StudioPreferences.defaults))
    }

    var body: some View {
        NavigationSplitView {
            StudioSidebar(workspace: workspace, models: models)
                .navigationSplitViewColumnWidth(min: 170, ideal: 205, max: 245)
        } detail: {
            MaterialCanvas(workspace: workspace, models: models)
        }
        .inspector(isPresented: $workspace.showInspector) {
            MaterialInspector(workspace: workspace, models: models)
                .inspectorColumnWidth(min: 300, ideal: 330, max: 400)
        }
        .focusedSceneValue(\.textureWorkspace, workspace)
        .focusedSceneValue(\.textureModels, models)
        .navigationTitle(workspace.source?.url.lastPathComponent ?? "Texture Studio")
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button { workspace.choosePhoto() } label: { Label("Import Photo", systemImage: "photo.badge.plus") }
                    .help("Import a surface photo")
                    .disabled(workspace.isBusy)
                Button { workspace.chooseRecipe() } label: { Label("Open Recipe", systemImage: "folder") }
                    .help("Restore a saved photo, material settings and model selection.")
                    .disabled(workspace.isBusy)
            }
            ToolbarSpacer(.flexible)
            ToolbarItemGroup(placement: .primaryAction) {
                Button { workspace.updatePreview(models: models) } label: {
                    Label(workspace.materialNeedsUpdate ? (workspace.hasEdits ? "Update Material" : "Generate Material") : "Material Is Current",
                          systemImage: workspace.materialNeedsUpdate ? "arrow.trianglehead.2.clockwise" : "checkmark.circle")
                }
                    .help("Generate all four maps at the selected output size. Full Quality and export reuse the finished maps until you change a setting.")
                    .disabled(workspace.source == nil || workspace.isBusy || !workspace.materialNeedsUpdate)
                Button { workspace.chooseExport(models: models) } label: { Label("Export Material…", systemImage: "square.and.arrow.up") }
                    .labelStyle(.titleAndIcon)
                    .help("Save diffuse PNG, roughness, OpenGL normal and displacement EXRs, plus a Blender setup script, in a material folder. ⌘E")
                    .disabled(workspace.source == nil || workspace.isBusy)
            }
            ToolbarSpacer(.fixed)
            ToolbarItemGroup(placement: .automatic) {
                Button { openWindow(id: "model-training") } label: { Label("Model Training", systemImage: "graduationcap") }
                    .help("Prepare crops, refine a model, compare checkpoints and inspect full-resolution material maps.")
                Button { workspace.showAdvice = true } label: { Label("Review Photo", systemImage: "eye") }
                    .help("Review bounded suggestions from local Clef")
                Button { workspace.showModels = true } label: { Label("Local Models", systemImage: "shippingbox") }
                    .help("See the active material model, locate model files and manage local runtimes.")
                Button { workspace.showInspector.toggle() } label: { Label("Inspector", systemImage: "sidebar.right") }
                    .help("Show or hide photo, material and export settings. ⌥⌘I")
            }
        }
        .sheet(isPresented: $workspace.showModels) {
            ModelLibraryView(models: models, adviser: adviser, workspace: workspace)
                .frame(width: 660, height: 570)
        }
        .sheet(isPresented: $workspace.showModelRecovery) {
            MissingModelView(workspace: workspace, models: models)
                .frame(width: 530, height: 400)
        }
        .sheet(isPresented: $workspace.showAdvice) {
            AdviceReviewView(workspace: workspace, adviser: adviser, models: models)
                .frame(width: 600, height: 660)
        }
        .alert(item: $workspace.notice) { notice in
            Alert(title: Text(notice.title), message: Text(notice.message), dismissButton: .default(Text("OK")))
        }
        .confirmationDialog("Large texture export", isPresented: $workspace.showMemoryWarning, titleVisibility: .visible) {
            Button("Export \(workspace.settings.outputSize) × \(workspace.settings.outputSize)") {
                workspace.chooseExport(models: models, memoryApproved: true)
            }
            Button("Use 2048 × 2048") {
                workspace.settings.outputSize = 2048
                workspace.chooseExport(models: models)
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Large float maps can use several gigabytes of unified memory. Inference capacity depends on the selected model. Export is refused if the available memory budget is too low.")
        }
        .onOpenURL { url in
            if url.pathExtension.lowercased() == "json" { workspace.openRecipe(url) }
            else { workspace.importPhoto(url) }
        }
        .task {
            models.refresh()
            workspace.reloadSelectedCheckpoint()
            if CommandLine.arguments.contains("--open-training") { openWindow(id: "model-training") }
            let args = CommandLine.arguments
            if let index = args.firstIndex(of: "--open"), index + 1 < args.count {
                workspace.importPhoto(URL(fileURLWithPath: args[index + 1]))
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            workspace.reloadSelectedCheckpoint()
        }
    }
}

struct StudioSidebar: View {
    @Environment(\.openWindow) private var openWindow
    @Bindable var workspace: TextureWorkspace
    let models: ModelManager

    var body: some View {
        ScrollView {
          VStack(alignment: .leading, spacing: 4) {
            Section("Surface material") {
                ForEach(MaterialPreview.allCases) { preview in
                    MaterialSidebarRow(selected: workspace.selectedPreview == preview, action: { workspace.selectPreview(preview) }) {
                        Label(preview.rawValue, systemImage: preview.symbol)
                    }
                    .disabled(workspace.source == nil || workspace.isBusy)
                }
            }
            Section("Project") {
                Button { workspace.chooseExport(models: models) } label: { Label("Export Material…", systemImage: "square.and.arrow.up") }
                    .help("Save all four final maps and a Blender setup script. ⌘E")
                    .disabled(workspace.source == nil || workspace.isBusy)
                Button { openWindow(id: "model-training") } label: { Label("Model Training", systemImage: "graduationcap") }
                    .help("Open the guided workspace for data preparation, training, comparison and model export.")
                Button { workspace.saveRecipe() } label: { Label("Save Recipe…", systemImage: "doc.badge.arrow.up") }
                    .disabled(workspace.source == nil || workspace.isBusy)
                Button { workspace.showModels = true } label: { Label("Local Models", systemImage: "shippingbox") }
            }
          }.padding(10)
        }
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Texture Studio").font(.headline)
                Text("Photo → Cycles material").font(.caption).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
        }
    }
}
