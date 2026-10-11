import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ModelLibraryView: View {
    @Bindable var models: ModelManager
    let adviser: OllamaDecisionService?
    let materialStore: WorkbenchStore
    var workspace: TextureWorkspace?
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismiss) private var dismiss
    @State private var pendingRemoval: LocalModelDescriptor?
    @State private var selectedModels = SelectedMaterialCheckpoint.readAll()
    @AppStorage(StudioPreferences.developerModeKey, store: StudioPreferences.defaults) private var developerMode = false

    init(models: ModelManager, materialStore: WorkbenchStore, adviser: OllamaDecisionService? = nil,
         workspace: TextureWorkspace? = nil) {
        self.models = models
        self.materialStore = materialStore
        self.adviser = adviser
        self.workspace = workspace
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Local Models").font(.title2.bold())
                    Text("Manage local inference models. Photos stay on this Mac.").foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    materialSection
                    if developerMode {
                        Text("Developer Core ML models").font(.headline)
                        ForEach(models.catalog) { descriptor in
                            ModelRow(models: models, descriptor: descriptor) { pendingRemoval = descriptor }
                            Divider()
                        }
                    }
                    if let adviser { OllamaModelControls(adviser: adviser) }
                }
            }
            if let error = models.lastError { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            Text("Remove deletes only downloads managed by Texture Studio. Located external models are unlinked; their files stay in place.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(24)
        .onAppear { models.refresh(); selectedModels = SelectedMaterialCheckpoint.readAll() }
        .onReceive(NotificationCenter.default.publisher(for: SelectedMaterialCheckpoint.changeNotification)) { _ in
            selectedModels = SelectedMaterialCheckpoint.readAll()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            selectedModels = SelectedMaterialCheckpoint.readAll()
        }
        .confirmationDialog("Remove this model?", isPresented: Binding(get: { pendingRemoval != nil }, set: { if !$0 { pendingRemoval = nil } }), titleVisibility: .visible) {
            Button("Remove Model", role: .destructive) {
                guard let descriptor = pendingRemoval else { return }
                do { try models.remove(id: descriptor.id) } catch { models.lastError = error.localizedDescription }
                pendingRemoval = nil
            }
            Button("Cancel", role: .cancel) { pendingRemoval = nil }
        } message: {
            Text("Managed downloads will be deleted. External model files will remain on disk.")
        }
    }

    @ViewBuilder
    private var materialSection: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("Material models").font(.headline)
            let active = workspace?.activeMaterialModels ?? selectedModels
            if !active.isEmpty {
                ForEach(["height", "roughness", "normal"].filter { active[$0] != nil }, id: \.self) { target in
                    if let checkpoint = active[target] {
                        Text("\(target.capitalized) · \(checkpoint.title)").font(.headline).textSelection(.enabled)
                        if let summary = checkpoint.modelSummary { Text(summary).font(.callout).foregroundStyle(.secondary) }
                        Text("SHA256 \(checkpoint.sha256.prefix(12))").font(.caption).foregroundStyle(.secondary)
                        Text(checkpoint.checkpointPath).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
            } else {
                Text("Locate a trained material safetensors model or its exported package, or select one in Model Training.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            HStack {
                Button("Locate Material Model…") { locateMaterialModel() }
                    .disabled(materialStore.isBusy)
                    .help("Use a trained height, roughness or normal safetensors model. Its files stay in place.")
                Button("Open Model Training…") { openWindow(id: "model-training") }
            }
            if materialStore.isBusy {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(materialStore.activity).font(.caption).foregroundStyle(.secondary)
                }
            } else if let error = materialStore.error {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
            }
        }
        Divider()
    }

    private func locateMaterialModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
        panel.message = "Choose a Texture Studio material .safetensors file or its exported package folder. Height, roughness and normal models use the native material runtime. The selected files stay in place."
        panel.prompt = "Use Material Model"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            materialStore.operation("Opening material model…") {
                let checkpoint = try await MaterialCheckpointImporter.activate(url, using: materialStore)
                models.lastError = nil
                models.refresh()
                selectedModels = SelectedMaterialCheckpoint.readAll()
                workspace?.reloadSelectedCheckpoint(activate: true, target: checkpoint.target)
            }
        }
    }
}

struct ModelRow: View {
    @Bindable var models: ModelManager
    let descriptor: LocalModelDescriptor
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text(descriptor.name).font(.headline)
            Text(descriptor.summary).font(.callout).foregroundStyle(.secondary)
            HStack {
                Link(descriptor.license, destination: descriptor.licenseURL)
                Link("Model source", destination: descriptor.sourceURL)
                if descriptor.downloadable {
                    Text(ByteCountFormatter.string(fromByteCount: descriptor.downloadBytes, countStyle: .file)).foregroundStyle(.secondary)
                }
            }
            .font(.caption)
            Text(models.status(for: descriptor.id).message).font(.caption).foregroundStyle(.secondary)
            if let progress = models.status(for: descriptor.id).progress {
                ProgressView(value: progress)
            }
            if let record = models.records[descriptor.id] {
                Text(record.path).font(.caption2).foregroundStyle(.secondary).lineLimit(2).textSelection(.enabled)
                if record.interface.outputs.count > 1 {
                    Picker("Depth output", selection: Binding(get: { record.selectedOutput ?? "" }, set: { name in
                        do { try models.chooseOutput(id: descriptor.id, name: name) }
                        catch { models.lastError = error.localizedDescription }
                    })) {
                        Text("Choose output…").tag("")
                        ForEach(record.interface.outputs, id: \.name) { output in
                            Text("\(output.name) · \(output.width) × \(output.height)").tag(output.name)
                        }
                    }
                }
            }
            HStack {
                if models.status(for: descriptor.id).isBusy {
                    Button("Cancel Download") { models.cancelDownload(id: descriptor.id) }
                } else {
                    if descriptor.downloadable && models.availableURL(for: descriptor.id) == nil {
                        Button("Download Model") { models.download(id: descriptor.id) }.buttonStyle(.borderedProminent)
                    }
                    Button("Locate Core ML Model…") { locateModel(descriptor: descriptor, models: models) }
                    if models.records[descriptor.id] != nil { Button("Remove…", role: .destructive, action: remove) }
                }
            }
        }
    }
}

struct MissingModelView: View {
    @Bindable var workspace: TextureWorkspace
    @Bindable var models: ModelManager
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Depth model unavailable", systemImage: "shippingbox").font(.title2.bold())
            Text("Locate or download the selected depth model, or continue with a flat surface. You can also attach your own height map.")
                .foregroundStyle(.secondary)
            if let descriptor = models.catalog.first(where: { $0.id == workspace.modelID }) {
                Text(models.status(for: descriptor.id).message).font(.caption)
                if let progress = models.status(for: descriptor.id).progress { ProgressView(value: progress) }
                if models.availableURL(for: descriptor.id) != nil {
                    Button("Use Model & Update Preview") { dismiss(); workspace.updatePreview(models: models) }
                        .buttonStyle(.glassProminent)
                } else if descriptor.downloadable && models.availableURL(for: descriptor.id) == nil {
                    Button("Download \(descriptor.name)") { models.download(id: descriptor.id) }
                        .buttonStyle(.glassProminent).disabled(models.status(for: descriptor.id).isBusy)
                }
                Button("Locate Existing Model…") { locateModel(descriptor: descriptor, models: models) }
                    .disabled(models.status(for: descriptor.id).isBusy)
                if models.status(for: descriptor.id).isBusy { Button("Cancel Download") { models.cancelDownload(id: descriptor.id) } }
            }
            Spacer()
            HStack {
            Button("Continue with Flat Surface") {
                    workspace.depthChoice = .photoDetail
                    dismiss()
                    workspace.updatePreview(models: models)
                }
                Spacer()
                Button("Close") { dismiss() }
            }
            Text("You can remove downloaded models later in Models. No photos are uploaded.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(26)
    }
}

@MainActor
private func locateModel(descriptor: LocalModelDescriptor, models: ModelManager) {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = true
    panel.allowsMultipleSelection = false
    panel.allowedContentTypes = ["mlpackage", "mlmodel", "mlmodelc"].compactMap { UTType(filenameExtension: $0) }
    panel.message = "Choose a compatible image-to-depth Core ML .mlpackage, .mlmodel or .mlmodelc. The selected model stays in place."
    panel.prompt = "Use Model"
    panel.begin { response in
        guard response == .OK, let url = panel.url else { return }
        Task {
            do { try await models.locate(id: descriptor.id, url: url) }
            catch { models.lastError = error.localizedDescription }
        }
    }
}
