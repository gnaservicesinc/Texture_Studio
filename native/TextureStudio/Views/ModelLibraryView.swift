import SwiftUI
import AppKit

struct ModelLibraryView: View {
    @Bindable var models: ModelManager
    let adviser: OllamaDecisionService?
    let runtime: PythonDepthService?
    @Environment(\.dismiss) private var dismiss
    @State private var pendingRemoval: LocalModelDescriptor?

    init(models: ModelManager, adviser: OllamaDecisionService? = nil, runtime: PythonDepthService? = nil) {
        self.models = models
        self.adviser = adviser
        self.runtime = runtime
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Local Models").font(.title2.bold())
                    Text("Optional enhancement. Photos stay on this Mac.").foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    if let runtime {
                        PythonRuntimeControls(runtime: runtime)
                        Divider()
                    }
                    ForEach(models.catalog) { descriptor in
                        ModelRow(models: models, descriptor: descriptor) { pendingRemoval = descriptor }
                        Divider()
                    }
                    if let adviser { OllamaModelControls(adviser: adviser) }
                }
            }
            if let error = models.lastError { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            Text("Remove deletes only downloads managed by Texture Studio. External models and Python environments are unlinked and their files stay in place.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(24)
        .onAppear { models.refresh() }
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
                    Button("Locate Model…") { locateModel(descriptor: descriptor, models: models) }
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
            Text("Locate or download DA3-GIANT-1.1, or continue with a flat surface. Portrait depth and the retired small V2 model are not used for material relief.")
                .foregroundStyle(.secondary)
            if let descriptor = models.catalog.first(where: { $0.id == workspace.modelID }) {
                Text(models.status(for: descriptor.id).message).font(.caption)
                if let progress = models.status(for: descriptor.id).progress { ProgressView(value: progress) }
                if descriptor.backend == .pytorchDA3 && !workspace.pythonDepthService.runtimeStatus.isReady {
                    Text(workspace.pythonDepthService.runtimeStatus.message).font(.caption)
                    Button("Set Up PyTorch Runtime…") { dismiss(); workspace.showModels = true }
                }
                if models.availableURL(for: descriptor.id) != nil &&
                    (descriptor.backend == .coreML || workspace.pythonDepthService.runtimeStatus.isReady) {
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
    panel.message = descriptor.backend == .pytorchDA3
        ? "Choose the DA3-GIANT-1.1 folder containing config.json and model.safetensors. The selected folder is validated and stays in place."
        : "Choose a compatible image-to-depth Core ML .mlpackage, .mlmodel or .mlmodelc. DA3 PyTorch models use their own backend."
    panel.prompt = "Use Model"
    panel.begin { response in
        guard response == .OK, let url = panel.url else { return }
        Task {
            do { try await models.locate(id: descriptor.id, url: url) }
            catch { models.lastError = error.localizedDescription }
        }
    }
}
