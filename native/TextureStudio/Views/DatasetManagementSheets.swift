import SwiftUI

struct NewMaterialDatasetSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var store: WorkbenchStore
    @State private var name = ""
    @State private var description = ""
    @State private var parentURL: URL?
    @FocusState private var nameFocused: Bool

    private var cleanName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("New Dataset", systemImage: "folder.badge.plus").font(.title2.bold())
            Text("Give your dataset a name, then add the material maps you want to review or train with.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Form {
                TextField("Name", text: $name).focused($nameFocused)
                TextField("Description", text: $description, axis: .vertical).lineLimit(3...5)
                LabeledContent("Save in") {
                    HStack {
                        Text(parentURL?.path ?? "Choose a folder")
                            .lineLimit(2).truncationMode(.middle).textSelection(.enabled)
                            .foregroundStyle(parentURL == nil ? .secondary : .primary)
                        Spacer()
                        Button("Choose…") { store.chooseDatasetParent(selected: parentURL) { parentURL = $0 } }
                    }
                }
            }.formStyle(.grouped).disabled(store.isBusy)
            Text("A new folder will hold the dataset information. Original map files stay where they are.")
                .font(.caption).foregroundStyle(.secondary)
            DatasetSheetOperationNotice(store: store)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction).disabled(store.isBusy)
                Button("Create Dataset") {
                    guard let parentURL else { return }
                    store.createDataset(name: cleanName, description: description, parentURL: parentURL)
                }
                .buttonStyle(.glassProminent).keyboardShortcut(.defaultAction)
                .disabled(cleanName.isEmpty || parentURL == nil || store.isBusy)
            }
        }
        .padding(24).frame(width: 570)
        .interactiveDismissDisabled(store.isBusy)
        .onAppear {
            if parentURL == nil {
                parentURL = store.datasetFolderURL?.deletingLastPathComponent()
                    ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                        .appendingPathComponent("Texture Studio Datasets", isDirectory: true)
            }
            store.error = nil
            nameFocused = true
        }
    }
}

struct MaterialDatasetInfoSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var store: WorkbenchStore
    @State private var name: String
    @State private var description: String
    @FocusState private var nameFocused: Bool

    init(store: WorkbenchStore) {
        self.store = store
        _name = State(initialValue: store.datasetName)
        _description = State(initialValue: store.datasetDescription)
    }

    private var cleanName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var hasChanges: Bool { cleanName != store.datasetName || description != store.datasetDescription }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Dataset Info", systemImage: "info.circle").font(.title2.bold())
            Text("Rename your dataset or edit its description.").foregroundStyle(.secondary)
            Form {
                TextField("Name", text: $name).focused($nameFocused)
                TextField("Description", text: $description, axis: .vertical).lineLimit(3...5)
                if let dataset = store.dataset {
                    LabeledContent("Contents", value: "\(dataset.materialCount ?? dataset.materials.count) materials · \(dataset.sampleCount ?? dataset.samples.count) map sets")
                }
                if let url = store.datasetFolderURL {
                    LabeledContent("Folder") {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(url.path).font(.caption).textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                            Button("Show in Finder") { store.revealDataset() }
                        }
                    }
                }
            }.formStyle(.grouped).disabled(store.isBusy)
            DatasetSheetOperationNotice(store: store)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction).disabled(store.isBusy)
                Button("Save Changes") {
                    store.updateDatasetInfo(name: cleanName, description: description)
                }
                .buttonStyle(.glassProminent).keyboardShortcut(.defaultAction)
                .disabled(cleanName.isEmpty || !hasChanges || store.isBusy)
            }
        }
        .padding(24).frame(width: 570)
        .interactiveDismissDisabled(store.isBusy)
        .onAppear { store.error = nil; nameFocused = true }
    }
}

struct AddDatasetMaterialSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var store: WorkbenchStore
    @State private var name = ""
    @State private var inputURL: URL?
    @State private var heightURL: URL?
    @State private var roughnessURL: URL?
    @State private var normalURL: URL?
    @State private var normalConvention = "opengl"
    @FocusState private var nameFocused: Bool

    private var cleanName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var canAdd: Bool {
        !cleanName.isEmpty && inputURL != nil && (heightURL != nil || roughnessURL != nil || normalURL != nil) && !store.isBusy
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Add Materials", systemImage: "photo.badge.plus").font(.title2.bold())
            Text("Add a named material with a diffuse map and at least one surface map. Choose original PNG maps with matching dimensions; the dataset references their full-quality files.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Form {
                TextField("Material name", text: $name).focused($nameFocused)
                mapRow("Diffuse", url: $inputURL, required: true)
                mapRow("Displacement", url: $heightURL)
                mapRow("Roughness", url: $roughnessURL)
                mapRow("Normal", url: $normalURL)
                if normalURL != nil {
                    Picker("Normal convention", selection: $normalConvention) {
                        Text("OpenGL (+Y)").tag("opengl")
                        Text("DirectX (−Y)").tag("directx")
                    }
                }
            }.formStyle(.grouped).disabled(store.isBusy)
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "folder").foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 5) {
                    Text("Have a folder of material maps?").font(.callout.bold())
                    Text("Import matching diffuse, displacement, roughness and normal maps together.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Import Folder…") { store.importMaterialFolder() }
                    .disabled(store.isBusy)
            }
            DatasetSheetOperationNotice(store: store)
            Divider()
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction).disabled(store.isBusy)
                Button("Add Material") {
                    guard let inputURL else { return }
                    store.addMaterial(name: cleanName, input: inputURL, height: heightURL, roughness: roughnessURL,
                                      normal: normalURL, normalConvention: normalConvention)
                }
                .buttonStyle(.glassProminent).keyboardShortcut(.defaultAction).disabled(!canAdd)
            }
        }
        .padding(24).frame(width: 640)
        .interactiveDismissDisabled(store.isBusy)
        .onAppear { store.error = nil; nameFocused = true }
    }

    private func mapRow(_ title: String, url: Binding<URL?>, required: Bool = false) -> some View {
        LabeledContent(required ? "\(title) (required)" : title) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(url.wrappedValue?.lastPathComponent ?? "Choose a PNG map")
                        .lineLimit(1).truncationMode(.middle)
                        .foregroundStyle(url.wrappedValue == nil ? .secondary : .primary)
                    if let selectedURL = url.wrappedValue {
                        Text(selectedURL.deletingLastPathComponent().path)
                            .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    }
                }
                Spacer()
                if url.wrappedValue != nil {
                    Button { url.wrappedValue = nil } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain).foregroundStyle(.secondary).help("Remove \(title.lowercased()) map")
                }
                Button("Choose…") {
                    store.chooseMaterialMap(title: "Choose \(title.lowercased()) PNG map", selected: url.wrappedValue) { selected in
                        url.wrappedValue = selected
                        if required && cleanName.isEmpty { name = selected.deletingPathExtension().lastPathComponent }
                    }
                }
            }
        }
    }
}

private struct DatasetSheetOperationNotice: View {
    let store: WorkbenchStore

    var body: some View {
        if let error = store.error {
            Label(error, systemImage: "exclamationmark.circle")
                .font(.callout).foregroundStyle(.red)
                .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
                .accessibilityLabel("Dataset operation failed: \(error)")
        }
        if store.isBusy {
            HStack(spacing: 9) {
                ProgressView().controlSize(.small)
                Text(store.activity).font(.callout).foregroundStyle(.secondary)
                Spacer()
                WorkbenchStopButtons(store: store)
            }
        }
    }
}
