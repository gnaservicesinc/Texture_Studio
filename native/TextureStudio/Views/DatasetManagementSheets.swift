import SwiftUI

struct NewMaterialDatasetSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var store: WorkbenchStore
    @State private var name = ""
    @State private var description = ""
    @State private var parentURL: URL?
    @State private var size = 1024
    @FocusState private var nameFocused: Bool

    private var cleanName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("New Dataset", systemImage: "folder.badge.plus").font(.title2.bold())
            Text("Choose the resolution first, then import and review the exact material crops and splits.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Form {
                TextField("Name", text: $name).focused($nameFocused)
                TextField("Description", text: $description, axis: .vertical).lineLimit(3...5)
                DatasetResolutionPicker(size: $size)
                if let folder = store.pendingSourceFolder {
                    LabeledContent("Source folder", value: folder.path)
                }
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
                Button("Cancel") { store.pendingSourceFolder = nil; dismiss() }.keyboardShortcut(.cancelAction).disabled(store.isBusy)
                Button("Create Dataset") {
                    guard let parentURL else { return }
                    store.createDataset(name: cleanName, description: description, parentURL: parentURL, size: size)
                }
                .buttonStyle(.glassProminent).keyboardShortcut(.defaultAction)
                .disabled(cleanName.isEmpty || parentURL == nil || store.isBusy)
            }
        }
        .padding(24).frame(width: 660, height: 600)
        .interactiveDismissDisabled(store.isBusy)
        .onDisappear { if !store.isBusy { store.pendingSourceFolder = nil } }
        .onAppear {
            if parentURL == nil {
                parentURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                        .appendingPathComponent("Texture Studio Datasets", isDirectory: true)
            }
            name = store.pendingSourceFolder?.lastPathComponent ?? name
            size = store.datasetResolution
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
    @State private var size: Int
    @FocusState private var nameFocused: Bool

    init(store: WorkbenchStore) {
        self.store = store
        _name = State(initialValue: store.datasetName)
        _description = State(initialValue: store.datasetDescription)
        _size = State(initialValue: store.datasetResolution)
    }

    private var cleanName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var hasChanges: Bool { cleanName != store.datasetName || description != store.datasetDescription || size != store.dataset?.trainingSize }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Dataset Info", systemImage: "info.circle").font(.title2.bold())
            Text("Set the grid used for dataset rendering, checkpoint comparisons and training.").foregroundStyle(.secondary)
            Form {
                TextField("Name", text: $name).focused($nameFocused)
                TextField("Description", text: $description, axis: .vertical).lineLimit(3...5)
                DatasetResolutionPicker(size: $size)
                if let plans = store.dataset?.resolutionPlans?[String(size)] {
                    DatasetPlanSummary(plans: plans)
                }
                if let dataset = store.dataset {
                    LabeledContent("Contents", value: "\(dataset.sourceSetCount ?? dataset.materials.count) original source sets")
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
                    store.updateDatasetInfo(name: cleanName, description: description, size: size)
                }
                .buttonStyle(.glassProminent).keyboardShortcut(.defaultAction)
                .disabled(cleanName.isEmpty || !hasChanges || store.isBusy)
            }
        }
        .padding(24).frame(width: 700, height: 700)
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
            Text("\(store.datasetName) · Rendering & training: \(store.datasetResolution) × \(store.datasetResolution)")
                .font(.callout.bold())
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

struct DatasetSheetOperationNotice: View {
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

struct DatasetResolutionPicker: View {
    @Binding var size: Int
    var body: some View {
        Picker("Rendering & training resolution", selection: $size) {
            ForEach([256, 512, 1024, 2048], id: \.self) { value in
                Text("\(value) × \(value)").tag(value)
            }
        }
        Text("Uses exact original pixels: one centered crop per source set, or three disjoint corner crops for sources at least 8192 × 8192. Smaller sources remain available for inspection but cannot train at this size. No resizing or padding is applied. Final material export has its own size setting.")
            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}

struct DatasetPlanSummary: View {
    let plans: [String: WorkbenchDatasetPlan]
    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("Crops and splits").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 7) {
                GridRow {
                    Text("Target"); Text("Training"); Text("Validation"); Text("No target")
                }.font(.caption.bold()).foregroundStyle(.secondary)
                ForEach(["height", "roughness", "normal"], id: \.self) { target in
                    if let plan = plans[target] {
                        GridRow {
                            Text(target == "height" ? "Displacement" : target.capitalized)
                            Text(plan.trainCount.formatted())
                            Text(plan.validationCount.formatted())
                            Text(plan.unavailableTargetCount.formatted())
                        }
                    }
                }
            }.font(.callout).monospacedDigit()
            if let plan = plans["height"] {
                Text("\(plan.sourceSetCount) source sets supply \(plan.cropCount) native crops. \(plan.undersizedSourceSetCount ?? 0) source sets are too small; \(plan.excludedCount) crops are excluded.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Automatic validation holds out source families (about 5%). Resolution and color siblings stay together. A lone 8K set can use disjoint regions. You can edit splits after import.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if plans.values.contains(where: { $0.trainCount > 0 && $0.validationCount == 0 }) {
                Text("Some targets have no independent validation data. Add another material family or an eligible 8K source set for a validation check.")
                    .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
        }.padding(.vertical, 6)
    }
}

struct ImportDatasetFolderSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var store: WorkbenchStore
    @State private var size: Int
    init(store: WorkbenchStore) {
        self.store = store
        _size = State(initialValue: store.datasetResolution)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Import Material Folder", systemImage: "folder.badge.plus").font(.title2.bold())
            Text("Add to “\(store.datasetName)”").font(.headline)
            Text(store.folderImportURL?.path ?? "Choose a source folder").font(.caption)
                .foregroundStyle(.secondary).textSelection(.enabled)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    DatasetResolutionPicker(size: $size).disabled(store.isBusy)
                    if let preview = store.folderImport {
                        Text("Found \(preview.sourceSetCount) paired source sets · \(preview.addedMaterialCount) new · \(preview.duplicateMaterialCount) already present")
                            .font(.headline)
                        if let plans = preview.plans[String(size)] { DatasetPlanSummary(plans: plans) }
                        if !preview.warnings.isEmpty {
                            DisclosureGroup("\(preview.warnings.count) source notices") {
                                ForEach(Array(preview.warnings.enumerated()), id: \.offset) { _, warning in
                                    Text(warning).font(.caption).frame(maxWidth: .infinity, alignment: .leading)
                                        .textSelection(.enabled).padding(.vertical, 3)
                                }
                            }
                        }
                        if preview.addedMaterialCount == 0 {
                            Text(preview.duplicateMaterialCount > 0 ? "These materials are already in this dataset." : "No importable pairs were found. Choose another folder or use Add Materials to assign maps manually.")
                                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        }
                    } else if store.isBusy {
                        Text("Scanning subfolders and verifying full-quality original maps. Large collections can take a minute. Resolution changes after the scan are immediate.")
                            .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                    DatasetSheetOperationNotice(store: store)
                }.padding(4)
            }
            HStack {
                Button("Choose Another Folder…") { store.importMaterialFolder() }.disabled(store.isBusy)
                if store.error != nil, let url = store.folderImportURL {
                    Button("Scan Again") { store.importMaterialFolder(url) }.disabled(store.isBusy)
                }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction).disabled(store.isBusy)
                Button("Import \(store.folderImport?.addedMaterialCount ?? 0) Sets") { store.commitFolderImport(size: size) }
                    .buttonStyle(.glassProminent).keyboardShortcut(.defaultAction)
                    .disabled(store.isBusy || (store.folderImport?.addedMaterialCount ?? 0) == 0)
            }
        }.padding(24).frame(width: 780, height: 650)
            .interactiveDismissDisabled(store.isBusy)
            .onDisappear { store.clearFolderImport() }
    }
}

struct DatasetManagementPresentation: ViewModifier {
    @Bindable var store: WorkbenchStore
    func body(content: Content) -> some View {
        content.sheet(item: $store.datasetSheet) { route in
            switch route {
            case .new: NewMaterialDatasetSheet(store: store)
            case .add: AddDatasetMaterialSheet(store: store)
            case .info: MaterialDatasetInfoSheet(store: store)
            case .folder: ImportDatasetFolderSheet(store: store)
            }
        }
    }
}
