import SwiftUI

struct NewMaterialDatasetSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var store: WorkbenchStore
    @State private var name = ""
    @State private var description = ""
    @State private var parentURL: URL?
    @State private var size = 1024
    @State private var validation = WorkbenchValidationSettings()
    @FocusState private var nameFocused: Bool

    private var cleanName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("New Dataset", systemImage: "folder.badge.plus").font(.title2.bold())
            Text("Choose the resolution first, then import and review the exact material crops and validation checks.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Form {
                TextField("Name", text: $name).focused($nameFocused)
                TextField("Description", text: $description, axis: .vertical).lineLimit(3...5)
                DatasetResolutionPicker(size: $size)
                DatasetValidationControls(settings: $validation)
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
                    store.createDataset(name: cleanName, description: description, parentURL: parentURL, size: size, validation: validation)
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
    @State private var validation: WorkbenchValidationSettings
    @FocusState private var nameFocused: Bool

    init(store: WorkbenchStore) {
        self.store = store
        _name = State(initialValue: store.datasetName)
        _description = State(initialValue: store.datasetDescription)
        _size = State(initialValue: store.datasetResolution)
        _validation = State(initialValue: store.dataset?.validation ?? .init())
    }

    private var cleanName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var hasChanges: Bool { cleanName != store.datasetName || description != store.datasetDescription || size != store.dataset?.trainingSize || validation != (store.dataset?.validation ?? .init()) }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Dataset Info", systemImage: "info.circle").font(.title2.bold())
            Text("Set the grid used for dataset rendering, checkpoint comparisons and training.").foregroundStyle(.secondary)
            Form {
                TextField("Name", text: $name).focused($nameFocused)
                TextField("Description", text: $description, axis: .vertical).lineLimit(3...5)
                DatasetResolutionPicker(size: $size)
                DatasetValidationControls(settings: $validation)
                if let plans = store.dataset?.resolutionPlans?[String(size)] {
                    DatasetPlanSummary(plans: plans, validation: validation)
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
                    store.updateDatasetInfo(name: cleanName, description: description, size: size, validation: validation)
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

struct DatasetValidationControls: View {
    @Binding var settings: WorkbenchValidationSettings
    var body: some View {
        Section("Validation · shared by all map targets") {
            Toggle("Generate separate validation crops", isOn: $settings.enabled)
            LabeledContent("Maximum subject percentage") {
                TextField("Percent", value: $settings.percent, format: .number).labelsHidden().frame(width: 65)
                Text("%")
            }.disabled(!settings.enabled)
            Stepper("Maximum validation crops: \(settings.maxCrops == 0 ? "percentage limit" : String(settings.maxCrops))",
                    value: $settings.maxCrops, in: 0...10000).disabled(!settings.enabled)
            Stepper("Quick checks: up to \(settings.quickCount) crops", value: $settings.quickCount, in: 1...100).disabled(!settings.enabled)
            Text("At most one different crop per subject folder. Prefer the unused 8K corner; another corner may overlap training pixels. All targets use the same folders and crops. Quick checks run during training; checkpoint saves and final exports check the entire configured pool. Zero maximum crops uses the percentage limit.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct DatasetPlanSummary: View {
    let plans: [String: WorkbenchDatasetPlan]
    var validation: WorkbenchValidationSettings? = nil
    private func limit(_ plan: WorkbenchDatasetPlan) -> Int {
        guard let validation else { return plan.validationLimit ?? 0 }
        guard validation.enabled, validation.percent.isFinite else { return 0 }
        let cap = Int((Double(plan.subjectCount ?? 0) * min(100, max(0, validation.percent)) / 100).rounded(.down))
        return validation.maxCrops > 0 ? min(cap, validation.maxCrops) : cap
    }
    private func count(_ plan: WorkbenchDatasetPlan) -> Int {
        guard let validation else { return plan.sharedValidationCount ?? plan.validationCount }
        let available = plan.subjects?.filter { $0.available && validation.folders[$0.id] != false }.count ?? plan.validationCandidateCount ?? 0
        return min(limit(plan), available)
    }
    var body: some View {
        if let plan = plans["height"] ?? plans.values.first {
            VStack(alignment: .leading, spacing: 9) {
                Text("Shared crop plan").font(.headline)
                Text("\(plan.subjectCount ?? plan.sourceSetCount) subject folders · \(plan.cropCount) training crops · \(count(plan)) extra validation crops")
                    .font(.callout).monospacedDigit()
                Text("Validation limit: \(limit(plan)) folders. \(plan.validationCandidateCount ?? 0) folders can supply a different crop at this size. Existing training crops stay in training.")
                    .font(.caption).foregroundStyle(.secondary)
                if (plan.undersizedSourceSetCount ?? 0) > 0 || plan.excludedCount > 0 {
                    Text("\(plan.undersizedSourceSetCount ?? 0) source sets are too small; \(plan.excludedCount) crops are excluded.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(["height", "roughness", "normal"], id: \.self) { target in
                    if let targetPlan = plans[target], targetPlan.unavailableTargetCount > 0 {
                        Text("\(target == "height" ? "Displacement" : target.capitalized): \(targetPlan.unavailableTargetCount) crops lack a supported target map.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                if count(plan) == 0 {
                    Text("No validation crops at these settings. Raise the percentage or use sources larger than the selected crop size.")
                        .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                }
                Text("These check learning on known materials. Review novel images after training to judge generalization.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(.vertical, 6)
        }
    }
}

struct ImportDatasetFolderSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var store: WorkbenchStore
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Import Material Folder", systemImage: "folder.badge.plus").font(.title2.bold())
            Text("Add to “\(store.datasetName)”").font(.headline)
            Text(store.folderImportURL?.path ?? "Choose a source folder").font(.caption)
                .foregroundStyle(.secondary).textSelection(.enabled)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    DatasetResolutionPicker(size: $store.folderImportSize).disabled(store.isBusy && !store.isScanningFolder)
                    if let preview = store.folderImport {
                        Text("Found \(preview.sourceSetCount) paired source sets · \(preview.addedMaterialCount) new · \(preview.duplicateMaterialCount) already present")
                            .font(.headline)
                        if let plans = preview.plans[String(store.folderImportSize)] { DatasetPlanSummary(plans: plans) }
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
                        Text("Scanning subfolders and verifying full-quality original maps. Large collections can take a minute. You can change resolution while this scan continues; no restart is needed.")
                            .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                    DatasetSheetOperationNotice(store: store)
                }.padding(4)
            }
            HStack {
                Button("Choose Another Folder…") { store.importMaterialFolder() }.disabled(store.isBusy)
                if store.error != nil || (!store.isBusy && store.folderImport == nil), let url = store.folderImportURL {
                    Button("Scan Again") { store.importMaterialFolder(url) }.disabled(store.isBusy)
                }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction).disabled(store.isBusy)
                Button("Import \(store.folderImport?.addedMaterialCount ?? 0) Sets") { store.commitFolderImport(size: store.folderImportSize) }
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
