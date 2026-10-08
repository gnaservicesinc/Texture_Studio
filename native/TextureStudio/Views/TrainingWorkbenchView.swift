import SwiftUI
import AppKit

struct TrainingWorkbenchView: View {
    @Bindable var store: WorkbenchStore
    @State private var showCheckpoints = false
    @State private var followLog = true

    private var selectedMaterial: WorkbenchMaterial? {
        store.dataset?.materials.first { $0.id == store.selectedMaterialId }
    }
    private var resourceIssue: String? {
        if !store.training.memoryGB.isFinite || !(2...30).contains(store.training.memoryGB) {
            return "Choose a memory cap between 2 and 30 GB."
        }
        if !store.training.maxMinutes.isFinite || !(1...240).contains(store.training.maxMinutes) {
            return "Choose a time limit between 1 and 240 minutes."
        }
        if !(1...10_000).contains(store.training.updatesPerCrop) {
            return "Choose between 1 and 10,000 updates per crop."
        }
        return nil
    }
    private var trainingIssue: String? {
        if let resourceIssue { return resourceIssue }
        guard let dataset = store.dataset else { return "Open a prepared material dataset." }
        let candidates = store.training.useSelectedMaterialOnly ? selectedMaterial?.samples ?? [] : dataset.samples
        let eligible = candidates.filter { sample in
            sample.status == "approved" || (store.training.allowUnreviewed && ["prepared", "unreviewed"].contains(sample.status))
        }
        if eligible.isEmpty { return "Select eligible crops or approve the prepared crops first." }
        if eligible.contains(where: { $0.width != store.training.size || $0.height != store.training.size }) {
            return "Choose \(store.training.size) × \(store.training.size) native crops. Training does not resize source targets."
        }
        if eligible.contains(where: { $0.maps[store.training.target] == nil }) {
            return "Every selected crop needs a \(store.training.target) map."
        }
        if !eligible.contains(where: { $0.split == "train" }) || !eligible.contains(where: { $0.split == "validation" }) {
            return "The selection needs training and validation crops."
        }
        return nil
    }

    var body: some View {
        HSplitView {
            VStack(spacing: 0) {
                HStack {
                    Text("Training setup").font(.headline)
                    Spacer()
                    Button("Open Dataset…") { store.chooseDataset() }.disabled(store.isBusy)
                }
                .padding(14)
                TrainingConfigurationForm(options: $store.training, dataset: store.dataset,
                                          material: selectedMaterial, checkpoint: store.selectedCheckpoint,
                                          checkpoints: store.checkpoints, selectedCheckpointId: $store.selectedCheckpointId,
                                          selectedSampleId: $store.selectedSampleId,
                                          disabled: store.isBusy)
                Divider()
                WorkbenchRuntimeControls(store: store)
                    .padding(14)
            }
            .frame(minWidth: 355, idealWidth: 400, maxWidth: 475)
            VStack(spacing: 0) {
                runControls
                Divider()
                liveLog
                Divider()
                runFooter
            }
            .frame(minWidth: 400)
        }
        .sheet(isPresented: $showCheckpoints) {
            CheckpointLibraryView(store: store)
                .frame(minWidth: 780, minHeight: 560)
        }
        .onChange(of: store.dataset?.indexSha256) { _, _ in
            let sizes = Set(store.samples.filter { $0.width == $0.height }.map(\.width))
            if sizes.count == 1, let size = sizes.first, [1024, 2048].contains(size) {
                store.training.size = size
            }
        }
    }

    private var runControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.isTraining ? "Training in progress" : "Material training").font(.title3.bold())
                    Text("Final saved checkpoint · native \(store.training.size) × \(store.training.size)")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button { showCheckpoints = true } label: { Label("Checkpoints", systemImage: "shippingbox") }
            }
            HStack {
                if store.isTraining {
                    ProgressView().controlSize(.small)
                    Button(store.isStopping ? "Saving checkpoint…" : "Stop & Save") { store.stop() }
                        .disabled(store.isStopping)
                } else {
                    Button("Start Training") { store.startTraining() }
                        .buttonStyle(.glassProminent)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(store.isBusy || trainingIssue != nil)
                    Button("Resume Saved Run…") { store.chooseResumeCheckpoint() }
                        .disabled(store.isBusy || resourceIssue != nil)
                }
                Spacer()
            }
            if !store.isBusy, let issue = trainingIssue {
                Label(issue, systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Resume restores saved optimizer state and source selections. Set updates per crop to the desired total before resuming.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(16)
    }

    private var liveLog: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Live log").font(.headline)
                Spacer()
                Toggle("Follow", isOn: $followLog).toggleStyle(.checkbox)
            }
            .padding(12)
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(store.logText.isEmpty ? "Training events and saved-checkpoint updates will appear here." : store.logText)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Color.clear.frame(height: 1).id("log-end")
                    }
                    .padding(14)
                }
                .onChange(of: store.logText) { _, _ in
                    if followLog { proxy.scrollTo("log-end", anchor: .bottom) }
                }
            }
        }
        .frame(maxHeight: .infinity)
    }

    private var runFooter: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !store.activity.isEmpty {
                Text(store.activity).font(.caption).foregroundStyle(.secondary)
            }
            if let error = store.error {
                Label(error, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            HStack {
                if let output = store.lastOutputURL {
                    Button("Show Run Folder") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
                }
                if let log = store.lastLogURL {
                    Button("Show Log") { NSWorkspace.shared.activateFileViewerSelecting([log]) }
                }
                Spacer()
            }
        }
        .padding(12)
    }
}

private struct TrainingConfigurationForm: View {
    @Binding var options: MaterialTrainingOptions
    let dataset: WorkbenchDataset?
    let material: WorkbenchMaterial?
    let checkpoint: WorkbenchCheckpoint?
    let checkpoints: [WorkbenchCheckpoint]
    @Binding var selectedCheckpointId: String?
    @Binding var selectedSampleId: String?
    let disabled: Bool

    var body: some View {
        Form {
            Section("Source dataset") {
                if let dataset {
                    Text(URL(fileURLWithPath: dataset.datasetPath).lastPathComponent)
                        .font(.headline).lineLimit(2).help(dataset.datasetPath)
                    LabeledContent("Materials", value: dataset.materials.count.formatted())
                    LabeledContent("Crops", value: dataset.samples.count.formatted())
                    if let scope = dataset.validationScope {
                        Text(scope.replacingOccurrences(of: "_", with: " "))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Toggle("Train only the selected material", isOn: $options.useSelectedMaterialOnly)
                        .disabled(material == nil)
                    if options.useSelectedMaterialOnly {
                        Picker("Material", selection: Binding(get: { material?.id ?? "" }, set: { materialId in
                            selectedSampleId = dataset.materials.first { $0.id == materialId }?.samples.first?.id
                        })) {
                            ForEach(dataset.materials) { item in
                                Text(item.materialId.replacingOccurrences(of: "_", with: " ")).tag(item.id)
                            }
                        }
                    }
                } else {
                    Text("Choose a dataset to configure its training run.")
                        .foregroundStyle(.secondary)
                }
            }
            Section("Output map") {
                Picker("Target", selection: $options.target) {
                    Text("Displacement").tag("height")
                    Text("Roughness").tag("roughness")
                    Text("OpenGL Normal").tag("normal")
                }
                Picker("Native crop size", selection: $options.size) {
                    Text("1024 × 1024").tag(1024)
                    Text("2048 × 2048").tag(2048)
                }
                Text("Crops train at their prepared native size. Choose a dataset with matching dimensions.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Training") {
                Stepper("Updates per crop: \(options.updatesPerCrop)", value: $options.updatesPerCrop, in: 1...10_000, step: 25)
                Toggle("Include prepared crops awaiting approval", isOn: $options.allowUnreviewed)
                Toggle("Exclude transparent input pixels from the loss", isOn: $options.maskTransparency)
                Toggle("Start from the selected checkpoint", isOn: $options.useWarmStart)
                if options.useWarmStart {
                    Picker("Checkpoint", selection: $selectedCheckpointId) {
                        Text("New head").tag(String?.none)
                        ForEach(checkpoints) { item in Text(item.title).tag(Optional(item.id)) }
                    }
                    if let checkpoint {
                        Text(checkpoint.title)
                            .font(.caption)
                            .lineLimit(2)
                        if checkpoint.target != options.target {
                            Text("The checkpoint supplies learned features; a new output head is created for \(options.target).")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    } else {
                        Text("Choose a checkpoint in the model library, or start a new head.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Section("Resource limits") {
                LabeledContent("Unified memory") {
                    HStack {
                        TextField("GB", value: $options.memoryGB, format: .number.precision(.fractionLength(0...1)))
                            .frame(width: 65)
                            .multilineTextAlignment(.trailing)
                        Text("GB").foregroundStyle(.secondary)
                    }
                }
                Slider(value: $options.memoryGB, in: 2...30, step: 1)
                    .accessibilityLabel("Maximum training memory in gigabytes")
                LabeledContent("Time limit") {
                    HStack {
                        TextField("Minutes", value: $options.maxMinutes, format: .number.precision(.fractionLength(0)))
                            .frame(width: 65)
                            .multilineTextAlignment(.trailing)
                        Text("minutes").foregroundStyle(.secondary)
                    }
                }
                Text("The run saves completed updates when stopped. Memory is checked between stages; large in-flight allocations can exceed the sampled value.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .disabled(disabled)
    }
}

struct WorkbenchRuntimeControls: View {
    @Bindable var store: WorkbenchStore

    var body: some View {
        DisclosureGroup("Local runtime") {
            VStack(alignment: .leading, spacing: 10) {
                runtimeRow("Python environment", path: store.pythonPath, action: store.choosePython)
                runtimeRow("Workspace", path: store.workspacePath, action: store.chooseWorkspace)
                runtimeRow("DINOv2 encoder", path: store.modelDirectory, action: store.chooseEncoder)
                runtimeRow("Encoder source", path: store.codeDirectory, action: store.chooseEncoderCode)
                HStack {
                    Button("Download Pinned Encoder", systemImage: "arrow.down.circle", action: store.installEncoder)
                    Spacer()
                    Button("Remove Downloaded Copy", role: .destructive, action: store.removeDownloadedEncoder)
                }
                Text("Downloads are checked against the pinned model and source hashes. Removing the managed copy leaves your checkpoints and other model folders intact.")
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 8)
        }
        .font(.caption)
        .disabled(store.isBusy)
    }

    private func runtimeRow(_ title: String, path: String, action: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title).fontWeight(.medium)
                Spacer()
                Button("Locate…", action: action)
            }
            Text(path.isEmpty ? "Choose a location" : path)
                .foregroundStyle(.secondary).lineLimit(2).textSelection(.enabled)
        }
    }
}
