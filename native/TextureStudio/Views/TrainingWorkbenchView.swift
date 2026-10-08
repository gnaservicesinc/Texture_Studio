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
        if let issue = store.resources.trainingMemoryIssue(store.training.memoryGB) { return issue }
        if !store.training.maxMinutes.isFinite || !(1...240).contains(store.training.maxMinutes) {
            return "Choose a time limit between 1 and 240 minutes."
        }
        if !(1...10_000).contains(store.training.updatesPerCrop) {
            return "Choose between 1 and 10,000 updates per crop."
        }
        return nil
    }
    private var trainingIssue: String? {
        store.trainingConfigurationIssue
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
                                          nativeSize: Binding(get: { store.training.size }, set: store.selectTrainingSize),
                                          locateCheckpoint: store.chooseCheckpoint,
                                          disabled: store.isBusy, resources: store.resources)
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
    }

    private var runControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.isPreparingDataset ? "Preparing native crops" : (store.isTraining ? "Training in progress" : "Material training")).font(.title3.bold())
                    Text(store.isResumingTraining ? "Restoring the saved run’s target and native crop size" : "Final saved checkpoint · native \(store.training.size) × \(store.training.size)")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button { showCheckpoints = true } label: { Label("Checkpoints", systemImage: "shippingbox") }
            }
            HStack {
                if store.isPreparingDataset {
                    ProgressView().controlSize(.small)
                    Text("Preparing \(store.training.size) × \(store.training.size)…").font(.callout)
                    Button(store.isStopping ? "Stopping…" : "Stop Preparation") { store.stop() }
                        .disabled(store.isStopping)
                } else if store.isTraining {
                    ProgressView().controlSize(.small)
                    Button(store.isStopping ? "Saving checkpoint…" : "Stop & Save") { store.stop() }
                        .disabled(store.isStopping)
                } else {
                    Button("Start Training") { store.startTraining() }
                        .buttonStyle(.glassProminent)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(store.isBusy || trainingIssue != nil)
                        .help("Prepare the selected native size if needed, then train with the starting point chosen below.")
                    Button("Resume Saved Run…") { store.chooseResumeCheckpoint() }
                        .disabled(store.isBusy || resourceIssue != nil)
                        .help("Continue checkpoint.latest.pt with its saved optimizer state and original crop selection.")
                }
                Spacer()
            }
            if !store.isBusy, let issue = trainingIssue {
                Label(issue, systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Start Training uses the setup on the left. Resume restores a saved run’s optimizer and source selections; updates per crop becomes its desired total.")
                .font(.caption).foregroundStyle(.secondary)
            if !store.datasetPreparationSummary.isEmpty {
                Label(store.datasetPreparationSummary, systemImage: "checkmark.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }
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
    @Binding var nativeSize: Int
    let locateCheckpoint: () -> Void
    let disabled: Bool
    let resources: MachineResources

    var body: some View {
        Form {
            Section("Getting started") {
                Text("Open a dataset, choose the map and native crop size, then choose a starting point. After training, inspect the saved maps and compare checkpoints visually before choosing one for Texture Studio.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Source dataset") {
                if let dataset {
                    Text(URL(fileURLWithPath: dataset.datasetPath).lastPathComponent)
                        .font(.headline).lineLimit(2).help(dataset.datasetPath)
                    LabeledContent("Materials", value: dataset.materials.count.formatted())
                    LabeledContent("Training crops", value: dataset.samples.filter { $0.split == "train" }.count.formatted())
                    Picker("What to train on", selection: $options.useSelectedMaterialOnly) {
                        Text("All materials in this dataset").tag(false)
                        Text("One material for a quick fit").tag(true)
                    }.pickerStyle(.radioGroup)
                        .help("All materials teaches the broader collection. One material checks how well the model can fit the surface you choose below.")
                    if options.useSelectedMaterialOnly {
                        Picker("Material", selection: Binding(get: { material?.id ?? "" }, set: { materialId in
                            selectedSampleId = dataset.materials.first { $0.id == materialId }?.samples.first?.id
                        })) {
                            ForEach(dataset.materials) { item in
                                Text(item.materialId.replacingOccurrences(of: "_", with: " ")).tag(item.id)
                            }
                        }
                        Text("Only \(material?.materialId.replacingOccurrences(of: "_", with: " ") ?? "the material you choose") updates the model in this run. Other materials remain in the dataset. This tests fitting one surface, not performance on new photos.")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("All usable training crops update the model. This takes longer but teaches the variety in your collection. Excluded crops are always skipped.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text("A unique crop from a repeatable random 5% of sources checks progress automatically. Quick fits use a check crop from the chosen material. Judge the result visually on new photographs later.")
                        .font(.caption).foregroundStyle(.secondary)
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
                .help("Each target learns from its own supplied source map. Normals use the OpenGL +Y convention.")
                Picker("Native crop size", selection: $nativeSize) {
                    Text("1024 × 1024").tag(1024)
                    Text("2048 × 2048").tag(2048)
                }
                Text("Changing size prepares matching crops from original material maps. Targets keep their native pixels; the original dataset stays intact. Larger crops need more memory and time.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Training") {
                Stepper("Updates per crop: \(options.updatesPerCrop)", value: $options.updatesPerCrop, in: 1...10_000, step: 25)
                    .help("Every eligible training crop receives this many optimizer updates. A time or memory limit may stop the run earlier and save its progress.")
                Toggle("Train with unreviewed crops", isOn: $options.allowUnreviewed)
                    .help("On: train on approved and not-yet-reviewed crops. Off: review the training crops first. Excluded crops are always skipped.")
                Text(options.allowUnreviewed
                     ? "Approved crops and crops you have not reviewed yet can train. Use Dataset → Exclude for any problem crop; exclusions are always respected."
                     : "Review the chosen training crops in Dataset first: approve usable crops and exclude problems before starting.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Exclude transparent input pixels from the loss", isOn: $options.maskTransparency)
                    .help("Ignore invalid photo pixels and their immediate boundary during supervision; source pixels remain unchanged.")
                Picker("Starting point", selection: $options.useWarmStart) {
                    Text("New material head · DINOv2 Base features").tag(false)
                    Text("Refine a checkpoint").tag(true)
                }
                .pickerStyle(.radioGroup)
                if options.useWarmStart {
                    Picker("Checkpoint", selection: $selectedCheckpointId) {
                        Text("Choose a checkpoint…").tag(String?.none)
                        ForEach(checkpoints) { item in
                            Text(item.title + (!item.supportsTrainingWarmStart ? " · inference only" : item.variant == "lora" ? " · fixed LoRA" : "")).tag(Optional(item.id))
                        }
                    }
                    Button("Locate Checkpoint…", systemImage: "folder", action: locateCheckpoint)
                    Text(checkpoint?.variant == "lora" && checkpoint?.supportsTrainingWarmStart == true
                         ? "Reuse the selected head and its saved LoRA adapters with a new optimizer. The encoder and adapters stay fixed; only the material head is refined."
                         : "Reuse the selected material head and begin a new optimizer run. The pretrained DINOv2 Base encoder stays frozen.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let checkpoint {
                        Text(checkpoint.title)
                            .font(.caption)
                            .lineLimit(2)
                        if checkpoint.target != options.target {
                            Text("The material head’s hidden layers are reused; its final output layer is replaced for \(options.target).")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if !checkpoint.supportsTrainingWarmStart {
                            Text("The selected backend has not verified refinement support for this checkpoint. Choose a supported checkpoint or start from Base DINOv2.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    } else {
                        Text("Locate a checkpoint in the model library, or start a new material head with DINOv2 Base features.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("DINOv2 Base supplies pretrained visual features; it has no pretrained height output. Your new material head learns displacement, roughness or normals from your paired maps. DA3 camera depth is a separate Studio option.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("Resource limits") {
                LabeledContent("Unified memory") {
                    HStack {
                        TextField("GiB", value: $options.memoryGB, format: .number.precision(.fractionLength(0...1)))
                            .frame(width: 65)
                            .multilineTextAlignment(.trailing)
                        Text("GiB").foregroundStyle(.secondary)
                    }
                }
                Slider(value: $options.memoryGB, in: resources.trainingMemoryRange, step: 0.1)
                    .accessibilityLabel("Maximum training memory in gibibytes")
                    .help("The upper limit uses this Mac’s installed unified memory and leaves space for macOS. Training allocates memory as needed; cached inputs avoid repeated decoding.")
                HStack {
                    Text("\(resources.physicalGiB, format: .number.precision(.fractionLength(0...1))) GiB installed · up to \(resources.maximumTrainingGiB, format: .number.precision(.fractionLength(0...1))) GiB for training")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Use recommended") { options.memoryGB = resources.defaultTrainingGiB }
                        .help("Use the recommended \(resources.defaultTrainingGiB.formatted(.number.precision(.fractionLength(0...1)))) GiB ceiling for this Mac.")
                }
                Text("Training uses memory as needed, up to this ceiling. The recommendation uses Metal’s working-set guidance and leaves \(resources.reservedGiB, format: .number.precision(.fractionLength(0...1))) GiB for macOS. A higher ceiling helps jobs that reach the limit; input caching uses spare capacity to reduce repeated work.")
                    .font(.caption).foregroundStyle(.secondary)
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
                runtimeRow("Working folder", path: store.workspacePath, action: store.chooseWorkspace)
                    .help("Saved training runs and comparisons go here. The default app-managed folder is created when first needed; a repository is not required.")
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
