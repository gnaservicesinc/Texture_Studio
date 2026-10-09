import SwiftUI
import AppKit

struct TrainingWorkbenchView: View {
    @Bindable var store: WorkbenchStore
    @State private var showCheckpoints = false
    @AppStorage(StudioPreferences.developerModeKey, store: StudioPreferences.defaults) private var developerMode = false

    var body: some View {
        HSplitView {
            Form {
                Section("Training data") {
                    HStack {
                        Button("Open Dataset…") { store.chooseDataset() }
                            .help("Choose an existing dataset folder containing your material maps.")
                        Button("Dataset Info…") { store.showDatasetInfoSheet = true }.disabled(store.dataset == nil)
                        Button("New Dataset…") { store.showNewDatasetSheet = true }
                            .help("Open Material Dataset to create or rename datasets, edit their info, and add or remove materials.")
                    }.disabled(store.isBusy)
                    if let dataset = store.dataset {
                        Text("\(dataset.materials.count) materials · \(dataset.samples.count) material sets")
                        Text(store.datasetNativeSizeLabel).font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("Create a dataset in Manage Datasets, add paired material maps, then open its folder here.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    // Avoid Swift 6.3 IRGen's actor-isolated bound-method conversion.
                    Picker("Training dimensions", selection: Binding(get: { store.training.size }, set: { store.selectTrainingSize($0) })) {
                        ForEach(Array(Set(store.supportedTrainingSizes + [store.training.size])).sorted(), id: \.self) { size in
                            Text("\(size) × \(size)").tag(size)
                        }
                    }.disabled(store.isBusy || store.supportedTrainingSizes.isEmpty)
                    if let plans = store.dataset?.trainingPlans { DatasetPlanSummary(plans: plans) }
                    Text("Every diffuse and target map uses this exact grid. Matching sources are referenced directly; each crop is saved once in temporary training storage; smaller source sets are excluded.")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("Train only selected material", isOn: $store.training.useSelectedMaterialOnly)
                    if !store.datasetPreparationSummary.isEmpty {
                        Text(store.datasetPreparationSummary).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("Refine a material model") {
                    Picker("Map", selection: Binding(get: { store.training.target }, set: { store.selectTrainingTarget($0) })) {
                        Text("Displacement").tag("height")
                        Text("Roughness").tag("roughness")
                        Text("Normals").tag("normal")
                    }
                    Toggle("Start from selected checkpoint", isOn: $store.training.useWarmStart)
                    Button("Choose Starting Checkpoint…") { store.chooseResumeCheckpoint() }
                    Stepper("Updates per map: \(store.training.updatesPerCrop)", value: $store.training.updatesPerCrop, in: 1...10_000)
                    LabeledContent("Time limit (minutes)") {
                        TextField("Minutes", value: $store.training.maxMinutes, format: .number).frame(width: 70)
                    }
                    LabeledContent("Memory budget (GiB)") {
                        Text(store.training.memoryGB.formatted(.number.precision(.fractionLength(0...1))))
                    }
                    Button("Check Supported Sizes") { store.refreshTrainingCapabilities() }
                    Text("\(store.resources.physicalGiB.formatted(.number.precision(.fractionLength(0)))) GiB installed. Sizes that exceed the training budget are omitted; generation sizes are independent.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text(developerMode ? "Exports a full fused safetensors checkpoint and a separate LoRA. Upload to Hugging Face from Saved Models." : "Exports a separate safetensors LoRA for refining your material base.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let issue = store.trainingConfigurationIssue {
                        Text(issue).font(.caption).foregroundStyle(.secondary)
                    }
                    Button("Train Material", systemImage: "play.fill") { store.startTraining() }
                        .buttonStyle(.borderedProminent).disabled(store.isBusy || store.trainingConfigurationIssue != nil)
                }.disabled(store.isBusy)
                if developerMode {
                    Section("Developer controls") {
                        Toggle("Automatic memory budget", isOn: $store.training.automaticMemory)
                        if !store.training.automaticMemory {
                            LabeledContent("Memory budget (GiB)") {
                                TextField("GiB", value: $store.training.memoryGB, format: .number).frame(width: 70)
                            }
                        }
                        Picker("Refinement scope", selection: $store.training.scope) {
                            Text("Final map layer").tag("final-map")
                            Text("Map decoder").tag("map-decoder")
                        }
                        Text("The final layer makes focused refinements with a small adapter. The map decoder changes more features and needs a larger training budget.")
                            .font(.caption).foregroundStyle(.secondary)
                        Toggle("Upload full checkpoint after training", isOn: $store.uploadAfterTraining)
                        Toggle("Public Hugging Face model", isOn: $store.uploadPublic)
                        TextField("Hugging Face repository (automatic when blank)", text: $store.uploadRepo)
                        Stepper("LoRA rank: \(store.training.loraRank)", value: $store.training.loraRank, in: 1...64)
                        LabeledContent("LoRA alpha") { TextField("Alpha", value: $store.training.loraAlpha, format: .number).frame(width: 70) }
                        LabeledContent("Decoded map cache (GiB)") { TextField("GiB", value: $store.training.cacheGB, format: .number).frame(width: 70) }
                    }.disabled(store.isBusy)
                }
                Section { Button("Saved Models & Export…") { showCheckpoints = true } }
            }.formStyle(.grouped).frame(minWidth: 380, idealWidth: 440, maxWidth: 520)
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(store.isTraining ? "Training progress" : "Operation log").font(.headline)
                    Spacer()
                    if store.isBusy {
                        ProgressView().controlSize(.small)
                        WorkbenchStopButtons(store: store)
                    }
                }
                ScrollView {
                    Text(store.logText.isEmpty ? "Training and dataset preparation progress appears here." : store.logText)
                        .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }.frame(maxHeight: .infinity)
                if !store.activity.isEmpty { Text(store.activity).font(.caption).foregroundStyle(.secondary) }
                if let error = store.error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
                HStack {
                    if let output = store.lastOutputURL { Button("Show Run Folder") { NSWorkspace.shared.activateFileViewerSelecting([output]) } }
                    if let log = store.lastLogURL { Button("Show Log") { NSWorkspace.shared.activateFileViewerSelecting([log]) } }
                }
            }.padding(20).frame(minWidth: 420)
        }
        .sheet(isPresented: $showCheckpoints) { CheckpointLibraryView(store: store).frame(minWidth: 780, minHeight: 560) }
        .onChange(of: store.training.memoryGB) { if !store.training.automaticMemory { store.refreshTrainingCapabilities() } }
        .onChange(of: store.training.automaticMemory) { store.refreshTrainingCapabilities() }
        .onChange(of: store.training.cacheGB) { store.refreshTrainingCapabilities() }
        .onChange(of: store.training.scope) { store.refreshTrainingCapabilities() }
    }
}

struct WorkbenchStopButtons: View {
    @Bindable var store: WorkbenchStore
    var body: some View {
        Button(store.isStopping && !store.isSavingTraining ? "Stopping…" : "Stop") { store.stop() }
            .disabled(store.isStopping && !store.isSavingTraining)
            .help("Abort immediately. Checkpoints already saved on disk are kept.")
        if store.hasTrainingStarted || store.isSavingTraining {
            Button(store.isSavingTraining ? "Saving…" : "Stop and Save") { store.stopAndSave() }
                .disabled(!store.canStopAndSave)
                .help("Finish the current update and save the material LoRA.")
        }
    }
}
