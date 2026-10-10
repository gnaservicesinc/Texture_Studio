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
                        Text(store.datasetName).font(.headline).textSelection(.enabled)
                        Text(store.datasetDisplayURL?.path ?? dataset.datasetPath)
                            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
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
                    if store.training.useSelectedMaterialOnly {
                        Text(store.selectedMaterialName.map { "Selected material: \($0)" } ?? "Select a material in Dataset.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if !store.datasetPreparationSummary.isEmpty {
                        Text(store.datasetPreparationSummary).font(.caption).foregroundStyle(.secondary)
                    }
                }.disabled(store.isBusy)
                Section("Refine a material model") {
                    Picker("Map", selection: Binding(get: { store.training.target }, set: { store.selectTrainingTarget($0) })) {
                        Text("Displacement").tag("height")
                        Text("Roughness").tag("roughness")
                        Text("Normals").tag("normal")
                    }
                    Toggle("Start from selected checkpoint", isOn: $store.training.useWarmStart)
                    Button("Choose Starting Checkpoint…") { store.chooseResumeCheckpoint() }
                    if store.training.useWarmStart {
                        if let checkpoint = store.selectedCheckpoint {
                            Text("Starting model: \(checkpoint.title)").font(.caption).textSelection(.enabled)
                            Text(checkpoint.modelSummary)
                                .font(.caption).foregroundStyle(.secondary)
                            Text("Refinement scope: \(checkpoint.scope == "map-decoder" ? "Map decoder" : "Final map layer")")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text("Choose a starting checkpoint before training.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    NumericField("Updates per map", value: $store.training.updatesPerCrop, in: 1...10_000, unit: "updates")
                    NumericField("Time limit", value: $store.training.maxMinutes, in: 1...240, unit: "minutes")
                    Text(developerMode ? "Exports a full fused safetensors checkpoint and a separate LoRA. Upload to Hugging Face from Saved Models." : "Exports a separate safetensors LoRA for refining your material base.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let issue = store.trainingConfigurationIssue {
                        Text(issue).font(.caption).foregroundStyle(.secondary)
                    }
                    Button("Train Material", systemImage: "play.fill") { store.startTraining() }
                        .buttonStyle(.borderedProminent).disabled(store.isBusy || store.trainingConfigurationIssue != nil)
                }.disabled(store.isBusy)
                Section("Validation & checkpoints") {
                    if let validation = store.dataset?.validation {
                        Text(validation.enabled ? "Up to \(validation.quickCount) crops per quick check; full \(validation.percent.formatted())% pool when saving." : "Validation is disabled in Dataset Info.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    NumericField("Quick check every", value: $store.training.validationEvery, in: 1...10_000, unit: "updates")
                    NumericField("Save checkpoint every", value: $store.training.checkpointEvery, in: 0...100_000, unit: "updates")
                    Text("Set checkpoint updates to 0 to save on request and at final export. Every saved checkpoint runs full validation. Save Checkpoint Now finishes the current update and continues training after saving.")
                        .font(.caption).foregroundStyle(.secondary)
                }.disabled(store.isBusy)
                if developerMode {
                    Section("Developer controls") {
                        Picker("Refinement scope", selection: $store.training.scope) {
                            Text("Final map layer").tag("final-map")
                            Text("Map decoder").tag("map-decoder")
                        }
                        Text("The final layer makes focused refinements with a small adapter. The map decoder changes more features.")
                            .font(.caption).foregroundStyle(.secondary)
                        Toggle("Upload full checkpoint after training", isOn: $store.uploadAfterTraining)
                        Toggle("Public Hugging Face model", isOn: $store.uploadPublic)
                        TextField("Hugging Face repository (automatic when blank)", text: $store.uploadRepo)
                        if store.uploadAfterTraining {
                            Text(store.uploadAccount.map { account in
                                let repository = store.effectiveUploadRepo.isEmpty ? "an automatic repository under \(account)" : store.effectiveUploadRepo
                                return "After training: upload to \(repository) · \(store.uploadPublic ? "Public" : "Private")"
                            } ?? "After training: upload when a saved Hugging Face login is available. Set the repository and visibility above.")
                                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                        NumericField("LoRA rank", value: $store.training.loraRank, in: 1...64)
                        NumericField("LoRA alpha", value: $store.training.loraAlpha, in: 0.01...128)
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
                    }
                }
                if store.isBusy {
                    HStack { WorkbenchStopButtons(store: store) }
                }
                ScrollView {
                    Text(store.logText.isEmpty ? "Training and dataset preparation progress appears here." : store.logText)
                        .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }.frame(maxHeight: .infinity)
                if !store.validationSummary.isEmpty { Text(store.validationSummary).font(.callout) }
                if !store.activity.isEmpty { Text(store.activity).font(.caption).foregroundStyle(.secondary) }
                if let error = store.error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
                HStack {
                    if let output = store.lastOutputURL { Button("Show Run Folder") { NSWorkspace.shared.activateFileViewerSelecting([output]) } }
                    if let log = store.lastLogURL { Button("Show Log") { NSWorkspace.shared.activateFileViewerSelecting([log]) } }
                }
            }.padding(20).frame(minWidth: 420)
        }
        .sheet(isPresented: $showCheckpoints) { CheckpointLibraryView(store: store).frame(minWidth: 780, minHeight: 560) }
    }
}

struct WorkbenchStopButtons: View {
    @Bindable var store: WorkbenchStore
    var body: some View {
        if store.isTraining {
            Button(store.isSavingTraining ? "Stopping…" : "Stop", systemImage: "stop.fill") { store.stop() }
                .disabled(!store.canStopAndSave)
                .accessibilityIdentifier("training.stop")
                .help(store.hasTrainingStarted || store.isSavingTraining
                    ? "Finish the current update, validate, and save the material LoRA. Abort remains available while saving."
                    : "Available once training starts. Use Abort to cancel dataset preparation or model setup.")
            Button(role: .destructive) { store.abort() } label: {
                Label(store.isStopping && !store.isSavingTraining ? "Aborting…" : "Abort", systemImage: "xmark.octagon.fill")
            }
                .disabled(!store.canAbort)
                .accessibilityIdentifier("training.abort")
                .help("Cancel setup or training without a final save. Previously saved checkpoints are kept.")
            Button(store.isCheckpointPending ? "Checkpoint Queued…" : "Save Checkpoint Now") { store.saveCheckpointNow() }
                .disabled(!store.canStopAndSave || store.isCheckpointPending)
                .accessibilityIdentifier("training.save-checkpoint")
                .help("Run full validation, save a checkpoint, and continue training.")
        } else {
            Button(store.isStopping ? "Stopping…" : "Stop") { store.stop() }
                .disabled(!store.canAbort)
                .help("Cancel the current operation.")
        }
    }
}
