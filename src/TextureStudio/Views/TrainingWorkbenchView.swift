import SwiftUI
import AppKit

struct TrainingWorkbenchView: View {
    @Bindable var store: WorkbenchStore
    @State private var showCheckpoints = false
    @AppStorage(StudioPreferences.developerModeKey, store: StudioPreferences.defaults) private var developerMode = false

    var body: some View {
        VStack(spacing: 0) {
          HSplitView {
            Form {
                Section("Training data") {
                    HStack {
                        Button("Choose Source Folder…") { store.chooseDataset() }
                            .help("Select the folder containing your original images and material maps.")
                        Button("Dataset Info…") { store.showDatasetInfoSheet = true }.disabled(store.dataset == nil)
                        Button("Show Folder") { store.revealDataset() }.disabled(store.dataset == nil)
                    }.disabled(store.isBusy)
                    if let dataset = store.dataset {
                        Text(store.datasetName).font(.headline).textSelection(.enabled)
                        Text(store.datasetDisplayURL?.path ?? dataset.datasetPath)
                            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        Text(store.datasetInventorySummary)
                        LabeledContent("Materials", value: store.trainingScopeSummary)
                            .font(.callout)
                        if store.specialistTrainingContext == nil {
                            Text("Choose a material category in Loom Room.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Text(store.datasetNativeSizeLabel).font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("Choose your source folder. Each material subfolder contains a diffuse image and its surface maps; image tags identify the material category and map type.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    // Avoid Swift 6.3 IRGen's actor-isolated bound-method conversion.
                    Picker("Training dimensions", selection: Binding(get: { store.training.size }, set: { store.selectTrainingSize($0) })) {
                        ForEach(Array(Set(store.supportedTrainingSizes + [store.training.size])).sorted(), id: \.self) { size in
                            Text("\(size) × \(size)").tag(size)
                        }
                    }.disabled(store.isBusy || store.supportedTrainingSizes.isEmpty)
                    if store.specialistTrainingContext != nil, let plans = store.dataset?.resolutionPlans?[String(store.training.size)] ?? store.dataset?.trainingPlans {
                        DatasetPlanSummary(plans: plans)
                    }
                    Text(store.trainingCurrentCropSummary).font(.callout).monospacedDigit()
                    Text("Diffuse and target crops use the same position and pixel size. Originals are never resized. A source smaller than this crop size remains available for inspection.")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("Randomize image order each epoch", isOn: $store.training.shuffleTrainingOrder)
                        .toggleStyle(.checkbox)
                        .accessibilityIdentifier("training.shuffle-order")
                        .help("Turn off to use the same image order each epoch.")
                    TrainingCropSelectionFields(options: $store.training)
                    if !store.datasetPreparationSummary.isEmpty {
                        Text(store.datasetPreparationSummary).font(.caption).foregroundStyle(.secondary)
                    }
                }.disabled(store.isBusy)
                Section(store.training.modelFamily == .depthProHeightRefiner ? "Train a DepthPro Height Refiner" : store.training.modelFamily.isCompact ? "Train a compact material model" : "Refine a material model") {
                    TrainingModelFamilyField(store: store)
                    TrainingModelNameField(name: $store.training.modelName, suggestedName: store.suggestedTrainingModelName)
                    Text("Shown in Saved Models and exported with the checkpoint. Clear the field to use the suggested name.")
                        .font(.caption).foregroundStyle(.secondary)
                    Picker("Map", selection: Binding(get: { store.training.target }, set: { store.selectTrainingTarget($0) })) {
                        Text("Displacement").tag("height")
                        if store.training.modelFamily != .depthProHeightRefiner {
                            Text("Roughness").tag("roughness")
                            Text("Normals").tag("normal")
                        }
                    }
                    if store.training.target == "height" {
                        Toggle("Exaggerate features", isOn: $store.training.exaggerateFeatures)
                            .toggleStyle(.checkbox)
                            .accessibilityIdentifier("training.exaggerate-features")
                            .help("Increase depth contrast with a nonlinear stretch. Originals stay unchanged.")
                    }
                    Toggle("Start from selected checkpoint", isOn: $store.training.useWarmStart)
                    if store.training.modelFamily.isCompact {
                        if store.training.useWarmStart {
                            if let width = store.selectedCheckpoint?.networkWidth,
                               let preset = MaterialTrainingModelSize(rawValue: width) {
                                Text("Recorded model size: \(preset.label(target: store.training.target, family: store.training.modelFamily))")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Text("A starting checkpoint retains its recorded model size.")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Picker("Model size", selection: $store.training.compactWidth) {
                                ForEach(MaterialTrainingModelSize.allCases) { preset in
                                    Text(preset.label(target: store.training.target, family: store.training.modelFamily)).tag(preset.rawValue)
                                }
                            }
                            Text("Larger models add capacity, training work and memory use. Compare learned quality on the same validation materials; size alone does not guarantee better maps.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Text(store.training.modelFamily.isCompact ? "Training scope: complete compact model" : "Training scope: \(store.training.scope == "map-decoder" ? "Map decoder and output branch" : "Map output branch")")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Choose Starting Checkpoint…") { store.chooseResumeCheckpoint() }
                    if store.training.useWarmStart {
                        if let checkpoint = store.selectedCheckpoint {
                            Text("Starting model: \(checkpoint.title)").font(.caption).textSelection(.enabled)
                            Text(checkpoint.modelSummary)
                                .font(.caption).foregroundStyle(.secondary)
                            Text(checkpoint.inputColorSummary)
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text("Choose a starting checkpoint before training.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if !store.training.useWarmStart, store.training.modelFamily.isCompact, store.training.modelFamily != .depthProHeightRefiner {
                        Text(store.training.target == "height" && store.training.exaggerateFeatures
                             ? "Diffuse input: linear sRGB. Depth targets use the selected feature contrast."
                             : "Diffuse input: linear sRGB. Height, roughness and normal targets keep their numerical values.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    NumericField("Updates per map", value: $store.training.updatesPerCrop, atLeast: 1, unit: "updates")
                    NumericField("Time limit", value: $store.training.maxMinutes, greaterThan: 0, unit: "minutes")
                    Text(store.training.modelFamily.isCompact ? "Exports a complete safetensors model. No PBRnxt base weights are required." : developerMode ? "Exports a full fused safetensors checkpoint and a separate LoRA. Upload to Hugging Face from Saved Models." : "Exports a separate safetensors LoRA for refining your material base.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let issue = store.trainingConfigurationIssue {
                        Text(issue).font(.caption).foregroundStyle(.secondary)
                    }
                }.disabled(store.isBusy)
                Section("Training settings") {
                    NumericField("Learning rate", value: $store.training.learningRate, greaterThan: 0)
                    Picker("Optimizer", selection: $store.training.optimizer) {
                        Text("AdamW").tag("adamw")
                        Text("Adam").tag("adam")
                        Text("Lion · sign momentum").tag("lion")
                    }
                    .onChange(of: store.training.optimizer) { old, new in
                        if new == "lion", store.training.optimizerBeta2 == 0.999 { store.training.optimizerBeta2 = 0.99 }
                        if old == "lion", new != "lion", store.training.optimizerBeta2 == 0.99 { store.training.optimizerBeta2 = 0.999 }
                    }
                    if store.training.optimizer == "lion" {
                        Text("Lion keeps one momentum tensor per parameter. Start with a learning rate 3–10× lower than AdamW; increase nonzero weight decay by the same factor to retain its strength. Compare validation quality for your material.")
                            .font(.caption).foregroundStyle(.secondary)
                        Button("Apply Lion starting settings (10× lower learning rate)") {
                            store.training.learningRate /= 10
                            store.training.weightDecay *= 10
                            store.training.optimizerBeta1 = 0.9
                            store.training.optimizerBeta2 = 0.99
                        }
                    }
                    NumericField("Gradient accumulation steps", value: $store.training.gradientAccumulationSteps, atLeast: 1, unit: "maps / update")
                    Text("One complete map is processed at a time. Gradients are averaged before each optimizer update. Increasing accumulation does more work per update.")
                        .font(.caption).foregroundStyle(.secondary)
                    NumericField("Weight decay", value: $store.training.weightDecay, atLeast: 0)
                    Picker("Learning rate schedule", selection: $store.training.learningRateSchedule) {
                        Text("Constant").tag("constant")
                        Text("Cosine decay").tag("cosine")
                    }
                    if store.training.learningRateSchedule == "cosine" {
                        NumericField("Minimum learning rate ratio", value: $store.training.minimumLearningRateRatio, in: Double(Float.leastNonzeroMagnitude)...1)
                        Text("Cosine decay finishes at this fraction of the starting learning rate.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    NumericField("Warmup", value: $store.training.warmupUpdates, atLeast: 0, unit: "updates")
                    DisclosureGroup("Advanced training settings") {
                        NumericField("Optimizer beta 1", value: $store.training.optimizerBeta1, in: 0...Double(Float(1).nextDown))
                        NumericField("Optimizer beta 2", value: $store.training.optimizerBeta2, in: 0...Double(Float(1).nextDown))
                        if store.training.optimizer != "lion" {
                            NumericField("Optimizer epsilon", value: $store.training.optimizerEpsilon, greaterThan: 0)
                        }
                        NumericField("Maximum gradient norm", value: $store.training.maxGradientNorm, atLeast: 0)
                        Text("0 disables gradient clipping. Warm starts restore weights and start fresh optimizer moments.")
                            .font(.caption).foregroundStyle(.secondary)
                        NumericField("Random seed", value: $store.training.seed, atLeast: 0)
                        if !store.training.modelFamily.isCompact {
                            NumericField("LoRA rank", value: $store.training.loraRank, atLeast: 1)
                                .disabled(store.training.useWarmStart && store.selectedCheckpoint?.schema == "texture-studio-material-lora-v1")
                            NumericField("LoRA alpha", value: $store.training.loraAlpha, greaterThan: 0)
                                .disabled(store.training.useWarmStart && store.selectedCheckpoint?.schema == "texture-studio-material-lora-v1")
                            if store.training.useWarmStart && store.selectedCheckpoint?.schema == "texture-studio-material-lora-v1" {
                                Text("The starting LoRA retains its recorded rank and alpha. These controls apply to new adapters.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        Text(store.training.modelFamily.isCompact ? "Float32 training of all compact model weights. The fixed objective is value L1 + 4 × multiscale detail L1." : "Float32 training. The fixed objective is value L1 + 4 × multiscale detail L1; the selected LoRA layers learn while base weights stay frozen.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }.disabled(store.isBusy)
                Section("Validation & checkpoints") {
                    if let validation = store.dataset?.validation {
                        Text(validation.enabled ? (validation.quickCount == 0 ? "Quick checks are off. The configured validation pool is checked when saving." : "Up to \(validation.quickCount) crops per quick check; the configured validation pool is checked when saving.") : "Validation is off in Dataset Info.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    TrainingIntervalField("Quick check every", value: $store.training.validationEvery, unit: $store.training.validationUnit)
                    Text("0 turns off periodic quick checks.")
                        .font(.caption).foregroundStyle(.secondary)
                    TrainingIntervalField("Save checkpoint every", value: $store.training.checkpointEvery, unit: $store.training.checkpointUnit)
                    Text("An epoch is one complete pass through the training materials. A step is one optimizer update. Each interval uses its selected unit.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("0 saves on request and at final export. Saving checks the full validation pool when validation is enabled. Save Checkpoint Now finishes the current step, saves, and continues training.")
                        .font(.caption).foregroundStyle(.secondary)
                }.disabled(store.isBusy)
                if developerMode {
                    Section("Developer controls") {
                        if !store.training.modelFamily.isCompact {
                            Picker("Refinement scope", selection: $store.training.scope) {
                                Text("Map output branch").tag("final-map")
                                Text("Map decoder").tag("map-decoder")
                            }
                            Text("The output branch refines the selected map's RRDB layers. The map decoder also adapts its decoder and tail, adding backward work and time per step.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
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
                    }.disabled(store.isBusy)
                }
            }.formStyle(.grouped).frame(minWidth: 380, idealWidth: 440, maxWidth: 520)
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(store.isTraining ? "Training progress" : "Operation log").font(.headline)
                    Spacer()
                    if store.isBusy {
                        ProgressView().controlSize(.small)
                    }
                }
                if let progress = store.trainingProgress { TrainingProgressView(progress: progress) }
                ScrollView {
                    Text(store.readableOperationLog.isEmpty ? "Training and dataset preparation progress appears here." : store.readableOperationLog)
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
          Divider()
          HStack(spacing: 12) {
              Button("Saved Models & Export…") { showCheckpoints = true }
              Spacer()
              if store.isBusy { WorkbenchStopButtons(store: store) }
              Button("Train Material", systemImage: "play.fill") { store.startTraining() }
                  .buttonStyle(.borderedProminent)
                  .disabled(store.isBusy || store.trainingConfigurationIssue != nil)
                  .accessibilityIdentifier("training.start")
          }
          .padding(.horizontal, 20).padding(.vertical, 12)
          .accessibilityElement(children: .contain)
          .accessibilityIdentifier("training.actions")
        }
        .sheet(isPresented: $showCheckpoints) { CheckpointLibraryView(store: store).frame(minWidth: 780, minHeight: 560) }
    }
}

struct TrainingModelFamilyField: View {
    @Bindable var store: WorkbenchStore

    var body: some View {
        Picker("Model family", selection: Binding(get: { store.training.modelFamily }, set: { store.selectTrainingModelFamily($0) })) {
            ForEach(MaterialTrainingModelFamily.allCases, id: \.self) { family in
                Text(family.label).tag(family)
            }
        }
        .accessibilityIdentifier("training.model-family")
        if store.training.modelFamily == .depthProHeightRefiner {
            Text("One DepthPro depth channel → one surface-height channel. Each crop gets a cached DepthPro prediction when needed for its update. Evaluate the learned relief on held-out materials.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                TextField("DepthPro Core ML package", text: $store.depthProModelPath)
                Button("Locate DepthPro…") { store.chooseDepthProModel() }
            }
            Text("DepthPro downloads automatically when needed and is verified before use. Production runs it once, then this refiner. Cached float data uses the same depth preparation as inference.")
                .font(.caption).foregroundStyle(.secondary)
        } else if store.training.modelFamily == .compactScalar {
            Text("RGB diffuse input has 3 channels. The model predicts one displacement or roughness value per pixel. New compact models start from scratch; review their learned quality before use.")
                .font(.caption).foregroundStyle(.secondary)
        } else if store.training.modelFamily == .compactNormal {
            Text("RGB diffuse input has 3 channels. The model predicts a normal vector with 3 components per pixel. New compact models start from scratch; review their learned quality before use.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct TrainingCropSelectionFields: View {
    @Binding var options: MaterialTrainingOptions

    var body: some View {
        Picker("Choose new crop positions", selection: $options.randomizeCrops) {
            ForEach(MaterialTrainingCropRandomization.allCases, id: \.self) { timing in
                Text(timing == .never ? "Keep initial crops" : timing.label).tag(timing)
            }
        }
        .pickerStyle(.menu)
        .accessibilityIdentifier("training.randomize-crops")
        Toggle("Randomly flip crops", isOn: $options.randomFlip)
            .disabled(!options.cropAugmentationEnabled)
            .accessibilityIdentifier("training.random-flip")
            .help("Randomly mirror each paired crop horizontally, vertically, both ways or neither. Diffuse and target maps always receive the same flip.")
        Toggle("Randomly rotate center crops", isOn: $options.randomSpinCentercut)
            .disabled(!options.cropAugmentationEnabled)
            .accessibilityIdentifier("training.random-spin-centercut")
            .help("Rotate paired center crops by 90°, 180° or 270°. Both maps receive the same rotation, with no resizing or interpolation.")
        Text(cropDescription)
            .font(.caption).foregroundStyle(.secondary)
    }

    private var cropDescription: String {
        switch options.randomizeCrops {
        case .never:
            "Keep each initial crop throughout training. Random flips and rotations are paused."
        case .run:
            "Choose random positions at the start of each run when an image has room for more than one crop. Keep those crops and any flips or rotations for the run."
        case .epoch:
            "Choose new random positions for each pass through the materials when an image has room for more than one crop. Refresh enabled flips and rotations with those crops."
        }
    }
}

struct TrainingIntervalField: View {
    let title: String
    @Binding var value: Int
    @Binding var unit: MaterialTrainingIntervalUnit

    init(_ title: String, value: Binding<Int>, unit: Binding<MaterialTrainingIntervalUnit>) {
        self.title = title
        _value = value
        _unit = unit
    }

    var body: some View {
        LabeledContent(title) {
            HStack(spacing: 8) {
                NumericTextField(title: title, value: $value, atLeast: 0)
                    .frame(width: 90)
                Picker("\(title) unit", selection: $unit) {
                    ForEach(MaterialTrainingIntervalUnit.allCases, id: \.self) { choice in
                        Text(choice.label).tag(choice)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 78)
                .accessibilityLabel("\(title) unit")
            }
        }
    }
}

struct WorkbenchStopButtons: View {
    @Bindable var store: WorkbenchStore
    var body: some View {
        if store.isTraining {
            Button(store.isCheckpointPending ? "Checkpoint Queued…" : "Save Checkpoint Now") { store.saveCheckpointNow() }
                .disabled(!store.canStopAndSave || store.isCheckpointPending)
                .accessibilityIdentifier("training.save-checkpoint")
                .help("Run full validation when enabled, save a checkpoint, and continue training.")
            Button(store.isSavingTraining ? "Stopping & Saving…" : "Stop & Save", systemImage: "stop.fill") { store.stop() }
                .disabled(!store.canStopAndSave)
                .accessibilityIdentifier("training.stop")
                .help(store.hasTrainingStarted || store.isSavingTraining
                    ? "Finish the current update, validate when enabled, and save the material LoRA. Abort remains available while saving."
                    : "Available once training starts. Use Abort to cancel dataset preparation or model setup.")
            Button(role: .destructive) { store.abort() } label: {
                Label(store.isStopping && !store.isSavingTraining ? "Aborting…" : "Abort", systemImage: "xmark.octagon.fill")
            }
                .disabled(!store.canAbort)
                .accessibilityIdentifier("training.abort")
                .help("Cancel setup or training without a final save. Previously saved checkpoints are kept.")
        } else {
            Button(store.isStopping ? "Stopping…" : "Stop") { store.stop() }
                .disabled(!store.canAbort)
                .help("Cancel the current operation.")
        }
    }
}
