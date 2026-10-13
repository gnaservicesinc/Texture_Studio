import SwiftUI

/// The setup remains readable while the results use the rest of the window.
struct CheckpointComparisonView: View {
    @Bindable var store: WorkbenchStore
    @State private var search = ""

    private var selected: [WorkbenchCheckpoint] {
        store.checkpoints.filter { store.comparisonCheckpointIds.contains($0.id) }
    }
    private var matchingCheckpoints: [WorkbenchCheckpoint] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        return store.checkpoints.filter { query.isEmpty || $0.comparisonName.localizedCaseInsensitiveContains(query) }
            .sorted {
                let order = $0.comparisonName.localizedStandardCompare($1.comparisonName)
                return order == .orderedSame ? $0.step > $1.step : order == .orderedAscending
            }
    }
    private var inputName: String {
        store.sourceImageURL?.lastPathComponent ?? store.selectedMaterialName ?? store.selectedSampleId ?? "Choose a test photo"
    }

    var body: some View {
        VStack(spacing: 0) {
            HSplitView {
                setup.frame(minWidth: 285, idealWidth: 320, maxWidth: 380)
                results.frame(minWidth: 620, maxWidth: .infinity, maxHeight: .infinity)
            }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Divider()
            WorkbenchActivityView(store: store)
        }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var setup: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    inputSection
                    Divider()
                    modelSection
                }.padding(16)
            }
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                Text("\(selected.count) checkpoint\(selected.count == 1 ? "" : "s") selected")
                    .font(.callout.bold())
                if let issue = store.comparisonConfigurationIssue, !store.isBusy {
                    Text(issue).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                Button("Run Comparison", systemImage: "play.fill") { store.compare() }
                    .buttonStyle(.borderedProminent)
                    .frame(maxWidth: .infinity)
                    .disabled(store.isBusy || store.comparisonConfigurationIssue != nil)
            }.padding(16)
        }.background(.bar)
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Test image", systemImage: "photo").font(.headline)
            Text(inputName).font(.callout).textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Text(store.sourceImageURL == nil ? "Selected dataset material" : "Imported photo")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Choose Photo…") { store.chooseSourceImage() }
                if store.sourceImageURL != nil {
                    Button("Use Dataset") { store.sourceImageURL = nil }
                        .disabled(store.selectedDiffuseMap == nil)
                }
            }.disabled(store.isBusy)
            Picker("Output size", selection: Binding(get: { store.training.size }, set: { store.selectTrainingSize($0) })) {
                ForEach(Array(Set(store.supportedTrainingSizes + [store.training.size])).sorted(), id: \.self) { size in
                    Text("\(size.formatted()) × \(size.formatted())").tag(size)
                }
            }.disabled(store.isBusy)
            Text("Every selected model receives the same image and crop.")
                .font(.caption).foregroundStyle(.secondary)
            if store.sourceImageURL != nil { photoPreparation }
        }
    }

    private var photoPreparation: some View {
        DisclosureGroup("Photo preparation") {
            VStack(alignment: .leading, spacing: 10) {
                DoubleControl(title: "Tilt X", value: $store.testPhotoSettings.rotationX, range: -70...70, suffix: "°")
                DoubleControl(title: "Tilt Y", value: $store.testPhotoSettings.rotationY, range: -70...70, suffix: "°")
                DoubleControl(title: "Rotate", value: $store.testPhotoSettings.rotationZ, range: -180...180, suffix: "°", enforcesSliderRange: false)
                FloatControl(title: "Lighting balance", value: $store.testPhotoSettings.lightingStrength, range: 0...1)
                FloatControl(title: "Lighting scale", value: $store.testPhotoSettings.lightingRadius, range: 0...1)
            }.padding(.top, 8)
        }.disabled(store.isBusy)
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Models").font(.headline)
                Spacer()
                Button("Add…", systemImage: "plus") { store.chooseCheckpoint() }.disabled(store.isBusy)
            }
            if store.checkpoints.count > 5 {
                TextField("Find a model", text: $search).textFieldStyle(.roundedBorder)
            }
            if store.checkpoints.isEmpty {
                Text("Add saved checkpoints to compare their predictions.")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                HStack {
                    Button("Clear Selection") { store.comparisonCheckpointIds.removeAll() }
                        .disabled(store.isBusy || selected.isEmpty)
                    Spacer()
                    Text("\(store.checkpoints.count) available").font(.caption).foregroundStyle(.secondary)
                }
                LazyVStack(spacing: 8) {
                    ForEach(matchingCheckpoints) { checkpoint in checkpointRow(checkpoint) }
                }
                if matchingCheckpoints.isEmpty {
                    Text("No matching models").font(.caption).foregroundStyle(.secondary)
                }
            }
            Divider()
            Toggle(store.comparisonBaselineLabel, isOn: $store.comparisonIncludesBase)
                .toggleStyle(.checkbox).font(.callout).disabled(store.isBusy)
                .help("Add the model's prediction before training as a separate result.")
        }
    }

    private func checkpointRow(_ checkpoint: WorkbenchCheckpoint) -> some View {
        let checked = store.comparisonCheckpointIds.contains(checkpoint.id)
        return Toggle(isOn: Binding(get: { checked }, set: { enabled in
            if enabled { store.comparisonCheckpointIds.insert(checkpoint.id) }
            else { store.comparisonCheckpointIds.remove(checkpoint.id) }
        })) {
            VStack(alignment: .leading, spacing: 5) {
                Text(checkpoint.comparisonName).font(.callout.bold())
                    .fixedSize(horizontal: false, vertical: true)
                Text("Step \(checkpoint.step.formatted())").font(.caption)
                Text(checkpoint.target == "height" ? "Displacement" : checkpoint.target == "normal" ? "OpenGL Normal" : checkpoint.target.capitalized)
                    .font(.caption).foregroundStyle(.secondary)
                if !checkpoint.supportsStudioInference {
                    Text("Unavailable for inference").font(.caption).foregroundStyle(.orange)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }.toggleStyle(.checkbox)
            .padding(10)
            .background(checked ? Color.accentColor.opacity(0.13) : Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(checked ? Color.accentColor.opacity(0.4) : Color.primary.opacity(0.08)))
            .disabled(store.isBusy || !checkpoint.supportsStudioInference)
            .help(checkpoint.url.lastPathComponent)
    }

    @ViewBuilder private var results: some View {
        if !store.comparisonCandidates.isEmpty {
            ReviewWorkbenchView(candidates: store.comparisonCandidates, comparisonLayout: true)
        } else {
            VStack(alignment: .leading, spacing: 20) {
                Label("Compare your models", systemImage: "rectangle.split.2x1").font(.title2.bold())
                Text("Choose a test image and check the models you want to compare. Run Comparison produces matching maps for each model.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(alignment: .top, spacing: 16) {
                    comparisonPreview("Model A", detail: "Name and training step")
                    comparisonPreview("Model B", detail: "Name and training step")
                    if store.sourceImageURL == nil, store.selectedSample != nil {
                        comparisonPreview("Reference map", detail: "Dataset target")
                    }
                }.frame(maxHeight: 260)
                Text("Inspect up to two model outputs beside the reference. Switch visible models after the run to review every checkpoint.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }.padding(28).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }

    private func comparisonPreview(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            Text(detail).font(.caption).foregroundStyle(.secondary)
            RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .controlBackgroundColor))
                .overlay(Image(systemName: "photo").font(.largeTitle).foregroundStyle(.secondary))
                .frame(minHeight: 140)
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .windowBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }
}
