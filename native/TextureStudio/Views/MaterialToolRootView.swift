import SwiftUI

struct MaterialToolRootView: View {
    let role: MaterialTool
    @Bindable var store: WorkbenchStore
    @State private var review = ReviewSessionStore()
    @State private var showModels = false
    @State private var showRuntime = false
    init(role: MaterialTool, store: WorkbenchStore, review: ReviewSessionStore? = nil) {
        self.role = role
        self.store = store
        self._review = State(initialValue: review ?? ReviewSessionStore())
    }
    private var comparisonSelection: [WorkbenchCheckpoint] {
        store.checkpoints.filter { store.comparisonCheckpointIds.contains($0.id) }
    }
    private var comparisonReady: Bool {
        !comparisonSelection.isEmpty && comparisonSelection.count + (store.comparisonIncludesBase ? 1 : 0) >= 2 && Set(comparisonSelection.map(\.target)).count == 1
            && (store.sourceImageURL != nil || store.selectedSample?.maps["input"] != nil)
    }

    var body: some View {
        Group {
            switch role {
            case .review: reviewView
            case .compare: compareView
            case .dataset: DatasetWorkbenchView(store: store)
            case .train: TrainingWorkbenchView(store: store)
            }
        }
        .navigationTitle(role.title)
        .frame(minWidth: 1050, minHeight: 700)
        .toolbar {
            ToolbarItemGroup {
                Button { showModels = true } label: { Label("Checkpoints", systemImage: "shippingbox") }
                    .help("Inspect saved checkpoints, compare their outputs or export a portable package.")
                Button { showRuntime = true } label: { Label("Runtime", systemImage: "gearshape") }
                    .help("Configure the local Python environment and working folder. Developer mode exposes material model controls.")
                Menu {
                    ForEach(MaterialTool.allCases) { tool in
                        Button(tool.title, systemImage: tool.symbol) { MaterialToolLauncher.open(tool) }
                    }
                    Button("Texture Studio", systemImage: "square.3.layers.3d") { MaterialToolLauncher.openStudio() }
                } label: { Label("Tools", systemImage: "macwindow.on.rectangle") }
            }
        }
        .sheet(isPresented: $showModels) { CheckpointLibraryView(store: store).frame(minWidth: 850, minHeight: 620) }
        .sheet(isPresented: $showRuntime) { WorkbenchRuntimeView(store: store).frame(width: 710, height: 500) }
        .alert("Material tool", isPresented: Binding(get: { store.error != nil || review.error != nil }, set: { if !$0 { store.error = nil; review.error = nil } })) {
            Button("OK") { store.error = nil; review.error = nil }
        } message: { Text(store.error ?? review.error ?? "") }
        .task {
            store.restore()
            if role == .review && review.groups.isEmpty { review.restore(workspace: store.workspacePath) }
        }
        .onOpenURL { url in
            if url.lastPathComponent == "dataset.json" { store.openDataset(url) }
            else if url.pathExtension == "json" { review.load(url) }
            else if url.pathExtension == "safetensors" { store.openCheckpoint(url) }
            else { review.groups = [MaterialReviewGroup(id: url.lastPathComponent, candidates: [MapReviewCandidate(id: url.path, label: url.lastPathComponent, mapURL: url, numeric: true)])]; review.selectedGroupId = url.lastPathComponent }
        }
    }

    private var reviewView: some View {
        NavigationSplitView {
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(review.groups) { group in
                        MaterialSidebarRow(selected: review.selectedGroupId == group.id, action: { review.selectedGroupId = group.id }) {
                            Label(group.id.replacingOccurrences(of: "_", with: " "), systemImage: "square.3.layers.3d")
                        }
                    }
                }.padding(8)
            }.focusable().focusEffectDisabled()
                .onKeyPress(.downArrow) { review.selectedGroupId = MaterialSidebarSelection.next(review.selectedGroupId, in: review.groups.map(\.id), direction: 1); return .handled }
                .onKeyPress(.upArrow) { review.selectedGroupId = MaterialSidebarSelection.next(review.selectedGroupId, in: review.groups.map(\.id), direction: -1); return .handled }
                .navigationSplitViewColumnWidth(min: 180, ideal: 230)
        } detail: {
            if let selected = review.selected {
                VStack(spacing: 0) {
                    HStack {
                        Label("Sample: \(selected.id)", systemImage: "photo").font(.headline).textSelection(.enabled)
                        Spacer()
                    }.padding(.horizontal, 12).padding(.top, 12)
                    HStack {
                        let candidateId = selected.candidates.first(where: { $0.id == review.selectedCandidateId })?.id ?? selected.candidates.last!.id
                        Picker("Candidate", selection: Binding(get: { candidateId }, set: { review.selectedCandidateId = $0 })) {
                            ForEach(selected.candidates) { candidate in Text(candidate.label).tag(candidate.id) }
                        }.frame(minWidth: 260, idealWidth: 350)
                            .help("The decision and note apply to this named candidate for the sample shown above. Map pane titles identify each visible result.")
                        Picker("Decision", selection: Binding(get: { review.decisions[candidateId] ?? "unreviewed" }, set: { review.decisions[candidateId] = $0 })) {
                            Text("Unreviewed").tag("unreviewed")
                            Text("Usable").tag("usable")
                            Text("Needs work").tag("needs_work")
                            Text("Reject").tag("reject")
                        }.frame(width: 240)
                            .help("Record whether this candidate produces useful surface detail. A score alone cannot judge material quality.")
                        TextField("Detail, noise or relief observations", text: Binding(get: { review.notes[candidateId] ?? "" }, set: { review.notes[candidateId] = $0 }))
                        Button("Save Decisions & Notes…") { review.saveReview() }
                            .help("Save decisions and notes to a separate file you can reopen. Original maps and selected models stay intact.")
                    }.padding(12)
                    Divider()
                    ReviewWorkbenchView(candidates: selected.candidates, blendURL: review.blendURL,
                        onMissingSource: { url in
                            review.removeMissingSource(url)
                        }, onConfirmReview: { candidateID, recommendation in
                            review.decisions[candidateID] = recommendation == .approve ? "usable" : recommendation == .exclude ? "reject" : "needs_work"
                            review.selectedCandidateId = candidateID
                        })
                }
            } else {
                ContentUnavailableView("Inspect material details", systemImage: "photo.on.rectangle", description: Text("Open original maps or a review manifest. Each map supports native resolution, pan, zoom, pop-out and lossless export."))
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button("Open Review…", systemImage: "folder") { review.chooseManifest() }
                    .help("Reopen saved review decisions or the review manifest produced by a checkpoint comparison.")
                Button("Open Maps…", systemImage: "photo") { review.chooseMaps() }
                    .help("Inspect original PNG or EXR maps at full source resolution. Export Original saves a map; Export Visible Maps saves all visible panes together.")
            }
        }
    }
    private var compareView: some View {
        VStack(spacing: 0) {
            HStack {
                Button("Test Photo…", systemImage: "photo") { store.chooseSourceImage() }
                    .help("Use the same prepared diffuse at the chosen grid, for every checkpoint. A selected dataset material is used when no separate photo is chosen.")
                VStack(alignment: .leading, spacing: 3) {
                    Text(store.sourceImageURL.map { "Source photo: \($0.lastPathComponent)" }
                         ?? store.selectedSampleId.map { "Dataset material: \($0)" } ?? "Choose a photo or dataset material")
                        .lineLimit(1).truncationMode(.middle)
                    if store.sourceImageURL != nil {
                        Text("This saved photo is used instead of the dataset selection.").font(.caption)
                    }
                }.foregroundStyle(.secondary)
                if store.sourceImageURL != nil {
                    Button("Use Dataset Material") { store.sourceImageURL = nil }
                        .disabled(store.isBusy || store.selectedSample?.maps["input"] == nil)
                        .help("Switch back to the material currently selected in Dataset. Its diffuse and reference map will be used for this comparison.")
                }
                Spacer()
                Button("Choose Checkpoints…") { store.chooseCheckpoint() }
                    .help("Select one checkpoint to compare with its material base, or multiple saved checkpoints predicting the same map type. Their exact hashes are recorded with the results.")
                Button("Run Comparison", systemImage: "play.fill") { store.compare() }
                    .buttonStyle(.glassProminent).disabled(store.isBusy || !comparisonReady)
                    .help("Run checked candidates one at a time on Metal, then inspect their matching full-resolution outputs.")
            }.padding()
            if store.sourceImageURL != nil {
                DisclosureGroup("Prepare test diffuse") {
                    HStack(alignment: .top, spacing: 20) {
                        VStack {
                            DoubleControl(title: "Tilt X", value: $store.testPhotoSettings.rotationX, range: -45...45, suffix: "°")
                            DoubleControl(title: "Tilt Y", value: $store.testPhotoSettings.rotationY, range: -45...45, suffix: "°")
                            DoubleControl(title: "Rotate", value: $store.testPhotoSettings.rotationZ, range: -45...45, suffix: "°")
                        }
                        VStack {
                            FloatControl(title: "Lighting balance", value: $store.testPhotoSettings.lightingStrength, range: 0...1)
                            FloatControl(title: "Lighting scale", value: $store.testPhotoSettings.lightingRadius, range: 0.01...0.5)
                            Text("The shared Studio pipeline prepares one diffuse map at the selected grid. Inspect it alongside predictions before rating the result.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }.padding(.horizontal).padding(.bottom, 10).disabled(store.isBusy)
            }
            ScrollView(.horizontal) {
                HStack {
                    ForEach(store.checkpoints) { checkpoint in
                        Toggle(isOn: Binding(get: { store.comparisonCheckpointIds.contains(checkpoint.id) }, set: { enabled in
                            if enabled { store.comparisonCheckpointIds.insert(checkpoint.id) } else { store.comparisonCheckpointIds.remove(checkpoint.id) }
                        })) { Text("\(checkpoint.title) · \(checkpoint.target) · step \(checkpoint.step)") }.toggleStyle(.checkbox)
                    }
                }.padding(.horizontal)
            }.disabled(store.isBusy)
            Toggle("Include the material base before refinement", isOn: $store.comparisonIncludesBase)
                .toggleStyle(.checkbox).disabled(store.isBusy).padding(.horizontal).padding(.vertical, 8)
            if !comparisonReady && !store.isBusy {
                Text("Choose a photo or dataset material, then select one checkpoint plus the material base, or two checkpoints for the same map type.")
                    .font(.caption).foregroundStyle(.secondary).padding(.horizontal).padding(.vertical, 8)
            }
            Divider()
            if !store.comparisonCandidates.isEmpty { ReviewWorkbenchView(candidates: store.comparisonCandidates) }
            else { ContentUnavailableView("Compare material models", systemImage: "rectangle.split.2x1", description: Text("Compare saved checkpoints on the same photo and inspect matching details. Each result identifies its source sample, training step and exact model. Source photos and dataset targets are available in Maps.")) }
            WorkbenchActivityView(store: store)
        }
    }
}

struct WorkbenchActivityView: View {
    @Bindable var store: WorkbenchStore
    var body: some View {
        HStack {
            if store.isBusy { ProgressView().controlSize(.small) }
            Text(store.activity).lineLimit(2).font(.caption)
            Spacer()
            if let url = store.lastOutputURL { Button("Show Results") { NSWorkspace.shared.activateFileViewerSelecting([url]) } }
            if store.isBusy { WorkbenchStopButtons(store: store) }
        }.padding(12)
    }
}

struct WorkbenchRuntimeView: View {
    @AppStorage(StudioPreferences.developerModeKey, store: StudioPreferences.defaults) private var developerMode = false
    @Bindable var store: WorkbenchStore
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack {
            Form {
                Section("Local Python and workspace") {
                    runtimeRow("Python", path: store.pythonPath, choose: store.choosePython)
                    runtimeRow("Workspace", path: store.workspacePath, choose: store.chooseWorkspace)
                    Text("Choose a working folder for datasets, results and logs. Use a local Python environment with the dependencies required by your model backend.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Material base") {
                    runtimeRow("Weights", path: store.modelDirectory, choose: store.chooseEncoder)
                    if developerMode {
                        runtimeRow("Architecture source", path: store.codeDirectory, choose: store.chooseEncoderCode)
                    }
                    HStack {
                        Button("Download Base Model") { store.installEncoder() }.disabled(store.isBusy)
                        Button("Remove Downloaded Base Weights", role: .destructive) { store.removeDownloadedEncoder() }.disabled(store.isBusy)
                    }
                    Text("Keep the base while refining a LoRA. A full fused checkpoint contains its weights; downloaded base weights can then be removed and obtained again.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Settings") {
                    Toggle("Developer mode", isOn: $developerMode)
                    Text("Expose adapter controls and export a full fused checkpoint alongside the separate LoRA.")
                        .font(.caption).foregroundStyle(.secondary)
                }

            }.formStyle(.grouped)
            Button("Done") { store.saveConfiguration(); dismiss() }.keyboardShortcut(.defaultAction).padding()
        }
    }
    private func runtimeRow(_ title: String, path: String, choose: @escaping () -> Void) -> some View {
        LabeledContent(title) { Text(path).lineLimit(2).font(.caption).textSelection(.enabled); Button("Locate…", action: choose) }
    }
}
