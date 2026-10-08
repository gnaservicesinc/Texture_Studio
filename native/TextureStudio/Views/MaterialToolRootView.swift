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
                    .help("Configure the local Python environment and working folder. Archived experiments have separate optional dependencies.")
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
            else if url.pathExtension == "pt" { store.openCheckpoint(url) }
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
                    ReviewWorkbenchView(candidates: selected.candidates, blendURL: review.blendURL)
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
                Button("Source Photo…", systemImage: "photo") { store.chooseSourceImage() }
                    .help("Use one native crop, up to 2048 pixels per side, for every checkpoint. A selected dataset crop is used when no separate photo is chosen.")
                VStack(alignment: .leading, spacing: 3) {
                    Text(store.sourceImageURL.map { "Source photo: \($0.lastPathComponent)" }
                         ?? store.selectedSampleId.map { "Dataset crop: \($0)" } ?? "Choose a photo or dataset crop")
                        .lineLimit(1).truncationMode(.middle)
                    if store.sourceImageURL != nil {
                        Text("This saved photo is used instead of the dataset selection.").font(.caption)
                    }
                }.foregroundStyle(.secondary)
                if store.sourceImageURL != nil {
                    Button("Use Dataset Crop") { store.sourceImageURL = nil }
                        .disabled(store.isBusy || store.selectedSample?.maps["input"] == nil)
                        .help("Switch back to the crop currently selected in Dataset. Its source photo and reference map will be used for this comparison.")
                }
                Spacer()
                Button("Choose Checkpoints…") { store.chooseCheckpoint() }
                    .help("Select one checkpoint to compare with its untrained base, or multiple saved checkpoints predicting the same map type. Their exact hashes are recorded with the results.")
                Button("Run Comparison", systemImage: "play.fill") { store.compare() }
                    .buttonStyle(.glassProminent).disabled(store.isBusy || !comparisonReady)
                    .help("Run checked candidates one at a time on Metal, then inspect their matching full-resolution outputs.")
            }.padding()
            ScrollView(.horizontal) {
                HStack {
                    ForEach(store.checkpoints) { checkpoint in
                        Toggle(isOn: Binding(get: { store.comparisonCheckpointIds.contains(checkpoint.id) }, set: { enabled in
                            if enabled { store.comparisonCheckpointIds.insert(checkpoint.id) } else { store.comparisonCheckpointIds.remove(checkpoint.id) }
                        })) { Text("\(checkpoint.title) · \(checkpoint.target) · step \(checkpoint.step)") }.toggleStyle(.checkbox)
                    }
                }.padding(.horizontal)
            }.disabled(store.isBusy)
            if store.checkpoints.contains(where: { !$0.supportsStudioInference }) {
                DisclosureGroup("Archive comparison options") {
                    Toggle("Include the experiment’s original untrained baseline", isOn: $store.comparisonIncludesBase)
                        .toggleStyle(.checkbox)
                        .help("For archived DINOv2 experiments, the fresh material head starts flat: 0.5 displacement/roughness or a flat OpenGL normal. This records the original before-training baseline; it does not enable Studio inference.")
                        .disabled(store.isBusy)
                }.padding(.horizontal).padding(.vertical, 8)
            }
            if !comparisonReady && !store.isBusy {
                Text("Choose a photo or dataset crop, then select one checkpoint plus the untrained base, or two checkpoints for the same map type.")
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
            if store.isBusy { Button(store.isStopping ? "Stopping…" : (store.isTraining ? "Stop and Save" : "Stop")) { store.stop() }.disabled(store.isStopping) }
        }.padding(12)
    }
}

struct WorkbenchRuntimeView: View {
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
                if store.checkpoints.contains(where: { !$0.supportsStudioInference }) {
                    DisclosureGroup("Archive · old experiment dependencies") {
                        runtimeRow("DINOv2 archive weights", path: store.modelDirectory, choose: store.chooseEncoder)
                        runtimeRow("DINOv2 archive source", path: store.codeDirectory, choose: store.chooseEncoderCode)
                        HStack {
                            Button("Download Archive Dependencies") { store.installEncoder() }.disabled(store.isBusy)
                                .help("Obtain the pinned DINOv2 weights and source only for comparing archived experiments. This does not enable training or Studio inference.")
                            Button("Remove Downloaded Archive Dependencies", role: .destructive) { store.removeDownloadedEncoder() }.disabled(store.isBusy)
                                .help("Remove only the downloaded archive dependencies. External copies and saved checkpoints remain intact.")
                        }
                        Text("These dependencies are optional compatibility tools for archived experiments. Their original model identity is retained; no current production model is substituted.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

            }.formStyle(.grouped)
            Button("Done") { store.saveConfiguration(); dismiss() }.keyboardShortcut(.defaultAction).padding()
        }
    }
    private func runtimeRow(_ title: String, path: String, choose: @escaping () -> Void) -> some View {
        LabeledContent(title) { Text(path).lineLimit(2).font(.caption).textSelection(.enabled); Button("Locate…", action: choose) }
    }
}
