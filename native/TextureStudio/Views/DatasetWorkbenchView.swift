import SwiftUI

struct DatasetWorkbenchView: View {
    @Bindable var store: WorkbenchStore
    @State private var query = ""
    @State private var splitFilter = "all"
    @State private var reviewNote = ""
    @State private var viewport = InspectionViewport()
    @State private var showRemoveMaterial = false

    private var matchingMaterials: [WorkbenchMaterial] {
        guard let original = store.dataset?.materials else { return [] }
        let materials = original.map { material in
            WorkbenchMaterial(materialId: material.id, samples: material.samples.filter { splitFilter == "all" || $0.split == splitFilter }, name: material.name)
        }
            .filter { !$0.samples.isEmpty }
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return materials }
        return materials.filter { material in
            material.materialId.localizedStandardContains(text)
                || (material.name?.localizedStandardContains(text) ?? false)
                || material.samples.contains { $0.id.localizedStandardContains(text) }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            datasetHeader
            Divider()
            HSplitView {
            VStack(spacing: 0) {
                if store.dataset != nil {
                    HStack {
                        Text("Materials").font(.headline)
                        Spacer()
                        Button { store.showAddMaterialSheet = true } label: {
                            Label("Add Materials…", systemImage: "plus")
                        }
                        .disabled(store.isBusy)
                    }
                    .padding(12)
                    Picker("Show", selection: $splitFilter) {
                        Text("All materials").tag("all")
                        Text("Training").tag("train")
                        Text("Validation").tag("validation")
                    }
                    .padding(.horizontal, 12).padding(.bottom, 12)
                    Divider()
                }
                if store.dataset == nil {
                    recentDatasets
                } else if store.samples.isEmpty {
                    ContentUnavailableView {
                        Label("No materials yet", systemImage: "photo.badge.plus")
                    } description: {
                        Text("Add your diffuse and surface maps to this dataset.")
                    } actions: {
                        Button("Add Materials…") { store.showAddMaterialSheet = true }
                            .buttonStyle(.glassProminent).disabled(store.isBusy)
                    }
                } else if matchingMaterials.isEmpty {
                    ContentUnavailableView("No matching materials", systemImage: "line.3.horizontal.decrease", description: Text("Choose All materials or clear the search to see the rest of this dataset."))
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 3) {
                                ForEach(matchingMaterials) { material in
                                    Text(material.name ?? material.materialId.replacingOccurrences(of: "_", with: " "))
                                        .font(.caption.bold()).foregroundStyle(.secondary).padding(.top, 12).padding(.horizontal, 10)
                                    ForEach(material.samples) { sample in
                                        MaterialSidebarRow(selected: store.selectedSampleId == sample.id,
                                                           action: { store.selectedSampleId = sample.id }) {
                                            DatasetCropRow(sample: sample)
                                        }.id(sample.id)
                                    }
                                }
                            }.padding(8)
                        }
                        .focusable().focusEffectDisabled()
                        .onKeyPress(.downArrow) { moveSelection(1, proxy: proxy); return .handled }
                        .onKeyPress(.upArrow) { moveSelection(-1, proxy: proxy); return .handled }
                    }
                    .disabled(store.isBusy)
                }
                if store.dataset != nil {
                    Divider()
                    HStack {
                        Button("Dataset Info…", systemImage: "info.circle") { store.showDatasetInfoSheet = true }
                        Spacer()
                        Button("Delete Dataset…", role: .destructive) { store.showTrashDatasetConfirmation = true }
                    }
                    .font(.caption).padding(12).disabled(store.isBusy)
                }
            }
            .frame(minWidth: 290, idealWidth: 330, maxWidth: 400)
            VStack(spacing: 0) {
                if let sample = store.selectedSample {
                    cropHeader(sample)
                    Divider()
                    if let map = store.selectedMap {
                        MapInspectionView(url: store.datasetReviewURL(map), numeric: store.selectedRole != "input",
                                          viewport: viewport, title: mapTitle(store.selectedRole),
                                          sourceSHA256: store.datasetReviewSHA256(map), onMissingSource: { missing in
                                              store.removeMissingSource(missing, sampleID: sample.id)
                                          }, displayTransform: store.datasetDisplayTransform(map, role: store.selectedRole))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        ContentUnavailableView("Map unavailable", systemImage: "photo", description: Text("Choose a map available in this material."))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    Divider()
                    cropReview(sample)
                } else {
                    datasetWelcome
                }
            }
            .frame(minWidth: 470)
            }
        }
        .searchable(text: $query, prompt: "Find a material")
        .sheet(isPresented: $store.showNewDatasetSheet) {
            NewMaterialDatasetSheet(store: store)
        }
        .sheet(isPresented: $store.showDatasetInfoSheet) {
            MaterialDatasetInfoSheet(store: store)
        }
        .sheet(isPresented: $store.showAddMaterialSheet) {
            AddDatasetMaterialSheet(store: store)
        }
        .safeAreaInset(edge: .bottom) {
            if store.isBusy || !store.activity.isEmpty {
                HStack(spacing: 9) {
                    if store.isBusy { ProgressView().controlSize(.small) }
                    Text(store.activity).font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    if store.isBusy { WorkbenchStopButtons(store: store) }
                }.padding(12)
            }
        }
        .confirmationDialog("Delete “\(store.datasetName)”?", isPresented: $store.showTrashDatasetConfirmation, titleVisibility: .visible) {
            Button("Move Dataset to Trash", role: .destructive) { store.trashDataset() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Remove this dataset from the library and move its dataset metadata to Trash. Original source maps are kept. You can restore the metadata from Trash.\n\n\(store.datasetFolderURL?.path ?? "")")
        }
        .confirmationDialog("Remove “\(store.selectedMaterialName ?? store.selectedMaterialId ?? "material")” from this dataset?", isPresented: $showRemoveMaterial, titleVisibility: .visible) {
            Button("Remove Material", role: .destructive) { store.removeSelectedMaterial() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("All map sets for this material will be removed from the dataset. The original map files stay in place.")
        }
        .onChange(of: store.selectedSampleId) { _, _ in
            reviewNote = store.selectedSample?.note ?? ""
            viewport.fitToView = true
            selectAvailableMap()
        }
        .onChange(of: store.dataset?.indexSha256) { _, _ in
            selectVisibleCrop()
            selectAvailableMap()
            reviewNote = store.selectedSample?.note ?? ""
        }
        .onChange(of: store.dataset?.reviewSha256) { _, _ in
            selectVisibleCrop()
            reviewNote = store.selectedSample?.note ?? ""
        }
        .onChange(of: splitFilter) { _, _ in selectVisibleCrop() }
        .onChange(of: query) { _, _ in selectVisibleCrop() }
        .onAppear { selectVisibleCrop(); selectAvailableMap(); reviewNote = store.selectedSample?.note ?? "" }
    }

    private var datasetHeader: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text(store.dataset == nil ? "Your datasets" : store.datasetName).font(.title3.bold()).lineLimit(1)
                if let dataset = store.dataset {
                    Text("\(dataset.materialCount ?? dataset.materials.count) materials · \(dataset.sampleCount ?? dataset.samples.count) map sets")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Create a dataset or open a dataset folder.").font(.caption).foregroundStyle(.secondary)
                }
            }
            if !store.recentDatasets.isEmpty {
                Menu {
                    ForEach(store.recentDatasets) { location in
                        Button(location.name) { store.openDataset(URL(fileURLWithPath: location.path)) }
                    }
                } label: { Label("Switch Dataset", systemImage: "folder.badge.gearshape") }
                .help("Open a dataset from your library")
            }
            Spacer()
            Button { store.showNewDatasetSheet = true } label: { Label("New Dataset…", systemImage: "folder.badge.plus") }
            Button { store.chooseDataset() } label: { Label("Open Folder…", systemImage: "folder") }
            if store.dataset != nil {
                Menu {
                    Button("Show in Finder", systemImage: "folder") { store.revealDataset() }
                    Button("Close Dataset", systemImage: "xmark") { store.closeDataset() }
                } label: { Label("Dataset", systemImage: "ellipsis.circle") }
                .help("Show the dataset folder or close the dataset")
            }
        }
        .padding(14).disabled(store.isBusy)
    }

    private var recentDatasets: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Recent datasets").font(.headline)
            if store.recentDatasets.isEmpty {
                Text("Datasets you create or open appear here.").foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                ScrollView {
                    VStack(spacing: 4) {
                        ForEach(store.recentDatasets) { location in
                            MaterialSidebarRow(selected: false, action: { store.openDataset(URL(fileURLWithPath: location.path)) }) {
                                Label {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(location.name).lineLimit(1)
                                        Text(location.path)
                                            .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                                    }
                                } icon: { Image(systemName: "folder") }
                            }
                            .contextMenu {
                                Button("Open Dataset") { store.openDataset(URL(fileURLWithPath: location.path)) }
                                Button("Remove from Recent Datasets") { store.forgetDataset(location) }
                            }
                        }
                    }
                }
            }
            Spacer(minLength: 0)
        }.padding(14).disabled(store.isBusy)
    }

    private var datasetWelcome: some View {
        ContentUnavailableView {
            Label(store.dataset == nil ? "Make your first dataset" : store.samples.isEmpty ? "Add your material maps" : "Select a material", systemImage: "square.stack.3d.up")
        } description: {
            if store.dataset == nil {
                Text("Create a named dataset, add paired diffuse and surface maps, then review and train from it.")
            } else if store.samples.isEmpty {
                Text("Choose a diffuse map with displacement, roughness or normal maps. You can also import a folder of matching material maps.")
            } else {
                Text("Choose a material on the left to inspect its maps, edit its note and assign its training split.")
            }
        } actions: {
            if store.dataset == nil {
                Button("New Dataset…") { store.showNewDatasetSheet = true }.buttonStyle(.glassProminent)
                Button("Open Dataset Folder…") { store.chooseDataset() }
            } else if store.samples.isEmpty {
                Button("Add Materials…") { store.showAddMaterialSheet = true }.buttonStyle(.glassProminent)
                Button("Import Material Folder…") { store.importMaterialFolder() }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).disabled(store.isBusy)
    }

    private func selectVisibleCrop() {
        if !matchingMaterials.flatMap(\.samples).contains(where: { $0.id == store.selectedSampleId }) {
            store.selectedSampleId = matchingMaterials.first?.samples.first?.id
        }
    }
    private func selectAvailableMap() {
        guard store.selectedMap == nil, let sample = store.selectedSample,
              let role = ["height", "input", "roughness", "normal"].first(where: { sample.maps[$0] != nil }) else { return }
        store.selectedRole = role
    }
    private func moveSelection(_ direction: Int, proxy: ScrollViewProxy) {
        let ids = matchingMaterials.flatMap(\.samples).map(\.id)
        store.selectedSampleId = MaterialSidebarSelection.next(store.selectedSampleId, in: ids, direction: direction)
        if let id = store.selectedSampleId { proxy.scrollTo(id) }
    }

    private func cropHeader(_ sample: WorkbenchSample) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(store.selectedMaterialName ?? (store.selectedMaterialId ?? sample.id).replacingOccurrences(of: "_", with: " "))
                        .font(.headline)
                    Text("\(sample.id) · \(sample.width) × \(sample.height) map set · \(sample.split.capitalized)")
                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    if let map = store.selectedMap, let width = map.originalSourceWidth, let height = map.originalSourceHeight {
                        Text("Original source \(width) × \(height)")
                            .font(.caption).foregroundStyle(.secondary)
                        if let rectangle = map.cropRectangle, rectangle.count == 4 {
                            Text("Native crop: x\(rectangle[0]), y\(rectangle[1]), \(rectangle[2]) × \(rectangle[3])")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                Spacer()
                Menu {
                    Button("Remove Material…", role: .destructive) { showRemoveMaterial = true }
                } label: { Label("Material Actions", systemImage: "ellipsis.circle") }
                    .disabled(store.isBusy)
                Button { store.inspectSelectedMap() } label: {
                    Label("Open Full Quality", systemImage: "arrow.up.left.and.arrow.down.right")
                }
                .disabled(store.selectedMap == nil)
                .help("Open the original map in an independent window with pixel zoom, pan, lossless export and GIMP access.")
            }
            if let variants = sample.inputVariants, variants.count > 1 {
                Picker("Diffuse color", selection: Binding(get: { store.selectedDiffuseMap?.variantId ?? "" }, set: { store.selectedInputVariantId = $0 })) {
                    ForEach(variants, id: \.path) { variant in
                        Text(variant.variantId ?? variant.url.lastPathComponent).tag(variant.variantId ?? "")
                    }
                }
                .disabled(store.isBusy)
                Text("Training randomly chooses a registered diffuse color for this same target crop.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Picker("Map", selection: $store.selectedRole) {
                    ForEach(["input", "height", "roughness", "normal"].filter { sample.maps[$0] != nil }, id: \.self) { role in
                        Text(mapTitle(role)).tag(role)
                    }
                }
                .frame(maxWidth: 260)
                Spacer()
                if let map = store.selectedMap {
                    Text([map.sourceBits.map { "\($0)-bit source" }, map.encoding?.replacingOccurrences(of: "_", with: " ")]
                        .compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .padding(14)
    }

    private func cropReview(_ sample: WorkbenchSample) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(reviewStatus(sample.status), systemImage: sample.status == "excluded" ? "eye.slash" : sample.status == "approved" ? "checkmark.circle" : "circle.dotted")
                    .foregroundStyle(sample.status == "excluded" ? .secondary : .primary)
                Spacer()
                Picker("Split", selection: Binding(get: { sample.split }, set: { split in
                    store.curateSelected(status: editableStatus(sample.status), split: split, note: reviewNote)
                })) {
                    Text("Training").tag("train")
                    Text("Validation").tag("validation")
                }
                .frame(width: 210)
                .help("Assign this material to training or validation data.")
                Button(sample.status == "excluded" ? "Reapprove" : "Approve") {
                    store.curateSelected(status: "approved", note: reviewNote)
                }
                .disabled(sample.status == "approved")
                .help("Mark this material as reviewed and suitable for training. The original map files remain unchanged.")
                Button("Exclude") {
                    store.curateSelected(status: "excluded", note: reviewNote)
                }
                .disabled(sample.status == "excluded")
                .help("Keep this material and its files, but leave it out of training. Reapprove it at any time.")
            }
            HStack {
                TextField("Review note", text: $reviewNote).textFieldStyle(.roundedBorder)
                Button("Save Note") {
                    store.curateSelected(status: editableStatus(sample.status), note: reviewNote)
                }
                .disabled(reviewNote == (sample.note ?? ""))
                Menu {
                    Button("Mark Unreviewed") { store.curateSelected(status: "unreviewed", note: reviewNote) }
                } label: { Label("More Review Actions", systemImage: "ellipsis") }
                .menuStyle(.borderlessButton).fixedSize().help("Mark this material for another review")
            }
            Text("Approve usable materials or exclude problems. Edit the note or split above; changes are saved to the dataset.")
                .font(.caption).foregroundStyle(.secondary)
            if let diffuse = store.selectedDiffuseMap, let map = store.selectedMap,
               ["height", "roughness", "normal"].contains(store.selectedRole) {
                MaterialQualityReviewPanel(diffuseURL: store.datasetReviewURL(diffuse), mapURL: store.datasetReviewURL(map), mapType: store.selectedRole,
                    purpose: .dataset, diffuseSHA256: store.datasetReviewSHA256(diffuse), mapSHA256: store.datasetReviewSHA256(map),
                    diffuseTransform: store.datasetDisplayTransform(diffuse, role: "input"),
                    mapTransform: store.datasetDisplayTransform(map, role: store.selectedRole), onApply: { recommendation in
                        let status = recommendation == .approve ? "approved" : recommendation == .exclude ? "excluded" : "unreviewed"
                        store.curateSelected(status: status, note: reviewNote)
                    })
            }
            if let map = store.selectedMap {
                DisclosureGroup("Original file") {
                    Text(store.datasetReviewURL(map).path).textSelection(.enabled)
                    if let hash = store.datasetReviewSHA256(map) { Text("SHA256: \(hash)").textSelection(.enabled) }
                }
                .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .disabled(store.isBusy)
    }

    private func editableStatus(_ status: String) -> String {
        ["approved", "excluded", "unreviewed"].contains(status) ? status : "unreviewed"
    }
    private func reviewStatus(_ status: String) -> String {
        switch status {
        case "approved": "Approved for training"
        case "excluded": "Excluded from training"
        default: "Awaiting review"
        }
    }
    private func mapTitle(_ role: String) -> String {
        switch role {
        case "input": "Diffuse"
        case "height": "Displacement"
        case "roughness": "Roughness"
        case "normal": "OpenGL Normal"
        default: role.capitalized
        }
    }
}

private struct DatasetCropRow: View {
    let sample: WorkbenchSample
    var body: some View {
        HStack(spacing: 9) {
            Image(systemName: sample.status == "excluded" ? "eye.slash" : "photo").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(sample.id).lineLimit(1)
                Text("\(sample.width) × \(sample.height) · \(sample.split.capitalized)")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }
}
