import SwiftUI

struct DatasetWorkbenchView: View {
    @Bindable var store: WorkbenchStore
    @State private var query = ""
    @State private var expandedFolders: Set<String> = []
    @State private var expandedCrops: Set<String> = []
    @State private var reviewNote = ""
    @State private var viewport = InspectionViewport()
    @State private var selectedNode: DatasetBrowserNode?
    @State private var selectedInventoryPath: String?
    @State private var unavailableSource: String?
    @State private var tagImage: URL?

    private var matchingMaterials: [WorkbenchMaterial] {
        let materials = store.datasetBrowserMaterials
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return materials }
        return materials.filter { material in
            material.materialId.localizedStandardContains(text)
                || (material.name?.localizedStandardContains(text) ?? false)
                || material.samples.contains { $0.id.localizedStandardContains(text) }
        }
    }

    private var matchingInventory: [WorkbenchSourceInventoryItem] {
        guard store.datasetBrowserMode == .originals else { return [] }
        return store.dataset?.browserInventory(query: query) ?? []
    }

    private var selectedInventory: WorkbenchSourceInventoryItem? {
        store.dataset?.sourceInventory?.first { $0.path == selectedInventoryPath }
    }

    private var subjectGroups: [DatasetSubjectGroup] {
        var groups: [String: DatasetSubjectGroup] = [:]
        for material in matchingMaterials {
            let key = material.sourceDirectory ?? material.subjectId ?? material.id
            var group = groups[key] ?? DatasetSubjectGroup(id: key,
                name: material.sourceDirectory.map { URL(fileURLWithPath: $0).lastPathComponent } ?? material.name ?? key,
                materials: [], inventory: [])
            group.materials.append(material)
            groups[key] = group
        }
        for item in matchingInventory {
            var group = groups[item.folder] ?? DatasetSubjectGroup(id: item.folder,
                name: URL(fileURLWithPath: item.folder).lastPathComponent, materials: [], inventory: [])
            group.inventory.append(item)
            groups[item.folder] = group
        }
        return groups.values.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    var body: some View {
        VStack(spacing: 0) {
            datasetHeader
            classifierHeader
            Divider()
            HSplitView {
            VStack(spacing: 0) {
                if store.dataset != nil {
                    HStack {
                        Text("Source folders").font(.headline)
                        Spacer()
                        Button { store.addMaterialFolder() } label: {
                            Label(store.dataset?.isSourceFolder == true ? "Manage in Finder" : "Add Folder…", systemImage: "folder.badge.plus")
                        }
                        .disabled(store.isBusy)
                    }
                    .padding(12)
                    Picker("Browse", selection: $store.datasetBrowserMode) {
                        ForEach(DatasetBrowserMode.allCases) { Text($0.title).tag($0) }
                    }.pickerStyle(.segmented).padding(.horizontal, 12).padding(.bottom, 8)
                    Text(store.datasetBrowserMode == .originals
                         ? (store.dataset?.isSourceFolder == true ? "Add images or folders here. Folder changes appear automatically." : "Every original file stays here. Training eligibility does not hide source images.")
                         : "Exact \(store.datasetResolution) × \(store.datasetResolution) crops only. Folder checkboxes enable separate validation crops.")
                        .font(.caption).foregroundStyle(.secondary).padding(.horizontal, 12).padding(.bottom, 10)
                    Divider()
                }
                if store.dataset == nil {
                    recentDatasets
                } else if store.datasetBrowserSamples.isEmpty && matchingInventory.isEmpty && query.isEmpty {
                    ContentUnavailableView {
                        Label("No materials yet", systemImage: "photo.badge.plus")
                    } description: {
                        Text(store.datasetBrowserMode == .originals ? "Add a folder of original images and maps." : "No original map set can provide an exact crop at this grid. Your originals remain available in Originals.")
                    } actions: {
                        Button("Add Folder…") { store.addMaterialFolder() }
                            .buttonStyle(.glassProminent).disabled(store.isBusy)
                    }
                } else if subjectGroups.isEmpty {
                    ContentUnavailableView("No matching materials", systemImage: "line.3.horizontal.decrease", description: Text("Clear the search to see the rest of this dataset."))
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 3) {
                                ForEach(subjectGroups) { group in
                                    VStack(alignment: .leading, spacing: 3) {
                                        DatasetDisclosureRow(title: group.name, nodeID: group.id,
                                            expanded: expandedFolders.contains(group.id),
                                            selected: selectedNode == .subject(group.id),
                                            select: { selectSubject(group) },
                                            disclose: {
                                                toggle(group.id, in: &expandedFolders)
                                                selectSubject(group, reveal: false)
                                            }) {
                                                Label(group.name, systemImage: "folder").font(.callout.bold())
                                            } accessory: {
                                                if store.datasetBrowserMode == .crops,
                                                   let subject = store.dataset?.subjects?.first(where: { $0.id == (group.materials.first?.subjectId ?? group.id) }) {
                                                    Toggle("Validation", isOn: Binding(get: { subject.selected }, set: {
                                                        store.setSubjectValidation(subject, enabled: $0)
                                                    })).labelsHidden().toggleStyle(.checkbox)
                                                        .accessibilityLabel("Validation for \(group.name)")
                                                        .disabled(!subject.available || store.dataset?.validation?.enabled == false)
                                                        .help(subject.reason ?? "Enable one extra learning-check crop for this subject, within the dataset percentage and count limits.")
                                                }
                                            }
                                        if expandedFolders.contains(group.id) {
                                            ForEach(group.materials) { material in
                                                ForEach(material.samples) { sample in
                                                    VStack(alignment: .leading, spacing: 3) {
                                                        DatasetDisclosureRow(title: cropTitle(sample), nodeID: sample.id,
                                                            expanded: expandedCrops.contains(sample.id),
                                                            selected: selectedNode == .crop(sample.id),
                                                            select: { selectCrop(sample) },
                                                            disclose: {
                                                                toggle(sample.id, in: &expandedCrops)
                                                                selectCrop(sample, reveal: false)
                                                            }) {
                                                                DatasetCropRow(sample: sample)
                                                            } accessory: { EmptyView() }
                                                        if expandedCrops.contains(sample.id) {
                                                            ForEach(availableRoles(sample), id: \.self) { role in
                                                                MaterialSidebarRow(selected: selectedNode == .map(sample.id, role),
                                                                    action: { selectMap(sample, role: role) }) {
                                                                    Label(mapTitle(role), systemImage: mapIcon(role))
                                                                }.padding(.leading, 22)
                                                            }
                                                        }
                                                    }.id(sample.id).padding(.leading, 20)
                                                }
                                            }
                                            ForEach(group.inventory) { item in
                                                MaterialSidebarRow(selected: selectedInventoryPath == item.path,
                                                                   action: { selectInventory(item) }) {
                                                    VStack(alignment: .leading, spacing: 2) {
                                                        Label(item.name, systemImage: "photo").lineLimit(1)
                                                        Text(inventorySubtitle(item)).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                                                    }
                                                }.id(item.path).padding(.leading, 20)
                                            }
                                        }
                                    }.padding(.vertical, 3)
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
                    }
                    .font(.caption).padding(12).disabled(store.isBusy)
                }
            }
            .frame(minWidth: 290, idealWidth: 330, maxWidth: 400)
            VStack(spacing: 0) {
                if let item = selectedInventory {
                    inventoryInspection(item)
                } else if let sample = store.selectedSample {
                    cropHeader(sample)
                    Divider()
                    if let map = store.selectedMap {
                        MapInspectionView(url: store.datasetReviewURL(map), numeric: store.selectedRole != "input",
                                          viewport: viewport, title: mapTitle(store.selectedRole),
                                          sourceSHA256: store.datasetReviewSHA256(map), onMissingSource: { missing in
                                              unavailableSource = store.reportMissingDatasetSource(missing)
                                          }, displayTransform: store.datasetDisplayTransform(map, role: store.selectedRole))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        ContentUnavailableView("Map unavailable", systemImage: "photo", description: Text("Choose a map available in this material."))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    if let unavailableSource {
                        Text(unavailableSource).font(.caption).foregroundStyle(.orange).textSelection(.enabled).padding(12)
                    }
                    mapControls(sample)
                    Divider()
                    if sample.split == "validation" {
                        Text("This extra crop checks the model. All of this folder’s ordinary crops remain in training. Change its validation flag in the folder tree.")
                            .font(.callout).foregroundStyle(.secondary).padding(14)
                    } else { cropReview(sample) }
                } else {
                    datasetWelcome
                }
            }
            .frame(minWidth: 470)
            }
        }
        .searchable(text: $query, prompt: "Find a material")
        .sheet(isPresented: $store.showClassifierDatasetSettings) {
            ClassifierDatasetSettingsSheet(store: store)
        }
        .sheet(isPresented: Binding(get: { tagImage != nil }, set: { if !$0 { tagImage = nil } })) {
            if let tagImage { SourceImageTagsSheet(store: store, url: tagImage) }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard !store.isBusy, let url = urls.first else { return false }
            if urls.count == 1 && (url.lastPathComponent == "dataset.json" || FileManager.default.fileExists(atPath: url.appendingPathComponent("dataset.json").path)) {
                store.openDataset(url)
                return true
            }
            guard urls.allSatisfy({ source in
                FileManager.default.fileExists(atPath: source.path)
            }) else { return false }
            store.addMaterialFolders(urls)
            return true
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

        .onChange(of: store.selectedSampleId) { _, _ in
            reviewNote = store.selectedSample?.note ?? ""
            viewport.fitToView = true
            selectAvailableMap()
            if selectedNode?.sampleID != store.selectedSampleId && !isSelectedSubject {
                selectedNode = store.selectedSampleId.map(DatasetBrowserNode.crop)
                revealSelection()
            }
        }
        .onChange(of: store.datasetBrowserMode) { _, _ in
            selectedInventoryPath = nil
            unavailableSource = nil
            selectVisibleCrop()
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
        .onChange(of: query) { _, _ in selectVisibleCrop() }
        .onAppear { selectVisibleCrop(); selectAvailableMap(); revealSelection(); selectedNode = store.selectedSampleId.map(DatasetBrowserNode.crop); viewport.fit(); reviewNote = store.selectedSample?.note ?? "" }
    }

    private var datasetHeader: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text(store.dataset == nil ? "Your datasets" : store.datasetName).font(.title3.bold()).lineLimit(1)
                if let dataset = store.dataset {
                    Text("\(store.datasetInventorySummary) · \(dataset.sampleCount ?? dataset.samples.count) training crops · \(store.datasetResolution) × \(store.datasetResolution)")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Open the folder containing your original images and maps. You can also drop folders here.").font(.caption).foregroundStyle(.secondary)
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
            Button { store.showNewDatasetSheet = true } label: { Label("New Folder…", systemImage: "folder.badge.plus") }
            Button { store.chooseDataset() } label: { Label("Open Folder…", systemImage: "folder") }
            if store.dataset != nil {
                Button("Add Images or Folders…", systemImage: "folder.badge.plus") { store.addMaterialFolder() }
                Menu {
                    Button("Assign Maps Manually…", systemImage: "photo.badge.plus") { store.showAddMaterialSheet = true }
                    if store.dataset?.isSourceFolder == true {
                        Button("Remember Missing Map Tags", systemImage: "tag") { store.repairSourceTags() }
                    }
                    Button("Show in Finder", systemImage: "folder") { store.revealDataset() }
                    Button("Close Dataset", systemImage: "xmark") { store.closeDataset() }
                } label: { Label("Dataset", systemImage: "ellipsis.circle") }
                .help("Show the dataset folder or close the dataset")
            }
        }
        .padding(14).disabled(store.isBusy)
    }

    private var classifierHeader: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Diffuse type classifier").font(.callout.bold())
                Text(store.classifierDatasetResult?.summary ?? "Refresh separately when you add new material types; review labels before Create ML training.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            Button("Create / Refresh Classifier Dataset", systemImage: "arrow.clockwise") { store.refreshClassifierDataset() }
            Button("Settings…", systemImage: "gearshape") { store.showClassifierDatasetSettings = true }
            Button("Show Folder", systemImage: "folder") { store.revealClassifierDataset() }
                .disabled(!FileManager.default.fileExists(atPath: store.classifierDatasetPath))
        }
        .padding(.horizontal, 14).padding(.bottom, 12).disabled(store.isBusy)
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
            Label(store.dataset == nil ? "Open your source folder" : store.datasetBrowserSamples.isEmpty && matchingInventory.isEmpty ? "Add your source images" : "Select an original or crop", systemImage: "square.stack.3d.up")
        } description: {
            if store.dataset == nil {
                Text("The folder is your dataset. Open an existing folder to inspect its current images, or create an empty source folder and add material subfolders.")
            } else if store.datasetBrowserSamples.isEmpty && matchingInventory.isEmpty {
                Text("Add original images and material maps. Use image tags to identify their material category and map type for training.")
            } else {
                Text("Choose a material on the left to inspect its maps, edit tags and review notes, or enable folder validation.")
            }
        } actions: {
            if store.dataset == nil {
                Button("Choose Source Folder…") { store.chooseDataset() }.buttonStyle(.glassProminent)
                Button("New Source Folder…") { store.showNewDatasetSheet = true }
            } else if store.datasetBrowserSamples.isEmpty && matchingInventory.isEmpty {
                Button("Add Folder…") { store.addMaterialFolder() }.buttonStyle(.glassProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).disabled(store.isBusy)
    }

    private var isSelectedSubject: Bool {
        guard case .subject(let id) = selectedNode else { return false }
        return subjectGroups.first(where: { $0.id == id })?.materials
            .contains(where: { $0.samples.contains(where: { $0.id == store.selectedSampleId }) }) ?? false
    }
    private func toggle(_ id: String, in values: inout Set<String>) {
        if values.contains(id) { values.remove(id) } else { values.insert(id) }
    }
    private func selectSubject(_ group: DatasetSubjectGroup, reveal: Bool = true) {
        let samples = group.materials.flatMap(\.samples)
        guard let sample = samples.first(where: { $0.id == store.selectedSampleId }) ?? samples.first else {
            if let item = group.inventory.first { selectInventory(item) }
            if reveal { expandedFolders.insert(group.id) }
            return
        }
        selectedInventoryPath = nil
        unavailableSource = nil
        selectedNode = .subject(group.id)
        store.selectDatasetSample(sample)
        viewport.fit()
        if reveal { expandedFolders.insert(group.id) }
        expandedCrops.insert(sample.id)
    }
    private func selectCrop(_ sample: WorkbenchSample, reveal: Bool = true) {
        selectedInventoryPath = nil
        unavailableSource = nil
        selectedNode = .crop(sample.id)
        store.selectDatasetSample(sample)
        viewport.fit()
        if reveal { expandedCrops.insert(sample.id) }
    }
    private func selectMap(_ sample: WorkbenchSample, role: String) {
        selectedInventoryPath = nil
        unavailableSource = nil
        selectedNode = .map(sample.id, role)
        store.selectDatasetSample(sample, role: role)
    }
    private func availableRoles(_ sample: WorkbenchSample) -> [String] {
        let preferred = ["input", "height", "roughness", "normal", "ao", "ancillary", "unknown"]
        return preferred.filter { sample.maps[$0] != nil } + sample.maps.keys.filter { !preferred.contains($0) }.sorted()
    }
    private func cropTitle(_ sample: WorkbenchSample) -> String {
        if sample.split == "original" { return "Original \(sample.width) × \(sample.height)" }
        return sample.split == "validation" ? "Learning check" : (sample.sourceRegionId ?? "original").replacingOccurrences(of: "_", with: " ").capitalized + " crop"
    }
    private func mapIcon(_ role: String) -> String {
        switch role {
        case "input": "photo"
        case "height": "square.3.layers.3d"
        case "roughness": "circle.lefthalf.filled"
        case "normal": "arrow.up.right"
        default: "photo"
        }
    }
    private func mapControls(_ sample: WorkbenchSample) -> some View {
        HStack(spacing: 10) {
            ForEach(availableRoles(sample), id: \.self) { role in
                Button { selectMap(sample, role: role) } label: {
                    Label(mapTitle(role), systemImage: mapIcon(role))
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .background(store.selectedRole == role ? Color.accentColor.opacity(0.28) : Color.secondary.opacity(0.12),
                            in: RoundedRectangle(cornerRadius: 6))
                .accessibilityAddTraits(store.selectedRole == role ? [.isSelected] : [])
                .accessibilityIdentifier("dataset-map-\(role)")
            }
            Spacer(minLength: 0)
            Button { viewport.fit() } label: { Image(systemName: "arrow.down.right.and.arrow.up.left").padding(6).contentShape(Rectangle()) }
                .buttonStyle(.plain)
                .help("Fit image").accessibilityLabel("Fit image")
            Button { viewport.setActualSize() } label: { Image(systemName: "1.magnifyingglass").padding(6).contentShape(Rectangle()) }
                .buttonStyle(.plain)
                .help("Actual pixels").accessibilityLabel("Actual pixels")
        }
        .padding(10).disabled(store.isBusy)
    }
    private func revealSelection() {
        guard let sample = store.selectedSampleId,
              let folder = subjectGroups.first(where: { $0.materials.contains { $0.samples.contains { $0.id == sample } } }) else { return }
        expandedFolders.insert(folder.id)
        if selectedNode != .subject(folder.id) { expandedCrops.insert(sample) }
    }
    private func selectVisibleCrop() {
        if let selectedInventoryPath, matchingInventory.contains(where: { $0.path == selectedInventoryPath }) { return }
        if !matchingMaterials.flatMap(\.samples).contains(where: { $0.id == store.selectedSampleId }) {
            store.selectedSampleId = matchingMaterials.first?.samples.first?.id
        }
        if store.selectedSampleId == nil, let item = matchingInventory.first {
            selectInventory(item)
            expandedFolders.insert(item.folder)
        }
    }
    private func selectAvailableMap() {
        guard store.selectedMap == nil, let sample = store.selectedSample,
              let role = ["input", "height", "roughness", "normal"].first(where: { sample.maps[$0] != nil }) else { return }
        store.selectedRole = role
    }
    private func moveSelection(_ direction: Int, proxy: ScrollViewProxy) {
        let ids = subjectGroups.flatMap { group in
            group.materials.flatMap(\.samples).map(\.id) + group.inventory.map(\.path)
        }
        let current = selectedInventoryPath ?? store.selectedSampleId
        guard let next = MaterialSidebarSelection.next(current, in: ids, direction: direction) else { return }
        if let item = matchingInventory.first(where: { $0.path == next }) {
            selectInventory(item)
            expandedFolders.insert(item.folder)
            proxy.scrollTo(item.path)
            return
        }
        selectedInventoryPath = nil
        unavailableSource = nil
        store.selectedSampleId = next
        selectedNode = store.selectedSampleId.map(DatasetBrowserNode.crop)
        revealSelection()
        if let id = store.selectedSampleId { proxy.scrollTo(id) }
    }

    private func cropHeader(_ sample: WorkbenchSample) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(store.selectedMaterialName ?? (store.selectedMaterialId ?? sample.id).replacingOccurrences(of: "_", with: " "))
                        .font(.headline)
                    Text("\(sample.id) · \(sample.width) × \(sample.height) · \(sample.split == "original" ? "Original source set" : sample.split == "validation" ? "Extra validation crop" : "Training crop")")
                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    if sample.split == "original", min(sample.width, sample.height) < store.datasetResolution {
                        Text("Available for inspection. This original cannot supply a \(store.datasetResolution) × \(store.datasetResolution) training crop.")
                            .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                    }
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
                    Group {
                        if let map = store.selectedMap {
                            Button("Edit Image Tags / Folder Name…", systemImage: "tag") { tagImage = URL(fileURLWithPath: map.originalSourcePath ?? map.path) }
                            if URL(fileURLWithPath: map.originalSourcePath ?? map.path).deletingLastPathComponent().resolvingSymlinksInPath().path != store.datasetFolderURL?.resolvingSymlinksInPath().path {
                                Button("Move Material Folder Out…", systemImage: "folder.badge.minus") {
                                    store.moveSourceFolderOut(URL(fileURLWithPath: map.originalSourcePath ?? map.path).deletingLastPathComponent())
                                }
                            }
                        }
                        Button("Show Source Folder in Finder") { store.revealSelectedOriginalSource() }
                    }
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
                Text(sample.split == "original" ? "Original diffuse color variants for this source set." : "Training randomly chooses an available diffuse color for this target crop.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Label(mapTitle(store.selectedRole), systemImage: mapIcon(store.selectedRole))
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
            Text("Approve usable materials or exclude problems. All maps share these review settings. Enable validation at the folder level.")
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

    private func selectInventory(_ item: WorkbenchSourceInventoryItem) {
        selectedInventoryPath = item.path
        selectedNode = nil
        unavailableSource = nil
        store.selectedSampleId = nil
        viewport.fit()
    }

    private func inventorySubtitle(_ item: WorkbenchSourceInventoryItem) -> String {
        let dimensions = item.width.flatMap { width in item.height.map { "\(width) × \($0)" } } ?? "Dimensions unavailable"
        let state: String
        switch item.state {
        case "needs_assignment": state = "Needs map assignment"
        case "ambiguous": state = "Check map assignment"
        case "header_error": state = "Image needs review"
        case "missing": state = "File unavailable"
        default: state = mapTitle(item.role ?? "unknown")
        }
        return "\(dimensions) · \(state)"
    }

    private func inventoryInspection(_ item: WorkbenchSourceInventoryItem) -> some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(item.name).font(.headline)
                        Text(inventorySubtitle(item)).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if store.dataset?.isSourceFolder == true {
                        Button("Edit Tags…", systemImage: "tag") { tagImage = item.url }
                    }
                    Button("Assign Maps…", systemImage: "photo.badge.plus") {
                        store.pendingMapAssignment = item
                        store.showAddMaterialSheet = true
                    }
                    Menu {
                        Button("Open Original", systemImage: "arrow.up.forward.app") { NSWorkspace.shared.open(item.url) }
                        Button("Show in Finder", systemImage: "folder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                        if store.dataset?.isSourceFolder == true,
                           let root = store.datasetFolderURL,
                           item.url.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL.path != root.resolvingSymlinksInPath().standardizedFileURL.path {
                            Divider()
                            Button("Move Folder Out…", systemImage: "folder.badge.minus") { store.moveSourceFolderOut(item.url.deletingLastPathComponent()) }
                        }
                    } label: { Label("Actions", systemImage: "ellipsis.circle") }
                }
                Text(item.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                if let label = item.classLabel, !label.isEmpty { Label(label, systemImage: "tag").font(.callout) }
                if item.sourceVerification == "verified" { Label("Original bytes verified", systemImage: "checkmark.seal").font(.caption).foregroundStyle(.secondary) }
                if let issue = item.issue {
                    Text(issue).font(.callout).foregroundStyle(.orange).textSelection(.enabled)
                }
                Text("Original file retained in this dataset. Map assignment and training eligibility do not change the source image.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(14)
            Divider()
            MapInspectionView(url: item.url, numeric: ["height", "roughness", "normal", "ancillary"].contains(item.role ?? ""),
                              viewport: viewport, title: item.name, sourceSHA256: nil, onMissingSource: { missing in
                unavailableSource = store.reportMissingDatasetSource(missing)
            })
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            if let unavailableSource {
                Text(unavailableSource).font(.caption).foregroundStyle(.orange).textSelection(.enabled).padding(12)
            }
        }
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
        case "ao": "Ambient Occlusion"
        case "ancillary": "Additional map"
        case "unknown": "Unassigned image"
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
                Text(sample.split == "original" ? "Original source" : sample.split == "validation" ? "Learning check" : (sample.sourceRegionId ?? "original").replacingOccurrences(of: "_", with: " ").capitalized + " crop").lineLimit(1)
                Text("\(sample.width) × \(sample.height) · \(sample.split == "original" ? "Original pixels" : sample.split == "validation" ? "Extra validation crop" : "Training crop")")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }
}

private struct DatasetSubjectGroup: Identifiable {
    let id: String
    let name: String
    var materials: [WorkbenchMaterial]
    var inventory: [WorkbenchSourceInventoryItem]
}
