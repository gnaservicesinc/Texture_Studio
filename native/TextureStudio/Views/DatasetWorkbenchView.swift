import SwiftUI

struct DatasetWorkbenchView: View {
    @Bindable var store: WorkbenchStore
    @State private var query = ""
    @State private var reviewNote = ""
    @State private var viewport = InspectionViewport()

    private var matchingMaterials: [WorkbenchMaterial] {
        guard let original = store.dataset?.materials else { return [] }
        let materials = original.map { WorkbenchMaterial(materialId: $0.id, samples: $0.samples.filter { $0.split == "train" }) }
            .filter { !$0.samples.isEmpty }
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return materials }
        return materials.filter { material in
            material.materialId.localizedStandardContains(text)
                || material.samples.contains { $0.id.localizedStandardContains(text) }
        }
    }

    var body: some View {
        HSplitView {
            VStack(spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Materials").font(.headline)
                        if let dataset = store.dataset {
                            Text("\(dataset.materials.count) materials · \(dataset.samples.filter { $0.split == "train" }.count) training crops")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    Button { store.chooseDataset() } label: {
                        Label("Open Dataset…", systemImage: "folder")
                    }
                    .labelStyle(.iconOnly)
                    .help("Open a prepared material dataset")
                    .disabled(store.isBusy)
                }
                .padding(12)
                Divider()
                if store.dataset == nil {
                    ContentUnavailableView {
                        Label("Open a dataset", systemImage: "square.stack.3d.up")
                    } description: {
                        Text("Choose your prepared crops to inspect and review their original maps.")
                    } actions: {
                        Button("Open Dataset…") { store.chooseDataset() }
                            .buttonStyle(.glassProminent)
                            .disabled(store.isBusy)
                    }
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 3) {
                                ForEach(matchingMaterials) { material in
                                    Text(material.materialId.replacingOccurrences(of: "_", with: " "))
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
            }
            .frame(minWidth: 245, idealWidth: 290, maxWidth: 380)
            VStack(spacing: 0) {
                if let sample = store.selectedSample {
                    cropHeader(sample)
                    Divider()
                    if let map = store.selectedMap {
                        MapInspectionView(url: map.url, numeric: store.selectedRole != "input",
                                          viewport: viewport, title: mapTitle(store.selectedRole))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        ContentUnavailableView("Map unavailable", systemImage: "photo", description: Text("Choose a map available in this crop."))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    Divider()
                    cropReview(sample)
                } else {
                    ContentUnavailableView("Select a crop", systemImage: "photo.on.rectangle", description: Text("Select a crop to inspect its maps and review its training use."))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(minWidth: 470)
        }
        .searchable(text: $query, prompt: "Find a material or crop")
        .onChange(of: store.selectedSampleId) { _, _ in
            reviewNote = store.selectedSample?.note ?? ""
            viewport.fitToView = true
            if store.selectedMap == nil {
                store.selectedRole = store.selectedSample?.maps["height"] == nil ? "input" : "height"
            }
        }
        .onChange(of: store.dataset?.indexSha256) { _, _ in
            selectVisibleCrop()
            reviewNote = store.selectedSample?.note ?? ""
        }
        .onAppear { selectVisibleCrop(); reviewNote = store.selectedSample?.note ?? "" }
    }

    private func selectVisibleCrop() {
        if store.selectedSample?.split != "train" {
            store.selectedSampleId = matchingMaterials.first?.samples.first?.id
        }
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
                    Text((store.selectedMaterialId ?? sample.id).replacingOccurrences(of: "_", with: " "))
                        .font(.headline)
                    Text("\(sample.id) · \(sample.width) × \(sample.height)")
                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Spacer()
                Button { store.inspectSelectedMap() } label: {
                    Label("Open Full Quality", systemImage: "arrow.up.left.and.arrow.down.right")
                }
                .disabled(store.selectedMap == nil)
                .help("Open the original map in an independent window with pixel zoom, pan, lossless export and GIMP access.")
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
                Button(sample.status == "excluded" ? "Reapprove" : "Approve") {
                    store.curateSelected(status: "approved", note: reviewNote.isEmpty ? nil : reviewNote)
                }
                .disabled(sample.status == "approved")
                .help("Mark this crop as reviewed and suitable for training. The original map files remain unchanged.")
                Button("Exclude") {
                    store.curateSelected(status: "excluded", note: reviewNote.isEmpty ? nil : reviewNote)
                }
                .disabled(sample.status == "excluded")
                .help("Keep this crop and its files, but leave it out of training. Reapprove it at any time.")
            }
            HStack {
                TextField("Review note", text: $reviewNote).textFieldStyle(.roundedBorder)
                Button("Save Note") {
                    store.curateSelected(status: editableStatus(sample.status), note: reviewNote)
                }
                .disabled(reviewNote.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Menu {
                    Button("Mark Unreviewed") { store.curateSelected(status: "unreviewed") }
                } label: { Label("More Review Actions", systemImage: "ellipsis") }
                .menuStyle(.borderlessButton).fixedSize().help("Mark this crop for another review")
            }
            Text("Approve usable crops or exclude problems. A small automatic check sample is managed in the background; original files stay in place.")
                .font(.caption).foregroundStyle(.secondary)
            if let map = store.selectedMap {
                DisclosureGroup("Original file") {
                    Text(map.path).textSelection(.enabled)
                    if let hash = map.sha256 { Text("SHA256: \(hash)").textSelection(.enabled) }
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
                Text("Crop \(sample.id.components(separatedBy: "_").last ?? sample.id)").lineLimit(1)
                Text("\(sample.width) × \(sample.height)")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }
}
