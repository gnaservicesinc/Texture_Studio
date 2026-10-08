import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct MapReviewCandidate: Identifiable, Hashable {
    let id: String
    let label: String
    let mapURL: URL
    let numeric: Bool
    let sampleLabel: String?
    let detail: String?
    let role: String
    init(id: String? = nil, label: String, mapURL: URL, numeric: Bool,
         sampleLabel: String? = nil, detail: String? = nil, role: String = "map") {
        self.id = id ?? mapURL.path + "|" + label
        self.label = label; self.mapURL = mapURL; self.numeric = numeric
        self.sampleLabel = sampleLabel; self.detail = detail; self.role = role
    }
    var accessibleLabel: String { [sampleLabel, label, detail].compactMap { $0 }.joined(separator: " · ") }
    var exportFilename: String {
        guard let sampleLabel else { return mapURL.lastPathComponent }
        let stem = (sampleLabel + "-" + label + (role == "checkpoint" ? "-" + String(id.prefix(12)) : ""))
            .replacingOccurrences(of: "[^A-Za-z0-9._-]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-._"))
        return String(stem.prefix(180)) + "." + mapURL.pathExtension
    }
}

struct ReviewWorkbenchView: View {
    let candidates: [MapReviewCandidate]
    var blendURL: URL? = nil
    @State private var viewport = InspectionViewport()
    @State private var notice: String?
    @State private var inspectedHashes: [String: String] = [:]
    @State private var visibleIDs: Set<String> = []
    private var visibleCandidates: [MapReviewCandidate] {
        let selected = candidates.filter { visibleIDs.contains($0.id) }
        return selected.isEmpty ? Self.initialCandidates(candidates) : selected
    }
    static func initialCandidates(_ candidates: [MapReviewCandidate]) -> [MapReviewCandidate] {
        let predictions = candidates.filter { ["base", "checkpoint"].contains($0.role) }
        if !predictions.isEmpty {
            // Keep every requested model visible. Source/reference maps remain
            // available in Maps without silently hiding a selected checkpoint.
            return candidates.filter { $0.role == "target" } + predictions
        }
        guard let target = candidates.first(where: { $0.label.lowercased() == "target" }) else {
            return Array(candidates.prefix(2))
        }
        let model = candidates.last { $0.id != target.id && $0.label.lowercased() != "flat" }
        return model.map { [target, $0] } ?? [target]
    }
    var body: some View {
        VStack(spacing: 0) {
            if let sample = candidates.compactMap(\.sampleLabel).first {
                HStack(alignment: .firstTextBaseline) {
                    Label("Sample: \(sample)", systemImage: "photo").font(.headline).textSelection(.enabled)
                    Spacer()
                    Text("\(candidates.count) maps · same source").font(.caption).foregroundStyle(.secondary)
                }.padding(.horizontal, 12).padding(.vertical, 8).background(.bar)
            }
            inspectionControls.padding(8).background(.bar)
            Text("Drag or scroll to pan · Pinch or Option-scroll to zoom · 100% shows original pixels")
                .font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 12).padding(.vertical, 5)
            if candidates.isEmpty {
                ContentUnavailableView("Choose maps to compare", systemImage: "square.split.2x1")
            } else {
                HSplitView {
                    ForEach(visibleCandidates) { candidate in
                        VStack(spacing: 0) {
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(candidate.label).font(.headline).textSelection(.enabled)
                                    if let detail = candidate.detail {
                                        Text(detail).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                                    }
                                    Text(candidate.mapURL.lastPathComponent).font(.caption2).foregroundStyle(.secondary)
                                        .lineLimit(1).truncationMode(.middle).help(candidate.mapURL.path)
                                }.accessibilityElement(children: .ignore).accessibilityLabel(candidate.accessibleLabel)
                                Spacer()
                                Button("Export Original…") { export(candidate) }.disabled(inspectedHashes[candidate.id] == nil)
                                    .help("Copy the original PNG or EXR exactly, at its original resolution and precision. Display contrast has no effect on the export.")
                                Menu {
                                    Button("Open Full Map in New Window") { ReviewWindowController.shared.open(candidates: [candidate]) }
                                    Button("Export Original…") { export(candidate) }.disabled(inspectedHashes[candidate.id] == nil)
                                    Button("Open in GIMP") { openInGIMP(candidate) }.disabled(inspectedHashes[candidate.id] == nil)
                                    Button("Show Original in Finder") { NSWorkspace.shared.activateFileViewerSelecting([candidate.mapURL]) }
                                } label: { Image(systemName: "ellipsis.circle") }
                            }.padding(10)
                            MapInspectionView(url: candidate.mapURL, numeric: candidate.numeric, viewport: viewport, onLoad: { value in
                                inspectedHashes[candidate.id] = value?.sourceSHA256
                            })
                        }.frame(minWidth: 240)
                    }
                }
            }
        }
        .alert("Map inspection", isPresented: Binding(get: { notice != nil }, set: { if !$0 { notice = nil } })) {
            Button("OK") { notice = nil }
        } message: { Text(notice ?? "") }
    }
    private var inspectionControls: some View {
        HStack {
                Button("Fit") { viewport.fit() }
                Button("100%") { viewport.setActualSize() }.help("One source pixel per display pixel")
                Button("200%") { viewport.setZoom(2) }
                Button { viewport.setZoom(viewport.zoom / 1.25) } label: { Image(systemName: "minus.magnifyingglass") }
                Button { viewport.setZoom(viewport.zoom * 1.25) } label: { Image(systemName: "plus.magnifyingglass") }
                Text(viewport.fitToView ? "Fit • linked pan" : "\(Int(viewport.zoom * 100))% • linked pan")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Menu("Maps") {
                    ForEach(candidates) { candidate in
                        Toggle(candidate.label, isOn: Binding(get: { visibleCandidates.contains(where: { $0.id == candidate.id }) }, set: { shown in
                            if visibleIDs.isEmpty { visibleIDs = Set(visibleCandidates.map(\.id)) }
                            if shown { visibleIDs.insert(candidate.id) } else if visibleIDs.count > 1 { visibleIDs.remove(candidate.id) }
                        }))
                    }
                }
                .help("Show the source photo, reference target, base model or checkpoint outputs. Every pane keeps its name and source identity when popped out.")
                Button("Pop Out Comparison") { ReviewWindowController.shared.open(candidates: visibleCandidates, blendURL: blendURL) }
                if let blendURL { Button("Open Displaced Surface in Blender") { NSWorkspace.shared.open(blendURL) } }
        }.help("Drag or scroll to pan. Pinch, or hold Option/Command while scrolling, to zoom. Pan and zoom are linked across maps.")
            .safeAreaInset(edge: .bottom, spacing: 8) {
                if visibleCandidates.contains(where: \.numeric) {
                    HStack {
                        Text("Display contrast")
                        Slider(value: $viewport.displayContrast, in: 1...32).frame(maxWidth: 160)
                        Text("\(viewport.displayContrast, specifier: "%.1f")×")
                        Text("Midpoint")
                        Slider(value: $viewport.displayMidpoint, in: 0...1).frame(maxWidth: 160)
                        Text("\(viewport.displayMidpoint, specifier: "%.3f")")
                        Button("Reset Display") { viewport.displayContrast = 1; viewport.displayMidpoint = 0.5 }
                        Spacer()
                        Text("Display only; shared across maps").foregroundStyle(.secondary)
                    }.font(.caption)
                }
            }
    }
    private func export(_ candidate: MapReviewCandidate) {
        guard let expectedHash = inspectedHashes[candidate.id] else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = candidate.exportFilename
        if let type = UTType(filenameExtension: candidate.mapURL.pathExtension) { panel.allowedContentTypes = [type] }
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        Task {
            do {
                try await Task.detached {
                    try ReviewImageLoader.exportOriginal(candidate.mapURL, expectedSHA256: expectedHash, to: destination)
                }.value
            } catch { notice = error.localizedDescription }
        }
    }
    private func openInGIMP(_ candidate: MapReviewCandidate) {
        guard let expectedHash = inspectedHashes[candidate.id] else { return }
        let identifiers = ["org.gimp.gimp", "org.gimp.GIMP"]
        let app = identifiers.compactMap({ NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0) }).first
        Task {
            do {
                let directory = FileManager.default.temporaryDirectory.appendingPathComponent("TextureStudio-Editor-\(UUID().uuidString)", isDirectory: true)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let copy = directory.appendingPathComponent(candidate.exportFilename)
                try await Task.detached {
                    try ReviewImageLoader.exportOriginal(candidate.mapURL, expectedSHA256: expectedHash, to: copy)
                }.value
                if let app {
                    _ = try await NSWorkspace.shared.open([copy], withApplicationAt: app, configuration: NSWorkspace.OpenConfiguration())
                } else {
                    NSWorkspace.shared.activateFileViewerSelecting([copy])
                    notice = "GIMP was not found. A byte-identical editable copy is selected in Finder; use Open With to choose your editor."
                }
            } catch { notice = error.localizedDescription }
        }
    }
}
