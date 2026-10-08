import AppKit
import SwiftUI
import UniformTypeIdentifiers

private struct ReviewPaneHeaderHeights: PreferenceKey {
    static let defaultValue: [String: CGFloat] = [:]
    static func reduce(value: inout [String: CGFloat], nextValue: () -> [String: CGFloat]) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}

struct ReviewWorkbenchView: View {
    let candidates: [MapReviewCandidate]
    var blendURL: URL? = nil
    @State private var viewport: InspectionViewport
    @State private var notice: String?
    @State private var inspectedHashes: [String: String] = [:]
    @State private var visibleIDs: Set<String> = []
    @State private var exportedURL: URL?
    @State private var isExporting = false
    @State private var paneHeaderHeight: CGFloat = 0
    private let preferences: UserDefaults
    init(candidates: [MapReviewCandidate], blendURL: URL? = nil,
         preferences: UserDefaults = UserDefaults(suiteName: "org.ipde.material-tools")!) {
        self.candidates = candidates
        self.blendURL = blendURL
        self.preferences = preferences
        _viewport = State(initialValue: InspectionViewport(preferences: preferences))
    }
    static func selectionPreferenceKey(_ candidates: [MapReviewCandidate]) -> String {
        // Persist pane choices only for this exact set of files and candidate
        // identities; a different comparison must never inherit stale IDs.
        let identities = candidates.map { [$0.id, $0.mapURL.standardizedFileURL.path].joined(separator: "\u{0}") }.sorted()
        return "reviewVisibleMaps." + ReviewImageLoader.hash(Data(identities.joined(separator: "\u{1}").utf8))
    }
    private var visibleCandidates: [MapReviewCandidate] {
        let selected = candidates.filter { visibleIDs.contains($0.id) }
        return selected.isEmpty ? Self.initialCandidates(candidates) : selected
    }
    static func initialCandidates(_ candidates: [MapReviewCandidate]) -> [MapReviewCandidate] {
        let predictions = candidates.filter { ["base", "checkpoint", "model"].contains($0.role) }
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
            HStack {
                Text("Drag or scroll to pan · Pinch or Option-scroll to zoom · 100% shows original pixels")
                Spacer()
                Button("Export Visible Maps (\(visibleCandidates.count))…", systemImage: "square.and.arrow.up") { exportSelected() }
                    .buttonStyle(.bordered)
                    .disabled(candidates.isEmpty || isExporting)
                    .help("Save every currently visible map into a new folder, with its sample and model name. Copies keep their original resolution, bit depth and exact bytes; display adjustments are not exported.")
            }.font(.caption).foregroundStyle(.secondary)
                .padding(.horizontal, 12).padding(.vertical, 7)
            if let exportedURL {
                HStack {
                    Label("Export complete · original data preserved", systemImage: "checkmark.circle")
                    Text(exportedURL.lastPathComponent).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button("Show Export in Finder") { NSWorkspace.shared.activateFileViewerSelecting([exportedURL]) }
                }.font(.caption).padding(.horizontal, 12).padding(.bottom, 5)
            }
            if candidates.isEmpty {
                ContentUnavailableView("Choose maps to compare", systemImage: "square.split.2x1")
            } else {
                HSplitView {
                    ForEach(Array(visibleCandidates.enumerated()), id: \.element.id) { position, candidate in
                        VStack(spacing: 0) {
                            VStack(spacing: 0) {
                                HStack(alignment: .top) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                                            Text("\(position + 1)").font(.caption.bold()).foregroundStyle(.secondary)
                                                .accessibilityLabel("Pane \(position + 1)")
                                            Text(candidate.label).font(.headline).textSelection(.enabled)
                                        }
                                        if let detail = candidate.detail {
                                            Text(detail).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                                        }
                                        Text(candidate.mapURL.lastPathComponent).font(.caption2).foregroundStyle(.secondary)
                                            .lineLimit(1).truncationMode(.middle).help(candidate.mapURL.path)
                                    }.accessibilityElement(children: .ignore).accessibilityLabel(candidate.accessibleLabel)
                                    Spacer()
                                    Menu {
                                        Button("Open Full Map in New Window") { ReviewWindowController.shared.open(candidates: [candidate]) }
                                        Button("Show Original in Finder") { NSWorkspace.shared.activateFileViewerSelecting([candidate.mapURL]) }
                                    } label: { Image(systemName: "ellipsis.circle") }
                                }.padding(10)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .layoutPriority(2)
                                    .background(.bar)
                                ViewThatFits(in: .horizontal) {
                                    HStack { exportControls(candidate); Spacer(minLength: 0) }
                                    VStack(alignment: .leading) { exportControls(candidate) }.frame(maxWidth: .infinity, alignment: .leading)
                                }.font(.caption).padding(.horizontal, 10).padding(.bottom, 8).disabled(isExporting)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .layoutPriority(1)
                                    .background(.bar)
                            }
                            .fixedSize(horizontal: false, vertical: true)
                            .background {
                                GeometryReader { geometry in
                                    Color.clear.preference(key: ReviewPaneHeaderHeights.self,
                                        value: [candidate.id: geometry.size.height])
                                }
                            }
                            .frame(minHeight: paneHeaderHeight, alignment: .top)
                            .layoutPriority(2)
                            .background(.bar)
                            Divider()
                            MapInspectionView(url: candidate.mapURL, numeric: candidate.numeric, viewport: viewport, onLoad: { value in
                                inspectedHashes[candidate.id] = value?.sourceSHA256
                            }).frame(maxWidth: .infinity, maxHeight: .infinity).clipped()
                        }.frame(minWidth: 240)
                    }
                }
            }
        }
        .alert("Map inspection", isPresented: Binding(get: { notice != nil }, set: { if !$0 { notice = nil } })) {
            Button("OK") { notice = nil }
        } message: { Text(notice ?? "") }
        .onAppear { restoreVisibleMaps() }
        .onPreferenceChange(ReviewPaneHeaderHeights.self) { heights in
            // Keep linked image coordinates aligned even when one checkpoint
            // has a longer identity than its neighbors. Measure before padding
            // to the shared height so resizing can also shrink the header.
            let height = heights.values.max() ?? 0
            if abs(height - paneHeaderHeight) > 0.5 { paneHeaderHeight = height }
        }
        .onChange(of: Self.selectionPreferenceKey(candidates)) { _, _ in restoreVisibleMaps() }
        .onChange(of: visibleIDs) { _, selected in
            preferences.set(Array(selected), forKey: Self.selectionPreferenceKey(candidates))
        }
    }
    @ViewBuilder private func exportControls(_ candidate: MapReviewCandidate) -> some View {
        Button("Export Original…", systemImage: "square.and.arrow.up") { export(candidate) }
            .help("Copy this original PNG or EXR exactly, at its original resolution and precision. You can export before its display finishes loading.")
        Button("Open Copy in GIMP", systemImage: "paintbrush") { openInGIMP(candidate) }
            .help("Open a byte-identical copy for closer inspection or editing. The original map is retained.")
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
                Menu("Visible Maps (\(visibleCandidates.count))") {
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
                        Text("Display only · saved for next time · exports keep raw values").foregroundStyle(.secondary)
                    }.font(.caption)
                }
            }
    }
    private func restoreVisibleMaps() {
        let known = Set(candidates.map(\.id))
        visibleIDs = Set(preferences.stringArray(forKey: Self.selectionPreferenceKey(candidates)) ?? []).intersection(known)
    }
    private func restoreExportDirectory(_ panel: NSSavePanel) {
        if let path = preferences.string(forKey: "reviewExportDirectory"), FileManager.default.fileExists(atPath: path) {
            panel.directoryURL = URL(fileURLWithPath: path, isDirectory: true)
        }
    }
    private func export(_ candidate: MapReviewCandidate) {
        let expectedHash = inspectedHashes[candidate.id]
        let panel = NSSavePanel()
        panel.title = "Export original \(candidate.label)"
        panel.message = "Save the original file at full resolution and precision. Display contrast and zoom do not change its data. Choose a new filename to keep existing files."
        panel.nameFieldStringValue = candidate.exportFilename
        restoreExportDirectory(panel)
        if let type = UTType(filenameExtension: candidate.mapURL.pathExtension) { panel.allowedContentTypes = [type] }
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        preferences.set(destination.deletingLastPathComponent().path, forKey: "reviewExportDirectory")
        isExporting = true
        Task {
            defer { isExporting = false }
            do {
                try await Task.detached {
                    try ReviewImageLoader.exportOriginal(candidate.mapURL, expectedSHA256: expectedHash, to: destination)
                }.value
                exportedURL = destination
            } catch { notice = error.localizedDescription }
        }
    }
    private func exportSelected() {
        let selected = visibleCandidates
        let panel = NSOpenPanel()
        panel.title = "Export \(selected.count) original maps"
        panel.message = "Choose a destination. A new folder will contain all visible maps, with their original data and sample/model labels."
        panel.prompt = "Export Maps"
        panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
        restoreExportDirectory(panel)
        guard panel.runModal() == .OK, let parent = panel.url else { return }
        preferences.set(parent.path, forKey: "reviewExportDirectory")
        let directory = parent.appendingPathComponent("Texture Studio Maps \(UUID().uuidString.prefix(8))", isDirectory: true)
        let copies = selected.map { ReviewOriginalExport(sourceURL: $0.mapURL, filename: $0.exportFilename,
                                                        expectedSHA256: inspectedHashes[$0.id]) }
        isExporting = true
        Task {
            defer { isExporting = false }
            do {
                try await Task.detached { try ReviewImageLoader.exportOriginals(copies, to: directory) }.value
                exportedURL = directory
            } catch { notice = error.localizedDescription }
        }
    }
    private func openInGIMP(_ candidate: MapReviewCandidate) {
        let expectedHash = inspectedHashes[candidate.id]
        let identifiers = ["org.gimp.gimp", "org.gimp.GIMP"]
        let app = identifiers.compactMap({ NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0) }).first
        isExporting = true
        Task {
            defer { isExporting = false }
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
