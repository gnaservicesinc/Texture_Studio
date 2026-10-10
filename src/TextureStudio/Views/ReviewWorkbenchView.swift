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
    @State private var inspectedDescriptions: [String: String] = [:]
    @State private var visibleIDs: Set<String> = []
    @State private var exportedURL: URL?
    @State private var isExporting = false
    @State private var paneHeaderHeight: CGFloat = 0
    @State private var qualityCandidateID: String?
    @State private var missingIDs: Set<String> = []
    private let preferences: UserDefaults
    private let onMissingSource: ((URL) -> Void)?
    private let onConfirmReview: ((String, MaterialQualityDecision.Recommendation) -> Void)?
    init(candidates: [MapReviewCandidate], blendURL: URL? = nil,
         preferences: UserDefaults = UserDefaults(suiteName: "org.ipde.material-tools")!,
         onMissingSource: ((URL) -> Void)? = nil,
         onConfirmReview: ((String, MaterialQualityDecision.Recommendation) -> Void)? = nil) {
        self.candidates = candidates
        self.blendURL = blendURL
        self.preferences = preferences
        self.onMissingSource = onMissingSource
        self.onConfirmReview = onConfirmReview
        _viewport = State(initialValue: InspectionViewport(preferences: preferences, preferenceKey: Self.displayPreferenceKey(candidates)))
    }
    static func selectionPreferenceKey(_ candidates: [MapReviewCandidate]) -> String {
        // Persist pane choices only for this exact set of files and candidate
        // identities; a different comparison must never inherit stale IDs.
        let identities = candidates.map { [$0.id, $0.mapURL.standardizedFileURL.path].joined(separator: "\u{0}") }.sorted()
        return "reviewVisibleMaps." + ReviewImageLoader.hash(Data(identities.joined(separator: "\u{1}").utf8))
    }
    static func displayPreferenceKey(_ candidates: [MapReviewCandidate]) -> String {
        selectionPreferenceKey(candidates) + ".display"
    }
    private var visibleCandidates: [MapReviewCandidate] {
        Self.resolvedCandidates(candidates.filter { !missingIDs.contains($0.id) }, visibleIDs: visibleIDs)
    }
    private var diffuseCandidate: MapReviewCandidate? {
        candidates.first { !missingIDs.contains($0.id) && !$0.numeric && ($0.role == "diffuse" || $0.label.lowercased().contains("diffuse")) }
    }
    private var qualityCandidates: [MapReviewCandidate] {
        candidates.filter { !missingIDs.contains($0.id) && $0.numeric && ["base", "checkpoint", "model"].contains($0.role)
            && ["height", "depth", "roughness", "normal"].contains($0.modelIdentity?.mapType ?? "") }
    }
    private var qualityCandidate: MapReviewCandidate? {
        qualityCandidates.first { $0.id == qualityCandidateID } ?? qualityCandidates.last
    }
    static func resolvedCandidates(_ candidates: [MapReviewCandidate], visibleIDs: Set<String>) -> [MapReviewCandidate] {
        let selected = candidates.filter { visibleIDs.contains($0.id) }
        let requested = selected.isEmpty ? initialCandidates(candidates) : selected
        // The real reference anchors every model comparison, even if an older
        // saved pane selection hid it. Never substitute a prediction for it.
        return candidates.filter(\.isReference) + requested.filter { !$0.isReference }
    }
    static func initialCandidates(_ candidates: [MapReviewCandidate]) -> [MapReviewCandidate] {
        let predictions = candidates.filter { ["base", "checkpoint", "model"].contains($0.role) }
        if !predictions.isEmpty {
            return candidates.filter(\.isReference) + predictions
        }
        guard let target = candidates.first(where: \.isReference) else {
            return Array(candidates.prefix(2))
        }
        return [target]
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
            if !candidates.contains(where: \.isReference), candidates.contains(where: { ["base", "checkpoint", "model"].contains($0.role) }) {
                Label("Model predictions only · this review has no recorded source-map reference", systemImage: "info.circle")
                    .font(.caption).padding(.horizontal, 12).padding(.vertical, 6)
                    .help("A real source displacement map must be identified explicitly in the review manifest. A model output is never substituted for it.")
            }
            inspectionControls.padding(8).background(.bar)
            if let diffuse = diffuseCandidate, let map = qualityCandidate {
                VStack(alignment: .leading, spacing: 6) {
                    Picker("Adviser output", selection: Binding(get: { map.id }, set: { qualityCandidateID = $0 })) {
                        ForEach(qualityCandidates) { Text($0.label).tag($0.id) }
                    }.frame(maxWidth: 440)
                    MaterialQualityReviewPanel(diffuseURL: diffuse.mapURL, mapURL: map.mapURL,
                        mapType: map.modelIdentity?.mapType ?? "height", purpose: .result,
                        diffuseTransform: diffuse.displayTransform, mapTransform: map.displayTransform,
                        onApply: onConfirmReview.map { confirm in { recommendation in confirm(map.id, recommendation) } })
                }.padding(.horizontal, 12).padding(.vertical, 8)
            }
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
                                        if let description = inspectedDescriptions[candidate.id] {
                                            Text(description).font(.caption.bold()).textSelection(.enabled)
                                        }
                                        Text(candidate.mapURL.lastPathComponent).font(.caption2).foregroundStyle(.secondary)
                                            .lineLimit(1).truncationMode(.middle).help(candidate.mapURL.path)
                                    }.accessibilityElement(children: .ignore).accessibilityLabel(candidate.accessibleLabel)
                                    Spacer()
                                    Menu {
                                        Button("Open Full Map in New Window") { ReviewWindowController.shared.open(candidates: [candidate]) }
                                        Button("Show Original in Finder") { NSWorkspace.shared.activateFileViewerSelecting([candidate.mapURL]) }
                                        if let source = candidate.fullSourceReference {
                                            Button("Show Full Source Map in Finder") { NSWorkspace.shared.activateFileViewerSelecting([source.mapURL]) }
                                            Button("Export Full Source Map…") { export(source) }
                                            Button("Open Full Source Map in GIMP") { openInGIMP(source) }
                                        }
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
                                if let source = candidate.fullSourceReference {
                                    Button("Open Full Source \(candidate.referenceMapName.capitalized) Map", systemImage: "arrow.up.left.and.arrow.down.right") {
                                        ReviewWindowController.shared.open(candidates: [source])
                                    }.font(.caption).padding(.horizontal, 10).padding(.bottom, 8)
                                        .help("Inspect the complete original source displacement map at its own resolution. The comparison above uses the recorded crop; this does not run a model or produce a full-source prediction.")
                                }
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
                                inspectedDescriptions[candidate.id] = value.map { "Displayed \(candidate.displayTransform == nil ? "file" : "training grid"): \($0.pixelWidth) × \($0.pixelHeight) · \($0.storageDescription)" }
                            }, onMissingSource: { source in
                                missingIDs.insert(candidate.id)
                                visibleIDs.remove(candidate.id)
                                onMissingSource?(source)
                            }, displayTransform: candidate.displayTransform).frame(maxWidth: .infinity, maxHeight: .infinity).clipped()
                        }.frame(minWidth: 240)
                    }
                }
            }
        }
        .alert("Map inspection", isPresented: Binding(get: { notice != nil }, set: { if !$0 { notice = nil } })) {
            Button("OK") { notice = nil }
        } message: { Text(notice ?? "") }
        .onAppear {
            viewport.useDisplayContext(Self.displayPreferenceKey(candidates))
            restoreVisibleMaps()
        }
        .onPreferenceChange(ReviewPaneHeaderHeights.self) { heights in
            // Keep linked image coordinates aligned even when one checkpoint
            // has a longer identity than its neighbors. Measure before padding
            // to the shared height so resizing can also shrink the header.
            let height = heights.values.max() ?? 0
            if abs(height - paneHeaderHeight) > 0.5 { paneHeaderHeight = height }
        }
        .onChange(of: Self.selectionPreferenceKey(candidates)) { _, _ in
            viewport.useDisplayContext(Self.displayPreferenceKey(candidates))
            restoreVisibleMaps()
        }
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
                Text("Zoom").font(.caption)
                NumericTextField(title: "Zoom percent", value: Binding(
                    get: { Double(viewport.zoom) * 100 },
                    set: { viewport.setZoom(CGFloat($0 / 100)) }), greaterThan: 0)
                    .frame(width: 85)
                Text(viewport.fitToView ? "% manual · Fit active" : "% · linked pan")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Menu("Visible Maps (\(visibleCandidates.count))") {
                    ForEach(candidates) { candidate in
                        Toggle(candidate.label, isOn: Binding(get: { visibleCandidates.contains(where: { $0.id == candidate.id }) }, set: { shown in
                            if visibleIDs.isEmpty { visibleIDs = Set(visibleCandidates.map(\.id)) }
                            if shown { visibleIDs.insert(candidate.id) } else if !candidate.isReference && visibleIDs.count > 1 { visibleIDs.remove(candidate.id) }
                        }))
                        .disabled(candidate.isReference)
                    }
                }
                .help("The real source reference stays visible beside model predictions. Choose additional photos and predictions here. Every pane retains its identity when popped out.")
                Button("Pop Out Comparison") { ReviewWindowController.shared.open(candidates: visibleCandidates, blendURL: blendURL) }
                if let blendURL { Button("Open Displaced Surface in Blender") { NSWorkspace.shared.open(blendURL) } }
        }.help("Drag or scroll to pan. Pinch, or hold Option/Command while scrolling, to zoom. Pan and zoom are linked across maps.")
            .safeAreaInset(edge: .bottom, spacing: 8) {
                if visibleCandidates.contains(where: \.numeric) {
                    HStack {
                        DoubleControl(title: "Display contrast", value: $viewport.displayContrast, range: 1...32, suffix: "×", enforcesSliderRange: false, greaterThan: 0)
                            .frame(maxWidth: 220)
                        DoubleControl(title: "Midpoint", value: $viewport.displayMidpoint, range: 0...1)
                            .frame(maxWidth: 200)
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
