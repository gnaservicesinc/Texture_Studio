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
    let comparisonLayout: Bool
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
    @State private var detailsCandidate: MapReviewCandidate?
    private let preferences: UserDefaults
    private let onMissingSource: ((URL) -> Void)?
    private let onConfirmReview: ((String, MaterialQualityDecision.Recommendation) -> Void)?
    init(candidates: [MapReviewCandidate], blendURL: URL? = nil, comparisonLayout: Bool = false,
         preferences: UserDefaults = UserDefaults(suiteName: "org.ipde.material-tools")!,
         onMissingSource: ((URL) -> Void)? = nil,
         onConfirmReview: ((String, MaterialQualityDecision.Recommendation) -> Void)? = nil) {
        self.candidates = candidates
        self.blendURL = blendURL
        self.comparisonLayout = comparisonLayout
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
        let available = candidates.filter { !missingIDs.contains($0.id) }
        return comparisonLayout ? Self.resolvedComparisonCandidates(available, visibleIDs: visibleIDs)
            : Self.resolvedCandidates(available, visibleIDs: visibleIDs)
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
    static func resolvedComparisonCandidates(_ candidates: [MapReviewCandidate], visibleIDs: Set<String>) -> [MapReviewCandidate] {
        let selected = candidates.filter { !$0.isReference && visibleIDs.contains($0.id) }
        let predictions = candidates.filter { ["base", "checkpoint", "model"].contains($0.role) }
        let outputs = selected.isEmpty ? Array((predictions.isEmpty ? candidates.filter { !$0.isReference } : predictions).suffix(2))
            : Array(selected.suffix(2))
        // Model names head the results; the real reference remains beside them.
        return outputs + Array(candidates.filter(\.isReference).prefix(1))
    }
    var body: some View {
        VStack(spacing: 0) {
            if let sample = candidates.compactMap(\.sampleLabel).first {
                HStack(alignment: .firstTextBaseline) {
                    Text(comparisonLayout ? "Results" : "Sample: \(sample)").font(.headline).textSelection(.enabled)
                    Spacer()
                    Text(comparisonLayout ? sample : "\(candidates.count) maps · same source")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }.padding(.horizontal, 12).padding(.vertical, 8).background(.bar)
            }
            if !candidates.contains(where: \.isReference), candidates.contains(where: { ["base", "checkpoint", "model"].contains($0.role) }) {
                Label("Model predictions only · this review has no recorded source-map reference", systemImage: "info.circle")
                    .font(.caption).padding(.horizontal, 12).padding(.vertical, 6)
                    .help("A real source displacement map must be identified explicitly in the review manifest. A model output is never substituted for it.")
            }
            inspectionControls.padding(8).background(.bar)
            if comparisonLayout {
                DisclosureGroup("Evaluate a model") { evaluationControls }
                    .padding(.horizontal, 12).padding(.vertical, 7)
            } else { evaluationControls }
            HStack {
                Text(comparisonLayout ? "Pan and zoom are linked across maps" : "Drag or scroll to pan · Pinch or Option-scroll to zoom · 100% shows original pixels")
                Spacer()
                Button(comparisonLayout ? "Export Maps…" : "Export Visible Maps (\(visibleCandidates.count))…", systemImage: "square.and.arrow.up") { exportSelected() }
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
                    ForEach(visibleCandidates) { candidate in
                        VStack(spacing: 0) {
                            VStack(spacing: 0) {
                                HStack(alignment: .top) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                                            Text(candidate.displayTitle).font(.headline).textSelection(.enabled)
                                        }
                                        if !candidate.displaySummary.isEmpty {
                                            Text(candidate.displaySummary).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                                        }
                                        if let description = inspectedDescriptions[candidate.id] {
                                            Text(description).font(.caption.bold()).textSelection(.enabled)
                                        }
                                    }.accessibilityElement(children: .ignore).accessibilityLabel(candidate.accessibleLabel)
                                    Spacer()
                                    Menu {
                                        Button("Details…", systemImage: "info.circle") { detailsCandidate = candidate }
                                        if comparisonLayout {
                                            Button("Export Original…", systemImage: "square.and.arrow.up") { export(candidate) }
                                            Button("Open Copy in GIMP", systemImage: "paintbrush") { openInGIMP(candidate) }
                                        }
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
                                if !comparisonLayout { ViewThatFits(in: .horizontal) {
                                    HStack { exportControls(candidate); Spacer(minLength: 0) }
                                    VStack(alignment: .leading) { exportControls(candidate) }.frame(maxWidth: .infinity, alignment: .leading)
                                }.font(.caption).padding(.horizontal, 10).padding(.bottom, 8).disabled(isExporting)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .layoutPriority(1)
                                    .background(.bar) }
                                if let source = candidate.fullSourceReference {
                                    Button(comparisonLayout ? "Open full original" : "Open Full Source \(candidate.referenceMapName.capitalized) Map", systemImage: "arrow.up.left.and.arrow.down.right") {
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
                                inspectedDescriptions[candidate.id] = value.map { "\($0.pixelWidth.formatted()) × \($0.pixelHeight.formatted()) · \($0.storageDescription)" }
                            }, onMissingSource: { source in
                                missingIDs.insert(candidate.id)
                                visibleIDs.remove(candidate.id)
                                onMissingSource?(source)
                            }, displayTransform: candidate.displayTransform).frame(maxWidth: .infinity, maxHeight: .infinity).clipped()
                        }.frame(minWidth: comparisonLayout ? 180 : 240)
                    }
                }
            }
        }
        .alert("Map inspection", isPresented: Binding(get: { notice != nil }, set: { if !$0 { notice = nil } })) {
            Button("OK") { notice = nil }
        } message: { Text(notice ?? "") }
        .popover(item: $detailsCandidate) { candidate in candidateDetails(candidate) }
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
    @ViewBuilder private var evaluationControls: some View {
        if let map = qualityCandidate {
            VStack(alignment: .leading, spacing: 6) {
                Picker("Model", selection: Binding(get: { map.id }, set: { qualityCandidateID = $0 })) {
                    ForEach(qualityCandidates) { candidate in Text(candidate.selectionTitle).tag(candidate.id) }
                }.frame(maxWidth: 440)
                if let diffuse = diffuseCandidate {
                    MaterialQualityReviewPanel(diffuseURL: diffuse.mapURL, mapURL: map.mapURL,
                        mapType: map.modelIdentity?.mapType ?? "height", purpose: .result,
                        diffuseTransform: diffuse.displayTransform, mapTransform: map.displayTransform,
                        onApply: onConfirmReview.map { confirm in { recommendation in confirm(map.id, recommendation) } })
                }
                if ["height", "roughness"].contains(map.modelIdentity?.mapType ?? ""),
                   let reference = candidates.first(where: { $0.isReference && $0.modelIdentity?.mapType == map.modelIdentity?.mapType }) {
                    MaterialScalarAgreementPanel(reference: reference, prediction: map)
                }
            }.padding(.horizontal, comparisonLayout ? 0 : 12).padding(.vertical, 8)
        }
    }

    private func candidateDetails(_ candidate: MapReviewCandidate) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text(candidate.displayTitle).font(.title3.bold())
                if !candidate.displaySummary.isEmpty { Text(candidate.displaySummary).foregroundStyle(.secondary) }
                if let sample = candidate.sampleLabel { LabeledContent("Test image", value: sample) }
                if let identity = candidate.modelIdentity {
                    if let architecture = identity.architecture { LabeledContent("Architecture", value: architecture) }
                    if let path = identity.checkpointPath { LabeledContent("Checkpoint", value: URL(fileURLWithPath: path).lastPathComponent) }
                    if let hash = identity.checkpointSHA256 { LabeledContent("SHA256", value: hash) }
                }
                if let source = candidate.sourceIdentity {
                    ForEach(source.recordedDetails, id: \.self) { Text($0).font(.caption) }
                }
                if let detail = candidate.detail { Text(detail).font(.caption).foregroundStyle(.secondary) }
                Divider()
                LabeledContent("Map file", value: candidate.mapURL.path)
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([candidate.mapURL]) }
            }.textSelection(.enabled).padding(18).frame(width: 430, alignment: .leading)
        }.frame(maxHeight: 450)
    }
    private var inspectionControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    zoomControls
                    Spacer(minLength: 8)
                    visibleMapMenu
                    Button("Pop Out", systemImage: "macwindow.on.rectangle") { ReviewWindowController.shared.open(candidates: visibleCandidates, blendURL: blendURL) }
                }
                VStack(alignment: .leading, spacing: 8) {
                    HStack { zoomControls; Spacer(minLength: 0) }
                    HStack {
                        visibleMapMenu
                        Spacer()
                        Button("Pop Out", systemImage: "macwindow.on.rectangle") { ReviewWindowController.shared.open(candidates: visibleCandidates, blendURL: blendURL) }
                    }
                }
            }
            if visibleCandidates.contains(where: \.numeric) {
                DisclosureGroup("Display contrast") {
                    HStack {
                        DoubleControl(title: "Contrast", value: $viewport.displayContrast, range: 1...32, suffix: "×", enforcesSliderRange: false, greaterThan: 0)
                        DoubleControl(title: "Midpoint", value: $viewport.displayMidpoint, range: 0...1)
                        Button("Reset") { viewport.displayContrast = 1; viewport.displayMidpoint = 0.5 }
                    }.font(.caption).padding(.top, 6)
                    Text("Adjusts the preview. Original map values are preserved.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if let blendURL { Button("Open Displaced Surface in Blender") { NSWorkspace.shared.open(blendURL) } }
        }.help("Drag or scroll to pan. Pinch, or hold Option/Command while scrolling, to zoom. Pan and zoom are linked across maps.")
    }
    private var zoomControls: some View {
        HStack(spacing: 6) {
            Button("Fit") { viewport.fit() }
            Button("100%") { viewport.setActualSize() }.help("One map pixel per display pixel")
            Button { viewport.setZoom(viewport.zoom / 1.25) } label: { Image(systemName: "minus.magnifyingglass") }
            Button { viewport.setZoom(viewport.zoom * 1.25) } label: { Image(systemName: "plus.magnifyingglass") }
            NumericTextField(title: "Zoom percent", value: Binding(
                get: { Double(viewport.zoom) * 100 },
                set: { viewport.setZoom(CGFloat($0 / 100)) }), greaterThan: 0)
                .frame(width: 65)
            Text("%").font(.caption).foregroundStyle(.secondary)
        }
    }
    private var visibleMapMenu: some View {
        Menu("Visible Maps (\(visibleCandidates.count))") {
            if comparisonLayout { Text("Show up to two outputs beside the reference") }
            ForEach(comparisonLayout ? candidates.filter({ ["checkpoint", "model", "base"].contains($0.role) })
                    + candidates.filter({ !["checkpoint", "model", "base"].contains($0.role) }) : candidates) { candidate in
                Toggle(candidate.selectionTitle, isOn: Binding(get: { visibleCandidates.contains(where: { $0.id == candidate.id }) }, set: { shown in
                    if visibleIDs.isEmpty { visibleIDs = Set(visibleCandidates.map(\.id)) }
                    if shown {
                        if comparisonLayout, !candidate.isReference,
                           let replaced = visibleCandidates.filter({ !$0.isReference }).first,
                           visibleCandidates.filter({ !$0.isReference }).count >= 2 {
                            visibleIDs.remove(replaced.id)
                        }
                        visibleIDs.insert(candidate.id)
                    } else if !candidate.isReference && visibleIDs.count > 1 { visibleIDs.remove(candidate.id) }
                })).disabled(candidate.isReference)
            }
        }.help("Choose the model outputs to inspect. The dataset reference stays visible beside them.")
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
