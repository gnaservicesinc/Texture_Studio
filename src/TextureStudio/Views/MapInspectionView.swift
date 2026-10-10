import AppKit
import Observation
import SwiftUI

@MainActor @Observable final class InspectionViewport {
    var zoom: CGFloat = 1 { didSet { persistDisplay() } }
    var normalizedCenter = CGPoint(x: 0.5, y: 0.5)
    var fitToView = false { didSet { persistDisplay() } }
    var displayContrast: Double = 1 { didSet { persistDisplay() } }
    var displayMidpoint: Double = 0.5 { didSet { persistDisplay() } }
    private let preferences: UserDefaults?
    @ObservationIgnored private var displayPreferenceKey: String?
    @ObservationIgnored private var isRestoringDisplay = false
    init(preferences: UserDefaults? = nil, preferenceKey: String? = nil) {
        self.preferences = preferences
        displayPreferenceKey = preferenceKey
        restoreDisplay()
    }
    func useDisplayContext(_ key: String?) {
        guard key != displayPreferenceKey else { return }
        displayPreferenceKey = key
        restoreDisplay()
    }
    private func restoreDisplay() {
        // Loading a context must not save intermediate defaults over either
        // comparison. Each newly opened map starts with a neutral display.
        isRestoringDisplay = true
        defer { isRestoringDisplay = false }
        zoom = 1; fitToView = false
        displayContrast = 1; displayMidpoint = 0.5
        normalizedCenter = CGPoint(x: 0.5, y: 0.5)
        if let displayPreferenceKey, let stored = preferences?.dictionary(forKey: displayPreferenceKey) {
            if let value = stored["zoom"] as? Double, value.isFinite, value > 0 { zoom = value }
            fitToView = stored["fit"] as? Bool ?? false
            if let value = stored["contrast"] as? Double, value.isFinite, value > 0 { displayContrast = value }
            if let value = stored["midpoint"] as? Double, value.isFinite { displayMidpoint = min(1, max(0, value)) }
        }
    }
    private func persistDisplay() {
        guard !isRestoringDisplay, let displayPreferenceKey else { return }
        preferences?.set(["zoom": Double(zoom), "fit": fitToView,
                          "contrast": displayContrast, "midpoint": displayMidpoint], forKey: displayPreferenceKey)
    }
    func setActualSize() { zoom = 1; fitToView = false }
    func fit() { normalizedCenter = CGPoint(x: 0.5, y: 0.5); fitToView = true }
    func setZoom(_ value: CGFloat) {
        guard value.isFinite, value > 0 else { return }
        zoom = value; fitToView = false
    }
    func pan(dx: CGFloat, dy: CGFloat) {
        guard dx.isFinite, dy.isFinite else { return }
        normalizedCenter = CGPoint(x: min(1, max(0, normalizedCenter.x + dx)),
                                   y: min(1, max(0, normalizedCenter.y + dy)))
    }
    func imageScale(imageSize: CGSize, viewSize: CGSize, backingScale: CGFloat) -> CGFloat {
        if fitToView { return max(0.00001, min(viewSize.width / imageSize.width, viewSize.height / imageSize.height)) }
        return zoom / max(1, backingScale)
    }
}

struct MapInspectionView: View {
    let url: URL
    let numeric: Bool
    let viewport: InspectionViewport
    var title = ""
    var onLoad: ((ReviewLoadedImage?) -> Void)? = nil
    var sourceSHA256: String? = nil
    var onMissingSource: ((URL) -> Void)? = nil
    var displayTransform: MapReviewDisplayTransform? = nil
    @State private var loaded: ReviewLoadedImage?
    @State private var failure: String?
    @State private var pinnedHash: String?
    @State private var pinnedURL: URL?
    @State private var pinnedRevision: String?
    @State private var isMissing = false
    @State private var reload = 0
    var body: some View {
        VStack(spacing: 0) {
            if let loaded {
                InspectionCanvas(image: loaded.image, viewport: viewport)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()
                HStack {
                    Text("\(loaded.pixelWidth) × \(loaded.pixelHeight) · \(loaded.storageDescription)")
                    Spacer()
                    Text(numeric ? "8-bit display only • original precision retained • shared contrast" : "Color display • original file retained")
                }.font(.caption).foregroundStyle(.secondary).padding(8)
                    .fixedSize(horizontal: false, vertical: true)
                    .background(.bar)
                if let failure { previewNotice(failure) }
            } else if let failure {
                ContentUnavailableView {
                    Label(isMissing ? "Source file missing" : "Preview unavailable", systemImage: isMissing ? "doc.badge.ellipsis" : "photo")
                } description: {
                    Text(failure)
                } actions: {
                    if !isMissing { Button("Retry Preview") { reload += 1 } }
                }
            } else { ProgressView("Loading full map…").frame(maxWidth: .infinity, maxHeight: .infinity) }
        }
        .task(id: url.path + (sourceSHA256 ?? "") + (displayTransform?.identity ?? "") + "\(reload)" + (numeric ? "numeric:\(viewport.displayContrast):\(viewport.displayMidpoint)" : "color")) {
            if pinnedURL != url || pinnedRevision != sourceSHA256 {
                loaded = nil; pinnedHash = sourceSHA256; pinnedURL = url; pinnedRevision = sourceSHA256; onLoad?(nil)
            }
            failure = nil; isMissing = false
            do {
                if loaded != nil { try await Task.sleep(for: .milliseconds(120)) }
                let value = try await ReviewImageLoader.shared.load(url, numeric: numeric,
                    contrast: viewport.displayContrast, midpoint: viewport.displayMidpoint, expectedSHA256: pinnedHash,
                    displayTransform: displayTransform)
                try Task.checkCancellation()
                loaded = value; pinnedHash = value.sourceSHA256
                onLoad?(value)
            } catch is CancellationError { }
            catch ReviewImageError.missingSource {
                if !Task.isCancelled {
                    loaded = nil; failure = ReviewImageError.missingSource.localizedDescription; isMissing = true
                    onLoad?(nil); onMissingSource?(url)
                }
            }
            catch {
                if !Task.isCancelled {
                    failure = error.localizedDescription
                    // A contrast refresh failure must not replace a valid map
                    // with an error panel or discard its pinned export hash.
                    if loaded == nil { onLoad?(nil) }
                }
            }
        }
    }
    private func previewNotice(_ message: String) -> some View {
        HStack {
            Text(message).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Button("Retry Preview") { reload += 1 }.font(.caption)
        }.padding(8).background(.bar)
    }
}

struct InspectionCanvas: NSViewRepresentable {
    let image: CGImage
    let viewport: InspectionViewport
    func makeNSView(context: Context) -> InspectionCanvasView { InspectionCanvasView() }
    func updateNSView(_ view: InspectionCanvasView, context: Context) {
        // Reading all observable fields establishes SwiftUI updates for both panes.
        _ = viewport.zoom; _ = viewport.fitToView; _ = viewport.normalizedCenter
        view.image = image; view.viewport = viewport; view.needsDisplay = true
    }
}

@MainActor final class InspectionCanvasView: NSView {
    var image: CGImage?
    var viewport: InspectionViewport?
    private var lastDrag: CGPoint?
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        clipsToBounds = true
    }
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        clipsToBounds = true
    }
    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    private var scale: CGFloat {
        guard let image, let viewport else { return 1 }
        return viewport.imageScale(imageSize: CGSize(width: image.width, height: image.height),
            viewSize: bounds.size, backingScale: window?.backingScaleFactor ?? 1)
    }
    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        context.saveGState()
        defer { context.restoreGState() }
        // Native-pixel images extend beyond this viewport when zoomed/panned.
        // Clip their drawing explicitly so they cannot cover pane identity,
        // export controls, or a neighboring comparison image.
        context.clip(to: bounds)
        NSColor.textBackgroundColor.setFill(); bounds.fill()
        guard let image, let viewport else { return }
        let width = CGFloat(image.width) * scale, height = CGFloat(image.height) * scale
        let rect = CGRect(x: bounds.midX - viewport.normalizedCenter.x * width,
                          y: bounds.midY - viewport.normalizedCenter.y * height, width: width, height: height)
        context.interpolationQuality = .none
        context.translateBy(x: rect.minX, y: rect.maxY); context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(origin: .zero, size: rect.size))
    }
    override func setFrameSize(_ newSize: NSSize) { super.setFrameSize(newSize); needsDisplay = true }
    override func mouseDown(with event: NSEvent) { window?.makeFirstResponder(self); lastDrag = convert(event.locationInWindow, from: nil) }
    override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        if let lastDrag, let image, let viewport {
            viewport.pan(dx: (lastDrag.x - point.x) / (CGFloat(image.width) * scale),
                         dy: (lastDrag.y - point.y) / (CGFloat(image.height) * scale)); needsDisplay = true
        }
        lastDrag = point
    }
    override func mouseUp(with event: NSEvent) { lastDrag = nil }
    override func scrollWheel(with event: NSEvent) {
        guard let image, let viewport else { return }
        if event.modifierFlags.contains(.command) || event.modifierFlags.contains(.option) {
            viewport.setZoom((viewport.fitToView ? scale * (window?.backingScaleFactor ?? 1) : viewport.zoom) * exp(-event.scrollingDeltaY * 0.015))
        } else {
            viewport.pan(dx: event.scrollingDeltaX / (CGFloat(image.width) * scale),
                         dy: event.scrollingDeltaY / (CGFloat(image.height) * scale))
        }
        needsDisplay = true
    }
    override func magnify(with event: NSEvent) {
        guard let viewport else { return }
        viewport.setZoom((viewport.fitToView ? scale * (window?.backingScaleFactor ?? 1) : viewport.zoom) * (1 + event.magnification)); needsDisplay = true
    }
}
