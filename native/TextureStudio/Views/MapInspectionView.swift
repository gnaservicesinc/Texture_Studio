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
    init(preferences: UserDefaults? = nil) {
        self.preferences = preferences
        if let stored = preferences?.dictionary(forKey: "reviewDisplay") {
            if let value = stored["zoom"] as? Double, value.isFinite { zoom = min(32, max(0.02, value)) }
            fitToView = stored["fit"] as? Bool ?? false
            if let value = stored["contrast"] as? Double, value.isFinite { displayContrast = min(32, max(1, value)) }
            if let value = stored["midpoint"] as? Double, value.isFinite { displayMidpoint = min(1, max(0, value)) }
        }
    }
    private func persistDisplay() {
        preferences?.set(["zoom": Double(zoom), "fit": fitToView,
                          "contrast": displayContrast, "midpoint": displayMidpoint], forKey: "reviewDisplay")
    }
    func setActualSize() { zoom = 1; fitToView = false }
    func fit() { normalizedCenter = CGPoint(x: 0.5, y: 0.5); fitToView = true }
    func setZoom(_ value: CGFloat) {
        guard value.isFinite else { return }
        zoom = min(32, max(0.02, value)); fitToView = false
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
    @State private var loaded: ReviewLoadedImage?
    @State private var failure: String?
    @State private var pinnedHash: String?
    @State private var pinnedURL: URL?
    var body: some View {
        VStack(spacing: 0) {
            if let loaded {
                InspectionCanvas(image: loaded.image, viewport: viewport)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()
                HStack {
                    Text("\(loaded.pixelWidth) × \(loaded.pixelHeight) • full source resolution")
                    Spacer()
                    Text(numeric ? "8-bit display only • original precision retained • shared contrast" : "Color display • original file retained")
                }.font(.caption).foregroundStyle(.secondary).padding(8)
                    .fixedSize(horizontal: false, vertical: true)
                    .background(.bar)
            } else if let failure {
                ContentUnavailableView("Map could not be opened", systemImage: "exclamationmark.triangle", description: Text(failure))
            } else { ProgressView("Loading full map…").frame(maxWidth: .infinity, maxHeight: .infinity) }
        }
        .task(id: url.path + (numeric ? "numeric:\(viewport.displayContrast):\(viewport.displayMidpoint)" : "color")) {
            if pinnedURL != url { loaded = nil; pinnedHash = nil; pinnedURL = url; onLoad?(nil) }
            failure = nil
            do {
                if loaded != nil { try await Task.sleep(for: .milliseconds(120)) }
                let value = try await ReviewImageLoader.shared.load(url, numeric: numeric,
                    contrast: viewport.displayContrast, midpoint: viewport.displayMidpoint, expectedSHA256: pinnedHash)
                try Task.checkCancellation()
                loaded = value; pinnedHash = value.sourceSHA256
                onLoad?(value)
            } catch is CancellationError { }
            catch { if !Task.isCancelled { failure = error.localizedDescription; loaded = nil; onLoad?(nil) } }
        }
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
