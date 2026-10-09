import AppKit
import SwiftUI

@MainActor final class ReviewWindowController: NSObject, NSWindowDelegate {
    static let shared = ReviewWindowController()
    private var windows: [ObjectIdentifier: (window: NSWindow, caches: [MaterialRenderCache])] = [:]
    func open(candidates: [MapReviewCandidate], blendURL: URL? = nil, retaining cache: MaterialRenderCache? = nil) {
        guard !candidates.isEmpty else { return }
        var retained = cache.map { [$0] } ?? []
        // A child inspector may outlive the window that opened it. Inherit
        // every matching lease before constructing its view, so closing a
        // parent or changing Studio's source cannot remove its original maps.
        for existing in windows.values.flatMap(\.caches) {
            guard !retained.contains(where: { $0 === existing }),
                  candidates.contains(where: { Self.mapURL($0.mapURL, isInCacheFolder: existing.folder) }) else { continue }
            retained.append(existing)
        }
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1200, height: 780),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        let sample = candidates.compactMap(\.sampleLabel).first
        window.title = candidates.count == 1 ? candidates[0].accessibleLabel : sample.map { "\($0) · Full Resolution Comparison" } ?? "Full Resolution Map Comparison"
        window.contentView = NSHostingView(rootView: ReviewWorkbenchView(candidates: candidates, blendURL: blendURL))
        window.minSize = NSSize(width: 550, height: 360)
        window.isReleasedWhenClosed = false
        window.delegate = self
        windows[ObjectIdentifier(window)] = (window, retained)
        window.center(); window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    func windowWillClose(_ notification: Notification) {
        if let window = notification.object as? NSWindow { windows.removeValue(forKey: ObjectIdentifier(window)) }
    }
    static func mapURL(_ url: URL, isInCacheFolder folder: URL) -> Bool {
        let fileComponents = url.standardizedFileURL.resolvingSymlinksInPath().pathComponents
        let folderComponents = folder.standardizedFileURL.resolvingSymlinksInPath().pathComponents
        return fileComponents.count > folderComponents.count && fileComponents.starts(with: folderComponents)
    }
}
