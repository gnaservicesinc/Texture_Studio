import AppKit
import SwiftUI

@MainActor final class ReviewWindowController: NSObject, NSWindowDelegate {
    static let shared = ReviewWindowController()
    private var windows: [ObjectIdentifier: NSWindow] = [:]
    func open(candidates: [MapReviewCandidate], blendURL: URL? = nil) {
        guard !candidates.isEmpty else { return }
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1200, height: 780),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = candidates.count == 1 ? candidates[0].label : "Full Resolution Map Comparison"
        window.contentView = NSHostingView(rootView: ReviewWorkbenchView(candidates: candidates, blendURL: blendURL))
        window.minSize = NSSize(width: 550, height: 360)
        window.isReleasedWhenClosed = false
        window.delegate = self
        windows[ObjectIdentifier(window)] = window
        window.center(); window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    func windowWillClose(_ notification: Notification) {
        if let window = notification.object as? NSWindow { windows.removeValue(forKey: ObjectIdentifier(window)) }
    }
}
