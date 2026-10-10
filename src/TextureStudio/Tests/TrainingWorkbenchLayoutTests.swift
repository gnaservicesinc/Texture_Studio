import AppKit
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class TrainingWorkbenchLayoutTests: XCTestCase {
    func testTrainingAndSavingReserveBottomActionBarAtMinimumWindowSize() async throws {
        let suite = "training-layout-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.operation("Training material", training: true) { try await Task.sleep(for: .seconds(60)) }
        defer { store.abort() }
        store.recordTrainingProgress("""
        {"event":"training_started","requested_updates":200,"updates_per_map":100,"initial_step":40}
        {"event":"update_started","completed_updates":12,"requested_updates":200,"current_update":13,"epoch":7,"total_epochs":100,"sample_position":1,"sample_total":2,"sample_id":"brick-center"}
        {"event":"operation_progress","phase":"training","operation":"Backward pass","completed":7,"total":18,"workflow_phase":3}

        """)
        let host = NSHostingView(rootView: TrainingWorkbenchView(store: store).environment(\.colorScheme, .dark))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 1050, height: 700),
                              styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .darkAqua)
        window.contentView = host
        window.orderFront(nil)
        defer { window.close() }
        try await waitFor(host) { Self.splitView(in: host) != nil }
        try assertBottomActionBar(host)
        XCTAssertEqual(store.trainingProgress?.updateSummary, "Total steps: 12 / 200")
        XCTAssertEqual(store.trainingProgress?.currentUpdateSummary, "Running step 13 of 200")
        XCTAssertEqual(store.trainingProgress?.operationLabel, "Backward pass")

        store.saveCheckpointNow()
        store.stop()
        try await Task.sleep(for: .milliseconds(100))
        host.layoutSubtreeIfNeeded()
        try assertBottomActionBar(host)
        XCTAssertTrue(store.canAbort)
    }

    private func assertBottomActionBar(_ host: NSView) throws {
        let split = try XCTUnwrap(Self.splitView(in: host))
        let frame = split.convert(split.bounds, to: host)
        let bottomSpace = host.isFlipped ? host.bounds.maxY - frame.maxY : frame.minY - host.bounds.minY
        XCTAssertGreaterThanOrEqual(bottomSpace, 44, "Settings and the log must leave visible space for the bottom run controls")
        XCTAssertLessThanOrEqual(bottomSpace, 80, "The action bar must not consume the training area")
        XCTAssertEqual(frame.width, host.bounds.width, accuracy: 1)
    }

    private func waitFor(_ host: NSView, condition: () -> Bool) async throws {
        for _ in 0..<200 {
            host.layoutSubtreeIfNeeded()
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Training controls did not appear")
    }

    private static func splitView(in view: NSView) -> NSSplitView? {
        if let split = view as? NSSplitView { return split }
        return view.subviews.lazy.compactMap { splitView(in: $0) }.first
    }
}
