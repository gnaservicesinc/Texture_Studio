import AppKit
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class PreparationSettingsControlTests: XCTestCase {
    func testActualSettingsStepperChangesAndPersistsWorkerCount() async throws {
        let suite = "preparation-settings-control-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let resources = MachineResources(physicalBytes: 16 * MachineResources.gibibyte, availableProcessorCount: 4)
        let store = WorkbenchStore(preferences: defaults, resources: resources)
        let host = NSHostingView(rootView: PreparationSettingsView(store: store).padding(20))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 560, height: 140),
                              styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderFront(nil)
        defer { window.close() }
        try await waitFor(host) { Self.workerControl(in: host) != nil }
        let stepper = try XCTUnwrap(Self.workerControl(in: host))
        XCTAssertEqual(stepper.accessibilityRole(), .incrementor)
        XCTAssertEqual(store.preparationWorkers, 4)
        _ = stepper.accessibilityPerformDecrement()
        try await waitFor(host) { store.preparationWorkers == 3 }
        XCTAssertEqual(WorkbenchStore(preferences: defaults, resources: resources).preparationWorkers, 3,
                       "A settings click must persist without dismissing Settings or starting preparation")
        _ = stepper.accessibilityPerformIncrement()
        try await waitFor(host) { store.preparationWorkers == 4 }
        XCTAssertEqual(WorkbenchStore(preferences: defaults, resources: resources).preparationWorkers, 4)
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let attachment = XCTAttachment(image: NSImage(cgImage: try XCTUnwrap(bitmap.cgImage), size: host.bounds.size))
        attachment.name = "Preparation Settings after real increment and decrement activation"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func waitFor(_ host: NSView, condition: () -> Bool) async throws {
        for _ in 0..<200 {
            host.layoutSubtreeIfNeeded()
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Preparation Settings did not reach the expected state")
    }

    private static func workerControl(in view: NSView) -> (any NSAccessibilityProtocol)? {
        var visited = Set<ObjectIdentifier>()
        func find(_ object: Any) -> (any NSAccessibilityProtocol)? {
            guard let element = object as? any NSAccessibilityProtocol else { return nil }
            guard visited.insert(ObjectIdentifier(element)).inserted else { return nil }
            if element.accessibilityRole() == .incrementor { return element }
            for child in element.accessibilityChildren() ?? [] { if let match = find(child) { return match } }
            if let view = object as? NSView {
                for child in view.subviews { if let match = find(child) { return match } }
            }
            return nil
        }
        return find(view)
    }
}
