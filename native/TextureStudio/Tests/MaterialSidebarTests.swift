import AppKit
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialSidebarTests: XCTestCase {
    func testKeyboardSelectionHandlesEmptyRowsAndRemovedSelection() {
        XCTAssertNil(MaterialSidebarSelection.next("removed", in: [String](), direction: 1))
        XCTAssertEqual(MaterialSidebarSelection.next("removed", in: ["a", "b"], direction: 1), "a")
        XCTAssertEqual(MaterialSidebarSelection.next("removed", in: ["a", "b"], direction: -1), "b")
        XCTAssertEqual(MaterialSidebarSelection.next("a", in: ["a", "b"], direction: -1), "a")
        XCTAssertEqual(MaterialSidebarSelection.next("b", in: ["a", "b"], direction: 1), "b")
    }

    func testDatasetBrowserHasNoAppKitTableDragSelectionHandler() {
        let name = "org.ipde.sidebar-test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let store = WorkbenchStore(preferences: defaults, workerOverride: { _, _ in "" })
        let sample = WorkbenchSample(sampleId: "soil_001", status: "prepared", split: "train",
                                     width: 1024, height: 1024, maps: [:], note: nil)
        store.dataset = WorkbenchDataset(datasetPath: "/fixture", indexSha256: "fixture", materials: [
            WorkbenchMaterial(materialId: "soil", samples: [sample])], validationScope: nil,
            crossSizeValidationNotice: nil, preparation: nil, automaticValidation: nil)
        let host = NSHostingView(rootView: DatasetWorkbenchView(store: store))
        host.frame = CGRect(x:0, y:0, width:1100, height:800)
        host.layoutSubtreeIfNeeded()
        func tables(_ view: NSView) -> Int { (view is NSTableView ? 1 : 0) + view.subviews.reduce(0) { $0 + tables($1) } }
        XCTAssertEqual(tables(host), 0, "The dataset must not re-enter NSTableView's negative-row drag-selection path")
        store.selectedSampleId = "removed"
        store.dataset = nil
        host.layoutSubtreeIfNeeded()
        XCTAssertEqual(tables(host), 0)
    }
}
