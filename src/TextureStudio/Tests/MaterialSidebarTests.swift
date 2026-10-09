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

@MainActor
final class DatasetBrowserControlTests: XCTestCase {
    func testFolderAndCropArrowsSelectImagesAndExposeMapButtons() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dataset-controls-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "dataset-controls-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults, workerOverride: { _, _ in "" })
        store.training.size = 256
        store.selectedRole = "input"
        var materials: [WorkbenchMaterial] = []
        for index in 0..<2 {
            let input = root.appendingPathComponent("input-\(index).png")
            let height = root.appendingPathComponent("height-\(index).png")
            try NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0),
                          pixels: Data(repeating: UInt8(70 + index * 100), count: 256 * 256 * 3), colorChunks: []).encoded().write(to: input)
            try NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 1, color: 0, interlace: 0),
                          pixels: Data(repeating: 100, count: 256 * 256), colorChunks: []).encoded().write(to: height)
            let maps = ["input": WorkbenchMap(path: input.path, sha256: nil, sourceBits: 8, encoding: nil, width: 256, height: 256),
                        "height": WorkbenchMap(path: height.path, sha256: nil, sourceBits: 8, encoding: nil, width: 256, height: 256)]
            let sample = WorkbenchSample(sampleId: "sample-\(index)", status: "unreviewed", split: "train", width: 256, height: 256, maps: maps, note: nil)
            materials.append(WorkbenchMaterial(materialId: "material-\(index)", samples: [sample],
                                              name: "Subject \(index)", subjectId: "subject-\(index)"))
        }
        store.dataset = WorkbenchDataset(datasetPath: root.path, indexSha256: "fixture", materials: materials,
                                         validationScope: nil, crossSizeValidationNotice: nil, preparation: nil, automaticValidation: nil)
        let host = NSHostingView(rootView: DatasetWorkbenchView(store: store).environment(\.colorScheme, .dark))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 1300, height: 850),
                              styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.makeKeyAndOrderFront(nil)
        defer { window.close() }
        try await waitFor(host) { Self.arrowFrames(in: host, indent: 0).count == 2 && Self.arrowFrames(in: host, indent: 20).count == 1 && Self.firstPixel(in: host) == 70 }
        let secondFolder = try XCTUnwrap(Self.arrowFrames(in: host, indent: 0).last)
        try click(secondFolder, in: host, window: window)
        try await waitFor(host) { store.selectedSampleId == "sample-1" && Self.firstPixel(in: host) == 170 }
        XCTAssertEqual(store.selectedRole, "input")
        XCTAssertTrue(try isHighlighted(secondFolder, in: host), "The selected subject row must visibly highlight")
        let cropFrame = secondFolder.offsetBy(dx: 20, dy: 40)
        try click(cropFrame, in: host, window: window)
        try await waitFor(host) { store.selectedSampleId == "sample-1" }
        XCTAssertEqual(store.selectedSampleId, "sample-1")
        XCTAssertTrue(try isHighlighted(cropFrame, in: host), "The selected crop row must visibly highlight")
        try click(cropFrame, in: host, window: window)
        host.layoutSubtreeIfNeeded()
        let mapButtons = Self.imageControlFrames(in: host)
        XCTAssertEqual(mapButtons.count, 4, "Two map icons and Fit/Actual pixels must be below the image")
        try click(mapButtons[3], in: host, window: window)
        try await waitFor(host) { Self.canvases(in: host).first?.viewport?.fitToView == false }
        try click(mapButtons[2], in: host, window: window)
        try await waitFor(host) { Self.canvases(in: host).first?.viewport?.fitToView == true }
        try click(mapButtons[3], in: host, window: window)
        try await waitFor(host) { Self.canvases(in: host).first?.viewport?.fitToView == false }
        try click(mapButtons[1], in: host, window: window)
        try await waitFor(host) { store.selectedRole == "height" && Self.firstPixel(in: host) == 100 }
        let canvas = try XCTUnwrap(Self.canvases(in: host).first)
        XCTAssertEqual(canvas.viewport?.zoom, 1)
        XCTAssertGreaterThan(canvas.bounds.height, 100)
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let image = NSImage(cgImage: try XCTUnwrap(bitmap.cgImage), size: host.bounds.size)
        let attachment = XCTAttachment(image: image)
        attachment.name = "Dataset subject and crop controls after activation"
        attachment.lifetime = .keepAlways
        add(attachment)
        let snapshot = FileManager.default.temporaryDirectory.appendingPathComponent("dataset-controls-validation.png")
        try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: snapshot)
        print("DATASET_CONTROLS_SNAPSHOT=\(snapshot.path)")
    }

    func testFolderSelectionResetsUnavailableRoleAndColorVariant() {
        let suite = "dataset-selection-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults, workerOverride: { _, _ in "" })
        let input = WorkbenchMap(path: "/input.png", sha256: nil, sourceBits: 8, encoding: nil, width: 256, height: 256)
        let sample = WorkbenchSample(sampleId: "diffuse-only", status: "unreviewed", split: "train",
                                     width: 256, height: 256, maps: ["input": input], note: nil)
        store.selectedRole = "normal"
        store.selectedInputVariantId = "old-color"
        store.selectDatasetSample(sample)
        XCTAssertEqual(store.selectedSampleId, sample.id)
        XCTAssertEqual(store.selectedRole, "input")
        XCTAssertNil(store.selectedInputVariantId)
        store.selectedInputVariantId = "chosen-color"
        store.selectDatasetSample(sample, role: "input")
        XCTAssertEqual(store.selectedInputVariantId, "chosen-color", "Map switching must retain this sample’s selected diffuse color")
    }

    private func waitFor(_ host: NSView, condition: () -> Bool) async throws {
        for _ in 0..<300 {
            host.layoutSubtreeIfNeeded()
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("The dataset UI did not reach the expected state")
    }
    private static func firstPixel(in view: NSView) -> UInt8? {
        guard let data = canvases(in: view).first?.image?.dataProvider?.data else { return nil }
        return (data as Data).first
    }
    private static func canvases(in view: NSView) -> [InspectionCanvasView] {
        (view as? InspectionCanvasView).map { [$0] } ?? view.subviews.flatMap { canvases(in: $0) }
    }
    // Hosted SwiftUI exposes native keyboard-focus rectangles even when the
    // test process has no external Accessibility client. Send real window
    // mouse events at those rendered controls and verify their visible result.
    private static func focusFrames(in host: NSView) -> [CGRect] {
        func views(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(views) }
        return views(host).filter { String(describing: type(of: $0)) == "KeyViewProxy" }.map {
            let rect = $0.convert($0.bounds, to: host)
            return host.isFlipped ? rect : CGRect(x: rect.minX, y: host.bounds.maxY - rect.maxY, width: rect.width, height: rect.height)
        }
    }
    private static func arrowFrames(in host: NSView, indent: CGFloat) -> [CGRect] {
        focusFrames(in: host).filter { abs($0.width - 22) < 1 && abs($0.height - 28) < 1 && abs($0.minX - (8 + indent)) < 1 }.sorted { $0.minY < $1.minY }
    }
    private static func imageControlFrames(in host: NSView) -> [CGRect] {
        guard let canvas = canvases(in: host).first else { return [] }
        let frame = canvas.convert(canvas.bounds, to: host)
        let bottom = host.isFlipped ? frame.maxY : host.bounds.maxY - frame.minY
        let candidates = focusFrames(in: host).filter { $0.minX > frame.minX && $0.minY > bottom && $0.minY < bottom + 160 && (20...36).contains($0.height) }
        guard let firstRow = candidates.map(\.minY).min() else { return [] }
        return candidates.filter { abs($0.minY - firstRow) < 3 }.sorted { $0.minX < $1.minX }
    }
    private func click(_ frame: CGRect, in host: NSView, window: NSWindow) throws {
        let topPoint = CGPoint(x: frame.midX, y: frame.midY)
        let point = host.isFlipped ? topPoint : CGPoint(x: topPoint.x, y: host.bounds.maxY - topPoint.y)
        let location = host.convert(point, to: nil)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            let event = try XCTUnwrap(NSEvent.mouseEvent(with: type, location: location, modifierFlags: [],
                timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
                context: nil, eventNumber: 0, clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0))
            window.sendEvent(event)
        }
    }
    private func isHighlighted(_ row: CGRect, in host: NSView) throws -> Bool {
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let color = try XCTUnwrap(bitmap.colorAt(x: 300, y: Int(row.midY))?.usingColorSpace(.deviceRGB))
        return color.blueComponent > color.redComponent + 0.1
    }
}
