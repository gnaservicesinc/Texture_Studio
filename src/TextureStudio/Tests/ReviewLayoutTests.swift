import AppKit
import CoreImage
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class ReviewLayoutTests: XCTestCase {
    func testFourNativeImagePanesRetainSpaceForTheirModelHeaders() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("review-layout-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "review-layout-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let titles = ["Reference displacement", "Flat baseline · no model",
                      "Starting trained displacement · four-material-adaptation-01/frozen",
                      "Trained 2K displacement · native-2k-material-cycle-01"]
        let details = ["Dataset reference · not a model output · 16-bit source · linear data",
                       "Constant height · no model inference",
                       "DINOv2 Base + material-height head · Step 1,200 · SHA256 2bec143e0c5f",
                       "DINOv2 Base + material-height head · Step 800 · SHA256 1a3b86ed7008"]
        var candidates: [MapReviewCandidate] = []
        let image = CIImage(color: CIColor(red: 0.9, green: 0.9, blue: 0.9))
            .cropped(to: CGRect(x: 0, y: 0, width: 512, height: 512))
        for index in 0..<4 {
            let map = root.appendingPathComponent("candidate-\(index).png")
            try CIContext().writePNGRepresentation(of: image, to: map, format: .RGBA8,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
            candidates.append(MapReviewCandidate(id: "candidate-\(index)", label: titles[index], mapURL: map,
                numeric: true, sampleLabel: "broken_brick_wall_auto_003", detail: details[index],
                role: index == 0 ? "target" : index == 1 ? "base" : "checkpoint"))
        }
        preferences.set(["zoom": 2.0, "fit": false, "contrast": 1.0, "midpoint": 0.5],
            forKey: ReviewWorkbenchView.displayPreferenceKey(candidates))
        let view = ReviewWorkbenchView(candidates: candidates, preferences: preferences)
        let hosting = NSHostingView(rootView: view.environment(\.colorScheme, .dark))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 1500, height: 850),
            styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = hosting
        window.orderFront(nil)
        defer { window.close() }
        var canvases: [InspectionCanvasView] = []
        for _ in 0..<500 {
            hosting.layoutSubtreeIfNeeded()
            canvases = Self.canvases(in: hosting).filter { $0.image != nil }
            let origins = canvases.map { canvas in
                let frame = canvas.convert(canvas.bounds, to: hosting)
                return hosting.isFlipped ? frame.minY : hosting.bounds.maxY - frame.maxY
            }
            if canvases.count == 4, (origins.max() ?? 0) - (origins.min() ?? 0) < 1 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(canvases.count, 4, "All four maps must load at native size")
        var imageOrigins: [CGFloat] = []
        for canvas in canvases {
            let frame = canvas.convert(canvas.bounds, to: hosting)
            let headerSpace = hosting.isFlipped ? frame.minY : hosting.bounds.maxY - frame.maxY
            XCTAssertGreaterThan(headerSpace, 200, "A native image must leave room for its identity and export controls")
            XCTAssertGreaterThan(frame.height, 200, "Headers cannot consume the entire inspection area")
            imageOrigins.append(headerSpace)
        }
        XCTAssertLessThan((imageOrigins.max() ?? 0) - (imageOrigins.min() ?? 0), 1,
                          "Linked comparisons must align their image areas despite differently sized model labels")
        let bitmap = try XCTUnwrap(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
        hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
        let attachment = XCTAttachment(image: NSImage(cgImage: try XCTUnwrap(bitmap.cgImage), size: hosting.bounds.size))
        attachment.name = "Four labeled native maps at 200 percent"
        attachment.lifetime = .keepAlways
        add(attachment)
        let snapshot = FileManager.default.temporaryDirectory.appendingPathComponent("review-labels-validation-\(UUID().uuidString).png")
        try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: snapshot)
        print("REVIEW_LAYOUT_SNAPSHOT=\(snapshot.path)")
    }

    private static func canvases(in view: NSView) -> [InspectionCanvasView] {
        (view as? InspectionCanvasView).map { [$0] } ?? view.subviews.flatMap { canvases(in: $0) }
    }
}
