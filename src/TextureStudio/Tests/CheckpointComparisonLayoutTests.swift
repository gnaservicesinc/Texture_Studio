import AppKit
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class CheckpointComparisonLayoutTests: XCTestCase {
    func testComparisonSetupAndResultsRenderAtMinimumAndLargeWindowSizes() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("comparison-layout-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "comparison-layout-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root)
        let diffuse = try map(in: root, name: "Brick test photo.png", color: true, shift: 0)
        let reference = try map(in: root, name: "reference.png", color: false, shift: 0)
        let predictionA = try map(in: root, name: "model-a.png", color: false, shift: 20)
        let predictionB = try map(in: root, name: "model-b.png", color: false, shift: 40)
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
        {"dataset_path":"\(root.path)","index_sha256":"layout-index","materials":[{"material_id":"brick","name":"Brick test photo","samples":[{"sample_id":"brick-native","status":"approved","split":"train","width":1024,"height":1024,"maps":{"input":{"path":"\(diffuse.path)","width":1024,"height":1024},"height":{"path":"\(reference.path)","width":1024,"height":1024}}}]}]}
        """)
        store.selectedSampleId = "brick-native"
        store.training.size = 1024
        store.comparisonIncludesBase = false
        store.checkpoints = try (0..<12).map { index in
            var checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
            {"checkpoint_path":"\(root.path)/export/checkpoint-step-\(index).safetensors","sha256":"layout-\(index)","schema":"texture-studio-compact-material-v1","architecture":"texture-studio-compact-scalar-native-v1","target":"height","step":\(330 * (index + 1)),"compatible":true}
            """)
            checkpoint.modelName = index < 6 ? "Bricks Small – studio surface detail model" : "Bricks Large – extended natural material variation model"
            return checkpoint
        }
        store.comparisonCheckpointIds = Set(store.checkpoints.suffix(3).map(\.id))
        let resultCandidates = [
            MapReviewCandidate(id: "diffuse", label: "Diffuse source", mapURL: diffuse, numeric: false,
                sampleLabel: "Brick test photo", role: "diffuse"),
            MapReviewCandidate(id: "reference", label: "Source displacement reference", mapURL: reference, numeric: true,
                sampleLabel: "Brick test photo", role: "target", modelIdentity: MapReviewModelIdentity(mapType: "height"))
        ] + store.checkpoints.enumerated().map { index, checkpoint in
            MapReviewCandidate(id: checkpoint.id, label: "export · displacement · old source label",
                mapURL: index.isMultiple(of: 2) ? predictionA : predictionB, numeric: true, sampleLabel: "Brick test photo",
                detail: "Technical checkpoint provenance belongs in Details", role: "checkpoint",
                modelIdentity: MapReviewModelIdentity(checkpointPath: checkpoint.checkpointPath,
                    checkpointSHA256: checkpoint.sha256, checkpointStep: checkpoint.step,
                    architecture: checkpoint.architecture, mapType: "height", modelName: checkpoint.modelName))
        }
        for size in [CGSize(width: 1050, height: 700), CGSize(width: 1600, height: 900)] {
            for populated in [false, true] {
                store.comparisonCandidates = populated ? resultCandidates : []
                let host = NSHostingView(rootView: CheckpointComparisonView(store: store).environment(\.colorScheme, .dark))
                let window = NSWindow(contentRect: CGRect(origin: .zero, size: size),
                    styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false
                window.appearance = NSAppearance(named: .darkAqua)
                window.contentView = host
                window.orderFront(nil)
                defer { window.close() }
                for _ in 0..<100 {
                    host.layoutSubtreeIfNeeded()
                    if Self.splitView(in: host) != nil { break }
                    try await Task.sleep(for: .milliseconds(10))
                }
                // Let asynchronous native map loading and SwiftUI header
                // measurements settle before inspecting the rendered artifact.
                try await Task.sleep(for: .milliseconds(200))
                host.layoutSubtreeIfNeeded(); host.displayIfNeeded()
                let split = try XCTUnwrap(Self.splitView(in: host))
                let frame = split.convert(split.bounds, to: host)
                XCTAssertEqual(frame.width, size.width, accuracy: 1)
                let columns = split.subviews.filter { $0.frame.width > 30 }
                XCTAssertEqual(columns.count, 2, "Setup and results occupy separate bounded columns")
                let rail = try XCTUnwrap(columns.first)
                XCTAssertGreaterThanOrEqual(rail.frame.width, 284)
                XCTAssertLessThanOrEqual(rail.frame.width, 381)
                let activityHeight = host.isFlipped ? host.bounds.maxY - frame.maxY : frame.minY - host.bounds.minY
                XCTAssertGreaterThanOrEqual(activityHeight, 30, "Activity remains visible below the comparison")
                if populated {
                    let displayed = ReviewWorkbenchView.resolvedComparisonCandidates(resultCandidates, visibleIDs: [])
                    XCTAssertEqual(displayed.count, 3)
                    XCTAssertTrue(displayed.first?.displayTitle.hasPrefix("Bricks") == true)
                    XCTAssertFalse(displayed.first?.displayTitle.contains("export") == true)
                }
                try snapshot(host, name: "ipde-comparison-\(Int(size.width))x\(Int(size.height))-\(populated ? "results" : "setup")")
                window.close()
            }
        }
    }

    private func map(in root: URL, name: String, color: Bool, shift: Int) throws -> URL {
        let size = 1024, channels = color ? 3 : 1
        let pixels = Data((0..<size * size * channels).map { index in
            let pixel = index / channels, x = pixel % size, y = pixel / size
            let mortar = x % 32 < 3 || y % 16 < 2
            return UInt8(mortar ? 30 : min(255, 100 + ((x + y + shift) % 90) + (color && index % 3 == 0 ? 40 : 0)))
        })
        let bytes = try NativePNG(header: .init(width: size, height: size, bits: 8, channels: channels,
            color: color ? 2 : 0, interlace: 0), pixels: pixels, colorChunks: []).encoded()
        let url = root.appendingPathComponent(name)
        try bytes.write(to: url)
        return url
    }

    private func snapshot(_ host: NSView, name: String) throws {
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let image = NSImage(cgImage: try XCTUnwrap(bitmap.cgImage), size: host.bounds.size)
        let attachment = XCTAttachment(image: image)
        attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(name + ".png")
        try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: path)
        print("COMPARISON_LAYOUT_SNAPSHOT=\(path.path)")
    }

    private static func splitView(in view: NSView) -> NSSplitView? {
        if let split = view as? NSSplitView { return split }
        return view.subviews.lazy.compactMap { splitView(in: $0) }.first
    }
}
