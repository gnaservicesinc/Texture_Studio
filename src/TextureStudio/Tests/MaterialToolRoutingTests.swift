import Foundation
import XCTest
@testable import TextureStudio

final class MaterialToolRoutingTests: XCTestCase {
    func testInstalledChildFindsItsParentStudioAndEmbeddedSiblings() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let studio = root.appendingPathComponent("Texture Studio.app")
        let trainer = studio.appendingPathComponent("Contents/Applications/Material Trainer.app")
        let review = studio.appendingPathComponent("Contents/Applications/Material Review.app")
        try FileManager.default.createDirectory(at: trainer, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: review, withIntermediateDirectories: true)
        XCTAssertEqual(MaterialToolLauncher.studioURL(containing: trainer).path, studio.path)
        XCTAssertEqual(MaterialToolLauncher.toolURL(.review, containing: trainer).path, review.path)
        XCTAssertEqual(MaterialToolLauncher.toolURL(.review, containing: studio).path, review.path)
    }

    func testSeparatelyCopiedToolRetainsSiblingFallback() {
        let trainer = URL(fileURLWithPath: "/a/local/folder/Material Trainer.app")
        XCTAssertEqual(MaterialToolLauncher.studioURL(containing: trainer).path, "/a/local/folder/Texture Studio.app")
        XCTAssertEqual(MaterialToolLauncher.toolURL(.review, containing: trainer).path, "/a/local/folder/Material Review.app")
    }
}
