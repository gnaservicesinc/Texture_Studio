import AppKit
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import TextureStudio

final class MapInspectionTests: XCTestCase {
    func testPNGFullDimensionsAndOriginalBytesRemainIntact() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("original.png")
        let width = 1537, height = 1025
        let samples = [UInt16](repeating: UInt16(32769).bigEndian, count: width * height)
        let data = samples.withUnsafeBytes { Data($0) }
        let image = try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 16,
            bytesPerRow: width * 2, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: [.byteOrder16Big],
            provider: CGDataProvider(data: data as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(source as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let original = try Data(contentsOf: source)
        let loaded = try await ReviewImageLoader.shared.load(source, numeric: true)
        XCTAssertEqual(loaded.pixelWidth, width)
        XCTAssertEqual(loaded.pixelHeight, height)
        let copy = directory.appendingPathComponent("copy.png")
        try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256, to: copy)
        XCTAssertEqual(try Data(contentsOf: copy), original)
        XCTAssertEqual(try Data(contentsOf: source), original)
        XCTAssertThrowsError(try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256, to: copy))
    }

    func testFloatEXRContrastIsAppliedBeforeDisplayQuantizationAndExportUsesPinnedHash() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("height.exr")
        let values: [Float] = [0.4995,0.4995,0.4995,1, 0.5005,0.5005,0.5005,1]
        let image = CIImage(bitmapData: values.withUnsafeBytes { Data($0) }, bytesPerRow: 32,
            size: CGSize(width: 2, height: 1), format: .RGBAf, colorSpace: nil)
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .workingFormat: CIFormat.RGBAf])
        try FloatEXRWriter.write(image, to: source, context: context, color: true)
        let original = try Data(contentsOf: source)
        let loaded = try await ReviewImageLoader.shared.load(source, numeric: true)
        let contrast = try await ReviewImageLoader.shared.load(source, numeric: true, contrast: 32, midpoint: 0.5,
            expectedSHA256: loaded.sourceSHA256)
        XCTAssertEqual(loaded.pixelWidth, 2)
        XCTAssertEqual(loaded.pixelHeight, 1)
        XCTAssertEqual(contrast.sourceSHA256, loaded.sourceSHA256)
        let codes = rgbaCodes(contrast.image)
        XCTAssertGreaterThan(Int(codes[4]) - Int(codes[0]), 5, "Sub-8-bit detail must be amplified before display conversion")
        XCTAssertEqual(try Data(contentsOf: source), original)
        let copy = directory.appendingPathComponent("copy.exr")
        try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256, to: copy)
        XCTAssertEqual(try Data(contentsOf: copy), original)
        try Data("changed".utf8).write(to: source)
        XCTAssertThrowsError(try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256,
            to: directory.appendingPathComponent("refused.exr")))
        do {
            _ = try await ReviewImageLoader.shared.load(source, numeric: true, expectedSHA256: loaded.sourceSHA256)
            XCTFail("A contrast refresh must reject changed originals")
        } catch ReviewImageError.sourceChanged { }
    }

    @MainActor func testLinkedViewportUsesNormalizedPanAndRetinaDevicePixelZoom() {
        let viewport = InspectionViewport()
        XCTAssertFalse(viewport.fitToView)
        XCTAssertEqual(viewport.imageScale(imageSize: CGSize(width: 2048, height: 2048),
            viewSize: CGSize(width: 500, height: 500), backingScale: 2), 0.5)
        viewport.pan(dx: 0.125, dy: -0.25)
        XCTAssertEqual(viewport.normalizedCenter, CGPoint(x: 0.625, y: 0.25))
        viewport.setZoom(2)
        XCTAssertEqual(viewport.imageScale(imageSize: CGSize(width: 1024, height: 1024),
            viewSize: CGSize(width: 500, height: 500), backingScale: 2), 1)
        viewport.fit()
        XCTAssertEqual(viewport.normalizedCenter, CGPoint(x: 0.5, y: 0.5))
        XCTAssertEqual(viewport.imageScale(imageSize: CGSize(width: 2048, height: 1024),
            viewSize: CGSize(width: 512, height: 512), backingScale: 2), 0.25)
        viewport.pan(dx: 99, dy: -99)
        XCTAssertEqual(viewport.normalizedCenter, CGPoint(x: 1, y: 0))
        viewport.setZoom(.nan)
        XCTAssertTrue(viewport.zoom.isFinite)
    }

    @MainActor func testReferenceAndLatestModelAreInitiallyVisible() {
        let candidates = ["flat", "target", "starting_head", "trained_2k"].map {
            MapReviewCandidate(id: $0, label: $0, mapURL: URL(fileURLWithPath: "/\($0).exr"), numeric: true)
        }
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates(candidates).map(\.label), ["target", "trained_2k"])
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates([candidates[0], candidates[1]]).map(\.label), ["target"])
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates([candidates[2], candidates[3]]).map(\.label), ["starting_head", "trained_2k"])
    }

    func testPhotoDisplayHonorsWideGamutProfileWhileNumericCodesStayUnmanaged() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("wide-gamut.png")
        let colors: [UInt8] = [179, 77, 26, 255, 179, 77, 26, 255]
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.displayP3))
        let image = try XCTUnwrap(CGImage(width: 2, height: 1, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 8, space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: CGDataProvider(data: Data(colors) as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(source as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let managed = try await ReviewImageLoader.shared.load(source, numeric: false)
        let raw = try await ReviewImageLoader.shared.load(source, numeric: true)
        let numericCodes = try XCTUnwrap(raw.image.dataProvider?.data) as Data
        let photoCodes = try XCTUnwrap(managed.image.dataProvider?.data) as Data
        XCTAssertEqual(Array(numericCodes.prefix(3)), Array(colors.prefix(3)))
        XCTAssertGreaterThan(abs(Int(photoCodes[0]) - Int(numericCodes[0])), 5)
        XCTAssertEqual(managed.sourceSHA256, raw.sourceSHA256)
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("MapInspectionTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
    private func rgbaCodes(_ image: CGImage) -> [UInt8] {
        var values = [UInt8](repeating: 0, count: image.width * image.height * 4)
        values.withUnsafeMutableBytes { bytes in
            let context = CGContext(data: bytes.baseAddress, width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return values
    }
}
