import AppKit
import XCTest
@testable import TextureStudio

@MainActor
final class InspectionClippingTests: XCTestCase {
    func testNativeSizeImageCannotPaintOverNeighboringHeadersOrPanes() throws {
        try assertCanvasContainsDrawing(zoom: 1, center: CGPoint(x: 0.5, y: 0.5))
    }

    func testMagnifiedAndPannedImageCannotPaintOverNeighboringHeadersOrPanes() throws {
        try assertCanvasContainsDrawing(zoom: 2, center: CGPoint(x: 0.18, y: 0.81))
    }

    private func assertCanvasContainsDrawing(zoom: CGFloat, center: CGPoint,
                                            file: StaticString = #filePath, line: UInt = #line) throws {
        // The bitmap includes the area occupied by neighboring UI. Calling draw
        // with no enclosing AppKit clip reproduces a representable whose large
        // native image escaped its own frame. A property-only clipping assertion
        // would not detect that drawing regression.
        let bitmapSize = 64
        let frame = CGRect(x: 20, y: 20, width: 24, height: 24)
        let context = try XCTUnwrap(CGContext(data: nil, width: bitmapSize, height: bitmapSize,
            bitsPerComponent: 8, bytesPerRow: bitmapSize * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 20.0 / 255, green: 100.0 / 255, blue: 180.0 / 255, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: bitmapSize, height: bitmapSize))
        let original = Array(UnsafeBufferPointer(start: try XCTUnwrap(context.data).assumingMemoryBound(to: UInt8.self),
                                                count: bitmapSize * bitmapSize * 4))

        let viewport = InspectionViewport()
        viewport.setZoom(zoom)
        viewport.normalizedCenter = center
        let view = InspectionCanvasView(frame: CGRect(origin: .zero, size: frame.size))
        view.viewport = viewport
        view.image = try solidRedImage()

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: true)
        context.saveGState()
        context.translateBy(x: frame.minX, y: frame.minY)
        view.draw(view.bounds)
        context.restoreGState()
        NSGraphicsContext.restoreGraphicsState()

        let pixels = Array(UnsafeBufferPointer(start: try XCTUnwrap(context.data).assumingMemoryBound(to: UInt8.self),
                                              count: bitmapSize * bitmapSize * 4))
        var changedOutside = 0
        for y in 0..<bitmapSize {
            for x in 0..<bitmapSize where !frame.contains(CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y) + 0.5)) {
                let offset = (y * bitmapSize + x) * 4
                if pixels[offset..<(offset + 4)] != original[offset..<(offset + 4)] { changedOutside += 1 }
            }
        }
        XCTAssertEqual(changedOutside, 0,
            "Zoomed map painted \(changedOutside) pixels outside the canvas, covering neighboring labels", file: file, line: line)
        let centerOffset = (32 * bitmapSize + 32) * 4
        XCTAssertGreaterThan(pixels[centerOffset], 240, "The test must draw the image inside its frame", file: file, line: line)
        // Quartz may color-manage DeviceRGB red into the bitmap. It must
        // still be red, rather than merely the pale canvas background.
        XCTAssertLessThan(pixels[centerOffset + 1], 80, file: file, line: line)
        XCTAssertLessThan(pixels[centerOffset + 2], 80, file: file, line: line)
    }

    private func solidRedImage() throws -> CGImage {
        let context = try XCTUnwrap(CGContext(data: nil, width: 512, height: 512, bitsPerComponent: 8,
            bytesPerRow: 512 * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 512, height: 512))
        return try XCTUnwrap(context.makeImage())
    }
}
