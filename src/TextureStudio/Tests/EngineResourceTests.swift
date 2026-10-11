import XCTest
import CoreImage
@testable import TextureStudio

final class EngineResourceTests: XCTestCase {
    func testLargePhotoAndAttachedMapImportRetainOriginalDimensionsWithoutOutputLimit() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("large-original.tiff")
        // An ordinary single-strip, 8-bit grayscale TIFF. Sparse storage supplies
        // valid black pixels without allocating a 156 MB fixture in the test.
        let width: UInt32 = 13_000, height: UInt32 = 12_000, offset: UInt32 = 122
        var header = Data([0x49, 0x49])
        func word(_ value: UInt16) { header.append(UInt8(value & 255)); header.append(UInt8(value >> 8)) }
        func dword(_ value: UInt32) { for shift in stride(from: 0, to: 32, by: 8) { header.append(UInt8((value >> shift) & 255)) } }
        word(42); dword(8); word(9)
        let entries: [(UInt16, UInt16, UInt32)] = [
            (256, 4, width), (257, 4, height), (258, 3, 8), (259, 3, 1), (262, 3, 1),
            (273, 4, offset), (277, 3, 1), (278, 4, height), (279, 4, width * height)
        ]
        for (tag, type, value) in entries { word(tag); word(type); dword(1); dword(value) }
        dword(0)
        XCTAssertEqual(header.count, Int(offset))
        try header.write(to: image)
        let file = try FileHandle(forWritingTo: image)
        try file.truncate(atOffset: UInt64(offset) + UInt64(width) * UInt64(height))
        try file.close()
        let engine = TextureEngine()
        let source = try await engine.importPhoto(image)
        XCTAssertGreaterThan(source.pixelWidth * source.pixelHeight, 150_000_000)
        XCTAssertEqual(source.pixelWidth, Int(width))
        XCTAssertEqual(source.pixelHeight, Int(height))
        XCTAssertEqual(source.orientedImage.extent.size, CGSize(width: Int(width), height: Int(height)))
        let attached = try await engine.importDepth(image, matching: source)
        XCTAssertEqual(attached.image.extent.size, source.orientedImage.extent.size)
    }

    func testSourceDimensionsRejectInvalidExtentsAndIntegerOverflow() {
        XCTAssertNotNil(TextureEngine.sourceDimensions(CGRect(x: 0, y: 0, width: 30_000, height: 20_000), minimumSide: 4))
        for rectangle in [CGRect.infinite, CGRect.zero,
                          CGRect(x: 0, y: 0, width: 20.5, height: 40),
                          CGRect(x: 0, y: 0, width: CGFloat(Int.max), height: 40),
                          CGRect(x: 0, y: 0, width: 1 << 32, height: 1 << 32)] {
            XCTAssertNil(TextureEngine.sourceDimensions(rectangle, minimumSide: 4))
        }
    }

    func testDiffusePreparationAcceptsTrainingGridsAndStillRejectsInvalidControls() async throws {
        let image = CIImage(color: CIColor(red: 0.3, green: 0.2, blue: 0.1)).cropped(to: CGRect(x: 0, y: 0, width: 320, height: 240))
        let source = TextureSource(url: URL(fileURLWithPath: "/test-photo.png"), orientedImage: image,
            camera: CameraMetadata(), pixelWidth: 320, pixelHeight: 240)
        let engine = TextureEngine()
        for side in [256, 512] {
            var settings = TextureSettings(); settings.outputSize = side
            let prepared = try await engine.prepareDiffuse(source: source, settings: settings)
            XCTAssertEqual(prepared.diffuse.extent, CGRect(x: 0, y: 0, width: side, height: side))
            settings.lightingStrength = .nan
            do {
                _ = try await engine.prepareDiffuse(source: source, settings: settings)
                XCTFail("Accepting a compact grid must not bypass finite control validation")
            } catch TextureError.invalidSettings(_) { }
        }
    }

    func testDepthCleanupRetainsAvailableSourceDetailThroughEightK() {
        XCTAssertEqual(TextureEngine.depthCleanupSide(outputSize:4098,
            depthExtent:CGRect(x:0,y:0,width:4096,height:4096)),4096)
        XCTAssertEqual(TextureEngine.depthCleanupSide(outputSize:8192,
            depthExtent:CGRect(x:0,y:0,width:8192,height:8192)),8192)
        XCTAssertEqual(TextureEngine.depthCleanupSide(outputSize:1024,
            depthExtent:CGRect(x:0,y:0,width:8192,height:8192)),1024)
        XCTAssertEqual(TextureEngine.depthCleanupSide(outputSize:8192,
            depthExtent:CGRect(x:0,y:0,width:1536,height:1024)),1536)
    }

    func testEXRStripUsesAvailableMemoryAndAccountsForPlanarOutput() {
        let oneColorRow:UInt64 = 8192*28+8
        XCTAssertEqual(FloatEXRWriter.renderStripHeight(width:8192,height:8192,color:true,
            workingMemoryBytes:oneColorRow*512),512)
        XCTAssertEqual(FloatEXRWriter.renderStripHeight(width:8192,height:8192,color:true,
            workingMemoryBytes:oneColorRow*8192),8192)
        XCTAssertEqual(FloatEXRWriter.renderStripHeight(width:8192,height:8192,color:true,
            workingMemoryBytes:0),1)
        XCTAssertEqual(FloatEXRWriter.renderStripHeight(width:8192,height:8192,color:false,
            workingMemoryBytes:(8192*20+8)*256),256)
    }

    func testEXRAdaptiveStripsKeepIdenticalFloatChannelsAndRowOrientation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:root) }
        let width=7,height=19
        var values = [Float](repeating:1,count:width*height*4)
        for row in 0..<height {
            for x in 0..<width {
                values[(row*width+x)*4] = Float(row)*0.041-Float(x)*0.1234567
                values[(row*width+x)*4+1] = Float(row*width+x)*0.01234567
                values[(row*width+x)*4+2] = Float(row)*0.8765432+Float(x)*0.56789
            }
        }
        let image = CIImage(bitmapData:values.withUnsafeBytes { Data($0) },bytesPerRow:width*16,
            size:CGSize(width:width,height:height),format:.RGBAf,colorSpace:nil)
        let context = CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull(),.workingFormat:CIFormat.RGBAf])
        for color in [true,false] {
            let single = root.appendingPathComponent("single-\(color).exr")
            let full = root.appendingPathComponent("full-\(color).exr")
            try FloatEXRWriter.write(image,to:single,context:context,color:color,workingMemoryBytes:0)
            try FloatEXRWriter.write(image,to:full,context:context,color:color,workingMemoryBytes:1_000_000)
            XCTAssertEqual(try Data(contentsOf:single),try Data(contentsOf:full))
            try FloatEXRWriter.verifyChannelPrecision(at:full,expected:.float32)
        }
    }
}
