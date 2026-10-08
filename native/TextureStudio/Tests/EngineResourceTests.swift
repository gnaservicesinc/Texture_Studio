import XCTest
import CoreImage
@testable import TextureStudio

final class EngineResourceTests: XCTestCase {
    func testRenderUsesWholeRecommendedWorkingSetAndCurrentMemory() {
        let gib:UInt64 = 1_073_741_824
        XCTAssertEqual(TextureEngine.renderMemoryBudget(practicalBytes:58*gib,
            metalRecommendedBytes:48*gib,availableBytes:50*gib),48*gib)
        XCTAssertEqual(TextureEngine.renderMemoryBudget(practicalBytes:58*gib,
            metalRecommendedBytes:48*gib,availableBytes:35*gib),35*gib)
        XCTAssertEqual(TextureEngine.renderMemoryBudget(practicalBytes:58*gib,
            metalRecommendedBytes:0,availableBytes:nil),58*gib)
        XCTAssertEqual(TextureEngine.renderMemoryBudget(practicalBytes:12*gib,
            metalRecommendedBytes:14*gib,availableBytes:16*gib),12*gib)
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
        let base = TextureEngine.workingMemoryEstimate(outputSize:8192,sourcePixels:24_000_000,cleanupSide:0)
        let full = TextureEngine.workingMemoryEstimate(outputSize:8192,sourcePixels:24_000_000,cleanupSide:8192)
        XCTAssertEqual(full-base,UInt64(8192)*8192*24)
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
