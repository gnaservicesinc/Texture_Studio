import Foundation
import CoreImage
import XCTest
@testable import TextureStudio

private enum SpatialFusionFixture {
    static let side = 512
    static func image(seed: UInt32, gains: [Float] = [1,1,1], changedDetail: Bool = false) -> CIImage {
        let gridSide = 65
        var random = seed
        var grid = [Float](repeating:0,count:gridSide*gridSide)
        for i in grid.indices {
            random = random &* 1664525 &+ 1013904223
            grid[i] = 0.14 + Float(random % 65536) / 65536 * 0.55
        }
        var samples = [Float](repeating:1,count:side*side*4)
        for y in 0..<side {
            for x in 0..<side {
                let gx = x/8, gy = y/8
                let tx = Float(x%8)/8, ty = Float(y%8)/8
                let v = (grid[gy*gridSide+gx]*(1-tx)+grid[gy*gridSide+gx+1]*tx)*(1-ty) +
                    (grid[(gy+1)*gridSide+gx]*(1-tx)+grid[(gy+1)*gridSide+gx+1]*tx)*ty
                let delta: Float = changedDetail ? sin(Float(x)*0.47+Float(y)*0.19)*0.012 : 0
                let p = (y*side+x)*4
                for c in 0..<3 {
                    let colour = v * [Float(1),0.92,0.81][c]
                    samples[p+c] = (colour+delta) * gains[c]
                }
            }
        }
        return CIImage(bitmapData:samples.withUnsafeBytes{Data($0)},bytesPerRow:side*16,
            size:CGSize(width:side,height:side),format:.RGBAf,
            colorSpace:CGColorSpace(name:CGColorSpace.extendedLinearSRGB))
    }
    static func companion(seed: UInt32, translated: Bool) -> CIImage {
        let image = image(seed:seed,gains:[0.65,0.80,0.72],changedDetail:true)
            .transformed(by:CGAffineTransform(scaleX:0.5,y:0.5))
        return translated ? image.transformed(by:CGAffineTransform(translationX:10,y:6))
            .composited(over:CIImage(color:CIColor.clear).cropped(to:CGRect(x:0,y:0,width:side/2,height:side/2)))
            .cropped(to:CGRect(x:0,y:0,width:side/2,height:side/2)) : image
    }
    static func source(reference: CIImage, companion: CIImage) -> TextureSource {
        TextureSource(url:URL(fileURLWithPath:"/tmp/spatial-surface.heic"),orientedImage:reference,
            camera:CameraMetadata(),pixelWidth:side,pixelHeight:side,supportingViews:[companion])
    }
    static func pixels(_ image: CIImage) -> [Float] {
        let space = CGColorSpace(name:CGColorSpace.extendedLinearSRGB)!
        let context = CIContext(options:[.workingColorSpace:space,.workingFormat:CIFormat.RGBAf])
        var values = [Float](repeating:0,count:side*side*4)
        values.withUnsafeMutableBytes{context.render(image,toBitmap:$0.baseAddress!,rowBytes:side*16,
            bounds:CGRect(x:0,y:0,width:side,height:side),format:.RGBAf,colorSpace:space)}
        return values
    }
}


/// Exercises the complete native Vision registration → exposure fit → validity gate →
/// actual pixel blend, rather than testing only the numerical patch assessor.
final class SpatialFusionTests: XCTestCase {
    func testVisionFusesSameSurfaceAtDifferentResolutionExposureAndColor() async throws {
        let reference = SpatialFusionFixture.image(seed:11)
        let companion = SpatialFusionFixture.companion(seed:11,translated:false)
        XCTAssertEqual(companion.extent.width,reference.extent.width/2)
        let result = try await PhotoEvidenceService.fuse(source:
            SpatialFusionFixture.source(reference:reference,companion:companion))
        XCTAssertTrue(result.warnings.contains { $0.contains("supporting samples blended") },result.warnings.joined())
        let metrics = compare(reference,result.image)
        // The companion contains a small independent detail signal. This must reach
        // the accepted output, proving that a positive status is not a no-op blend.
        XCTAssertGreaterThan(metrics.meanDifference,0.0002)
        XCTAssertLessThan(metrics.meanDifference,0.008)
        XCTAssertLessThan(metrics.maximumDifference,0.04)
        XCTAssertEqual(metrics.minimumAlpha,1,accuracy:1e-6)
        XCTAssertEqual(result.image.extent,reference.extent)
    }

    func testVisionRegistersTranslationAndKeepsUncoveredEdgesExactlyPrimary() async throws {
        let reference = SpatialFusionFixture.image(seed:11)
        let companion = SpatialFusionFixture.companion(seed:11,translated:true)
        let result = try await PhotoEvidenceService.fuse(source:
            SpatialFusionFixture.source(reference:reference,companion:companion))
        XCTAssertTrue(result.warnings.contains { $0.contains("supporting samples blended") },result.warnings.joined())
        let metrics = compare(reference,result.image)
        XCTAssertGreaterThan(metrics.meanDifference,0.0002)
        XCTAssertLessThan(metrics.meanDifference,0.008)
        XCTAssertLessThan(metrics.maximumDifference,0.04)
        // Translating the companion +20,+12 source pixels leaves the right and
        // bottom of its registered frame uncovered. Those pixels keep the primary.
        XCTAssertEqual(metrics.uncoveredEdgeDifference,0,accuracy:1e-6)
        XCTAssertEqual(metrics.minimumAlpha,1,accuracy:1e-6)
        XCTAssertEqual(result.image.extent,reference.extent)
    }

    func testVisionRejectsUnrelatedSurfaceAndKeepsEveryPrimaryPixel() async throws {
        let reference = SpatialFusionFixture.image(seed:11)
        let unrelated = SpatialFusionFixture.companion(seed:971,translated:false)
        let result = try await PhotoEvidenceService.fuse(source:
            SpatialFusionFixture.source(reference:reference,companion:unrelated))
        XCTAssertFalse(result.warnings.contains { $0.contains("supporting samples blended") })
        XCTAssertTrue(result.warnings.contains { $0.contains("primary photo retained") })
        let metrics = compare(reference,result.image)
        XCTAssertEqual(metrics.meanDifference,0)
        XCTAssertEqual(metrics.maximumDifference,0)
        XCTAssertEqual(result.image.extent,reference.extent)
    }

    private func compare(_ reference: CIImage, _ fused: CIImage) ->
        (meanDifference:Float,maximumDifference:Float,uncoveredEdgeDifference:Float,minimumAlpha:Float) {
        let a = SpatialFusionFixture.pixels(reference), b = SpatialFusionFixture.pixels(fused)
        var total:Float = 0, maximum:Float = 0, edges:Float = 0, alpha:Float = 1
        for y in 0..<SpatialFusionFixture.side {
            for x in 0..<SpatialFusionFixture.side {
                let p = (y*SpatialFusionFixture.side+x)*4
                alpha = min(alpha,b[p+3])
                for channel in 0..<3 {
                    let difference = abs(a[p+channel]-b[p+channel])
                    total += difference
                    maximum = max(maximum,difference)
                    if x >= 506 || y < 6 { edges = max(edges,difference) }
                }
            }
        }
        return (total/Float(SpatialFusionFixture.side*SpatialFusionFixture.side*3),maximum,edges,alpha)
    }
}
