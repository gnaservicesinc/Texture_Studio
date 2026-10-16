import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

final class PhotoEnhancerTests: XCTestCase {
    func testIncludedRGBModelAppliesStrengthOnCPUWhenFixtureIsProvided() async throws {
        guard let path = ProcessInfo.processInfo.environment["LOOM_PHOTO_STUDIO_FIXTURE"] else { throw XCTSkip("Optional staged Core ML RGB fixture; no GPU work.") }
        let service = PhotoEnhancerService(resourceRoot:URL(fileURLWithPath:path),computeUnits:.cpuOnly)
        let image = CIImage(color:CIColor(red:0.4,green:0.5,blue:0.2)).cropped(to:CGRect(x:0,y:0,width:1024,height:1024))
        let result = try await service.apply(image,repair:0.5,lighting:0)
        let context = CIContext(options:[.useSoftwareRenderer:true,.workingFormat:CIFormat.RGBAf])
        func pixel(_ value:CIImage) -> [Float] {
            var out = [Float](repeating:0,count:4)
            out.withUnsafeMutableBytes { context.render(value,toBitmap:$0.baseAddress!,rowBytes:16,bounds:CGRect(x:0,y:0,width:1,height:1),format:.RGBAf,colorSpace:CGColorSpace(name:CGColorSpace.sRGB)!) }
            return out
        }
        let before = pixel(image), after = pixel(result)
        for i in 0..<3 { XCTAssertEqual(after[i],min(1,max(0,before[i]+[Float(0.1),-0.15,0.2][i])),accuracy:0.0002) }
        XCTAssertEqual(after[3],1)
    }
    func testFreshPhotoIsNeutralAndOldExplicitBalanceSurvives() throws {
        let settings = TextureSettings()
        XCTAssertEqual(settings.lightingStrength, 0)
        XCTAssertEqual(settings.photoRepairStrength, 0); XCTAssertEqual(settings.photoLightingStrength, 0)
        let old = try JSONDecoder().decode(TextureSettings.self, from: Data("{\"lightingStrength\":0.65}".utf8))
        XCTAssertEqual(old.lightingStrength, 0.65)
        XCTAssertEqual(old.photoRepairStrength,0)
        var updated = settings; updated.photoRepairStrength = 0.35; updated.photoLightingStrength = 0.6
        XCTAssertEqual(try JSONDecoder().decode(TextureSettings.self,from:JSONEncoder().encode(updated)), updated)
        XCTAssertNotEqual(MaterialRenderKey(sourceIdentity:"a",depthIdentity:"b",settings:settings),MaterialRenderKey(sourceIdentity:"a",depthIdentity:"b",settings:updated))
    }
    func testDisabledEnhancersBypassWithoutModelsOrPixelEvaluation() async throws {
        let image = CIImage(color:CIColor(red:0.2,green:0.4,blue:0.8)).cropped(to:CGRect(x:0,y:0,width:39,height:41))
        let output = try await PhotoEnhancerService().apply(image,repair:0,lighting:0)
        XCTAssertTrue(output === image)
        XCTAssertEqual(try PhotoEnhancerDescriptor.identities(TextureSettings()),[:])
    }
    func testDescriptorRejectsNormalProductBeforeUsingItAsRGB() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:directory) }
        let row:[String:Any] = ["schema":"loom.photo-enhancer.v1","name":"wrong normal","product":"normal","architecture":"loom-factory-material-compact-naf-v1", "width":32,"input":"rgb","output":"material_map","pixelContract":PhotoEnhancerDescriptor.contract,"modelFile":"Enhancer.mlmodel","modelSHA256":"x","weights":"weights.safetensors","weightsSHA256":"y","runID":"test","updates":1,"qualified":false]
        try JSONSerialization.data(withJSONObject:row).write(to:directory.appendingPathComponent("enhancer.json"))
        XCTAssertThrowsError(try PhotoEnhancerDescriptor.load(directory, product:"photo-repair-v1"))
    }
}
