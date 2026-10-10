import XCTest
import CoreImage
import ImageIO
@testable import TextureStudio

final class EngineTests: XCTestCase {
    func testMaximumSquareMatchesAnalyticTrapezoid() throws {
        let polygon = [CGPoint(x:0,y:0),CGPoint(x:1000,y:0),CGPoint(x:900,y:1000),CGPoint(x:100,y:1000)]
        let crop = try TextureGeometry.maximumCrop(in:polygon,inset:0)
        XCTAssertEqual(crop.width,1000/1.2,accuracy:1e-7)
        XCTAssertEqual(crop.minY,0,accuracy:1e-7)
        try assertValid(crop,in:polygon,inset:0)
    }

    func testMaximumCropKeepsLandscapeCentredAndSquare() throws {
        let polygon = try TextureGeometry.projectedCorners(width:4000,height:3000,xDegrees:0,yDegrees:0,zDegrees:0,focalPixels:5000)
        let crop = try TextureGeometry.maximumCrop(in:polygon)
        XCTAssertEqual(crop.width,2997,accuracy:1e-7)
        XCTAssertEqual(crop.midX,2000,accuracy:1e-7)
        XCTAssertEqual(crop.midY,1500,accuracy:1e-7)
        try assertValid(crop,in:polygon,inset:1.5)
    }

    func testUserExampleAndExtremeTransformsLeaveNoBlankCropCorners() throws {
        for angles in [(7.93,0.0,0.22),(-40.0,20.0,15.0),(0.0,0.0,45.0),(70.0,0.0,450.0)] {
            let polygon = try TextureGeometry.projectedCorners(width:1024,height:1024,
                xDegrees:angles.0,yDegrees:angles.1,zDegrees:angles.2,focalPixels:1800)
            let crop = try TextureGeometry.maximumCrop(in:polygon)
            XCTAssertEqual(crop.width,crop.height,accuracy:1e-7)
            try assertValid(crop,in:polygon,inset:1.5)
            for scale in [0.5, 0.01] {
                let framed = try TextureGeometry.framedCrop(crop,scale:scale,offsetX:1,offsetY:-1)
                XCTAssertGreaterThanOrEqual(framed.minX,crop.minX-1e-7)
                XCTAssertGreaterThanOrEqual(framed.minY,crop.minY-1e-7)
                XCTAssertLessThanOrEqual(framed.maxX,crop.maxX+1e-7)
                XCTAssertLessThanOrEqual(framed.maxY,crop.maxY+1e-7)
                try assertValid(framed,in:polygon,inset:1.5)
            }
        }
    }

    func testInvalidGeometryAndDepthSamplesAreRejected() throws {
        XCTAssertThrowsError(try TextureGeometry.projectedCorners(width:1024,height:1024,xDegrees:71,yDegrees:0,zDegrees:0,focalPixels:1800))
        XCTAssertThrowsError(try TextureGeometry.framedCrop(CGRect(x:0,y:0,width:10,height:10),scale:1.2,offsetX:0,offsetY:0))
        XCTAssertThrowsError(try TextureDepth(width:2,height:2,values:[0,1,.nan,3],sourceLabel:"invalid"))
        XCTAssertThrowsError(try TextureDepth(width:2,height:2,values:[1],sourceLabel:"invalid"))
    }

    func testStereoCompanionsRequireDocumentedGroupMembership() {
        let nested:[String:Any] = [kCGImagePropertyFileContentsDictionary as String:[
            kCGImagePropertyGroups as String:[[
                kCGImagePropertyGroupType as String:kCGImagePropertyGroupTypeStereoPair as String,
                kCGImagePropertyGroupImageIndexLeft as String:1,
                kCGImagePropertyGroupImageIndexRight as String:2,
                kCGImagePropertyGroupImageIndexMonoscopic as String:0]]]]
        XCTAssertEqual(TextureEngine.supportingImageIndices(globalProperties:nested,
            perImageProperties:[[:],[:],[:],[:]],primaryIndex:0),[1,2])
        let left:[String:Any] = [kCGImagePropertyGroups as String:[
            kCGImagePropertyGroupType as String:kCGImagePropertyGroupTypeStereoPair as String,
            kCGImagePropertyGroupIndex as String:0,kCGImagePropertyGroupImageIsLeftImage as String:true]]
        let right:[String:Any] = [kCGImagePropertyGroups as String:[
            kCGImagePropertyGroupType as String:kCGImagePropertyGroupTypeStereoPair as String,
            kCGImagePropertyGroupIndex as String:0,kCGImagePropertyGroupImageIsRightImage as String:true]]
        XCTAssertEqual(TextureEngine.supportingImageIndices(globalProperties:[:],
            perImageProperties:[left,right,[:]],primaryIndex:0),[1])
        XCTAssertEqual(TextureEngine.supportingImageIndices(globalProperties:[:],
            perImageProperties:[[:],[:],[:]],primaryIndex:0),[])
    }

    func testOldSettingsDecodeWithSafeSurfaceHeightDefaults() throws {
        let data=Data("{\"outputSize\":2048,\"useEmbeddedDepth\":true}".utf8)
        let settings=try JSONDecoder().decode(TextureSettings.self,from:data)
        let encoded = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(settings)) as? [String: Any])
        XCTAssertNil(encoded["useEmbeddedDepth"], "Retired portrait-depth settings must not be written into recipes or exports")
        XCTAssertEqual(settings.heightDetail,0)
        XCTAssertEqual(settings.surfacePlaneRemoval,1)
        XCTAssertEqual(settings.depthCleanup,0.25)
        XCTAssertTrue(settings.adaptiveRelief)
        XCTAssertTrue(settings.attachedMapIsHeight)
        XCTAssertEqual(settings.outputSize,2048)
    }

    func testDistancePlaneBecomesFlatWithoutAlbedoGeometry() throws {
        let side=64
        let values=(0..<side*side).map { Float(2)+Float($0%side)*0.007+Float($0/side)*0.003 }
        let result=try SurfaceHeightProcessor.derive(values:values,width:side,height:side,
            interpretation:.distance,planeRemoval:1,cleanup:0.25,strength:1,invert:false,adaptive:false)
        XCTAssertFalse(result.hasRelief)
        XCTAssertTrue(result.values.allSatisfy { $0 == 0.5 })
    }

    func testSurfaceDepthKeepsBroadBumpInsteadOfOnlyEdgeHaloAndPreservesRawSamples() throws {
        let side=96
        var values=[Float](repeating:0,count:side*side)
        for y in 0..<side {
            for x in 0..<side {
                let dx=Float(x-48)/20,dy=Float(y-48)/20
                values[y*side+x]=2+Float(x)*0.002+Float(y)*0.001-0.08*exp(-(dx*dx+dy*dy))
            }
        }
        let original=values.map(\.bitPattern)
        let result=try SurfaceHeightProcessor.derive(values:values,width:side,height:side,
            interpretation:.distance,planeRemoval:1,cleanup:0.25,strength:1,invert:false,adaptive:false)
        XCTAssertTrue(result.hasRelief)
        XCTAssertGreaterThan(result.values[48*side+48],0.8)
        XCTAssertGreaterThan(result.values[48*side+58],0.7)
        XCTAssertLessThan(abs(result.values[5*side+5]-0.5),0.08)
        XCTAssertEqual(values.map(\.bitPattern),original)
        let inverted=try SurfaceHeightProcessor.derive(values:values,width:side,height:side,
            interpretation:.distance,planeRemoval:1,cleanup:0.25,strength:1,invert:true,adaptive:false)
        XCTAssertLessThan(inverted.values[48*side+48],0.2)
    }

    func testNearFlatDistanceProtectionDoesNotStretchSmallResidualToFullRange() throws {
        let side=64
        let values=(0..<side*side).map { Float(10)+($0%2 == 0 ? Float(0.01) : Float(-0.01)) }
        let protected=try SurfaceHeightProcessor.derive(values:values,width:side,height:side,
            interpretation:.distance,planeRemoval:0,cleanup:0,strength:1,invert:false)
        let unprotected=try SurfaceHeightProcessor.derive(values:values,width:side,height:side,
            interpretation:.distance,planeRemoval:0,cleanup:0,strength:1,invert:false,adaptive:false)
        XCTAssertEqual(protected.amplitudeGain,0.04,accuracy:0.00001)
        XCTAssertLessThan(protected.values.max()!-protected.values.min()!,0.037)
        XCTAssertEqual(unprotected.values.max()!-unprotected.values.min()!,0.9,accuracy:0.00001)
        XCTAssertEqual(unprotected.amplitudeGain,1)
    }

    func testProtectionPreservesRichDistanceAndArtistInverseHeightMaps() throws {
        let side=64
        let rich=(0..<side*side).map { Float(10)+($0%2 == 0 ? Float(1) : Float(-1)) }
        let strong=try SurfaceHeightProcessor.derive(values:rich,width:side,height:side,
            interpretation:.distance,planeRemoval:0,cleanup:0,strength:1,invert:false)
        let lowContrast=rich.map { 10+($0-10)*0.01 }
        let artist=try SurfaceHeightProcessor.derive(values:lowContrast,width:side,height:side,
            interpretation:.inverseDepth,planeRemoval:0,cleanup:0,strength:1,invert:false)
        XCTAssertEqual(strong.amplitudeGain,1)
        XCTAssertEqual(artist.amplitudeGain,1)
        XCTAssertEqual(strong.values.max()!-strong.values.min()!,0.9,accuracy:0.00001)
        XCTAssertEqual(artist.values.max()!-artist.values.min()!,0.9,accuracy:0.00001)
    }

    func testDepthOnlyCleanupReducesIsolatedSpikeAndRetainsStep() throws {
        let side=64
        var values=[Float](repeating:1,count:side*side)
        for y in 0..<side {
            for x in 32..<side { values[y*side+x]=1.1 }
        }
        values[16*side+16]=1.9
        let result=try SurfaceHeightProcessor.derive(values:values,width:side,height:side,
            interpretation:.inverseDepth,planeRemoval:0,cleanup:0.25,strength:1,invert:false)
        XCTAssertLessThan(abs(result.values[16*side+16]-result.values[16*side+15]),0.01)
        XCTAssertGreaterThan(result.values[32*side+40]-result.values[32*side+24],0.7)
    }

    func testDepthBitmapRowsAreTopDownWithoutAnExtraFlip() throws {
        let depth=try TextureDepth(width:2,height:2,values:[1,2,3,4],sourceLabel:"orientation")
        let context=CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull(),.workingFormat:CIFormat.RGBAf])
        var top=[Float](repeating:0,count:2),bottom=top
        top.withUnsafeMutableBytes { context.render(depth.image,toBitmap:$0.baseAddress!,rowBytes:8,
            bounds:CGRect(x:0,y:1,width:2,height:1),format:.Rf,colorSpace:nil) }
        bottom.withUnsafeMutableBytes { context.render(depth.image,toBitmap:$0.baseAddress!,rowBytes:8,
            bounds:CGRect(x:0,y:0,width:2,height:1),format:.Rf,colorSpace:nil) }
        XCTAssertEqual(top,[1,2]);XCTAssertEqual(bottom,[3,4])
    }

    func testNormalKernelMatchesAnalyticNonflatOpenGLRamp() throws {
        let library=try XCTUnwrap(Bundle.main.url(forResource:"TextureKernels",withExtension:"metallib"))
        let kernel=try CIKernel(functionName:"textureNormal",fromMetalLibraryData:Data(contentsOf:library))
        let side=64
        var pixels=[Float](repeating:1,count:side*side*4)
        for row in 0..<side {
            for x in 0..<side {
                let value=Float(x)*0.004+Float(side-1-row)*0.008
                for channel in 0..<3 { pixels[(row*side+x)*4+channel]=value }
            }
        }
        let height=CIImage(bitmapData:pixels.withUnsafeBytes { Data($0) },bytesPerRow:side*16,
            size:CGSize(width:side,height:side),format:.RGBAf,colorSpace:nil)
        let normal=try XCTUnwrap(kernel.apply(extent:height.extent,roiCallback:{ _,r in r.insetBy(dx:-2,dy:-2) },
            arguments:[height.clampedToExtent(),Float(3)]))
        let context=CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull(),.workingFormat:CIFormat.RGBAf])
        var actual=[Float](repeating:0,count:4)
        actual.withUnsafeMutableBytes { context.render(normal,toBitmap:$0.baseAddress!,rowBytes:16,
            bounds:CGRect(x:32,y:32,width:1,height:1),format:.RGBAf,colorSpace:nil) }
        let magnitude=sqrt(Float(1)+0.024*0.024+0.048*0.048)
        let expected=[Float(-0.024)/magnitude*0.5+0.5,Float(-0.048)/magnitude*0.5+0.5,1/magnitude*0.5+0.5,1]
        for index in 0..<4 { XCTAssertEqual(actual[index],expected[index],accuracy:1e-6) }
    }

    func testPhotoPatternsAndLegacyPortraitSettingsNeverBecomeDefaultRelief() async throws {
        let engine=TextureEngine()
        let side=64
        var colours=[Float](repeating:1,count:side*side*4)
        for y in 0..<side {
            for x in 0..<side {
                let pattern:Float = (x/8+y/8)%2 == 0 ? 0.1 : 0.9
                for channel in 0..<3 { colours[(y*side+x)*4+channel]=pattern }
            }
        }
        let image=CIImage(bitmapData:colours.withUnsafeBytes { Data($0) },bytesPerRow:side*16,
            size:CGSize(width:side,height:side),format:.RGBAf,colorSpace:CGColorSpace(name:CGColorSpace.extendedLinearSRGB))
        let source=TextureSource(url:URL(fileURLWithPath:"/tmp/colour-pattern.png"),orientedImage:image,
            camera:CameraMetadata(),pixelWidth:side,pixelHeight:side)
        let settings=try JSONDecoder().decode(TextureSettings.self,
            from:Data("{\"useEmbeddedDepth\":true}".utf8))
        let result=try await engine.process(source:source,settings:settings)
        let context=CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull(),.workingFormat:CIFormat.RGBAf])
        var values=[Float](repeating:0,count:32*32)
        let resized=result.height.transformed(by:CGAffineTransform(scaleX:1.0/32,y:1.0/32))
        values.withUnsafeMutableBytes { context.render(resized,toBitmap:$0.baseAddress!,rowBytes:32*4,
            bounds:resized.extent,format:.Rf,colorSpace:nil) }
        XCTAssertTrue(values.allSatisfy { abs($0-0.5) < 1e-6 })
        XCTAssertTrue(result.depthOrigin.contains("Flat"))
    }

    func testMaterialPhotoImportRetainsColourAuxiliariesButNeverReadsPortraitDepth() {
        let types = TextureEngine.materialPhotoAuxiliaryTypes.map { $0.0 as String }
        XCTAssertFalse(types.contains(kCGImageAuxiliaryDataTypeDepth as String))
        XCTAssertFalse(types.contains(kCGImageAuxiliaryDataTypeDisparity as String))
        XCTAssertEqual(Set(types), Set([
            kCGImageAuxiliaryDataTypeHDRGainMap as String,
            kCGImageAuxiliaryDataTypeISOGainMap as String,
            kCGImageAuxiliaryDataTypePortraitEffectsMatte as String,
            kCGImageAuxiliaryDataTypeSemanticSegmentationHairMatte as String,
            kCGImageAuxiliaryDataTypeSemanticSegmentationSkinMatte as String,
            kCGImageAuxiliaryDataTypeSemanticSegmentationSkyMatte as String]))
    }

    func testLowResolutionDepthUpsamplingDoesNotCreateRaisedBorderRim() async throws {
        let engine=TextureEngine()
        let side=512,depthSide=32
        let photo=CIImage(color:CIColor(red:0.4,green:0.4,blue:0.4))
            .cropped(to:CGRect(x:0,y:0,width:side,height:side))
        let source=TextureSource(url:URL(fileURLWithPath:"/tmp/edge-surface.png"),orientedImage:photo,
            camera:CameraMetadata(),pixelWidth:side,pixelHeight:side)
        let values=(0..<depthSide*depthSide).map { index in
            Float(2)+0.04*sin(Float(index%depthSide)*2*Float.pi/Float(depthSide-1))
        }
        let depth=try TextureDepth(width:depthSide,height:depthSide,values:values,
            sourceLabel:"smooth low-resolution surface",interpretation:.distance)
        var settings=TextureSettings();settings.surfacePlaneRemoval=0;settings.depthCleanup=0;settings.adaptiveRelief=false
        let result=try await engine.process(source:source,settings:settings,attachedDepth:depth)
        let context=CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull(),.workingFormat:CIFormat.RGBAf])
        func pixel(_ image: CIImage,_ x: Int,_ y: Int) -> [Float] {
            var p=[Float](repeating:0,count:4)
            p.withUnsafeMutableBytes { context.render(image,toBitmap:$0.baseAddress!,rowBytes:16,
                bounds:CGRect(x:x,y:y,width:1,height:1),format:.RGBAf,colorSpace:nil) }
            return p
        }
        for x in [0,1,1022,1023] {
            XCTAssertEqual(pixel(result.height,x,512)[0],0.5,accuracy:0.005)
            XCTAssertEqual(pixel(result.normal,x,512)[0],0.5,accuracy:0.005)
        }
        XCTAssertLessThan(pixel(result.height,256,512)[0],0.15)
        XCTAssertGreaterThan(pixel(result.height,768,512)[0],0.85)
    }

    func testMaterialNormalUsesExactlyFinalDepthHeightAndFloatExport() async throws {
        let engine=TextureEngine()
        let side=64
        let photo=CIImage(color:CIColor(red:0.4,green:0.4,blue:0.4)).cropped(to:CGRect(x:0,y:0,width:side,height:side))
        let source=TextureSource(url:URL(fileURLWithPath:"/tmp/surface.png"),orientedImage:photo,
            camera:CameraMetadata(),pixelWidth:side,pixelHeight:side)
        let raw=(0..<side*side).map { Float(2)+Float($0%side)*0.005+Float($0/side)*0.01 }
        let depth=try TextureDepth(width:side,height:side,values:raw,sourceLabel:"distance ramp",interpretation:.distance)
        var settings=TextureSettings();settings.surfacePlaneRemoval=0;settings.depthCleanup=0
        let result=try await engine.process(source:source,settings:settings,attachedDepth:depth)
        let context=CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull(),.workingFormat:CIFormat.RGBAf])
        func pixel(_ map: CIImage,_ x: Int,_ y: Int) -> [Float] {
            var sample=[Float](repeating:0,count:4)
            sample.withUnsafeMutableBytes { context.render(map,toBitmap:$0.baseAddress!,rowBytes:16,
                bounds:CGRect(x:x,y:y,width:1,height:1),format:.RGBAf,colorSpace:nil) }
            return sample
        }
        XCTAssertGreaterThan(pixel(result.height,512,768)[0],pixel(result.height,512,256)[0])
        let slope=Float(settings.displacementScaleMeters/settings.materialWidthMeters)*512
        let dx = -(pixel(result.height,513,512)[0]-pixel(result.height,511,512)[0])*slope
        let dy = -(pixel(result.height,512,513)[0]-pixel(result.height,512,511)[0])*slope
        let magnitude=sqrt(dx*dx+dy*dy+1)
        let actual=pixel(result.normal,512,512)
        for (sample,expected) in zip(actual,[dx/magnitude*0.5+0.5,dy/magnitude*0.5+0.5,1/magnitude*0.5+0.5,1]) {
            XCTAssertEqual(sample,expected,accuracy:1e-6)
        }
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:folder) }
        _ = try await engine.export(result,to:folder,precision:.float32)
        let restored=try XCTUnwrap(CIImage(contentsOf:folder.appendingPathComponent("displacement.exr"),options:[.colorSpace:NSNull()]))
        XCTAssertEqual(pixel(restored,512,768)[0].bitPattern,pixel(result.height,512,768)[0].bitPattern)
        XCTAssertEqual(pixel(restored,512,256)[0].bitPattern,pixel(result.height,512,256)[0].bitPattern)
    }

    func testFloatEXRWritesTrue32BitChannelsAndPreservesSamplesAndOrientation() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:folder) }
        let values:[Float] = [0.12345679,0.23456790,0.34567891,1, 0.45678912,0.56789123,0.67891234,1,
                             0.78912345,0.89123456,0.91234567,1, 1.23456789,-0.12345679,2.34567890,1]
        let image = CIImage(bitmapData:values.withUnsafeBytes { Data($0) },bytesPerRow:32,
                            size:CGSize(width:2,height:2),format:.RGBAf,colorSpace:nil)
        let context = CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull(),.workingFormat:CIFormat.RGBAf])
        let url = folder.appendingPathComponent("float32.exr")
        try FloatEXRWriter.write(image,to:url,context:context,color:true)
        try FloatEXRWriter.verifyChannelPrecision(at:url,expected:.float32)
        XCTAssertEqual(try FloatEXRWriter.channelTypes(Data(contentsOf:url)),[2,2,2])
        let restored = try XCTUnwrap(CIImage(contentsOf:url,options:[.colorSpace:NSNull()]))
        var samples = [Float](repeating:0,count:16)
        samples.withUnsafeMutableBytes { context.render(restored,toBitmap:$0.baseAddress!,rowBytes:32,
                                                        bounds:restored.extent,format:.RGBAf,colorSpace:nil) }
        for index in values.indices { XCTAssertEqual(samples[index].bitPattern,values[index].bitPattern,"sample \(index)") }
        let grayURL = folder.appendingPathComponent("gray.exr")
        try FloatEXRWriter.write(image,to:grayURL,context:context,color:false)
        XCTAssertEqual(try FloatEXRWriter.channelTypes(Data(contentsOf:grayURL)),[2])
    }

    func testAppleHalfEXRAndFullEngineFlatMaterial() async throws {
        let library = try XCTUnwrap(Bundle.main.url(forResource:"TextureKernels",withExtension:"metallib"))
        let engine = TextureEngine(libraryURL:library)
        let image = CIImage(color:CIColor(red:0.4,green:0.4,blue:0.4)).cropped(to:CGRect(x:0,y:0,width:1200,height:1000))
        let source = TextureSource(url:URL(fileURLWithPath:"/tmp/synthetic-source.png"),orientedImage:image,
                                  camera:CameraMetadata(),pixelWidth:1200,pixelHeight:1000)
        let settings = TextureSettings()
        let material = try await engine.process(source:source,settings:settings)
        let numeric = CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull(),.workingFormat:CIFormat.RGBAf])
        for (map,expected) in [(material.height,[Float(0.5),0.5,0.5,1]),
                               (material.normal,[Float(0.5),0.5,1,1]),
                               (material.roughness,[settings.roughnessBase,settings.roughnessBase,settings.roughnessBase,1])] {
            var sample = [Float](repeating:0,count:4)
            sample.withUnsafeMutableBytes { numeric.render(map,toBitmap:$0.baseAddress!,rowBytes:16,
                bounds:CGRect(x:512,y:512,width:1,height:1),format:.RGBAf,colorSpace:nil) }
            for index in 0..<4 { XCTAssertEqual(sample[index],expected[index],accuracy:1e-4) }
        }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:folder) }
        let outputs = try await engine.export(material,to:folder,precision:.float16)
        XCTAssertEqual(outputs.count,6)
        try FloatEXRWriter.verifyChannelPrecision(at:folder.appendingPathComponent("normal.exr"),expected:.float16)
        XCTAssertThrowsError(try FloatEXRWriter.verifyChannelPrecision(at:folder.appendingPathComponent("normal.exr"),expected:.float32))
        let png = try XCTUnwrap(CGImageSourceCreateWithURL(folder.appendingPathComponent("diffuse.png") as CFURL,nil))
        let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(png,0,nil) as? [String:Any])
        XCTAssertEqual(properties[kCGImagePropertyDepth as String] as? Int,16)
    }

    private func assertValid(_ crop: CGRect, in polygon: [CGPoint], inset: Double) throws {
        let planes = try TextureGeometry.halfPlanes(polygon,inset:inset)
        for point in [CGPoint(x:crop.minX,y:crop.minY),CGPoint(x:crop.maxX,y:crop.minY),
                      CGPoint(x:crop.maxX,y:crop.maxY),CGPoint(x:crop.minX,y:crop.maxY)] {
            for plane in planes { XCTAssertLessThanOrEqual(plane.x*point.x+plane.y*point.y,plane.limit+1e-7) }
        }
    }
}
