import CoreImage
import CoreML
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class TextureLoomCompactContractTests: XCTestCase {
    func testConservativeMemoryAdmissionUsesGridWidthAndLiveBudget() {
        let gib = UInt64(1 << 30)
        let budget = TextureLoomPackage.compactMemoryBudget(physical: 64 * gib,
            recommendedMetal: 48 * gib, allocatedMetal: 0, availableMemory: 64 * gib)
        let oneK = TextureLoomPackage.compactMemoryForecast(width: 1024, height: 1024, featureWidth: 32, map: "depth")
        let twoK = TextureLoomPackage.compactMemoryForecast(width: 2048, height: 2048, featureWidth: 32, map: "depth")
        let fourK = TextureLoomPackage.compactMemoryForecast(width: 4096, height: 4096, featureWidth: 32, map: "depth")
        XCTAssertEqual(oneK, 8 * gib + 999_937 * 12)
        XCTAssertEqual(twoK, 32 * gib + 999_937 * 12)
        XCTAssertLessThan(oneK, budget); XCTAssertLessThan(twoK, budget)
        XCTAssertGreaterThan(fourK, budget)
        let busy = TextureLoomPackage.compactMemoryBudget(physical: 64 * gib,
            recommendedMetal: 48 * gib, allocatedMetal: 0, availableMemory: 24 * gib)
        XCTAssertEqual(busy, 18 * gib)
        XCTAssertGreaterThan(twoK, busy)
        XCTAssertEqual(TextureLoomPackage.compactMemoryForecast(width: Int.max, height: Int.max,
            featureWidth: 32, map: "depth"), .max)
    }

    func testCompactScalarAndNormalImagesKeepFiniteRawFloat32Values() throws {
        let width = 32, height = 40, plane = width * height
        for map in ["depth", "normal"] {
            let channels = map == "normal" ? 3 : 1
            let tensor = try MLMultiArray(shape: [1, NSNumber(value: channels), NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
            let values = tensor.dataPointer.assumingMemoryBound(to: Float.self)
            let scalar: [Float] = [-0.25, 1.25, 0.5, Float(0.5).nextUp]
            for channel in 0..<channels {
                for pixel in 0..<plane { values[channel * plane + pixel] = scalar[(pixel + channel) % scalar.count] }
            }
            let image = try TextureLoomService.materialImage(from: tensor, map: map,
                architecture: TextureLoomPackage.compactArchitecture, width: width, height: height)
            let output = render(image, channels: channels)
            let components = channels == 3 ? 4 : 1
            for pixel in 0..<plane {
                for channel in 0..<channels {
                    XCTAssertEqual(output[pixel * components + channel].bitPattern, values[channel * plane + pixel].bitPattern,
                        "Raw compact values cannot be clipped, normalized or rounded to Float16")
                }
            }
        }
    }

    func testExistingArchitectureKeepsItsBoundedOutputInterpretation() throws {
        let width = 8, height = 4
        let tensor = try MLMultiArray(shape: [1, 1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let values = tensor.dataPointer.assumingMemoryBound(to: Float.self)
        for pixel in 0..<(width * height) { values[pixel] = pixel.isMultiple(of: 2) ? -0.25 : 1.25 }
        let image = try TextureLoomService.materialImage(from: tensor, map: "depth",
            architecture: "loom-factory-material-residual-v1", width: width, height: height)
        let output = render(image, channels: 1)
        for pixel in output.indices { XCTAssertEqual(output[pixel], pixel.isMultiple(of: 2) ? 0 : 1) }
    }

    func testCompactGridAndNonfiniteOutputFailBeforeMaterialImageCreation() throws {
        for grid in [(31, 32), (32, 31), (33, 40), (40, 33)] {
            XCTAssertThrowsError(try TextureLoomPackage.validateGrid(architecture: TextureLoomPackage.compactArchitecture,
                width: grid.0, height: grid.1))
        }
        XCTAssertNoThrow(try TextureLoomPackage.validateGrid(architecture: TextureLoomPackage.compactArchitecture, width: 32, height: 40))
        let tensor = try MLMultiArray(shape: [1, 1, 32, 32], dataType: .float32)
        tensor.dataPointer.assumingMemoryBound(to: Float.self).initialize(repeating: 0.5, count: tensor.count)
        tensor[0] = NSNumber(value: Float.nan)
        XCTAssertThrowsError(try TextureLoomService.materialImage(from: tensor, map: "depth",
            architecture: TextureLoomPackage.compactArchitecture, width: 32, height: 32))
    }

    func testCompactDescriptorRejectsMissingOrChangedInferenceContracts() throws {
        let descriptor = compactDescriptor()
        XCTAssertNoThrow(try TextureLoomPackage.validateCompactDescriptor(descriptor, map: "depth"))
        for (key, replacement) in [("head_activation", "sigmoid"), ("padding", "zero"),
                                   ("normal_output_transform", "normalize"), ("output", "float32 NCHW [0,1]")] {
            var changed = descriptor
            var configuration = changed["architecture_config"] as! [String: Any]
            configuration[key] = replacement; changed["architecture_config"] = configuration
            XCTAssertThrowsError(try TextureLoomPackage.validateCompactDescriptor(changed, map: "depth"), key)
        }
        var changed = descriptor; changed["training"] = ["scalar_loss": "absolute-v1"]
        XCTAssertThrowsError(try TextureLoomPackage.validateCompactDescriptor(changed, map: "depth"))
        changed = descriptor; changed["image_size"] = 33
        XCTAssertThrowsError(try TextureLoomPackage.validateCompactDescriptor(changed, map: "depth"))
    }

    func testPackageRequiresTheRegisteredCompactDescriptor() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("studio-compact-contract-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var contents = ["classifier.mlmodelc/fixture": Data("classifier contract fixture".utf8),
            "model/depth.mlmodelc/fixture": Data("material contract fixture".utf8),
            "model/weights.safetensors": Data("weights contract fixture".utf8)]
        func save(includeDescriptor: Bool) throws {
            if includeDescriptor { contents["model/model.json"] = try JSONSerialization.data(withJSONObject: compactDescriptor()) }
            for (name, bytes) in contents {
                let url = root.appendingPathComponent(name)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try bytes.write(to: url)
            }
            let manifest: [String: Any] = ["schema": "texture-loom.package.v1", "id": UUID().uuidString,
                "name": "Compact contract fixture", "version": "1",
                "classifier": ["format": "coreml", "path": "classifier.mlmodelc", "architecture": "fixture",
                    "labels": [["id": "bricks", "name": "Bricks"]]],
                "categories": [["id": "bricks", "name": "Bricks", "version": "1", "models": [[
                    "map": "depth", "format": "coreml", "path": "model/depth.mlmodelc",
                    "weightsPath": "model/weights.safetensors", "architecture": TextureLoomPackage.compactArchitecture]]]],
                "files": contents.map { ["path": $0.key, "sha256": TextureLoomPackage.hash($0.value)] }]
            try JSONSerialization.data(withJSONObject: manifest).write(to: root.appendingPathComponent("manifest.json"))
        }
        try save(includeDescriptor: false)
        XCTAssertThrowsError(try TextureLoomPackage(root: root))
        try save(includeDescriptor: true)
        XCTAssertNoThrow(try TextureLoomPackage(root: root))
    }

    private func compactDescriptor() -> [String: Any] {
        ["schema": "loom-factory-model-v1", "status": "completed", "kind": "map", "map": "depth",
         "architecture": TextureLoomPackage.compactArchitecture, "image_size": 32,
         "training": ["scalar_loss": "linear-detail-v1"],
         "architecture_config": ["id": TextureLoomPackage.compactArchitecture, "width": 32, "blocks": 1,
            "input": "RGB sRGB float32 NCHW [0,1]", "output": "float32 NCHW raw linear values",
            "input_channels": 3, "output_channels": 1, "normal_convention": NSNull(), "head_activation": "linear",
            "padding": "reflect-v1", "scale_channels": [32, 64, 128, 256], "blocks_per_stage": 1,
            "stage_count": 7, "upsampling": "nearest-2x-exact-repeat", "grid_multiple": 8,
            "minimum_grid_size": 32, "normal_output_transform": "none", "lora_scope": "dense-convolutions-only"]]
    }

    private func render(_ image: CIImage, channels: Int) -> [Float] {
        let width = Int(image.extent.width), height = Int(image.extent.height), components = channels == 3 ? 4 : 1
        let context = CIContext(options: [.useSoftwareRenderer: true, .workingFormat: CIFormat.RGBAf,
            .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        var values = [Float](repeating: 0, count: width * height * components)
        values.withUnsafeMutableBytes {
            context.render(image, toBitmap: $0.baseAddress!, rowBytes: width * components * 4,
                bounds: image.extent, format: channels == 3 ? .RGBAf : .Rf, colorSpace: nil)
        }
        return values
    }
}
