import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialPackageTests: XCTestCase {
    func testRanksAbove4096LoadExportInspectAndCombineWithoutGraphAllocation() throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        let baseURL = fixture.root.appendingPathComponent("base.safetensors")
        try NativeSafetensors.write(tensors: [fixture.layer + ".weight": .floats([7, 8], shape: [1, 2])],
            metadata: [:], to: baseURL)
        let model = try NativeMaterialModel.load(checkpointURL: nil, baseURL: baseURL,
            target: "height", rank: 4097, alpha: 4097, training: true)
        XCTAssertEqual(model.layers[fixture.layer]?.rank, 4097)
        XCTAssertEqual(model.adapterWeights[fixture.layer + ".lora_A"]?.shape, [4097, 2])
        XCTAssertEqual(model.adapterWeights[fixture.layer + ".lora_B"]?.shape, [1, 4097])
        let output = fixture.root.appendingPathComponent("Large Rank")
        let config = try model.checkpointConfiguration(size: 256, step: 4)
        _ = try NativeMaterialPackage.export(model: model, configuration: config, to: output, developer: false)
        let adapterURL = output.appendingPathComponent("adapter.safetensors")
        XCTAssertNoThrow(try NativeMaterialCheckpoint.inspect(at: adapterURL))
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output))
        let reloaded = try NativeMaterialModel.load(checkpointURL: adapterURL, baseURL: baseURL, target: "height")
        XCTAssertEqual(reloaded.layers[fixture.layer]?.rank, 4097)
        let snapshot = try NativeSafetensors(contentsOf: adapterURL)
        let (combinedConfig, tensors) = try NativeMaterialPackage.combine([(snapshot, 0.5), (snapshot, 0.5)])
        let specs = try JSONDecoder().decode([String: NativeMaterialModel.AdapterLayer].self,
            from: JSONSerialization.data(withJSONObject: combinedConfig["layers"]!))
        XCTAssertEqual(specs[fixture.layer]?.rank, 8194)
        XCTAssertEqual(tensors[fixture.layer + ".lora_A"]?.shape, [8194, 2])
        XCTAssertEqual(tensors[fixture.layer + ".lora_B"]?.shape, [1, 8194])
        let combined = try NativeMaterialModel(baseWeights: model.baseWeights, adapterWeights: tensors, layers: specs,
            configuration: combinedConfig, baseSHA256: model.baseSHA256)
        let combinedOutput = fixture.root.appendingPathComponent("Combined Large Rank")
        _ = try NativeMaterialPackage.export(model: combined, configuration: combinedConfig, to: combinedOutput, developer: false)
        XCTAssertNoThrow(try NativeMaterialCheckpoint.inspect(at: combinedOutput.appendingPathComponent("adapter.safetensors")))
        XCTAssertNoThrow(try NativeMaterialPackage.verify(combinedOutput))
    }

    func testCombinedRankOverflowFailsBeforeAllocatingFactors() throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        var config = fixture.configuration
        config["layers"] = [fixture.layer: ["weight_shape": [1, 2], "rank": Int.max, "alpha": 1]]
        let tensors: [String: NativeTensor] = [fixture.layer + ".lora_A": .floats([1, 2], shape: [1, 2]),
            fixture.layer + ".lora_B": .floats([3], shape: [1, 1])]
        let extreme = try NativeSafetensors(bytes: NativeSafetensors.encoded(tensors: tensors,
            metadata: ["configuration": NativeMaterialTransfer.json(config)]))
        let ordinary = try fixture.adapter()
        XCTAssertThrowsError(try NativeMaterialPackage.combine([(extreme, 1), (ordinary, 1)])) { error in
            XCTAssertTrue(error.localizedDescription.contains("rank overflows"))
        }
    }

    func testPackageCarriesAllLicensesAndPreservesUnchangedSnapshotExactly() async throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        let package = try fixture.package()
        let (_, hashes) = try NativeMaterialPackage.verify(package)
        XCTAssertEqual(hashes.count, 9)
        for license in NativeMaterialPackage.licenses {
            XCTAssertGreaterThan(try Data(contentsOf: package.appendingPathComponent("ModelLicenses/" + license)).count, 20)
        }
        // Nonstandard metadata must remain part of the selected snapshot.
        let selected = fixture.root.appendingPathComponent("selected.safetensors")
        let snapshot = try fixture.adapter(extraMetadata: ["provenance": "untouched snapshot"])
        try snapshot.bytes.write(to: selected)
        let exported = fixture.root.appendingPathComponent("Exact Export")
        _ = try await NativeMaterialPackage.run(arguments: ["--checkpoint", selected.path, "--expected-sha256", snapshot.sha256, "--output", exported.path])
        XCTAssertEqual(try Data(contentsOf: exported.appendingPathComponent("adapter.safetensors")), snapshot.bytes)
        _ = try NativeMaterialPackage.verify(exported)
    }

    func testRejectsChangedSidecarUnlistedFileMissingNoticeAndMismatchedFullPartner() throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        let package = try fixture.package()
        var config = try NativeMaterialTransfer.object(package.appendingPathComponent("config.json"))
        config["step"] = 700
        try NativeMaterialTransfer.writeJSON(config, to: package.appendingPathComponent("config.json"))
        try fixture.refreshManifest(package)
        XCTAssertThrowsError(try NativeMaterialPackage.verify(package))
        let clean = try fixture.package("Clean")
        try Data("unlisted".utf8).write(to: clean.appendingPathComponent("unexpected.bin"))
        XCTAssertThrowsError(try NativeMaterialPackage.verify(clean))
        try FileManager.default.removeItem(at: clean.appendingPathComponent("unexpected.bin"))
        try FileManager.default.removeItem(at: clean.appendingPathComponent("ModelLicenses/PBRnxt_LICENSE"))
        XCTAssertThrowsError(try NativeMaterialPackage.verify(clean))
        let full = try fixture.package("Full")
        var fullConfig = fixture.configuration
        fullConfig["schema"] = "texture-studio-material-checkpoint-v1"
        fullConfig["fused_adapter_sha256"] = String(repeating: "9", count: 64)
        let modelBytes = try NativeSafetensors.encoded(tensors: ["ups.3.model.10.weight": .floats([7, 8], shape: [1, 2])], metadata: ["configuration": NativeMaterialTransfer.json(fullConfig)])
        try modelBytes.write(to: full.appendingPathComponent("model.safetensors"))
        fullConfig["checkpoint_filename"] = "model.safetensors"; fullConfig["adapter_filename"] = "adapter.safetensors"
        fullConfig["full_checkpoint"] = true; fullConfig["optimizer_included"] = false; fullConfig["source_images_included"] = false
        try NativeMaterialTransfer.writeJSON(fullConfig, to: full.appendingPathComponent("config.json"))
        try fixture.refreshManifest(full, extra: ["model.safetensors"])
        XCTAssertThrowsError(try NativeMaterialPackage.verify(full))
    }

    func testWeightedMixtureMatchesSumOfNumericAdapterDeltasAndRejectsDifferentBase() throws {
        let fixture = try NativeMaterialPackageFixture()
        defer { fixture.remove() }
        let first = try fixture.adapter(a: [1, 2], b: [3], alpha: 2)
        let second = try fixture.adapter(a: [4, 5], b: [6], alpha: 1)
        let (configuration, tensors) = try NativeMaterialPackage.combine([(first, 0.5), (second, -0.25)])
        let a = try XCTUnwrap(tensors[fixture.layer + ".lora_A"]).floatValues()
        let b = try XCTUnwrap(tensors[fixture.layer + ".lora_B"]).floatValues()
        XCTAssertEqual(a, [1, 2, 4, 5]); XCTAssertEqual(b, [3, -1.5])
        XCTAssertEqual(b[0] * a[0] + b[1] * a[2], -3, accuracy: 1e-7)
        XCTAssertEqual(b[0] * a[1] + b[1] * a[3], -1.5, accuracy: 1e-7)
        let specs = try JSONDecoder().decode([String: NativeMaterialModel.AdapterLayer].self, from: JSONSerialization.data(withJSONObject: configuration["layers"]!))
        XCTAssertEqual(specs[fixture.layer]?.rank, 2); XCTAssertEqual(specs[fixture.layer]?.alpha, 2)
        var changed = fixture.configuration
        changed["base"] = ["sha256": String(repeating: "b", count: 64)]
        let mismatch = try fixture.adapter(configuration: changed)
        XCTAssertThrowsError(try NativeMaterialPackage.combine([(first, 1), (mismatch, 1)]))
        XCTAssertThrowsError(try NativeMaterialPackage.combine([(first, .infinity)]))
    }
}

struct NativeMaterialPackageFixture {
    let root: URL
    let layer = "ups.3.model.10"
    let base = String(repeating: "a", count: 64)
    var configuration: [String: Any] {
        ["schema": "texture-studio-material-lora-v1", "architecture": "pbrnxt-native-v1", "target": "height", "step": 4,
         "training_size": 256, "image_padding": false, "image_resizing": false, "scope": "final-map", "base": ["sha256": base],
         "layers": [layer: ["weight_shape": [1, 2], "rank": 1, "alpha": 1]]]
    }
    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("native-package-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }
    func remove() { try? FileManager.default.removeItem(at: root) }
    func adapter(a: [Float] = [1, 2], b: [Float] = [3], alpha: Float = 1,
                 configuration: [String: Any]? = nil, extraMetadata: [String: String] = [:]) throws -> NativeSafetensors {
        var config = configuration ?? self.configuration
        config["layers"] = [layer: ["weight_shape": [1, 2], "rank": 1, "alpha": alpha]]
        var metadata = extraMetadata; metadata["configuration"] = try NativeMaterialTransfer.json(config)
        return try NativeSafetensors(bytes: NativeSafetensors.encoded(tensors: [layer + ".lora_A": .floats(a, shape: [1, 2]), layer + ".lora_B": .floats(b, shape: [1, 1])], metadata: metadata))
    }
    func package(_ name: String = "Package") throws -> URL {
        let snapshot = try adapter()
        let model = try NativeMaterialModel(baseWeights: [layer + ".weight": .floats([7, 8], shape: [1, 2])],
            adapterWeights: snapshot.nativeTensors(), layers: [layer: .init(weightShape: [1, 2], rank: 1, alpha: 1)], baseSHA256: base)
        let output = root.appendingPathComponent(name)
        _ = try NativeMaterialPackage.export(model: model, configuration: configuration, to: output, developer: false)
        return output
    }
    func refreshManifest(_ package: URL, extra: [String] = []) throws {
        var hashes = try NativeMaterialTransfer.object(package.appendingPathComponent("SHA256SUMS.json"))
        for name in Set(Array(hashes.keys) + extra) { hashes[name] = try NativeMaterialTransfer.hash(package.appendingPathComponent(name)) }
        try NativeMaterialTransfer.writeJSON(hashes, to: package.appendingPathComponent("SHA256SUMS.json"))
    }
    func files(_ package: URL) throws -> [String: Data] {
        let hashes = try NativeMaterialTransfer.object(package.appendingPathComponent("SHA256SUMS.json"))
        return try Dictionary(uniqueKeysWithValues: (Array(hashes.keys) + ["SHA256SUMS.json"]).map { ($0, try Data(contentsOf: package.appendingPathComponent($0))) })
    }
}
