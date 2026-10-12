import Foundation

/// Mutable run-owned crop storage. Original and prepared dataset files remain
/// immutable; an epoch flip reads the preceding cached crop, never its source.
final class NativeMaterialCropAugmentation {
    typealias Sample = NativeMaterialDatasetService.TrainingSample
    static let centerSpinSampling = "exact-quarter-turn-pixel-permutation-v1"
    let directory: URL
    let target: String
    private var random: NativeMaterialRandom
    private(set) var history: [[String: Any]] = []
    private var activeEpoch: Int?
    private var decisions: [String: (Int, Bool, Bool, Bool)] = [:]
    private var targets: [String: (URL, String, String)] = [:]

    init(directory: URL, target: String, seed: UInt64) {
        self.directory = directory; self.target = target
        random = .init(seed: seed ^ 0x63726f70732d7632)
    }

    func refresh(_ samples: [Sample], epoch: Int, flip: Bool, spin: Bool,
                 depthProCache: NativeDepthProInputCache?, check: () throws -> Void,
                 onProgress: (_ completed: Int, _ total: Int, _ sampleID: String) throws -> Void = { _, _, _ in },
                 onOperation: (_ operation: String, _ sampleID: String) throws -> Void = { _, _ in }) throws -> [Sample] {
        try check()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if activeEpoch != epoch {
            activeEpoch = epoch
            decisions.removeAll(keepingCapacity: true)
            targets.removeAll(keepingCapacity: true)
        }
        var refreshed: [Sample] = []
        for sample in samples {
            try check()
            try onOperation("Reading cached crop", sample.id)
            // All color variants share target geometry and orientation.
            let geometry = String(sample.id.split(separator: "|", maxSplits: 1)[0])
            if decisions[geometry] == nil {
                var degrees = spin && sample.cropKind == "center" ? (Int(random.next() % 3) + 1) * 90 : 0
                var insufficientSupport = false
                if degrees > 0 {
                    guard let inputSource = sample.augmentationSourceInputURL, sample.augmentationSourceInputSHA256 != nil,
                          let targetSource = sample.augmentationSourceTargetURL, sample.augmentationSourceTargetSHA256 != nil else {
                        throw StudioError("Center spinning needs both original diffuse and target source bindings.")
                    }
                    let inputHeader = try NativePNG.inspect(inputSource), targetHeader = try NativePNG.inspect(targetSource)
                    if [inputHeader.width, inputHeader.height, targetHeader.width, targetHeader.height].contains(where: { $0 < sample.width }) {
                        // Quarter turns need only the exact center crop, with
                        // no additional support, border pixels or interpolation.
                        degrees = 0; insufficientSupport = true
                    }
                }
                let choice = flip ? random.next() % 4 : 0
                decisions[geometry] = (degrees, choice & 1 != 0, choice & 2 != 0, insufficientSupport)
            }
            let (degrees, horizontal, vertical, insufficientSupport) = decisions[geometry]!
            let stem = NativeMaterialTrainer.checksum(Data(sample.id.utf8))
            let inputURL = directory.appendingPathComponent(stem + "-diffuse.png")
            let targetStem = NativeMaterialTrainer.checksum(Data(geometry.utf8))
            let targetURL = directory.appendingPathComponent(targetStem + "-" + target + ".png")
            if degrees > 0 {
                guard sample.augmentationSourceInputURL != nil, sample.augmentationSourceInputSHA256 != nil,
                      sample.augmentationSourceTargetURL != nil, sample.augmentationSourceTargetSHA256 != nil else {
                    throw StudioError("Center spinning needs both original diffuse and target source bindings.")
                }
            }
            var input: NativePNG?
            var beforeBytes: Data?
            var beforeSHA = sample.inputSHA256
            var convention = sample.targetConvention
            if degrees > 0, let source = sample.augmentationSourceInputURL,
               let sha = sample.augmentationSourceInputSHA256 {
                try onOperation("Spinning center crop", sample.id)
                input = try Self.spunCenter(source: source, expectedSHA256: sha, size: sample.width, degrees: degrees,
                    rectangle: sample.cropRectangle)
            } else {
                // An unchanged crop keeps its exact compressed bytes and cache
                // identity. Decode only when a pixel permutation is requested.
                beforeBytes = try readBytes(sample.inputURL, expected: sample.inputSHA256)
            }
            if let cache = depthProCache, flip {
                // Predict a newly spun field before orientation-only permutations.
                let beforeURL: URL
                if let input {
                    beforeBytes = try input.encoded()
                    beforeSHA = NativeMaterialTrainer.checksum(beforeBytes!)
                    beforeURL = directory.appendingPathComponent(stem + "-before-flip.png")
                    try beforeBytes!.write(to: beforeURL, options: .atomic)
                } else { beforeURL = sample.inputURL }
                defer { if beforeURL != sample.inputURL { try? FileManager.default.removeItem(at: beforeURL) } }
                try onOperation("Caching DepthPro input for crop", sample.id)
                _ = try cache.input(imageURL: beforeURL, expectedSHA256: beforeSHA, width: sample.width,
                    height: sample.height, checkCancellation: check)
            }
            try check()
            try onOperation("Applying paired crop transform", sample.id)
            let inputBytes: Data
            if horizontal || vertical {
                let decoded = try input ?? NativePNG.decode(beforeBytes!)
                inputBytes = try Self.flipped(decoded, horizontal: horizontal, vertical: vertical).encoded()
            } else { inputBytes = try beforeBytes ?? input!.encoded() }
            let inputSHA = NativeMaterialTrainer.checksum(inputBytes)
            if let cache = depthProCache, flip {
                _ = try cache.remapFlips(fromSourceSHA256: beforeSHA, toSourceSHA256: inputSHA,
                    width: sample.width, height: sample.height, horizontal: horizontal, vertical: vertical,
                    checkCancellation: check)
            }
            if targets[geometry] == nil {
                var output: NativePNG?
                var originalBytes: Data?
                if degrees > 0, let source = sample.augmentationSourceTargetURL,
                   let sha = sample.augmentationSourceTargetSHA256 {
                    output = try Self.spunCenter(source: source, expectedSHA256: sha, size: sample.width,
                        degrees: degrees, normal: target == "normal", normalConvention: sample.augmentationSourceTargetConvention ?? sample.targetConvention,
                        rectangle: sample.cropRectangle)
                    if target == "normal" { convention = "opengl" }
                } else { originalBytes = try readBytes(sample.targetURL, expected: sample.targetSHA256) }
                let bytes: Data
                if horizontal || vertical {
                    let decoded = try output ?? NativePNG.decode(originalBytes!)
                    bytes = try Self.flipped(decoded, horizontal: horizontal, vertical: vertical, normal: target == "normal").encoded()
                } else { bytes = try originalBytes ?? output!.encoded() }
                try check()
                try onOperation("Saving cached crop", sample.id)
                if targetURL != sample.targetURL || bytes != originalBytes { try bytes.write(to: targetURL, options: .atomic) }
                targets[geometry] = (targetURL, NativeMaterialTrainer.checksum(bytes), convention)
            }
            try check()
            try onOperation("Saving cached crop", sample.id)
            if inputURL != sample.inputURL || degrees > 0 || inputBytes != beforeBytes { try inputBytes.write(to: inputURL, options: .atomic) }
            let paired = targets[geometry]!
            var changed = sample
            changed.inputURL = inputURL; changed.inputSHA256 = inputSHA
            changed.targetURL = paired.0; changed.targetSHA256 = paired.1; changed.targetConvention = paired.2
            refreshed.append(changed)
            let receipt: [String: Any] = ["sample_id": sample.id, "epoch": epoch, "source_center_spin_degrees": degrees,
                "center_spin_sampling": degrees > 0 ? Self.centerSpinSampling : "exact-crop-or-flip",
                "horizontal_flip": horizontal, "vertical_flip": vertical,
                "input_sha256": inputSHA, "target_sha256": paired.1,
                "center_spin_skipped_insufficient_support": insufficientSupport,
                "source_pixels_resized": false, "source_pixels_interpolated": false,
                "progressive_cached_flip": degrees == 0, "original_sources_modified": false]
            try appendHistory(receipt)
            history.append(receipt)
            try onProgress(refreshed.count, samples.count, sample.id)
        }
        return refreshed
    }

    /// Append bounded records during an update; serialize the full history only
    /// at checkpoint/return boundaries, rather than once per requested crop.
    private func appendHistory(_ receipt: [String: Any]) throws {
        let url = directory.appendingPathComponent("crop-history.jsonl")
        if !FileManager.default.fileExists(atPath: url.path) {
            guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                throw StudioError("Crop history could not be created.")
            }
        }
        let file = try FileHandle(forWritingTo: url)
        defer { try? file.close() }
        try file.seekToEnd()
        var record = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
        record.append(10)
        try file.write(contentsOf: record)
    }

    func saveHistory() throws {
        guard !history.isEmpty else { return }
        try JSONSerialization.data(withJSONObject: ["schema": "texture-studio-progressive-crops-v1", "history": history],
            options: [.prettyPrinted, .sortedKeys]).write(to: directory.appendingPathComponent("crop-history.json"), options: .atomic)
    }

    private func readBytes(_ url: URL, expected: String) throws -> Data {
        let bytes = try Data(contentsOf: url)
        guard NativeMaterialTrainer.checksum(bytes) == expected else { throw StudioError("A crop changed before augmentation.") }
        return bytes
    }

    /// Integer pixel permutations preserve every scalar/color code exactly.
    /// Tangent normal X/Y signs follow the reflected image axes.
    static func flipped(_ image: NativePNG, horizontal: Bool, vertical: Bool, normal: Bool = false) throws -> NativePNG {
        guard horizontal || vertical else { return image }
        let h = image.header, bpp = h.bytesPerPixel, component = h.bits / 8
        guard !normal || h.channels >= 3 else { throw StudioError("Normal augmentation needs RGB channels.") }
        var pixels = Data(count: image.pixels.count)
        try image.pixels.withUnsafeBytes { original in
            try pixels.withUnsafeMutableBytes { transformed in
                let source = original.bindMemory(to: UInt8.self).baseAddress!, output = transformed.bindMemory(to: UInt8.self).baseAddress!
                for y in 0..<h.height {
                    try Task.checkCancellation()
                    for x in 0..<h.width {
                        let from = ((vertical ? h.height - y - 1 : y) * h.width + (horizontal ? h.width - x - 1 : x)) * bpp
                        let to = (y * h.width + x) * bpp
                        (output + to).update(from: source + from, count: bpp)
                        if normal {
                            if horizontal { for byte in 0..<component { output[to + byte] = 255 - output[to + byte] } }
                            if vertical { for byte in component..<(component * 2) { output[to + byte] = 255 - output[to + byte] } }
                        }
                    }
                }
            }
        }
        return NativePNG(header: h, pixels: pixels, colorChunks: image.colorChunks)
    }

    /// Take the recorded integer-aligned crop, then permute its pixels. Older
    /// samples without a rectangle retain their original center behavior.
    /// Restrict rotation to quarter turns so diffuse and scalar targets retain
    /// every original code; normal X/Y are only swapped and/or complemented.
    static func spunCenter(source: URL, expectedSHA256: String, size: Int, degrees: Int,
                           normal: Bool = false, normalConvention: String = "opengl", rectangle: [Int]? = nil) throws -> NativePNG {
        let bytes = try Data(contentsOf: source, options: .mappedIfSafe)
        guard NativeMaterialTrainer.checksum(bytes) == expectedSHA256, size > 0,
              [90, 180, 270].contains(degrees) else {
            throw StudioError("Center crops rotate only by 90, 180 or 270 degrees to preserve exact source pixels.")
        }
        let header = try NativePNG.inspect(source)
        guard size <= header.width, size <= header.height else {
            throw StudioError("The original source cannot cover this center crop without resizing or padding.")
        }
        let selectedRectangle = rectangle ?? [(header.width - size) / 2, (header.height - size) / 2, size, size]
        guard selectedRectangle.count == 4, selectedRectangle[2] == size, selectedRectangle[3] == size else {
            throw StudioError("The recorded center crop must match the selected native training grid.")
        }
        let selected = try NativePNG.crop(bytes,
            rectangle: selectedRectangle,
            flipGreen: normal && normalConvention == "directx")
        let rotated = try quarterTurn(selected, degrees: degrees, normal: normal)
        guard NativeMaterialTrainer.checksum(try Data(contentsOf: source, options: .mappedIfSafe)) == expectedSHA256 else {
            throw StudioError("Original source changed during center rotation.")
        }
        return rotated
    }

    static func quarterTurn(_ image: NativePNG, degrees: Int, normal: Bool = false) throws -> NativePNG {
        let h = image.header, bpp = h.bytesPerPixel, component = h.bits / 8
        guard h.width == h.height, [90, 180, 270].contains(degrees) else {
            throw StudioError("Exact crop rotations need a square grid and a quarter turn.")
        }
        guard !normal || h.channels >= 3 else { throw StudioError("Normal augmentation needs RGB channels.") }
        var pixels = Data(count: image.pixels.count)
        try image.pixels.withUnsafeBytes { original in
            try pixels.withUnsafeMutableBytes { transformed in
                let source = original.bindMemory(to: UInt8.self).baseAddress!, output = transformed.bindMemory(to: UInt8.self).baseAddress!
                for y in 0..<h.height {
                    try Task.checkCancellation()
                    for x in 0..<h.width {
                        let sx: Int, sy: Int
                        switch degrees {
                        case 90: sx = y; sy = h.width - x - 1
                        case 180: sx = h.width - x - 1; sy = h.height - y - 1
                        default: sx = h.height - y - 1; sy = x
                        }
                        let from = (sy * h.width + sx) * bpp, to = (y * h.width + x) * bpp
                        (output + to).update(from: source + from, count: bpp)
                        if normal {
                            for byte in 0..<component {
                                let nx = source[from + byte], ny = source[from + component + byte]
                                switch degrees {
                                case 90:
                                    output[to + byte] = ny; output[to + component + byte] = 255 - nx
                                case 180:
                                    output[to + byte] = 255 - nx; output[to + component + byte] = 255 - ny
                                default:
                                    output[to + byte] = 255 - ny; output[to + component + byte] = nx
                                }
                            }
                        }
                    }
                }
            }
        }
        return NativePNG(header: h, pixels: pixels, colorChunks: image.colorChunks)
    }
}
