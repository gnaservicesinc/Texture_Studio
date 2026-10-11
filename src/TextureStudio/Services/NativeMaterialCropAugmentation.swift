import Foundation

/// Mutable run-owned crop storage. Original and prepared dataset files remain
/// immutable; an epoch flip reads the preceding cached crop, never its source.
final class NativeMaterialCropAugmentation {
    typealias Sample = NativeMaterialDatasetService.TrainingSample
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
                var degrees = spin && sample.cropKind == "center" ? Int(random.next() % 179) + 1 : 0
                var insufficientSupport = false
                if degrees > 0 {
                    guard let inputSource = sample.augmentationSourceInputURL, sample.augmentationSourceInputSHA256 != nil,
                          let targetSource = sample.augmentationSourceTargetURL, sample.augmentationSourceTargetSHA256 != nil else {
                        throw StudioError("Center spinning needs both original diffuse and target source bindings.")
                    }
                    let inputHeader = try NativePNG.inspect(inputSource), targetHeader = try NativePNG.inspect(targetSource)
                    let bound = Self.spinBound(size: sample.width, degrees: degrees)
                    if [inputHeader.width, inputHeader.height, targetHeader.width, targetHeader.height].contains(where: { $0 < bound }) {
                        // Near-native fields can be center-cut by one row but
                        // cannot supply a rotated crop without new border data.
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
                input = try Self.spunCenter(source: source, expectedSHA256: sha, size: sample.width, degrees: degrees)
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
                        degrees: degrees, normal: target == "normal", normalConvention: sample.augmentationSourceTargetConvention ?? sample.targetConvention)
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
                "horizontal_flip": horizontal, "vertical_flip": vertical,
                "input_sha256": inputSHA, "target_sha256": paired.1,
                "center_spin_skipped_insufficient_support": insufficientSupport,
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

    /// Sample the rotated full source's center without decoding its entire grid.
    /// Nearest integer sampling keeps scalar target codes and color precision.
    static func spunCenter(source: URL, expectedSHA256: String, size: Int, degrees: Int,
                           normal: Bool = false, normalConvention: String = "opengl") throws -> NativePNG {
        let bytes = try Data(contentsOf: source, options: .mappedIfSafe)
        guard NativeMaterialTrainer.checksum(bytes) == expectedSHA256, (1...179).contains(degrees) else {
            throw StudioError("The center-spin source or angle is invalid.")
        }
        let header = try NativePNG.inspect(source)
        let angle = Double(degrees) * .pi / 180, c = cos(angle), s = sin(angle)
        let bound = spinBound(size: size, degrees: degrees)
        guard bound <= header.width, bound <= header.height else { throw StudioError("The original source cannot cover this center spin without padding.") }
        let left = (header.width - bound) / 2, top = (header.height - bound) / 2
        let selected = try NativePNG.crop(bytes, rectangle: [left, top, bound, bound],
            flipGreen: normal && normalConvention == "directx")
        let h = selected.header, bpp = h.bytesPerPixel, component = h.bits / 8
        guard !normal || h.channels >= 3 else { throw StudioError("Normal augmentation needs RGB channels.") }
        let maximum = h.bits == 16 ? 65535 : 255
        var pixels = Data(count: size * size * bpp)
        try selected.pixels.withUnsafeBytes { original in
            try pixels.withUnsafeMutableBytes { transformed in
                let input = original.bindMemory(to: UInt8.self).baseAddress!, output = transformed.bindMemory(to: UInt8.self).baseAddress!
                func code(_ p: UnsafePointer<UInt8>, _ channel: Int) -> Int {
                    let offset = channel * component
                    return component == 1 ? Int(p[offset]) : Int(p[offset]) * 256 + Int(p[offset + 1])
                }
                func put(_ offset: Int, _ value: Double) {
                    let integer = min(maximum, max(0, Int(((value + 1) * 0.5 * Double(maximum)).rounded())))
                    if component == 1 { output[offset] = UInt8(integer) }
                    else { output[offset] = UInt8(integer >> 8); output[offset + 1] = UInt8(integer & 255) }
                }
                for y in 0..<size {
                    try Task.checkCancellation()
                    for x in 0..<size {
                        let dx = Double(x) + 0.5 - Double(size) / 2, dy = Double(y) + 0.5 - Double(size) / 2
                        let sx = Int((c * dx + s * dy + Double(header.width) / 2 - 0.5).rounded()) - left
                        let sy = Int((-s * dx + c * dy + Double(header.height) / 2 - 0.5).rounded()) - top
                        guard sx >= 0, sy >= 0, sx < bound, sy < bound else { throw StudioError("Center spin escaped its original pixel support.") }
                        let from = input + (sy * bound + sx) * bpp, to = (y * size + x) * bpp
                        (output + to).update(from: from, count: bpp)
                        if normal {
                            let nx = Double(code(from, 0)) / Double(maximum) * 2 - 1
                            let ny = Double(code(from, 1)) / Double(maximum) * 2 - 1
                            put(to, c * nx + s * ny); put(to + component, -s * nx + c * ny)
                        }
                    }
                }
            }
        }
        guard NativeMaterialTrainer.checksum(try Data(contentsOf: source, options: .mappedIfSafe)) == expectedSHA256 else {
            throw StudioError("Original source changed during center spin.")
        }
        return NativePNG(header: .init(width: size, height: size, bits: h.bits, channels: h.channels, color: h.color, interlace: 0),
            pixels: pixels, colorChunks: selected.colorChunks)
    }

    private static func spinBound(size: Int, degrees: Int) -> Int {
        let angle = Double(degrees) * .pi / 180
        return Int(ceil(Double(size) * (abs(cos(angle)) + abs(sin(angle))))) + 2
    }
}
