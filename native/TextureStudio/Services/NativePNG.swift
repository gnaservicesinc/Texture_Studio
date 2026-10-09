import Foundation
import zlib

/// Integer PNG storage. This decoder never enters a color-managed image
/// pipeline: filters, compression and Adam7 are reversed directly on bytes.
struct NativePNG: Sendable {
    struct Header: Equatable, Sendable {
        let width: Int
        let height: Int
        let bits: Int
        let channels: Int
        let color: UInt8
        let interlace: UInt8
        var bytesPerPixel: Int { channels * bits / 8 }
    }
    let header: Header
    var pixels: Data
    let colorChunks: [(String, Data)]
    private static let signature = Data([137, 80, 78, 71, 13, 10, 26, 10])
    private static let colorTypes: [UInt8: Int] = [0: 1, 2: 3, 4: 2, 6: 4]
    private static let retainedChunks: Set<String> = ["cHRM", "gAMA", "iCCP", "sRGB", "sBIT", "cICP", "mDCV", "cLLI"]

    static func inspect(_ url: URL) throws -> Header {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        return try parseHeader(file.read(upToCount: 33) ?? Data())
    }
    private static func parseHeader(_ data: Data) throws -> Header {
        guard data.count >= 33, data.prefix(8) == signature, integer(data, 8) == 13,
              data.subdata(in: 12..<16) == Data("IHDR".utf8),
              checksum(data.subdata(in: 12..<29)) == integer(data, 29) else { throw invalid("Invalid PNG IHDR or checksum.") }
        let width = Int(integer(data, 16)), height = Int(integer(data, 20)), bits = Int(data[24]), color = data[25]
        guard width > 0, height > 0, width <= 65536, height <= 65536,
              width * height <= 150_000_000, [8, 16].contains(bits), let channels = colorTypes[color],
              data[26] == 0, data[27] == 0, data[28] <= 1 else { throw invalid("Unsupported native PNG storage.") }
        return Header(width: width, height: height, bits: bits, channels: channels, color: color, interlace: data[28])
    }
    static func decode(_ data: Data) throws -> NativePNG {
        let header = try parseHeader(data)
        var cursor = 8, compressed = Data(), colors: [(String, Data)] = [], ended = false
        while cursor + 12 <= data.count {
            let length = Int(integer(data, cursor))
            guard length <= data.count - cursor - 12 else { throw invalid("Truncated PNG chunk.") }
            let kind = String(decoding: data[cursor + 4..<cursor + 8], as: UTF8.self)
            let payload = data.subdata(in: cursor + 8..<cursor + 8 + length)
            guard checksum(data.subdata(in: cursor + 4..<cursor + 8 + length)) == integer(data, cursor + 8 + length) else {
                throw invalid("PNG chunk checksum changed.")
            }
            if kind == "IHDR", cursor != 8 { throw invalid("PNG contains more than one image header.") }
            if kind == "tRNS" { throw invalid("PNG transparency tables need an explicit native channel conversion before training.") }
            if kind == "IDAT" { compressed.append(payload) }
            else if retainedChunks.contains(kind) { colors.append((kind, payload)) }
            else if kind == "IEND" { ended = true; cursor += length + 12; break }
            else if kind != "IHDR", let first = kind.utf8.first, first & 32 == 0 { throw invalid("Unsupported critical PNG chunk.") }
            cursor += length + 12
        }
        guard ended, cursor == data.count, !compressed.isEmpty else { throw invalid("PNG image is incomplete.") }
        let passes = header.interlace == 0 ? [(0, 0, 1, 1)] : [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
        let bytesPerPixel = header.bytesPerPixel
        let layouts = passes.map { x, y, dx, dy in
            (x, y, dx, dy, max(0, (header.width - x + dx - 1) / dx), max(0, (header.height - y + dy - 1) / dy))
        }
        let expected = layouts.reduce(0) { $0 + ($1.4 > 0 && $1.5 > 0 ? ($1.4 * bytesPerPixel + 1) * $1.5 : 0) }
        var filtered = Data(count: expected), actual = uLongf(expected)
        let status = filtered.withUnsafeMutableBytes { destination in
            compressed.withUnsafeBytes { source in
                uncompress(destination.bindMemory(to: Bytef.self).baseAddress!, &actual,
                    source.bindMemory(to: Bytef.self).baseAddress!, uLong(compressed.count))
            }
        }
        guard status == Z_OK, actual == expected else { throw invalid("PNG pixel stream has an invalid size.") }
        var pixels = Data(count: header.width * header.height * bytesPerPixel), offset = 0
        for (x, y, dx, dy, width, height) in layouts where width > 0 && height > 0 {
            let rowBytes = width * bytesPerPixel
            var previous = [UInt8](repeating: 0, count: rowBytes)
            for rowIndex in 0..<height {
                try Task.checkCancellation()
                let filter = filtered[offset]; offset += 1
                guard filter <= 4 else { throw invalid("Unknown PNG row filter.") }
                var row = Array(filtered[offset..<offset + rowBytes]); offset += rowBytes
                for column in 0..<rowBytes {
                    let left = column >= bytesPerPixel ? row[column - bytesPerPixel] : 0
                    let up = previous[column]
                    let upperLeft = column >= bytesPerPixel ? previous[column - bytesPerPixel] : 0
                    let prediction: UInt8
                    switch filter {
                    case 0: prediction = 0
                    case 1: prediction = left
                    case 2: prediction = up
                    case 3: prediction = UInt8((Int(left) + Int(up)) / 2)
                    default: prediction = paeth(left, up, upperLeft)
                    }
                    row[column] = row[column] &+ prediction
                }
                if dx == 1 {
                    let destination = ((y + rowIndex * dy) * header.width + x) * bytesPerPixel
                    pixels.replaceSubrange(destination..<destination + rowBytes, with: row)
                } else {
                    for column in 0..<width {
                        let destination = ((y + rowIndex * dy) * header.width + x + column * dx) * bytesPerPixel
                        pixels.replaceSubrange(destination..<destination + bytesPerPixel,
                            with: row[column * bytesPerPixel..<(column + 1) * bytesPerPixel])
                    }
                }
                previous = row
            }
        }
        return NativePNG(header: header, pixels: pixels, colorChunks: colors)
    }
    static func sourceMetadata(_ data: Data) throws -> [String: Any] {
        let header = try parseHeader(data)
        var metadata: [String: Any] = ["width": header.width, "height": header.height, "sample_bits": header.bits, "channels": header.channels, "png_color_type": Int(header.color), "interlace": Int(header.interlace)]
        var cursor = 8, colors: [[String: String]] = [], ended = false
        while cursor + 12 <= data.count {
            let length = Int(integer(data, cursor))
            guard length <= data.count - cursor - 12 else { throw invalid("Truncated PNG chunk.") }
            let kind = String(decoding: data[cursor + 4..<cursor + 8], as: UTF8.self), payload = data.subdata(in: cursor + 8..<cursor + 8 + length)
            guard checksum(data.subdata(in: cursor + 4..<cursor + 8 + length)) == integer(data, cursor + 8 + length) else { throw invalid("PNG chunk checksum changed.") }
            if kind == "IHDR", cursor != 8 { throw invalid("PNG contains more than one image header.") }
            if kind == "tRNS" { throw invalid("PNG transparency tables need an explicit native channel conversion before training.") }
            if retainedChunks.contains(kind) {
                colors.append(["type": kind, "data_base64": payload.base64EncodedString()])
                if kind == "gAMA", payload.count == 4 { metadata["png_gamma"] = Double(integer(payload, 0)) / 100000 }
                if kind == "sRGB", payload.count == 1 { metadata["srgb_rendering_intent"] = Int(payload[0]) }
            }
            cursor += length + 12
            if kind == "IEND" { ended = true; break }
        }
        guard ended, cursor == data.count else { throw invalid("PNG image is incomplete.") }
        metadata["color_chunks"] = colors; return metadata
    }
    func crop(_ rectangle: [Int], flipGreen: Bool = false) throws -> NativePNG {
        guard rectangle.count == 4, rectangle[0] >= 0, rectangle[1] >= 0,
              rectangle[2] > 0, rectangle[3] > 0, rectangle[2] <= header.width,
              rectangle[3] <= header.height, rectangle[0] <= header.width - rectangle[2],
              rectangle[1] <= header.height - rectangle[3] else { throw Self.invalid("Crop is outside the original PNG grid.") }
        guard !flipGreen || [3, 4].contains(header.channels) else { throw Self.invalid("DirectX normal maps need RGB integer channels.") }
        let width = rectangle[2], height = rectangle[3], rowBytes = width * header.bytesPerPixel
        var selected = Data(); selected.reserveCapacity(rowBytes * height)
        for row in 0..<height {
            try Task.checkCancellation()
            let start = ((rectangle[1] + row) * header.width + rectangle[0]) * header.bytesPerPixel
            selected.append(pixels[start..<start + rowBytes])
        }
        if flipGreen {
            let componentBytes = header.bits / 8
            for pixel in 0..<width * height {
                let green = pixel * header.bytesPerPixel + componentBytes
                // max - unsigned code is a byte complement at either precision.
                for byte in 0..<componentBytes { selected[green + byte] ^= 255 }
            }
        }
        return NativePNG(header: Header(width: width, height: height, bits: header.bits,
            channels: header.channels, color: header.color, interlace: 0), pixels: selected, colorChunks: colorChunks)
    }
    func encoded() throws -> Data {
        let rowBytes = header.width * header.bytesPerPixel
        guard pixels.count == rowBytes * header.height else { throw Self.invalid("PNG sample storage has an invalid size.") }
        var scanlines = Data(); scanlines.reserveCapacity(pixels.count + header.height)
        for row in 0..<header.height { scanlines.append(0); scanlines.append(pixels[row * rowBytes..<(row + 1) * rowBytes]) }
        var length = compressBound(uLong(scanlines.count)), compressed = Data(count: Int(length))
        let status = compressed.withUnsafeMutableBytes { destination in
            scanlines.withUnsafeBytes { source in
                compress2(destination.bindMemory(to: Bytef.self).baseAddress!, &length,
                    source.bindMemory(to: Bytef.self).baseAddress!, uLong(scanlines.count), 3)
            }
        }
        guard status == Z_OK else { throw Self.invalid("Lossless PNG encoding failed.") }
        compressed.count = Int(length)
        var ihdr = Data(); ihdr.appendBE(UInt32(header.width)); ihdr.appendBE(UInt32(header.height))
        ihdr.append(contentsOf: [UInt8(header.bits), header.color, 0, 0, 0])
        var output = Self.signature
        output.append(Self.chunk("IHDR", ihdr))
        for (kind, payload) in colorChunks { output.append(Self.chunk(kind, payload)) }
        output.append(Self.chunk("IDAT", compressed)); output.append(Self.chunk("IEND", Data()))
        return output
    }
    /// Model tensors are a separate, explicit conversion from immutable integer
    /// source storage. Returned values use planar CHW order, matching training.
    func modelFloatSamples(role: String, encoding: String = "linear_data", normalConvention: String = "opengl") throws -> [Float] {
        guard ["input", "height", "roughness", "normal"].contains(role),
              role != "height" || header.bits == 16,
              !["input", "normal"].contains(role) || [3, 4].contains(header.channels) else { throw Self.invalid("The native map channels or precision do not support this training target.") }
        let count = header.width * header.height, outputChannels = ["input", "normal"].contains(role) ? 3 : 1
        let maximum = header.bits == 16 ? 65535 : 255
        var result = [Float](repeating: 0, count: count * outputChannels)
        func code(_ pixel: Int, _ channel: Int) -> Int {
            let index = pixel * header.bytesPerPixel + channel * header.bits / 8
            return header.bits == 16 ? Int(pixels[index]) * 256 + Int(pixels[index + 1]) : Int(pixels[index])
        }
        let linear = ["linear", "linear_rgb", "linear_color", "linear_light"].contains(encoding)
        if role == "input", !linear, !["srgb", "sRGB", "source_srgb_assumed", "srgb_display", "srgb_color"].contains(encoding) { throw Self.invalid("Diffuse color transfer must be explicit.") }
        for pixel in 0..<count {
            if pixel & 4095 == 0 { try Task.checkCancellation() }
            if [2, 4].contains(header.channels), code(pixel, header.channels - 1) != maximum { throw Self.invalid("Transparent material crops need review; no context pixels are fabricated.") }
            if ["height", "roughness"].contains(role), header.channels >= 3,
               (code(pixel, 0) != code(pixel, 1) || code(pixel, 0) != code(pixel, 2)) { throw Self.invalid("Scalar material maps need identical RGB codes.") }
            for channel in 0..<outputChannels {
                var value = Float(code(pixel, channel)) / Float(maximum)
                if role == "normal", channel == 1, normalConvention.lowercased() == "directx" { value = Float(maximum - code(pixel, channel)) / Float(maximum) }
                if role == "input", linear { value = value <= 0.0031308 ? 12.92 * value : 1.055 * pow(value, 1 / 2.4) - 0.055 }
                result[channel * count + pixel] = value
            }
        }
        return result
    }
    private static func chunk(_ kind: String, _ payload: Data) -> Data {
        var result = Data(); result.appendBE(UInt32(payload.count))
        let content = Data(kind.utf8) + payload
        result.append(content); result.appendBE(checksum(content)); return result
    }
    private static func checksum(_ data: Data) -> UInt32 {
        data.withUnsafeBytes { UInt32(crc32(0, $0.bindMemory(to: Bytef.self).baseAddress, uInt(data.count))) }
    }
    private static func integer(_ data: Data, _ offset: Int) -> UInt32 {
        data[offset..<offset + 4].reduce(0) { ($0 << 8) | UInt32($1) }
    }
    private static func paeth(_ a: UInt8, _ b: UInt8, _ c: UInt8) -> UInt8 {
        let p = Int(a) + Int(b) - Int(c), pa = abs(p - Int(a)), pb = abs(p - Int(b)), pc = abs(p - Int(c))
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c
    }
    private static func invalid(_ message: String) -> StudioError { StudioError(message) }
}

private extension Data {
    mutating func appendBE(_ value: UInt32) { append(contentsOf: [UInt8(truncatingIfNeeded: value >> 24), UInt8(truncatingIfNeeded: value >> 16), UInt8(truncatingIfNeeded: value >> 8), UInt8(truncatingIfNeeded: value)]) }
}
