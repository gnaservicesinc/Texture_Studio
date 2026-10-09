import Foundation
import XCTest
import zlib
@testable import TextureStudio

final class NativePNGTests: XCTestCase {
    func testEveryFilterAndAdam7PreserveEightAndSixteenBitIntegerChannels() throws {
        for bits in [8, 16] { for (color, channels) in [(UInt8(0), 1), (2, 3), (4, 2), (6, 4)] {
            for (width, height) in [(1, 1), (1, 7), (9, 7)] {
                let header = NativePNG.Header(width: width, height: height, bits: bits, channels: channels, color: color, interlace: 0)
                let codes = Data((0..<width * height * header.bytesPerPixel).map { UInt8(truncatingIfNeeded: $0 * 97 + $0 / 7 * 31) })
                for interlace: UInt8 in [0, 1] { for filter: UInt8 in 0...4 {
                    let fixture = try independentPNG(header: header, codes: codes, filter: filter, interlace: interlace)
                    let decoded = try NativePNG.decode(fixture)
                    XCTAssertEqual(decoded.pixels, codes, "\(bits)-bit color \(color), filter \(filter), interlace \(interlace), \(width)×\(height)")
                    XCTAssertEqual(decoded.header.bits, bits); XCTAssertEqual(decoded.header.channels, channels)
                    XCTAssertEqual(try NativePNG.decode(decoded.encoded()).pixels, codes)
                } }
            }
        } }
    }

    func testCroppedNormalComplementsOnlyGreenAtBothPrecisions() throws {
        for bits in [8, 16] { for (color, channels) in [(UInt8(2), 3), (6, 4)] {
            let header = NativePNG.Header(width: 9, height: 7, bits: bits, channels: channels, color: color, interlace: 0)
            let pixels = Data((0..<9 * 7 * header.bytesPerPixel).map { UInt8(truncatingIfNeeded: $0 * 97) })
            let source = NativePNG(header: header, pixels: pixels, colorChunks: [("gAMA", Data([0, 0, 177, 143]))])
            let crop = try source.crop([2, 1, 5, 4], flipGreen: true)
            var expected = Data()
            for row in 1..<5 { for column in 2..<7 { for byte in 0..<header.bytesPerPixel {
                let original = pixels[(row * 9 + column) * header.bytesPerPixel + byte]
                expected.append(byte >= bits / 8 && byte < 2 * bits / 8 ? original ^ 255 : original)
            } } }
            XCTAssertEqual(crop.pixels, expected)
            XCTAssertEqual(try crop.crop([0, 0, 5, 4], flipGreen: true).pixels, try source.crop([2, 1, 5, 4]).pixels)
            XCTAssertEqual(try NativePNG.decode(crop.encoded()).pixels, expected)
            XCTAssertEqual(source.pixels, pixels)
            XCTAssertEqual(crop.colorChunks.first?.1, source.colorChunks.first?.1)
        } }
    }

    func testPlanarModelSamplesKeepExactFloatConversionAndRejectInvalidScalarOrAlpha() throws {
        for bits in [8, 16] {
            let maximum = bits == 16 ? 65535 : 255
            let values = [0, maximum / 2, maximum, maximum, maximum / 4, maximum * 3 / 4]
            var bytes = Data()
            for code in values { if bits == 16 { bytes.append(UInt8(code >> 8)) }; bytes.append(UInt8(truncatingIfNeeded: code)) }
            let png = NativePNG(header: .init(width: 2, height: 1, bits: bits, channels: 3, color: 2, interlace: 0), pixels: bytes, colorChunks: [])
            let samples = try png.modelFloatSamples(role: "normal", normalConvention: "DirectX")
            XCTAssertEqual(samples, [Float(values[0]) / Float(maximum), Float(values[3]) / Float(maximum), Float(maximum - values[1]) / Float(maximum), Float(maximum - values[4]) / Float(maximum), Float(values[2]) / Float(maximum), Float(values[5]) / Float(maximum)])
            XCTAssertEqual(png.pixels, bytes)
            XCTAssertThrowsError(try png.modelFloatSamples(role: "roughness"))
        }
        let transparent = NativePNG(header: .init(width: 1, height: 1, bits: 8, channels: 4, color: 6, interlace: 0), pixels: Data([128, 128, 255, 0]), colorChunks: [])
        XCTAssertThrowsError(try transparent.modelFloatSamples(role: "normal"))
    }

    func testInvalidStorageAndUnknownFiltersFailBeforeUnsafeBufferAccess() throws {
        let invalid = NativePNG(header: .init(width: 0, height: 1, bits: 8, channels: 1, color: 0, interlace: 0), pixels: Data(), colorChunks: [])
        XCTAssertThrowsError(try invalid.encoded()); XCTAssertThrowsError(try invalid.crop([0, 0, 1, 1])); XCTAssertThrowsError(try invalid.modelFloatSamples(role: "roughness"))
        let short = NativePNG(header: .init(width: 2, height: 1, bits: 16, channels: 3, color: 2, interlace: 0), pixels: Data([0]), colorChunks: [])
        XCTAssertThrowsError(try short.encoded()); XCTAssertThrowsError(try short.crop([0, 0, 1, 1])); XCTAssertThrowsError(try short.modelFloatSamples(role: "input"))
        let header = NativePNG.Header(width: 1, height: 1, bits: 8, channels: 1, color: 0, interlace: 0)
        XCTAssertThrowsError(try NativePNG.decode(independentPNG(header: header, codes: Data([17]), filter: 5, interlace: 0)))
    }

    func testNativeOperationsHonorTaskCancellation() async throws {
        let header = NativePNG.Header(width: 9, height: 7, bits: 16, channels: 1, color: 0, interlace: 0)
        let png = NativePNG(header: header, pixels: Data(repeating: 97, count: 9 * 7 * 2), colorChunks: [])
        let encoded = try png.encoded()
        let task = Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            var cancelled = 0
            for operation in [
                { _ = try NativePNG.decode(encoded) },
                { _ = try NativePNG.sourceMetadata(encoded) },
                { _ = try png.crop([1, 1, 5, 4]) },
                { _ = try png.encoded() },
                { _ = try png.modelFloatSamples(role: "height") }
            ] {
                do { try operation() } catch is CancellationError { cancelled += 1 }
            }
            return cancelled
        }
        let cancelled = try await task.value
        XCTAssertEqual(cancelled, 5)
    }

    // Deliberately independent Array implementation produces filtered/Adam7
    // fixtures, so native buffer restoration is checked against raw codes.
    private func independentPNG(header: NativePNG.Header, codes: Data, filter: UInt8, interlace: UInt8) throws -> Data {
        let passes = interlace == 0 ? [(0, 0, 1, 1)] : [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
        var filtered = Data()
        for (x, y, dx, dy) in passes {
            let width = max(0, (header.width - x + dx - 1) / dx), height = max(0, (header.height - y + dy - 1) / dy)
            guard width > 0, height > 0 else { continue }
            var previous = [UInt8](repeating: 0, count: width * header.bytesPerPixel)
            for row in 0..<height {
                var raw: [UInt8] = []
                for column in 0..<width {
                    let position = ((y + row * dy) * header.width + x + column * dx) * header.bytesPerPixel
                    raw.append(contentsOf: codes[position..<position + header.bytesPerPixel])
                }
                filtered.append(filter)
                for column in raw.indices {
                    let left = column >= header.bytesPerPixel ? raw[column - header.bytesPerPixel] : 0
                    let up = previous[column], upperLeft = column >= header.bytesPerPixel ? previous[column - header.bytesPerPixel] : 0
                    let prediction: UInt8
                    switch filter {
                    case 1: prediction = left
                    case 2: prediction = up
                    case 3: prediction = UInt8((Int(left) + Int(up)) / 2)
                    case 4:
                        let p = Int(left) + Int(up) - Int(upperLeft)
                        let distance = [abs(p - Int(left)), abs(p - Int(up)), abs(p - Int(upperLeft))]
                        prediction = distance[0] <= distance[1] && distance[0] <= distance[2] ? left : distance[1] <= distance[2] ? up : upperLeft
                    default: prediction = 0
                    }
                    filtered.append(raw[column] &- prediction)
                }
                previous = raw
            }
        }
        var compressedLength = compressBound(uLong(filtered.count)), compressed = Data(count: Int(compressedLength))
        let status = compressed.withUnsafeMutableBytes { destination in filtered.withUnsafeBytes { source in compress2(destination.bindMemory(to: UInt8.self).baseAddress!, &compressedLength, source.bindMemory(to: UInt8.self).baseAddress!, uLong(filtered.count), 1) } }
        guard status == Z_OK else { throw NSError(domain: "PNGFixture", code: Int(status)) }
        compressed.count = Int(compressedLength)
        func bigEndian(_ value: UInt32) -> Data { Data([UInt8(truncatingIfNeeded: value >> 24), UInt8(truncatingIfNeeded: value >> 16), UInt8(truncatingIfNeeded: value >> 8), UInt8(truncatingIfNeeded: value)]) }
        func chunk(_ name: String, _ payload: Data) -> Data {
            let contents = Data(name.utf8) + payload
            let crc = contents.withUnsafeBytes { crc32(0, $0.bindMemory(to: UInt8.self).baseAddress, uInt(contents.count)) }
            return bigEndian(UInt32(payload.count)) + contents + bigEndian(UInt32(crc))
        }
        let ihdr = bigEndian(UInt32(header.width)) + bigEndian(UInt32(header.height)) + Data([UInt8(header.bits), header.color, 0, 0, interlace])
        return Data([137, 80, 78, 71, 13, 10, 26, 10]) + chunk("IHDR", ihdr) + chunk("IDAT", compressed) + chunk("IEND", Data())
    }
}
