import Foundation
import CoreImage

/// Apple's Core Image EXR writer produces HALF channels, with no format option in the SDK.
/// This minimal standard scanline writer supplies genuine 32-bit FLOAT channels. No transfer
/// function, quantization, normalization, or lossy compression is applied to these material maps.
enum FloatEXRWriter {
    static func write(_ image: CIImage, to url: URL, context: CIContext, color: Bool,
                      workingMemoryBytes:UInt64? = nil) throws {
        let width = Int(image.extent.width), height = Int(image.extent.height)
        guard image.extent.origin == .zero, width > 0, height > 0, width <= 8192, height <= 8192 else {
            throw TextureError.processing("Invalid EXR dimensions.")
        }
        let channels = color ? ["B","G","R"] : ["R"]
        let channelCount = channels.count
        let bytesPerScanline = width * channelCount * 4
        var header = Data()
        header.u32(20_000_630); header.u32(2)
        var list = Data()
        for channel in channels {
            list.cstring(channel); list.u32(2) // FLOAT in the OpenEXR pixel-type enumeration.
            list.append(contentsOf:[0,0,0,0]); list.u32(1); list.u32(1)
        }
        list.append(0)
        header.attribute("channels",type:"chlist",value:list)
        header.attribute("compression",type:"compression",value:Data([0]))
        var window = Data(); window.i32(0); window.i32(0); window.i32(Int32(width-1)); window.i32(Int32(height-1))
        header.attribute("dataWindow",type:"box2i",value:window)
        header.attribute("displayWindow",type:"box2i",value:window)
        header.attribute("lineOrder",type:"lineOrder",value:Data([0]))
        var scalar = Data(); scalar.f32(1)
        header.attribute("pixelAspectRatio",type:"float",value:scalar)
        var centre = Data(); centre.f32(0); centre.f32(0)
        header.attribute("screenWindowCenter",type:"v2f",value:centre)
        header.attribute("screenWindowWidth",type:"float",value:scalar)
        header.append(0)
        let firstChunk = UInt64(header.count + height*8)
        for row in 0..<height { header.u64(firstChunk+UInt64(row*(bytesPerScanline+8))) }
        guard FileManager.default.createFile(atPath:url.path,contents:nil) else {
            throw TextureError.processing("Cannot create the EXR file.")
        }
        let file = try FileHandle(forWritingTo:url)
        defer { try? file.close() }
        try file.write(contentsOf:header)
        // Re-evaluating the lazy image graph for every fixed 32-row strip is
        // expensive for lighting and blur kernels. Render larger strips when
        // memory permits, while retaining a bounded buffer on smaller Macs.
        // The budget includes both RGBA Float32 input and planar EXR chunks.
        let machineBudget = MachineResources.current.practicalBytes/16
        let availableBudget = TextureEngine.availableMemoryBytes().map { $0/8 } ?? machineBudget
        let stripHeight = renderStripHeight(width:width,height:height,color:color,
            workingMemoryBytes:workingMemoryBytes ?? min(machineBudget,availableBudget))
        var samples = [Float](repeating:0,count:width*stripHeight*4)
        for firstRow in stride(from:0,to:height,by:stripHeight) {
            try Task.checkCancellation()
            let count = min(stripHeight,height-firstRow)
            samples.withUnsafeMutableBytes {
                context.render(image,toBitmap:$0.baseAddress!,rowBytes:width*16,
                    bounds:CGRect(x:0,y:height-firstRow-count,width:width,height:count),format:.RGBAf,colorSpace:nil)
            }
            // Fill one contiguous strip directly. Appending four bytes for every
            // Float separately caused hundreds of millions of tiny Data appends
            // at 8K. This keeps exactly the same little-endian Float32 words.
            let chunkWords = bytesPerScanline/4+2
            var chunks = Data(count:count*(bytesPerScanline+8))
            try chunks.withUnsafeMutableBytes { bytes in
                let words = bytes.bindMemory(to:UInt32.self)
                for row in 0..<count {
                    if row%32 == 0 { try Task.checkCancellation() }
                    let chunkStart = row*chunkWords
                    words[chunkStart] = UInt32(firstRow+row).littleEndian
                    words[chunkStart+1] = UInt32(bytesPerScanline).littleEndian
                    for (channelIndex,channel) in channels.enumerated() {
                        let component = channel == "B" ? 2 : channel == "G" ? 1 : 0
                        let channelStart = chunkStart+2+channelIndex*width
                        for x in 0..<width {
                            words[channelStart+x] = samples[(row*width+x)*4+component].bitPattern.littleEndian
                        }
                    }
                }
            }
            try file.write(contentsOf:chunks)
        }
        try file.synchronize()
    }

    static func renderStripHeight(width:Int, height:Int, color:Bool,
                                  workingMemoryBytes:UInt64) -> Int {
        guard width > 0,height > 0 else { return 1 }
        let bytesPerRow = UInt64(width)*UInt64(color ? 28 : 20)+8
        return Int(min(UInt64(height),max(1,workingMemoryBytes/bytesPerRow)))
    }

    static func verifyChannelPrecision(at url: URL, expected: EXRPrecision) throws {
        let file = try FileHandle(forReadingFrom:url)
        defer { try? file.close() }
        let header = try file.read(upToCount:64*1024) ?? Data()
        let types = try channelTypes(header)
        guard !types.isEmpty, types.allSatisfy({ $0 == (expected == .float16 ? 1 : 2) }) else {
            throw TextureError.processing("The EXR channel precision does not match the requested format.")
        }
    }

    static func channelTypes(_ data: Data) throws -> [UInt32] {
        var reader = HeaderReader(data:data)
        guard try reader.u32() == 20_000_630 else { throw TextureError.processing("Not an OpenEXR file.") }
        _ = try reader.u32()
        while true {
            let name = try reader.string()
            if name.isEmpty { break }
            let type = try reader.string()
            let size = Int(try reader.u32())
            let end = reader.offset + size
            guard end <= data.count else { throw TextureError.processing("Incomplete EXR header.") }
            if name == "channels", type == "chlist" {
                var result = [UInt32]()
                while reader.offset < end {
                    if try reader.string().isEmpty { break }
                    result.append(try reader.u32()); reader.offset += 12
                }
                return result
            }
            reader.offset = end
        }
        throw TextureError.processing("No EXR channels were found.")
    }

    private struct HeaderReader {
        let data: Data
        var offset = 0
        mutating func u32() throws -> UInt32 {
            guard offset+4 <= data.count else { throw TextureError.processing("Incomplete EXR header.") }
            let value = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset:offset,as:UInt32.self) }
            offset += 4
            return UInt32(littleEndian:value)
        }
        mutating func string() throws -> String {
            let start = offset
            while offset < data.count, data[offset] != 0 { offset += 1 }
            guard offset < data.count else { throw TextureError.processing("Incomplete EXR header.") }
            let string = String(decoding:data[start..<offset],as:UTF8.self)
            offset += 1
            return string
        }
    }
}

private extension Data {
    mutating func cstring(_ string: String) { append(contentsOf:string.utf8); append(0) }
    mutating func u32(_ value: UInt32) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of:&value) { append(contentsOf:$0) }
    }
    mutating func i32(_ value: Int32) { u32(UInt32(bitPattern:value)) }
    mutating func u64(_ value: UInt64) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of:&value) { append(contentsOf:$0) }
    }
    mutating func f32(_ value: Float) { u32(value.bitPattern) }
    mutating func attribute(_ name: String, type: String, value: Data) {
        cstring(name); cstring(type); u32(UInt32(value.count)); append(value)
    }
}
