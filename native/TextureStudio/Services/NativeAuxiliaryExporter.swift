import CoreVideo
import CryptoKit
import Foundation
import ImageIO

/// Copies ImageIO auxiliary buffers without passing samples through a renderer.
/// The BIN retains row padding and the original data description. NPY removes
/// padding only, when the system pixel format has a known scalar representation.
enum NativeAuxiliaryExporter {
    static var types: [(String, CFString)] { [
        ("depth", kCGImageAuxiliaryDataTypeDepth),
        ("disparity", kCGImageAuxiliaryDataTypeDisparity),
        ("portrait_effects_matte", kCGImageAuxiliaryDataTypePortraitEffectsMatte),
        ("semantic_skin_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationSkinMatte),
        ("semantic_hair_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationHairMatte),
        ("semantic_teeth_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationTeethMatte),
        ("semantic_glasses_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationGlassesMatte),
        ("semantic_sky_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationSkyMatte),
        ("hdr_gain_map", kCGImageAuxiliaryDataTypeHDRGainMap),
        ("iso_gain_map", kCGImageAuxiliaryDataTypeISOGainMap)
    ] }

    struct Report: Codable, Sendable {
        let schema: String
        let sourceName: String
        let sourceSHA256: String
        let imageCount: Int
        let auxiliaryCount: Int
        let records: [Record]
        let coverage: String
    }

    struct Record: Codable, Sendable {
        let semantic: String
        let parentImageIndex: Int
        let rawFile: String
        let rawSHA256: String
        let descriptionFile: String
        let metadataFile: String?
        let arrayFile: String?
        let arraySHA256: String?
        let sampleType: String?
        let width: Int?
        let height: Int?
    }

    /// Publishes a complete new directory only after every emitted file verifies.
    /// Reading one immutable source snapshot binds metadata and buffers to its hash.
    static func export(sourceURL: URL, to destination: URL) async throws -> Report {
        try Task.checkCancellation()
        let job = Task.detached(priority: .userInitiated) {
            try exportSynchronously(sourceURL: sourceURL, to: destination)
        }
        return try await withTaskCancellationHandler {
            try await job.value
        } onCancel: { job.cancel() }
    }

    static func exportSynchronously(sourceURL: URL, to destination: URL) throws -> Report {
        let manager = FileManager.default
        let destination = destination.standardizedFileURL
        guard !manager.fileExists(atPath: destination.path) else {
            throw StudioError("Choose a new auxiliary export folder.")
        }
        try Task.checkCancellation()
        let snapshot = try Data(contentsOf: sourceURL)
        guard let source = CGImageSourceCreateWithData(snapshot as CFData,
            [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetCount(source) > 0, CGImageSourceGetStatus(source) == .statusComplete else {
            throw StudioError("ImageIO cannot open this source photograph.")
        }
        let stage = destination.deletingLastPathComponent().appendingPathComponent(".auxiliary-export-\(UUID().uuidString)")
        try manager.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { try? manager.removeItem(at: stage) }
        var records: [Record] = []
        let count = CGImageSourceGetCount(source)
        for index in 0..<count {
            for (semantic, type) in types {
                try Task.checkCancellation()
                guard let auxiliary = CGImageSourceCopyAuxiliaryDataInfoAtIndex(source, index, type) else { continue }
                let info = auxiliary as NSDictionary
                guard let raw = info[kCGImageAuxiliaryDataInfoData] as? Data,
                      let description = info[kCGImageAuxiliaryDataInfoDataDescription] as? [String: Any] else {
                    throw StudioError("The \(semantic) auxiliary has no native buffer or data description.")
                }
                let stem = "image-\(index)-\(semantic)"
                let rawFile = stem + ".bin", descriptionFile = stem + ".plist"
                try writeVerified(raw, to: stage.appendingPathComponent(rawFile))
                let descriptionData = try PropertyListSerialization.data(fromPropertyList: description, format: .binary, options: 0)
                try writeVerified(descriptionData, to: stage.appendingPathComponent(descriptionFile))
                var metadataFile: String?
                if let metadata = info[kCGImageAuxiliaryDataInfoMetadata] {
                    let reference = metadata as CFTypeRef
                    guard CFGetTypeID(reference) == CGImageMetadataGetTypeID() else {
                        throw StudioError("The \(semantic) auxiliary metadata has an unsupported type.")
                    }
                    let typed = unsafeDowncast(reference, to: CGImageMetadata.self)
                    guard let xmp = CGImageMetadataCreateXMPData(typed, nil) else {
                        throw StudioError("Could not preserve \(semantic) auxiliary metadata.")
                    }
                    metadataFile = stem + ".xmp"
                    try writeVerified(xmp as Data, to: stage.appendingPathComponent(metadataFile!))
                }
                var arrayFile: String?, arraySHA256: String?, sampleType: String?
                if let array = try scalarArray(raw: raw, description: description) {
                    arrayFile = stem + ".npy"
                    sampleType = array.descriptor
                    let data = try array.npy()
                    try writeVerified(data, to: stage.appendingPathComponent(arrayFile!))
                    arraySHA256 = checksum(data)
                }
                records.append(Record(semantic: semantic, parentImageIndex: index, rawFile: rawFile,
                    rawSHA256: checksum(raw), descriptionFile: descriptionFile, metadataFile: metadataFile,
                    arrayFile: arrayFile, arraySHA256: arraySHA256, sampleType: sampleType,
                    width: (description["Width"] as? NSNumber)?.intValue,
                    height: (description["Height"] as? NSNumber)?.intValue))
            }
        }
        let report = Report(schema: "ipde-imageio-auxiliary-v1", sourceName: sourceURL.lastPathComponent,
            sourceSHA256: checksum(snapshot), imageCount: count, auxiliaryCount: records.count, records: records,
            coverage: "Known auxiliary types exposed by Apple ImageIO at every image index. Unexposed HEIF items are not inventoried. Buffers retain encoded orientation, native values and precision; no gamma, tone mapping or normalization.")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try writeVerified(encoder.encode(report), to: stage.appendingPathComponent("auxiliary.json"))
        try Task.checkCancellation()
        try manager.moveItem(at: stage, to: destination)
        return report
    }

    struct ScalarArray {
        let width: Int
        let height: Int
        let descriptor: String
        let bytes: Data

        /// NPY 1.0 is a public binary array format, with no runtime dependency.
        func npy() throws -> Data {
            let text = "{'descr': '\(descriptor)', 'fortran_order': False, 'shape': (\(height), \(width)), }"
            let padding = (64 - (10 + text.utf8.count + 1) % 64) % 64
            let header = Data((text + String(repeating: " ", count: padding) + "\n").utf8)
            guard header.count <= Int(UInt16.max) else { throw StudioError("Array header is too large.") }
            var result = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0,
                UInt8(header.count & 255), UInt8(header.count >> 8)])
            result.append(header)
            result.append(bytes)
            return result
        }
    }

    static func scalarArray(raw: Data, description: [String: Any]) throws -> ScalarArray? {
        guard let pixelFormat = (description["PixelFormat"] as? NSNumber)?.uint32Value else { return nil }
        let descriptor: String, byteCount: Int
        switch pixelFormat {
        case kCVPixelFormatType_DepthFloat16, kCVPixelFormatType_DisparityFloat16, kCVPixelFormatType_OneComponent16Half:
            descriptor = "<f2"; byteCount = 2
        case kCVPixelFormatType_DepthFloat32, kCVPixelFormatType_DisparityFloat32, kCVPixelFormatType_OneComponent32Float:
            descriptor = "<f4"; byteCount = 4
        case kCVPixelFormatType_OneComponent8:
            descriptor = "|u1"; byteCount = 1
        case kCVPixelFormatType_OneComponent16:
            descriptor = "<u2"; byteCount = 2
        default: return nil // Preserve opaque/planar formats as the original BIN.
        }
        guard let width = integer(description["Width"]), let height = integer(description["Height"]),
              let stride = integer(description["BytesPerRow"]), width > 0, height > 0,
              width <= Int.max / byteCount, stride >= width * byteCount,
              height <= Int.max / stride, raw.count >= stride * height else {
            throw StudioError("Invalid dimensions, row stride or truncated auxiliary buffer.")
        }
        let packedStride = width * byteCount
        var bytes = Data(capacity: packedStride * height)
        for row in 0..<height {
            if row % 64 == 0 { try Task.checkCancellation() }
            bytes.append(raw[(row * stride)..<(row * stride + packedStride)])
        }
        return ScalarArray(width: width, height: height, descriptor: descriptor, bytes: bytes)
    }

    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite, number.doubleValue.rounded(.towardZero) == number.doubleValue,
              number.doubleValue > Double(Int.min), number.doubleValue < Double(Int.max) else { return nil }
        return number.intValue
    }

    static func checksum(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func writeVerified(_ data: Data, to file: URL) throws {
        try data.write(to: file, options: .withoutOverwriting)
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.synchronize()
        guard try Data(contentsOf: file) == data else { throw StudioError("Auxiliary export failed byte verification.") }
    }
}
