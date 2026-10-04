// Native ImageIO/Vision bridge. The Python caller independently verifies every
// retained array, image inventory, and spatial calibration before installation.
import Foundation
import ImageIO
import CoreGraphics
import CoreImage
import CoreVideo
import Vision

struct BridgeFailure: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}
func jsonValue(_ value: Any) -> Any {
    if let data = value as? Data { return ["encoding": "base64", "data": data.base64EncodedString()] }
    if let values = value as? [String: Any] { return values.mapValues(jsonValue) }
    if let values = value as? [Any] { return values.map(jsonValue) }
    if value is String || value is NSNumber || value is NSNull { return value }
    return String(describing: value)
}
func emit(_ result: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: jsonValue(result), options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
}
let auxiliaryTypes: [(String, CFString)] = [
    ("depth", kCGImageAuxiliaryDataTypeDepth), ("disparity", kCGImageAuxiliaryDataTypeDisparity),
    ("portrait_effects_matte", kCGImageAuxiliaryDataTypePortraitEffectsMatte),
    ("semantic_skin_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationSkinMatte),
    ("semantic_hair_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationHairMatte),
    ("semantic_teeth_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationTeethMatte),
    ("semantic_glasses_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationGlassesMatte),
    ("semantic_sky_matte", kCGImageAuxiliaryDataTypeSemanticSegmentationSkyMatte),
    ("hdr_gain_map", kCGImageAuxiliaryDataTypeHDRGainMap)]
func source(_ path: String) throws -> CGImageSource {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else {
        throw BridgeFailure("ImageIO cannot recognize this image")
    }
    return source
}
func allProperties(_ source: CGImageSource) -> [String: Any] {
    let images = (0..<CGImageSourceGetCount(source)).map {
        (CGImageSourceCopyPropertiesAtIndex(source, $0, nil) as? [String: Any]) ?? [:]
    }
    return ["global": CGImageSourceCopyProperties(source, nil) as? [String: Any] ?? [:], "images": images]
}
func auxiliaryRecord(_ info: CFDictionary, semantic: String, parent: Int,
                     directory: URL, ordinal: Int) throws -> [String: Any] {
    let dictionary = info as NSDictionary
    guard let data = dictionary[kCGImageAuxiliaryDataInfoData] as? Data,
          let description = dictionary[kCGImageAuxiliaryDataInfoDataDescription] as? [String: Any] else {
        throw BridgeFailure("Auxiliary \(semantic) is missing its native data description")
    }
    let file = directory.appendingPathComponent("aux-\(ordinal).bin")
    try data.write(to: file)
    var result: [String: Any] = ["semantic": semantic, "parent": parent, "path": file.path,
                               "description": description]
    if let metadata = dictionary[kCGImageAuxiliaryDataInfoMetadata] {
        let typed = unsafeBitCast(metadata as CFTypeRef, to: CGImageMetadata.self)
        if let xmp = CGImageMetadataCreateXMPData(typed, nil) {
            result["xmp"] = (xmp as Data).base64EncodedString()
        }
    }
    return result
}
func inspect(_ input: String, _ output: String, render: Bool = true) throws {
    let src = try source(input)
    let directory = URL(fileURLWithPath: output, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    var report: [String: Any] = ["properties": allProperties(src), "image_count": CGImageSourceGetCount(src)]
    var auxiliaries: [[String: Any]] = []
    var types = auxiliaryTypes
    if #available(macOS 15.0, *) { types.append(("iso_gain_map", kCGImageAuxiliaryDataTypeISOGainMap)) }
    for parent in 0..<CGImageSourceGetCount(src) {
        for (semantic, type) in types {
            if let info = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, parent, type) {
                auxiliaries.append(try auxiliaryRecord(info, semantic: semantic, parent: parent,
                                                      directory: directory, ordinal: auxiliaries.count))
            }
        }
    }
    report["auxiliary"] = auxiliaries
    if !render { try emit(report); return }
    // RAW conversion is explicitly derived. Render linear float32 RGB to keep
    // values above 1 rather than writing an 8-bit screen representation.
    let url = URL(fileURLWithPath: input)
    let image: CIImage?
    var rendering = "ImageIO/CoreImage decoded color (derived RGB, not sensor mosaic)"
    if #available(macOS 12.0, *), let raw = CIRAWFilter(imageURL: url) {
        raw.isDraftModeEnabled = false
        image = raw.outputImage
        rendering = "CIRAWFilter linear processed camera RGB (derived, not sensor mosaic)"
    } else {
        image = CIImage(contentsOf: url, options: [.applyOrientationProperty: false])
    }
    guard let rgb = image else { throw BridgeFailure("Native RAW/color rendering is unavailable") }
    let extent = rgb.extent.integral
    let width = Int(extent.width), height = Int(extent.height)
    guard width > 0 && height > 0 else { throw BridgeFailure("Invalid native rendered grid") }
    let colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
    let context = CIContext(options: [.workingColorSpace: colorspace])
    var samples = [Float](repeating: 0, count: width * height * 4)
    samples.withUnsafeMutableBytes { buffer in
        context.render(rgb, toBitmap: buffer.baseAddress!, rowBytes: width * 16,
                       bounds: extent, format: .RGBAf, colorSpace: colorspace)
    }
    let file = directory.appendingPathComponent("rendered-rgba-f32.bin")
    try samples.withUnsafeBytes { try Data($0).write(to: file) }
    report["rendered"] = ["path": file.path, "width": width, "height": height,
                          "dtype": "float32", "channels": 4, "description": rendering,
                          "color_space": "extended linear sRGB", "derived": true]
    try emit(report)
}
func personMask(_ input: String, _ output: String) throws {
    let src = try source(input)
    guard let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw BridgeFailure("ImageIO could not decode a Vision reference image")
    }
    let request = VNGeneratePersonSegmentationRequest()
    request.qualityLevel = .accurate
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8
    let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
    try handler.perform([request])
    guard let buffer = request.results?.first?.pixelBuffer else {
        throw BridgeFailure("Vision returned no person segmentation mask")
    }
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    guard let bytes = CVPixelBufferGetBaseAddress(buffer) else { throw BridgeFailure("Vision mask buffer is unavailable") }
    var packed = Data(capacity: width * height)
    for y in 0..<height { packed.append(bytes.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self), count: width) }
    try packed.write(to: URL(fileURLWithPath: output))
    try emit(["path": output, "width": width, "height": height, "dtype": "uint8", "derived": true,
              "model": "Apple Vision accurate person segmentation", "reference_width": image.width,
              "reference_height": image.height, "orientation": "encoded image grid"])
}
let forbiddenTokens = ["gps", "location", "latitude", "longitude", "altitude", "datetime", "timestamp",
                       "serialnumber", "ownername", "artist", "author", "copyright", "description", "comment",
                       "makernote", "makerapple", "documentname", "imagename", "originalfilename",
                       "hostcomputer", "software", "lensmodel", "lensmake", "cameramake", "cameramodel"]
func forbidden(_ key: String) -> Bool {
    let text = key.lowercased().filter { $0.isLetter || $0.isNumber }
    return forbiddenTokens.contains { text.contains($0) }
}
// Privacy is fail-closed: only known functional image, color, HDR and
// stereo-camera fields are accepted. Unrecognized metadata requires refusal.
let functionalKeys: Set<String> = [
    "global", "images", "{FileContents}", "{Groups}", "{HEIF}", "{TIFF}", "{Exif}",
    "CanAnimate", "FileSize", "ImageCount", "Images", "AuxiliaryData", "AuxiliaryDataType",
    "Width", "Height", "PixelWidth", "PixelHeight", "PixelFormat", "Orientation", "ImageIndex",
    "PrimaryImage", "ColorModel", "Depth", "DPIHeight", "DPIWidth", "ProfileName", "NamedColorSpace",
    "ChromaSubsampling", "FlexRange", "Headroom", "HDRHeadroom", "DerivationDetails",
    "TonemapAlternateHDRHeadroom", "TonemapBaseColorIsWorkingColor", "TonemapBaseHDRHeadroom",
    "TonemapChannelMetadata", "AlternateOffset", "BaseOffset", "GainMapMax", "GainMapMin", "Gamma",
    "ThumbnailImages", "ThumbnailOffset", "ThumbnailSize", "ResolutionUnit", "XResolution", "YResolution",
    "TileLength", "TileWidth", "ColorSpace", "ComponentsConfiguration", "PixelXDimension", "PixelYDimension",
    "GroupType", "GroupIndex", "GroupImageIndexLeft", "GroupImageIndexRight", "GroupImageIndexMonoscopic",
    "GroupImageIndexMonoscopicImageLocation", "GroupImageDisparityAdjustment", "GroupImageStereoAggressors",
    "CameraExtrinsics", "CameraModel", "Position", "Rotation", "CoordinateSystemID", "Intrinsics", "ModelType"
]
func privacyPropertiesAllowed(_ value: Any) -> Bool {
    if let dictionary = value as? [String: Any] {
        return dictionary.allSatisfy { functionalKeys.contains($0.key) && privacyPropertiesAllowed($0.value) }
    }
    if let items = value as? [Any] { return items.allSatisfy { privacyPropertiesAllowed($0) } }
    return value is String || value is NSNumber || value is NSNull
}
let functionalAuxTags: Set<String> = ["Version", "DataType", "Accuracy", "Quality", "Near", "Far",
    "MeasureType", "Format", "Units", "Min", "Max", "DepthNear", "DepthFar", "DisparityNear", "DisparityFar",
    "AuxiliaryImageType", "Orientation", "Width", "Height", "HDRGainMapVersion", "HDRGainMapHeadroom",
    "HDRGainMapMin", "HDRGainMapMax", "HDRGainMapGamma", "HDRGainMapOffsetSDR", "HDRGainMapOffsetHDR",
    "HDRGainMapCapacityMin", "HDRGainMapCapacityMax", "BaseRenditionIsHDR", "HDRCapacityMin", "HDRCapacityMax",
    "GainMapMin", "GainMapMax", "Gamma", "OffsetSDR", "OffsetHDR"]
func rewrite(_ input: String, _ output: String, _ configPath: String) throws {
    let config = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: configPath))) as! [String: Any]
    let privacy = config["privacy"] as? Bool ?? false
    let replacements = config["replacements"] as? [[String: Any]] ?? []
    let src = try source(input)
    guard let type = CGImageSourceGetType(src),
          let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: output) as CFURL,
                                                           type, CGImageSourceGetCount(src), nil) else {
        throw BridgeFailure("ImageIO cannot create the requested image destination")
    }
    let cleanMetadata = CGImageMetadataCreateMutable()
    if privacy {
        // A nonempty structural metadata object prevents ImageIO treating an
        // empty metadata object as an omitted update on some HEIC versions.
        let first = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [String: Any] ?? [:]
        let orientation = first[kCGImagePropertyOrientation as String] as? NSNumber ?? 1
        guard CGImageMetadataSetValueMatchingImageProperty(cleanMetadata, kCGImagePropertyTIFFDictionary,
                                                           kCGImagePropertyTIFFOrientation, orientation) else {
            throw BridgeFailure("Cannot construct structural-only privacy metadata")
        }
    }
    if replacements.isEmpty {
        var options: [CFString: Any] = [:]
        if privacy {
            options[kCGImageDestinationMetadata] = cleanMetadata
            options[kCGImageDestinationMergeMetadata] = false
            options[kCGImageMetadataShouldExcludeGPS] = true
        }
        var error: Unmanaged<CFError>?
        guard CGImageDestinationCopyImageSource(destination, src, options as CFDictionary, &error) else {
            let reason = error.map { String(describing: $0.takeRetainedValue()) } ?? "unsupported container"
            throw BridgeFailure("ImageIO refused lossless source-to-destination HEIC copy: \(reason). No recompression fallback is used.")
        }
        // CopyImageSource saves the destination itself. Finalize is forbidden.
    } else {
        // ImageIO has no lossless source-copy API that replaces an auxiliary
        // plane. Attempt its native source-image writer; Python rejects the
        // candidate if even one retained RGB/auxiliary bit changes.
        let global = (CGImageSourceCopyProperties(src, nil) as? [String: Any]) ?? [:]
        CGImageDestinationSetProperties(destination, global as CFDictionary)
        var types = auxiliaryTypes
        if #available(macOS 15.0, *) { types.append(("iso_gain_map", kCGImageAuxiliaryDataTypeISOGainMap)) }
        for parent in 0..<CGImageSourceGetCount(src) {
            var properties = (CGImageSourceCopyPropertiesAtIndex(src, parent, nil) as? [String: Any]) ?? [:]
            properties[kCGImageDestinationLossyCompressionQuality as String] = 1.0
            if privacy {
                properties[kCGImagePropertyExifDictionary as String] = NSNull()
                properties[kCGImagePropertyExifAuxDictionary as String] = NSNull()
                properties[kCGImagePropertyGPSDictionary as String] = NSNull()
                properties[kCGImagePropertyIPTCDictionary as String] = NSNull()
                properties[kCGImagePropertyMakerAppleDictionary as String] = NSNull()
            }
            CGImageDestinationAddImageFromSource(destination, src, parent, properties as CFDictionary)
            for (semantic, auxType) in types {
                guard let info = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, parent, auxType) else { continue }
                var values = info as! [String: Any]
                if let replacement = replacements.first(where: { ($0["parent"] as? Int) == parent && ($0["semantic"] as? String) == semantic }) {
                    values[kCGImageAuxiliaryDataInfoData as String] = try Data(contentsOf: URL(fileURLWithPath: replacement["path"] as! String))
                    // Replacement has the exact original native layout,
                    // including row padding and floating-point representation.
                    values[kCGImageAuxiliaryDataInfoDataDescription as String] = replacement["description"]

                }
                CGImageDestinationAddAuxiliaryDataInfo(destination, auxType, values as CFDictionary)
            }
        }
        guard CGImageDestinationFinalize(destination) else { throw BridgeFailure("ImageIO failed to finalize native HEIC repair") }
    }
    if privacy {
        // Randomize textual UUIDs consistently, including references and item
        // identifiers; subsequent Python decode comparison verifies fidelity.
        var bytes = try Data(contentsOf: URL(fileURLWithPath: output))
        var text = String(decoding: bytes, as: UTF8.self)
        let regex = try NSRegularExpression(pattern: "[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}")
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        var identifiers = Set<String>()
        for match in matches { if let range = Range(match.range, in: text) { identifiers.insert(String(text[range])) } }
        for identifier in identifiers {
            let old = Data(identifier.utf8), new = Data(UUID().uuidString.lowercased().utf8)
            while let range = bytes.range(of: old) { bytes.replaceSubrange(range, with: new) }
        }
        let filename = URL(fileURLWithPath: input).lastPathComponent
        let stem = URL(fileURLWithPath: input).deletingPathExtension().lastPathComponent
        for oldText in [filename, stem] where oldText.utf8.count >= 6 {
            let old = Data(oldText.utf8)
            let alphabet = Array("abcdefghijklmnopqrstuvwxyz0123456789".utf8)
            let random = Data((0..<old.count).map { _ in alphabet.randomElement()! })
            while let range = bytes.range(of: old) { bytes.replaceSubrange(range, with: random) }
        }
        try bytes.write(to: URL(fileURLWithPath: output))
        let check = try source(output)
        if !privacyPropertiesAllowed(allProperties(check)) {
            throw BridgeFailure("Native writer retained identifying metadata; privacy output refused")
        }
        // Inspect auxiliary XMP too; unsupported personal tags cause refusal.
        for parent in 0..<CGImageSourceGetCount(check) {
            for (_, type) in auxiliaryTypes {
                guard let info = CGImageSourceCopyAuxiliaryDataInfoAtIndex(check, parent, type) else { continue }
                let values = info as NSDictionary
                if let metadata = values[kCGImageAuxiliaryDataInfoMetadata] {
                    let typed = unsafeBitCast(metadata as CFTypeRef, to: CGImageMetadata.self)
                    var bad = false
                    CGImageMetadataEnumerateTagsUsingBlock(typed, nil, nil) { path, _ in
                        let name = (path as String).split(separator: ":").last.map(String.init) ?? ""
                        if !functionalAuxTags.contains(name) { bad = true }; return true
                    }
                    if bad { throw BridgeFailure("Identifying auxiliary XMP remains; privacy output refused") }
                }
            }
        }
        text = "" // avoid retaining a second full-file textual copy
    }
    try emit(["privacy_verified": privacy, "replacement_metadata_verified": false,
              "backend": "macOS ImageIO", "copy_mode": replacements.isEmpty ? "lossless source copy" : "native source-image auxiliary replacement",
              "replacement_count": replacements.count])
}
do {
    let args = CommandLine.arguments
    guard args.count >= 4 else { throw BridgeFailure("Usage: media-bridge inspect|mask|rewrite SOURCE OUTPUT [CONFIG]") }
    switch args[1] {
    case "inspect": try inspect(args[2], args[3])
    case "inventory": try inspect(args[2], args[3], render: false)
    case "mask": try personMask(args[2], args[3])
    case "rewrite": guard args.count == 5 else { throw BridgeFailure("Rewrite requires configuration") }; try rewrite(args[2], args[3], args[4])
    default: throw BridgeFailure("Unknown native media action")
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(2)
}
