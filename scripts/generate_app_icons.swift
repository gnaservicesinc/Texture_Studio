#!/usr/bin/env swift
// Texture Studio's original vector artwork. Regenerate with:
//   swift scripts/generate_app_icons.swift out/native-material-tools/icons
// No photos, model output, fonts, or third-party icon assets are used.
import AppKit
import Foundation

let destination = URL(fileURLWithPath: CommandLine.arguments.count > 1
    ? CommandLine.arguments[1] : "out/native-material-tools/icons", isDirectory: true)
let manager = FileManager.default
try manager.createDirectory(at: destination, withIntermediateDirectories: true)
let temporary = manager.temporaryDirectory.appendingPathComponent("texture-studio-icons-\(UUID().uuidString)")
try manager.createDirectory(at: temporary, withIntermediateDirectories: true)
defer { try? manager.removeItem(at: temporary) }

func color(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: red, green: green, blue: blue, alpha: alpha)
}
func path(_ points: [CGPoint], close: Bool = true) -> CGPath {
    let result = CGMutablePath()
    result.addLines(between: points)
    if close { result.closeSubpath() }
    return result
}
func point(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: x, y: y) }
func polygon(_ context: CGContext, _ points: [CGPoint], _ fill: CGColor) {
    context.addPath(path(points)); context.setFillColor(fill); context.fillPath()
}
func gradient(_ context: CGContext, _ first: CGColor, _ last: CGColor,
              from: CGPoint, to: CGPoint) {
    let fill = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                          colors: [first, last] as CFArray, locations: [0, 1])!
    context.drawLinearGradient(fill, start: from, end: to,
        options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
}

// A lit sample of a real surface floats above the height and normal-map layers.
// Geometry stays broad and readable at Dock/sidebar sizes; contour detail only
// becomes visible when macOS chooses a larger representation.
func drawTile(_ context: CGContext, offset: CGFloat, top: CGColor,
              side: CGColor, edge: CGColor, thickness: CGFloat) {
    let vertices = [point(512, 800 + offset), point(846, 602 + offset),
                    point(512, 404 + offset), point(178, 602 + offset)]
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -16), blur: 22,
                      color: color(0.01, 0.035, 0.075, 0.32))
    polygon(context, [vertices[3], vertices[2], vertices[1],
                      point(846, 602 + offset - thickness),
                      point(512, 404 + offset - thickness),
                      point(178, 602 + offset - thickness)], side)
    context.restoreGState()
    context.saveGState()
    context.addPath(path(vertices)); context.clip()
    gradient(context, top, edge, from: point(350, 800 + offset), to: point(740, 390 + offset))
    context.restoreGState()
    context.addPath(path([vertices[3], vertices[2], vertices[1]], close: false))
    context.setStrokeColor(color(1, 1, 1, 0.35)); context.setLineWidth(2.5); context.strokePath()
    polygon(context, [vertices[2], vertices[1], point(846, 602 + offset - thickness),
                      point(512, 404 + offset - thickness)], color(0.02, 0.08, 0.15, 0.16))
}

func drawSurface(_ context: CGContext) {
    let top = path([point(512, 800), point(846, 602), point(512, 404), point(178, 602)])
    context.saveGState()
    context.addPath(top); context.clip()
    // Nested organic contours imply relief without relying on busy photographic
    // texture or color noise. The deterministic curves are resolution independent.
    for row in -2...10 {
        let curve = CGMutablePath()
        let baseline = CGFloat(row) * 54 + 375
        curve.move(to: point(120, baseline))
        for column in 0..<7 {
            let x = CGFloat(column) * 112 + 120
            let wave = sin(CGFloat(row) * 1.17 + CGFloat(column) * 0.92)
            curve.addCurve(to: point(x + 112, baseline + sin(CGFloat(column + 1) * 0.92 + CGFloat(row) * 1.17) * 22),
                           control1: point(x + 32, baseline + wave * 24 + 18),
                           control2: point(x + 76, baseline + wave * 24 - 18))
        }
        context.addPath(curve); context.setStrokeColor(color(0.38, 0.25, 0.16, 0.21))
        context.setLineWidth(row % 3 == 0 ? 7 : 3); context.strokePath()
        context.addPath(curve); context.setStrokeColor(color(1, 0.96, 0.82, 0.37))
        context.setLineWidth(1.5); context.strokePath()
    }
    for index in 0..<15 {
        let x = 250 + CGFloat((index * 163) % 510)
        let y = 470 + CGFloat((index * 127) % 270)
        let width = 15 + CGFloat((index * 7) % 26)
        let rect = CGRect(x: x, y: y, width: width, height: width * 0.58)
        context.setFillColor(color(0.99, 0.91, 0.72, 0.22)); context.fillEllipse(in: rect)
    }
    context.restoreGState()
}

func drawBadge(_ context: CGContext, role: String) {
    guard role != "studio" else { return }
    let accents: [String: CGColor] = ["review": color(0.16, 0.58, 0.76),
        "compare": color(0.41, 0.39, 0.76), "dataset": color(0.20, 0.60, 0.48),
        "train": color(0.84, 0.42, 0.20)]
    let badge = CGRect(x: 660, y: 120, width: 220, height: 220)
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -8), blur: 14, color: color(0, 0, 0, 0.27))
    context.setFillColor(accents[role]!); context.fillEllipse(in: badge)
    context.restoreGState()
    context.setStrokeColor(color(1, 1, 1, 0.94)); context.setFillColor(color(1, 1, 1, 0.94))
    context.setLineWidth(13); context.setLineCap(.round); context.setLineJoin(.round)
    switch role {
    case "review":
        context.strokeEllipse(in: CGRect(x: 715, y: 206, width: 75, height: 75))
        context.addPath(path([point(779, 216), point(824, 173)], close: false)); context.strokePath()
    case "compare":
        context.stroke(CGRect(x: 713, y: 179, width: 114, height: 103))
        context.addPath(path([point(770, 179), point(770, 282)], close: false)); context.strokePath()
    case "dataset":
        for x: CGFloat in [714, 781] {
            for y: CGFloat in [175, 242] { context.fill(CGRect(x: x, y: y, width: 43, height: 43)) }
        }
    default:
        for (x, y): (CGFloat, CGFloat) in [(716, 225), (770, 270), (824, 201)] {
            context.addPath(path([point(x, 177), point(x, 285)], close: false)); context.strokePath()
            context.setFillColor(accents[role]!); context.fillEllipse(in: CGRect(x: x - 15, y: y - 15, width: 30, height: 30))
            context.strokeEllipse(in: CGRect(x: x - 15, y: y - 15, width: 30, height: 30))
        }
    }
}

func render(size: Int, role: String) -> Data {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    bitmap.size = NSSize(width: size, height: size)
    let context = NSGraphicsContext(bitmapImageRep: bitmap)!.cgContext
    context.scaleBy(x: CGFloat(size) / 1024, y: CGFloat(size) / 1024)
    context.setShouldAntialias(true)
    let background = CGPath(roundedRect: CGRect(x: 88, y: 88, width: 848, height: 848),
                            cornerWidth: 188, cornerHeight: 188, transform: nil)
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -12), blur: 20, color: color(0.01, 0.02, 0.045, 0.22))
    context.addPath(background); context.setFillColor(color(0.08, 0.14, 0.22)); context.fillPath()
    context.restoreGState()
    context.saveGState()
    context.addPath(background); context.clip()
    gradient(context, color(0.16, 0.25, 0.35), color(0.055, 0.10, 0.18), from: point(230, 900), to: point(760, 120))
    context.restoreGState()
    context.addPath(background); context.setStrokeColor(color(0.65, 0.83, 0.95, 0.20))
    context.setLineWidth(2); context.strokePath()
    drawTile(context, offset: -170, top: color(0.30, 0.40, 0.84), side: color(0.18, 0.23, 0.48),
             edge: color(0.31, 0.64, 0.86), thickness: 28)
    drawTile(context, offset: -86, top: color(0.37, 0.84, 0.77), side: color(0.12, 0.44, 0.47),
             edge: color(0.13, 0.52, 0.60), thickness: 30)
    drawTile(context, offset: 0, top: color(0.95, 0.87, 0.70), side: color(0.58, 0.43, 0.29),
             edge: color(0.75, 0.62, 0.43), thickness: 46)
    drawSurface(context)
    drawBadge(context, role: role)
    return bitmap.representation(using: .png, properties: [:])!
}

let icons = ["studio": "TextureStudio", "review": "MaterialReview", "compare": "CheckpointCompare",
             "dataset": "MaterialDataset", "train": "MaterialTrainer"]
for role in icons.keys.sorted() {
    let name = icons[role]!
    let iconset = temporary.appendingPathComponent("\(name).iconset")
    try manager.createDirectory(at: iconset, withIntermediateDirectories: true)
    for size in [16, 32, 128, 256, 512] {
        for scale in [1, 2] {
            let suffix = scale == 2 ? "@2x" : ""
            try render(size: size * scale, role: role).write(to:
                iconset.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
        }
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
    process.arguments = ["--convert", "icns", "--output", destination.appendingPathComponent("\(name).icns").path,
                         iconset.path]
    try process.run(); process.waitUntilExit()
    guard process.terminationStatus == 0 else { fatalError("iconutil failed for \(name)") }
    print("Created \(name).icns")
}
