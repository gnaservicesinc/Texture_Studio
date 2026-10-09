import Foundation
import CoreGraphics

/// Project a plane about its centre. Coordinates follow Core Image: origin at bottom left.
enum TextureGeometry {
    struct HalfPlane {
        let x: Double, y: Double, limit: Double
    }
    static func projectedCorners(width: Double, height: Double, xDegrees: Double,
                                 yDegrees: Double, zDegrees: Double, focalPixels: Double) throws -> [CGPoint] {
        guard [width, height, xDegrees, yDegrees, zDegrees, focalPixels].allSatisfy(\.isFinite),
              width > 2, height > 2, focalPixels > 0, abs(xDegrees) <= 70, abs(yDegrees) <= 70 else {
            throw TextureError.invalidSettings("Use finite dimensions, a positive focal length, and X/Y angles within ±70°.")
        }
        let rx = xDegrees * .pi / 180, ry = yDegrees * .pi / 180, rz = zDegrees * .pi / 180
        let centre = CGPoint(x: width / 2, y: height / 2)
        // Counter-clockwise corners: bottom left, bottom right, top right, top left.
        return try [CGPoint(x: 0, y: 0), CGPoint(x: width, y: 0),
                    CGPoint(x: width, y: height), CGPoint(x: 0, y: height)].map { point in
            let px = point.x - centre.x, py = point.y - centre.y
            let a = px, b = py * cos(rx), c = py * sin(rx)
            let d = a * cos(ry) + c * sin(ry), e = b, f = -a * sin(ry) + c * cos(ry)
            let rotatedX = d * cos(rz) - e * sin(rz), rotatedY = d * sin(rz) + e * cos(rz)
            guard focalPixels - f > focalPixels * 0.05 else {
                throw TextureError.invalidSettings("This angle crosses the camera plane. Reduce rotation or increase focal length.")
            }
            let scale = focalPixels / (focalPixels - f)
            return CGPoint(x: centre.x + rotatedX * scale, y: centre.y + rotatedY * scale)
        }
    }

    static func halfPlanes(_ polygon: [CGPoint], inset: Double = 0) throws -> [HalfPlane] {
        guard polygon.count >= 3 else { throw TextureError.processing("A valid crop needs a convex polygon.") }
        var result = [HalfPlane]()
        for index in polygon.indices {
            let a = polygon[index], b = polygon[(index + 1) % polygon.count]
            let dx = b.x - a.x, dy = b.y - a.y, length = hypot(dx, dy)
            guard length > 1e-9 else { throw TextureError.processing("Degenerate transformed image.") }
            // Outward normal for counter-clockwise vertices.
            let nx = dy / length, ny = -dx / length
            result.append(HalfPlane(x: nx, y: ny, limit: nx * a.x + ny * a.y - inset))
        }
        guard polygon.allSatisfy({ p in result.allSatisfy { $0.x*p.x + $0.y*p.y <= $0.limit + inset + 1e-7 } }) else {
            throw TextureError.processing("Transform must remain a convex, front-facing image.")
        }
        return result
    }

    /// Exact maximum-area fixed-aspect axis-aligned rectangle inside a convex polygon.
    /// For fixed aspect the problem is a 3-variable linear program (cx, cy, half-height).
    /// Enumerating triples of active constraints finds its global optimum, without raster masks.
    static func maximumCrop(in polygon: [CGPoint], aspect: Double = 1, inset: Double = 1.5) throws -> CGRect {
        guard aspect.isFinite, aspect > 0 else { throw TextureError.invalidSettings("Crop aspect must be positive.") }
        let planes = try halfPlanes(polygon, inset: inset)
        var best: (x: Double, y: Double, h: Double)?
        let preferred = CGPoint(x: polygon.map(\.x).reduce(0,+) / Double(polygon.count),
                                y: polygon.map(\.y).reduce(0,+) / Double(polygon.count))
        for i in 0..<(planes.count - 2) {
            for j in (i+1)..<(planes.count - 1) {
                for k in (j+1)..<planes.count {
                    let selected = [planes[i], planes[j], planes[k]]
                    let matrix = selected.map { [$0.x, $0.y, abs($0.x) * aspect + abs($0.y), $0.limit] }
                    guard let solved = solve3(matrix), solved[2] > 0 else { continue }
                    let valid = planes.allSatisfy {
                        $0.x*solved[0] + $0.y*solved[1] + solved[2]*(abs($0.x)*aspect + abs($0.y)) <= $0.limit + 1e-7
                    }
                    guard valid else { continue }
                    let candidate = (x: solved[0], y: solved[1], h: solved[2])
                    let oldDistance = best.map { hypot($0.x-preferred.x, $0.y-preferred.y) } ?? .infinity
                    if best == nil || candidate.h > best!.h + 1e-7 ||
                        (abs(candidate.h-best!.h) < 1e-7 && hypot(candidate.x-preferred.x,candidate.y-preferred.y) < oldDistance) {
                        best = candidate
                    }
                }
            }
        }
        guard var best, best.h >= 1 else { throw TextureError.processing("The transform leaves no usable image area.") }
        // The optimal centre may lie on a segment (e.g. a landscape rectangle's square crop).
        // Project the polygon centre into that feasible segment to keep the user's central framing.
        var cx = preferred.x, cy = preferred.y
        for _ in 0..<32 {
            for plane in planes {
                let excess = plane.x*cx + plane.y*cy + best.h*(abs(plane.x)*aspect+abs(plane.y)) - plane.limit
                if excess > 0 { cx -= excess*plane.x; cy -= excess*plane.y }
            }
            if planes.allSatisfy({ $0.x*cx+$0.y*cy+best.h*(abs($0.x)*aspect+abs($0.y)) <= $0.limit+1e-7 }) {
                best.x = cx; best.y = cy; break
            }
        }
        return CGRect(x: best.x-best.h*aspect, y: best.y-best.h,
                      width: best.h*2*aspect, height: best.h*2)
    }

    static func framedCrop(_ maximum: CGRect, scale: Double, offsetX: Double, offsetY: Double) throws -> CGRect {
        guard [scale,offsetX,offsetY].allSatisfy(\.isFinite), scale > 0, scale <= 1,
              abs(offsetX) <= 1, abs(offsetY) <= 1 else {
            throw TextureError.invalidSettings("Crop scale must be in (0,1], and offsets within ±1.")
        }
        let width = maximum.width * scale, height = maximum.height * scale
        return CGRect(x: maximum.midX - width/2 + offsetX*(maximum.width-width)/2,
                      y: maximum.midY - height/2 + offsetY*(maximum.height-height)/2,
                      width: width, height: height)
    }

    private static func solve3(_ augmented: [[Double]]) -> [Double]? {
        var a = augmented
        for column in 0..<3 {
            let pivot = (column..<3).max(by: { abs(a[$0][column]) < abs(a[$1][column]) })!
            guard abs(a[pivot][column]) > 1e-10 else { return nil }
            a.swapAt(column,pivot)
            let factor = a[column][column]
            for entry in column..<4 { a[column][entry] /= factor }
            for row in 0..<3 where row != column {
                let multiplier = a[row][column]
                for entry in column..<4 { a[row][entry] -= multiplier*a[column][entry] }
            }
        }
        return (0..<3).map { a[$0][3] }
    }
}
