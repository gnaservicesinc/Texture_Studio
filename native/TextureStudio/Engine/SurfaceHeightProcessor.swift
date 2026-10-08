import Foundation

/// Converts a registered depth prediction into an explicitly derived, artistic surface height.
/// The photograph never supplies geometry here. Raw distance/inverse-depth samples stay unchanged.
enum SurfaceHeightProcessor {
    struct Result {
        let values: [Float]
        let hasRelief: Bool
        let repairedSamples: Int
        let removedPlane: [Double]
        let amplitudeGain: Float
    }

    static func derive(values: [Float], width: Int, height: Int, interpretation: DepthInterpretation,
                       planeRemoval: Float, cleanup: Float, strength: Float, invert: Bool,
                       adaptive: Bool = true) throws -> Result {
        guard width > 1,height > 1,values.count == width*height else {
            throw TextureError.invalidDepth("Invalid surface-depth grid.")
        }
        let finite = values.filter(\.isFinite).sorted()
        guard finite.count >= max(16,values.count/2) else {
            throw TextureError.invalidDepth("The selected depth map has too many missing samples to form a material surface.")
        }
        let replacement = percentile(finite,0.5)
        let rawSpan = percentile(finite,0.98)-percentile(finite,0.02)
        let rawMagnitude = max(abs(finite.first!),abs(finite.last!))
        let numericalFloor = max(Float.leastNormalMagnitude,rawMagnitude*Float.ulpOfOne*16)
        var repaired = values
        var repairedCount = 0
        for index in repaired.indices where !repaired[index].isFinite {
            let x=index%width,y=index/width
            var neighbours = [Float]()
            for dy in -2...2 {
                for dx in -2...2 {
                    let sx=min(width-1,max(0,x+dx)),sy=min(height-1,max(0,y+dy))
                    let value=values[sy*width+sx]
                    if value.isFinite { neighbours.append(value) }
                }
            }
            repaired[index] = neighbours.isEmpty ? replacement : percentile(neighbours.sorted(),0.5)
            repairedCount += 1
        }
        let plane = try robustPlane(repaired,width:width,height:height)
        let direction: Float = (interpretation == .distance ? -1 : 1) * (invert ? -1 : 1)
        var relief = [Float](repeating:0,count:values.count)
        for y in 0..<height {
            if y % 64 == 0 { try Task.checkCancellation() }
            let ny = Double(y)*2/Double(height-1)-1
            for x in 0..<width {
                let nx = Double(x)*2/Double(width-1)-1
                // Remove the global tilt, not a broad Gaussian: genuine large surface
                // ridges and depressions survive instead of collapsing to edge halos.
                let baseline = plane[0] + Double(planeRemoval)*(plane[1]*nx+plane[2]*ny)
                relief[y*width+x] = direction*Float(Double(repaired[y*width+x])-baseline)
            }
        }
        let initial = relief.sorted()
        let initialSpan = percentile(initial,0.98)-percentile(initial,0.02)
        guard initialSpan > max(numericalFloor,rawSpan*0.00001) else {
            return Result(values:[Float](repeating:0.5,count:values.count),hasRelief:false,
                          repairedSamples:repairedCount,removedPlane:plane,amplitudeGain:0)
        }
        if cleanup > 0 {
            relief = try clean(relief,width:width,height:height,span:initialSpan,amount:cleanup)
        }
        let sorted = relief.sorted()
        let centre = percentile(sorted,0.5)
        let radius = max(centre-percentile(sorted,0.02),percentile(sorted,0.98)-centre)
        guard radius > numericalFloor else {
            return Result(values:[Float](repeating:0.5,count:values.count),hasRelief:false,
                          repairedSamples:repairedCount,removedPlane:plane,amplitudeGain:0)
        }
        // Distance models can leave small broad residual errors on an almost flat
        // wall. Stretching every residual to the full artistic range exaggerates
        // those errors. This optional artistic safeguard fades relief below 5%
        // relative distance contrast; it is not a confidence or accuracy measure.
        // Artist-supplied inverse-depth/height maps retain their chosen contrast.
        let amplitudeGain: Float = adaptive && interpretation == .distance && replacement > 0
            ? min(1,initialSpan/(abs(replacement)*0.05)) : 1
        // Symmetric mapping preserves the zero-height plane at 0.5. The 2–98% range
        // uses most of the material range while isolated predictions cannot dominate it.
        let output = relief.map { min(1,max(0,0.5+($0-centre)/radius*0.45*strength*amplitudeGain)) }
        return Result(values:output,hasRelief:true,repairedSamples:repairedCount,removedPlane:plane,
                      amplitudeGain:amplitudeGain)
    }

    private static func clean(_ input: [Float], width: Int, height: Int, span: Float, amount: Float) throws -> [Float] {
        var output = input
        let rangeSigma = max(span*(0.01+amount*0.04),Float.leastNormalMagnitude)
        let rangeDenominator = 2*rangeSigma*rangeSigma
        for y in 0..<height {
            if y % 32 == 0 { try Task.checkCancellation() }
            for x in 0..<width {
                let index=y*width+x,centre=input[index]
                var neighbourhood = [Float]()
                neighbourhood.reserveCapacity(9)
                for dy in -1...1 {
                    for dx in -1...1 {
                        neighbourhood.append(input[min(height-1,max(0,y+dy))*width+min(width-1,max(0,x+dx))])
                    }
                }
                let median = percentile(neighbourhood.sorted(),0.5)
                let mad = percentile(neighbourhood.map { abs($0-median) }.sorted(),0.5)
                let isSpike = abs(centre-median) > max(span*0.02,mad*6)
                // Strong isolated outliers are repaired even with a mild cleanup setting;
                // bilateral smoothing remains mild and preserves actual depth steps.
                let anchor = isSpike ? centre+(median-centre)*min(1,amount*4) : centre
                var weighted:Float=0,weightSum:Float=0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let neighbour=input[min(height-1,max(0,y+dy))*width+min(width-1,max(0,x+dx))]
                        let delta=neighbour-anchor
                        let spatial:Float = dx == 0 && dy == 0 ? 1 : dx == 0 || dy == 0 ? 0.7 : 0.5
                        let weight=spatial*exp(-delta*delta/rangeDenominator)
                        weighted += neighbour*weight;weightSum += weight
                    }
                }
                let smoothed=weightSum > 0 ? weighted/weightSum : anchor
                output[index]=anchor+(smoothed-anchor)*amount
            }
        }
        return output
    }

    private static func robustPlane(_ values: [Float], width: Int, height: Int) throws -> [Double] {
        // A bounded sample grid keeps full-resolution render working memory predictable.
        let strideX=max(1,width/96),strideY=max(1,height/96)
        var samples = [(Double,Double,Double)]()
        for y in stride(from:0,to:height,by:strideY) {
            for x in stride(from:0,to:width,by:strideX) {
                samples.append((Double(x)*2/Double(width-1)-1,Double(y)*2/Double(height-1)-1,Double(values[y*width+x])))
            }
        }
        var weights=[Double](repeating:1,count:samples.count)
        var plane=[Double](repeating:0,count:3)
        for _ in 0..<5 {
            try Task.checkCancellation()
            var matrix=[[Double]](repeating:[Double](repeating:0,count:4),count:3)
            for (index,sample) in samples.enumerated() {
                let p=[1.0,sample.0,sample.1],weight=weights[index]
                for row in 0..<3 {
                    for column in 0..<3 { matrix[row][column] += weight*p[row]*p[column] }
                    matrix[row][3] += weight*p[row]*sample.2
                }
            }
            guard let fit=solve3(matrix) else { return [Double(values[values.count/2]),0,0] }
            plane=fit
            let residuals=samples.map { $0.2-plane[0]-plane[1]*$0.0-plane[2]*$0.1 }
            let absolute=residuals.map(abs).sorted()
            let scale=max(1e-12,absolute[absolute.count/2]*1.4826*1.5)
            for index in weights.indices { weights[index]=min(1,scale/max(abs(residuals[index]),1e-12)) }
        }
        return plane
    }

    private static func percentile(_ sorted: [Float], _ fraction: Double) -> Float {
        let position=Double(sorted.count-1)*fraction
        let lower=Int(position),upper=min(sorted.count-1,lower+1)
        return sorted[lower]+(sorted[upper]-sorted[lower])*Float(position-Double(lower))
    }
    private static func solve3(_ matrix: [[Double]]) -> [Double]? {
        var a=matrix
        for column in 0..<3 {
            let pivot=(column..<3).max { abs(a[$0][column]) < abs(a[$1][column]) }!
            guard abs(a[pivot][column]) > 1e-12 else { return nil }
            a.swapAt(column,pivot)
            let denominator=a[column][column]
            for entry in column..<4 { a[column][entry] /= denominator }
            for row in 0..<3 where row != column {
                let multiple=a[row][column]
                for entry in column..<4 { a[row][entry] -= multiple*a[column][entry] }
            }
        }
        return (0..<3).map { a[$0][3] }
    }
}
