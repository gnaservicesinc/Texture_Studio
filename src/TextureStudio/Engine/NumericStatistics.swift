import Accelerate
import Foundation

/// Accelerate sorts an owned copy of finite derived samples. Source auxiliary
/// buffers never enter this path; their exported words remain byte-for-byte.
enum NumericStatistics {
    static func sorted(_ values: [Float]) -> [Float] {
        guard values.count > 1 else { return values }
        var copy = values
        copy.withUnsafeMutableBufferPointer { buffer in
            vDSP_vsort(buffer.baseAddress!, vDSP_Length(buffer.count), 1)
        }
        return copy
    }
}
