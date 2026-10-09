import XCTest
@testable import TextureStudio

final class NumericStatisticsTests: XCTestCase {
    func testAccelerateOrderingMatchesScalarStatisticsAndPreservesSourceWords() {
        var samples = (0..<16384).map { Float(($0 * 137) % 8191 - 4096) / 256 }
        samples += [Float.leastNonzeroMagnitude, -Float.leastNonzeroMagnitude, Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, 0, -0.0]
        let original = samples.map(\.bitPattern)
        let native = NumericStatistics.sorted(samples)
        let scalar = samples.sorted()
        XCTAssertEqual(native, scalar)
        XCTAssertEqual(samples.map(\.bitPattern), original)
        XCTAssertEqual(native.map(\.bitPattern).sorted(), original.sorted(), "Sorting must not alter sample encodings")
        for fraction in [0.02, 0.5, 0.98] {
            let index = Int(Double(native.count - 1) * fraction)
            XCTAssertEqual(native[index], scalar[index])
        }
    }

    func testEmptyAndSingletonArrays() {
        XCTAssertEqual(NumericStatistics.sorted([]), [])
        let singleton: [Float] = [-0.0]
        XCTAssertEqual(NumericStatistics.sorted(singleton).map(\.bitPattern), singleton.map(\.bitPattern))
    }
}
