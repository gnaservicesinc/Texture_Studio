import XCTest
@testable import TextureStudio

final class MachineResourcesTests: XCTestCase {
    private let gib = MachineResources.gibibyte

    func test64GiBMacAllowsMoreThanHalfOfItsMemoryAndUsesAnAdaptiveDefault() {
        let resources = MachineResources(physicalBytes: 64 * gib, metalRecommendedBytes: 52 * gib)
        XCTAssertEqual(resources.maximumTrainingGiB, 57.6, accuracy: 0.000_001)
        XCTAssertEqual(resources.defaultTrainingGiB, 51.2, accuracy: 0.000_001)
        XCTAssertEqual(resources.maximumTrainingBytes, 61_847_529_062)
        XCTAssertEqual(resources.defaultTrainingBytes, 54_975_581_388)
        XCTAssertNil(resources.trainingMemoryIssue(56))
        XCTAssertNil(resources.trainingMemoryIssue(57.6), "The displayed ceiling can be entered in the text field")
        XCTAssertEqual(resources.trainingMemoryBytes(57.6), Int64(resources.maximumTrainingBytes))
        XCTAssertNotNil(resources.trainingMemoryIssue(57.7))
        XCTAssertNotNil(resources.trainingMemoryIssue(64))
        XCTAssertEqual(resources.trainingMemoryBytes(56), 56 * Int64(gib))
    }

    func testMemoryCeilingAndRecommendationScaleWithMachineCapacity() {
        let small = MachineResources(physicalBytes: 8 * gib, metalRecommendedBytes: 6 * gib)
        XCTAssertEqual(small.maximumTrainingGiB, 4)
        let large = MachineResources(physicalBytes: 128 * gib, metalRecommendedBytes: 104 * gib)
        XCTAssertEqual(large.maximumTrainingGiB, 115.2, accuracy: 0.000_001)
        XCTAssertEqual(large.defaultTrainingGiB, 102.4, accuracy: 0.000_001)
    }

    func testExplicitAllocatorLimitIsRespectedAndDisabledAllocatorKeepsOSHeadroom() {
        let limited = MachineResources(physicalBytes: 64 * gib, metalRecommendedBytes: 48 * gib, highWatermarkRatio: 0.75)
        XCTAssertEqual(limited.maximumTrainingGiB, 36)
        XCTAssertEqual(limited.defaultTrainingGiB, 36)
        let allocatorDisabled = MachineResources(physicalBytes: 64 * gib, metalRecommendedBytes: 48 * gib, highWatermarkRatio: 0)
        XCTAssertEqual(allocatorDisabled.maximumTrainingGiB, 57.6, accuracy: 0.000_001)
        let smallGPU = MachineResources(physicalBytes: 64 * gib, metalRecommendedBytes: 8 * gib)
        XCTAssertEqual(smallGPU.maximumTrainingGiB, 13.6, accuracy: 0.000_001)
    }

    func testInvalidMemoryNeverReachesByteConversion() {
        let resources = MachineResources(physicalBytes: 64 * gib)
        for value in [Double.nan, .infinity, -.infinity, 0, -1, 1000] {
            XCTAssertNotNil(resources.trainingMemoryIssue(value))
        }
        XCTAssertEqual(resources.trainingMemoryBytes(12), 12_884_901_888)
    }
}
