import Foundation
import Metal

/// Machine-local capacity. Map dimensions and model contracts have separate limits.
struct MachineResources: Equatable, Sendable {
    static let gibibyte: UInt64 = 1_073_741_824
    let physicalBytes: UInt64
    let metalRecommendedBytes: UInt64?
    let practicalBytes: UInt64
    let maximumTrainingBytes: UInt64
    let defaultTrainingBytes: UInt64

    init(physicalBytes: UInt64, metalRecommendedBytes: UInt64? = nil, highWatermarkRatio: Double? = 1.7) {
        self.physicalBytes = physicalBytes
        self.metalRecommendedBytes = metalRecommendedBytes.flatMap { $0 > 0 ? $0 : nil }
        let tenthRoundedUp = physicalBytes / 10 + (physicalBytes % 10 == 0 ? 0 : 1)
        let reserve = min(physicalBytes / 2, max(4 * Self.gibibyte, tenthRoundedUp))
        practicalBytes = physicalBytes - reserve
        var maximum = practicalBytes
        if let recommended = self.metalRecommendedBytes,
           let ratio = highWatermarkRatio, ratio.isFinite, ratio > 0 {
            let allocationLimit = Double(recommended) * ratio
            if allocationLimit < Double(maximum) { maximum = UInt64(max(0, allocationLimit)) }
        }
        maximumTrainingBytes = maximum
        let fifthRoundedUp = physicalBytes / 5 + (physicalBytes % 5 == 0 ? 0 : 1)
        defaultTrainingBytes = min(maximum, physicalBytes - fifthRoundedUp,
                                   self.metalRecommendedBytes ?? maximum)
    }

    static let current: MachineResources = {
        let ratio = ProcessInfo.processInfo.environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"].flatMap(Double.init) ?? 1.7
        return MachineResources(physicalBytes: ProcessInfo.processInfo.physicalMemory,
                                metalRecommendedBytes: MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize,
                                highWatermarkRatio: ratio)
    }()

    var maximumTrainingGiB: Double { Double(maximumTrainingBytes) / Double(Self.gibibyte) }
    var defaultTrainingGiB: Double { Double(defaultTrainingBytes) / Double(Self.gibibyte) }
    var physicalGiB: Double { Double(physicalBytes) / Double(Self.gibibyte) }
    var reservedGiB: Double { Double(physicalBytes - practicalBytes) / Double(Self.gibibyte) }
    var trainingMemoryRange: ClosedRange<Double> { min(2, maximumTrainingGiB)...maximumTrainingGiB }

    func trainingMemoryIssue(_ value: Double) -> String? {
        // Validate the integer byte count sent to Python. The exact 64 GiB
        // ceiling is a fraction of a byte below the displayed 57.6 GiB value;
        // both convert to the same allowed byte count after truncation.
        let bytes = (value * Double(Self.gibibyte)).rounded(.down)
        guard value.isFinite, value >= trainingMemoryRange.lowerBound,
              bytes <= Double(maximumTrainingBytes) else {
            return "Choose a memory limit between \(trainingMemoryRange.lowerBound.formatted(.number.precision(.fractionLength(0...1)))) and \(maximumTrainingGiB.formatted(.number.precision(.fractionLength(0...1)))) GiB for this Mac."
        }
        return nil
    }

    func trainingMemoryBytes(_ value: Double) -> Int64 {
        Int64(value * Double(Self.gibibyte))
    }
}
