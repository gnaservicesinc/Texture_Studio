import Foundation

@MainActor
final class WorkbenchLifecycle {
    static let shared = WorkbenchLifecycle()
    private var runners: [ObjectIdentifier: NativeMaterialTrainingControl] = [:]
    private var cancellations: [ObjectIdentifier: @Sendable () -> Void] = [:]
    private(set) var isTerminating = false
    var hasOperations: Bool { !runners.isEmpty }
    func add(_ runner: NativeMaterialTrainingControl, cancel: @escaping @Sendable () -> Void = {}) {
        runners[ObjectIdentifier(runner)] = runner; cancellations[ObjectIdentifier(runner)] = cancel
    }
    func remove(_ runner: NativeMaterialTrainingControl) {
        runners.removeValue(forKey: ObjectIdentifier(runner)); cancellations.removeValue(forKey: ObjectIdentifier(runner))
    }
    func stopAll() {
        isTerminating = true
        for runner in runners.values { runner.stop() }
        for cancel in cancellations.values { cancel() }
    }
}
