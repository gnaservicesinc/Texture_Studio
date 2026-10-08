import Foundation

@MainActor
final class WorkbenchLifecycle {
    static let shared = WorkbenchLifecycle()
    private var runners: [ObjectIdentifier: WorkbenchProcess] = [:]
    private(set) var isTerminating = false
    var hasOperations: Bool { !runners.isEmpty }
    func add(_ runner: WorkbenchProcess) { runners[ObjectIdentifier(runner)] = runner }
    func remove(_ runner: WorkbenchProcess) { runners.removeValue(forKey: ObjectIdentifier(runner)) }
    func stopAll() { isTerminating = true; for runner in runners.values { runner.stop() } }
}
