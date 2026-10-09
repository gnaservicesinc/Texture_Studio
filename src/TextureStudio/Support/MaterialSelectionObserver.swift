import Foundation

/// The training hub and Studio can also run as separate application processes.
@MainActor
final class MaterialSelectionObserver: NSObject {
    private let changed: @MainActor () -> Void
    private var lastSelectionID: String?
    private(set) var changedTarget: String?

    init(changed: @escaping @MainActor () -> Void) {
        self.changed = changed
        super.init()
        NotificationCenter.default.addObserver(self, selector: #selector(selectionChanged),
            name: SelectedMaterialCheckpoint.changeNotification, object: nil)
        DistributedNotificationCenter.default().addObserver(self, selector: #selector(selectionChanged),
            name: SelectedMaterialCheckpoint.changeNotification, object: nil, suspensionBehavior: .deliverImmediately)
    }

    @objc private func selectionChanged(_ notification: Notification) {
        if let identifier = notification.userInfo?["selectionID"] as? String {
            guard identifier != lastSelectionID else { return }
            lastSelectionID = identifier
        }
        changedTarget = notification.userInfo?["target"] as? String
        changed()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        DistributedNotificationCenter.default().removeObserver(self)
    }
}
