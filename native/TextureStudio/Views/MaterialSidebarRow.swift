import SwiftUI

/// Explicit button selection avoids NSTableView's drag-selection recognizer,
/// which can assert on a negative row while a sectioned list scrolls/changes.
struct MaterialSidebarRow<Content: View>: View {
    let selected: Bool
    let action: () -> Void
    @ViewBuilder let content: () -> Content

    var body: some View {
        Button(action: action) {
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10).padding(.vertical, 8)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(selected ? Color.accentColor.opacity(0.18) : .clear, in: RoundedRectangle(cornerRadius: 7))
        .accessibilityAddTraits(selected ? [.isSelected] : [])
    }
}

enum MaterialSidebarSelection {
    static func next<ID: Equatable>(_ current: ID?, in ids: [ID], direction: Int) -> ID? {
        guard !ids.isEmpty else { return nil }
        guard let current, let index = ids.firstIndex(of: current) else {
            return direction < 0 ? ids.last : ids.first
        }
        return ids[min(max(index + direction, 0), ids.count - 1)]
    }
}
