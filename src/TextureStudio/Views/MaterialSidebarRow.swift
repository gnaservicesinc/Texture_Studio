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

/// The arrow and the row name both select; disclosure additionally opens or closes children.
struct DatasetDisclosureRow<Content: View, Accessory: View>: View {
    let title: String
    let nodeID: String
    let expanded: Bool
    let selected: Bool
    let select: () -> Void
    let disclose: () -> Void
    @ViewBuilder let content: () -> Content
    @ViewBuilder let accessory: () -> Accessory

    var body: some View {
        HStack(spacing: 3) {
            Button(action: disclose) {
                Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 16, height: 18)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 3))
                    .frame(width: 22, height: 28)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(expanded ? "Collapse" : "Expand") and select \(title)")
            .accessibilityIdentifier("dataset-disclosure-\(nodeID)")
            Button(action: select) {
                content().frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 7).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Select \(title)")
            accessory().padding(.trailing, 8)
        }
        .background(selected ? Color.accentColor.opacity(0.28) : .clear,
                    in: RoundedRectangle(cornerRadius: 7))
        .accessibilityAddTraits(selected ? [.isSelected] : [])
        .accessibilityIdentifier(selected ? "dataset-selected-row-\(nodeID)" : "dataset-row-\(nodeID)")
    }
}

enum DatasetBrowserNode: Equatable {
    case subject(String), crop(String), map(String, String)
    var sampleID: String? {
        switch self {
        case .subject: nil
        case .crop(let id), .map(let id, _): id
        }
    }
}
