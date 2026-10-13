import Foundation

struct MapReviewCandidate: Identifiable {
    var id: String { mapURL.path }
    let label: String
    let mapURL: URL
    let numeric: Bool
    var accessibleLabel: String { label }
}
