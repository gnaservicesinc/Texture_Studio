import AppKit
import Observation
import UniformTypeIdentifiers

struct MaterialReviewGroup: Identifiable {
    let id: String
    let candidates: [MapReviewCandidate]
}

@MainActor @Observable
final class ReviewSessionStore {
    var groups: [MaterialReviewGroup] = []
    var selectedGroupId: String?
    var blendURL: URL?
    var error: String?
    var decisions: [String: String] = [:]
    var notes: [String: String] = [:]
    var manifestURL: URL?
    var selectedCandidateId: String?
    var selected: MaterialReviewGroup? { groups.first { $0.id == selectedGroupId } }

    func restore(workspace: String) {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--review"), args.indices.contains(i + 1) {
            load(URL(fileURLWithPath: args[i + 1])); return
        }
        let preferences = UserDefaults(suiteName: "org.ipde.material-tools")!
        if let path = preferences.string(forKey: "reviewManifest"), FileManager.default.fileExists(atPath: path) {
            load(URL(fileURLWithPath: path)); return
        }
        let root = URL(fileURLWithPath: workspace)
        let path = root.appendingPathComponent("out/material-training/visual-candidates-2k-20261008/review-manifest.json")
        if FileManager.default.fileExists(atPath: path.path) { load(path) }
    }
    func chooseManifest() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.title = "Open material review manifest"
        panel.begin { response in if response == .OK, let url = panel.url { self.load(url) } }
    }
    func chooseMaps() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
        panel.title = "Open original maps at full resolution"
        panel.begin { response in
            guard response == .OK else { return }
            let candidates = panel.urls.map { MapReviewCandidate(id: $0.path, label: $0.lastPathComponent, mapURL: $0, numeric: true) }
            self.groups = [MaterialReviewGroup(id: "Opened maps", candidates: candidates)]
            self.selectedGroupId = self.groups.first?.id
            self.blendURL = nil
            self.manifestURL = nil
            self.decisions = [:]; self.notes = [:]; self.selectedCandidateId = nil
        }
    }
    func saveReview() {
        let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "material-review.json"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do {
                try self.writeReview(to: url)
            } catch { self.error = error.localizedDescription }
        }
    }
    func writeReview(to url: URL) throws {
        let materials: [[String: Any]] = groups.map { group in
            ["material_id": group.id, "variants": group.candidates.map {
                ["candidate_id": $0.id, "name": $0.label, "path": $0.mapURL.path,
                 "numeric": $0.numeric,
                 "decision": decisions[$0.id] ?? "unreviewed", "note": notes[$0.id] ?? ""] as [String: Any]
            }]
        }
        let review: [String: Any] = ["schema": "texture-studio-native-material-review-v1",
            "source_manifest": manifestURL?.path ?? "", "materials": materials,
            "blend_scene": blendURL?.path ?? "",
            "automatic_model_promotion": false]
        try JSONSerialization.data(withJSONObject: review, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    }
    func load(_ url: URL) {
        do {
            guard let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
                  let materials = manifest["materials"] as? [[String: Any]] else { throw StudioError("Choose a material review manifest containing materials and variants.") }
            var parsed: [MaterialReviewGroup] = []
            var parsedDecisions: [String: String] = [:]
            var parsedNotes: [String: String] = [:]
            let target = manifest["comparison_target"] as? String ?? "height"
            for material in materials {
                guard let id = material["material_id"] as? String, let variants = material["variants"] as? [[String: Any]] else { continue }
                let candidates = variants.compactMap { variant -> MapReviewCandidate? in
                    guard let name = variant["name"] as? String,
                          let path = variant[target] as? String ?? variant["path"] as? String else { return nil }
                    let map = path.hasPrefix("/") ? URL(fileURLWithPath: path) : url.deletingLastPathComponent().appendingPathComponent(path)
                    let candidateId = variant["candidate_id"] as? String ?? "\(id)/\(name)"
                    if let decision = variant["decision"] as? String { parsedDecisions[candidateId] = decision }
                    if let note = variant["note"] as? String { parsedNotes[candidateId] = note }
                    return MapReviewCandidate(id: candidateId, label: name, mapURL: map, numeric: variant["numeric"] as? Bool ?? true)
                }
                if !candidates.isEmpty { parsed.append(MaterialReviewGroup(id: id, candidates: candidates)) }
            }
            guard !parsed.isEmpty else { throw StudioError("This manifest contains no map variants.") }
            groups = parsed; selectedGroupId = parsed.first?.id; error = nil; manifestURL = url
            decisions = parsedDecisions; notes = parsedNotes; selectedCandidateId = nil
            let nearby = url.deletingLastPathComponent().appendingPathComponent("material-quality-review.blend")
            let legacy = url.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("quality-review-2k-20261008/material-quality-review.blend")
            let explicit = (manifest["blend_scene"] as? String).flatMap { path -> URL? in
                guard !path.isEmpty else { return nil }
                return path.hasPrefix("/") ? URL(fileURLWithPath: path) : url.deletingLastPathComponent().appendingPathComponent(path)
            }
            blendURL = ([explicit].compactMap { $0 } + [nearby, legacy]).first { FileManager.default.fileExists(atPath: $0.path) }
            UserDefaults(suiteName: "org.ipde.material-tools")!.set(url.path, forKey: "reviewManifest")
        } catch { self.error = error.localizedDescription }
    }
}
