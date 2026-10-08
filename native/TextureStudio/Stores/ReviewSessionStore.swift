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
                 "sample_label": $0.sampleLabel ?? group.id, "detail": $0.detail ?? "", "role": $0.role,
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
                var candidates = variants.compactMap { variant -> MapReviewCandidate? in
                    guard let name = variant["name"] as? String,
                          let path = variant[target] as? String ?? variant["path"] as? String else { return nil }
                    let map = path.hasPrefix("/") ? URL(fileURLWithPath: path) : url.deletingLastPathComponent().appendingPathComponent(path)
                    let candidateId = variant["candidate_id"] as? String ?? "\(id)/\(name)"
                    if let decision = variant["decision"] as? String { parsedDecisions[candidateId] = decision }
                    if let note = variant["note"] as? String { parsedNotes[candidateId] = note }
                    let role = variant["role"] as? String ?? (name == "target" ? "target" : name == "flat" ? "base" : variant["checkpoint"] != nil ? "checkpoint" : "map")
                    let title = name == "target" ? "Reference \(target == "height" ? "displacement" : target)" : name == "flat" ? "Flat baseline" : name.replacingOccurrences(of: "_", with: " ")
                    var details: [String] = []
                    if let checkpoint = variant["checkpoint"] as? String {
                        let file = URL(fileURLWithPath: checkpoint)
                        details.append(file.deletingLastPathComponent().lastPathComponent + " · " + file.lastPathComponent)
                    }
                    if let step = variant["checkpoint_step"] as? Int { details.append("Step \(step.formatted())") }
                    if let checksum = variant["checkpoint_sha256"] as? String { details.append("SHA256 \(checksum.prefix(12))") }
                    if name == "target", let bits = material["target_original_bits"] as? Int { details.append("\(bits)-bit source · linear data") }
                    if name == "flat" { details.append("No surface relief; comparison reference") }
                    let detail = variant["detail"] as? String ?? (details.isEmpty ? nil : details.joined(separator: " · "))
                    return MapReviewCandidate(id: candidateId, label: title, mapURL: map, numeric: variant["numeric"] as? Bool ?? true,
                        sampleLabel: variant["sample_label"] as? String ?? id, detail: detail, role: role)
                }
                if !candidates.contains(where: { $0.role == "source" }), let source = material["diffuse"] as? String {
                    let map = source.hasPrefix("/") ? URL(fileURLWithPath: source) : url.deletingLastPathComponent().appendingPathComponent(source)
                    candidates.insert(MapReviewCandidate(id: "\(id)/source", label: "Source photo", mapURL: map, numeric: false,
                        sampleLabel: id, detail: map.lastPathComponent, role: "source"), at: 0)
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
