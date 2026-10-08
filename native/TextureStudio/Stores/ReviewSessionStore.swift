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
    var selectedGroupId: String? {
        didSet {
            if selectedGroupId != oldValue, let selectedCandidateId,
               selected?.candidates.contains(where: { $0.id == selectedCandidateId }) != true { self.selectedCandidateId = nil }
            persistSelection()
        }
    }
    var blendURL: URL?
    var error: String?
    var decisions: [String: String] = [:]
    var notes: [String: String] = [:]
    var manifestURL: URL?
    var selectedCandidateId: String? { didSet { persistSelection() } }
    private let preferences: UserDefaults
    init(preferences: UserDefaults = UserDefaults(suiteName: "org.ipde.material-tools")!) { self.preferences = preferences }
    var selected: MaterialReviewGroup? { groups.first { $0.id == selectedGroupId } }

    private var selectionKey: String? {
        manifestURL.map { "reviewSelection." + ReviewImageLoader.hash(Data($0.standardizedFileURL.path.utf8)) }
    }
    private func persistSelection() {
        guard let selectionKey else { return }
        var values: [String: String] = [:]
        if let selectedGroupId { values["group"] = selectedGroupId }
        if let selectedCandidateId, selected?.candidates.contains(where: { $0.id == selectedCandidateId }) == true {
            values["candidate"] = selectedCandidateId
        }
        preferences.set(values, forKey: selectionKey)
    }

    func restore(workspace: String) {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--review"), args.indices.contains(i + 1) {
            load(URL(fileURLWithPath: args[i + 1])); return
        }
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
            self.manifestURL = nil
            self.groups = [MaterialReviewGroup(id: "Opened maps", candidates: candidates)]
            self.selectedGroupId = self.groups.first?.id
            self.blendURL = nil
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
            ["material_id": group.id, "variants": group.candidates.map { candidate in
                var fields: [String: Any] = ["candidate_id": candidate.id, "name": candidate.label, "path": candidate.mapURL.path,
                 "numeric": candidate.numeric,
                 "sample_label": candidate.sampleLabel ?? group.id, "detail": candidate.detail ?? "", "role": candidate.role,
                 "decision": decisions[candidate.id] ?? "unreviewed", "note": notes[candidate.id] ?? ""]
                if let identity = candidate.modelIdentity {
                    fields.merge(identity.manifestFields) { _, recorded in recorded }
                }
                return fields
            }]
        }
        let review: [String: Any] = ["schema": "texture-studio-native-material-review-v1",
            "source_manifest": manifestURL?.path ?? "", "materials": materials,
            "blend_scene": blendURL?.path ?? "",
            "selected_material_id": selectedGroupId ?? "", "selected_candidate_id": selectedCandidateId ?? "",
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
                    let identity = Self.modelIdentity(variant, target: target, relativeTo: url)
                    let deepBump = (identity.modelName ?? name).localizedCaseInsensitiveContains("deepbump")
                    let role = variant["role"] as? String ?? (name == "target" ? "target" : name == "flat" ? "base" : identity.checkpointPath != nil ? "checkpoint" : deepBump ? "model" : "map")
                    let title = Self.candidateTitle(name: name, role: role, identity: identity, deepBump: deepBump)
                    var details = identity.recordedDetails
                    if role == "checkpoint", identity.architecture == nil { details.append("Architecture not recorded") }
                    if role == "target" {
                        details.insert("Dataset reference · not a model output", at: 0)
                        if let bits = material["target_original_bits"] as? Int { details.append("\(bits)-bit source · linear data") }
                    }
                    if name == "flat" { details.append("Constant height · no model inference") }
                    if deepBump { details.append("DeepBump model output · not the dataset reference") }
                    let savedDetail = Self.nonemptyString(variant["detail"])
                    let detail = savedDetail ?? (details.isEmpty ? nil : details.joined(separator: " · "))
                    return MapReviewCandidate(id: candidateId, label: title, mapURL: map, numeric: variant["numeric"] as? Bool ?? true,
                        sampleLabel: variant["sample_label"] as? String ?? id, detail: detail, role: role, modelIdentity: identity)
                }
                if !candidates.contains(where: { $0.role == "source" }), let source = material["diffuse"] as? String {
                    let map = source.hasPrefix("/") ? URL(fileURLWithPath: source) : url.deletingLastPathComponent().appendingPathComponent(source)
                    candidates.insert(MapReviewCandidate(id: "\(id)/source", label: "Source photo", mapURL: map, numeric: false,
                        sampleLabel: id, detail: map.lastPathComponent, role: "source"), at: 0)
                }
                if !candidates.isEmpty { parsed.append(MaterialReviewGroup(id: id, candidates: candidates)) }
            }
            guard !parsed.isEmpty else { throw StudioError("This manifest contains no map variants.") }
            let key = "reviewSelection." + ReviewImageLoader.hash(Data(url.standardizedFileURL.path.utf8))
            let saved = preferences.dictionary(forKey: key) as? [String: String] ?? [:]
            let requestedGroup = saved["group"] ?? manifest["selected_material_id"] as? String
            let requestedCandidate = saved["candidate"] ?? manifest["selected_candidate_id"] as? String
            manifestURL = url; groups = parsed
            selectedGroupId = parsed.contains(where: { $0.id == requestedGroup }) ? requestedGroup : parsed.first?.id
            selectedCandidateId = selected?.candidates.contains(where: { $0.id == requestedCandidate }) == true ? requestedCandidate : nil
            error = nil; decisions = parsedDecisions; notes = parsedNotes
            let nearby = url.deletingLastPathComponent().appendingPathComponent("material-quality-review.blend")
            let legacy = url.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("quality-review-2k-20261008/material-quality-review.blend")
            let explicit = (manifest["blend_scene"] as? String).flatMap { path -> URL? in
                guard !path.isEmpty else { return nil }
                return path.hasPrefix("/") ? URL(fileURLWithPath: path) : url.deletingLastPathComponent().appendingPathComponent(path)
            }
            blendURL = ([explicit].compactMap { $0 } + [nearby, legacy]).first { FileManager.default.fileExists(atPath: $0.path) }
            preferences.set(url.path, forKey: "reviewManifest")
        } catch { self.error = error.localizedDescription }
    }

    private static func nonemptyString(_ value: Any?) -> String? {
        guard let string = value as? String, !string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return string
    }

    private static func modelIdentity(_ variant: [String: Any], target: String, relativeTo manifest: URL) -> MapReviewModelIdentity {
        let checkpoint = nonemptyString(variant["checkpoint"]).map { path in
            path.hasPrefix("/") ? path : manifest.deletingLastPathComponent().appendingPathComponent(path).path
        }
        // Architecture must be recorded explicitly. Filenames and old trial
        // aliases do not establish whether the model is DINOv2, DA3 or another.
        let architecture = nonemptyString(variant["model_architecture"]) ?? nonemptyString(variant["model_summary"])
            ?? nonemptyString(variant["base_encoder"])
        return MapReviewModelIdentity(checkpointPath: checkpoint,
            checkpointSHA256: nonemptyString(variant["checkpoint_sha256"]),
            checkpointStep: variant["checkpoint_step"] as? Int, architecture: architecture,
            mapType: nonemptyString(variant["map_type"]) ?? target, modelName: nonemptyString(variant["model_name"]))
    }

    private static func candidateTitle(name: String, role: String, identity: MapReviewModelIdentity, deepBump: Bool) -> String {
        let map = identity.mapType == "height" ? "displacement" : identity.mapType ?? "map"
        if name == "target" { return "Reference \(map)" }
        if name == "flat" { return "Flat baseline · no model" }
        if deepBump { return "DeepBump · \(map)" }
        var title: String
        switch name {
        case "starting_head": title = "Starting trained \(map)"
        case "trained_2k": title = "Trained 2K \(map)"
        // Preserve native/saved descriptive labels exactly, including a run's
        // underscores. Only the known legacy aliases above need translation.
        default: title = name
        }
        if role == "checkpoint", let run = identity.runName, !title.contains(run) {
            title += " · " + run
        }
        return title
    }
}
