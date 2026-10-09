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

    func removeMissingSource(_ url: URL) {
        do { _ = try ReviewImageLoader.readSource(url); return }
        catch ReviewImageError.missingSource { }
        catch { return }
        groups = groups.compactMap { group in
            let kept = group.candidates.filter { $0.mapURL.standardizedFileURL != url.standardizedFileURL }
            return kept.isEmpty ? nil : MaterialReviewGroup(id: group.id, candidates: kept)
        }
        if selected == nil { selectedGroupId = groups.first?.id }
        if selected?.candidates.contains(where: { $0.id == selectedCandidateId }) != true { selectedCandidateId = nil }
    }

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
            self.openMaps(panel.urls)
        }
    }
    func openMaps(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        let candidates = urls.map { MapReviewCandidate(id: $0.path, label: $0.lastPathComponent, mapURL: $0, numeric: true) }
        manifestURL = nil
        preferences.removeObject(forKey: "reviewManifest")
        blendURL = nil
        decisions = [:]; notes = [:]; selectedCandidateId = nil
        error = nil
        groups = [MaterialReviewGroup(id: "Opened maps", candidates: candidates)]
        selectedGroupId = groups.first?.id
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
                if let identity = candidate.sourceIdentity {
                    fields.merge(identity.manifestFields) { _, recorded in recorded }
                }
                if let transform = candidate.displayTransform {
                    fields.merge(transform.manifestFields) { _, recorded in recorded }
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
        manifestURL = url
        preferences.set(url.path, forKey: "reviewManifest")
        persistSelection()
        error = nil
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
                guard let id = material["material_id"] as? String, var variants = material["variants"] as? [[String: Any]] else { continue }
                let hasReference = variants.contains { Self.isReferenceVariant($0) && Self.mapPath($0, target: target) != nil }
                if !hasReference, let path = Self.nonemptyString(material["reference_\(target)"])
                    ?? Self.nonemptyString(material[target]) ?? (target == "height" ? Self.nonemptyString(material["reference_exr"]) : nil) {
                    variants.insert(["name": "target", "role": "target", target: path], at: 0)
                }
                var candidates = try variants.compactMap { variant -> MapReviewCandidate? in
                    let variantTarget = variant["map_type"] as? String ?? target
                    guard let name = variant["name"] as? String,
                          let path = Self.mapPath(variant, target: variantTarget) else { return nil }
                    let map = path.hasPrefix("/") ? URL(fileURLWithPath: path) : url.deletingLastPathComponent().appendingPathComponent(path)
                    let candidateId = variant["candidate_id"] as? String ?? "\(id)/\(name)"
                    if let decision = variant["decision"] as? String { parsedDecisions[candidateId] = decision }
                    if let note = variant["note"] as? String { parsedNotes[candidateId] = note }
                    let identity = Self.modelIdentity(variant, target: variantTarget, relativeTo: url)
                    let recordedRole = variant["role"] as? String
                    let role = recordedRole == "reference" ? "target" : recordedRole ?? "map"
                    let title = Self.candidateTitle(name: name, role: role, identity: identity)
                    let transform = ["target", "diffuse"].contains(role) ? try Self.displayTransform(variant, mapType: role == "diffuse" ? "input" : variantTarget) : nil
                    let sourceIdentity = role == "target" ? Self.sourceIdentity(variant.merging(["source_path": variant["source_path"] ?? material["source_path"] ?? map.path]) { recorded, _ in recorded }, material: material, relativeTo: url) : nil
                    var details = identity.recordedDetails
                    if role == "checkpoint", identity.architecture == nil { details.append("Architecture not recorded") }
                    if role == "target" {
                        details.insert("Real source map · not a model output", at: 0)
                        details.append(contentsOf: sourceIdentity?.recordedDetails ?? [])
                    }
                    if role == "base" { details.insert("Base model prediction · not the real source map", at: 0) }
                    if role == "checkpoint" { details.insert("Trained model prediction · not the real source map", at: 0) }
                    if let transform { details.append("Displayed on the \(transform.size) × \(transform.size) training grid; original-source exports retain original bytes") }
                    let savedDetail = Self.nonemptyString(variant["detail"])
                    // A saved freeform description cannot hide the structured
                    // role, source or checkpoint identity. Keep this merge
                    // idempotent across repeated save/reopen cycles.
                    let mandatory = details.filter { savedDetail?.range(of: $0, options: .caseInsensitive) == nil }
                    let combined = mandatory + [savedDetail].compactMap { $0 }
                    let detail = combined.isEmpty ? nil : combined.joined(separator: " · ")
                    return MapReviewCandidate(id: candidateId, label: title, mapURL: map, numeric: variant["numeric"] as? Bool ?? true,
                        sampleLabel: variant["sample_label"] as? String ?? id, detail: detail, role: role, modelIdentity: identity,
                        sourceIdentity: sourceIdentity, displayTransform: transform)
                }
                if !candidates.contains(where: { $0.role == "diffuse" }), let source = material["diffuse"] as? String {
                    let map = source.hasPrefix("/") ? URL(fileURLWithPath: source) : url.deletingLastPathComponent().appendingPathComponent(source)
                    var transform: MapReviewDisplayTransform?
                    if let size = material["diffuse_native_size"] as? Int {
                        transform = try Self.displayTransform(["native_dimensions": [size, size],
                            "source_sha256": material["diffuse_source_sha256"] as Any,
                            "source_resize_algorithm": material["diffuse_resize_algorithm"] as Any,
                            "source_crop_rectangle": material["diffuse_source_crop_rectangle"] as Any], mapType: "input")
                    }
                    candidates.insert(MapReviewCandidate(id: "\(id)/diffuse", label: "Diffuse · model input", mapURL: map, numeric: false,
                        sampleLabel: id, detail: transform.map { "Exact \($0.size) × \($0.size) training grid reconstructed from original source" } ?? map.lastPathComponent,
                        role: "diffuse", displayTransform: transform), at: 0)
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
            let explicit = (manifest["blend_scene"] as? String).flatMap { path -> URL? in
                guard !path.isEmpty else { return nil }
                return path.hasPrefix("/") ? URL(fileURLWithPath: path) : url.deletingLastPathComponent().appendingPathComponent(path)
            }
            blendURL = ([explicit].compactMap { $0 } + [nearby]).first { FileManager.default.fileExists(atPath: $0.path) }
            preferences.set(url.path, forKey: "reviewManifest")
        } catch { self.error = error.localizedDescription }
    }

    private static func nonemptyString(_ value: Any?) -> String? {
        guard let string = value as? String, !string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return string
    }
    private static func displayTransform(_ fields: [String: Any], mapType: String) throws -> MapReviewDisplayTransform? {
        guard fields["source_resize_algorithm"] != nil || fields["native_dimensions"] != nil else { return nil }
        guard let dimensions = fields["native_dimensions"] as? [Int], dimensions.count == 2,
              dimensions[0] == dimensions[1], (1...16384).contains(dimensions[0]),
              let hash = fields["source_sha256"] as? String, hash.count == 64, hash.allSatisfy(\.isHexDigit),
              fields["source_resize_algorithm"] as? String == MapReviewDisplayTransform.exactCrop else {
            throw StudioError("A training review needs its original source checksum and exact crop metadata. Regenerate this review with the material trainer.")
        }
        let convention = (fields["source_normal_convention"] as? String ?? "opengl").lowercased()
        guard ["input", "height", "roughness", "normal"].contains(mapType), ["opengl", "directx"].contains(convention) else {
            throw StudioError("The review's map type or normal convention is unsupported.")
        }
        let rectangle = fields["source_crop_rectangle"] as? [Int]
        if let rectangle {
            guard rectangle.count == 4, rectangle[0] >= 0, rectangle[1] >= 0,
                  rectangle[2] == dimensions[0], rectangle[3] == dimensions[1] else {
                throw StudioError("The review needs an exact native pixel crop matching its training grid.")
            }
        }
        return MapReviewDisplayTransform(size: dimensions[0], sourceSHA256: hash.lowercased(),
            algorithm: MapReviewDisplayTransform.exactCrop, mapType: mapType, normalConvention: convention, cropRectangle: rectangle)
    }

    private static func isReferenceVariant(_ variant: [String: Any]) -> Bool {
        if let role = variant["role"] as? String { return role == "target" || role == "reference" }
        return false
    }

    private static func mapPath(_ variant: [String: Any], target: String) -> String? {
        if let raw = nonemptyString(variant["raw_path"]) { return raw }
        if target == "height", isReferenceVariant(variant),
           let reference = nonemptyString(variant["reference_height"]) ?? nonemptyString(variant["reference_exr"]) {
            return reference
        }
        return nonemptyString(variant[target]) ?? nonemptyString(variant["path"])
            ?? (target == "height" ? nonemptyString(variant["height_exr"]) : nil)
    }

    private static func modelIdentity(_ variant: [String: Any], target: String, relativeTo manifest: URL) -> MapReviewModelIdentity {
        let checkpoint = nonemptyString(variant["checkpoint"]).map { path in
            path.hasPrefix("/") ? path : manifest.deletingLastPathComponent().appendingPathComponent(path).path
        }
        // A model architecture is recorded explicitly, never inferred from a filename.
        let architecture = nonemptyString(variant["model_architecture"]) ?? nonemptyString(variant["model_summary"])
            ?? nonemptyString(variant["base_encoder"])
        return MapReviewModelIdentity(checkpointPath: checkpoint,
            checkpointSHA256: nonemptyString(variant["checkpoint_sha256"]),
            checkpointStep: variant["checkpoint_step"] as? Int, architecture: architecture,
            mapType: nonemptyString(variant["map_type"]) ?? target, modelName: nonemptyString(variant["model_name"]))
    }

    private static func sourceIdentity(_ variant: [String: Any], material: [String: Any], relativeTo manifest: URL) -> MapReviewSourceIdentity {
        let path = nonemptyString(variant["source_path"] ?? material["source_path"]).map { value in
            value.hasPrefix("/") ? value : manifest.deletingLastPathComponent().appendingPathComponent(value).path
        }
        func integers(_ key: String, count: Int) -> [Int]? {
            guard let values = variant[key] as? [Int] ?? material[key] as? [Int], values.count == count,
                  values.allSatisfy({ $0 >= 0 }), values.suffix(2).allSatisfy({ $0 > 0 }) else { return nil }
            return values
        }
        return MapReviewSourceIdentity(path: path,
            sha256: nonemptyString(variant["source_sha256"] ?? material["source_sha256"]),
            url: nonemptyString(variant["source_url"] ?? material["source_url"]),
            bits: variant["source_bits"] as? Int ?? material["source_bits"] as? Int ?? material["target_original_bits"] as? Int,
            cropRectangle: integers("source_crop_rectangle", count: 4), pixelDimensions: integers("source_pixel_dimensions", count: 2))
    }

    private static func candidateTitle(name: String, role: String, identity: MapReviewModelIdentity) -> String {
        let map = identity.mapType == "height" ? "displacement" : identity.mapType ?? "map"
        if role == "target" { return "Source \(map) · reference" }
        var title = name
        if role == "checkpoint", let run = identity.runName, !title.contains(run) { title += " · " + run }
        return title
    }
}
