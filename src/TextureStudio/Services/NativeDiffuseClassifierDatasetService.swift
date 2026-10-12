import CryptoKit
import Darwin
import Foundation
import ImageIO

/// A reviewable Create ML folder dataset, independent of material crop refresh.
/// It copies original diffuse bytes and never enters an image rendering pipeline.
enum NativeDiffuseClassifierDatasetService {
    static let defaultDestination = URL(fileURLWithPath: "/opt/ipde/classifer_ds", isDirectory: true)
    static let command = "refresh-classifier-dataset"
    static let schema = "texture-studio-diffuse-classifier-dataset-v1"

    struct TextureClass: Codable, Sendable {
        let id: String
        let name: String
        let description: String
    }
    static let classes: [TextureClass] = [
        .init(id: "bricks", name: "Bricks", description: "Brick masonry, including painted and weathered bricks."),
        .init(id: "concrete", name: "Cement Concrete Pavement", description: "Cement, concrete, asphalt and continuous pavement."),
        .init(id: "bark", name: "Bark", description: "Tree bark, distinct from worked wood surfaces."),
        .init(id: "wood", name: "Wood", description: "Planks, boards, beams, panels and woven wood."),
        .init(id: "small_particles", name: "Small Particles", description: "Fine dirt, flour, sugar, snow, sand and soil."),
        .init(id: "ground", name: "Ground", description: "Leaves, grass and mixed organic ground cover."),
        .init(id: "medium_particles", name: "Medium Particles", description: "Pebbles, gravel, loose stones and wood chips."),
        .init(id: "metal", name: "Metal", description: "Metal sheets, plates, grids and rusted metal."),
        .init(id: "rock", name: "Rock Cliff Marble", description: "Solid rock, cliff faces, marble and stone walls."),
        .init(id: "tiles", name: "Tiles and Pavers", description: "Tiles, pavers and regular laid stone flooring."),
        .init(id: "fabric", name: "Fabric", description: "Woven cloth, knit, leather and other flexible textiles."),
        .init(id: "paper", name: "Paper", description: "Paper, cardboard and related fiber sheets."),
        .init(id: "wall_finish", name: "Paint Wallpaper Drywall Stucco", description: "Paint, wallpaper, plaster, drywall and stucco."),
        .init(id: "plastic", name: "Plastic Rubber Clay", description: "Plastic, rubber, linoleum and unlaid clay."),
        .init(id: "other_diffuse", name: "Other Diffuse Textures", description: "User-populated, evenly lit material textures that do not fit the other types."),
        .init(id: "other_fallback", name: "Other Fallback", description: "User-populated objects, rooms, people and other images for a fallback model.")
    ]

    struct ProviderMetadata: Codable, Sendable {
        let name: String
        let tags: [String]
        let categories: [String]
        let pageURL: String?
    }
    struct Source: Sendable {
        let family: String
        let variant: String
        let url: URL
        let width: Int
        let height: Int
        let metadata: ProviderMetadata?
        var resolutionAware = false
        var directoryScope: String? = nil
        var familyVariantID: String { family.lowercased() + "|" + variant.lowercased() }
        var logicalID: String { familyVariantID + (directoryScope.map { "|folder_" + $0 } ?? "") }
        var id: String { logicalID + (resolutionAware ? "|native_\(width)x\(height)" : "") }
    }
    struct Discovery: Sendable {
        let sources: [Source]
        let minimumSize: Int
        let discoveredCount: Int
        let eligibleCount: Int
        let undersizedCount: Int
        let unreadableCount: Int
        let warnings: [String]
    }
    struct Assignment: Sendable {
        let classID: String?
        let confidence: String
        let evidence: [String]
    }
    struct Entry: Codable, Sendable {
        let id: String
        let family: String
        let variant: String
        let sourcePath: String
        let sourceSHA256: String
        let width: Int
        let height: Int
        let filename: String
        var suggestedClass: String?
        var confidence: String
        var evidence: [String]
        var providerMetadata: ProviderMetadata?
        /// Stable semantic hint for future specialization. Older manifests
        /// decode without this optional field; filenames keep their identity.
        var subtype: String?
        var currentPath: String?
        var currentClass: String?
        var status: String
        var needsReview: Bool
        /// File identity keeps a user rename distinguishable from deletion,
        /// even when several originals contain exactly the same pixels.
        var currentFileIdentity: NativeSourceFileTags.Identity? = nil
    }
    struct Manifest: Codable, Sendable {
        let schema: String
        let updatedAt: String
        let datasetPath: String
        let selectionPolicy: String
        var minimumNativeSize: Int? = nil
        let classes: [TextureClass]
        var entries: [Entry]
    }
    struct TrainingViewFile: Codable, Sendable {
        let relativePath: String
        let sourcePath: String
        let sha256: String
        let bytes: Int
    }
    struct TrainingViewManifest: Codable, Sendable {
        let schema: String
        let datasetPath: String
        let trainingPath: String
        let curationManifestSHA256: String
        let updatedAt: String
        let classes: [String]
        let files: [TrainingViewFile]
    }
    private struct TrainingViewResult: Sendable {
        let path: URL
        let imageCount: Int
        let classCount: Int
        let ready: Bool
        let warnings: [String]
        var undersizedCount = 0
        var unreadableCount = 0
        var retiredCount = 0
    }
    private static let trainingViewSchema = "texture-studio-diffuse-classifier-createml-view-v1"
    private static let trainingViewManifestName = ".classifier-training-view.json"
    private typealias Object = [String: Any]

    /// Labels use local source metadata and prior curation. Refresh never waits
    /// for a network catalog before reading the user's images.
    static func run(arguments: [String], providerMetadataData: Data? = nil,
                    onEvent: @escaping @Sendable (String) -> Void = { _ in }) async throws -> String? {
        guard arguments.first == command else { return nil }
        func value(_ flag: String) -> String? {
            arguments.firstIndex(of: flag).flatMap { arguments.indices.contains($0 + 1) ? arguments[$0 + 1] : nil }
        }
        let destination = value("--destination").map { URL(fileURLWithPath: $0, isDirectory: true) } ?? defaultDestination
        let dataset = value("--dataset").map { URL(fileURLWithPath: $0) }
        let folder = value("--source-folder").map { URL(fileURLWithPath: $0, isDirectory: true) }
        guard dataset != nil || folder != nil else { throw StudioError("Choose the material dataset or original diffuse source folder.") }
        let minimumSize: Int?
        if let requested = value("--size") {
            guard let size = Int(requested), size > 0, size <= 32768 else { throw StudioError("Choose a positive classifier image resolution up to 32768 pixels.") }
            minimumSize = size
        } else { minimumSize = nil }
        var warnings: [String] = [], provider: [String: ProviderMetadata] = [:]
        try Task.checkCancellation()
        if providerMetadataData != nil || value("--provider-tags") != nil {
            onEvent(try event("Reading local material tags…", completed: 0, total: 0))
            do {
                let bytes = try providerMetadataData ?? Data(contentsOf: URL(fileURLWithPath: value("--provider-tags")!))
                provider = try parseProviderMetadata(bytes)
            } catch {
                if Task.isCancelled { throw CancellationError() }
                warnings.append("Local provider tags unavailable; filenames and saved labels remain reviewable: \(error.localizedDescription)")
            }
        }
        let metadata = provider, initialWarnings = warnings
        let operation = Task.detached(priority: .userInitiated) {
            onEvent(try event("Reading local diffuse image headers…", completed: 0, total: 0))
            let discovery = try discoverReport(dataset: dataset, sourceFolder: folder, minimumSize: minimumSize)
            let output = try refresh(sources: discovery.sources, destination: destination, providerMetadata: metadata,
                                    warnings: initialWarnings + discovery.warnings, minimumSize: discovery.minimumSize, onEvent: onEvent)
            var result = try object(Data(output.utf8))
            result["discovered_count"] = discovery.discoveredCount
            result["eligible_count"] = discovery.eligibleCount
            result["undersized_count"] = discovery.undersizedCount
            result["unreadable_count"] = discovery.unreadableCount + (result["unreadable_count"] as? Int ?? 0)
            return try json(result)
        }
        return try await withTaskCancellationHandler { try await operation.value } onCancel: { operation.cancel() }
    }

    static func parseProviderMetadata(_ bytes: Data) throws -> [String: ProviderMetadata] {
        let catalog = try object(bytes)
        return catalog.reduce(into: [:]) { result, element in
            guard let item = element.value as? Object else { return }
            let categories = item["categories"] as? [String] ?? []
            // Provider category is useful evidence; descriptions are prose and
            // can mention contrasting materials that are absent from the image.
            let category = (item["category"] as? String).map { [$0] } ?? []
            result[element.key.lowercased()] = ProviderMetadata(name: item["name"] as? String ?? element.key,
                tags: item["tags"] as? [String] ?? [], categories: categories + category,
                pageURL: "https://polyhaven.com/a/" + element.key)
        }
    }

    /// Semantic name takes precedence over broad provider categories (e.g.
    /// wood-chip paths are particles and marble tiles are laid tiles).
    static func classify(family: String, filename: String, metadata: ProviderMetadata?) -> Assignment {
        // These mixed surfaces were reviewed against the original diffuse
        // images. Keep the evidence visible so the user can change the label.
        let reviewed: [String: (String, String)] = [
            "baseball_playground": ("small_particles", "Reviewed original image: fine sand and dirt."),
            "excavated_soil_wall": ("small_particles", "Reviewed original image: compact dirt/soil; review whether this belongs with Ground."),
            "old_linoleum_flooring_01": ("plastic", "Reviewed original image: linoleum with a tile-like pattern; material identity takes precedence over pattern."),
            "wood_inlaid_stone_wall": ("rock", "Reviewed original image: mostly stone masonry with an inlaid wooden stripe; mixed material."),
            "red_mud_stones": ("ground", "Reviewed original image: mostly fine dirt with scattered rocks; mixed ground surface."),
            "asphalt_snow": ("concrete", "Reviewed original image: asphalt covered with some snow; mixed material."),
            "leaf_scattered_gravel": ("ground", "Mixed leaf-covered gravel; review the dominant visible surface.")
        ]
        let familyKey = normalized(family).replacingOccurrences(of: " ", with: "_")
        if let (id, evidence) = reviewed[familyKey] {
            return Assignment(classID: id, confidence: "reviewed_mixed_material", evidence: [evidence])
        }
        let primary = normalized([family, URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent,
                                  metadata?.name ?? ""].joined(separator: " "))
        let secondary = normalized(((metadata?.categories ?? []) + (metadata?.tags ?? [])).joined(separator: " "))
        let rules: [(String, [String])] = [
            ("bricks", ["brick", "bricks", "brickwork"]),
            ("tiles", ["tile", "tiles", "tiling", "paver", "pavers", "paving", "cobblestone", "cobblestones"]),
            ("medium_particles", ["wood chip", "wood chips", "pebble", "pebbles", "gravel", "rocks ground", "red mud stones"]),
            ("ground", ["leaves", "leaf", "grass", "forest floor", "forest ground", "forest leaves", "mud forest", "ground cover"]),
            ("small_particles", ["flour", "sugar", "sand", "snow", "soil", "dirt", "mud", "furrows", "baseball playground"]),
            ("fabric", ["fabric", "cloth", "textile", "denim", "cotton", "wool", "satin", "jersey", "leather", "corduroy", "boucle", "carpet"]),
            ("paper", ["paper", "cardboard", "card board"]),
            ("wall_finish", ["plaster", "plastered", "stucco", "wallpaper", "drywall", "paint"]),
            ("plastic", ["plastic", "rubber", "linoleum", "clay"]),
            ("metal", ["metal", "steel", "iron", "aluminum", "aluminium", "copper", "brass"]),
            ("concrete", ["cement", "concrete", "asphalt", "pavement"]),
            ("bark", ["bark"]),
            ("wood", ["wood", "wooden", "plank", "planks", "board", "boards", "beam", "beams", "panel", "panels", "timber", "pine", "cedar", "bamboo", "wicker", "hinoki", "oriented strand board"]),
            ("rock", ["rock", "rocks", "cliff", "marble", "granite", "sandstone", "stone", "stones"])
        ]
        func matches(_ text: String, _ term: String) -> Bool { (" " + text + " ").contains(" " + term + " ") }
        for (id, terms) in rules {
            if let term = terms.first(where: { matches(primary, $0) }) {
                return Assignment(classID: id, confidence: "filename_or_name", evidence: ["Name contains '\(term)'", "\(family) · \(metadata?.name ?? filename)"])
            }
        }
        for (id, terms) in rules {
            if let term = terms.first(where: { matches(secondary, $0) }) {
                return Assignment(classID: id, confidence: "provider_tags_review", evidence: ["Provider category/tag contains '\(term)'", secondary])
            }
        }
        return Assignment(classID: nil, confidence: "unclassified", evidence: ["No reliable material-type name or tag; inspect the original image."])
    }

    /// Infer only named surface forms or materials. Generic wood/stone labels
    /// do not imply planks, boards or another more specific construction.
    static func subtype(family: String, filename: String, metadata: ProviderMetadata?, classID: String?) -> String? {
        let rules: [String: [(String, [String])]] = [
            "wood": [("oriented_strand_board", ["oriented strand board", "osb"]),
                ("planks", ["plank", "planks", "floorboard", "floorboards"]), ("boards", ["board", "boards"]),
                ("beams", ["beam", "beams"]), ("panels", ["panel", "panels", "paneled", "panelled"]),
                ("bamboo", ["bamboo"]), ("wicker", ["wicker"]), ("shutters", ["shutter", "shutters"]),
                ("doors", ["door", "doors"]), ("barrels", ["barrel", "barrels"]), ("tables", ["table", "tables"]),
                ("flooring", ["floor", "flooring"])],
            "bark": [("pine", ["pine"]), ("cedar", ["cedar"]), ("maple", ["maple"]), ("birch", ["birch"]), ("oak", ["oak"])],
            "concrete": [("asphalt", ["asphalt"]), ("concrete", ["concrete"]), ("cement", ["cement"])],
            "small_particles": [("flour", ["flour"]), ("sugar", ["sugar"]), ("snow", ["snow"]),
                ("sand", ["sand"]), ("soil", ["soil"]), ("dirt", ["dirt"]), ("mud", ["mud"])],
            "ground": [("leaf_litter", ["leaf", "leaves"]), ("grass", ["grass"]), ("forest_ground", ["forest floor", "forest ground"])],
            "medium_particles": [("wood_chips", ["wood chip", "wood chips"]), ("pebbles", ["pebble", "pebbles"]), ("gravel", ["gravel"])],
            "metal": [("grid", ["grid"]), ("sheet", ["sheet", "sheets"]), ("plate", ["plate", "plates"])],
            "rock": [("marble", ["marble"]), ("granite", ["granite"]), ("sandstone", ["sandstone"]), ("cliff", ["cliff"])],
            "tiles": [("pavers", ["paver", "pavers", "paving"]), ("terracotta", ["terracotta"]), ("marble", ["marble"]), ("granite", ["granite"])],
            "fabric": [("leather", ["leather"]), ("denim", ["denim"]), ("cotton", ["cotton"]),
                ("wool", ["wool"]), ("satin", ["satin"]), ("corduroy", ["corduroy"])],
            "wall_finish": [("stucco", ["stucco"]), ("plaster", ["plaster", "plastered"]),
                ("wallpaper", ["wallpaper"]), ("drywall", ["drywall"])],
            "plastic": [("linoleum", ["linoleum"]), ("rubber", ["rubber"]), ("clay", ["clay"]), ("plastic", ["plastic"])]
        ]
        guard let classID, let candidates = rules[classID] else { return nil }
        let name = normalized([family, URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent,
                               metadata?.name ?? ""].joined(separator: " "))
        let tags = normalized(((metadata?.tags ?? []) + (metadata?.categories ?? [])).joined(separator: " "))
        for text in [name, tags] {
            for (id, terms) in candidates where terms.contains(where: { (" " + text + " ").contains(" " + $0 + " ") }) { return id }
        }
        return nil
    }

    /// Select one original tier that supplies the chosen grid. Repeated tiers
    /// would leak the same scene into Create ML's automatic validation split.
    static func discover(dataset: URL?, sourceFolder: URL?, minimumSize: Int? = nil) throws -> [Source] {
        try discoverReport(dataset: dataset, sourceFolder: sourceFolder, minimumSize: minimumSize).sources
    }

    static func discoverReport(dataset: URL?, sourceFolder: URL?, minimumSize requestedSize: Int? = nil) throws -> Discovery {
        if let dataset {
            let root = dataset.pathExtension.lowercased() == "json" ? dataset.deletingLastPathComponent() : dataset
            let manifest = root.appendingPathComponent("dataset.json")
            if !FileManager.default.fileExists(atPath: manifest.path) {
                return try discoverReport(dataset: nil, sourceFolder: root, minimumSize: requestedSize)
            }
            if let index = try? object(Data(contentsOf: manifest)), let folder = index["source_folder_dataset_path"] as? String {
                return try discoverReport(dataset: nil, sourceFolder: URL(fileURLWithPath: folder),
                    minimumSize: requestedSize ?? index["crop_size"] as? Int ?? index["training_size"] as? Int)
            }
        }
        var candidates: [Source] = []
        var minimumSize = requestedSize ?? 2048, seen = Set<String>()
        var discovered = 0, undersized = 0, unreadable = 0, warnings: [String] = []
        func candidate(_ file: URL, family: String, variant: String, metadata: ProviderMetadata?) throws {
            try Task.checkCancellation()
            let url = file.resolvingSymlinksInPath().standardizedFileURL
            if let role = NativeSourceFileTags.read(url)?.role, role != "input" { return }
            guard seen.insert(url.path).inserted else { return }
            discovered += 1
            do {
                let dimensions = try nativeDimensions(url)
                guard dimensions.0 >= minimumSize, dimensions.1 >= minimumSize else { undersized += 1; return }
                candidates.append(Source(family: family, variant: variant, url: url,
                    width: dimensions.0, height: dimensions.1, metadata: metadata, resolutionAware: true))
            } catch {
                if Task.isCancelled { throw CancellationError() }
                unreadable += 1
                warnings.append("Could not read original diffuse image \(url.path): \(error.localizedDescription)")
            }
        }
        if let dataset {
            var root = dataset.pathExtension.lowercased() == "json" ? dataset.deletingLastPathComponent() : dataset
            root = root.resolvingSymlinksInPath().standardizedFileURL
            var index = try object(Data(contentsOf: root.appendingPathComponent("dataset.json")))
            if requestedSize == nil { minimumSize = index["crop_size"] as? Int ?? index["training_size"] as? Int ?? 2048 }
            if let lineage = index["native_size_preparation"] as? Object, let original = lineage["source_dataset_path"] as? String {
                root = URL(fileURLWithPath: original).resolvingSymlinksInPath().standardizedFileURL
                index = try object(Data(contentsOf: root.appendingPathComponent("dataset.json")))
            }
            guard let entries = index["samples"] as? [Object] else { throw StudioError("Material dataset source inventory is missing.") }
            for entry in entries {
                try Task.checkCancellation()
                guard let relative = entry["path"] as? String else { throw StudioError("Material sample source path is missing.") }
                let sampleURL = try contained(root, relative).appendingPathComponent("sample.json")
                let sample: Object
                do { sample = try object(Data(contentsOf: sampleURL)) }
                catch {
                    unreadable += 1
                    warnings.append("Could not read material source registration \(sampleURL.path): \(error.localizedDescription)")
                    continue
                }
                guard !["excluded", "rejected"].contains(sample["status"] as? String ?? "") else { continue }
                let family = sample["source_family_id"] as? String ?? sample["asset_family_id"] as? String ?? sample["name"] as? String ?? "Material"
                let maps = sample["map_metadata"] as? [String: Object] ?? [:]
                let variants = sample["input_variants"] as? [Object] ?? maps["input"].map { [$0] } ?? []
                for map in variants {
                    let source = map["source"] as? Object ?? [:]
                    guard let path = source["path"] as? String, path.hasPrefix("/") else {
                        unreadable += 1
                        warnings.append("Original diffuse path is missing from \(sampleURL.path). Its registration remains preserved.")
                        continue
                    }
                    let url = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
                    let parsed = parsedDiffuse(url)
                    let meta = ProviderMetadata(name: sample["name"] as? String ?? family,
                        tags: source["tags"] as? [String] ?? [], categories: source["categories"] as? [String] ?? [],
                        pageURL: source["published_url"] as? String)
                    let recordedVariant = map["variant_id"] as? String ?? parsed?.1 ?? "color_1"
                    let variant = recordedVariant == "color_default" ? "color_1" : recordedVariant
                    try candidate(url, family: family, variant: variant, metadata: meta)
                }
            }
        }
        if let sourceFolder {
            let folder = sourceFolder.resolvingSymlinksInPath().standardizedFileURL
            guard let files = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles]) else {
                throw StudioError("Cannot read the original diffuse source folder.")
            }
            for case let file as URL in files {
                try Task.checkCancellation()
                let state = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                if state.isSymbolicLink == true { files.skipDescendants(); continue }
                guard state.isRegularFile == true else { continue }
                let parsed = parsedDiffuse(file) ?? (imageExtensions.contains(file.pathExtension.lowercased()) && NativeSourceFileTags.read(file)?.role == "input"
                    ? (file.deletingLastPathComponent().lastPathComponent + "_" + file.deletingPathExtension().lastPathComponent, "color_1") : nil)
                guard let parsed else { continue }
                try candidate(file, family: parsed.0, variant: parsed.1, metadata: nil)
            }
        }
        guard minimumSize > 0, minimumSize <= 32768 else { throw StudioError("The selected classifier image resolution is invalid.") }
        let families = Dictionary(grouping: candidates, by: \.familyVariantID)
        for index in candidates.indices {
            let source = candidates[index]
            let directories = Set((families[source.familyVariantID] ?? []).map { $0.url.deletingLastPathComponent().path })
            if directories.count > 1 {
                candidates[index].directoryScope = String(hash(Data(source.url.deletingLastPathComponent().path.utf8)).prefix(16))
            }
        }
        let grouped = Dictionary(grouping: candidates, by: \.logicalID)
        let selected = grouped.keys.sorted().compactMap { grouped[$0]?.min { left, right in
            let leftPixels = UInt64(left.width) * UInt64(left.height), rightPixels = UInt64(right.width) * UInt64(right.height)
            return leftPixels == rightPixels ? left.url.path < right.url.path : leftPixels < rightPixels
        } }
        if undersized > 0 { warnings.append("\(undersized) original diffuse images are smaller than \(minimumSize) × \(minimumSize). Their files are preserved; a larger native tier is selected when available.") }
        return Discovery(sources: selected, minimumSize: minimumSize, discoveredCount: discovered,
                         eligibleCount: candidates.count, undersizedCount: undersized, unreadableCount: unreadable, warnings: warnings)
    }

    static func refresh(sources: [Source], destination requested: URL,
                        providerMetadata: [String: ProviderMetadata] = [:], warnings: [String] = [],
                        minimumSize: Int? = nil,
                        onEvent: @Sendable (String) -> Void = { _ in }) throws -> String {
        try Task.checkCancellation()
        let requestedRoot = requested.standardizedFileURL
        let root = requestedRoot.resolvingSymlinksInPath().standardizedFileURL
        guard root.isFileURL, root.path != "/", (try? requestedRoot.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true,
              !sources.contains(where: {
                  let path = $0.url.resolvingSymlinksInPath().standardizedFileURL.path
                  return path == root.path || path.hasPrefix(root.path + "/")
              }) else {
            throw StudioError("Choose a classifier destination separate from the original material maps.")
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let descriptor = open(root.appendingPathComponent(".classifier-refresh.lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw StudioError("Cannot lock the classifier dataset.") }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else { throw StudioError("The classifier dataset is already refreshing.") }
        defer { flock(descriptor, LOCK_UN) }
        let manifestURL = root.appendingPathComponent(".classifier-manifest.json")
        var previous: [Entry] = [], previousClasses: [TextureClass] = []
        let hasManifest = FileManager.default.fileExists(atPath: manifestURL.path)
        if hasManifest {
            let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
            guard manifest.schema == schema, manifest.datasetPath == root.path,
                  Set(manifest.entries.map(\.id)).count == manifest.entries.count else { throw StudioError("Classifier dataset provenance does not match this folder.") }
            previous = manifest.entries; previousClasses = manifest.classes
        }
        for item in hasManifest ? [] : classes {
            let folder = root.appendingPathComponent(item.name, isDirectory: true)
            guard (try? folder.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { throw StudioError("Classifier class folders cannot be symbolic links.") }
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }
        var catalog = try classCatalog(root: root, previous: previousClasses)
        let pending = root.deletingLastPathComponent().appendingPathComponent(root.lastPathComponent + "-pending-review", isDirectory: true)
        var existing = try imageFiles(in: root) + (FileManager.default.fileExists(atPath: pending.path) ? try imageFiles(in: pending) : [])
        var entries: [Entry] = [], added = 0, preserved = 0, reclassified = 0
        var unreadable = 0
        var refreshWarnings = warnings
        let old = Dictionary(uniqueKeysWithValues: previous.map { ($0.id, $0) })
        let ambiguousHashes = Set(Dictionary(grouping: previous, by: \.sourceSHA256).filter { $0.value.count > 1 }.keys)
        func locatePrevious(_ entry: Entry) throws -> URL? {
            try locate(entry, files: existing, allowHashFallback: !ambiguousHashes.contains(entry.sourceSHA256))
        }
        var liveIDs = Set<String>()
        for (offset, source) in sources.enumerated() {
            try Task.checkCancellation()
            let sourceHash: String
            do {
                let bytes = try Data(contentsOf: source.url, options: .mappedIfSafe)
                sourceHash = hash(bytes)
                let dimensions = try verifiedDimensions(bytes, url: source.url)
                guard dimensions.0 == source.width, dimensions.1 == source.height else { throw StudioError("Original diffuse dimensions changed during classifier refresh.") }
                guard minimumSize.map({ source.width >= $0 && source.height >= $0 }) ?? true else { throw StudioError("Original diffuse image is smaller than the selected classifier grid.") }
            } catch {
                if Task.isCancelled { throw CancellationError() }
                unreadable += 1
                refreshWarnings.append("Could not add original diffuse image \(source.url.path): \(error.localizedDescription)")
                continue
            }
            liveIDs.insert(source.id)
            let familyEntries = previous.filter { $0.family.caseInsensitiveCompare(source.family) == .orderedSame && $0.variant.caseInsensitiveCompare(source.variant) == .orderedSame }
            let directory = source.url.deletingLastPathComponent().path
            let inDirectory = familyEntries.filter { URL(fileURLWithPath: $0.sourcePath).deletingLastPathComponent().path == directory }
            let currentDirectories = Set(sources.filter { $0.familyVariantID == source.familyVariantID }.map { $0.url.deletingLastPathComponent().path })
            let previousDirectories = Set(familyEntries.map { URL(fileURLWithPath: $0.sourcePath).deletingLastPathComponent().path })
            let related = !inDirectory.isEmpty ? inDirectory : currentDirectories.count == 1 && previousDirectories.count <= 1 ? familyEntries : []
            // Adopt a same-resolution legacy entry in place. A higher native
            // tier has a separate identity and never rewrites curated bytes.
            let sameOriginal = related.filter {
                $0.width == source.width && $0.height == source.height && $0.sourceSHA256 == sourceHash
            }
            let priorEntry = old[source.id] ?? (source.resolutionAware && sameOriginal.count == 1 ? sameOriginal[0] : nil)
            if let priorEntry { liveIDs.insert(priorEntry.id) }
            let metadata = providerMetadata[source.family.lowercased()] ?? priorEntry?.providerMetadata ?? source.metadata ?? related.compactMap(\.providerMetadata).first
            let savedTags = NativeSourceFileTags.read(source.url)
            let manualLabel = savedTags?.classifierIdentity == "user" ? savedTags?.classLabel : nil
            let hasManualLabel = manualLabel != nil
            if let manualLabel, catalog.allSatisfy({ $0.id != manualLabel && $0.name.caseInsensitiveCompare(manualLabel) != .orderedSame }), validClassFolderName(manualLabel) {
                let folder = root.appendingPathComponent(manualLabel, isDirectory: true)
                guard (try? folder.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { throw StudioError("Classifier class folders cannot be symbolic links.") }
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                catalog.append(TextureClass(id: "user_" + String(hash(Data(manualLabel.utf8)).prefix(16)), name: manualLabel, description: "A material class assigned by the user."))
            }
            let suggested = classify(family: source.family, filename: source.url.lastPathComponent, metadata: metadata)
            let materialName = normalized(source.family + " " + source.url.lastPathComponent + " " + (metadata?.name ?? ""))
            let custom = catalog.first { item in
                item.id.hasPrefix("user_") && (" " + materialName + " ").contains(" " + normalized(item.name) + " ")
            }
            let manualClass = manualLabel.flatMap { label in catalog.first { $0.id == label || $0.name.caseInsensitiveCompare(label) == .orderedSame } }
            let assignedID = hasManualLabel ? manualClass?.id : custom?.id ?? suggested.classID.flatMap { id in catalog.first { $0.id == id }?.id }
            let assignment = Assignment(classID: assignedID,
                confidence: hasManualLabel ? "user_file_tag" : (custom == nil ? suggested.confidence : "user_class_name"),
                evidence: hasManualLabel ? manualClass.map { ["The original file is tagged by the user as '\($0.name)'."] }
                    ?? ["The original file has no material class assigned by the user."]
                    : suggested.evidence + (custom.map { ["Name matches user class '\($0.name)'."] } ?? []))
            var entry: Entry
            if var prior = priorEntry {
                let current = try locatePrevious(prior)
                if let current {
                    let currentClass = classID(for: current, root: root, catalog: catalog)
                    let isEdited = try hash(Data(contentsOf: current, options: .mappedIfSafe)) != prior.sourceSHA256
                    let hasAutoStatus = ["suggested", "preserved", "taxonomy_refreshed", "taxonomy_conflict_review"].contains(prior.status)
                    let untouched = hasAutoStatus && !isEdited && sourceHash == prior.sourceSHA256 &&
                        prior.currentPath == current.path && current.lastPathComponent == prior.filename && currentClass == prior.suggestedClass
                    let relocated = prior.currentPath != current.path || current.lastPathComponent != prior.filename || currentClass != prior.suggestedClass
                    prior.currentPath = current.path; prior.currentClass = currentClass
                    if isEdited && !hasManualLabel { prior.status = "modified_by_user"; prior.needsReview = true }
                    else if relocated && !hasManualLabel { prior.status = "moved_by_user"; prior.needsReview = false }
                    else if untouched || hasManualLabel {
                        var canUpdate = true
                        if assignment.classID != prior.suggestedClass || (hasManualLabel && currentClass != assignment.classID) {
                            let targetFolder: URL
                            if let id = assignment.classID, let textureClass = catalog.first(where: { $0.id == id }) {
                                targetFolder = root.appendingPathComponent(textureClass.name)
                            } else {
                                guard (try? pending.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
                                    throw StudioError("Classifier pending-review folder cannot be a symbolic link.")
                                }
                                try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: true)
                                targetFolder = pending
                            }
                            let output = targetFolder.appendingPathComponent(current.lastPathComponent)
                            if output.path == current.path {
                                prior.currentClass = assignment.classID
                                prior.status = hasManualLabel ? "user_file_tag" : "taxonomy_refreshed"
                                prior.needsReview = !hasManualLabel
                            } else if FileManager.default.fileExists(atPath: output.path) {
                                // An occupied destination can be a user's added
                                // image. Leave both files intact for review.
                                canUpdate = false; prior.status = "taxonomy_conflict_review"; prior.needsReview = true
                                refreshWarnings.append("Classifier taxonomy move needs review; destination already contains an image: \(output.path)")
                            } else {
                                try FileManager.default.moveItem(at: current, to: output)
                                existing.removeAll { $0.path == current.path }; existing.append(output)
                                prior.currentPath = output.path; prior.currentClass = assignment.classID
                                prior.status = hasManualLabel ? "user_file_tag" : "taxonomy_refreshed"
                                prior.needsReview = !hasManualLabel; reclassified += 1
                            }
                        } else { prior.status = "preserved" }
                        if canUpdate {
                            prior.suggestedClass = assignment.classID; prior.confidence = assignment.confidence
                            prior.evidence = assignment.evidence; prior.providerMetadata = metadata
                        }
                    }
                    if isEdited { prior.status = "modified_by_user"; prior.needsReview = true }
                } else { prior.currentPath = nil; prior.currentClass = nil; prior.status = "removed_by_user"; prior.needsReview = false }
                if sourceHash != prior.sourceSHA256 { prior.status = "source_changed_review"; prior.needsReview = true }
                if prior.subtype == nil {
                    prior.subtype = subtype(family: source.family, filename: source.url.lastPathComponent, metadata: metadata, classID: assignment.classID)
                }
                entry = prior; preserved += 1
            } else {
                let removed = try related.contains { prior in
                    if prior.status == "removed_by_user", prior.currentPath == nil { return true }
                    guard prior.currentPath != nil else { return false }
                    return try locatePrevious(prior) == nil
                }
                var inheritedClass: String?
                for prior in related where !["suggested", "preserved", "taxonomy_refreshed", "source_not_in_current_inventory"].contains(prior.status) {
                    if let current = try locatePrevious(prior), let label = classID(for: current, root: root, catalog: catalog) {
                        inheritedClass = label; break
                    }
                }
                // A manual move since the previous refresh is already curation.
                if inheritedClass == nil {
                    for prior in related {
                        if let current = try locatePrevious(prior), let label = classID(for: current, root: root, catalog: catalog),
                           current.path != prior.currentPath || label != prior.suggestedClass {
                            inheritedClass = label; break
                        }
                    }
                }
                if hasManualLabel { inheritedClass = manualClass?.id }
                let manualMatches = try existing.filter { file in
                    guard file.lastPathComponent == source.url.lastPathComponent,
                          let label = classID(for: file, root: root, catalog: catalog),
                          !hasManualLabel || label == assignedID else { return false }
                    return hash(try Data(contentsOf: file, options: .mappedIfSafe)) == sourceHash
                }
                if removed {
                    entry = Entry(id: source.id, family: source.family, variant: source.variant, sourcePath: source.url.path,
                        sourceSHA256: sourceHash, width: source.width, height: source.height, filename: "",
                        suggestedClass: assignment.classID, confidence: "user_removed_material", evidence: ["Your removal of this material is preserved across native resolution changes."],
                        providerMetadata: metadata, subtype: nil, currentPath: nil, currentClass: nil, status: "removed_by_user", needsReview: false)
                    preserved += 1
                } else if manualMatches.count == 1, let curated = manualMatches.first {
                    entry = Entry(id: source.id, family: source.family, variant: source.variant, sourcePath: source.url.path,
                        sourceSHA256: sourceHash, width: source.width, height: source.height, filename: curated.lastPathComponent,
                        suggestedClass: assignment.classID, confidence: "user_curated_original", evidence: ["The existing user-classified image matches the original source bytes."],
                        providerMetadata: metadata, subtype: subtype(family: source.family, filename: source.url.lastPathComponent, metadata: metadata, classID: assignment.classID),
                        currentPath: curated.path, currentClass: classID(for: curated, root: root, catalog: catalog),
                        status: "adopted_user_class", needsReview: false)
                    preserved += 1
                } else {
                    let filename = safeName(source.family) + "--" + safeName(source.variant) + "--" + String(hash(Data(source.id.utf8)).prefix(12)) + "." + source.url.pathExtension.lowercased()
                    let targetFolder: URL
                    if let id = inheritedClass ?? assignment.classID, let textureClass = catalog.first(where: { $0.id == id }) { targetFolder = root.appendingPathComponent(textureClass.name) }
                    else {
                        guard (try? pending.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { throw StudioError("Classifier pending-review folder cannot be a symbolic link.") }
                        try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: true)
                        targetFolder = pending
                    }
                    let output = targetFolder.appendingPathComponent(filename)
                    // A manually supplied image with this stable name stays intact.
                    if FileManager.default.fileExists(atPath: output.path) {
                        guard try hash(Data(contentsOf: output, options: .mappedIfSafe)) == sourceHash else {
                            throw StudioError("Classifier filename is already occupied by another image: \(output.path)")
                        }
                    } else { try copyOriginal(source.url, to: output, expectedSHA256: sourceHash); added += 1 }
                    entry = Entry(id: source.id, family: source.family, variant: source.variant, sourcePath: source.url.path,
                        sourceSHA256: sourceHash, width: source.width, height: source.height, filename: filename,
                        suggestedClass: assignment.classID, confidence: assignment.confidence, evidence: assignment.evidence,
                        providerMetadata: metadata, subtype: subtype(family: source.family, filename: source.url.lastPathComponent, metadata: metadata, classID: assignment.classID),
                        currentPath: output.path, currentClass: inheritedClass ?? assignment.classID,
                        status: inheritedClass == nil ? "suggested" : "inherited_user_class", needsReview: inheritedClass == nil)
                }
            }
            entries.append(entry)
            // Every successful copy is durable even if the user cancels later.
            try saveManifest(entries + previous.filter { !liveIDs.contains($0.id) }, to: manifestURL, root: root, catalog: catalog, minimumSize: minimumSize)
            if savedTags?.classifierIdentity != "user", let label = catalog.first(where: { $0.id == entry.currentClass })?.name {
                do {
                    let userCurated = ["moved_by_user", "adopted_user_class", "inherited_user_class"].contains(entry.status)
                    try NativeSourceFileTags.update(source.url, role: "input", classLabel: label,
                        classifierIdentity: userCurated ? "user" : "filename-taxonomy-v2", verifiedSHA256: sourceHash,
                        reviewStatus: entry.needsReview ? "needs_review" : "reviewed")
                } catch { refreshWarnings.append("Classifier label was saved, but the original file tag could not be updated for \(source.url.path): \(error.localizedDescription)") }
            }
            onEvent(try event("Classifier diffuse: \(source.family)", completed: offset + 1, total: sources.count))
        }
        for var entry in previous where !liveIDs.contains(entry.id) {
            if entry.status != "removed_by_user" {
                if let current = try locatePrevious(entry) {
                    entry.currentPath = current.path; entry.currentClass = classID(for: current, root: root, catalog: catalog)
                }
                entry.status = "source_not_in_current_inventory"; entry.needsReview = true
            }
            entries.append(entry)
        }
        try saveManifest(entries, to: manifestURL, root: root, catalog: catalog, minimumSize: minimumSize)
        let counts = try catalog.map { item -> Object in
            ["id": item.id, "name": item.name, "count": try imageFiles(in: root.appendingPathComponent(item.name)).count]
        }
        // Curation is already saved. A derived-view problem is a follow-up
        // notice and must not turn that completed operation into a failure.
        let training = refreshTrainingView(root: root, manifestURL: manifestURL, minimumSize: minimumSize, onEvent: onEvent)
        refreshWarnings.append(contentsOf: training.warnings)
        let reviewURL = root.deletingLastPathComponent().appendingPathComponent(root.lastPathComponent + "-review.html")
        do {
            try Data(reviewHTML(entries: entries, counts: counts, root: root, pending: pending, training: training, minimumSize: minimumSize, warnings: refreshWarnings).utf8).write(to: reviewURL, options: .atomic)
        } catch { refreshWarnings.append("Classifier curation is saved, but its review page could not be refreshed: \(error.localizedDescription)") }
        return try json(["dataset_path": root.path, "manifest_path": manifestURL.path, "review_path": reviewURL.path,
            "training_path": training.path.path, "training_ready": training.ready,
            "training_count": training.imageCount, "training_class_count": training.classCount,
            "minimum_native_size": minimumSize as Any? ?? NSNull(), "eligible_count": sources.count,
            "undersized_count": 0, "unreadable_count": unreadable,
            "training_undersized_count": training.undersizedCount, "training_unreadable_count": training.unreadableCount,
            "training_retired_source_count": training.retiredCount,
            "empty_classes": counts.filter { $0["count"] as? Int == 0 }.compactMap { $0["name"] as? String },
            "pending_review_path": pending.path, "total": counts.reduce(0) { $0 + ($1["count"] as? Int ?? 0) }, "source_count": sources.count,
            "added": added, "preserved": preserved, "reclassified": reclassified, "needs_review": entries.filter(\.needsReview).count,
            "unclassified": entries.filter { $0.currentClass == nil && $0.currentPath != nil }.count,
            "classes": counts, "warnings": refreshWarnings,
            "selection_policy": selectionPolicy(minimumSize)])
    }

    private static func copyOriginal(_ source: URL, to destination: URL, expectedSHA256: String) throws {
        let staging = destination.deletingLastPathComponent().appendingPathComponent(".classifier-copy-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: staging) }
        // APFS clones share storage without symlink or hard-link dependence.
        // Other volumes use an ordinary copy with the same byte verification.
        if clonefile(source.path, staging.path, 0) != 0 {
            try FileManager.default.copyItem(at: source, to: staging)
        }
        try Task.checkCancellation()
        guard try hash(Data(contentsOf: staging, options: .mappedIfSafe)) == expectedSHA256 else {
            throw StudioError("Original diffuse changed while copying into the classifier dataset.")
        }
        try FileManager.default.moveItem(at: staging, to: destination)
    }

    private static func refreshTrainingView(root: URL, manifestURL: URL, minimumSize: Int?,
                                            onEvent: @Sendable (String) -> Void) -> TrainingViewResult {
        let training = root.deletingLastPathComponent().appendingPathComponent(root.lastPathComponent + "-createml", isDirectory: true)
        do { return try publishTrainingView(root: root, training: training, manifestURL: manifestURL, minimumSize: minimumSize, onEvent: onEvent) }
        catch {
            return TrainingViewResult(path: training, imageCount: 0, classCount: 0, ready: false,
                warnings: ["Classifier curation is saved. The Create ML training view was left unchanged because it could not be refreshed: \(error.localizedDescription)"])
        }
    }

    /// Empty class directories make Create ML reject labeledDirectories.
    /// Keep them in the curation dataset, but publish only populated labels in
    /// a disposable sibling view containing the current curated image bytes.
    private static func publishTrainingView(root: URL, training: URL, manifestURL: URL, minimumSize: Int?,
                                            onEvent: @Sendable (String) -> Void) throws -> TrainingViewResult {
        let hadView = pathExists(training)
        let oldManifest = hadView ? try validateTrainingView(at: training, dataset: root, training: training) : nil
        let staging = root.deletingLastPathComponent().appendingPathComponent(".\(root.lastPathComponent)-createml-stage-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        var removeStaging = true
        defer { if removeStaging { try? FileManager.default.removeItem(at: staging) } }
        var inventory: [TrainingViewFile] = [], labels: [String] = []
        var undersized = 0, unreadable = 0, retired = 0, warnings: [String] = []
        let curation = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        let retiredPaths = Set(curation.entries.filter { $0.status == "source_not_in_current_inventory" }.compactMap(\.currentPath))
        let selectedPaths = Set(curation.entries.filter { $0.status != "source_not_in_current_inventory" }.compactMap(\.currentPath))
        for textureClass in curation.classes {
            try Task.checkCancellation()
            let classFolder = root.appendingPathComponent(textureClass.name)
            let files = try imageFiles(in: classFolder).filter { file in
                guard let minimumSize else { return true }
                do {
                    let dimensions = try nativeDimensions(file)
                    guard dimensions.0 >= minimumSize, dimensions.1 >= minimumSize else { undersized += 1; return false }
                    if retiredPaths.contains(file.path), !selectedPaths.contains(file.path) { retired += 1; return false }
                    return true
                } catch {
                    if Task.isCancelled { throw CancellationError() }
                    unreadable += 1
                    warnings.append("Curated image was preserved but omitted from Create ML because its native dimensions could not be read: \(file.path)")
                    return false
                }
            }
            guard !files.isEmpty else { continue }
            labels.append(textureClass.name)
            let outputFolder = staging.appendingPathComponent(textureClass.name)
            try FileManager.default.createDirectory(at: outputFolder, withIntermediateDirectories: false)
            let filenameCounts = Dictionary(grouping: files, by: \.lastPathComponent).mapValues(\.count)
            for file in files {
                let bytes = try Data(contentsOf: file, options: .mappedIfSafe), checksum = hash(bytes)
                let filename: String
                if filenameCounts[file.lastPathComponent] == 1 { filename = file.lastPathComponent }
                else {
                    let relative = String(file.path.dropFirst(classFolder.path.count + 1))
                    filename = file.deletingPathExtension().lastPathComponent + "--" + String(hash(Data(relative.utf8)).prefix(12)) + "." + file.pathExtension
                }
                try copyOriginal(file, to: outputFolder.appendingPathComponent(filename), expectedSHA256: checksum)
                inventory.append(TrainingViewFile(relativePath: textureClass.name + "/" + filename,
                    sourcePath: file.path, sha256: checksum, bytes: bytes.count))
            }
            onEvent(try event("Create ML class: \(textureClass.name)", completed: inventory.count, total: 0))
        }
        let viewManifest = TrainingViewManifest(schema: trainingViewSchema, datasetPath: root.path, trainingPath: training.path,
            curationManifestSHA256: hash(try Data(contentsOf: manifestURL)), updatedAt: ISO8601DateFormatter().string(from: Date()),
            classes: labels, files: inventory.sorted { $0.relativePath < $1.relativePath })
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(viewManifest).write(to: staging.appendingPathComponent(trainingViewManifestName), options: .atomic)
        _ = try validateTrainingView(at: staging, dataset: root, training: training)
        try Task.checkCancellation()
        if hadView {
            // Revalidate immediately before the atomic exchange. A user edit
            // during preparation must keep the prior view intact.
            let current = try validateTrainingView(at: training, dataset: root, training: training)
            guard current.curationManifestSHA256 == oldManifest?.curationManifestSHA256,
                  current.updatedAt == oldManifest?.updatedAt else { throw StudioError("The generated training view changed during refresh.") }
        } else if pathExists(training) { throw StudioError("An unrelated folder appeared at the Create ML view destination.") }
        let mode = hadView ? UInt32(RENAME_SWAP) : UInt32(RENAME_EXCL)
        guard renameatx_np(AT_FDCWD, staging.path, AT_FDCWD, training.path, mode) == 0 else {
            throw StudioError("Could not atomically publish the Create ML view: \(String(cString: strerror(errno)))")
        }
        if undersized > 0 { warnings.append("\(undersized) curated images are below the selected \(minimumSize ?? 0) × \(minimumSize ?? 0) grid. They remain in their class folders and are omitted from the Create ML view.") }
        if retired > 0 { warnings.append("\(retired) prior native source copies remain reviewable in their class folders; only the currently selected native tier is exported to Create ML.") }
        if hadView {
            // After the swap, staging holds the old view. Never remove it if
            // it acquired a user edit after the last pre-swap inspection.
            removeStaging = false
            do {
                _ = try validateTrainingView(at: staging, dataset: root, training: training)
                try FileManager.default.removeItem(at: staging)
            } catch { warnings.append("The new Create ML view is ready. Its previous view was preserved at \(staging.path): \(error.localizedDescription)") }
        }
        return TrainingViewResult(path: training, imageCount: inventory.count, classCount: labels.count,
                                  ready: !inventory.isEmpty, warnings: warnings,
                                  undersizedCount: undersized, unreadableCount: unreadable, retiredCount: retired)
    }

    private static func nativeDimensions(_ url: URL) throws -> (Int, Int) {
        if url.pathExtension.lowercased() == "png" {
            let header = try NativePNG.inspect(url)
            return (header.width, header.height)
        }
        guard let image = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              let properties = CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let height = properties[kCGImagePropertyPixelHeight] as? NSNumber,
              width.intValue > 0, height.intValue > 0 else { throw StudioError("Image dimensions are unavailable.") }
        return (width.intValue, height.intValue)
    }

    private static func verifiedDimensions(_ bytes: Data, url: URL) throws -> (Int, Int) {
        if url.pathExtension.lowercased() == "png" {
            let metadata = try NativePNG.sourceMetadata(bytes)
            guard let width = metadata["width"] as? Int, let height = metadata["height"] as? Int else { throw StudioError("Original diffuse dimensions are unavailable.") }
            return (width, height)
        }
        guard let source = CGImageSourceCreateWithData(bytes as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetStatusAtIndex(source, 0) == .statusComplete else { throw StudioError("Original diffuse image could not be decoded completely.") }
        return (image.width, image.height)
    }

    /// Replacement owns exactly this manifest and its declared directories and
    /// hashes. Any unexpected file, symlink, changed image or malformed binding
    /// makes the view user data that must be preserved.
    private static func validateTrainingView(at location: URL, dataset: URL, training: URL) throws -> TrainingViewManifest {
        let state = try location.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard state.isDirectory == true, state.isSymbolicLink != true else { throw StudioError("The Create ML view destination is not an owned directory.") }
        let canonicalLocation = location.resolvingSymlinksInPath().standardizedFileURL
        let manifestURL = canonicalLocation.appendingPathComponent(trainingViewManifestName)
        guard (try? manifestURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])).map({ $0.isRegularFile == true && $0.isSymbolicLink != true }) == true else {
            throw StudioError("The Create ML view destination has no generated ownership manifest.")
        }
        let manifest = try JSONDecoder().decode(TrainingViewManifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.schema == trainingViewSchema, manifest.datasetPath == dataset.path, manifest.trainingPath == training.path,
              Set(manifest.classes).count == manifest.classes.count, Set(manifest.files.map(\.relativePath)).count == manifest.files.count,
              manifest.classes.allSatisfy(validClassFolderName) else { throw StudioError("The Create ML view ownership binding is invalid.") }
        let expectedFiles = Dictionary(uniqueKeysWithValues: manifest.files.map { ($0.relativePath, $0) })
        var found = Set<String>(), foundDirectories = Set<String>()
        guard let iterator = FileManager.default.enumerator(at: canonicalLocation, includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]) else {
            throw StudioError("Cannot inspect the generated Create ML view.")
        }
        for case let discovered as URL in iterator {
            try Task.checkCancellation()
            let values = try discovered.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey])
            let file = discovered.resolvingSymlinksInPath().standardizedFileURL
            guard file.path.hasPrefix(canonicalLocation.path + "/") else { throw StudioError("The Create ML view contains an image outside its owned directory.") }
            let relative = String(file.path.dropFirst(canonicalLocation.path.count + 1))
            guard values.isSymbolicLink != true else { throw StudioError("The Create ML view contains a symbolic link: \(relative)") }
            if values.isDirectory == true {
                guard manifest.classes.contains(relative) else { throw StudioError("The Create ML view contains an undeclared folder: \(relative)") }
                foundDirectories.insert(relative); continue
            }
            guard values.isRegularFile == true else { throw StudioError("The Create ML view contains an undeclared special file.") }
            if relative == trainingViewManifestName { continue }
            if file.lastPathComponent == ".DS_Store", let bytes = try? Data(contentsOf: file), bytes.count >= 8,
               bytes.prefix(8) == Data([0, 0, 0, 1, 66, 117, 100, 49]) { continue }
            guard let expected = expectedFiles[relative] else { throw StudioError("The Create ML view contains a user-added file: \(relative)") }
            let bytes = try Data(contentsOf: file, options: .mappedIfSafe)
            guard bytes.count == expected.bytes, hash(bytes) == expected.sha256 else { throw StudioError("The Create ML view image was edited: \(relative)") }
            found.insert(relative)
        }
        guard found == Set(expectedFiles.keys), foundDirectories == Set(manifest.classes),
              manifest.classes.allSatisfy({ label in manifest.files.contains { $0.relativePath.hasPrefix(label + "/") } }) else {
            throw StudioError("The generated Create ML view has missing images or class folders.")
        }
        return manifest
    }

    private static func pathExists(_ url: URL) -> Bool { var state = stat(); return lstat(url.path, &state) == 0 }
    private static func locate(_ entry: Entry, files: [URL], allowHashFallback: Bool) throws -> URL? {
        if entry.currentPath == nil, entry.status == "removed_by_user" { return nil }
        if let path = entry.currentPath, let match = files.first(where: { $0.path == path }) { return match }
        if let identity = entry.currentFileIdentity {
            let renamed = files.filter { file in
                guard let current = try? NativeSourceFileTags.identity(file) else { return false }
                return current.inode == identity.inode && current.device == identity.device
            }
            if renamed.count == 1 { return renamed[0] }
        }
        let named = files.filter { $0.lastPathComponent == entry.filename }
        if named.count == 1 { return named[0] }
        // Equal source bytes do not identify a material when another curated
        // entry used those bytes. A deletion must not adopt its neighbor.
        guard allowHashFallback else { return nil }
        let matching = try files.filter { try hash(Data(contentsOf: $0, options: .mappedIfSafe)) == entry.sourceSHA256 }
        return matching.count == 1 ? matching[0] : nil
    }
    private static func classID(for file: URL, root: URL, catalog: [TextureClass]) -> String? {
        guard file.path.hasPrefix(root.path + "/") else { return nil }
        let components = file.path.dropFirst(root.path.count + 1).split(separator: "/")
        return catalog.first { $0.name == components.first.map(String.init) }?.id
    }
    private static func validClassFolderName(_ name: String) -> Bool {
        !name.isEmpty && !name.hasPrefix(".") && !name.contains("/") && !name.contains("\\") && !name.contains("\0")
    }
    private static func classCatalog(root: URL, previous: [TextureClass]) throws -> [TextureClass] {
        var result: [TextureClass] = []
        let children = try FileManager.default.contentsOfDirectory(at: root,
            includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles])
        for folder in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let state = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard state.isDirectory == true else { continue }
            guard state.isSymbolicLink != true, validClassFolderName(folder.lastPathComponent) else {
                throw StudioError("Classifier class folders must be ordinary child directories.")
            }
            let name = folder.lastPathComponent
            if let known = (previous + classes).first(where: { $0.name == name }) { result.append(known) }
            else { result.append(.init(id: "user_" + String(hash(Data(name.utf8)).prefix(12)), name: name,
                                      description: "User-created material type.")) }
        }
        guard Set(result.map(\.id)).count == result.count else { throw StudioError("Classifier class identities must be unique.") }
        return result
    }
    private static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "jpe", "tif", "tiff", "heic", "heif", "bmp", "gif", "webp", "exr"]
    private static func imageFiles(in folder: URL) throws -> [URL] {
        guard let iterator = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles]) else { return [] }
        var files: [URL] = []
        for case let file as URL in iterator {
            try Task.checkCancellation()
            let state = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if state.isSymbolicLink == true { iterator.skipDescendants(); continue }
            if state.isRegularFile == true, imageExtensions.contains(file.pathExtension.lowercased()) { files.append(file) }
        }
        return files.map { $0.resolvingSymlinksInPath().standardizedFileURL }.sorted { $0.path < $1.path }
    }
    private static func parsedDiffuse(_ file: URL) -> (String, String)? {
        guard imageExtensions.contains(file.pathExtension.lowercased()) else { return nil }
        return NativeMaterialDatasetService.diffuseSourceFilename(file)
    }
    private static func contained(_ root: URL, _ relative: String) throws -> URL {
        let url = root.appendingPathComponent(relative).resolvingSymlinksInPath().standardizedFileURL
        guard !relative.hasPrefix("/"), url.path.hasPrefix(root.path + "/") else { throw StudioError("Material sample path leaves its dataset.") }
        return url
    }
    private static func selectionPolicy(_ minimumSize: Int?) -> String {
        minimumSize.map { "one_native_original_at_least_\($0)_pixels_per_axis_per_family_and_color_variant_no_resizing" }
            ?? "one_native_original_per_family_and_color_variant_no_resizing"
    }
    private static func saveManifest(_ entries: [Entry], to url: URL, root: URL, catalog: [TextureClass], minimumSize: Int?) throws {
        let recorded = entries.map { original -> Entry in
            var entry = original
            if let path = entry.currentPath, let identity = try? NativeSourceFileTags.identity(URL(fileURLWithPath: path)) {
                entry.currentFileIdentity = identity
            }
            return entry
        }
        let manifest = Manifest(schema: schema, updatedAt: ISO8601DateFormatter().string(from: Date()), datasetPath: root.path,
            selectionPolicy: selectionPolicy(minimumSize), minimumNativeSize: minimumSize, classes: catalog,
            entries: recorded.sorted { $0.id < $1.id })
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(to: url, options: .atomic)
    }
    private static func reviewHTML(entries: [Entry], counts: [Object], root: URL, pending: URL, training: TrainingViewResult, minimumSize: Int?, warnings: [String]) -> String {
        func escape(_ text: String) -> String { text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;").replacingOccurrences(of: "\"", with: "&quot;") }
        let summary = counts.map { "<li><b>\(escape($0["name"] as? String ?? ""))</b>: \($0["count"] as? Int ?? 0)</li>" }.joined()
        let policy = minimumSize.map { "Only originals at least \($0) × \($0) pixels supply the selected grid. One native tier per material/color variant is selected; smaller originals stay preserved and visible for review." } ?? "One native original per material/color variant is selected."
        let notices = warnings.isEmpty ? "" : "<h2>Refresh notices</h2><ul>" + warnings.map { "<li>\(escape($0))</li>" }.joined() + "</ul>"
        let cards = entries.map { entry in
            let image = entry.currentPath.map { "<a href=\"\(escape(URL(fileURLWithPath: $0).absoluteString))\"><img loading=\"lazy\" src=\"\(escape(URL(fileURLWithPath: $0).absoluteString))\" alt=\"\(escape(entry.family))\"></a>" } ?? "<p>Removed image</p>"
            let label = counts.first { $0["id"] as? String == entry.currentClass }?["name"] as? String ?? "Pending review"
            let subtype = entry.subtype.map { "<p>Subtype: <b>\(escape($0.replacingOccurrences(of: "_", with: " ")))</b></p>" } ?? ""
            return "<article>\(image)<h2>\(escape(entry.family))</h2><p><b>\(escape(label))</b> · \(escape(entry.variant)) · \(entry.width) × \(entry.height)</p>\(subtype)<p>\(escape(entry.status)) · \(escape(entry.confidence))</p><p>\(escape(entry.evidence.joined(separator: "; ")))</p><details><summary>Original provenance</summary><p>\(escape(entry.sourcePath))</p><code>\(entry.sourceSHA256)</code></details></article>"
        }.joined()
        return """
        <!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Diffuse texture classifier dataset review</title>
        <style>body{font:16px system-ui;margin:32px;background:#171a1d;color:#eef1f4}h1{font-size:28px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:20px}article{background:#252a30;padding:16px;border-radius:12px;overflow-wrap:anywhere}img{width:100%;aspect-ratio:1;object-fit:contain;background:#111}h2{font-size:18px}code{font-size:11px}a{color:#8dc9ff}li{margin:6px}details{font-size:12px}</style>
        <h1>Diffuse texture classifier dataset</h1><p>\(training.ready ? "Choose <b>\(escape(training.path.path))</b> as the parent image folder in Create ML (\(training.imageCount) images across \(training.classCount) populated classes)." : "The Create ML training view is not ready; review the refresh notices or add class examples.")</p>
        <p>Curate your images in <b>\(escape(root.path))</b>. Each class is a child folder. Review the initial suggestions by moving images between class folders in Finder; refresh preserves moves, renames, edits, removals and your added images. Refresh creates a separate byte-preserving Create ML view containing only populated classes. Keep your edits in the curation folder; changed images in an existing generated view are preserved and reported for review.</p>
        <p>Untouched generated suggestions follow updated class definitions. Bark has its own class. Subtype hints such as planks and boards come from explicit names or tags and remain available for future specialized models. Original family names and existing image filenames are preserved.</p>
        <p>\(escape(policy)) Images are copied byte-for-byte. No resizing, crops, gamma changes, noise or blur are applied. The Create ML view checks the actual native dimensions of every curated image, including user-added images. Repeated resolutions are excluded to reduce validation leakage. Create ML’s feature extractor may resize internally; that does not change these files or material-model training crops.</p>
        <p>Unclassified images remain outside the training folder at \(escape(pending.path)). Add or rename class folders in the curation folder to define your material types; refresh preserves your chosen classes. Empty classes stay out of the generated import view until you add examples. Automatic validation is a preliminary check; use distinct material families for an independent test set.</p>
        \(notices)<ul>\(summary)</ul><main>\(cards)</main></html>
        """
    }
    private static func normalized(_ value: String) -> String {
        value.lowercased().replacingOccurrences(of: #"(?<=[a-z])(?=\d)|[^a-z0-9]+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    private static func safeName(_ value: String) -> String { String(normalized(value).replacingOccurrences(of: " ", with: "_").prefix(100)) }
    private static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private static func object(_ bytes: Data) throws -> Object {
        guard let result = try JSONSerialization.jsonObject(with: bytes) as? Object else { throw StudioError("Classifier metadata must be a JSON object.") }
        return result
    }
    private static func json(_ value: Object) throws -> String { String(decoding: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]), as: UTF8.self) }
    private static func event(_ message: String, completed: Int, total: Int) throws -> String {
        try json(["event": "classifier_dataset_progress", "message": message, "completed": completed, "total": total]) + "\n"
    }
}
