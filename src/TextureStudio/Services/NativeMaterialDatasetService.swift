import CryptoKit
import Darwin
import Foundation

/// Dataset manifests, crop planning and metadata transactions run in Swift.
enum NativeMaterialDatasetService {
    /// Preparation uses CPU work on immutable original maps. Leave most
    /// practical memory available to the app, Metal and other applications.
    struct PreparationPolicy: Sendable {
        let memoryBudgetBytes: UInt64
        let maximumWorkers: Int
        init(memoryBudgetBytes: UInt64, maximumWorkers: Int) {
            self.memoryBudgetBytes = memoryBudgetBytes; self.maximumWorkers = max(1, maximumWorkers)
        }
        init(resources: MachineResources, processorCount: Int) {
            self.init(memoryBudgetBytes: min(resources.practicalBytes / 4, 16 * MachineResources.gibibyte), maximumWorkers: processorCount)
        }
        static var current: Self { .init(resources: .current, processorCount: MachineResources.current.availableProcessorCount) }
        func limitingWorkers(to requested: Int) throws -> Self {
            guard requested > 0 else { throw StudioError("Preparation workers must be a positive whole number.") }
            return .init(memoryBudgetBytes: memoryBudgetBytes, maximumWorkers: min(maximumWorkers, requested))
        }
        func workerCount(jobCount: Int, estimatedPeakBytes: UInt64) throws -> Int {
            guard jobCount > 0, estimatedPeakBytes > 0, estimatedPeakBytes <= memoryBudgetBytes else {
                throw StudioError("The original map needs more working memory than is available for safe native crop preparation.")
            }
            return min(jobCount, maximumWorkers, Int(min(UInt64(Int.max), memoryBudgetBytes / estimatedPeakBytes)))
        }
    }
    struct TrainingSample: Sendable {
        let id: String
        let split: String
        let width: Int
        let height: Int
        let inputURL: URL
        let targetURL: URL
        let inputSHA256: String
        let targetSHA256: String
        let inputEncoding: String
        let targetConvention: String
    }
    static func trainingSamples(datasetURL: URL, size: Int, target: String, materials: [String] = []) throws -> [TrainingSample] {
        let root = canonical(datasetURL), index = try object(root.appendingPathComponent("dataset.json"))
        let records = try readRecords(root, index: index)
        return try records.flatMap { record -> [TrainingSample] in
            let sample = record.sample, status = sample["status"] as? String ?? "unreviewed", material = sample["material_id"] as? String ?? ""
            guard !["excluded", "rejected"].contains(status), materials.isEmpty || materials.contains(material), sample["sample_pixel_dimensions"] as? [Int] == [size, size], let metadata = sample["map_metadata"] as? [String: Object], let input = metadata["input"], let output = metadata[target], let targetName = (sample["maps"] as? [String: String])?[target] else { return [] }
            let folder = record.path.deletingLastPathComponent()
            func resolve(_ name: String, _ details: Object) throws -> URL {
                if details["storage"] as? String == "source_reference" {
                    guard let source = details["source"] as? Object, name == source["path"] as? String, details["sample_sha256"] as? String == source["file_sha256"] as? String else { throw StudioError("Training source reference lost its exact original binding.") }
                    return URL(fileURLWithPath: name)
                }
                return try contained(folder, name)
            }
            let colors = sample["input_variants"] as? [Object] ?? []
            let variants = colors.isEmpty ? [input] : colors
            let outputURL = try resolve(targetName, output)
            let transforms = output["transforms"] as? [Object] ?? []
            let convention = transforms.contains { $0["type"] as? String == "directx_to_opengl" && $0["applied_in_memory"] as? Bool == true } ? "directx" : "opengl"
            return try variants.enumerated().map { offset, color in
                guard let inputName = color["filename"] as? String ?? (sample["maps"] as? [String: String])?["input"], let inputHash = color["sample_sha256"] as? String, let targetHash = output["sample_sha256"] as? String else { throw StudioError("Training map checksums are missing.") }
                return TrainingSample(id: (sample["sample_id"] as? String ?? "") + (offset == 0 ? "" : "|" + (color["variant_id"] as? String ?? "\(offset)")), split: sample["split"] as? String ?? "train", width: size, height: size, inputURL: try resolve(inputName, color), targetURL: outputURL, inputSHA256: inputHash, targetSHA256: targetHash, inputEncoding: color["encoding"] as? String ?? "source_srgb_assumed", targetConvention: convention)
            }
        }
    }
    private typealias Object = [String: Any]
    private struct Record { var entry: Object; let path: URL; var sample: Object }
    static let commands: Set<String> = ["dataset", "create-dataset", "edit-dataset", "remove-material", "validate-delete", "curate", "remove-missing", "add-material", "scan-folder", "import-folder", "prepare-size", "cleanup-size"]
    private static let sizes = [256, 512, 1024, 2048]
    private static let schema = "texture-studio-material-workbench-v1"
    private static let management = "texture-studio-dataset-management-v1"
    private static let journal = ".material-workbench-journal.json"

    /// nil means this service does not own the requested native command.
    static func run(arguments: [String], preparationPolicy: PreparationPolicy? = nil,
                    onEvent: @escaping @Sendable (String) -> Void = { _ in }) async throws -> String? {
        guard let command = arguments.first, commands.contains(command) else { return nil }
        try Task.checkCancellation()
        let operation = Task.detached(priority: .userInitiated) { try runSynchronously(arguments, preparationPolicy: preparationPolicy, onEvent: onEvent) }
        return try await withTaskCancellationHandler {
            let result = try await operation.value
            try Task.checkCancellation()
            return result
        } onCancel: { operation.cancel() }
    }
    private static func runSynchronously(_ arguments: [String], preparationPolicy: PreparationPolicy?, onEvent: @Sendable (String) -> Void) throws -> String? {
        try Task.checkCancellation()
        var options: [String: String] = [:]
        var cursor = 1
        while cursor < arguments.count {
            if arguments[cursor] == "--automatic-validation" { options[arguments[cursor]] = "true"; cursor += 1; continue }
            guard cursor + 1 < arguments.count, arguments[cursor].hasPrefix("--") else { throw StudioError("Invalid native dataset arguments.") }
            options[arguments[cursor]] = arguments[cursor + 1]; cursor += 2
        }
        guard let requested = options["--dataset"] else { throw StudioError("Choose a dataset folder.") }
        let command = arguments[0]
        let root = canonical(URL(fileURLWithPath: requested))
        if command == "create-dataset" { try create(root, options: options) }
        if command == "cleanup-size" { return try jsonString(cleanup(root)) }
        if command == "prepare-size" {
            var policy = preparationPolicy ?? .current
            if let value = options["--preparation-workers"] {
                guard let requested = Int(value) else { throw StudioError("Preparation workers must be a positive whole number.") }
                policy = try policy.limitingWorkers(to: requested)
            }
            return try jsonString(prepare(root, options: options, policy: policy, onEvent: onEvent))
        }
        return try locked(root) {
            try Task.checkCancellation()
            var index = try object(root.appendingPathComponent("dataset.json"))
            try compactStoredRecords(root, index: index)
            if try sourceInventoryNeedsRefresh(root, index: index) { index = try refreshSourceIndex(root, index: index) }
            let records = try readRecords(root, index: index)
            if ["curate", "remove-missing"].contains(command), let lineage = index["native_size_preparation"] as? Object {
                try checkIndex(root, expected: options["--expected-index-sha256"])
                guard lineage["schema"] as? String == preparationSchema, let source = lineage["source_dataset_path"] as? String else { throw StudioError("Prepared dataset source lineage is invalid.") }
                let original = canonical(URL(fileURLWithPath: source))
                guard original != root else { throw StudioError("Prepared dataset source lineage is circular.") }
                if command == "remove-missing", let path = options["--path"], path.hasPrefix(root.path + "/"), let id = options["--sample"], let sample = records.first(where: { $0.sample["sample_id"] as? String == id }) {
                    let details = sample.sample["map_metadata"] as? [String: Object] ?? [:]
                    for map in details.values { if let originalSource = map["source"] as? Object, let recovered = try recoverSource(originalSource, root: original) {
                        return try jsonString(["dataset_path": root.path, "sample_id": id, "removed": false, "recoverable": true, "source_path": recovered.path])
                    } }
                }
                return try locked(original) {
                    var sourceOptions = options
                    sourceOptions["--expected-index-sha256"] = nil
                    sourceOptions["--review-size"] = String(index["crop_size"] as? Int ?? 1024)
                    let sourceIndex = try object(original.appendingPathComponent("dataset.json")), sourceRecords = try readRecords(original, index: sourceIndex)
                    let response = try command == "curate" ? curate(original, index: sourceIndex, records: sourceRecords, options: sourceOptions) : removeMissing(original, index: sourceIndex, records: sourceRecords, options: sourceOptions)
                    if command == "curate", let id = options["--sample"], let status = options["--status"], let cached = records.first(where: { $0.sample["sample_id"] as? String == id }) {
                        var sample = cached.sample
                        sample["status"] = status
                        sample["review_status"] = ["approved": "user_approved", "excluded": "user_excluded", "unreviewed": "unreviewed"][status]
                        if let note = options["--note"] { sample["curation_note"] = note }
                        var entries = index["samples"] as? [Object] ?? []
                        for offset in entries.indices where entries[offset]["sample_id"] as? String == id { entries[offset]["status"] = status }
                        index["samples"] = entries
                        try commit(root, updates: [(cached.path, sample), (root.appendingPathComponent("dataset.json"), index)])
                    }
                    return try jsonString(response)
                }
            }
            if command != "dataset", command != "create-dataset" {
                try checkIndex(root, expected: options["--expected-index-sha256"])
                guard index["native_size_preparation"] == nil else { throw StudioError("Open the original dataset before managing its information.") }
            }
            if command == "curate" { return try jsonString(curate(root, index: index, records: records, options: options)) }
            if command == "remove-missing" { return try jsonString(removeMissing(root, index: index, records: records, options: options)) }
            if command == "add-material" {
                var paths: [String: URL] = [:]
                for role in ["input", "height", "roughness", "normal"] { if let path = options["--" + role] { paths[role] = URL(fileURLWithPath: path) } }
                let material = try sourceMaterial(options["--name"] ?? "", paths: paths, convention: options["--normal-convention"] ?? "opengl")
                return try jsonString(register(root, index: index, records: records, materials: [material], options: options))
            }
            if command == "scan-folder" { return try jsonString(scan(root, index: index, records: records, options: options)) }
            if command == "import-folder" { return try jsonString(importFolder(root, index: index, records: records, options: options)) }
            if command == "edit-dataset" {
                if let name = options["--name"] { index["name"] = try checkedName(name) }
                if let description = options["--description"] { index["description"] = try checkedDescription(description) }
                if let size = options["--training-size"] { index["training_size"] = try checkedSize(size) }
                var settings = try validation(index["validation"] as? Object)
                if let subject = options["--validation-subject"] {
                    guard records.contains(where: { subjectID($0.sample) == subject }) else { throw StudioError("The selected subject folder is no longer in this dataset.") }
                }
                if options["--validation-subject"] != nil, options["--subject-validation"] != "automatic" {
                    let reviews = try optionalObject(root.appendingPathComponent(".material-size-reviews.json"))
                    let plan = try regionPlan(records, size: index["training_size"] as? Int ?? 1024, reviews: reviews, target: "height", settings: settings)
                    var flags = settings["folders"] as? [String: Bool] ?? [:]
                    for item in plan["subjects"] as? [Object] ?? [] { if let id = item["subject_id"] as? String { flags[id] = item["selected"] as? Bool ?? false } }
                    settings["folders"] = flags
                }
                settings = try editedValidation(settings, options: options)
                if let subject = options["--validation-subject"], options["--subject-validation"] == "enabled" {
                    let plan = try regionPlan(records, size: index["training_size"] as? Int ?? 1024,
                        reviews: optionalObject(root.appendingPathComponent(".material-size-reviews.json")), target: "height", settings: settings)
                    let subjects = plan["subjects"] as? [Object] ?? []
                    guard subjects.contains(where: { $0["subject_id"] as? String == subject && $0["available"] as? Bool == true }) else { throw StudioError("This folder cannot supply a different validation crop at this resolution.") }
                    let flags = settings["folders"] as? [String: Bool] ?? [:]
                    guard subjects.filter({ flags[$0["subject_id"] as? String ?? ""] == true }).count <= (plan["validation_limit"] as? Int ?? 0) else { throw StudioError("Folder flags exceed the validation limit. Disable another folder or raise the limit.") }
                }
                index["validation"] = settings; index["updated_utc"] = timestamp()
                try commit(root, updates: [(root.appendingPathComponent("dataset.json"), index)])
            } else if command == "remove-material" {
                let selected = records.filter { $0.sample["material_id"] as? String == options["--material"] }
                guard !selected.isEmpty else { throw StudioError("Select a material in the open dataset before removing it.") }
                index["samples"] = (index["samples"] as? [Object] ?? []).filter { $0["material_id"] as? String != options["--material"] }
                var tombstones = index["removed_materials"] as? [Object] ?? []
                for record in selected {
                    let removed: Object = ["material_id": record.sample["material_id"] ?? NSNull(), "source_set_id": record.sample["source_set_id"] ?? NSNull()]
                    if !tombstones.contains(where: { NSDictionary(dictionary: $0).isEqual(to: removed) }) { tombstones.append(removed) }
                }
                index["removed_materials"] = tombstones; index["updated_utc"] = timestamp()
                try commit(root, updates: [(root.appendingPathComponent("dataset.json"), index)])
            } else if command == "validate-delete" {
                return try jsonString(deletion(root, index: index, records: records))
            }
            let refreshed = try object(root.appendingPathComponent("dataset.json"))
            return try jsonString(info(root, index: refreshed, records: readRecords(root, index: refreshed), options: options))
        }
    }

    private static func info(_ root: URL, index: Object, records original: [Record], options: [String: String]) throws -> Object {
        let preparation = index["native_size_preparation"] as? Object
        let originalRoot = (preparation?["source_dataset_path"] as? String).map { URL(fileURLWithPath: $0) } ?? root
        let metadata = preparation != nil && FileManager.default.fileExists(atPath: originalRoot.appendingPathComponent("dataset.json").path) ? try object(originalRoot.appendingPathComponent("dataset.json")) : index
        let size = options["--review-size"].flatMap(Int.init) ?? metadata["training_size"] as? Int ?? options["--default-review-size"].flatMap(Int.init)
        let reviews = try optionalObject(root.appendingPathComponent(".material-size-reviews.json"))
        let settings = try validation(metadata["validation"] as? Object)
        var resolutions: Object = [:], plans: Object = [:]
        if preparation == nil, let size {
            guard sizes.contains(size) else { throw StudioError("Choose a supported native training size.") }
            for candidate in sizes {
                let shared = try regionPlan(original, size: candidate, reviews: reviews, target: "height", settings: settings)
                var targets: Object = ["height": shared]
                for target in ["roughness", "normal"] { targets[target] = planForTarget(shared, records: original, target: target) }
                resolutions[String(candidate)] = targets
            }
            plans = resolutions[String(size)] as? Object ?? [:]
        }
        let selected = plans[options["--target"] ?? "height"] as? Object ?? [:]
        let assignments = selected["assignments"] as? [String: Object] ?? [:]
        var records = preparation == nil && size != nil ? reviewRecords(original, size: size!) : original
        for (id, assignment) in assignments.sorted(by: { $0.key < $1.key }) where assignment["split"] as? String == "validation" {
            if var record = original.first(where: { $0.sample["material_id"] as? String == assignment["material_id"] as? String }) {
                record.sample["sample_id"] = id; record.sample["source_region_id"] = "validation"
                record.sample["sample_pixel_dimensions"] = [size!, size!]; record.sample["crop_rectangle_top_left_xywh"] = assignment["crop_rectangle"]
                records.append(record)
            }
        }
        var grouped: [String: Object] = [:]
        for record in records {
            var sample = record.sample
            let id = sample["sample_id"] as? String ?? ""
            if preparation == nil, let size {
                sample.merge(currentReview(reviews, key: "\(size):\(id)", sample: sample)) { _, new in new }
                if let assignment = assignments[id] { for key in ["split", "split_assignment", "validation_scope", "status"] { sample[key] = assignment[key] } }
                else { sample["split"] = "train"; sample["split_assignment"] = "automatic" }
            }
            var maps: Object = [:]
            for (role, filename) in sample["maps"] as? [String: String] ?? [:] {
                let details = (sample["map_metadata"] as? [String: Object])?[role] ?? [:]
                maps[role] = try mapInfo(filename, details: details, sample: sample, folder: record.path.deletingLastPathComponent(), originalRoot: originalRoot, role: role)
            }
            var variants: [Object] = []
            for variant in sample["input_variants"] as? [Object] ?? [] {
                let source = variant["source"] as? Object ?? [:]
                let inputSource = ((sample["map_metadata"] as? [String: Object])?["input"]?["source"] as? Object) ?? [:]
                var details: Object
                if source["file_sha256"] as? String == inputSource["file_sha256"] as? String, let input = maps["input"] as? Object { details = input }
                else { details = try mapInfo(variant["filename"] as? String ?? variant["path"] as? String ?? source["path"] as? String ?? "", details: variant, sample: sample, folder: record.path.deletingLastPathComponent(), originalRoot: originalRoot, role: "input") }
                details["variant_id"] = variant["variant_id"] ?? "default"; variants.append(details)
            }
            let material = sample["material_id"] as? String ?? ""
            let dimensions = sample["sample_pixel_dimensions"] as? [Int] ?? [0, 0]
            guard dimensions.count == 2 else { throw StudioError("Dataset sample dimensions are invalid.") }
            var group = grouped[material] ?? ["material_id": material, "name": sample["name"] ?? material.replacingOccurrences(of: "_", with: " "), "subject_id": subjectID(sample), "source_directory": sample["source_directory"] ?? NSNull(), "samples": [Object]()]
            var samples = group["samples"] as? [Object] ?? []
            samples.append(["sample_id": id, "status": sample["status"] ?? "unreviewed", "split": sample["split"] ?? "train", "split_assignment": sample["split_assignment"] ?? NSNull(), "width": dimensions[0], "height": dimensions[1], "maps": maps, "note": sample["curation_note"] ?? NSNull(), "input_variants": variants, "source_family_id": familyID(sample), "source_set_id": sample["source_set_id"] ?? NSNull(), "source_region_id": sample["source_region_id"] ?? NSNull(), "available_targets": sample["available_targets"] ?? ["height", "roughness", "normal"].filter { maps[$0] != nil }])
            group["samples"] = samples; grouped[material] = group
        }
        let eligible = sizes.filter { candidate in original.contains { record in
            !["excluded", "rejected"].contains(record.sample["status"] as? String ?? "") && sources(record.sample).allSatisfy { min($0["width"] as? Int ?? 0, $0["height"] as? Int ?? 0) >= candidate } && !sources(record.sample).isEmpty
        } }
        func withoutAssignments(_ values: Object) -> Object { values.mapValues { value in var plan = value as? Object ?? [:]; plan.removeValue(forKey: "assignments"); return plan } }
        return ["dataset_path": root.path, "index_sha256": try digest(root.appendingPathComponent("dataset.json")), "name": metadata["name"] ?? originalRoot.lastPathComponent, "description": metadata["description"] ?? "", "review_sha256": try optionalDigest(originalRoot.appendingPathComponent(".material-size-reviews.json")) ?? hash(Data()), "training_size": metadata["training_size"] ?? NSNull(), "review_size": size as Any? ?? NSNull(), "validation": settings, "subjects": selected["subjects"] ?? [], "material_count": grouped.count, "sample_count": records.count, "source_set_count": original.count, "training_plans": withoutAssignments(plans), "resolution_plans": resolutions.mapValues { withoutAssignments($0 as? Object ?? [:]) }, "supported_training_sizes": eligible, "validation_scope": selected.isEmpty ? index["validation_scope"] ?? NSNull() : "known_subject_diagnostic", "cross_size_validation_notice": preparation?["cross_size_validation_notice"] ?? NSNull(), "automatic_validation": index["automatic_validation"] ?? NSNull(), "preparation": NSNull(), "materials": grouped.keys.sorted().compactMap { grouped[$0] }]
    }
    private static func mapInfo(_ filename: String, details: Object, sample: Object, folder: URL, originalRoot: URL, role: String) throws -> Object {
        let source = details["source"] as? Object ?? [:]
        var path: URL
        if details["storage"] as? String == "source_reference" {
            guard filename.hasPrefix("/"), filename == source["path"] as? String,
                  details["sample_sha256"] as? String == source["file_sha256"] as? String,
                  (source["file_sha256"] as? String)?.count == 64 else { throw StudioError("Original map reference differs from its recorded source identity.") }
            path = URL(fileURLWithPath: filename)
        } else { path = try contained(folder, filename) }
        var original = (source["path"] as? String).map { URL(fileURLWithPath: $0) }
        if let current = original, !FileManager.default.fileExists(atPath: current.path), let recovered = try recoverSource(source, root: originalRoot) {
            original = recovered; if details["storage"] as? String == "source_reference" { path = recovered }
        }
        let header = try? NativePNG.inspect(path)
        return ["path": path.path, "sha256": details["sample_sha256"] ?? NSNull(), "source_bits": details["sample_bits"] ?? source["sample_bits"] ?? NSNull(), "encoding": details["encoding"] ?? NSNull(), "width": header?.width as Any? ?? NSNull(), "height": header?.height as Any? ?? NSNull(), "original_source_path": original?.path as Any? ?? NSNull(), "original_source_sha256": source["file_sha256"] ?? NSNull(), "original_source_width": source["width"] ?? NSNull(), "original_source_height": source["height"] ?? NSNull(), "crop_rectangle": sample["crop_rectangle_top_left_xywh"] ?? NSNull(), "source_normal_convention": role == "normal" ? (details["storage"] as? String == "source_reference" && source["suffix"] as? String == "nor_dx" ? "directx" : "opengl") : NSNull(), "original_normal_convention": role == "normal" ? (source["suffix"] as? String == "nor_dx" ? "directx" : "opengl") : NSNull()]
    }
    private static func recoverSource(_ source: Object, root: URL) throws -> URL? {
        guard let expected = source["file_sha256"] as? String, let path = source["path"] as? String else { return nil }
        let filename = source["filename"] as? String ?? URL(fileURLWithPath: path).lastPathComponent
        for folder in [root.appendingPathComponent("sources"), URL(fileURLWithPath: path).deletingLastPathComponent()] {
            guard let entries = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey]) else { continue }
            for case let candidate as URL in entries where candidate.lastPathComponent == filename {
                if (try? digest(candidate)) == expected { return candidate.resolvingSymlinksInPath().standardizedFileURL }
            }
        }
        return nil
    }

    private static func regionPlan(_ records: [Record], size: Int, reviews: Object, target: String, settings: Object) throws -> Object {
        let materials = Dictionary(records.filter { !sources($0.sample).isEmpty && sources($0.sample).allSatisfy { min($0["width"] as? Int ?? 0, $0["height"] as? Int ?? 0) >= size } }.map { ($0.sample["material_id"] as? String ?? "", $0) }, uniquingKeysWith: { _, new in new })
        var assignments: [String: Object] = [:], subjects: [String: Set<String>] = [:], occupied: [String: [[Double]]] = [:]
        func eligible(_ sample: Object) -> Bool {
            guard let map = (sample["map_metadata"] as? [String: Object])?[target] else { return false }
            if target == "height", (map["source"] as? Object)?["sample_bits"] as? Int != 16 { return false }
            return (sample["available_targets"] as? [String] ?? [target]).contains(target)
        }
        for material in materials.keys.sorted() {
            let sample = materials[material]!.sample
            let dimensions = sample["source_pixel_dimensions"] as? [Int] ?? [0, 0]
            guard dimensions.count == 2 else { throw StudioError("Invalid original material dimensions.") }
            let subject = subjectID(sample), family = familyID(sample)
            for (region, rectangle) in cropLayout(dimensions, size: size) {
                let identity = material + "_" + region
                let review = currentReview(reviews, key: "\(size):\(identity)", sample: sample)
                let status = review["status"] as? String ?? sample["status"] as? String ?? "unreviewed"
                let included = !["excluded", "rejected"].contains(status)
                assignments[identity] = ["material_id": material, "region": region, "subject_id": subject, "split": "train", "split_assignment": "automatic", "validation_scope": "known_subject_diagnostic", "eligible": included && eligible(sample), "target_available": eligible(sample), "status": status, "crop_rectangle": rectangle]
                if included {
                    subjects[subject, default: []].insert(material)
                    let normalized = normalize(rectangle, dimensions)
                    occupied["subject:" + subject, default: []].append(normalized)
                    occupied["family:" + family, default: []].append(normalized)
                }
            }
        }
        var cap = Int(floor(Double(subjects.count) * (settings["percent"] as? Double ?? 5) / 100))
        if let max = settings["max_crops"] as? Int, max > 0 { cap = min(cap, max) }
        if settings["enabled"] as? Bool == false { cap = 0 }
        let flags = settings["folders"] as? [String: Bool] ?? [:]
        var candidates: [String: (String, [Int], Bool)] = [:], subjectPlans: [String: Object] = [:]
        for subject in subjects.keys.sorted() {
            let mids = subjects[subject]!.sorted { a, b in
                let ad = materials[a]!.sample["source_pixel_dimensions"] as? [Int] ?? [0, 0]
                let bd = materials[b]!.sample["source_pixel_dimensions"] as? [Int] ?? [0, 0]
                return ad[0] * ad[1] == bd[0] * bd[1] ? a < b : ad[0] * ad[1] > bd[0] * bd[1]
            }
            for material in mids {
                let sample = materials[material]!.sample, dimensions = sample["source_pixel_dimensions"] as! [Int]
                let blocked = assignments.values.filter { $0["material_id"] as? String == material && !["excluded", "rejected"].contains($0["status"] as? String ?? "") }.compactMap { ($0["crop_rectangle"] as? [Int]).map { normalize($0, dimensions) } }
                let allViews = (occupied["subject:" + subject] ?? []) + (occupied["family:" + familyID(sample)] ?? [])
                var corners: [[Int]] = []
                if let spare = spareRectangle(dimensions, size: size, occupied: blocked) { corners.append(spare) }
                corners += [[0, dimensions[1] - size, size, size], [dimensions[0] - size, dimensions[1] - size, size, size], [0, 0, size, size], [dimensions[0] - size, 0, size, size]]
                if let rectangle = corners.first(where: { !allViews.contains(normalize($0, dimensions)) }) {
                    candidates[subject] = (material, rectangle, blocked.contains { overlaps(normalize(rectangle, dimensions), $0) }); break
                }
            }
            subjectPlans[subject] = ["subject_id": subject, "name": URL(fileURLWithPath: subject).lastPathComponent, "selected": false, "available": candidates[subject] != nil, "preference": flags[subject] as Any? ?? NSNull(), "reason": candidates[subject] == nil ? "No different crop fits this source at the selected resolution." : NSNull()]
        }
        let ordered = candidates.keys.filter { flags[$0] != false }.sorted { a, b in
            if (flags[a] == true) != (flags[b] == true) { return flags[a] == true }
            if candidates[a]!.2 != candidates[b]!.2 { return !candidates[a]!.2 }
            return hash(Data(a.utf8)) < hash(Data(b.utf8))
        }
        var checks = Set<String>()
        for subject in ordered.prefix(cap) {
            let (material, rectangle, _) = candidates[subject]!, sample = materials[material]!.sample
            let id = material + "_validation"
            assignments[id] = ["material_id": material, "region": "validation", "subject_id": subject, "split": "validation", "split_assignment": "automatic", "validation_scope": "known_subject_diagnostic", "eligible": eligible(sample), "target_available": eligible(sample), "status": "unreviewed", "crop_rectangle": rectangle]
            subjectPlans[subject]?["selected"] = true; checks.insert(familyID(sample))
        }
        let values = Array(assignments.values)
        return ["assignments": assignments, "subjects": subjectPlans.keys.sorted().compactMap { subjectPlans[$0] }, "size": size, "crop_count": values.filter { $0["split"] as? String == "train" }.count, "source_set_count": materials.count, "subject_count": subjects.count, "train_count": values.filter { $0["eligible"] as? Bool == true && $0["split"] as? String == "train" }.count, "validation_count": values.filter { $0["eligible"] as? Bool == true && $0["split"] as? String == "validation" }.count, "excluded_count": values.filter { ["excluded", "rejected"].contains($0["status"] as? String ?? "") }.count, "unavailable_target_count": values.filter { $0["target_available"] as? Bool == false }.count, "undersized_source_set_count": Set(records.compactMap { $0.sample["material_id"] as? String }).count - materials.count, "shared_validation_count": min(cap, ordered.count), "validation_limit": cap, "validation_candidate_count": candidates.count, "regional_families": checks.sorted()]
    }
    /// Region membership and validation selections are shared by all maps.
    /// Compute that geometry once, then adjust only target availability counts.
    private static func planForTarget(_ shared: Object, records: [Record], target: String) -> Object {
        var plan = shared
        let available = Dictionary(records.map { record in
            let sample = record.sample
            let supported = (sample["map_metadata"] as? [String: Object])?[target] != nil && (sample["available_targets"] as? [String] ?? [target]).contains(target)
            return (sample["material_id"] as? String ?? "", supported)
        }, uniquingKeysWith: { _, new in new })
        let assignments = (shared["assignments"] as? [String: Object] ?? [:]).mapValues { value in
            var assignment = value
            let targetAvailable = available[value["material_id"] as? String ?? ""] ?? false
            assignment["target_available"] = targetAvailable
            assignment["eligible"] = targetAvailable && !["excluded", "rejected"].contains(value["status"] as? String ?? "")
            return assignment
        }
        let values = Array(assignments.values)
        plan["assignments"] = assignments
        plan["train_count"] = values.filter { $0["eligible"] as? Bool == true && $0["split"] as? String == "train" }.count
        plan["validation_count"] = values.filter { $0["eligible"] as? Bool == true && $0["split"] as? String == "validation" }.count
        plan["unavailable_target_count"] = values.filter { $0["target_available"] as? Bool == false }.count
        return plan
    }
    private static func cropLayout(_ dimensions: [Int], size: Int) -> [(String, [Int])] {
        guard dimensions.count == 2, min(dimensions[0], dimensions[1]) >= size else { return [] }
        let width = dimensions[0], height = dimensions[1]
        if width == size && height == size { return [("full", [0, 0, size, size])] }
        if min(width, height) >= 8192 { return [("top_left", [0, 0, size, size]), ("top_right", [width - size, 0, size, size]), ("bottom_right", [width - size, height - size, size, size])] }
        return [("center", [(width - size) / 2, (height - size) / 2, size, size])]
    }
    private static func reviewRecords(_ originals: [Record], size: Int) -> [Record] {
        originals.flatMap { original -> [Record] in
            let layout = cropLayout(original.sample["source_pixel_dimensions"] as? [Int] ?? [], size: size)
            if layout.isEmpty { return [original] }
            return layout.map { region, rectangle in
                var record = original
                record.sample["sample_id"] = (original.sample["material_id"] as? String ?? "") + "_" + region
                record.sample["sample_pixel_dimensions"] = [size, size]
                record.sample["crop_rectangle_top_left_xywh"] = rectangle
                record.sample["source_region_id"] = region
                return record
            }
        }
    }
    private static func normalize(_ rectangle: [Int], _ dimensions: [Int]) -> [Double] { [Double(rectangle[0]) / Double(dimensions[0]), Double(rectangle[1]) / Double(dimensions[1]), Double(rectangle[2]) / Double(dimensions[0]), Double(rectangle[3]) / Double(dimensions[1])] }
    private static func overlaps(_ a: [Double], _ b: [Double]) -> Bool { a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3] }
    private static func spareRectangle(_ dimensions: [Int], size: Int, occupied: [[Double]]) -> [Int]? {
        let width = dimensions[0], height = dimensions[1]
        var xs: Set<Int> = [0, width - size], ys: Set<Int> = [0, height - size]
        for r in occupied { xs.insert(Int(ceil((r[0] + r[2]) * Double(width)))); xs.insert(Int(floor(r[0] * Double(width))) - size); ys.insert(Int(ceil((r[1] + r[3]) * Double(height)))); ys.insert(Int(floor(r[1] * Double(height))) - size) }
        for y in ys.sorted(by: >) { for x in xs.sorted() where x >= 0 && x <= width - size && y >= 0 && y <= height - size {
            let rectangle = [x, y, size, size]
            if !occupied.contains(where: { overlaps(normalize(rectangle, dimensions), $0) }) { return rectangle }
        } }
        return nil
    }
    private static func subjectID(_ sample: Object) -> String { (sample["source_directory"] as? String).map { canonical(URL(fileURLWithPath: $0)).path } ?? sample["source_family_id"] as? String ?? sample["asset_family_id"] as? String ?? sample["material_id"] as? String ?? "" }
    private static func familyID(_ sample: Object) -> String { sample["asset_family_id"] as? String ?? sample["source_family_id"] as? String ?? sample["material_id"] as? String ?? "" }
    private static func sources(_ sample: Object, target: String? = nil) -> [Object] {
        let maps = (sample["map_metadata"] as? [String: Object] ?? [:]).filter { target == nil || $0.key == "input" || $0.key == target }
        let candidates = Array(maps.values).map { $0["source"] as? Object ?? [:] } + (sample["input_variants"] as? [Object] ?? []).map { $0["source"] as? Object ?? [:] }
        var seen = Set<String>()
        return candidates.filter { seen.insert(($0["path"] as? String ?? "") + "|" + ($0["file_sha256"] as? String ?? "")).inserted }
    }
    private static func sourceBinding(_ sample: Object) -> String {
        if sample["native_size_preparation"] != nil, let binding = sample["source_binding_sha256"] as? String { return binding }
        var maps: Object = [:]
        for (role, details) in sample["map_metadata"] as? [String: Object] ?? [:] { maps[role] = (details["source"] as? Object)?["file_sha256"] ?? NSNull() }
        let colors = Set((sample["input_variants"] as? [Object] ?? []).map { ($0["source"] as? Object)?["file_sha256"] as? String ?? "" }).sorted()
        // This checksum is the existing cross-language manifest contract.
        let identities: Object = ["maps": maps, "colors": colors]
        guard let data = try? JSONSerialization.data(withJSONObject: identities, options: [.sortedKeys, .withoutEscapingSlashes]), let compact = String(data: data, encoding: .utf8) else { return "" }
        let canonicalJSON = spacedJSON(compact)
        return hash(Data(canonicalJSON.utf8))
    }
    private static func currentReview(_ reviews: Object, key: String, sample: Object) -> Object {
        let review = reviews[key] as? Object ?? [:]
        guard let binding = review["source_binding_sha256"] as? String else { return review }
        return sourceBinding(sample) == binding ? review : [:]
    }
    /// Sorted JSON with fixed separators is part of the dataset evidence format.
    private static func spacedJSON(_ input: String) -> String {
        var quoted = false, escaped = false, output = ""
        for character in input {
            output.append(character)
            if escaped { escaped = false; continue }
            if character == "\\", quoted { escaped = true; continue }
            if character == "\"" { quoted.toggle() }
            if !quoted, character == ":" || character == "," { output.append(" ") }
        }
        return output
    }

    private static func curate(_ root: URL, index sourceIndex: Object, records original: [Record], options: [String: String]) throws -> Object {
        guard let id = options["--sample"], let status = options["--status"], ["approved", "excluded", "unreviewed"].contains(status) else { throw StudioError("Select an exact sample and review status.") }
        let reviewPath = root.appendingPathComponent(".material-size-reviews.json")
        if let expected = options["--expected-review-sha256"], try optionalDigest(reviewPath) ?? hash(Data()) != expected { throw StudioError("Dataset reviews changed since selection; reload them before curation.") }
        var index = sourceIndex, reviews = try optionalObject(reviewPath)
        let size = options["--review-size"].flatMap(Int.init)
        let records = size.map { reviewRecords(original, size: $0) } ?? original
        let matches = records.filter { $0.sample["sample_id"] as? String == id }
        guard matches.count == 1 else { throw StudioError("Select one exact indexed sample ID.") }
        let selected = matches[0], family = familyID(selected.sample), split = options["--split"]
        guard split == nil || ["train", "validation", "automatic"].contains(split!) else { throw StudioError("Choose a training, validation or automatic split.") }
        if split == "automatic" {
            let siblings = original.filter { familyID($0.sample) == family }, ids = siblings.compactMap { $0.sample["material_id"] as? String }
            var updates: [(URL, Object)] = []
            var entries = index["samples"] as? [Object] ?? []
            for record in siblings {
                var sample = record.sample; sample.removeValue(forKey: "split_assignment"); sample["split"] = "train"
                updates.append((record.path, sample))
                for offset in entries.indices where entries[offset]["sample_id"] as? String == sample["sample_id"] as? String { entries[offset]["split"] = "train" }
            }
            for key in reviews.keys {
                let parts = key.split(separator: ":", maxSplits: 1)
                if parts.count == 2, size == nil || String(parts[0]) == String(size!), ids.contains(where: { parts[1].hasPrefix($0 + "_") }) {
                    var review = reviews[key] as? Object ?? [:]; review.removeValue(forKey: "split_assignment"); review.removeValue(forKey: "split"); reviews[key] = review
                }
            }
            index["samples"] = entries
            if FileManager.default.fileExists(atPath: reviewPath.path) { updates.append((reviewPath, reviews)) }
            try commit(root, updates: updates + [(root.appendingPathComponent("dataset.json"), index)])
            return ["dataset_path": root.path, "sample_id": id, "split": "automatic"]
        }
        let reviewStatus = ["approved": "user_approved", "excluded": "user_excluded", "unreviewed": "unreviewed"][status]!
        let virtual = size.map { !cropLayout(selected.sample["source_pixel_dimensions"] as? [Int] ?? [], size: $0).isEmpty } ?? false
        if virtual, let size {
            let key = "\(size):\(id)"
            var review = currentReview(reviews, key: key, sample: selected.sample)
            review["status"] = status; review["review_status"] = reviewStatus; review["source_binding_sha256"] = sourceBinding(selected.sample)
            review["split"] = split ?? review["split"] ?? selected.sample["split"] ?? "train"
            if split != nil { review["split_assignment"] = "manual" }
            if let note = options["--note"] { review["curation_note"] = note }
            reviews[key] = review
            if let split {
                let siblings = records.filter { familyID($0.sample) == family }
                if Set(siblings.map { $0.sample["source_set_id"] as? String ?? $0.sample["material_id"] as? String ?? "" }).count > 1 {
                    for record in siblings {
                        let siblingKey = "\(size):\(record.sample["sample_id"] as? String ?? "")"
                        var sibling = currentReview(reviews, key: siblingKey, sample: record.sample)
                        sibling["split"] = split; sibling["split_assignment"] = "manual"; sibling["source_binding_sha256"] = sourceBinding(record.sample)
                        reviews[siblingKey] = sibling
                    }
                }
                let training = siblings.filter { currentReview(reviews, key: "\(size):\($0.sample["sample_id"] as? String ?? "")", sample: $0.sample)["split"] as? String ?? $0.sample["split"] as? String == "train" }
                let checks = siblings.filter { currentReview(reviews, key: "\(size):\($0.sample["sample_id"] as? String ?? "")", sample: $0.sample)["split"] as? String ?? $0.sample["split"] as? String == "validation" }
                try checkSplitOverlap(training, checks)
            }
            try commit(root, updates: [(reviewPath, reviews)])
            return ["dataset_path": root.path, "sample_id": id, "status": status, "split": review["split"] ?? "train", "review_sha256": try digest(reviewPath)]
        }
        var selectedSample = selected.sample; selectedSample["status"] = status; selectedSample["review_status"] = reviewStatus
        if let note = options["--note"] { selectedSample["curation_note"] = note }
        var entries = index["samples"] as? [Object] ?? [], changed: [(URL, Object)] = []
        let siblings = original.filter { familyID($0.sample) == family }
        for record in original {
            let isSelected = record.sample["sample_id"] as? String == id
            var sample = isSelected ? selectedSample : record.sample
            if split != nil, familyID(sample) == family { sample["split"] = split; sample["split_assignment"] = "manual" }
            if isSelected || (split != nil && familyID(sample) == family) {
                changed.append((record.path, sample))
                for offset in entries.indices where entries[offset]["sample_id"] as? String == sample["sample_id"] as? String { entries[offset]["status"] = sample["status"]; entries[offset]["split"] = sample["split"] }
            }
        }
        if split != nil {
            let final = siblings.map { record -> Record in var result = record; result.sample["split"] = split; return result }
            try checkSplitOverlap(final.filter { $0.sample["split"] as? String == "train" }, final.filter { $0.sample["split"] as? String == "validation" })
        }
        if let size, reviews.removeValue(forKey: "\(size):\(id)") != nil { changed.append((reviewPath, reviews)) }
        index["samples"] = entries; try commit(root, updates: changed + [(root.appendingPathComponent("dataset.json"), index)])
        return ["dataset_path": root.path, "sample_id": id, "status": status, "split": split ?? selected.sample["split"] ?? "train", "index_sha256": try digest(root.appendingPathComponent("dataset.json"))]
    }
    private static func checkSplitOverlap(_ training: [Record], _ checks: [Record]) throws {
        for check in checks { for train in training {
            if let a = check.sample["crop_rectangle_top_left_xywh"] as? [Int], let b = train.sample["crop_rectangle_top_left_xywh"] as? [Int], a.count == 4, b.count == 4, overlaps(a.map(Double.init), b.map(Double.init)) { throw StudioError("Validation cannot overlap training pixels from the same source family.") }
        } }
    }
    private static func removeMissing(_ root: URL, index initial: Object, records originals: [Record], options: [String: String]) throws -> Object {
        guard let id = options["--sample"], let requested = options["--path"] else { throw StudioError("Choose the exact missing source map.") }
        let requestedPath = URL(fileURLWithPath: requested).resolvingSymlinksInPath().standardizedFileURL.path
        let records = options["--review-size"].flatMap(Int.init).map { reviewRecords(originals, size: $0) } ?? originals
        var response: Object = ["dataset_path": root.path, "sample_id": id, "removed": false]
        guard let record = records.first(where: { $0.sample["sample_id"] as? String == id || ($0.sample["material_id"] as? String ?? "") + "_validation" == id }) else { return response }
        let details = record.sample["map_metadata"] as? [String: Object] ?? [:]
        let matched = details.filter { _, value in (value["source"] as? Object)?["path"] as? String == requestedPath }
        let variants = (record.sample["input_variants"] as? [Object] ?? []).filter { ($0["source"] as? Object)?["path"] as? String == requestedPath }
        guard matched.count == 1 || (matched.isEmpty && variants.count == 1) else { throw StudioError("Missing map path differs from the selected listed material.") }
        // Only ENOENT authorizes membership removal; access/IO errors do not.
        var state = stat()
        if stat(requestedPath, &state) == 0 { return response }
        guard errno == ENOENT else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let source = (matched.values.first ?? variants[0])["source"] as? Object ?? [:]
        if let recovered = try recoverSource(source, root: root) { response["recoverable"] = true; response["source_path"] = recovered.path; return response }
        var index = initial
        if matched.isEmpty {
            let variant = variants[0], variantID = variant["variant_id"] as? String ?? ""
            var stored = try object(record.path)
            stored["input_variants"] = (stored["input_variants"] as? [Object] ?? []).filter { $0["variant_id"] as? String != variantID }
            var tombstones = index["removed_missing_color_variants"] as? [Object] ?? []
            tombstones.append(["source_set_id": record.sample["source_set_id"] ?? NSNull(), "material_id": record.sample["material_id"] ?? NSNull(), "variant_id": variantID, "path": source["path"] ?? NSNull(), "file_sha256": source["file_sha256"] ?? NSNull()]); index["removed_missing_color_variants"] = tombstones
            try commit(root, updates: [(record.path, stored), (root.appendingPathComponent("dataset.json"), index)])
            response["removed_variant_id"] = variantID
        } else {
            let material = record.sample["material_id"] as? String
            index["samples"] = (index["samples"] as? [Object] ?? []).filter { $0["material_id"] as? String != material }
            var tombstones = index["removed_missing_sources"] as? [Object] ?? []
            tombstones.append(["source_set_id": record.sample["source_set_id"] ?? NSNull(), "material_id": material as Any? ?? NSNull(), "path": source["path"] ?? NSNull(), "file_sha256": source["file_sha256"] ?? NSNull()]); index["removed_missing_sources"] = tombstones
            try commit(root, updates: [(root.appendingPathComponent("dataset.json"), index)])
        }
        response["removed"] = true; response["source_images_modified"] = false; response["index_sha256"] = try digest(root.appendingPathComponent("dataset.json")); return response
    }
    private static func fileState(_ path: URL) throws -> Object {
        var state = stat()
        guard stat(path.path, &state) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        return ["size": Int64(state.st_size), "mtime_ns": Int64(state.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(state.st_mtimespec.tv_nsec), "ctime_ns": Int64(state.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(state.st_ctimespec.tv_nsec)]
    }
    private static func sourceSummary(_ requested: URL, cached: Object? = nil) throws -> Object {
        try Task.checkCancellation()
        let path = requested.resolvingSymlinksInPath().standardizedFileURL, before = try fileState(path)
        if let cached, let state = cached["source_stat"] as? Object, NSDictionary(dictionary: state).isEqual(to: before) { return compactMetadata(cached) as! Object }
        let data = try Data(contentsOf: path, options: .mappedIfSafe)
        var result = try NativePNG.sourceMetadata(data)
        let after = try fileState(path)
        guard NSDictionary(dictionary: before).isEqual(to: after) else { throw StudioError("Original map changed while inspecting: \(path.path)") }
        result["file_sha256"] = hash(data)
        result["filename"] = path.lastPathComponent; result["file_bytes"] = after["size"]; result["source_stat"] = after
        return result
    }
    private static func matchesPublishedMD5(_ url: URL, expected: String?) throws -> Bool {
        guard let expected else { return false }
        // Provider download audits sometimes require MD5. Compute it only for
        // that audit, without persisting a second checksum on every map.
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return Insecure.MD5.hash(data: data).map { String(format: "%02x", $0) }.joined() == expected.lowercased()
    }
    private static func identifier(_ value: String) -> String {
        let ascii = value.decomposedStringWithCompatibilityMapping.unicodeScalars.filter { $0.value < 128 }.map(String.init).joined().lowercased()
        let cleaned = ascii.replacingOccurrences(of: "[^a-z0-9]+", with: "_", options: .regularExpression).trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return cleaned.isEmpty ? "material_" + hash(Data(value.utf8)).prefix(16) : cleaned
    }
    private static func encoding(_ role: String, source: Object) -> String { role != "input" ? "linear_data" : source["png_gamma"] as? Double == 1 && source["srgb_rendering_intent"] == nil ? "linear" : "source_srgb_assumed" }
    private static func sourceMaterial(_ name: String, paths: [String: URL], convention: String = "opengl") throws -> Object {
        let name = try checkedName(name)
        guard paths["input"] != nil, ["height", "roughness", "normal"].contains(where: { paths[$0] != nil }) else { throw StudioError("Choose an input map and at least one height, normal or roughness map.") }
        let family = identifier(name)
        var maps: [String: Object] = [:], dimensions: [Int]?
        for role in paths.keys.sorted() {
            let path = paths[role]!.resolvingSymlinksInPath().standardizedFileURL
            var source = try sourceSummary(path)
            guard !["input", "normal"].contains(role) || [3, 4].contains(source["channels"] as? Int ?? 0) else { throw StudioError("Color and normal maps must contain native RGB or RGBA channels.") }
            let actual = [source["width"] as? Int ?? 0, source["height"] as? Int ?? 0]
            if let dimensions, dimensions != actual { throw StudioError("All maps must have exactly matching native dimensions; no resizing is applied.") }
            dimensions = actual
            source["path"] = path.path; source["suffix"] = ["input": "diff", "height": "disp", "roughness": "rough", "normal": convention.lowercased() == "directx" ? "nor_dx" : "nor_gl"][role]
            source["source_family_id"] = family; source["asset_family_id"] = family; maps[role] = source
        }
        let size = dimensions!, width = size[0], height = size[1], resolution = width == height && width >= 1024 && width % 1024 == 0 ? "\(width / 1024)k" : "\(width)x\(height)"
        for role in maps.keys { maps[role]?["resolution_label"] = resolution }
        maps["input"]?["variant_id"] = "color_default"
        return ["name": name, "material_id": family + "_" + resolution, "source_family_id": family, "source_set_id": "\(family)_\(width)x\(height)", "source_directory": paths["input"]!.resolvingSymlinksInPath().deletingLastPathComponent().path, "common_pixel_dimensions": size, "resolution_label": resolution, "maps": maps, "input_variants": [maps["input"]!], "source_files": Array(maps.values), "warnings": [String](), "ignored_files": [String](), "problems": [String]()]
    }
    private static func sourceRecord(_ material: Object) throws -> Object {
        guard let materialID = material["material_id"] as? String, identifier(materialID) == materialID,
              let maps = material["maps"] as? [String: Object], let dimensions = material["common_pixel_dimensions"] as? [Int], dimensions.count == 2, maps["input"] != nil else { throw StudioError("Invalid native material source set.") }
        let variants = material["input_variants"] as? [Object] ?? []
        for source in Array(maps.values) + variants {
            guard [source["width"] as? Int ?? 0, source["height"] as? Int ?? 0] == dimensions else { throw StudioError("All maps and colors need matching native dimensions.") }
        }
        guard variants.allSatisfy({ [3, 4].contains($0["channels"] as? Int ?? 0) }), maps["normal"] == nil || [3, 4].contains(maps["normal"]?["channels"] as? Int ?? 0) else { throw StudioError("Diffuse and normal maps must contain RGB or RGBA integer channels.") }
        var filenames: [String: String] = [:], metadata: [String: Object] = [:]
        for (role, source) in maps {
            guard let path = source["path"] as? String else { throw StudioError("Missing original map path.") }
            filenames[role] = path
            metadata[role] = ["source": source, "storage": "source_reference", "filename": path, "sample_sha256": source["file_sha256"] ?? NSNull(), "sample_bits": source["sample_bits"] ?? NSNull(), "channels": source["channels"] ?? NSNull(), "encoding": encoding(role, source: source), "transforms": [Object](), "exact_source_crop": true]
        }
        let inputVariants: [Object] = try variants.map { source in
            guard let path = source["path"] as? String else { throw StudioError("Missing original color path.") }
            return ["source": source, "storage": "source_reference", "filename": path, "path": path, "sample_sha256": source["file_sha256"] ?? NSNull(), "sample_bits": source["sample_bits"] ?? NSNull(), "channels": source["channels"] ?? NSNull(), "encoding": encoding("input", source: source), "transforms": [Object](), "exact_source_crop": true, "variant_id": source["variant_id"] ?? "color_default"]
        }
        return ["schema_version": 2, "generator": "ipde-material-dataset-v2", "sample_id": materialID + "_full", "name": material["name"] ?? material["source_family_id"] ?? materialID, "material_id": materialID, "asset_family_id": material["source_family_id"] ?? materialID, "source_family_id": material["source_family_id"] ?? materialID, "source_set_id": material["source_set_id"] ?? materialID, "source_directory": material["source_directory"] ?? NSNull(), "source_resolution_label": material["resolution_label"] ?? "native", "sample_pixel_dimensions": dimensions, "source_pixel_dimensions": dimensions, "crop_rectangle_top_left_xywh": [0, 0] + dimensions, "status": "unreviewed", "split": "train", "review_status": "unreviewed", "maps": filenames, "map_metadata": metadata, "input_variants": inputVariants, "source_precision_verified": true, "crop_values_verified": true, "source_discovery_warnings": material["warnings"] ?? [], "ignored_source_files": material["ignored_files"] ?? [], "available_targets": ["height", "roughness", "normal"].filter { maps[$0] != nil && ($0 != "height" || (maps[$0]?["sample_bits"] as? Int ?? 0) >= 16) }]
    }
    private static func sourceIdentity(_ sample: Object) -> String {
        let maps = sample["map_metadata"] as? [String: Object] ?? [:]
        var entries: [[String]] = maps.keys.sorted().map { role in
            let source = maps[role]?["source"] as? Object ?? [:]
            return [role, (source["path"] as? String).map { URL(fileURLWithPath: $0).resolvingSymlinksInPath().path } ?? "", source["file_sha256"] as? String ?? ""]
        }
        let colorSources = (sample["input_variants"] as? [Object] ?? []).map { $0["source"] as? Object ?? [:] } + [maps["input"]?["source"] as? Object ?? [:]]
        let colors = Set(colorSources.map { ($0["path"] as? String ?? "") + "|" + ($0["file_sha256"] as? String ?? "") }).sorted()
        entries.append(contentsOf: colors.map { ["color", $0] })
        return hash((try? JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data())
    }
    private static func register(_ root: URL, index initial: Object, records: [Record], materials: [Object], options: [String: String]) throws -> Object {
        guard !materials.isEmpty else { throw StudioError("No paired diffuse and surface PNG maps were found.") }
        var index = initial, current = Dictionary(records.map { ($0.sample["material_id"] as? String ?? "", $0.sample) }, uniquingKeysWith: { _, new in new })
        var identities = Set(current.values.map(sourceIdentity)), pending: [String: Object] = [:], duplicates = 0
        let sizeChanged = options["--training-size"].map { index["training_size"] as? Int != Int($0) } ?? false
        if let size = options["--training-size"] { index["training_size"] = try checkedSize(size) }
        for material in materials {
            guard (material["problems"] as? [String] ?? []).isEmpty else { throw StudioError("Ambiguous source maps need explicit map selection.") }
            var sample = try sourceRecord(material), identity = sourceIdentity(sample), id = sample["material_id"] as! String
            if identities.contains(identity) { duplicates += 1; continue }
            guard current[id] == nil, pending[id] == nil else { throw StudioError("A different material already uses this name and native size. Choose another name.") }
            let folder = try contained(root, "samples/" + (sample["sample_id"] as! String))
            if let prior = try? object(folder.appendingPathComponent("sample.json")), sourceBinding(prior) == sourceBinding(sample) {
                for key in ["status", "split", "split_assignment", "review_status", "curation_note", "source_binding_sha256"] { if let value = prior[key] { sample[key] = value } }
            }
            pending[id] = sample; current[id] = sample; identities.insert(identity)
        }
        var entries = index["samples"] as? [Object] ?? [], updates: [(URL, Object)] = []
        let reviewsPath = root.appendingPathComponent(".material-size-reviews.json")
        var reviews = try optionalObject(reviewsPath), changedReviews = false
        for id in pending.keys.sorted() {
            let sample = pending[id]!, sampleID = sample["sample_id"] as! String
            let folder = try contained(root, "samples/" + sampleID)
            if let prior = try? object(folder.appendingPathComponent("sample.json")), sourceBinding(prior) != sourceBinding(sample) {
                for key in reviews.keys where key.split(separator: ":", maxSplits: 1).last?.hasPrefix(id + "_") == true { reviews.removeValue(forKey: key); changedReviews = true }
            }
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            updates.append((folder.appendingPathComponent("sample.json"), sample))
            entries.append(["sample_id": sampleID, "material_id": id, "status": sample["status"] ?? "unreviewed", "split": sample["split"] ?? "train", "path": "samples/" + sampleID])
        }
        if !pending.isEmpty || sizeChanged {
            index["samples"] = entries; index["updated_utc"] = timestamp()
            index["removed_materials"] = (index["removed_materials"] as? [Object] ?? []).filter { pending[$0["material_id"] as? String ?? ""] == nil }
            if changedReviews { updates.append((reviewsPath, reviews)) }
            try commit(root, updates: updates + [(root.appendingPathComponent("dataset.json"), index)])
        }
        var result = try info(root, index: index, records: readRecords(root, index: index), options: options)
        result["added_material_count"] = pending.count; result["duplicate_material_count"] = duplicates; return result
    }
    private static func inventory(_ folder: URL, dataset: URL) throws -> [[Any]] {
        var result: [[Any]] = [], failed: Error?
        guard let enumerator = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles], errorHandler: { _, error in failed = error; return false }) else { throw StudioError("Choose an existing folder of material maps.") }
        for case let file as URL in enumerator {
            try Task.checkCancellation()
            let values = try file.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
            if values.isDirectory == true {
                if file.resolvingSymlinksInPath() == dataset || FileManager.default.fileExists(atPath: file.appendingPathComponent("dataset.json").path) { enumerator.skipDescendants() }
                continue
            }
            guard values.isRegularFile == true, file.pathExtension.lowercased() == "png" else { continue }
            let state = try fileState(file)
            result.append([file.path, file.resolvingSymlinksInPath().path, state["size"]!, state["mtime_ns"]!, state["ctime_ns"]!])
        }
        if let failed { throw failed }
        return result.sorted { ($0[0] as? String ?? "") < ($1[0] as? String ?? "") }
    }
    private static func parsedProvider(_ filename: String) -> (String, String, String)? {
        let known = #"nor_gl|nor_dx|disp_gl|rough_ao|translucent|(?:diffuse|diff|color|col|albedo)(?:_?\d+)?|coll\d+|displacement|roughness|disp|rough|anisotropy_rotation|anisotropy_strength|spec_ior|spec|bump|metal|ao|arm"#
        let patterns = ["^(.+?)_(" + known + #")_(\d+k)(?: \(\d+\))?\.png$"#, #"^([A-Za-z][A-Za-z0-9]*)_([1248]K)-PNG_(Color|Displacement|NormalGL|NormalDX|Roughness)\.png$"#]
        for (offset, pattern) in patterns.enumerated() {
            guard let expression = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]), let match = expression.firstMatch(in: filename, range: NSRange(filename.startIndex..., in: filename)) else { continue }
            let fields = (1...3).map { String(filename[Range(match.range(at: $0), in: filename)!]).lowercased() }
            if offset == 1 { return (fields[0], ["color": "diff", "displacement": "disp", "normalgl": "nor_gl", "normaldx": "nor_dx", "roughness": "rough"][fields[2]]!, fields[1]) }
            return (fields[0], fields[1].hasPrefix("coll") ? "col_" + fields[1].dropFirst(4) : fields[1], fields[2])
        }
        return nil
    }
    private static func providerRole(_ suffix: String) -> String? {
        if suffix.range(of: #"^(?:diffuse|diff|color|col|albedo)(?:_?\d+)?$"#, options: .regularExpression) != nil { return "input" }
        return ["disp": "height", "disp_gl": "height", "displacement": "height", "nor_gl": "normal", "nor_dx": "normal", "rough": "roughness", "roughness": "roughness"][suffix]
    }

    private static func discover(_ folder: URL, dataset: URL, records: [Record] = []) throws -> ([Object], [String]) {
        var cache: [String: Object] = [:]
        for record in records { for source in sources(record.sample) { if let path = source["path"] as? String { cache[path] = source } } }
        if FileManager.default.fileExists(atPath: folder.appendingPathComponent("dataset.json").path) {
            guard folder != dataset else { throw StudioError("This dataset is already open. Choose another folder to import.") }
            return try locked(folder) {
                let index = try object(folder.appendingPathComponent("dataset.json")), imported = try readRecords(folder, index: index)
                let materials: [Object] = try imported.map { record in
                    var maps: [String: Object] = [:]
                    for (role, details) in record.sample["map_metadata"] as? [String: Object] ?? [:] {
                        guard ["input", "height", "roughness", "normal"].contains(role), var source = details["source"] as? Object, let path = source["path"] as? String else { continue }
                        let actual = try sourceSummary(URL(fileURLWithPath: path), cached: cache[path])
                        guard actual["file_sha256"] as? String == source["file_sha256"] as? String else { throw StudioError("Imported original source changed since registration.") }
                        source.merge(actual) { _, new in new }; maps[role] = source
                    }
                    guard let input = maps["input"] else { throw StudioError("Imported material has no diffuse source.") }
                    var variants: [Object] = []
                    for detail in record.sample["input_variants"] as? [Object] ?? [] {
                        guard var source = detail["source"] as? Object, let path = source["path"] as? String else { continue }
                        let actual = try sourceSummary(URL(fileURLWithPath: path), cached: cache[path])
                        guard actual["file_sha256"] as? String == source["file_sha256"] as? String else { throw StudioError("Imported original color changed since registration.") }
                        source.merge(actual) { _, new in new }; source["variant_id"] = detail["variant_id"] ?? "color_default"; variants.append(source)
                    }
                    if variants.isEmpty { var color = input; color["variant_id"] = "color_default"; variants = [color] }
                    return ["name": record.sample["name"] ?? record.sample["material_id"] ?? "Material", "material_id": record.sample["material_id"] ?? "", "source_family_id": record.sample["source_family_id"] ?? record.sample["material_id"] ?? "", "source_set_id": record.sample["source_set_id"] ?? record.sample["material_id"] ?? "", "source_directory": URL(fileURLWithPath: input["path"] as? String ?? "").deletingLastPathComponent().path, "common_pixel_dimensions": [input["width"] as? Int ?? 0, input["height"] as? Int ?? 0], "resolution_label": record.sample["source_resolution_label"] ?? "native", "maps": maps, "input_variants": variants]
                }
                return (materials, [])
            }
        }
        let files = try inventory(folder, dataset: dataset).compactMap { ($0[0] as? String).map { URL(fileURLWithPath: $0) } }
        var warnings: [String] = [], groups: [String: Object] = [:]
        let directories = Set(files.map { $0.deletingLastPathComponent() }).sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
        for directory in directories {
            try Task.checkCancellation()
            let local = files.filter { $0.deletingLastPathComponent() == directory }.sorted { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() }
            let provider = local.contains { parsedProvider($0.lastPathComponent) != nil }
            if !provider {
                let aliases: [String: String] = ["input": "input", "color": "input", "colour": "input", "albedo": "input", "basecolor": "input", "base_color": "input", "diffuse": "input", "height": "height", "displacement": "height", "disp": "height", "roughness": "roughness", "rough": "roughness", "normal": "normal", "normalgl": "normal", "normal_gl": "normal", "normaldx": "normal", "normal_dx": "normal"]
                var paths: [String: URL] = [:], ambiguous = false, convention = "opengl"
                for path in local { if let role = aliases[path.deletingPathExtension().lastPathComponent.lowercased()] {
                    if paths[role] != nil { warnings.append("Skipped \(directory.lastPathComponent): multiple \(role) maps need explicit selection."); ambiguous = true }
                    paths[role] = path
                    if role == "normal", path.deletingPathExtension().lastPathComponent.lowercased().contains("dx") { convention = "directx" }
                } }
                if !ambiguous, paths["input"] != nil, ["height", "normal", "roughness"].contains(where: { paths[$0] != nil }) {
                    do { let material = try sourceMaterial(directory.lastPathComponent, paths: paths, convention: convention); groups["manual:" + directory.path] = material }
                    catch { warnings.append("Skipped \(directory.lastPathComponent): \(error.localizedDescription)") }
                }
                continue
            }
            let manifests: [(URL, Object)] = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey, .isSymbolicLinkKey])) ?? []).filter { $0.lastPathComponent.hasPrefix("material-source") && $0.pathExtension.lowercased() == "json" }.compactMap { path in
                guard let values = try? path.resourceValues(forKeys: [.fileSizeKey, .isSymbolicLinkKey]), values.isSymbolicLink != true, (values.fileSize ?? Int.max) <= 16 * 1024 * 1024, let value = try? object(path) else { return nil }; return (path, value)
            }
            for path in local {
                guard let (asset, suffix, label) = parsedProvider(path.lastPathComponent), let role = providerRole(suffix) else { continue }
                var family = identifier(asset)
                if family.hasSuffix("_" + label) { family = String(family.dropLast(label.count + 1)) }
                let resolved = path.resolvingSymlinksInPath().standardizedFileURL
                var source: Object
                do { source = try sourceSummary(resolved, cached: cache[resolved.path]) }
                catch { warnings.append("Skipped \(path.path): \(error.localizedDescription)"); continue }
                source["path"] = resolved.path; source["suffix"] = suffix; source["resolution_label"] = label; source["source_family_id"] = family; source["asset_family_id"] = family
                for (manifestPath, manifest) in manifests where manifest["material_id"] as? String == family && (manifest["resolution"] as? String)?.lowercased() == label {
                    for (downloadRole, downloaded) in manifest["downloaded_maps"] as? [String: Object] ?? [:] where URL(fileURLWithPath: downloaded["path"] as? String ?? "").lastPathComponent == path.lastPathComponent && downloaded["sha256"] as? String == source["file_sha256"] as? String && downloaded["bytes"] as? Int == source["file_bytes"] as? Int {
                        if manifest["provider"] as? String == "Poly Haven", manifest["license"] as? String == "CC0-1.0", let api = manifest["api"] as? Object, api["api_url"] as? String == "https://api.polyhaven.com/files/" + family, let published = (manifest["maps"] as? [String: Object])?[downloadRole], published["published_bytes"] as? Int == source["file_bytes"] as? Int, try matchesPublishedMD5(resolved, expected: published["published_md5"] as? String), (published["url"] as? String)?.hasPrefix("https://dl.polyhaven.org/file/ph-assets/Textures/png/") == true {
                            source.merge(["provider": "Poly Haven", "asset_id": family, "asset_url": "https://polyhaven.com/a/" + family, "license": "CC0-1.0", "license_url": "https://polyhaven.com/license", "license_basis": "Current full source SHA256 and byte count match the verified official download manifest", "published_url": published["url"]!, "published_md5": published["published_md5"]!, "published_bytes": published["published_bytes"]!, "published_api_url": api["api_url"]!, "published_audit_timestamp_utc": manifest["completed_utc"] ?? NSNull(), "published_manifest_path": manifestPath.path]) { _, new in new }
                        }
                        if manifest["provider"] as? String == "ambientCG", let auditPath = manifest["package_audit_path"] as? String,
                           let values = try? URL(fileURLWithPath: auditPath).resourceValues(forKeys: [.fileSizeKey, .isSymbolicLinkKey]),
                           values.isSymbolicLink != true, (values.fileSize ?? Int.max) <= 16 * 1024 * 1024,
                           let audit = try? object(URL(fileURLWithPath: auditPath)), var evidence = ambientEvidence(source, family: family, audit: audit) {
                            evidence["published_manifest_path"] = manifestPath.path
                            source.merge(evidence) { _, new in new }
                        }
                    }
                }
                let dimensions = [source["width"] as? Int ?? 0, source["height"] as? Int ?? 0]
                let key = "\(directory.path)|\(family)|\(dimensions[0])x\(dimensions[1])"
                var group = groups[key] ?? ["source_family_id": family, "source_set_id": "\(family)_\(dimensions[0])x\(dimensions[1])", "source_directory": directory.resolvingSymlinksInPath().path, "common_pixel_dimensions": dimensions, "maps": [String: Object](), "input_variants": [Object](), "source_files": [Object](), "warnings": [String](), "ignored_files": [String](), "problems": [String](), "resolution_labels": [String]()]
                var labels = group["resolution_labels"] as? [String] ?? []; if !labels.contains(label) { labels.append(label) }; group["resolution_labels"] = labels
                var allSources = group["source_files"] as? [Object] ?? []; allSources.append(source); group["source_files"] = allSources
                var maps = group["maps"] as? [String: Object] ?? [:], variants = group["input_variants"] as? [Object] ?? [], problems = group["problems"] as? [String] ?? [], ignored = group["ignored_files"] as? [String] ?? [], issues = group["warnings"] as? [String] ?? []
                let expectedSize = (Int(label.dropLast()) ?? 0) * 1024
                if dimensions != [expectedSize, expectedSize] { issues.append("\(path.lastPathComponent): named \(label) but actual registration is \(dimensions[0])×\(dimensions[1]); actual pixels are used") }
                if role == "input" {
                    let digits = suffix.replacingOccurrences(of: "^(?:diffuse|diff|color|col|albedo)_?", with: "", options: .regularExpression)
                    source["variant_id"] = digits.isEmpty ? "color_default" : "color_\(Int(digits) ?? 0)"
                    if let prior = variants.first(where: { $0["variant_id"] as? String == source["variant_id"] as? String }) {
                        if prior["file_sha256"] as? String == source["file_sha256"] as? String { ignored.append(path.lastPathComponent) } else { problems.append("Different maps claim the same color variant; choose one explicitly.") }
                    } else { variants.append(source) }
                } else if let prior = maps[role] {
                    if role == "normal", prior["suffix"] as? String == "nor_dx", suffix == "nor_gl" { ignored.append(prior["filename"] as? String ?? ""); maps[role] = source }
                    else if role == "normal", prior["suffix"] as? String == "nor_gl", suffix == "nor_dx" { ignored.append(path.lastPathComponent) }
                    else if prior["file_sha256"] as? String == source["file_sha256"] as? String { ignored.append(path.lastPathComponent) }
                    else { problems.append("Multiple source maps for \(role).") }
                } else { maps[role] = source }
                group["maps"] = maps; group["input_variants"] = variants; group["problems"] = problems; group["ignored_files"] = ignored; group["warnings"] = issues; groups[key] = group
            }
        }
        var materials: [Object] = [], seen = Set<String>()
        for key in groups.keys.sorted() {
            var group = groups[key]!
            if !key.hasPrefix("manual:") {
                let dimensions = group["common_pixel_dimensions"] as! [Int], labels = (group.removeValue(forKey: "resolution_labels") as? [String] ?? []).sorted()
                let resolution = dimensions[0] == dimensions[1] && dimensions[0] >= 1024 && dimensions[0] % 1024 == 0 ? "\(dimensions[0] / 1024)k" : labels.count == 1 ? labels[0] : "\(dimensions[0])x\(dimensions[1])"
                group["resolution_label"] = resolution; group["material_id"] = (group["source_family_id"] as! String) + "_" + resolution
                let variants = (group["input_variants"] as? [Object] ?? []).sorted { a, b in
                    let ai = a["variant_id"] as? String ?? "", bi = b["variant_id"] as? String ?? ""
                    if (ai == "color_default") != (bi == "color_default") { return ai == "color_default" }; return ai < bi
                }
                group["input_variants"] = variants
                var maps = group["maps"] as? [String: Object] ?? [:]; if let input = variants.first { maps["input"] = input }; group["maps"] = maps
            }
            let maps = group["maps"] as? [String: Object] ?? [:]
            guard maps["input"] != nil, ["height", "normal", "roughness"].contains(where: { maps[$0] != nil }) else { continue }
            let id = group["material_id"] as! String
            if !seen.insert(id).inserted { warnings.append("Skipped \(id): duplicate normalized material identity."); continue }
            let problems = group["problems"] as? [String] ?? []
            if !problems.isEmpty { warnings.append("Skipped \(id): " + problems.joined(separator: "; ")); continue }
            do { _ = try sourceRecord(group); materials.append(group); warnings += (group["warnings"] as? [String] ?? []).map { id + ": " + $0 } }
            catch { warnings.append("Skipped \(id): \(error.localizedDescription)") }
        }
        return (materials.sorted { ($0["material_id"] as? String ?? "") < ($1["material_id"] as? String ?? "") }, warnings)
    }
    private static func ambientEvidence(_ source: Object, family: String, audit: Object) -> Object? {
        func validHash(_ value: Any?) -> Bool { (value as? String)?.range(of: #"^[a-f0-9]{64}$"#, options: .regularExpression) != nil }
        guard audit["schema"] as? String == "ipde-material-package-audit-v1", audit["provider"] as? String == "ambientCG",
              audit["material_id"] as? String == family, let package = audit["package"] as? Object,
              let filename = package["filename"] as? String, filename.range(of: #"^[A-Za-z0-9]+_[1248]K-PNG\.zip$"#, options: .regularExpression) != nil,
              let asset = audit["asset_id"] as? String, filename.hasPrefix(asset + "_"), asset.lowercased() == family.lowercased(),
              let urlText = package["url"] as? String, let url = URLComponents(string: urlText), url.scheme == "https", url.host == "ambientcg.com", url.path == "/get",
              url.queryItems?.count == 1, url.queryItems?.first?.name == "file", url.queryItems?.first?.value == filename,
              validHash(package["sha256"]), (package["file_bytes"] as? Int ?? 0) > 0,
              let license = audit["license"] as? Object, license["spdx"] as? String == "CC0-1.0", license["url"] as? String == "https://docs.ambientcg.com/license/", validHash(license["snapshot_sha256"]),
              validHash(source["file_sha256"]), (source["file_bytes"] as? Int ?? 0) > 0 else { return nil }
        for entry in audit["files"] as? [Object] ?? [] {
            guard entry["source_filename"] as? String == source["filename"] as? String,
                  entry["exact_full_file_match"] as? Bool == true, entry["source_stable"] as? Bool != false,
                  entry["source_sha256"] as? String == source["file_sha256"] as? String,
                  entry["member_sha256"] as? String == source["file_sha256"] as? String,
                  entry["source_bytes"] as? Int == source["file_bytes"] as? Int,
                  entry["member_bytes"] as? Int == source["file_bytes"] as? Int,
                  let member = entry["archive_member"] as? String, !member.hasPrefix("/"), !member.split(separator: "/").contains(".."),
                  member.hasPrefix(String(filename.dropLast(4)) + "_"), member.hasSuffix(".png") else { continue }
            return ["provider": "ambientCG", "asset_id": asset, "asset_url": audit["asset_url"] ?? NSNull(), "license": "CC0-1.0", "license_url": license["url"]!, "license_basis": "Current full source SHA256 and byte count match a verified member of the official ambientCG archive", "creation_method": audit["creation_method"] ?? NSNull(), "published_package_url": urlText, "published_package_sha256": package["sha256"]!, "published_package_bytes": package["file_bytes"]!, "published_archive_member": member, "published_member_sha256": entry["member_sha256"]!, "published_member_bytes": entry["member_bytes"]!, "published_audit_timestamp_utc": audit["completed_utc"] ?? NSNull()]
        }
        return nil
    }
    private static func scan(_ root: URL, index: Object, records: [Record], options: [String: String]) throws -> Object {
        guard let requested = options["--folder"], let planPath = options["--plan"] else { throw StudioError("Choose a folder and preview path.") }
        let folder = URL(fileURLWithPath: requested).resolvingSymlinksInPath().standardizedFileURL, before = try inventory(folder, dataset: root)
        let (materials, discoveredWarnings) = try discover(folder, dataset: root, records: records)
        guard NSArray(array: before).isEqual(to: try inventory(folder, dataset: root)) else { throw StudioError("Original folder changed while scanning. Scan it again.") }
        var identities = Set(records.map { sourceIdentity($0.sample) }), names = Set(records.compactMap { $0.sample["material_id"] as? String }), accepted: [Object] = [], warnings = discoveredWarnings, duplicates = 0
        for material in materials {
            let sample = try sourceRecord(material), identity = sourceIdentity(sample), id = sample["material_id"] as! String
            if identities.contains(identity) { duplicates += 1; continue }
            if names.contains(id) { warnings.append("Skipped \(id): another map set already uses this name. Add it manually with another name."); continue }
            identities.insert(identity); names.insert(id); accepted.append(material)
        }
        let recognized = Set(materials.flatMap { ($0["source_files"] as? [Object] ?? Array(($0["maps"] as? [String: Object] ?? [:]).values)).compactMap { $0["path"] as? String } })
        let ignored = before.filter { !recognized.contains($0[1] as? String ?? "") }.count
        if ignored > 0 { warnings.append("\(ignored) PNG files are not part of a recognized paired diffuse and surface map set.") }
        if materials.isEmpty { warnings.append("No paired PNG material maps were recognized. Add maps manually or choose another folder.") }
        let reviews = try optionalObject(root.appendingPathComponent(".material-size-reviews.json")), settings = try validation(index["validation"] as? Object)
        let combined = try records + accepted.map { Record(entry: [:], path: root.appendingPathComponent("dataset.json"), sample: try sourceRecord($0)) }
        var plans: Object = [:]
        for size in sizes {
            let shared = try regionPlan(combined, size: size, reviews: reviews, target: "height", settings: settings)
            var targets: Object = [:]
            for target in ["height", "roughness", "normal"] { var plan = target == "height" ? shared : planForTarget(shared, records: combined, target: target); plan.removeValue(forKey: "assignments"); targets[target] = plan }
            plans[String(size)] = targets
        }
        let indexHash = try digest(root.appendingPathComponent("dataset.json")), reviewHash = try optionalDigest(root.appendingPathComponent(".material-size-reviews.json")) ?? hash(Data())
        let plan: Object = ["schema": "texture-studio-folder-import-v1", "dataset_path": root.path, "folder_path": folder.path, "index_sha256": indexHash, "review_sha256": reviewHash, "inventory": before, "materials": accepted]
        let output = URL(fileURLWithPath: planPath); try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true); try atomic(output, data: jsonData(plan))
        return ["folder_path": folder.path, "plan_path": output.path, "plan_sha256": try digest(output), "index_sha256": indexHash, "source_set_count": materials.count, "added_material_count": accepted.count, "duplicate_material_count": duplicates, "ignored_file_count": ignored, "warnings": warnings, "plans": plans]
    }
    private static func importFolder(_ root: URL, index: Object, records: [Record], options: [String: String]) throws -> Object {
        guard let requested = options["--folder"] else { throw StudioError("Choose a folder to import.") }
        let folder = URL(fileURLWithPath: requested).resolvingSymlinksInPath().standardizedFileURL
        let materials: [Object]
        if let path = options["--plan"] {
            let url = URL(fileURLWithPath: path)
            guard try digest(url) == options["--expected-plan-sha256"] else { throw StudioError("Import preview changed. Scan the folder again.") }
            let plan = try object(url)
            let currentIndex = try digest(root.appendingPathComponent("dataset.json")), currentReviews = try optionalDigest(root.appendingPathComponent(".material-size-reviews.json")) ?? hash(Data()), currentInventory = try inventory(folder, dataset: root)
            guard plan["schema"] as? String == "texture-studio-folder-import-v1", plan["dataset_path"] as? String == root.path, plan["folder_path"] as? String == folder.path, plan["index_sha256"] as? String == currentIndex, plan["review_sha256"] as? String == currentReviews, let inventoryProof = plan["inventory"] as? [[Any]], NSArray(array: inventoryProof).isEqual(to: currentInventory), let pending = plan["materials"] as? [Object] else { throw StudioError("Dataset or original folder changed since preview. Scan the folder again.") }
            for material in pending { for source in Array((material["maps"] as? [String: Object] ?? [:]).values) + (material["input_variants"] as? [Object] ?? []) {
                guard let path = source["path"] as? String, let expected = source["source_stat"] as? Object, NSDictionary(dictionary: expected).isEqual(to: try fileState(URL(fileURLWithPath: path))) else { throw StudioError("Original map changed since preview. Scan the folder again.") }
            } }
            materials = pending
        } else { materials = try discover(folder, dataset: root, records: records).0 }
        return try register(root, index: index, records: records, materials: materials, options: options)
    }
    private static func refreshSourceIndex(_ root: URL, index: Object) throws -> Object {
        guard index["storage_policy"] as? String == "original-source-references-v1" else { throw StudioError("This dataset uses a retired material schema. Create a native dataset and import its original source maps.") }
        let inventoryBefore = try sourceInventoryDigest(root)
        // Current manifests are authoritative evidence. Discovery may append new
        // sets but never replace their source checksums or scientific reviews.
        let records = try readRecords(root, index: index)
        let (materials, _) = try discover(root.appendingPathComponent("sources"), dataset: root, records: records)
        let removed = Set((index["removed_materials"] as? [Object] ?? []).compactMap { $0["material_id"] as? String })
        let currentIDs = Set(records.compactMap { $0.sample["material_id"] as? String })
        let fresh = materials.filter { !removed.contains($0["material_id"] as? String ?? "") && !currentIDs.contains($0["material_id"] as? String ?? "") }
        guard try sourceInventoryDigest(root) == inventoryBefore else { throw StudioError("Original source inventory changed while it was inspected. Reload the dataset.") }
        if !fresh.isEmpty { _ = try register(root, index: index, records: records, materials: fresh, options: [:]) }
        var refreshed = try object(root.appendingPathComponent("dataset.json"))
        refreshed["source_discovery"] = "registered-resolution-and-color-sets-v2"
        refreshed["source_inventory_sha256"] = inventoryBefore
        try commit(root, updates: [(root.appendingPathComponent("dataset.json"), refreshed)])
        return refreshed
    }

    private static let preparationSchema = "texture-studio-native-crops-v1"
    private static func prepare(_ supplied: URL, options: [String: String], policy: PreparationPolicy, onEvent: @Sendable (String) -> Void) throws -> Object {
        let size = try checkedSize(options["--size"] ?? ""), target = options["--target"] ?? "height"
        guard ["height", "roughness", "normal"].contains(target) else { throw StudioError("Choose a material training target.") }
        let initial = try locked(supplied) { try object(supplied.appendingPathComponent("dataset.json")) }
        let lineage = initial["native_size_preparation"] as? Object
        let original: URL
        if let lineage {
            guard lineage["schema"] as? String == preparationSchema, let source = lineage["source_dataset_path"] as? String, source.hasPrefix("/") else { throw StudioError("Native crop dataset has an invalid source lineage.") }
            original = canonical(URL(fileURLWithPath: source)); guard original != supplied else { throw StudioError("Native dataset has circular source lineage.") }
        } else { original = supplied }
        let roots = Set([original, supplied]).sorted { $0.path < $1.path }
        func acquire(_ offset: Int, body: () throws -> Object) throws -> Object { if offset == roots.count { return try body() }; return try locked(roots[offset]) { try acquire(offset + 1, body: body) } }
        return try acquire(0) {
            try Task.checkCancellation(); try checkIndex(supplied, expected: options["--expected-index-sha256"])
            let sourceIndex = try object(original.appendingPathComponent("dataset.json")), current = try object(supplied.appendingPathComponent("dataset.json"))
            guard NSDictionary(dictionary: current["native_size_preparation"] as? Object ?? [:]).isEqual(to: lineage ?? [:]) else { throw StudioError("Native source lineage changed; reopen the dataset.") }
            let reviewPath = original.appendingPathComponent(".material-size-reviews.json")
            if let expected = options["--expected-review-sha256"], try optionalDigest(reviewPath) ?? hash(Data()) != expected { throw StudioError("Dataset reviews changed; reload before preparing native crops.") }
            let allRecords = try readRecords(original, index: sourceIndex)
            let selectedID = options["--material"]
            let selected = allRecords.filter { selectedID == nil || $0.sample["material_id"] as? String == selectedID }
            guard !selected.isEmpty else { throw StudioError("Dataset has no selected original material maps.") }
            let records = selected.filter {
                let sample = $0.sample, selectedSources = sources(sample, target: target)
                return !["excluded", "rejected"].contains(sample["status"] as? String ?? "") &&
                    (sample["map_metadata"] as? [String: Object])?[target] != nil &&
                    (sample["available_targets"] as? [String] ?? [target]).contains(target) &&
                    !selectedSources.isEmpty && selectedSources.allSatisfy { min($0["width"] as? Int ?? 0, $0["height"] as? Int ?? 0) >= size }
            }
            guard !records.isEmpty else { throw StudioError("No original source set supplies the selected native training size.") }
            var reviews = try optionalObject(reviewPath)
            if supplied != original {
                let cached = try readRecords(supplied, index: current)
                for record in cached {
                    let key = "\(current["crop_size"] as? Int ?? size):\(record.sample["sample_id"] as? String ?? "")", review = currentReview(reviews, key: key, sample: record.sample)
                    if let snapshot = record.sample["source_review_snapshot"] as? Object, NSDictionary(dictionary: snapshot).isEqual(to: review) || review.isEmpty { reviews[key] = savedReview(record.sample) }
                }
            }
            var settings = try validation(sourceIndex["validation"] as? Object)
            if options["--automatic-validation"] != "true" { settings["enabled"] = false }
            // Folder percentages and geometry remain common to all targets;
            // only target-eligible assignments become preparation jobs.
            let plan = try regionPlan(selected, size: size, reviews: reviews, target: target, settings: settings)
            let assignments = (plan["assignments"] as? [String: Object] ?? [:]).filter { $0.value["eligible"] as? Bool == true }
            guard !assignments.isEmpty else { throw StudioError("No included crops supply the selected training target.") }
            var metadataProof: [String: String] = [:]
            for record in allRecords { metadataProof[String(record.path.path.dropFirst(original.path.count + 1))] = try digest(record.path) }
            let proof: Object = ["dataset_path": original.path, "index_sha256": try digest(original.appendingPathComponent("dataset.json")), "sample_metadata_sha256": metadataProof]
            let signature: Object = ["schema": preparationSchema, "map_scope": "selected-target-v2", "size": size, "target": target, "snapshot": proof, "reviews": reviews, "validation": settings, "material": selectedID as Any? ?? NSNull()]
            let identity = hash(try jsonData(signature)), stagingRoot = original.appendingPathComponent(".training-data")
            guard (try? stagingRoot.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { throw StudioError("Native training data cannot be a symbolic link.") }
            if !FileManager.default.fileExists(atPath: stagingRoot.path) { try FileManager.default.createDirectory(at: stagingRoot, withIntermediateDirectories: false) }
            let destination = stagingRoot.appendingPathComponent("\(size)-\(identity.prefix(20))")
            if FileManager.default.fileExists(atPath: destination.appendingPathComponent("dataset.json").path) {
                if try validatePrepared(destination, size: size, proof: proof) {
                    let result = try preparedInfo(original, destination: destination, size: size, proof: proof, reused: true)
                    let binding = try object(destination.appendingPathComponent("dataset.json"))["native_size_preparation"] as? Object ?? [:]
                    emitPreparation(onEvent, event: "preparation_completed", completed: records.count, total: records.count,
                        workerCount: binding["preparation_workers"] as? Int ?? 1, size: size, extra: ["reused": true, "dataset_path": destination.path])
                    return result
                }
                try cleanupLocked(destination, original: original)
            }
            var verified: [String: Object] = [:], bytesRequired: Int64 = 0
            for record in records {
                try Task.checkCancellation()
                let dimensions = record.sample["source_pixel_dimensions"] as? [Int] ?? []
                guard dimensions.count == 2 else { throw StudioError("Invalid native source dimensions.") }
                let regionCount = assignments.values.filter { $0["material_id"] as? String == record.sample["material_id"] as? String && $0["crop_rectangle"] as? [Int] != [0, 0] + dimensions }.count
                for source in sources(record.sample, target: target) {
                    try Task.checkCancellation()
                    guard let path = source["path"] as? String, let expected = source["file_sha256"] as? String else { throw StudioError("Original source is not checksum bound.") }
                    if verified[path] == nil {
                        let currentPath = FileManager.default.fileExists(atPath: path) ? URL(fileURLWithPath: path) : try recoverSource(source, root: original)
                        guard let currentPath else { throw StudioError("Original source is missing: \(path). Restore its recorded original map.") }
                        let state = try fileState(currentPath)
                        let recordedState = source["source_stat"] as? Object
                        let unchanged = recordedState.map { NSDictionary(dictionary: $0).isEqual(to: state) } ?? false
                        guard try unchanged || digest(currentPath) == expected else { throw StudioError("Original source changed: \(path). Restore its recorded original map.") }
                        let header = try NativePNG.inspect(currentPath)
                        guard header.width == source["width"] as? Int, header.height == source["height"] as? Int, header.bits == source["sample_bits"] as? Int, header.channels == source["channels"] as? Int else { throw StudioError("Original source dimensions or precision disagree with its evidence.") }
                        var bound = source; bound["path"] = currentPath.path; bound["source_stat"] = state; verified[path] = bound
                    }
                    bytesRequired += Int64(size * size * (source["channels"] as? Int ?? 1) * (source["sample_bits"] as? Int ?? 8) / 8 * regionCount)
                }
            }
            let disk = try stagingRoot.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage ?? 0
            guard disk >= bytesRequired + 64 * 1024 * 1024 else { throw StudioError("Insufficient storage for lossless native crops. Free space or choose a smaller native size.") }
            let stage = stagingRoot.appendingPathComponent(".preparing-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
            defer { try? FileManager.default.removeItem(at: stage) }
            try atomic(stage.appendingPathComponent(".native-crop-preparation.json"), data: jsonData(["schema": preparationSchema, "source_dataset_path": original.path, "pid": Int(getpid())]))
            let activeMaterials = Set(assignments.values.compactMap { $0["material_id"] as? String })
            let work = try records.filter { activeMaterials.contains($0.sample["material_id"] as? String ?? "") }.sorted(by: { ($0.sample["material_id"] as? String ?? "") < ($1.sample["material_id"] as? String ?? "") }).map { record -> PreparationJob in
                try Task.checkCancellation()
                let material = record.sample["material_id"] as? String ?? ""
                let selectedAssignments = assignments.filter { $0.value["material_id"] as? String == material }
                let sourcePaths = Set(sources(record.sample, target: target).compactMap { $0["path"] as? String })
                return PreparationJob(sample: try jsonData(record.sample), assignments: try jsonData(selectedAssignments),
                    reviews: try jsonData(reviews.filter { $0.key.hasPrefix("\(size):\(material)_") }), verified: try jsonData(verified.filter { sourcePaths.contains($0.key) }),
                    stage: stage, original: original, size: size, target: target,
                    estimatedPeakBytes: try preparationPeakBytes(sources: sources(record.sample, target: target), size: size, cropCount: selectedAssignments.count))
            }
            let peakBytes = work.map(\.estimatedPeakBytes).max() ?? 0
            let workerCount = try policy.workerCount(jobCount: work.count, estimatedPeakBytes: peakBytes)
            emitPreparation(onEvent, event: "preparation_started", completed: 0, total: work.count, workerCount: workerCount, size: size,
                extra: ["estimated_worker_peak_bytes": peakBytes, "preparation_memory_budget_bytes": policy.memoryBudgetBytes])
            let completed = try performPreparation(work, workerCount: workerCount) { completed in
                emitPreparation(onEvent, event: "preparation_progress", completed: completed, total: work.count, workerCount: workerCount, size: size)
            }
            let prepared = try completed.flatMap { try JSONSerialization.jsonObject(with: $0.samples) as? [Object] ?? [] }
            let cropped = completed.contains { $0.cropped }
            let checkFamilies = Set(prepared.filter { $0["split"] as? String == "validation" && !["excluded", "rejected"].contains($0["status"] as? String ?? "") }.map(familyID))
            let automatic: Object = ["policy": "subject-extra-crops-v2", "fraction": (settings["percent"] as? Double ?? 5) / 100, "source_family_ids": checkFamilies.sorted(), "material_ids": records.filter { checkFamilies.contains(familyID($0.sample)) }.compactMap { $0.sample["material_id"] as? String }.sorted(), "quick_fit_material_id": selectedID as Any? ?? NSNull(), "target": target]
            let binding: Object = ["schema": preparationSchema, "ephemeral": true, "source_dataset_path": original.path, "source_snapshot": proof, "target": target, "map_scope": "selected-target-v2", "target_resized": false, "target_cropped": cropped, "estimated_staging_bytes": bytesRequired, "preparation_workers": workerCount, "estimated_worker_peak_bytes": peakBytes, "preparation_memory_budget_bytes": policy.memoryBudgetBytes, "automatic_validation": automatic]
            let derived: Object = ["schema_version": 2, "generator": "ipde-material-dataset-v2", "crop_size": size, "training_pixel_dimensions": [size, size], "split_strategy": "subject-extra-crops-v2", "validation": settings, "validation_scope": "known_subject_diagnostic", "original_sources_required_for_training": true, "source_images_modified": false, "native_size_preparation": binding, "automatic_validation": automatic, "samples": prepared.map { ["sample_id": $0["sample_id"]!, "material_id": $0["material_id"]!, "status": $0["status"]!, "split": $0["split"]!, "asset_family_id": familyID($0), "source_set_id": $0["source_set_id"] ?? NSNull(), "path": "samples/" + ($0["sample_id"] as! String)] }]
            try atomic(stage.appendingPathComponent("dataset.json"), data: jsonData(derived))
            guard try validatePrepared(stage, size: size, proof: proof) else { throw StudioError("Native training grids failed validation.") }
            guard try digest(original.appendingPathComponent("dataset.json")) == proof["index_sha256"] as? String else { throw StudioError("Original dataset changed during preparation.") }
            for (path, expected) in metadataProof { guard try digest(contained(original, path)) == expected else { throw StudioError("Original review metadata changed during preparation.") } }
            try Task.checkCancellation()
            try FileManager.default.removeItem(at: stage.appendingPathComponent(".native-crop-preparation.json"))
            guard !FileManager.default.fileExists(atPath: destination.path) else { throw StudioError("Concurrent native dataset publication.") }
            try FileManager.default.moveItem(at: stage, to: destination); try sync(stagingRoot)
            let result = try preparedInfo(original, destination: destination, size: size, proof: proof, reused: false)
            emitPreparation(onEvent, event: "preparation_completed", completed: work.count, total: work.count, workerCount: workerCount, size: size,
                extra: ["reused": false, "dataset_path": destination.path])
            return result
        }
    }
    private static func emitPreparation(_ onEvent: @Sendable (String) -> Void, event: String, completed: Int, total: Int,
                                        workerCount: Int, size: Int, extra: Object = [:]) {
        var object: Object = ["event": event, "completed": completed, "total": total, "worker_count": workerCount, "training_size": size]
        object.merge(extra) { _, new in new }
        if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) { onEvent(String(decoding: data, as: UTF8.self) + "\n") }
    }
    private struct PreparationJob: Sendable {
        let sample: Data, assignments: Data, reviews: Data, verified: Data
        let stage: URL, original: URL
        let size: Int, target: String
        let estimatedPeakBytes: UInt64
    }
    private struct PreparationResult: Sendable { let samples: Data; let cropped: Bool }

    /// Only a finite number of persistent workers can hold decoded source maps.
    /// JSON creates independent Foundation objects for each worker; mutable
    /// manifest dictionaries never cross an executor boundary.
    private final class PreparationQueue: @unchecked Sendable {
        private let lock = NSLock()
        private var next = 0
        private var results: [PreparationResult?]
        private var failure: Error?
        private var finished = 0
        init(count: Int) { results = Array(repeating: nil, count: count) }
        func take() -> Int? { lock.withLock { guard failure == nil, next < results.count else { return nil }; defer { next += 1 }; return next } }
        func finish(_ result: PreparationResult, at index: Int) { lock.withLock { results[index] = result; finished += 1 } }
        func fail(_ error: Error) { lock.withLock { if failure == nil { failure = error } } }
        var failed: Bool { lock.withLock { failure != nil } }
        var completedCount: Int { lock.withLock { finished } }
        func completed() throws -> [PreparationResult] {
            try lock.withLock {
                if let failure { throw failure }
                guard results.allSatisfy({ $0 != nil }) else { throw StudioError("Native crop workers did not complete their inventory.") }
                return results.map { $0! }
            }
        }
    }
    private static func performPreparation(_ work: [PreparationJob], workerCount: Int, onProgress: (Int) -> Void) throws -> [PreparationResult] {
        let queue = PreparationQueue(count: work.count), completion = DispatchGroup()
        let workers = (0..<workerCount).map { _ -> Task<Void, Never> in
            completion.enter()
            return Task.detached(priority: .userInitiated) {
                defer { completion.leave() }
                do {
                    while let index = queue.take() {
                        try Task.checkCancellation()
                        let result = try autoreleasepool { try prepareMaterial(work[index]) }
                        queue.finish(result, at: index)
                    }
                } catch { queue.fail(error) }
            }
        }
        // Keep the lock and staging directory alive until every writer stops.
        // Swift workers retain their own cancellation context inside PNG loops.
        var reported = 0
        while completion.wait(timeout: .now() + .milliseconds(20)) == .timedOut {
            if Task.isCancelled || queue.failed { workers.forEach { $0.cancel() } }
            let count = queue.completedCount
            if count != reported { reported = count; onProgress(count) }
        }
        try Task.checkCancellation()
        let result = try queue.completed()
        if result.count != reported { onProgress(result.count) }
        return result
    }
    static func preparationPeakBytes(sources: [[String: Any]], size: Int, cropCount: Int) throws -> UInt64 {
        func product(_ values: [UInt64]) throws -> UInt64 {
            try values.reduce(1) { value, factor in
                let result = value.multipliedReportingOverflow(by: factor)
                guard !result.overflow else { throw StudioError("Native map working dimensions overflow.") }
                return result.partialValue
            }
        }
        var peak: UInt64 = 0
        for source in sources {
            guard let width = source["width"] as? Int, let height = source["height"] as? Int,
                  let bits = source["sample_bits"] as? Int, let channels = source["channels"] as? Int,
                  width > 0, height > 0, [8, 16].contains(bits), (1...4).contains(channels) else {
                throw StudioError("Original source working dimensions or precision are invalid.")
            }
            let row = try product([UInt64(width), UInt64(channels), UInt64(bits / 8)])
            let crop = try product([UInt64(size), UInt64(size), UInt64(channels), UInt64(bits / 8)])
            // The source is memory mapped and inflated scanline by scanline.
            // Only selected crops, row buffers and one encoded export are live.
            let fileBytes = (source["file_bytes"] as? NSNumber)?.uint64Value ?? 0
            let terms = try [fileBytes, product([row, 4]), product([crop, UInt64(max(1, cropCount) + 6)]), UInt64(64 * 1024 * 1024)]
            let total = try terms.reduce(UInt64(0)) { value, term in
                let result = value.addingReportingOverflow(term)
                guard !result.overflow else { throw StudioError("Native crop working memory exceeds its format budget.") }
                return result.partialValue
            }
            peak = max(peak, total)
        }
        return peak
    }
    private static func prepareMaterial(_ job: PreparationJob) throws -> PreparationResult {
        func read(_ bytes: Data) throws -> Object {
            guard let value = try JSONSerialization.jsonObject(with: bytes) as? Object else { throw StudioError("Native preparation metadata is invalid.") }; return value
        }
        let originalSample = try read(job.sample), assignments = try read(job.assignments) as? [String: Object] ?? [:]
        let reviews = try read(job.reviews), verified = try read(job.verified) as? [String: Object] ?? [:]
        let size = job.size, dimensions = originalSample["source_pixel_dimensions"] as! [Int]
        let originalMaps = originalSample["map_metadata"] as? [String: Object] ?? [:]
        var variants = originalSample["input_variants"] as? [Object] ?? []
        if variants.isEmpty, var input = originalMaps["input"] { input["variant_id"] = "color_default"; variants = [input] }
        guard let primary = originalMaps["input"]?["source"] as? Object,
              let primaryPath = primary["path"] as? String, let primaryHash = primary["file_sha256"] as? String,
              let canonicalOffset = variants.firstIndex(where: {
                  let source = $0["source"] as? Object ?? [:]
                  return source["path"] as? String == primaryPath && source["file_sha256"] as? String == primaryHash
              }) else { throw StudioError("Canonical diffuse is absent from the source colors.") }
        var maps = variants.enumerated().map { offset, details -> (String, Object, String) in
            // Separate source variants may contain identical bytes. Only the
            // recorded canonical path receives the canonical output filename.
            return ("input", details, offset == canonicalOffset ? "diffuse.png" : "input-variant-\(offset + 1).png")
        }
        for role in [job.target] { if let details = originalMaps[role] { maps.append((role, details, ["height": "displacement.png", "roughness": "roughness.png", "normal": "normal.png"][role]!)) } }
        var active: [(Object, URL, [Int])] = [], cropped = false
        let binding = sourceBinding(originalSample)
        for id in assignments.keys.sorted() {
            try Task.checkCancellation()
            let assignment = assignments[id]!, rectangle = assignment["crop_rectangle"] as! [Int], region = assignment["region"] as! String
            var sample = originalSample
            let review = currentReview(reviews, key: "\(size):\(id)", sample: sample)
            sample.merge(review) { _, new in new }
            sample["sample_id"] = id; sample["sample_pixel_dimensions"] = [size, size]; sample["crop_rectangle_top_left_xywh"] = rectangle
            sample["maps"] = [String: String](); sample["map_metadata"] = [String: Object](); sample["input_variants"] = [Object]()
            sample["available_targets"] = [job.target]
            sample.removeValue(forKey: "source_files")
            sample.removeValue(forKey: "source_normalized_rectangle_xywh")
            sample["split"] = assignment["split"]; sample["split_assignment"] = "automatic"; sample["source_region_role"] = assignment["split"]; sample["source_region_id"] = region
            sample["source_review_snapshot"] = review; sample["source_binding_sha256"] = binding
            sample["split_strategy"] = "subject-extra-crops-v2"; sample["validation_scope"] = "known_subject_diagnostic"; sample["source_precision_verified"] = true; sample["crop_values_verified"] = true
            let changed = rectangle != [0, 0] + dimensions; cropped = cropped || changed
            sample["native_size_preparation"] = ["schema": preparationSchema, "source_dataset_path": job.original.path, "target_resized": false, "target_cropped": changed]
            let folder = try contained(job.stage, "samples/" + id); try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            active.append((sample, folder, rectangle))
        }
        for (role, originalDetails, filename) in maps {
            try Task.checkCancellation()
            guard let source = originalDetails["source"] as? Object, let sourcePath = source["path"] as? String, let verifiedSource = verified[sourcePath], let resolved = verifiedSource["path"] as? String else { throw StudioError("Source identity could not be verified.") }
            let sourceURL = URL(fileURLWithPath: resolved)
            let before = try fileState(sourceURL)
            guard let verifiedState = verifiedSource["source_stat"] as? Object, NSDictionary(dictionary: verifiedState).isEqual(to: before) else { throw StudioError("Original source changed during preparation.") }
            let flip = role == "normal" && ["nor_dx", "normal_dx"].contains(source["suffix"] as? String ?? "")
            let cropOffsets = active.indices.filter { active[$0].2 != [0, 0] + dimensions }
            let crops = cropOffsets.isEmpty ? [] : try NativePNG.crops(sourceURL, rectangles: cropOffsets.map { active[$0].2 }, flipGreen: flip)
            let selectedCrops = Dictionary(uniqueKeysWithValues: zip(cropOffsets, crops))
            guard NSDictionary(dictionary: before).isEqual(to: try fileState(sourceURL)) else { throw StudioError("Original source changed during preparation.") }
            for offset in active.indices {
                try Task.checkCancellation()
                let rectangle = active[offset].2, output = active[offset].1.appendingPathComponent(filename)
                let isCrop = rectangle != [0, 0] + dimensions
                var details = originalDetails, transforms: [Object] = [], path = resolved
                if isCrop {
                    guard let selected = selectedCrops[offset] else { throw StudioError("Native crop decode is absent.") }
                    let encoded = try selected.encoded()
                    try Task.checkCancellation()
                    try encoded.write(to: output, options: .withoutOverwriting); path = filename
                    details["storage"] = "prepared_crop"; details["sample_sha256"] = hash(encoded); details["prepared_stat"] = try fileState(output)
                    transforms.append(["type": "native_crop", "algorithm": "exact_native_integer_codes", "range_normalization": false, "gamma_applied": false])
                } else { details["storage"] = "source_reference"; details["sample_sha256"] = source["file_sha256"] }
                if flip { transforms.append(["type": "directx_to_opengl", "component": "G", "applied_in_memory": !isCrop]) }
                details["source"] = verifiedSource; details["path"] = path; details["filename"] = path; details["sample_bits"] = source["sample_bits"]; details["channels"] = source["channels"]; details["transforms"] = transforms
                details.removeValue(forKey: "source_pixel_dimensions"); details.removeValue(forKey: "crop_rectangle_top_left_xywh"); details["exact_source_crop"] = true
                var sample = active[offset].0
                if role == "input" { var colors = sample["input_variants"] as? [Object] ?? []; colors.append(details); sample["input_variants"] = colors }
                if role != "input" || filename == "diffuse.png" {
                    var filenames = sample["maps"] as? [String: String] ?? [:], metadata = sample["map_metadata"] as? [String: Object] ?? [:]
                    filenames[role] = path; metadata[role] = details; sample["maps"] = filenames; sample["map_metadata"] = metadata
                }
                active[offset].0 = sample
            }
        }
        var prepared: [Object] = []
        for (sample, folder, _) in active {
            try Task.checkCancellation()
            guard (sample["maps"] as? [String: String])?["input"] != nil else { throw StudioError("Canonical diffuse is absent from the prepared colors.") }
            try atomic(folder.appendingPathComponent("sample.json"), data: jsonData(sample)); prepared.append(sample)
        }
        return PreparationResult(samples: try JSONSerialization.data(withJSONObject: prepared, options: [.sortedKeys, .withoutEscapingSlashes]), cropped: cropped)
    }
    static func validatePrepared(_ root: URL, size: Int, proof: [String: Any], checksumFile: ((URL) throws -> String)? = nil) throws -> Bool {
        let index = try object(root.appendingPathComponent("dataset.json"))
        guard index["crop_size"] as? Int == size, let binding = index["native_size_preparation"] as? Object, binding["schema"] as? String == preparationSchema, let snapshot = binding["source_snapshot"] as? Object, NSDictionary(dictionary: snapshot).isEqual(to: proof) else { return false }
        let verifyFile = checksumFile ?? digest
        var verified: [String: (String, Object)] = [:]
        for record in try readRecords(root, index: index) {
            try Task.checkCancellation()
            guard record.sample["sample_pixel_dimensions"] as? [Int] == [size, size] else { return false }
            var changed = false
            func validateMap(_ map: inout Object) throws -> Bool {
                try Task.checkCancellation()
                guard let name = map["filename"] as? String, let expected = map["sample_sha256"] as? String else { return false }
                let path = name.hasPrefix("/") ? URL(fileURLWithPath: name) : try contained(record.path.deletingLastPathComponent(), name)
                let header = try NativePNG.inspect(path)
                guard header.width == size, header.height == size else { return false }
                let state = try fileState(path), reference = map["storage"] as? String == "source_reference"
                var source = map["source"] as? Object ?? [:]
                if reference, source["file_sha256"] as? String != expected { return false }
                let recorded = reference ? source["source_stat"] as? Object : map["prepared_stat"] as? Object
                let unchanged = recorded.map { NSDictionary(dictionary: $0).isEqual(to: state) } ?? false
                if let prior = verified[path.path] {
                    guard prior.0 == expected, NSDictionary(dictionary: prior.1).isEqual(to: state) else { return false }
                } else {
                    guard try unchanged || verifyFile(path) == expected else { return false }
                    guard NSDictionary(dictionary: state).isEqual(to: try fileState(path)) else { return false }
                    verified[path.path] = (expected, state)
                }
                if !unchanged {
                    if reference { source["source_stat"] = state; map["source"] = source }
                    else { map["prepared_stat"] = state }
                    changed = true
                }
                return true
            }
            var maps = record.sample["map_metadata"] as? [String: Object] ?? [:]
            for role in maps.keys.sorted() {
                var map = maps[role]!
                guard try validateMap(&map) else { return false }
                maps[role] = map
            }
            var variants = record.sample["input_variants"] as? [Object] ?? []
            for offset in variants.indices {
                var map = variants[offset]
                guard try validateMap(&map) else { return false }
                variants[offset] = map
            }
            if changed {
                var sample = record.sample
                sample["map_metadata"] = maps; sample["input_variants"] = variants
                try atomic(record.path, data: jsonData(sample))
            }
        }
        return true
    }
    private static func preparedInfo(_ original: URL, destination: URL, size: Int, proof: Object, reused: Bool) throws -> Object {
        let index = try object(destination.appendingPathComponent("dataset.json")), binding = index["native_size_preparation"] as? Object ?? [:]
        var result = try info(destination, index: index, records: readRecords(destination, index: index), options: [:])
        result["preparation"] = ["source_dataset_path": original.path, "source_index_sha256": proof["index_sha256"] ?? "", "prepared_dataset_path": destination.path, "crop_size": size, "reused": reused, "target_resized": false, "target_cropped": binding["target_cropped"] ?? false, "original_dataset_modified": false, "split_lineage_changed": true, "cross_size_validation_notice": "Known-material learning checks use a different crop, which may share pixels with training. Test novel images separately."]
        var preparation = result["preparation"] as! Object
        for key in ["preparation_workers", "estimated_worker_peak_bytes", "preparation_memory_budget_bytes"] { preparation[key] = binding[key] }
        result["preparation"] = preparation
        return result
    }
    private static func savedReview(_ sample: Object) -> Object {
        var result: Object = [:]
        for key in ["status", "split", "split_assignment", "review_status", "curation_note"] { result[key] = sample[key] ?? NSNull() }
        result["source_binding_sha256"] = sourceBinding(sample); return result
    }
    private static func cleanup(_ supplied: URL) throws -> Object {
        guard FileManager.default.fileExists(atPath: supplied.appendingPathComponent("dataset.json").path) else { return ["dataset_path": supplied.path, "removed": false] }
        let index = try object(supplied.appendingPathComponent("dataset.json")), binding = index["native_size_preparation"] as? Object ?? [:]
        guard binding["schema"] as? String == preparationSchema, binding["ephemeral"] as? Bool == true, let source = binding["source_dataset_path"] as? String else { return ["dataset_path": supplied.path, "removed": false] }
        let original = URL(fileURLWithPath: source).resolvingSymlinksInPath().standardizedFileURL
        return try locked(original) { try locked(supplied) {
            try cleanupLocked(supplied, original: original)
            return ["dataset_path": supplied.path, "source_dataset_path": original.path, "removed": true, "source_images_modified": false, "reviews_preserved": true]
        } }
    }
    private static func cleanupLocked(_ supplied: URL, original: URL) throws {
        let staging = original.appendingPathComponent(".training-data")
        guard supplied.deletingLastPathComponent() == staging, (try? supplied.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true, (try? staging.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { throw StudioError("Temporary cleanup scope differs from its owned source location.") }
        let index = try object(supplied.appendingPathComponent("dataset.json")), binding = index["native_size_preparation"] as? Object ?? [:]
        guard binding["schema"] as? String == preparationSchema, binding["ephemeral"] as? Bool == true, binding["source_dataset_path"] as? String == original.path else { throw StudioError("Temporary training data ownership changed.") }
        let reviewPath = original.appendingPathComponent(".material-size-reviews.json")
        var reviews = try optionalObject(reviewPath)
        let records = try readRecords(supplied, index: index)
        var ownedFiles = Set(["dataset.json", ".material-workbench.lock", ".native-crop-preparation.json"])
        for record in records {
            ownedFiles.insert(String(record.path.path.dropFirst(supplied.path.count + 1)))
            for detail in Array((record.sample["map_metadata"] as? [String: Object] ?? [:]).values) + (record.sample["input_variants"] as? [Object] ?? []) {
                guard let source = detail["source"] as? Object, let sourcePath = source["path"] as? String,
                      !isInside(URL(fileURLWithPath: sourcePath).resolvingSymlinksInPath().standardizedFileURL, supplied) else {
                    throw StudioError("Temporary training metadata references an original inside its cleanup scope.")
                }
                if detail["storage"] as? String == "prepared_crop" {
                    guard let name = detail["filename"] as? String, let expectedHash = detail["sample_sha256"] as? String else { throw StudioError("Prepared crop identity is missing.") }
                    let crop = try contained(record.path.deletingLastPathComponent(), name)
                    var state = stat()
                    if lstat(crop.path, &state) == 0 {
                        guard state.st_mode & S_IFMT == S_IFREG, try digest(crop) == expectedHash else { throw StudioError("Temporary crop contents changed; keep this folder and inspect its files before cleanup.") }
                    } else if errno != ENOENT { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                    ownedFiles.insert(String(crop.path.dropFirst(supplied.path.count + 1)))
                }
            }
            let key = "\(index["crop_size"] as? Int ?? 0):\(record.sample["sample_id"] as? String ?? "")", current = currentReview(reviews, key: key, sample: record.sample)
            if let snapshot = record.sample["source_review_snapshot"] as? Object { if !NSDictionary(dictionary: snapshot).isEqual(to: current) { continue } }
            else if !current.isEmpty { continue }
            reviews[key] = savedReview(record.sample)
        }
        // Ownership authorizes generated PNGs and manifests only. New unrelated
        // files prevent cleanup rather than disappearing with the cache.
        var ownedDirectories = Set<String>()
        for path in ownedFiles {
            var components = path.split(separator: "/")
            while components.count > 1 { components.removeLast(); ownedDirectories.insert(components.joined(separator: "/")) }
        }
        var failed = false
        guard let files = FileManager.default.enumerator(at: supplied, includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey], errorHandler: { _, _ in failed = true; return false }) else { throw StudioError("Cannot inspect temporary training data.") }
        for case let file as URL in files {
            try Task.checkCancellation()
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey])
            guard values.isSymbolicLink != true else { throw StudioError("Temporary training data contains an unexpected symbolic link.") }
            let checkedFile = file.resolvingSymlinksInPath().standardizedFileURL
            guard isInside(checkedFile, supplied) else { throw StudioError("Temporary training data contains an unexpected outside path.") }
            let relative = String(checkedFile.path.dropFirst(supplied.path.count + 1))
            if values.isRegularFile == true {
                guard ownedFiles.contains(relative) else { throw StudioError("Temporary training data contains new unrelated files; originals are kept.") }
            } else if values.isDirectory == true {
                guard ownedDirectories.contains(relative) else { throw StudioError("Temporary training data contains an unrelated folder; originals are kept.") }
            } else { throw StudioError("Temporary training data contains an unrelated special file; originals are kept.") }
        }
        guard !failed else { throw StudioError("Temporary training data could not be fully inspected.") }
        try Task.checkCancellation(); try commit(original, updates: [(reviewPath, reviews)])
        try FileManager.default.removeItem(at: supplied); try sync(staging)
    }

    private static func sourceInventoryNeedsRefresh(_ root: URL, index: Object) throws -> Bool {
        guard index["native_size_preparation"] == nil else { return false }
        guard index["storage_policy"] as? String == "original-source-references-v1" else { throw StudioError("This dataset uses a retired material schema. Create a native dataset and import its original source maps.") }
        let sourceRoot = root.appendingPathComponent("sources")
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: sourceRoot.path, isDirectory: &isDirectory), isDirectory.boolValue else { return false }
        guard index["source_discovery"] as? String == "registered-resolution-and-color-sets-v2" else { return true }
        return try sourceInventoryDigest(root) != index["source_inventory_sha256"] as? String
    }
    private static func sourceInventoryDigest(_ root: URL) throws -> String {
        let sourceRoot = root.appendingPathComponent("sources")
        let known = "nor_gl|nor_dx|disp_gl|rough_ao|translucent|(?:diffuse|diff|color|col|albedo)(?:_?\\d+)?|coll\\d+|displacement|roughness|disp|rough|anisotropy_rotation|anisotropy_strength|spec_ior|spec|bump|metal|ao|arm"
        let patterns = [try NSRegularExpression(pattern: "^(.+?)_(" + known + ")_(\\d+k)(?: \\(\\d+\\))?\\.png$", options: [.caseInsensitive]), try NSRegularExpression(pattern: "^([A-Za-z][A-Za-z0-9]*)_([1248]K)-PNG_(Color|Displacement|NormalGL|NormalDX|Roughness)\\.png$", options: [.caseInsensitive])]
        var entries: [[Any]] = []
        for folder in try FileManager.default.contentsOfDirectory(at: sourceRoot, includingPropertiesForKeys: [.isDirectoryKey]).sorted(by: { $0.path < $1.path }) {
            try Task.checkCancellation()
            guard (try folder.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true else { continue }
            for file in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isRegularFileKey]).sorted(by: { $0.path < $1.path }) {
                try Task.checkCancellation()
                let name = file.lastPathComponent
                let recognized = (name.hasPrefix("material-source") && file.pathExtension.lowercased() == "json") || patterns.contains { $0.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)) != nil }
                guard recognized, (try file.resourceValues(forKeys: [.isRegularFileKey])).isRegularFile == true else { continue }
                var state = stat()
                guard stat(file.path, &state) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                entries.append([String(file.path.dropFirst(sourceRoot.path.count + 1)), canonical(file).path, state.st_size, Int64(state.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(state.st_mtimespec.tv_nsec), Int64(state.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(state.st_ctimespec.tv_nsec)])
            }
        }
        let compact = try JSONSerialization.data(withJSONObject: entries, options: [.withoutEscapingSlashes])
        let ascii = String(decoding: compact, as: UTF8.self).utf16.map { $0 > 127 ? String(format: "\\u%04x", $0) : String(UnicodeScalar($0)!) }.joined()
        return hash(Data(ascii.utf8))
    }
    private static func compactMetadata(_ value: Any) -> Any {
        if let object = value as? Object {
            return object.filter { !["data_base64", "color_chunks", "file_md5", "source_files"].contains($0.key) }.mapValues(compactMetadata)
        }
        if let array = value as? [Any] { return array.map(compactMetadata) }
        return value
    }
    private static func compactStoredRecords(_ root: URL, index: Object) throws {
        for entry in index["samples"] as? [Object] ?? [] {
            try Task.checkCancellation()
            guard let directory = entry["path"] as? String else { continue }
            try autoreleasepool {
                let path = try contained(root, directory).appendingPathComponent("sample.json")
                let prior = try object(path), compact = compactMetadata(prior) as! Object
                if !NSDictionary(dictionary: prior).isEqual(to: compact) { try commit(root, updates: [(path, compact)]) }
            }
        }
    }
    private static func readRecords(_ root: URL, index: Object) throws -> [Record] {
        guard index["schema_version"] as? Int == 2, let entries = index["samples"] as? [Object] else { throw StudioError("Expected a schema-2 material dataset.") }
        var seen = Set<String>()
        return try entries.map { entry in
            guard let directory = entry["path"] as? String else { throw StudioError("Dataset sample path is missing.") }
            let path = try contained(root, directory).appendingPathComponent("sample.json")
            let sample = compactMetadata(try object(path)) as! Object
            for key in ["sample_id", "material_id", "status", "split"] {
                guard let expected = entry[key] as? String, sample[key] as? String == expected else { throw StudioError("Dataset and sample identity disagree: \(path.path)") }
            }
            guard seen.insert(sample["sample_id"] as! String).inserted else { throw StudioError("Duplicate dataset sample identity.") }
            return Record(entry: entry, path: path, sample: sample)
        }
    }
    private static func validation(_ value: Object?) throws -> Object {
        var settings: Object = ["enabled": true, "percent": 5.0, "max_crops": 0, "quick_count": 4, "folders": [String: Bool]()]
        settings.merge(value ?? [:]) { _, new in new }
        guard let enabled = settings["enabled"] as? NSNumber, CFGetTypeID(enabled) == CFBooleanGetTypeID(),
              let percent = settings["percent"] as? NSNumber, CFGetTypeID(percent) != CFBooleanGetTypeID(), percent.doubleValue.isFinite, (0...100).contains(percent.doubleValue),
              let max = settings["max_crops"] as? NSNumber, CFGetTypeID(max) != CFBooleanGetTypeID(), max.doubleValue >= 0, max.doubleValue.rounded() == max.doubleValue,
              let quick = settings["quick_count"] as? NSNumber, CFGetTypeID(quick) != CFBooleanGetTypeID(), quick.doubleValue >= 1, quick.doubleValue.rounded() == quick.doubleValue,
              let folders = settings["folders"] as? Object, folders.values.allSatisfy({ ($0 as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? false }) else { throw StudioError("Validation needs an enabled flag, a percentage from 0 to 100, a nonnegative crop limit and at least one quick crop.") }
        settings["enabled"] = enabled.boolValue; settings["percent"] = percent.doubleValue; settings["max_crops"] = max.intValue; settings["quick_count"] = quick.intValue
        return settings
    }
    private static func editedValidation(_ previous: Object, options: [String: String]) throws -> Object {
        var settings = previous
        if let enabled = options["--validation-enabled"] { guard ["yes", "no"].contains(enabled) else { throw StudioError("Invalid validation flag.") }; settings["enabled"] = enabled == "yes" }
        if let percent = options["--validation-percent"] { guard let number = Double(percent) else { throw StudioError("Invalid validation percentage.") }; settings["percent"] = number }
        for key in ["max-crops", "quick-count"] { if let value = options["--validation-" + key] { guard let number = Int(value) else { throw StudioError("Invalid validation crop count.") }; settings[key.replacingOccurrences(of: "-", with: "_")] = number } }
        if let subject = options["--validation-subject"] {
            var flags = settings["folders"] as? [String: Bool] ?? [:]
            switch options["--subject-validation"] { case "automatic": flags.removeValue(forKey: subject); case "enabled": flags[subject] = true; case "disabled": flags[subject] = false; default: throw StudioError("Choose automatic, enabled or disabled folder validation.") }
            settings["folders"] = flags
        }
        return try validation(settings)
    }
    private static func create(_ root: URL, options: [String: String]) throws {
        try Task.checkCancellation()
        let name = try checkedName(options["--name"] ?? ""), description = try checkedDescription(options["--description"] ?? "")
        let size = try checkedSize(options["--training-size"] ?? "1024")
        guard !["dataset.json", "sources", ".training-data"].contains(root.lastPathComponent), (try? root.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { throw StudioError("Choose a new dataset folder.") }
        try FileManager.default.createDirectory(at: root.deletingLastPathComponent(), withIntermediateDirectories: true)
        var created = false
        if !FileManager.default.fileExists(atPath: root.path) { try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); created = true }
        do {
            try locked(root) {
                guard try FileManager.default.contentsOfDirectory(atPath: root.path).allSatisfy({ $0 == ".material-workbench.lock" }) else { throw StudioError("That folder already contains files. Choose a new or empty dataset folder.") }
                let now = timestamp()
                let index: Object = ["schema_version": 2, "generator": "ipde-material-dataset-v2", "dataset_id": UUID().uuidString.lowercased(), "name": name, "description": description, "created_utc": now, "updated_utc": now, "dataset_management": ["schema": management, "owns_directory": true], "storage_policy": "original-source-references-v1", "source_images_modified": false, "original_sources_required_for_training": true, "split_strategy": "whole-material-v1", "validation_scope": "known_subject_diagnostic", "validation": try editedValidation(validation(nil), options: options), "training_size": size, "samples": [Object]()]
                try Task.checkCancellation()
                try atomic(root.appendingPathComponent("dataset.json"), data: jsonData(index))
            }
        } catch {
            if created, !FileManager.default.fileExists(atPath: root.appendingPathComponent("dataset.json").path), let remaining = try? FileManager.default.contentsOfDirectory(atPath: root.path), remaining.allSatisfy({ $0 == ".material-workbench.lock" }) { for file in remaining { try? FileManager.default.removeItem(at: root.appendingPathComponent(file)) }; try? FileManager.default.removeItem(at: root) }
            throw error
        }
    }
    private static func checkedName(_ name: String) throws -> String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.unicodeScalars.count <= 200, !name.unicodeScalars.contains(where: { $0.value < 32 }) else { throw StudioError("Enter a name with 1–200 characters and no control characters.") }
        return trimmed
    }
    private static func checkedDescription(_ value: String) throws -> String { guard !value.contains("\0"), value.unicodeScalars.count <= 100_000 else { throw StudioError("Dataset description must be text without null characters.") }; return value }
    private static func checkedSize(_ value: String) throws -> Int { guard let size = Int(value), sizes.contains(size) else { throw StudioError("Choose a native training size of 256, 512, 1024 or 2048.") }; return size }
    private static func deletion(_ root: URL, index: Object, records: [Record]) throws -> Object {
        guard root.path != "/", root.path != NSHomeDirectory(), !FileManager.default.fileExists(atPath: root.appendingPathComponent(".git").path) else { throw StudioError("This is a protected workspace folder.") }
        let stages = root.appendingPathComponent(".training-data")
        for stage in (try? FileManager.default.contentsOfDirectory(at: stages, includingPropertiesForKeys: nil)) ?? [] where stage.lastPathComponent.hasPrefix(".preparing-") {
            if let ownership = try? object(stage.appendingPathComponent(".native-crop-preparation.json")), ownership["source_dataset_path"] as? String == root.path, let pid = ownership["pid"] as? Int, kill(pid_t(pid), 0) == 0 { throw StudioError("Finish or cancel active dataset preparation before deleting its dataset.") }
        }
        let owner = index["dataset_management"] as? Object ?? [:]
        var safe = owner["schema"] as? String == management && owner["owns_directory"] as? Bool == true
        for record in records { for source in sources(record.sample) { guard let path = source["path"] as? String, !isInside(canonical(URL(fileURLWithPath: path)), root) else { safe = false; continue } } }
        let permitted: Set<String> = ["dataset.json", ".material-workbench.lock", ".material-size-reviews.json", ".DS_Store"]
        var enumerationFailed = false
        if let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], errorHandler: { _, _ in enumerationFailed = true; return false }) {
            for case let file as URL in files {
                let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                let checkedFile = file.resolvingSymlinksInPath().standardizedFileURL
                guard isInside(checkedFile, root) else { safe = false; continue }
                let relative = String(checkedFile.path.dropFirst(root.path.count + 1)), parts = relative.split(separator: "/")
                if values.isSymbolicLink == true { safe = false }
                if values.isRegularFile == true && !permitted.contains(relative) && !(parts.count == 3 && parts.first == "samples" && parts.last == "sample.json") { safe = false }
            }
        } else { safe = false }
        if enumerationFailed { safe = false }
        return ["dataset_path": root.path, "index_sha256": try digest(root.appendingPathComponent("dataset.json")), "safe_to_trash_folder": safe, "trash_paths": [safe ? root.path : root.appendingPathComponent("dataset.json").path]]
    }
    private static func checkIndex(_ root: URL, expected: String?) throws { if let expected, try digest(root.appendingPathComponent("dataset.json")) != expected { throw StudioError("Dataset changed since selection; reload it before editing.") } }
    private static func locked<T>(_ root: URL, body: () throws -> T) throws -> T {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &directory), directory.boolValue else { throw StudioError("The dataset folder is missing. Create a dataset or choose an existing dataset folder.") }
        let descriptor = Darwin.open(root.appendingPathComponent(".material-workbench.lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(descriptor) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else { throw StudioError("This dataset is being edited in another window. Try again when it finishes.") }
        defer { flock(descriptor, LOCK_UN) }
        try recover(root)
        return try body()
    }
    private static func commit(_ root: URL, updates: [(URL, Object)]) throws {
        try Task.checkCancellation()
        let stage = root.appendingPathComponent(".material-workbench-transaction-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { if !FileManager.default.fileExists(atPath: root.appendingPathComponent(journal).path) { try? FileManager.default.removeItem(at: stage) } }
        var changes: [Object] = []
        for (offset, update) in updates.enumerated() {
            try Task.checkCancellation()
            let (path, value) = update, data = try jsonData(value)
            let staged = stage.appendingPathComponent("\(offset).json")
            try atomic(staged, data: data)
            changes.append(["path": String(path.path.dropFirst(root.path.count + 1)),
                            "old_sha256": try optionalDigest(path) as Any? ?? NSNull(),
                            "new_sha256": hash(data), "staged_path": String(staged.path.dropFirst(root.path.count + 1))])
        }
        try atomic(root.appendingPathComponent(journal), data: jsonData(["schema": schema, "changes": changes]))
        try recover(root)
    }
    private static func recover(_ root: URL) throws {
        let path = root.appendingPathComponent(journal)
        guard FileManager.default.fileExists(atPath: path.path) else { return }
        let document = try object(path)
        guard Set(document.keys) == ["schema", "changes"], document["schema"] as? String == schema, let changes = document["changes"] as? [Object] else { throw StudioError("Unrecognized dataset recovery journal.") }
        var stages = Set<URL>()
        func bytes(_ change: Object) throws -> Data {
            if let relative = change["staged_path"] as? String {
                let staged = try contained(root, relative), directory = staged.deletingLastPathComponent()
                guard directory.deletingLastPathComponent() == root, directory.lastPathComponent.hasPrefix(".material-workbench-transaction-"), staged.pathExtension == "json" else { throw StudioError("Invalid dataset recovery staging path.") }
                stages.insert(directory)
                return try Data(contentsOf: staged)
            }
            // Finish recovery journals written by earlier app versions.
            guard let encoded = change["new_base64"] as? String, let data = Data(base64Encoded: encoded) else { throw StudioError("Invalid dataset recovery identity.") }
            return data
        }
        // Validate every change before publishing any of them. Keep only one
        // small metadata file in memory; the journal contains paths, not bytes.
        for change in changes {
            guard let relative = change["path"] as? String, let expected = change["new_sha256"] as? String else { throw StudioError("Invalid dataset recovery identity.") }
            let file = try contained(root, relative), current = try optionalDigest(file)
            guard ["sample.json", "dataset.json", ".material-size-reviews.json"].contains(file.lastPathComponent),
                  hash(try bytes(change)) == expected,
                  current == change["old_sha256"] as? String || current == expected else { throw StudioError("Dataset recovery identity changed; original data remains untouched.") }
        }
        for change in changes { try atomic(contained(root, change["path"] as! String), data: bytes(change)) }
        try FileManager.default.removeItem(at: path)
        for stage in stages { try FileManager.default.removeItem(at: stage) }
        try sync(root)
    }
    private static func atomic(_ path: URL, data: Data) throws {
        let temporary = path.deletingLastPathComponent().appendingPathComponent(".\(path.lastPathComponent)-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try data.write(to: temporary, options: .withoutOverwriting)
        let descriptor = Darwin.open(temporary.path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let status = fsync(descriptor), issue = errno; Darwin.close(descriptor)
        guard status == 0 else { throw POSIXError(POSIXErrorCode(rawValue: issue) ?? .EIO) }
        guard Darwin.rename(temporary.path, path.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        try sync(path.deletingLastPathComponent())
    }
    private static func sync(_ directory: URL) throws {
        let descriptor = Darwin.open(directory.path, O_RDONLY)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(descriptor) }
        guard fsync(descriptor) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    }
    private static func canonical(_ url: URL) -> URL {
        var root = url
        if root.lastPathComponent == "dataset.json", FileManager.default.fileExists(atPath: root.path) { root.deleteLastPathComponent() }
        if root.lastPathComponent == "sources", FileManager.default.fileExists(atPath: root.deletingLastPathComponent().appendingPathComponent("dataset.json").path) { root.deleteLastPathComponent() }
        return root.resolvingSymlinksInPath().standardizedFileURL
    }
    private static func contained(_ root: URL, _ name: String) throws -> URL {
        guard !name.hasPrefix("/"), !name.split(separator: "/").contains("..") else { throw StudioError("Dataset metadata path must stay inside its directory.") }
        let path = root.appendingPathComponent(name).resolvingSymlinksInPath().standardizedFileURL
        guard isInside(path, root) else { throw StudioError("Dataset metadata path must stay inside its directory.") }; return path
    }
    private static func isInside(_ path: URL, _ root: URL) -> Bool { path.path == root.path || path.path.hasPrefix(root.path + "/") }
    private static func object(_ url: URL) throws -> Object { guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? Object else { throw StudioError("Invalid dataset metadata: \(url.path)") }; return value }
    private static func optionalObject(_ url: URL) throws -> Object { FileManager.default.fileExists(atPath: url.path) ? try object(url) : [:] }
    private static func jsonData(_ value: Object) throws -> Data { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]) + Data([10]) }
    private static func jsonString(_ value: Object) throws -> String { String(decoding: try jsonData(value), as: UTF8.self) }
    private static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private static func digest(_ url: URL) throws -> String { try hash(Data(contentsOf: url, options: .mappedIfSafe)) }
    private static func optionalDigest(_ url: URL) throws -> String? { FileManager.default.fileExists(atPath: url.path) ? try digest(url) : nil }
    private static func timestamp() -> String { ISO8601DateFormatter().string(from: Date()) }
}
