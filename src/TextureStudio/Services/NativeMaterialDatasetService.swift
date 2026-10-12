import CryptoKit
import Darwin
import Foundation
import ImageIO

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
        var inputURL: URL
        var targetURL: URL
        var inputSHA256: String
        var targetSHA256: String
        let inputEncoding: String
        var targetConvention: String
        let augmentationSourceInputURL: URL?
        let augmentationSourceInputSHA256: String?
        let augmentationSourceTargetURL: URL?
        let augmentationSourceTargetSHA256: String?
        let augmentationSourceTargetConvention: String?
        let cropRectangle: [Int]?
        let subjectID: String?
        let cropKind: String?
        var materialName: String?

        init(id: String, split: String, width: Int, height: Int, inputURL: URL, targetURL: URL,
             inputSHA256: String, targetSHA256: String, inputEncoding: String, targetConvention: String,
             augmentationSourceInputURL: URL? = nil, augmentationSourceInputSHA256: String? = nil,
             augmentationSourceTargetURL: URL? = nil, augmentationSourceTargetSHA256: String? = nil,
             augmentationSourceTargetConvention: String? = nil, cropRectangle: [Int]? = nil,
             subjectID: String? = nil, cropKind: String? = nil, materialName: String? = nil) {
            self.id = id; self.split = split; self.width = width; self.height = height
            self.inputURL = inputURL; self.targetURL = targetURL; self.inputSHA256 = inputSHA256
            self.targetSHA256 = targetSHA256; self.inputEncoding = inputEncoding; self.targetConvention = targetConvention
            self.augmentationSourceInputURL = augmentationSourceInputURL; self.augmentationSourceInputSHA256 = augmentationSourceInputSHA256
            self.augmentationSourceTargetURL = augmentationSourceTargetURL; self.augmentationSourceTargetSHA256 = augmentationSourceTargetSHA256
            self.augmentationSourceTargetConvention = augmentationSourceTargetConvention
            self.cropRectangle = cropRectangle; self.subjectID = subjectID; self.cropKind = cropKind
            self.materialName = materialName
        }
    }
    /// Prepared datasets are disposable children of the original dataset.
    /// Predictions and per-run augmentation files are siblings of those crops,
    /// so cleanup-size never deletes reusable model inputs or run provenance.
    static func trainingCacheRoot(datasetURL: URL) throws -> URL {
        let dataset = datasetURL.resolvingSymlinksInPath().standardizedFileURL
        var original = dataset
        let manifest = dataset.appendingPathComponent("dataset.json")
        if FileManager.default.fileExists(atPath: manifest.path),
           let binding = try object(manifest)["native_size_preparation"] as? Object {
            guard binding["schema"] as? String == preparationSchema,
                  let path = binding["source_dataset_path"] as? String, path.hasPrefix("/") else {
                throw StudioError("Native crop dataset has an invalid source lineage.")
            }
            original = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
            guard dataset.deletingLastPathComponent() == original.appendingPathComponent(".training-data") else {
                throw StudioError("Native crop dataset differs from its owned source location.")
            }
        }
        let root = original.appendingPathComponent(".training-data")
        guard (try? root.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
            throw StudioError("Native training data cannot be a symbolic link.")
        }
        return root
    }

    static func trainingSamples(datasetURL: URL, size: Int, target: String, materials: [String] = []) throws -> [TrainingSample] {
        let root = canonical(datasetURL), index = try object(root.appendingPathComponent("dataset.json"))
        let preparation = index["native_size_preparation"] as? Object
        if let preparation, preparation["target_resized"] as? Bool != false {
            throw StudioError("Refresh training crops from the original dataset. Scaled training maps are not supported.")
        }
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
            guard ([output] + variants).allSatisfy({
                let source = $0["source"] as? Object ?? [:]
                return ($0["sample_sha256"] as? String)?.count == 64 &&
                    ($0["storage"] as? String != "source_reference" || (source["file_sha256"] as? String)?.count == 64)
            }) else { throw StudioError("Prepare training crops to verify these newly registered originals before training.") }
            guard ([output] + variants).allSatisfy({ trainingMapIsExact($0, prepared: preparation != nil) }) else {
                throw StudioError("Refresh training crops from the original dataset. Training maps must use exact crops without scaling or interpolation.")
            }
            guard ([output] + variants).allSatisfy({ map in
                guard map["storage"] as? String == "source_reference" else { return true }
                let source = map["source"] as? Object ?? [:]
                return source["width"] as? Int == size && source["height"] as? Int == size
            }) else {
                throw StudioError("Prepare exact training crops before training. Original map dimensions differ from the selected grid.")
            }
            let outputURL = try resolve(targetName, output)
            let transforms = output["transforms"] as? [Object] ?? []
            let convention = transforms.contains { $0["type"] as? String == "directx_to_opengl" && $0["applied_in_memory"] as? Bool == true } ? "directx" : "opengl"
            return try variants.enumerated().map { offset, color in
                guard let inputName = color["filename"] as? String ?? (sample["maps"] as? [String: String])?["input"], let inputHash = color["sample_sha256"] as? String, let targetHash = output["sample_sha256"] as? String else { throw StudioError("Training map checksums are missing.") }
                let inputSource = color["source"] as? Object ?? [:], targetSource = output["source"] as? Object ?? [:]
                return TrainingSample(id: (sample["sample_id"] as? String ?? "") + (offset == 0 ? "" : "|" + (color["variant_id"] as? String ?? "\(offset)")), split: sample["split"] as? String ?? "train", width: size, height: size, inputURL: try resolve(inputName, color), targetURL: outputURL, inputSHA256: inputHash, targetSHA256: targetHash, inputEncoding: color["encoding"] as? String ?? "source_srgb_assumed", targetConvention: convention,
                    augmentationSourceInputURL: (inputSource["path"] as? String).map { URL(fileURLWithPath: $0) },
                    augmentationSourceInputSHA256: inputSource["file_sha256"] as? String,
                    augmentationSourceTargetURL: (targetSource["path"] as? String).map { URL(fileURLWithPath: $0) },
                    augmentationSourceTargetSHA256: targetSource["file_sha256"] as? String,
                    augmentationSourceTargetConvention: ["nor_dx", "normal_dx"].contains(targetSource["suffix"] as? String ?? "") ? "directx" : "opengl",
                    cropRectangle: sample["crop_rectangle_top_left_xywh"] as? [Int], subjectID: subjectID(sample),
                    cropKind: sample["crop_kind"] as? String ?? sample["source_region_id"] as? String,
                    materialName: sample["name"] as? String)
            }
        }
    }

    /// Inspect the registered projection without inventory maintenance,
    /// downloads or metadata writes, including for audits of live datasets.
    static func readOnlyDatasetInfo(datasetURL: URL, size: Int, target: String = "height") throws -> [String: Any] {
        let root = canonical(datasetURL), index = try object(root.appendingPathComponent("dataset.json"))
        return try info(root, index: index, records: readRecords(root, index: index), options: ["--review-size": String(size), "--target": target])
    }

    /// A saved resized cache may have the expected dimensions and checksum.
    /// Its declared pixel transformation must also satisfy the crop contract.
    private static func trainingMapIsExact(_ map: Object, prepared: Bool) -> Bool {
        guard map["exact_source_crop"] as? Bool != false,
              !["resized", "scaled", "resampled", "image_resizing"].contains(where: { map[$0] as? Bool == true }) else { return false }
        let transforms = map["transforms"] as? [Object] ?? []
        if map["transforms"] != nil, !(map["transforms"] is [Object]) { return false }
        for transform in transforms {
            switch transform["type"] as? String {
            case "native_crop":
                guard transform["algorithm"] as? String == "exact_native_integer_codes",
                      transform["range_normalization"] as? Bool == false,
                      transform["gamma_applied"] as? Bool == false else { return false }
            case "directx_to_opengl":
                guard transform["component"] as? String == "G" else { return false }
            default: return false
            }
        }
        guard prepared else { return true }
        switch map["storage"] as? String {
        case "source_reference":
            return !transforms.contains { $0["type"] as? String == "native_crop" }
        case "prepared_crop":
            return map["exact_source_crop"] as? Bool == true && transforms.contains { $0["type"] as? String == "native_crop" }
        default: return false
        }
    }
    private typealias Object = [String: Any]
    private struct Record { var entry: Object; let path: URL; var sample: Object }
    static let commands: Set<String> = ["dataset", "create-dataset", "edit-dataset", "remove-material", "validate-delete", "curate", "remove-missing", "add-material", "scan-folder", "import-folder", "prepare-size", "cleanup-size", "add-source", "rename-source", "move-source", "tag-source", "repair-source-tags"]
    private static let sizes = [256, 512, 1024, 2048]
    private static let schema = "texture-studio-material-workbench-v1"
    private static let management = "texture-studio-dataset-management-v1"
    private static let journal = ".material-workbench-journal.json"

    /// nil means this service does not own the requested native command.
    static func run(arguments: [String], preparationPolicy: PreparationPolicy? = nil,
                    sourceRecovery: NativeMaterialSourceRecovery? = nil,
                    onEvent: @escaping @Sendable (String) -> Void = { _ in }) async throws -> String? {
        guard let command = arguments.first, commands.contains(command) else { return nil }
        try Task.checkCancellation()
        let operation = Task.detached(priority: .userInitiated) {
            let context = try recoveryContext(arguments)
            let recovery = sourceRecovery ?? NativeMaterialSourceRecovery()
            var attempts: [RecoveryAttempt] = []
            if !context.requests.isEmpty, let event = try? jsonString(["event": "source_recovery_started", "completed": 0, "total": context.requests.count, "training_size": context.trainingSize as Any? ?? NSNull()]) { onEvent(event + "\n") }
            for (offset, request) in context.requests.enumerated() {
                try Task.checkCancellation()
                let result = try await recovery.recover(request) { message in
                    if let event = try? jsonString(["event": "source_recovery_progress", "message": message, "completed": offset, "total": context.requests.count, "asset": request.asset, "resolution": request.resolution, "target": request.target]) { onEvent(event + "\n") }
                }
                attempts.append(RecoveryAttempt(request: request, result: result))
                if let event = try? jsonString(["event": "source_recovery_progress", "message": "Checked \(request.asset) \(request.resolution) \(request.target)", "completed": offset + 1, "total": context.requests.count, "asset": request.asset, "resolution": request.resolution, "target": request.target]) { onEvent(event + "\n") }
            }
            if !attempts.isEmpty {
                var event = recoverySummary(attempts, trainingSize: context.trainingSize)
                event["event"] = "source_recovery_completed"
                if let text = try? jsonString(event) { onEvent(text + "\n") }
            }
            try Task.checkCancellation()
            let revised = try attachRecoveredTargets(arguments, context: context)
            let result = try runSynchronously(revised, preparationPolicy: preparationPolicy, onEvent: onEvent)
            return try recoveryResponse(result, attempts: attempts, trainingSize: context.trainingSize)
        }
        return try await withTaskCancellationHandler {
            let result = try await operation.value
            try Task.checkCancellation()
            return result
        } onCancel: { operation.cancel() }
    }
    private static func runSynchronously(_ arguments: [String], preparationPolicy: PreparationPolicy?, onEvent: @Sendable (String) -> Void) throws -> String? {
        try Task.checkCancellation()
        let options = try parsedOptions(arguments)
        guard let requested = options["--dataset"] else { throw StudioError("Choose a dataset folder.") }
        let command = arguments[0]
        let root = canonical(URL(fileURLWithPath: requested))
        if command == "cleanup-size" { return try jsonString(cleanup(root)) }
        if command == "create-dataset", options["--source-folder"] == "true" {
            guard !FileManager.default.fileExists(atPath: root.path) else { throw StudioError("A folder already exists at this location. Open it directly.") }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        if !FileManager.default.fileExists(atPath: root.appendingPathComponent("dataset.json").path),
           command != "create-dataset" || options["--source-folder"] == "true" {
            return try jsonString(sourceFolderCommand(root, command: command, options: options,
                policy: preparationPolicy ?? .current, onEvent: onEvent))
        }
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
            // Check the user's revision before our own inventory maintenance.
            // Recovery may legitimately change source file statistics.
            if command != "dataset", command != "create-dataset" { try checkIndex(root, expected: options["--expected-index-sha256"]) }
            var index = try object(root.appendingPathComponent("dataset.json"))
            let quickPreview = command == "scan-folder" && options["--quick-preview"] == "true"
            let plannedImport = command == "import-folder" && options["--plan"] != nil
            let managed = options["--managed-import"] == "true" || index["managed_source_folders"] as? Bool == true
            if !quickPreview && !plannedImport && !managed {
                try compactStoredRecords(root, index: index)
                if try sourceInventoryNeedsRefresh(root, index: index) { index = try refreshSourceIndex(root, index: index) }
            }
            let records = try readRecords(root, index: index)
            if ["curate", "remove-missing"].contains(command), let lineage = index["native_size_preparation"] as? Object {
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
                if let tiers = options["--source-tier-count"] { index["source_tier_count"] = try checkedSourceTierCount(tiers) }
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
                    let providerFamilies = Set(sources(record.sample).compactMap { source -> String? in
                        guard let path = source["path"] as? String, let name = recoveryProviderName(URL(fileURLWithPath: path)) else { return nil }
                        return identifier(name.asset)
                    }).sorted()
                    let removed: Object = ["material_id": record.sample["material_id"] ?? NSNull(), "source_set_id": record.sample["source_set_id"] ?? NSNull(),
                        "source_family_id": familyID(record.sample), "subject_id": subjectID(record.sample), "provider_families": providerFamilies]
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

    private struct RecoveryContext: Sendable {
        var requests: [NativeMaterialSourceRecovery.Request] = []
        var original: URL?
        var indexSHA256: String?
        var reviewSHA256: String?
        var trainingSize: Int?
    }
    private struct RecoveryAttempt: Sendable {
        let request: NativeMaterialSourceRecovery.Request
        let result: NativeMaterialSourceRecovery.Result
    }
    private static func parsedOptions(_ arguments: [String]) throws -> [String: String] {
        var options: [String: String] = [:], cursor = 1
        while cursor < arguments.count {
            if ["--automatic-validation", "--skip-source-recovery"].contains(arguments[cursor]) { options[arguments[cursor]] = "true"; cursor += 1; continue }
            guard cursor + 1 < arguments.count, arguments[cursor].hasPrefix("--") else { throw StudioError("Invalid native dataset arguments.") }
            options[arguments[cursor]] = arguments[cursor + 1]; cursor += 2
        }
        return options
    }
    private struct RemovedSourceFamilies {
        var scoped: Set<String> = [], legacy: Set<String> = []
        func excludes(family: String, subject: String) -> Bool { legacy.contains(family) || scoped.contains(subject + "|" + family) }
    }
    /// Removal covers automatic zoom expansion until the user explicitly adds
    /// the family again. Retained originals alone do not authorize new tiers.
    private static func removedSourceFamilies(index: Object, records: [Record]) -> RemovedSourceFamilies {
        func families(_ sample: Object) -> Set<String> {
            var result: Set<String> = [familyID(sample)]
            for source in sources(sample) {
                if let path = source["path"] as? String, let name = recoveryProviderName(URL(fileURLWithPath: path)) { result.insert(identifier(name.asset)) }
            }
            return result
        }
        let active = Set(records.flatMap { record in families(record.sample).map { subjectID(record.sample) + "|" + $0 } })
        let activeFamilies = Set(records.flatMap { families($0.sample) })
        var result = RemovedSourceFamilies()
        for removed in index["removed_materials"] as? [Object] ?? [] {
            let materialID = removed["material_id"] as? String ?? "", sourceSetID = removed["source_set_id"] as? String ?? ""
            let materialFamily = materialID.range(of: #"_(?:\d+k|\d+x\d+)$"#, options: .regularExpression).map { String(materialID[..<$0.lowerBound]) }
            let sourceSetFamily = sourceSetID.range(of: #"_\d+x\d+$"#, options: .regularExpression).map { String(sourceSetID[..<$0.lowerBound]) }
            let legacyFamily = materialFamily ?? sourceSetFamily ?? materialID
            let family = removed["source_family_id"] as? String ?? legacyFamily
            let removedFamilies = Set([family] + (removed["provider_families"] as? [String] ?? []))
            if let subject = removed["subject_id"] as? String {
                for family in removedFamilies where !family.isEmpty && !active.contains(subject + "|" + family) { result.scoped.insert(subject + "|" + family) }
            } else {
                for family in removedFamilies where !family.isEmpty && !activeFamilies.contains(family) { result.legacy.insert(family) }
            }
        }
        return result
    }
    /// Inspect and download before taking the manifest transaction lock or
    /// capturing an import preview. The preview then binds the repaired files.
    private static func recoveryContext(_ arguments: [String]) throws -> RecoveryContext {
        let options = try parsedOptions(arguments), command = arguments[0]
        if options["--skip-source-recovery"] == "true" || options["--managed-import"] == "true" || (command == "scan-folder" && options["--quick-preview"] == "true") { return RecoveryContext() }
        let setup = ["dataset", "scan-folder", "import-folder", "prepare-size"].contains(command)
            || (command == "edit-dataset" && options["--training-size"] != nil)
        guard setup, !(command == "import-folder" && options["--plan"] != nil),
              let requested = options["--dataset"] else { return RecoveryContext() }
        let supplied = canonical(URL(fileURLWithPath: requested))
        guard FileManager.default.fileExists(atPath: supplied.appendingPathComponent("dataset.json").path) else { return RecoveryContext() }
        let initial = try locked(supplied) {
            try checkIndex(supplied, expected: options["--expected-index-sha256"])
            return try object(supplied.appendingPathComponent("dataset.json"))
        }
        if initial["managed_source_folders"] as? Bool == true { return RecoveryContext() }
        let original: URL
        if let lineage = initial["native_size_preparation"] as? Object {
            guard command == "prepare-size", lineage["schema"] as? String == preparationSchema,
                  let path = lineage["source_dataset_path"] as? String, path.hasPrefix("/") else { return RecoveryContext() }
            original = canonical(URL(fileURLWithPath: path))
            guard original != supplied else { throw StudioError("Native dataset has circular source lineage.") }
        } else { original = supplied }
        var context = RecoveryContext(original: original)
        var files: [URL] = [], recorded: [NativeMaterialSourceRecovery.Request] = []
        var sourceTierCount = 1
        var removedFamilies = RemovedSourceFamilies()
        try locked(original) {
            let index = try object(original.appendingPathComponent("dataset.json"))
            context.trainingSize = try checkedSize(options["--size"] ?? options["--training-size"] ?? options["--review-size"]
                ?? (index["training_size"] as? Int).map(String.init) ?? options["--default-review-size"] ?? "1024")
            sourceTierCount = try checkedSourceTierCount(options["--source-tier-count"] ?? (index["source_tier_count"] as? Int).map(String.init) ?? "1")
            context.indexSHA256 = try digest(original.appendingPathComponent("dataset.json"))
            context.reviewSHA256 = try optionalDigest(original.appendingPathComponent(".material-size-reviews.json")) ?? hash(Data())
            if let expected = options["--expected-review-sha256"], expected != context.reviewSHA256 { throw StudioError("Dataset reviews changed; reload before preparing native crops.") }
            let records = try readRecords(original, index: index)
            removedFamilies = removedSourceFamilies(index: index, records: records)
            let selected = options["--material"].flatMap { id in records.first { $0.sample["material_id"] as? String == id } }
            for record in records {
                if options["--material"] != nil {
                    guard let selected, subjectID(record.sample) == subjectID(selected.sample), familyID(record.sample) == familyID(selected.sample) else { continue }
                }
                for source in sources(record.sample) {
                    guard let path = source["path"] as? String else { continue }
                    let url = URL(fileURLWithPath: path)
                    files.append(url)
                    if !FileManager.default.fileExists(atPath: path), let name = recoveryProviderName(url),
                       let role = providerRole(name.suffix), let sha = source["file_sha256"] as? String {
                        recorded.append(.init(asset: name.asset, resolution: name.resolution,
                            target: name.suffix == "nor_dx" ? "normal_dx" : role, destination: url,
                            width: source["width"] as? Int ?? 0, height: source["height"] as? Int ?? 0,
                            expectedSHA256: sha, publishedRole: role == "input" && name.suffix.hasPrefix("col_") ? name.suffix : nil,
                            provider: name.provider))
                    }
                }
            }
        }
        // Existing records authorize only their own provider asset in each
        // folder. Other assets beside a selected source remain outside scope.
        var authorizedFolders: [URL: Set<String>] = [:]
        for file in files {
            guard let name = recoveryProviderName(file) else { continue }
            authorizedFolders[canonical(file.deletingLastPathComponent()), default: []].insert(providerAssetKey(name))
        }
        for folder in authorizedFolders.keys.sorted(by: { $0.path < $1.path }) {
            let allowed = authorizedFolders[folder]!
            let siblings = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            files += siblings.filter { recoveryProviderName($0).map { allowed.contains(providerAssetKey($0)) } ?? false }
        }
        if command != "prepare-size", let folder = options["--folder"] {
            let url = canonical(URL(fileURLWithPath: folder))
            if !FileManager.default.fileExists(atPath: url.appendingPathComponent("dataset.json").path) {
                let candidates = try inventory(url, dataset: original).compactMap { ($0.first as? String).map { URL(fileURLWithPath: $0) } }
                files += candidates.filter { file in
                    guard options["--material"] != nil else { return true }
                    guard let name = recoveryProviderName(file) else { return false }
                    return authorizedFolders[canonical(file.deletingLastPathComponent())]?.contains(providerAssetKey(name)) == true
                }
            }
        }
        // Fresh owned sources also need their zoom tiers before the first
        // native read discovers and registers them.
        let ownedSources = original.appendingPathComponent("sources")
        if options["--material"] == nil, FileManager.default.fileExists(atPath: ownedSources.path) {
            files += try inventory(ownedSources, dataset: original).compactMap { ($0.first as? String).map { URL(fileURLWithPath: $0) } }.filter { file in
                guard let name = recoveryProviderName(file) else { return true }
                return !removedFamilies.excludes(family: identifier(name.asset), subject: canonical(file.deletingLastPathComponent()).path)
            }
        }
        var requests = Dictionary(recorded.map { ($0.destination.path, $0) }, uniquingKeysWith: { first, _ in first })
        let targets = command == "prepare-size" ? [options["--target"] ?? "height"] : ["height", "roughness", "normal"]
        for target in targets {
            for request in croppingRecoveryRequests(files: files, size: context.trainingSize!, target: target, tierCount: sourceTierCount) {
                if requests[request.destination.path] == nil { requests[request.destination.path] = request }
            }
        }
        // Ambient archives are shared by all maps of one tier. Group requests
        // so a bounded archive cache needs only the active tier.
        context.requests = requests.values.sorted {
            ($0.provider.rawValue, $0.asset, $0.resolution, $0.target, $0.destination.path)
                < ($1.provider.rawValue, $1.asset, $1.resolution, $1.target, $1.destination.path)
        }
        return context
    }
    static func croppingRecoveryRequests(files: [URL], size: Int, target: String, tierCount: Int = 1) -> [NativeMaterialSourceRecovery.Request] {
        guard ["height", "roughness", "normal"].contains(target), size > 0, size <= Int.max / 4, (1...3).contains(tierCount) else { return [] }
        let firstTier = max(1024, size)
        // Explicit generic map names do not prove a downloadable provider
        // identity, even if their color filename resembles a provider name.
        let genericFamilies = Set(files.compactMap { file -> String? in
            guard parsedProvider(file.lastPathComponent) == nil, let name = sourceMapFilename(file.lastPathComponent) else { return nil }
            return canonical(file.deletingLastPathComponent()).path + "|" + identifier(name.0)
        })
        struct Field {
            var roles: Set<String>
            let width: Int
            let height: Int
            var dimensionsByRole: [String: Set<[Int]>] = [:]
        }
        var fields: [String: [Int: Field]] = [:], locations: [String: (ProviderSourceName, URL)] = [:]
        var inputRoles: [String: Set<String>] = [:]
        for file in Set(files) {
            guard let name = recoveryProviderName(file), let role = providerRole(name.suffix),
                  let scale = Int(name.resolution.dropLast()), let header = try? NativePNG.inspect(file) else { continue }
            guard isProviderResolutionField(dimensions: [header.width, header.height], resolution: name.resolution) else { continue }
            let nominal = scale * 1024
            let folder = file.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL
            guard !genericFamilies.contains(folder.path + "|" + identifier(name.asset)) else { continue }
            let key = folder.path + "|" + providerAssetKey(name)
            locations[key] = (name, folder)
            var field = fields[key]?[nominal] ?? Field(roles: [], width: header.width, height: header.height)
            field.roles.insert(role)
            field.dimensionsByRole[role, default: []].insert([header.width, header.height])
            fields[key, default: [:]][nominal] = field
            if role == "input", name.provider == .polyHaven { inputRoles[key, default: []].insert(name.suffix) }
        }
        var requests: [NativeMaterialSourceRecovery.Request] = []
        for key in fields.keys.sorted() {
            let (name, folder) = locations[key]!, current = fields[key]!
            let inputSuffix = inputRoles[key]?.sorted { left, right in
                func variant(_ value: String) -> Int {
                    let digits = value.replacingOccurrences(of: #"^(?:diffuse|diff|color|col|albedo)_?"#, with: "", options: .regularExpression)
                    return Int(digits) ?? 0
                }
                return (variant(left), left) < (variant(right), right)
            }.first
            // Existing higher-resolution originals stay discoverable. Only
            // expansion downloads follow the user's selected tier budget.
            let tiers = Set((0..<tierCount).map { firstTier * (1 << $0) }.filter { [1024, 2048, 4096, 8192, 16384].contains($0) })
            // Rectangular providers may begin above the requested square grid.
            // A complete native pair already covers that grid; do not acquire
            // absent smaller tiers merely to imitate square-source filenames.
            let firstCompleteRectangle = current.keys.filter { pixels in
                let field = current[pixels]!
                let paired = (field.dimensionsByRole["input"] ?? []).intersection(field.dimensionsByRole[target] ?? [])
                return paired.contains { dimensions in
                    let edge = min(dimensions[0], dimensions[1])
                    return edge >= size && max(dimensions[0], dimensions[1]) / edge >= 2
                }
            }.min()
            for pixels in tiers.sorted() {
                let field = current[pixels], roles = field?.roles ?? []
                if field == nil, let firstCompleteRectangle, pixels < firstCompleteRectangle { continue }
                for role in ["input", target] where !roles.contains(role) {
                    let label = "\(pixels / 1024)k"
                    let selectedInput = role == "input" ? inputSuffix : nil
                    let publishedRole = selectedInput.flatMap { ["diff", "diffuse"].contains($0) ? nil : $0 }
                    requests.append(.init(asset: name.asset, resolution: label, target: role,
                        destination: providerDestination(folder: folder, name: name, resolution: label, target: role, inputSuffix: selectedInput),
                        width: field?.width ?? pixels, height: field?.height ?? pixels, publishedRole: publishedRole,
                        provider: name.provider, preserveOriginalDimensions: true))
                }
            }
        }
        return requests
    }
    /// Published native fields can have a different aspect ratio. A matching
    /// nominal edge establishes the provider tier; actual crop eligibility is
    /// checked separately against both original dimensions.
    static func isProviderResolutionField(dimensions: [Int], resolution: String) -> Bool {
        guard dimensions.count == 2, dimensions.allSatisfy({ $0 > 0 }),
              ["1k", "2k", "4k", "8k", "16k"].contains(resolution.lowercased()),
              let scale = Int(resolution.dropLast()) else { return false }
        let nominal = scale * 1024, tolerance = max(2, Int(ceil(Double(nominal) * 0.02)))
        return dimensions.contains { abs($0 - nominal) <= tolerance }
    }
    private struct ProviderSourceName {
        let provider: NativeMaterialSourceRecovery.Provider
        let asset: String
        let suffix: String
        let resolution: String
    }
    private static func recoveryProviderName(_ file: URL) -> ProviderSourceName? {
        guard let (_, suffix, resolution) = parsedProvider(file.lastPathComponent) else { return nil }
        let stem = file.deletingPathExtension().lastPathComponent
        for (offset, expression) in providerFilenameExpressions.enumerated() {
            guard let match = expression.firstMatch(in: stem, range: NSRange(stem.startIndex..., in: stem)),
                  let range = Range(match.range(at: 1), in: stem) else { continue }
            let originalAsset = String(stem[range])
            let ambient = offset == 1 || originalAsset.range(of: #"^[A-Z][A-Za-z]+[0-9]{3,}[A-Za-z]?$"#, options: .regularExpression) != nil
            return ProviderSourceName(provider: ambient ? .ambientCG : .polyHaven,
                asset: ambient ? originalAsset : identifier(originalAsset), suffix: suffix, resolution: resolution)
        }
        return nil
    }
    private static func providerAssetKey(_ name: ProviderSourceName) -> String { name.provider.rawValue + "|" + identifier(name.asset) }
    private static func providerDestination(folder: URL, name: ProviderSourceName, resolution: String, target: String, inputSuffix: String? = nil) -> URL {
        if name.provider == .ambientCG {
            let map = ["input": "Color", "height": "Displacement", "roughness": "Roughness", "normal": "NormalGL"] [target]!
            return folder.appendingPathComponent("\(name.asset)_\(resolution.uppercased())-PNG_\(map).png")
        }
        let suffix = target == "input" && inputSuffix != nil && !["diff", "diffuse"].contains(inputSuffix!) ? inputSuffix!
            : ["input": "diff", "height": "disp", "roughness": "rough", "normal": "nor_gl"] [target]!
        return folder.appendingPathComponent("\(name.asset)_\(suffix)_\(resolution).png")
    }
    /// Add only absent roles. Previously recorded pixels and their reviews stay
    /// authoritative; a download never replaces changed or competing originals.
    private static func attachRecoveredTargets(_ arguments: [String], context: RecoveryContext) throws -> [String] {
        guard let root = context.original, let snapshot = context.indexSHA256 else { return arguments }
        let options = try parsedOptions(arguments)
        return try locked(root) {
            try Task.checkCancellation()
            try checkIndex(root, expected: snapshot)
            let reviewPath = root.appendingPathComponent(".material-size-reviews.json")
            guard try optionalDigest(reviewPath) ?? hash(Data()) == context.reviewSHA256 else { throw StudioError("Dataset reviews changed during source recovery. Reload the dataset.") }
            var index = try object(root.appendingPathComponent("dataset.json")), reviews = try optionalObject(reviewPath)
            var updates: [(URL, Object)] = [], changedReviews = false
            let originalRecords = try readRecords(root, index: index)
            let selected = options["--material"].flatMap { id in originalRecords.first { $0.sample["material_id"] as? String == id } }
            let authorized = originalRecords.filter { record in
                guard options["--material"] != nil else { return true }
                guard let selected else { return false }
                return subjectID(record.sample) == subjectID(selected.sample) && familyID(record.sample) == familyID(selected.sample)
            }
            var authorizedFolders: [URL: Set<String>] = [:]
            for record in authorized {
                var metadata = record.sample["map_metadata"] as? [String: Object] ?? [:]
                guard let input = metadata["input"]?["source"] as? Object, let path = input["path"] as? String,
                      let name = recoveryProviderName(URL(fileURLWithPath: path)),
                      let dimensions = record.sample["source_pixel_dimensions"] as? [Int], dimensions.count == 2 else { continue }
                let folder = URL(fileURLWithPath: path).deletingLastPathComponent()
                authorizedFolders[canonical(folder), default: []].insert(identifier(name.asset))
                var additions: [String: Object] = [:]
                let siblings = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
                for target in ["height", "roughness", "normal"] where metadata[target] == nil {
                    let preferred = providerDestination(folder: folder, name: name, resolution: name.resolution, target: target)
                    let candidates = [preferred] + siblings.sorted { $0.path < $1.path }.filter { url in
                        guard let candidate = recoveryProviderName(url) else { return false }
                        return providerAssetKey(candidate) == providerAssetKey(name) && candidate.resolution == name.resolution && providerRole(candidate.suffix) == target
                    }
                    for url in candidates {
                        guard var source = try? sourceSummary(url), [source["width"] as? Int ?? 0, source["height"] as? Int ?? 0] == dimensions,
                              target != "height" || source["sample_bits"] as? Int == 16,
                              target != "normal" || [3, 4].contains(source["channels"] as? Int ?? 0) else { continue }
                        source["path"] = canonical(url).path
                        source["suffix"] = recoveryProviderName(url)?.suffix
                        additions[target] = source; break
                    }
                }
                guard !additions.isEmpty else { continue }
                guard sources(record.sample).allSatisfy({ source in
                    guard let path = source["path"] as? String, let sha = source["file_sha256"] as? String else { return false }
                    return (try? digest(URL(fileURLWithPath: path))) == sha
                }) else { continue }
                var sample = record.sample, maps = sample["maps"] as? [String: String] ?? [:]
                for (role, summary) in additions {
                    var source = summary
                    source["resolution_label"] = name.resolution; source["source_family_id"] = identifier(name.asset); source["asset_family_id"] = identifier(name.asset)
                    let path = source["path"] as! String; maps[role] = path
                    metadata[role] = ["source": source, "storage": "source_reference", "filename": path, "sample_sha256": source["file_sha256"]!, "sample_bits": source["sample_bits"]!, "channels": source["channels"]!, "encoding": "linear_data", "transforms": [Object](), "exact_source_crop": true]
                }
                sample["maps"] = maps; sample["map_metadata"] = metadata
                sample["available_targets"] = ["height", "roughness", "normal"].filter { metadata[$0] != nil && ($0 != "height" || ((metadata[$0]?["source"] as? Object)?["sample_bits"] as? Int == 16)) }
                let oldBinding = sourceBinding(record.sample), newBinding = sourceBinding(sample), id = sample["material_id"] as? String ?? ""
                for key in reviews.keys where key.split(separator: ":", maxSplits: 1).last?.hasPrefix(id + "_") == true {
                    guard var review = reviews[key] as? Object, review["source_binding_sha256"] as? String == oldBinding else { continue }
                    review["source_binding_sha256"] = newBinding; reviews[key] = review; changedReviews = true
                }
                updates.append((record.path, sample))
            }
            try Task.checkCancellation()
            if !updates.isEmpty {
                index["updated_utc"] = timestamp()
                if changedReviews { updates.append((reviewPath, reviews)) }
                try commit(root, updates: updates + [(root.appendingPathComponent("dataset.json"), index)])
            }
            // Imported folders remain authoritative outside root/sources too.
            // Register only new sets for assets already authorized by the
            // original selection; existing records retain their exact evidence.
            if context.trainingSize != nil {
                var records = try readRecords(root, index: index)
                var registered = Set(records.compactMap { $0.sample["material_id"] as? String })
                let removed = Set((index["removed_materials"] as? [Object] ?? []).compactMap { $0["material_id"] as? String })
                for folder in authorizedFolders.keys.sorted(by: { $0.path < $1.path }) {
                    try Task.checkCancellation()
                    let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isRegularFileKey])
                        .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true }
                    let allowed = authorizedFolders[folder]!
                    let discovered = try discover(folder, dataset: root, records: records, inventoriedFiles: files).0
                    // Newly found color variants belong to the same authored
                    // native set when its canonical input and paired targets
                    // still reference the exact recorded maps. Preserve the
                    // original ID and reviews instead of manufacturing a
                    // second material through the name-collision namespace.
                    var colorUpdates: [(URL, Object)] = [], reboundReviews = try optionalObject(reviewPath)
                    var colorsChangedReviews = false
                    for candidate in discovered {
                        let incoming = try sourceRecord(candidate)
                        let incomingMetadata = incoming["map_metadata"] as? [String: Object] ?? [:]
                        let incomingVariants = incoming["input_variants"] as? [Object] ?? []
                        guard allowed.contains(candidate["source_family_id"] as? String ?? ""),
                              let record = records.first(where: { existing in
                                  guard familyID(existing.sample) == candidate["source_family_id"] as? String,
                                        subjectID(existing.sample) == folder.path,
                                        existing.sample["source_pixel_dimensions"] as? [Int] == incoming["source_pixel_dimensions"] as? [Int],
                                        let primary = ((existing.sample["map_metadata"] as? [String: Object])?["input"]?["source"] as? Object),
                                        incomingVariants.contains(where: { sameSourceContents($0["source"] as? Object ?? [:], primary) &&
                                            ($0["source"] as? Object)?["path"] as? String == primary["path"] as? String }) else { return false }
                                  return (existing.sample["map_metadata"] as? [String: Object] ?? [:]).allSatisfy { role, details in
                                      guard role != "input" else { return true }
                                      let old = details["source"] as? Object ?? [:], new = incomingMetadata[role]?["source"] as? Object ?? [:]
                                      return sameSourceContents(old, new) && old["path"] as? String == new["path"] as? String
                                  }
                              }) else { continue }
                        let oldVariants = record.sample["input_variants"] as? [Object] ?? []
                        var variants = oldVariants
                        for variant in incomingVariants {
                            let source = variant["source"] as? Object ?? [:]
                            if !variants.contains(where: { ($0["source"] as? Object)?["path"] as? String == source["path"] as? String &&
                                sameSourceContents($0["source"] as? Object ?? [:], source) }) { variants.append(variant) }
                        }
                        guard variants.count != oldVariants.count else { continue }
                        var sample = record.sample; sample["input_variants"] = variants
                        let oldBinding = sourceBinding(record.sample), newBinding = sourceBinding(sample)
                        let id = sample["material_id"] as? String ?? ""
                        for key in reboundReviews.keys where key.split(separator: ":", maxSplits: 1).last?.hasPrefix(id + "_") == true {
                            guard var review = reboundReviews[key] as? Object, review["source_binding_sha256"] as? String == oldBinding else { continue }
                            review["source_binding_sha256"] = newBinding; reboundReviews[key] = review; colorsChangedReviews = true
                        }
                        colorUpdates.append((record.path, sample))
                    }
                    if !colorUpdates.isEmpty {
                        if colorsChangedReviews { colorUpdates.append((reviewPath, reboundReviews)) }
                        index["updated_utc"] = timestamp()
                        try commit(root, updates: colorUpdates + [(root.appendingPathComponent("dataset.json"), index)])
                        records = try readRecords(root, index: index)
                    }
                    let fresh = discovered.filter {
                        let id = $0["material_id"] as? String ?? ""
                        return allowed.contains($0["source_family_id"] as? String ?? "") && !registered.contains(id) && !removed.contains(id)
                    }
                    if !fresh.isEmpty {
                        _ = try register(root, index: index, records: records, materials: fresh, options: [:])
                        index = try object(root.appendingPathComponent("dataset.json"))
                        records = try readRecords(root, index: index)
                        registered = Set(records.compactMap { $0.sample["material_id"] as? String })
                    }
                }
            }
            var revised = arguments
            func replace(_ option: String, with value: String) {
                if let offset = revised.firstIndex(of: option), offset + 1 < revised.count { revised[offset + 1] = value }
            }
            if canonical(URL(fileURLWithPath: options["--dataset"] ?? "")) == root { replace("--expected-index-sha256", with: try digest(root.appendingPathComponent("dataset.json"))) }
            replace("--expected-review-sha256", with: try optionalDigest(reviewPath) ?? hash(Data()))
            return revised
        }
    }
    private static func recoverySummary(_ attempts: [RecoveryAttempt], trainingSize: Int?) -> Object {
        var summary: Object = ["completed": attempts.count, "total": attempts.count, "training_size": trainingSize as Any? ?? NSNull(),
                               "recovered": 0, "not_published": 0, "failed": 0, "already_present": 0]
        for attempt in attempts {
            let key: String
            switch attempt.result { case .recovered: key = "recovered"; case .notPublished: key = "not_published"; case .failed: key = "failed"; case .alreadyPresent: key = "already_present" }
            summary[key] = (summary[key] as! Int) + 1
        }
        return summary
    }
    private static func recoveryResponse(_ text: String?, attempts: [RecoveryAttempt], trainingSize: Int?) throws -> String? {
        guard let text, !attempts.isEmpty, var result = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? Object else { return text }
        func annotate(_ value: Any) -> Any {
            if let array = value as? [Any] { return array.map(annotate) }
            guard var object = value as? Object else { return value }
            if object["code"] as? String == "missing_target", let target = object["target"] as? String,
               let path = object["source_path"] as? String, let name = recoveryProviderName(URL(fileURLWithPath: path)),
               let attempt = attempts.first(where: { $0.request.provider == name.provider && $0.request.asset == name.asset && $0.request.resolution == name.resolution && $0.request.target == target && canonical($0.request.destination.deletingLastPathComponent()).path == canonical(URL(fileURLWithPath: path).deletingLastPathComponent()).path }) {
                let label = target == "height" ? "displacement" : target
                switch attempt.result {
                case .notPublished:
                    object["code"] = "not_published"
                    object["reason"] = "\(name.provider == .ambientCG ? "ambientCG" : "Poly Haven") does not publish a \(label) map for this source set. Import continues with its available maps; only \(label) training skips this set."
                case .failed(let reason):
                    object["code"] = "recovery_failed"
                    object["reason"] = "\(label.capitalized) is unavailable: \(reason) Other maps remain usable; recovery will retry on the next scan."
                default: break
                }
            }
            for key in object.keys where key != "reason" { object[key] = annotate(object[key]!) }
            return object
        }
        result = annotate(result) as! Object
        result["source_recovery"] = recoverySummary(attempts, trainingSize: trainingSize)
        result["source_recovery_results"] = attempts.map { attempt -> Object in
            var item: Object = ["provider": attempt.request.provider.rawValue, "asset": attempt.request.asset, "resolution": attempt.request.resolution,
                               "target": attempt.request.target, "source_path": attempt.request.destination.path]
            switch attempt.result {
            case .recovered: item["status"] = "recovered"
            case .alreadyPresent: item["status"] = "already_present"
            case .notPublished: item["status"] = "not_published"; item["reason"] = "The provider does not publish this map at the requested tier."
            case .failed(let reason): item["status"] = "failed"; item["reason"] = reason
            }
            return item
        }
        result["recovered_map_count"] = attempts.filter { if case .recovered = $0.result { return true }; return false }.count
        result["recovered_sources"] = attempts.compactMap { attempt -> Object? in
            guard case .recovered(let proof) = attempt.result else { return nil }
            return ["source_path": proof.destination.path, "download_url": proof.downloadURL.absoluteString, "published_md5": proof.publishedMD5,
                    "published_bytes": proof.publishedBytes, "sha256": proof.sha256, "crop_rectangle": proof.cropRectangle as Any? ?? NSNull(),
                    "provider": proof.provider.rawValue, "archive_sha256": proof.archiveSHA256 as Any? ?? NSNull(), "archive_sha1": proof.archiveSHA1 as Any? ?? NSNull(),
                    "archive_member": proof.archiveMember as Any? ?? NSNull(), "archive_member_crc32": proof.archiveMemberCRC32 as Any? ?? NSNull(), "archive_member_bytes": proof.archiveMemberBytes as Any? ?? NSNull(),
                    "published_bytes_kind": proof.provider == .ambientCG ? "archive" : "image",
                    "provenance_sidecar_path": proof.provenanceSidecarURL?.path as Any? ?? NSNull(), "provenance_warning": proof.provenanceWarning as Any? ?? NSNull()]
        }
        return try jsonString(result)
    }

    /// A source folder is the dataset. This view requires no manifest, lock,
    /// copied image or writes in that folder. Optional teaching choices and
    /// training evidence live in the app cache, independently of membership.
    private static func sourceFolderCache(_ root: URL) -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appendingPathComponent("org.ipde.texture-studio/SourceFolders")
            .appendingPathComponent(String(hash(Data(root.path.utf8)).prefix(24)))
    }
    private static func sourceFolderChoices(_ cache: URL) throws -> Object {
        var choices = try optionalObject(cache.appendingPathComponent("teaching.json"))
        var reviews = choices["crop_reviews"] as? Object ?? [:]
        // Preparation verifies source bindings and cleanup saves later crop
        // decisions here. Those newer decisions survive the next folder open.
        reviews.merge(try optionalObject(cache.appendingPathComponent(".material-size-reviews.json"))) { _, current in current }
        choices["crop_reviews"] = reviews
        return choices
    }
    private static func sourceFolderSnapshot(_ root: URL, options: [String: String]) throws -> (index: Object, records: [Record]) {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &directory), directory.boolValue else {
            throw StudioError("The source folder is unavailable. Choose an existing folder of original images.")
        }
        let cache = sourceFolderCache(root)
        let choices = try sourceFolderChoices(cache)
        let size = try checkedSize(options["--size"] ?? options["--review-size"] ?? options["--training-size"]
            ?? (choices["training_size"] as? Int).map(String.init) ?? "2048")
        let entries = try inventory(root, dataset: cache, managedImages: true)
        let files = entries.compactMap { ($0[0] as? String).map { URL(fileURLWithPath: $0) } }
        let priorIndex = (try? object(cache.appendingPathComponent("dataset.json"))) ?? [:]
        let priorRecords = (try? readRecords(cache, index: priorIndex)) ?? []
        var (materials, warnings) = try discover(root, dataset: cache, records: priorRecords,
            inventoriedFiles: files, headersOnly: true)
        for assignment in choices["assignments"] as? [Object] ?? [] {
            let paths = (assignment["source_files"] as? [Object] ?? []).compactMap { $0["path"] as? String }
            guard !paths.isEmpty, paths.allSatisfy({ FileManager.default.fileExists(atPath: $0) && isInside(URL(fileURLWithPath: $0), root) }) else { continue }
            var current = assignment
            do {
                var selected: [String: URL] = [:]
                for (oldRole, source) in assignment["maps"] as? [String: Object] ?? [:] {
                    guard let path = source["path"] as? String else { continue }
                    let file = URL(fileURLWithPath: path), role = NativeSourceFileTags.read(file)?.role ?? oldRole
                    if ["input", "height", "roughness", "normal"].contains(role) { selected[role] = file }
                }
                let summaries = try sourceSummaries(Array(selected.values), cached: [:], headersOnly: true)
                current = try sourceMaterial(assignment["name"] as? String ?? root.lastPathComponent,
                    paths: selected, summaries: summaries)
            } catch is CancellationError { throw CancellationError() }
            catch { warnings.append("Manual map assignment needs review: " + paths.joined(separator: ", ") + ". " + error.localizedDescription); continue }
            let inputs = Set((assignment["input_variants"] as? [Object] ?? []).compactMap { $0["path"] as? String })
                .union([(assignment["maps"] as? [String: Object])?["input"]?["path"] as? String ?? ""])
            materials.removeAll { !inputs.isDisjoint(with: Set(($0["source_files"] as? [Object] ?? []).compactMap { $0["path"] as? String })) }
            materials.append(current)
        }
        var occupied = Set<String>(), records: [Record] = [], paired: [String: (String, Object)] = [:]
        let reviews = choices["reviews"] as? [String: Object] ?? [:]
        for original in materials {
            try Task.checkCancellation()
            let base = original["material_id"] as? String ?? "material"
            let material = occupied.contains(base) ? disambiguatedMaterial(original, occupied: occupied) : original
            var sample = try sourceRecord(material)
            if let path = (sample["maps"] as? [String: String])?["input"],
               let status = NativeSourceFileTags.read(URL(fileURLWithPath: path))?.reviewStatus {
                sample["status"] = status; sample["review_status"] = status
            }
            let id = sample["material_id"] as! String; occupied.insert(id)
            if let prior = priorRecords.first(where: { sourceLocationIdentity($0.sample) == sourceLocationIdentity(sample) }),
               sources(prior.sample).allSatisfy({ source in
                   guard let path = source["path"] as? String, let state = source["source_stat"] as? Object,
                         let actual = try? fileState(URL(fileURLWithPath: path)) else { return false }
                   return NSDictionary(dictionary: state).isEqual(to: actual)
               }) { sample = prior.sample }
            if let review = reviews[id] { for key in ["status", "review_status", "curation_note"] { sample[key] = review[key] } }
            let sampleID = sample["sample_id"] as! String
            let entry: Object = ["material_id": id, "sample_id": sampleID, "path": "samples/" + sampleID, "status": sample["status"] ?? "unreviewed", "split": "train"]
            records.append(Record(entry: entry, path: cache.appendingPathComponent("samples/" + sampleID + "/sample.json"), sample: sample))
            for source in sources(sample) { if let path = source["path"] as? String { paired[path] = (id, source) } }
        }
        var catalog: [Object] = []
        for file in files {
            try Task.checkCancellation()
            let path = file.resolvingSymlinksInPath().standardizedFileURL.path, role = inferredSourceRole(file)
            var item: Object = ["path": path, "filename": file.lastPathComponent, "directory": file.deletingLastPathComponent().path,
                                "role": role, "state": "needs_assignment"]
            let tags = NativeSourceFileTags.read(file)
            item["class_label"] = tags?.classLabel
            item["source_verification"] = tags?.verifiedSHA256 == nil ? "header_inspected" : "verified"
            do {
                let header = try paired[path]?.1 ?? managedImageHeader(file)
                for key in ["width", "height", "channels"] { item[key] = header[key] }
                item["bits"] = header["sample_bits"]
                if let match = paired[path] { item["state"] = "paired"; item["material_id"] = match.0 }
                else if let issue = warnings.first(where: { $0.contains(path) }) { item["issue"] = issue }
                if role == "ancillary", item["issue"] == nil { item["issue"] = "Supplemental map retained with its source folder." }
                if role == "unknown", item["issue"] == nil { item["issue"] = "Original image. Assign its maps when it is used for material training." }
            } catch is CancellationError { throw CancellationError() }
            catch { item["state"] = "header_error"; item["issue"] = error.localizedDescription }
            catalog.append(item)
        }
        var index: Object = ["schema_version": 2, "generator": "ipde-material-dataset-v2", "name": root.lastPathComponent, "description": "Folder membership is read directly from disk.",
            "is_source_folder": true, "managed_source_folders": true, "source_inventory": catalog,
            "source_folders": [root.path], "training_size": size, "source_tier_count": 1,
            "storage_policy": "original-source-references-v1", "samples": records.map(\.entry),
            "validation": try validation(choices["validation"] as? Object), "warnings": warnings,
            "source_folder_crop_reviews": choices["crop_reviews"] ?? [:]]
        index["inventory_sha256"] = hash(try jsonData(["files": entries, "choices": choices]))
        return (index, records)
    }
    static func sourceFolderRecords(_ root: URL) throws -> [(entry: [String: Any], sample: [String: Any])] {
        try sourceFolderSnapshot(canonical(root), options: [:]).records.map { ($0.entry, $0.sample) }
    }
    private static func sourceFolderCommand(_ root: URL, command: String, options: [String: String],
        policy: PreparationPolicy, onEvent: @Sendable (String) -> Void) throws -> Object {
        let cache = sourceFolderCache(root)
        if ["add-source", "rename-source", "move-source", "tag-source", "repair-source-tags"].contains(command) {
            let changed = try sourceFolderMutation(root, command: command, options: options, onEvent: onEvent)
            if options["--defer-refresh"] == "true" { return ["source_folder_path": root.path, "completed": true, "added_source_path": changed as Any? ?? NSNull()] }
        }
        if ["edit-dataset", "curate", "add-material"].contains(command) {
            // Only an explicit teaching/settings action creates app state.
            try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
            var choices = try sourceFolderChoices(cache)
            if command == "edit-dataset" {
                if let size = options["--training-size"] { choices["training_size"] = try checkedSize(size) }
                let snapshot = try sourceFolderSnapshot(root, options: options)
                var settings = try validation(choices["validation"] as? Object)
                if let subject = options["--validation-subject"] {
                    guard snapshot.records.contains(where: { subjectID($0.sample) == subject }) else { throw StudioError("The selected subject folder is no longer in this dataset.") }
                }
                let size = choices["training_size"] as? Int ?? snapshot.index["training_size"] as? Int ?? 2048
                let reviews = choices["crop_reviews"] as? Object ?? [:]
                if options["--validation-subject"] != nil, options["--subject-validation"] != "automatic" {
                    let plan = try regionPlan(snapshot.records, size: size, reviews: reviews, target: "height", settings: settings)
                    var flags = settings["folders"] as? [String: Bool] ?? [:]
                    for item in plan["subjects"] as? [Object] ?? [] { if let id = item["subject_id"] as? String { flags[id] = item["selected"] as? Bool ?? false } }
                    settings["folders"] = flags
                }
                settings = try editedValidation(settings, options: options)
                if let subject = options["--validation-subject"], options["--subject-validation"] == "enabled" {
                    let plan = try regionPlan(snapshot.records, size: size, reviews: reviews, target: "height", settings: settings)
                    let subjects = plan["subjects"] as? [Object] ?? []
                    guard subjects.contains(where: { $0["subject_id"] as? String == subject && $0["available"] as? Bool == true }) else { throw StudioError("This folder cannot supply a different validation crop at this resolution.") }
                    let flags = settings["folders"] as? [String: Bool] ?? [:]
                    guard subjects.filter({ flags[$0["subject_id"] as? String ?? ""] == true }).count <= (plan["validation_limit"] as? Int ?? 0) else { throw StudioError("Folder flags exceed the validation limit. Disable another folder or raise the limit.") }
                }
                choices["validation"] = settings
            } else if command == "curate" {
                let snapshot = try sourceFolderSnapshot(root, options: options)
                guard let id = options["--sample"], let status = options["--status"], ["approved", "excluded", "unreviewed"].contains(status) else { throw StudioError("Select an exact sample and review status.") }
                let split = options["--split"]
                guard split == nil || ["train", "validation", "automatic"].contains(split!) else { throw StudioError("Choose a training, validation or automatic split.") }
                let reviewStatus = ["approved": "user_approved", "excluded": "user_excluded", "unreviewed": "unreviewed"][status]!
                if options["--original-source"] == "true" {
                    guard let record = snapshot.records.first(where: { $0.sample["sample_id"] as? String == id }), let material = record.sample["material_id"] as? String else { throw StudioError("Select one exact original source ID.") }
                    var reviews = choices["reviews"] as? [String: Object] ?? [:], review = reviews[material] ?? [:]
                    review["status"] = status; review["review_status"] = reviewStatus
                    if let note = options["--note"] { review["curation_note"] = note }
                    reviews[material] = review; choices["reviews"] = reviews
                    for source in sources(record.sample) { if let path = source["path"] as? String {
                        try? NativeSourceFileTags.update(URL(fileURLWithPath: path), reviewStatus: status)
                    } }
                } else {
                    let size = try checkedSize(options["--review-size"] ?? String(snapshot.index["training_size"] as? Int ?? 2048))
                    var reviews = choices["crop_reviews"] as? Object ?? [:]
                    if let expected = options["--expected-review-sha256"], hash(try jsonData(reviews)) != expected { throw StudioError("Dataset reviews changed since selection; reload them before curation.") }
                    var projection = reviewRecords(snapshot.records, size: size)
                    let plan = try regionPlan(snapshot.records, size: size, reviews: reviews, target: "height", settings: validation(choices["validation"] as? Object))
                    for (sampleID, assignment) in plan["assignments"] as? [String: Object] ?? [:] where assignment["region"] as? String == "validation" {
                        if var record = snapshot.records.first(where: { $0.sample["material_id"] as? String == assignment["material_id"] as? String }) {
                            record.sample["sample_id"] = sampleID; record.sample["sample_pixel_dimensions"] = [size, size]
                            record.sample["crop_rectangle_top_left_xywh"] = assignment["crop_rectangle"]
                            record.sample["source_region_id"] = "validation"; record.sample["split"] = "validation"
                            projection.append(record)
                        }
                    }
                    let matches = projection.filter { $0.sample["sample_id"] as? String == id }
                    guard matches.count == 1 else { throw StudioError("Select one exact crop ID at the current resolution.") }
                    let selected = matches[0], key = "\(size):\(id)"
                    var review = currentReview(reviews, key: key, sample: selected.sample)
                    review["status"] = status; review["review_status"] = reviewStatus
                    review["source_binding_sha256"] = sourceBinding(selected.sample)
                    if split == "automatic" { review.removeValue(forKey: "split"); review.removeValue(forKey: "split_assignment") }
                    else if let split { review["split"] = split; review["split_assignment"] = "manual" }
                    if let note = options["--note"] { review["curation_note"] = note }
                    reviews[key] = review
                    if split != nil {
                        let siblings = projection.filter { familyID($0.sample) == familyID(selected.sample) }
                        let manual = siblings.map { record -> Record in
                            var record = record
                            let review = currentReview(reviews, key: "\(size):\(record.sample["sample_id"] as? String ?? "")", sample: record.sample)
                            record.sample.merge(review) { _, value in value }
                            return record
                        }.filter { !["excluded", "rejected"].contains($0.sample["status"] as? String ?? "") }
                        try checkSplitOverlap(manual.filter { $0.sample["split"] as? String ?? "train" == "train" }, manual.filter { $0.sample["split"] as? String == "validation" })
                    }
                    choices["crop_reviews"] = reviews
                }
            } else {
                var paths: [String: URL] = [:]
                for role in ["input", "height", "roughness", "normal"] { if let path = options["--" + role] {
                    let file = canonical(URL(fileURLWithPath: path))
                    guard isInside(file, root) else { throw StudioError("Place the original maps in this source folder before assigning them.") }
                    paths[role] = file
                } }
                let material = try sourceMaterial(options["--name"] ?? "", paths: paths, convention: options["--normal-convention"] ?? "opengl")
                for (role, file) in paths { try NativeSourceFileTags.update(file, role: role) }
                var assignments = choices["assignments"] as? [Object] ?? []
                assignments.removeAll { $0["material_id"] as? String == material["material_id"] as? String }
                assignments.append(material); choices["assignments"] = assignments
            }
            try atomic(cache.appendingPathComponent("teaching.json"), data: jsonData(choices))
            try atomic(cache.appendingPathComponent(".material-size-reviews.json"), data: jsonData(choices["crop_reviews"] as? Object ?? [:]))
        }
        let snapshot = try sourceFolderSnapshot(root, options: options)
        if command == "prepare-size" {
            try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
            var index = snapshot.index
            index["is_source_folder"] = false; index["source_folder_dataset_path"] = root.path
            for record in snapshot.records {
                try FileManager.default.createDirectory(at: record.path.deletingLastPathComponent(), withIntermediateDirectories: true)
                try atomic(record.path, data: jsonData(record.sample))
            }
            try atomic(cache.appendingPathComponent("dataset.json"), data: jsonData(index))
            try atomic(cache.appendingPathComponent(".material-size-reviews.json"), data: jsonData(snapshot.index["source_folder_crop_reviews"] as? Object ?? [:]))
            var preparationOptions = options
            preparationOptions["--expected-index-sha256"] = nil; preparationOptions["--expected-review-sha256"] = nil
            var result = try prepare(cache, options: preparationOptions, policy: policy, onEvent: onEvent)
            try atomic(cache.appendingPathComponent("teaching.json"), data: jsonData(sourceFolderChoices(cache)))
            if let path = result["dataset_path"] as? String {
                let destination = URL(fileURLWithPath: path)
                var prepared = try object(destination.appendingPathComponent("dataset.json"))
                prepared["source_folder_dataset_path"] = root.path
                try atomic(destination.appendingPathComponent("dataset.json"), data: jsonData(prepared))
                let proof = ["index_sha256": snapshot.index["inventory_sha256"] ?? ""]
                result = try preparedInfo(cache, destination: destination, size: try checkedSize(options["--size"] ?? ""), proof: proof, reused: false)
            }
            return result
        }
        guard ["dataset", "create-dataset", "edit-dataset", "curate", "add-material", "add-source", "rename-source", "move-source", "tag-source", "repair-source-tags"].contains(command) else {
            throw StudioError("This action requires a source folder or a prepared training dataset.")
        }
        return try info(root, index: snapshot.index, records: snapshot.records, options: options)
    }
    private static func sourceFolderMutation(_ root: URL, command: String, options: [String: String], onEvent: @Sendable (String) -> Void) throws -> String? {
        func name(_ value: String) throws -> String {
            let value = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty, value != ".", value != "..", !value.contains("/"), !value.contains(":"), !value.hasPrefix(".") else { throw StudioError("Choose a visible folder name without slashes.") }
            return value
        }
        if command == "repair-source-tags" {
            let entries = try inventory(root, dataset: sourceFolderCache(root), managedImages: true)
            var failures: [String] = []
            for (offset, entry) in entries.enumerated() {
                try Task.checkCancellation()
                let file = URL(fileURLWithPath: entry[0] as! String), role = inferredSourceRole(file)
                if role != "unknown", NativeSourceFileTags.read(file)?.role == nil {
                    do { try NativeSourceFileTags.update(file, role: role) }
                    catch { failures.append(file.lastPathComponent + ": " + error.localizedDescription) }
                }
                if offset % 100 == 0, let event = try? jsonString(["event": "source_tag_progress", "completed": offset, "total": entries.count]) { onEvent(event + "\n") }
            }
            if !failures.isEmpty { throw StudioError("Some file tags could not be saved: " + failures.prefix(5).joined(separator: "; ")) }
            return nil
        }
        guard let value = options["--source"] else { throw StudioError("Choose an image or material folder.") }
        let source = canonical(URL(fileURLWithPath: value))
        guard FileManager.default.fileExists(atPath: source.path) else { throw StudioError("This source is no longer present.") }
        if command == "add-source" {
            if isInside(source, root) { return source.path }
            guard !isInside(root, source) else { throw StudioError("Choose a folder outside this dataset's parent chain.") }
            var directory: ObjCBool = false
            _ = FileManager.default.fileExists(atPath: source.path, isDirectory: &directory)
            var parent = root
            if !directory.boolValue {
                guard isManagedImageCandidate(source) else { throw StudioError("Choose an image file or material folder.") }
                let folderName = try name(options["--folder-name"] ?? source.deletingPathExtension().lastPathComponent)
                parent = root.appendingPathComponent(folderName)
                guard isInside(canonical(parent), root), (try? parent.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { throw StudioError("The destination material folder must stay inside the open source folder.") }
                try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            }
            guard isInside(canonical(parent), root) else { throw StudioError("The destination material folder must stay inside the open source folder.") }
            let base = source.lastPathComponent
            var destination = parent.appendingPathComponent(base), suffix = 2
            while FileManager.default.fileExists(atPath: destination.path) {
                let stem = source.deletingPathExtension().lastPathComponent
                let ext = source.pathExtension
                destination = parent.appendingPathComponent(directory.boolValue ? base + " \(suffix)" : stem + " \(suffix)" + (ext.isEmpty ? "" : "." + ext))
                suffix += 1
            }
            try Task.checkCancellation()
            let teaching = sourceTeachingTags(source)
            if options["--move"] == "true" { try FileManager.default.moveItem(at: source, to: destination) }
            else { try FileManager.default.copyItem(at: source, to: destination) }
            try restoreSourceTeachingTags(teaching, at: destination)
            return destination.path
        }
        guard source != root, isInside(source, root) else { throw StudioError("Select an image or material folder inside the open source folder.") }
        if command == "tag-source" {
            let role = options["--role"]
            guard role == nil || ["input", "height", "roughness", "normal", "ancillary", "unknown"].contains(role!) else { throw StudioError("Choose a supported map role.") }
            try NativeSourceFileTags.update(source, role: role, classLabel: options["--class-label"], classifierIdentity: options["--class-label"] == nil ? nil : "user")
        } else if command == "rename-source" {
            let destination = source.deletingLastPathComponent().appendingPathComponent(try name(options["--name"] ?? ""))
            guard destination.path != source.path else { return source.path }
            guard !FileManager.default.fileExists(atPath: destination.path) else { throw StudioError("That name is already in use.") }
            try FileManager.default.moveItem(at: source, to: destination)
            let choicesURL = sourceFolderCache(root).appendingPathComponent("teaching.json")
            if FileManager.default.fileExists(atPath: choicesURL.path) {
                func rebase(_ value: Any) -> Any {
                    if let text = value as? String, text == source.path || text.hasPrefix(source.path + "/") { return destination.path + text.dropFirst(source.path.count) }
                    if let array = value as? [Any] { return array.map(rebase) }
                    if let object = value as? Object { return object.mapValues(rebase) }
                    return value
                }
                var choices = rebase(try object(choicesURL)) as! Object
                if var reviews = choices["reviews"] as? [String: Object] {
                    let oldPrefix = identifier(source.lastPathComponent) + "_", newPrefix = identifier(destination.lastPathComponent) + "_"
                    for (key, review) in reviews where key.hasPrefix(oldPrefix) { reviews[newPrefix + key.dropFirst(oldPrefix.count)] = review }
                    choices["reviews"] = reviews
                }
                try atomic(choicesURL, data: jsonData(choices))
            }
            return destination.path
        } else {
            guard let path = options["--destination"] else { throw StudioError("Choose where to move this material.") }
            let parent = canonical(URL(fileURLWithPath: path))
            guard !isInside(parent, root), !isInside(parent, source) else { throw StudioError("Move the material to a folder outside this source folder.") }
            let destination = parent.appendingPathComponent(source.lastPathComponent)
            guard !FileManager.default.fileExists(atPath: destination.path) else { throw StudioError("A material with that name already exists at the destination.") }
            let teaching = sourceTeachingTags(source)
            try FileManager.default.moveItem(at: source, to: destination)
            try restoreSourceTeachingTags(teaching, at: destination)
            return destination.path
        }
        return source.path
    }
    private static func sourceTeachingTags(_ source: URL) -> [String: NativeSourceFileTags.Tags] {
        var result: [String: NativeSourceFileTags.Tags] = [:]
        if let tag = NativeSourceFileTags.read(source) { result[""] = tag }
        if let files = FileManager.default.enumerator(at: source, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
            for case let entry as URL in files {
                // Foundation may enumerate /private/var URLs for a /var root.
                // Normalize both paths before deriving the copied-file key.
                let file = entry.resolvingSymlinksInPath().standardizedFileURL
                guard file.path.hasPrefix(source.path + "/") else { continue }
                if let tag = NativeSourceFileTags.read(file) { result[String(file.path.dropFirst(source.path.count + 1))] = tag }
            }
        }
        return result
    }
    private static func restoreSourceTeachingTags(_ tags: [String: NativeSourceFileTags.Tags], at destination: URL) throws {
        for (relative, tag) in tags {
            let file = relative.isEmpty ? destination : destination.appendingPathComponent(relative)
            if NativeSourceFileTags.read(file) != nil { continue }
            // Copies receive their own identity and retain teaching decisions;
            // their verification will be established when they enter training.
            try NativeSourceFileTags.update(file, role: tag.role, classLabel: tag.classLabel,
                classifierIdentity: tag.classifierIdentity, reviewStatus: tag.reviewStatus)
        }
    }
    private static func info(_ root: URL, index: Object, records original: [Record], options: [String: String]) throws -> Object {
        let preparation = index["native_size_preparation"] as? Object
        let folderSnapshot = try (index["source_folder_dataset_path"] as? String).map {
            try sourceFolderSnapshot(URL(fileURLWithPath: $0), options: ["--review-size": String(index["crop_size"] as? Int ?? index["training_size"] as? Int ?? 2048)])
        }
        let originalRoot = (index["source_folder_dataset_path"] as? String ?? preparation?["source_dataset_path"] as? String)
            .map { URL(fileURLWithPath: $0) } ?? root
        let metadata = try folderSnapshot?.index ?? (preparation != nil && FileManager.default.fileExists(atPath: originalRoot.appendingPathComponent("dataset.json").path)
            ? object(originalRoot.appendingPathComponent("dataset.json")) : index)
        let registered = try folderSnapshot?.records ?? (preparation != nil && FileManager.default.fileExists(atPath: originalRoot.appendingPathComponent("dataset.json").path)
            ? readRecords(originalRoot, index: metadata) : original)
        let sourceTierCount = try checkedSourceTierCount((metadata["source_tier_count"] as? Int).map(String.init) ?? "1")
        let datasetGrid = preparation == nil ? metadata["training_size"] as? Int : index["crop_size"] as? Int
        let size = options["--review-size"].flatMap(Int.init) ?? datasetGrid ?? options["--default-review-size"].flatMap(Int.init)
        let reviews = try metadata["is_source_folder"] as? Bool == true
            ? metadata["source_folder_crop_reviews"] as? Object ?? [:]
            : optionalObject(root.appendingPathComponent(".material-size-reviews.json"))
        let settings = try validation(metadata["validation"] as? Object)
        var resolutions: Object = [:], plans: Object = [:]
        if preparation == nil, let size {
            guard sizes.contains(size) else { throw StudioError("Choose a supported native training size.") }
            for candidate in metadata["is_source_folder"] as? Bool == true ? [size] : sizes {
                let shared = try regionPlan(original, size: candidate, reviews: reviews, target: "height", settings: settings)
                var targets: Object = ["height": shared]
                for target in ["roughness", "normal"] { targets[target] = planForTarget(shared, records: original, target: target) }
                resolutions[String(candidate)] = targets
            }
            plans = resolutions[String(size)] as? Object ?? [:]
        }
        let selected = plans[options["--target"] ?? "height"] as? Object ?? [:]
        let assignments = selected["assignments"] as? [String: Object] ?? [:]
        var records = preparation == nil && size != nil ? reviewRecords(original, size: size!) : original.filter {
            size == nil || $0.sample["sample_pixel_dimensions"] as? [Int] == [size!, size!]
        }
        for (id, assignment) in assignments.sorted(by: { $0.key < $1.key }) where assignment["region"] as? String == "validation" {
            if var record = original.first(where: { $0.sample["material_id"] as? String == assignment["material_id"] as? String }) {
                record.sample["sample_id"] = id; record.sample["source_region_id"] = "validation"
                record.sample["sample_pixel_dimensions"] = [size!, size!]; record.sample["crop_rectangle_top_left_xywh"] = assignment["crop_rectangle"]
                records.append(record)
            }
        }
        let managed = metadata["managed_source_folders"] as? Bool == true
        let cropMaterials = try browserMaterials(records, originalRoot: originalRoot, size: size, reviews: reviews,
                                                assignments: assignments, applyReviews: preparation == nil,
                                                recoverMissing: !managed)
        let originals = registered.map { record -> Record in
            var record = record
            record.sample["split"] = "original"; record.sample["source_region_id"] = "original"
            return record
        }
        let originalMaterials = try browserMaterials(originals, originalRoot: originalRoot, recoverMissing: !managed)
        let sourceInventory = visibleSourceInventory(index: metadata, records: registered)
        let eligible = sizes.filter { candidate in registered.contains { record in
            !["excluded", "rejected"].contains(record.sample["status"] as? String ?? "") && suppliesCrop(record.sample, size: candidate)
        } }
        func withoutAssignments(_ values: Object) -> Object { values.mapValues { value in var plan = value as? Object ?? [:]; plan.removeValue(forKey: "assignments"); return plan } }
        let revision = try index["is_source_folder"] as? Bool == true ? index["inventory_sha256"] as? String ?? "" : digest(root.appendingPathComponent("dataset.json"))
        let reviewRevision = try metadata["is_source_folder"] as? Bool == true
            ? hash(jsonData(reviews)) : optionalDigest(originalRoot.appendingPathComponent(".material-size-reviews.json")) ?? hash(Data())
        return ["dataset_path": root.path, "index_sha256": revision, "is_source_folder": metadata["is_source_folder"] ?? false,
            "source_folder_path": metadata["is_source_folder"] as? Bool == true ? originalRoot.path as Any : NSNull(),
            "name": metadata["name"] ?? originalRoot.lastPathComponent, "description": metadata["description"] ?? "", "review_sha256": reviewRevision, "training_size": datasetGrid as Any? ?? NSNull(), "source_tier_count": sourceTierCount, "review_size": size as Any? ?? NSNull(), "validation": settings, "subjects": selected["subjects"] ?? [], "material_count": cropMaterials.count, "sample_count": records.count, "source_set_count": registered.count, "training_plans": withoutAssignments(plans), "resolution_plans": resolutions.mapValues { withoutAssignments($0 as? Object ?? [:]) }, "supported_training_sizes": eligible, "validation_scope": selected.isEmpty ? index["validation_scope"] ?? NSNull() : "known_subject_diagnostic", "cross_size_validation_notice": preparation?["cross_size_validation_notice"] ?? NSNull(), "automatic_validation": index["automatic_validation"] ?? NSNull(), "preparation": NSNull(), "materials": cropMaterials, "original_materials": originalMaterials, "source_inventory": sourceInventory, "source_folders": metadata["source_folders"] ?? [], "warnings": metadata["warnings"] ?? []]
    }
    private static func browserMaterials(_ records: [Record], originalRoot: URL, size: Int? = nil,
                                         reviews: Object = [:], assignments: [String: Object] = [:],
                                         applyReviews: Bool = false, recoverMissing: Bool = true) throws -> [Object] {
        var grouped: [String: Object] = [:]
        for record in records {
            var sample = record.sample
            let id = sample["sample_id"] as? String ?? ""
            if applyReviews, let size {
                sample.merge(currentReview(reviews, key: "\(size):\(id)", sample: sample)) { _, new in new }
                if let assignment = assignments[id] { for key in ["split", "split_assignment", "validation_scope", "status"] { sample[key] = assignment[key] } }
                else { sample["split"] = "train"; sample["split_assignment"] = "automatic" }
            }
            var maps: Object = [:]
            for (role, filename) in sample["maps"] as? [String: String] ?? [:] {
                let details = (sample["map_metadata"] as? [String: Object])?[role] ?? [:]
                maps[role] = try mapInfo(filename, details: details, sample: sample, folder: record.path.deletingLastPathComponent(), originalRoot: originalRoot, role: role, recoverMissing: recoverMissing)
            }
            var variants: [Object] = []
            for variant in sample["input_variants"] as? [Object] ?? [] {
                let source = variant["source"] as? Object ?? [:]
                let inputSource = ((sample["map_metadata"] as? [String: Object])?["input"]?["source"] as? Object) ?? [:]
                var details: Object
                if sameSourceContents(source, inputSource), let input = maps["input"] as? Object { details = input }
                else { details = try mapInfo(variant["filename"] as? String ?? variant["path"] as? String ?? source["path"] as? String ?? "", details: variant, sample: sample, folder: record.path.deletingLastPathComponent(), originalRoot: originalRoot, role: "input", recoverMissing: recoverMissing) }
                details["variant_id"] = variant["variant_id"] ?? "default"; variants.append(details)
            }
            let material = sample["material_id"] as? String ?? ""
            let dimensions = sample["sample_pixel_dimensions"] as? [Int] ?? [0, 0]
            guard dimensions.count == 2 else { throw StudioError("Dataset sample dimensions are invalid.") }
            var group = grouped[material] ?? ["material_id": material, "name": sample["name"] ?? material.replacingOccurrences(of: "_", with: " "), "subject_id": subjectID(sample), "source_directory": sample["source_directory"] ?? NSNull(), "samples": [Object]()]
            var samples = group["samples"] as? [Object] ?? []
            samples.append(["sample_id": id, "status": sample["status"] ?? "unreviewed", "split": sample["split"] ?? "train", "split_assignment": sample["split_assignment"] ?? NSNull(), "width": dimensions[0], "height": dimensions[1], "maps": maps, "note": sample["curation_note"] ?? NSNull(), "input_variants": variants, "source_family_id": familyID(sample), "source_set_id": sample["source_set_id"] ?? NSNull(), "source_verification": sample["source_verification"] ?? "verified", "source_region_id": sample["source_region_id"] ?? NSNull(), "available_targets": sample["available_targets"] ?? ["height", "roughness", "normal"].filter { maps[$0] != nil }])
            group["samples"] = samples; grouped[material] = group
        }
        return grouped.keys.sorted().compactMap { grouped[$0] }
    }
    private static func mapInfo(_ filename: String, details: Object, sample: Object, folder: URL, originalRoot: URL, role: String, recoverMissing: Bool = true) throws -> Object {
        let source = details["source"] as? Object ?? [:]
        var path: URL
        if details["storage"] as? String == "source_reference" {
            let verified = (source["file_sha256"] as? String)?.count == 64 && details["sample_sha256"] as? String == source["file_sha256"] as? String
            let pending = source["source_verification"] as? String == "header_inspected" && source["source_stat"] is Object
            guard filename.hasPrefix("/"), filename == source["path"] as? String, verified || pending else {
                throw StudioError("Original map reference differs from its recorded source identity.")
            }
            path = URL(fileURLWithPath: filename)
        } else { path = try contained(folder, filename) }
        var original = (source["path"] as? String).map { URL(fileURLWithPath: $0) }
        if recoverMissing, let current = original, !FileManager.default.fileExists(atPath: current.path), let recovered = try recoverSource(source, root: originalRoot) {
            original = recovered; if details["storage"] as? String == "source_reference" { path = recovered }
        }
        let header = try? NativePNG.inspect(path)
        return ["path": path.path, "sha256": details["sample_sha256"] ?? NSNull(), "source_bits": details["sample_bits"] ?? source["sample_bits"] ?? NSNull(), "encoding": details["encoding"] ?? NSNull(), "width": header?.width as Any? ?? NSNull(), "height": header?.height as Any? ?? NSNull(), "original_source_path": original?.path as Any? ?? NSNull(), "original_source_sha256": source["file_sha256"] ?? NSNull(), "original_source_width": source["width"] ?? NSNull(), "original_source_height": source["height"] ?? NSNull(), "source_verification": source["source_verification"] ?? ((source["file_sha256"] as? String)?.count == 64 ? "verified" : "header_inspected"), "missing": !FileManager.default.fileExists(atPath: path.path), "crop_rectangle": sample["crop_rectangle_top_left_xywh"] ?? NSNull(), "source_normal_convention": role == "normal" ? (details["storage"] as? String == "source_reference" && source["suffix"] as? String == "nor_dx" ? "directx" : "opengl") : NSNull(), "original_normal_convention": role == "normal" ? (source["suffix"] as? String == "nor_dx" ? "directx" : "opengl") : NSNull()]
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

    private static func regionPlan(_ records: [Record], size: Int, reviews: Object, target: String, settings: Object, cropSeed: UInt64 = 0) throws -> Object {
        let materials = Dictionary(records.filter { suppliesCrop($0.sample, size: size) }.map { ($0.sample["material_id"] as? String ?? "", $0) }, uniquingKeysWith: { _, new in new })
        let reserved: [String: Object]
        if cropSeed == 0 { reserved = [:] }
        else {
            let baseline = try regionPlan(records, size: size, reviews: reviews, target: target, settings: settings)
            reserved = (baseline["assignments"] as? [String: Object] ?? [:]).filter { $0.value["split"] as? String == "validation" }
        }
        var excludedAnchors: [String: Set<Int>] = [:]
        for assignment in reserved.values {
            guard let material = assignment["material_id"] as? String, let record = materials[material],
                  let dimensions = record.sample["source_pixel_dimensions"] as? [Int], dimensions.count == 2,
                  min(dimensions[0], dimensions[1]) <= size * 4,
                  let rectangle = assignment["crop_rectangle"] as? [Int] else { continue }
            let positions = selectionRectangles(dimensions, size: size)
            if let anchor = positions.firstIndex(of: rectangle) { excludedAnchors[subjectID(record.sample), default: []].insert(anchor) }
        }
        var assignments: [String: Object] = [:], subjects: [String: Set<String>] = [:], occupied: [String: [[Double]]] = [:]
        func eligible(_ sample: Object) -> Bool {
            guard let map = (sample["map_metadata"] as? [String: Object])?[target] else { return false }
            if target == "height", (map["source"] as? Object)?["sample_bits"] as? Int != 16 { return false }
            return (sample["available_targets"] as? [String] ?? [target]).contains(target)
        }
        for material in materials.keys.sorted() {
            let sample = materials[material]!.sample
            let dimensions = sample["source_pixel_dimensions"] as? [Int] ?? [0, 0]
            guard dimensions.count == 2, min(dimensions[0], dimensions[1]) >= size else { throw StudioError("Invalid original material dimensions.") }
            let subject = subjectID(sample), family = familyID(sample)
            for (region, selectedRectangle) in cropLayout(dimensions, size: size, subject: subject, sourceSet: sample["source_set_id"] as? String ?? material, seed: cropSeed, excludedAnchors: excludedAnchors[subject] ?? []) {
                var rectangle = selectedRectangle
                guard let field = squareFields(dimensions).first(where: { rectangle[0] >= $0[0] && rectangle[1] >= $0[1] && rectangle[0] + size <= $0[0] + $0[2] && rectangle[1] + size <= $0[1] + $0[3] }) else {
                    throw StudioError("The selected native crop is outside its original source field.")
                }
                if region == "random", reserved.values.contains(where: { $0["material_id"] as? String == material && $0["crop_rectangle"] as? [Int] == rectangle }) {
                    if field[2] > size { rectangle[0] = field[0] + (rectangle[0] - field[0] + 1) % (field[2] - size + 1) }
                    else { rectangle[1] = field[1] + (rectangle[1] - field[1] + 1) % (field[3] - size + 1) }
                }
                let identity = material + "_" + region
                let review = currentReview(reviews, key: "\(size):\(identity)", sample: sample)
                let status = review["status"] as? String ?? sample["status"] as? String ?? "unreviewed"
                let included = !["excluded", "rejected"].contains(status)
                let center = [field[0] + (field[2] - size) / 2, field[1] + (field[3] - size) / 2, size, size]
                let isCenter = rectangle == center && (field[2] > size || field[3] > size)
                let kind = isCenter ? "center" : region == "selected" ? "anchor" : region
                assignments[identity] = ["material_id": material, "region": region, "crop_kind": kind, "source_square_rectangle_top_left_xywh": field, "crop_selection_policy": cropSelectionPolicy, "crop_selection_seed": String(cropSeed), "subject_id": subject, "split": review["split"] ?? "train", "split_assignment": review["split_assignment"] ?? "automatic", "validation_scope": "known_subject_diagnostic", "eligible": included && eligible(sample), "target_available": eligible(sample), "status": status, "crop_rectangle": rectangle]
                if included {
                    subjects[subject, default: []].insert(material)
                    // Diagnostic source membership remains stable when a new
                    // run chooses different training pixels.
                    let baseline = cropLayout(dimensions, size: size, subject: subject, sourceSet: sample["source_set_id"] as? String ?? material).first!.1
                    let normalized = normalize(baseline, dimensions)
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
                let blocked = cropLayout(dimensions, size: size, subject: subject, sourceSet: sample["source_set_id"] as? String ?? material).map { normalize($0.1, dimensions) }
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
            let review = currentReview(reviews, key: "\(size):\(id)", sample: sample)
            let status = review["status"] as? String ?? "unreviewed"
            assignments[id] = ["material_id": material, "region": "validation", "subject_id": subject, "split": review["split"] ?? "validation", "split_assignment": review["split_assignment"] ?? "automatic", "validation_scope": "known_subject_diagnostic", "eligible": eligible(sample) && !["excluded", "rejected"].contains(status), "target_available": eligible(sample), "status": status, "crop_rectangle": rectangle]
            subjectPlans[subject]?["selected"] = true; checks.insert(familyID(sample))
        }
        let values = Array(assignments.values)
        return ["assignments": assignments, "subjects": subjectPlans.keys.sorted().compactMap { subjectPlans[$0] }, "size": size, "crop_selection_policy": cropSelectionPolicy, "crop_selection_seed": String(cropSeed), "crop_count": values.filter { $0["split"] as? String == "train" }.count, "source_set_count": materials.count, "subject_count": subjects.count, "train_count": values.filter { $0["eligible"] as? Bool == true && $0["split"] as? String == "train" }.count, "validation_count": values.filter { $0["eligible"] as? Bool == true && $0["split"] as? String == "validation" }.count, "excluded_count": values.filter { ["excluded", "rejected"].contains($0["status"] as? String ?? "") }.count, "unavailable_target_count": values.filter { $0["target_available"] as? Bool == false }.count, "undersized_source_set_count": Set(records.compactMap { $0.sample["material_id"] as? String }).count - materials.count, "smaller_alternate_source_set_count": smallerAlternateMaterials(records, size: size).count, "shared_validation_count": min(cap, ordered.count), "validation_limit": cap, "validation_candidate_count": candidates.count, "regional_families": checks.sorted(), "source_issues": planSourceIssues(records, size: size, target: target, assignments: assignments)]
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
        plan["source_issues"] = planSourceIssues(records, size: shared["size"] as? Int ?? 0, target: target, assignments: assignments)
        return plan
    }
    /// Preserve the source identity and specific cause behind an unavailable
    /// target count, so the preview can explain what the user can repair.
    private static func planSourceIssues(_ records: [Record], size: Int, target: String, assignments: [String: Object]) -> [Object] {
        let materials = Dictionary(records.map { ($0.sample["material_id"] as? String ?? "", $0.sample) }, uniquingKeysWith: { _, new in new })
        let byMaterial = Dictionary(grouping: assignments.values) { $0["material_id"] as? String ?? "" }
        let smallerAlternates = smallerAlternateMaterials(records, size: size)
        let label = target == "height" ? "displacement" : target
        return materials.keys.sorted().compactMap { material -> Object? in
            let sample = materials[material]!, metadata = sample["map_metadata"] as? [String: Object] ?? [:]
            let input = metadata["input"]?["source"] as? Object ?? [:]
            let crops = byMaterial[material] ?? []
            var issue: Object = ["material_id": material, "source_path": input["path"] ?? sample["source_directory"] ?? "", "target": target, "crop_count": crops.count]
            if crops.isEmpty {
                guard !smallerAlternates.contains(material) else { return nil }
                guard let source = sources(sample).first(where: { min($0["width"] as? Int ?? 0, $0["height"] as? Int ?? 0) < size }) else { return nil }
                issue["source_path"] = source["path"] ?? issue["source_path"]
                issue["target"] = NSNull(); issue["code"] = "undersized"
                issue["reason"] = "Original map is \(source["width"] as? Int ?? 0) × \(source["height"] as? Int ?? 0) pixels. Its originals are retained; this set is omitted from \(size) × \(size) crops while other sets continue. To train this set, choose a smaller resolution."
                return issue
            }
            guard crops.contains(where: { $0["target_available"] as? Bool == false }) else { return nil }
            guard let map = metadata[target] else {
                issue["code"] = "missing_target"
                issue["reason"] = "This set has no \(label) map. Import continues with its available maps; only \(label) training skips this set. A matching PNG can be added later."
                return issue
            }
            let source = map["source"] as? Object ?? [:]
            issue["source_path"] = source["path"] ?? issue["source_path"]
            let bits = source["sample_bits"] as? Int ?? 0
            if target == "height", bits != 16 {
                issue["code"] = "unsupported_precision"
                issue["reason"] = bits > 0 ? "Displacement is \(bits)-bit; training requires a 16-bit PNG displacement map." : "Displacement precision could not be verified; training requires a 16-bit PNG displacement map."
            } else {
                issue["code"] = "unavailable_target"
                issue["reason"] = "The registered \(label) map is not available for training. Select a supported matching map."
            }
            return issue
        }
    }
    private static func suppliesCrop(_ sample: Object, size: Int) -> Bool {
        guard let dimensions = sample["source_pixel_dimensions"] as? [Int], dimensions.count == 2,
              size > 0, dimensions.allSatisfy({ $0 >= size }) else { return false }
        let originals = sources(sample)
        return !originals.isEmpty && originals.allSatisfy { min($0["width"] as? Int ?? 0, $0["height"] as? Int ?? 0) >= size }
    }
    /// Lower-resolution copies are ordinary alternatives when the same material
    /// in the same subject folder has enough original pixels for this grid.
    /// An unrelated large material must not hide a genuinely undersized set.
    private static func smallerAlternateMaterials(_ records: [Record], size: Int) -> Set<String> {
        var largerFamilies: [String: Set<String>] = [:]
        for record in records where suppliesCrop(record.sample, size: size) {
            largerFamilies[subjectID(record.sample), default: []].insert(familyID(record.sample))
        }
        return Set(records.compactMap { record in
            let sample = record.sample
            guard !suppliesCrop(sample, size: size),
                  largerFamilies[subjectID(sample)]?.contains(familyID(sample)) == true else { return nil }
            return sample["material_id"] as? String
        })
    }
    static let cropSelectionPolicy = "balanced-native-resolution-crops-v2"
    /// One spatial sample per source resolution. Every paired map and diffuse
    /// variant uses this rectangle; choices share a folder seed. Long native
    /// fields first choose a short-edge square, without resizing or padding.
    static func cropLayout(_ dimensions: [Int], size: Int, subject: String = "", sourceSet: String = "", seed: UInt64 = 0, excludedAnchors: Set<Int> = []) -> [(String, [Int])] {
        guard size > 0, size <= Int.max / 4, dimensions.count == 2, min(dimensions[0], dimensions[1]) >= size else { return [] }
        let width = dimensions[0], height = dimensions[1]
        if width == size && height == size { return [("full", [0, 0, size, size])] }
        func random(_ axis: String) -> UInt64 {
            let identity = "\(cropSelectionPolicy)|\(size)|\(subject)|\(seed)|\(axis)"
            return SHA256.hash(data: Data(identity.utf8)).prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
        }
        let fields = squareFields(dimensions)
        if min(width, height) <= size * 4 {
            let positions = selectionRectangles(dimensions, size: size)
            let perField = positions.count / fields.count
            let allowedFields = fields.indices.filter { field in
                (0..<perField).contains { !excludedAnchors.contains(field * perField + $0) }
            }
            guard !allowedFields.isEmpty else { return [] }
            let field = allowedFields[Int(random("field") % UInt64(allowedFields.count))]
            let allowed = (0..<perField).map { field * perField + $0 }.filter { !excludedAnchors.contains($0) }
            let position = positions[allowed[Int(random("anchor") % UInt64(allowed.count))]]
            let region = perField == 1 && (fields[field][2] > size || fields[field][3] > size) ? "center" : "selected"
            return [(region, position)]
        }
        let field = fields[Int(random("field") % UInt64(fields.count))]
        let x = field[0] + Int(random("\(sourceSet)|x") % UInt64(field[2] - size + 1))
        let y = field[1] + Int(random("\(sourceSet)|y") % UInt64(field[3] - size + 1))
        return [("random", [x, y, size, size])]
    }
    /// Preserve both ends of a non-square field. Evenly separated native
    /// squares also cover a trailing partial strip without padding it.
    private static func squareFields(_ dimensions: [Int]) -> [[Int]] {
        let width = dimensions[0], height = dimensions[1], edge = min(width, height), length = max(width, height)
        let count = length / edge
        guard count >= 2 else { return [[0, 0, width, height]] }
        return (0..<count).map { index in
            let product = UInt64(length - edge).multipliedFullWidth(by: UInt64(index))
            let offset = Int(UInt64(count - 1).dividingFullWidth(product).quotient)
            return width < height ? [0, offset, edge, edge] : [offset, 0, edge, edge]
        }
    }
    private static func selectionRectangles(_ dimensions: [Int], size: Int) -> [[Int]] {
        let fields = squareFields(dimensions)
        return fields.flatMap { field in
            let local: [[Int]]
            if field[2] == size && field[3] == size { local = [[0, 0, size, size]] }
            else if min(field[2], field[3]) <= size * 2 { local = [[(field[2] - size) / 2, (field[3] - size) / 2, size, size]] }
            else { local = anchorRectangles([field[2], field[3]], size: size) }
            return local.map { [field[0] + $0[0], field[1] + $0[1], size, size] }
        }
    }
    private static func anchorRectangles(_ dimensions: [Int], size: Int) -> [[Int]] {
        let width = dimensions[0], height = dimensions[1]
        return [[0, 0, size, size], [width - size, 0, size, size], [0, height - size, size, size], [width - size, height - size, size, size], [(width - size) / 2, (height - size) / 2, size, size]]
    }
    private static func reviewRecords(_ originals: [Record], size: Int) -> [Record] {
        originals.flatMap { original -> [Record] in
            guard suppliesCrop(original.sample, size: size) else { return [] }
            let layout = cropLayout(original.sample["source_pixel_dimensions"] as? [Int] ?? [], size: size, subject: subjectID(original.sample), sourceSet: original.sample["source_set_id"] as? String ?? original.sample["material_id"] as? String ?? "")
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
        func identity(_ source: Object) -> Any {
            if let hash = source["file_sha256"] as? String, hash.count == 64 { return hash }
            return ["path": source["path"] ?? "", "stat": source["source_stat"] ?? [:],
                    "width": source["width"] ?? 0, "height": source["height"] ?? 0,
                    "bits": source["sample_bits"] ?? 0, "channels": source["channels"] ?? 0] as Object
        }
        var maps: Object = [:]
        for (role, details) in sample["map_metadata"] as? [String: Object] ?? [:] { maps[role] = identity(details["source"] as? Object ?? [:]) }
        let colors = Set((sample["input_variants"] as? [Object] ?? []).map { detail -> String in
            let value = identity(detail["source"] as? Object ?? [:])
            if let hash = value as? String { return hash }
            return hash((try? jsonData(value as? Object ?? [:])) ?? Data())
        }).sorted()
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
        let originalView = options["--original-source"] == "true"
        let size = originalView ? nil : options["--review-size"].flatMap(Int.init)
        let projection = size.map { reviewRecords(original, size: $0) } ?? original
        let records = projection.contains(where: { $0.sample["sample_id"] as? String == id }) ? projection : original
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
        let virtual = !originalView && size.map { !cropLayout(selected.sample["source_pixel_dimensions"] as? [Int] ?? [], size: $0).isEmpty } ?? false
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
    /// Bound simultaneously mapped source bytes as well as CPU use. A scan
    /// retains compact metadata only; it never decodes full-resolution pixels.
    static func sourceScanWorkerCount(fileCount: Int, largestFileBytes: UInt64, resources: MachineResources) -> Int {
        guard fileCount > 0 else { return 0 }
        let budget = max(1, min(resources.practicalBytes / 8, 2 * MachineResources.gibibyte))
        let estimated = largestFileBytes.addingReportingOverflow(16 * 1_048_576)
        let footprint = max(1, estimated.overflow ? UInt64.max : estimated.partialValue)
        let memoryWorkers = max(1, Int(min(8, budget / footprint)))
        return min(fileCount, 8, max(1, resources.availableProcessorCount - 2), memoryWorkers)
    }
    private final class SourceScanQueue: @unchecked Sendable {
        private let lock = NSLock()
        private var next = 0
        private var cancelled = false
        private var results: [Result<Data, Error>?]
        init(count: Int) { results = Array(repeating: nil, count: count) }
        func take() -> Int? {
            lock.withLock {
                guard !cancelled, next < results.count else { return nil }
                defer { next += 1 }; return next
            }
        }
        func finish(_ result: Result<Data, Error>, at index: Int) { lock.withLock { results[index] = result } }
        func cancel() { lock.withLock { cancelled = true } }
        var isCancelled: Bool { lock.withLock { cancelled } }
        func completed() throws -> [Result<Data, Error>] {
            try lock.withLock {
                guard !cancelled else { throw CancellationError() }
                guard results.allSatisfy({ $0 != nil }) else { throw StudioError("Source scan workers did not finish their inventory.") }
                return results.map { $0! }
            }
        }
    }
    /// Workers own Swift cancellation contexts, including the PNG chunk loop.
    /// A corrupt file is reported independently; cancelling waits for all
    /// readers to stop before returning or publishing an import preview.
    static func scanSourceFiles(_ paths: [URL], maximumWorkers: Int,
                                inspect: @escaping @Sendable (URL) throws -> Data) throws -> [Result<Data, Error>] {
        try Task.checkCancellation()
        guard !paths.isEmpty else { return [] }
        if maximumWorkers <= 1 || paths.count == 1 {
            return try paths.map { path in
                try Task.checkCancellation()
                do {
                    let data = try autoreleasepool { try inspect(path) }
                    try Task.checkCancellation()
                    return .success(data)
                } catch is CancellationError { throw CancellationError() }
                catch { return .failure(error) }
            }
        }
        let queue = SourceScanQueue(count: paths.count), completion = DispatchGroup()
        let workers = (0..<min(paths.count, max(1, maximumWorkers))).map { _ -> Task<Void, Never> in
            completion.enter()
            return Task.detached(priority: .userInitiated) {
                defer { completion.leave() }
                while let index = queue.take() {
                    do {
                        try Task.checkCancellation()
                        let data = try autoreleasepool { try inspect(paths[index]) }
                        try Task.checkCancellation()
                        queue.finish(.success(data), at: index)
                    } catch is CancellationError { queue.cancel(); return }
                    catch { queue.finish(.failure(error), at: index) }
                }
            }
        }
        while completion.wait(timeout: .now() + .milliseconds(20)) == .timedOut {
            if Task.isCancelled || queue.isCancelled { queue.cancel(); workers.forEach { $0.cancel() } }
        }
        try Task.checkCancellation()
        return try queue.completed()
    }
    private static func sourceSummaries(_ paths: [URL], cached: [String: Object],
                                        manifests: [String: [URL]] = [:], headersOnly: Bool = false) throws -> [String: Result<Object, Error>] {
        let unique = Array(Set(paths.map { $0.resolvingSymlinksInPath().standardizedFileURL })).sorted { $0.path < $1.path }
        var compactCache: [String: Data] = [:]
        for path in unique { if let prior = cached[path.path] { compactCache[path.path] = try jsonData(prior) } }
        let largest = unique.reduce(UInt64(0)) { max($0, UInt64(max(0, (try? fileState($1)["size"] as? Int64) ?? 0))) }
        let savedCache = compactCache
        let results = try scanSourceFiles(unique, maximumWorkers: sourceScanWorkerCount(fileCount: unique.count,
                largestFileBytes: largest, resources: .current)) { path in
            let prior: Object? = try savedCache[path.path].flatMap { try JSONSerialization.jsonObject(with: $0) as? Object }
            var summary = try headersOnly ? previewSourceSummary(path, cached: prior) : sourceSummary(path, cached: prior)
            if headersOnly { return try jsonData(summary) }
            let expected = publishedMD5Candidates(path, source: summary, manifests: manifests[path.deletingLastPathComponent().path] ?? [])
            if !expected.isEmpty {
                // Cached license evidence was verified against these same
                // immutable bytes. New evidence is audited in this worker,
                // rather than serially rereading every provider PNG later.
                if summary["provider"] as? String == "Poly Haven", let verified = summary["published_md5"] as? String,
                   expected.contains(verified.lowercased()) { summary["scan_verified_md5"] = verified.lowercased() }
                else { summary["scan_verified_md5"] = try publishedMD5(path) }
            }
            return try jsonData(summary)
        }
        var summaries: [String: Result<Object, Error>] = [:]
        for (path, result) in zip(unique, results) {
            summaries[path.path] = result.flatMap { bytes in
                Result {
                    guard let summary = try JSONSerialization.jsonObject(with: bytes) as? Object else { throw StudioError("Invalid original map summary.") }
                    return summary
                }
            }
        }
        return summaries
    }
    private static func scannedSummary(_ resolvedPath: URL, summaries: [String: Result<Object, Error>]) throws -> Object {
        guard let result = summaries[resolvedPath.path] else { throw StudioError("Original map location changed while scanning. Scan it again.") }
        return try result.get()
    }
    private static func sourceSummary(_ requested: URL, cached: Object? = nil) throws -> Object {
        try Task.checkCancellation()
        let path = requested.resolvingSymlinksInPath().standardizedFileURL, before = try fileState(path)
        if let tags = NativeSourceFileTags.read(path), let text = tags.verifiedMetadata,
           let metadata = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? Object,
           metadata["file_sha256"] as? String == tags.verifiedSHA256 {
            var result = metadata; result["source_stat"] = before; result["filename"] = path.lastPathComponent
            return result
        }
        if let cached, (cached["file_sha256"] as? String)?.count == 64,
           let state = cached["source_stat"] as? Object, NSDictionary(dictionary: state).isEqual(to: before) {
            return compactMetadata(cached) as! Object
        }
        let data = try Data(contentsOf: path, options: .mappedIfSafe)
        var result = try NativePNG.sourceMetadata(data)
        let after = try fileState(path)
        guard NSDictionary(dictionary: before).isEqual(to: after) else { throw StudioError("Original map changed while inspecting: \(path.path)") }
        let sha256 = hash(data)
        result["file_sha256"] = sha256; result["source_verification"] = "verified"
        result["filename"] = path.lastPathComponent; result["file_bytes"] = after["size"]; result["source_stat"] = after
        if let evidence = NativeMaterialSourceRecovery.sourceProvenance(for: path, sha256: sha256, bytes: data.count,
                width: result["width"] as? Int ?? 0, height: result["height"] as? Int ?? 0) {
            result.merge(evidence) { _, new in new }
        }
        // Attributes are optional on read-only files or filesystems without
        // xattr support. Verification remains valid in the prepared cache.
        return result
    }
    /// Folder intake needs the grid and map roles, not compressed pixel
    /// checksums. Managed references retain this pending state until training
    /// preparation verifies the selected source bytes.
    private static func previewSourceSummary(_ path: URL, cached: Object?) throws -> Object {
        try Task.checkCancellation()
        let before = try fileState(path)
        if let cached, let state = cached["source_stat"] as? Object,
           NSDictionary(dictionary: state).isEqual(to: before) { return compactMetadata(cached) as! Object }
        let header = try NativePNG.inspect(path)
        guard NSDictionary(dictionary: before).isEqual(to: try fileState(path)) else { throw StudioError("Original map changed while inspecting: \(path.path)") }
        if let tags = NativeSourceFileTags.read(path), let text = tags.verifiedMetadata,
           var result = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? Object,
           result["file_sha256"] as? String == tags.verifiedSHA256 {
            result["source_stat"] = before; result["filename"] = path.lastPathComponent
            return result
        }
        return ["width": header.width, "height": header.height, "sample_bits": header.bits,
                "channels": header.channels, "png_color_type": Int(header.color), "interlace": Int(header.interlace),
                "filename": path.lastPathComponent, "file_bytes": before["size"]!, "source_stat": before, "source_verification": "header_inspected"]
    }
    private static func publishedMD5(_ url: URL) throws -> String {
        // Provider download audits sometimes require MD5. Compute it only for
        // that audit, without persisting a second checksum on every map.
        try Task.checkCancellation()
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return Insecure.MD5.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private static func sourceManifest(_ path: URL) -> Object? {
        guard let values = try? path.resourceValues(forKeys: [.fileSizeKey, .isSymbolicLinkKey]),
              values.isSymbolicLink != true, (values.fileSize ?? Int.max) <= 16 * 1024 * 1024 else { return nil }
        return try? object(path)
    }
    private static func publishedMD5Candidates(_ path: URL, source: Object, manifests: [URL]) -> Set<String> {
        guard !manifests.isEmpty, let (asset, _, label) = parsedProvider(path.lastPathComponent),
              let sourceBytes = source["file_bytes"] as? NSNumber else { return [] }
        var family = identifier(asset)
        if family.hasSuffix("_" + label) { family = String(family.dropLast(label.count + 1)) }
        var expected = Set<String>()
        for manifestPath in manifests {
            guard let manifest = sourceManifest(manifestPath), manifest["material_id"] as? String == family,
                  (manifest["resolution"] as? String)?.lowercased() == label, manifest["provider"] as? String == "Poly Haven",
                  manifest["license"] as? String == "CC0-1.0", let api = manifest["api"] as? Object,
                  api["api_url"] as? String == "https://api.polyhaven.com/files/" + family else { continue }
            for (role, downloaded) in manifest["downloaded_maps"] as? [String: Object] ?? [:] {
                guard URL(fileURLWithPath: downloaded["path"] as? String ?? "").lastPathComponent == source["filename"] as? String,
                      downloaded["sha256"] as? String == source["file_sha256"] as? String,
                      (downloaded["bytes"] as? NSNumber)?.int64Value == sourceBytes.int64Value,
                      let published = (manifest["maps"] as? [String: Object])?[role],
                      (published["published_bytes"] as? NSNumber)?.int64Value == sourceBytes.int64Value,
                      (published["url"] as? String)?.hasPrefix("https://dl.polyhaven.org/file/ph-assets/Textures/png/") == true,
                      let md5 = published["published_md5"] as? String else { continue }
                expected.insert(md5.lowercased())
            }
        }
        return expected
    }
    private static func identifier(_ value: String) -> String {
        let ascii = value.decomposedStringWithCompatibilityMapping.unicodeScalars.filter { $0.value < 128 }.map(String.init).joined().lowercased()
        let cleaned = ascii.replacingOccurrences(of: "[^a-z0-9]+", with: "_", options: .regularExpression).trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return cleaned.isEmpty ? "material_" + hash(Data(value.utf8)).prefix(16) : cleaned
    }
    private static func encoding(_ role: String, source: Object) -> String { role != "input" ? "linear_data" : source["png_gamma"] as? Double == 1 && source["srgb_rendering_intent"] == nil ? "linear" : "source_srgb_assumed" }
    /// Authored paired maps may differ by a few edge pixels. Their shared
    /// top-left intersection is cropable without resampling either original.
    static func commonNativeDimensions(_ originals: [[String: Any]]) throws -> [Int] {
        guard !originals.isEmpty else { throw StudioError("Choose original material maps.") }
        let widths = originals.compactMap { $0["width"] as? Int }, heights = originals.compactMap { $0["height"] as? Int }
        guard widths.count == originals.count, heights.count == originals.count,
              let width = widths.min(), let height = heights.min(), width > 0, height > 0,
              widths.max()! - width <= 8, heights.max()! - height <= 8 else {
            throw StudioError("Paired maps must have matching native dimensions or differ by at most eight edge pixels; only their common pixels can be cropped.")
        }
        return [width, height]
    }
    private static func sourceMaterial(_ name: String, paths: [String: URL], convention: String = "opengl",
                                       summaries: [String: Result<Object, Error>]? = nil) throws -> Object {
        let name = try checkedName(name)
        guard paths["input"] != nil, ["height", "roughness", "normal"].contains(where: { paths[$0] != nil }) else { throw StudioError("Choose an input map and at least one height, normal or roughness map.") }
        let family = identifier(name)
        var maps: [String: Object] = [:], warnings: [String] = []
        for role in paths.keys.sorted() {
            let path = paths[role]!.resolvingSymlinksInPath().standardizedFileURL
            var source = try summaries.map { try scannedSummary(path, summaries: $0) } ?? sourceSummary(path)
            if let notice = sourceFormatNotice(path) { warnings.append(notice) }
            guard role != "input" || [1, 2, 3, 4].contains(source["channels"] as? Int ?? 0),
                  role != "normal" || [3, 4].contains(source["channels"] as? Int ?? 0) else { throw StudioError("Diffuse maps need native gray, gray-alpha, RGB or RGBA channels; normal maps need RGB or RGBA channels.") }
            source["path"] = path.path; source["suffix"] = ["input": "diff", "height": "disp", "roughness": "rough", "normal": convention.lowercased() == "directx" ? "nor_dx" : "nor_gl"][role]
            source["source_family_id"] = family; source["asset_family_id"] = family; maps[role] = source
        }
        let size = try commonNativeDimensions(Array(maps.values)), width = size[0], height = size[1], resolution = width == height && width >= 1024 && width % 1024 == 0 ? "\(width / 1024)k" : "\(width)x\(height)"
        for role in maps.keys { maps[role]?["resolution_label"] = resolution }
        maps["input"]?["variant_id"] = "color_default"
        return ["name": name, "material_id": family + "_" + resolution, "source_family_id": family, "source_set_id": "\(family)_\(width)x\(height)", "source_directory": paths["input"]!.resolvingSymlinksInPath().deletingLastPathComponent().path, "common_pixel_dimensions": size, "resolution_label": resolution, "maps": maps, "input_variants": [maps["input"]!], "source_files": Array(maps.values), "warnings": warnings, "ignored_files": [String](), "problems": [String]()]
    }
    private static func sourceRecord(_ material: Object) throws -> Object {
        guard let materialID = material["material_id"] as? String, identifier(materialID) == materialID,
              let maps = material["maps"] as? [String: Object], let dimensions = material["common_pixel_dimensions"] as? [Int], dimensions.count == 2, maps["input"] != nil else { throw StudioError("Invalid native material source set.") }
        let variants = material["input_variants"] as? [Object] ?? []
        let verified = (Array(maps.values) + variants).allSatisfy { ($0["file_sha256"] as? String)?.count == 64 }
        guard try commonNativeDimensions(Array(maps.values) + variants) == dimensions else { throw StudioError("The shared native crop must match the paired maps' common pixels.") }
        guard variants.allSatisfy({ [1, 2, 3, 4].contains($0["channels"] as? Int ?? 0) }), maps["normal"] == nil || [3, 4].contains(maps["normal"]?["channels"] as? Int ?? 0) else { throw StudioError("Diffuse maps need native gray, gray-alpha, RGB or RGBA channels; normal maps need RGB or RGBA channels.") }
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
        return ["schema_version": 2, "generator": "ipde-material-dataset-v2", "sample_id": materialID + "_full", "name": material["name"] ?? material["source_family_id"] ?? materialID, "material_id": materialID, "asset_family_id": material["source_family_id"] ?? materialID, "source_family_id": material["source_family_id"] ?? materialID, "source_set_id": material["source_set_id"] ?? materialID, "source_directory": material["source_directory"] ?? NSNull(), "source_resolution_label": material["resolution_label"] ?? "native", "sample_pixel_dimensions": dimensions, "source_pixel_dimensions": dimensions, "crop_rectangle_top_left_xywh": [0, 0] + dimensions, "status": "unreviewed", "split": "train", "review_status": "unreviewed", "maps": filenames, "map_metadata": metadata, "input_variants": inputVariants, "source_verification": verified ? "verified" : "header_inspected", "source_precision_verified": verified, "crop_values_verified": verified, "source_discovery_warnings": material["warnings"] ?? [], "ignored_source_files": material["ignored_files"] ?? [], "available_targets": ["height", "roughness", "normal"].filter { maps[$0] != nil && ($0 != "height" || (maps[$0]?["sample_bits"] as? Int ?? 0) >= 16) }]
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
    private static func sourceLocationIdentity(_ sample: Object) -> String {
        let maps = sample["map_metadata"] as? [String: Object] ?? [:]
        var values: [String] = maps.keys.sorted().map { role in
            let source = maps[role]?["source"] as? Object ?? [:]
            return role + "|" + (source["path"] as? String ?? "")
        }
        values += Set((sample["input_variants"] as? [Object] ?? []).map {
            ($0["source"] as? Object)?["path"] as? String ?? ""
        }).sorted().map { "color|" + $0 }
        return hash(Data(values.joined(separator: "\n").utf8))
    }
    private static func disambiguatedMaterial(_ original: Object, occupied: Set<String>) -> Object {
        var material = original
        let base = original["material_id"] as? String ?? "material"
        let paths = (original["source_files"] as? [Object] ?? Array((original["maps"] as? [String: Object] ?? [:]).values))
            .compactMap { $0["path"] as? String }.sorted()
        let token = String(hash(Data(((original["source_directory"] as? String ?? "") + "|" + paths.joined(separator: "|")).utf8)).prefix(10))
        var id = base + "_" + token, suffix = 2
        while occupied.contains(id) { id = base + "_" + token + "_" + String(suffix); suffix += 1 }
        material["material_id"] = id
        material["source_set_id"] = (original["source_set_id"] as? String ?? base) + "_" + token
        if material["name"] == nil { material["name"] = original["source_family_id"] ?? base }
        return material
    }
    private static func register(_ root: URL, index initial: Object, records: [Record], materials: [Object], options: [String: String]) throws -> Object {
        guard !materials.isEmpty || options["--managed-import"] == "true" else { throw StudioError("No paired diffuse and surface PNG maps were found.") }
        var index = initial, current = Dictionary(records.map { ($0.sample["material_id"] as? String ?? "", $0.sample) }, uniquingKeysWith: { _, new in new })
        let managed = options["--managed-import"] == "true"
        var identities = Set(current.values.map { managed ? sourceLocationIdentity($0) : sourceIdentity($0) }), pending: [String: Object] = [:], duplicates = 0
        let sizeChanged = options["--training-size"].map { index["training_size"] as? Int != Int($0) } ?? false
        if let size = options["--training-size"] { index["training_size"] = try checkedSize(size) }
        for requested in materials {
            var material = requested
            guard (material["problems"] as? [String] ?? []).isEmpty else { throw StudioError("Ambiguous source maps need explicit map selection.") }
            var sample = try sourceRecord(material)
            let identity = managed ? sourceLocationIdentity(sample) : sourceIdentity(sample)
            if identities.contains(identity) { duplicates += 1; continue }
            if current[sample["material_id"] as! String] != nil || pending[sample["material_id"] as! String] != nil {
                guard managed || material["automatic_discovery"] as? Bool == true else { throw StudioError("A different material already uses this name and native size. Choose another name.") }
                material = disambiguatedMaterial(material, occupied: Set(current.keys).union(pending.keys)); sample = try sourceRecord(material)
            }
            let id = sample["material_id"] as! String
            let folder = try contained(root, "samples/" + (sample["sample_id"] as! String))
            if let prior = try? object(folder.appendingPathComponent("sample.json")), sourceBinding(prior) == sourceBinding(sample) {
                for key in ["status", "split", "split_assignment", "review_status", "curation_note", "source_binding_sha256"] { if let value = prior[key] { sample[key] = value } }
            }
            if var inventory = index["source_inventory"] as? [Object] {
                let paths = Set(sources(sample).compactMap { $0["path"] as? String })
                for offset in inventory.indices where paths.contains(inventory[offset]["path"] as? String ?? "") {
                    inventory[offset]["material_id"] = id; inventory[offset]["state"] = "paired"
                }
                index["source_inventory"] = inventory
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
        if !pending.isEmpty || sizeChanged || options["--managed-import"] == "true" {
            index["samples"] = entries; index["updated_utc"] = timestamp()
            index["removed_materials"] = (index["removed_materials"] as? [Object] ?? []).filter { pending[$0["material_id"] as? String ?? ""] == nil }
            if changedReviews { updates.append((reviewsPath, reviews)) }
            try commit(root, updates: updates + [(root.appendingPathComponent("dataset.json"), index)])
        }
        var result = try info(root, index: index, records: readRecords(root, index: index), options: options)
        result["added_material_count"] = pending.count; result["duplicate_material_count"] = duplicates; return result
    }
    /// Managed folders register locations and headers immediately. Byte-level
    /// evidence belongs to the selected training preparation, not folder intake.
    private static func importManagedFolder(_ root: URL, index initial: Object, records: [Record], folder: URL,
                                            options: [String: String]) throws -> Object {
        let entries = try inventory(folder, dataset: root, managedImages: true)
        let files = entries.compactMap { ($0[0] as? String).map { URL(fileURLWithPath: $0) } }
        let (materials, warnings) = try discover(folder, dataset: root, records: records,
                                                inventoriedFiles: files, headersOnly: true)
        var index = initial
        let previous = index["source_inventory"] as? [Object] ?? []
        var catalog = Dictionary(previous.compactMap { item -> (String, Object)? in
            guard let path = item["path"] as? String else { return nil }; return (path, item)
        }, uniquingKeysWith: { _, new in new })
        let existingCount = catalog.count
        var paired: [String: (String, Object)] = [:]
        for record in records { for source in sources(record.sample) { if let path = source["path"] as? String {
            paired[path] = (record.sample["material_id"] as? String ?? "", source)
        } } }
        for material in materials { for source in (material["source_files"] as? [Object] ?? []) {
            if let path = source["path"] as? String { paired[path] = (material["material_id"] as? String ?? "", source) }
        } }
        for file in files {
            try Task.checkCancellation()
            let path = file.resolvingSymlinksInPath().standardizedFileURL.path
            let role = inferredSourceRole(file)
            var item: Object = ["id": hash(Data(path.utf8)), "path": path, "filename": file.lastPathComponent,
                                "directory": file.deletingLastPathComponent().path, "role": role,
                                "state": "needs_assignment", "source_folder": folder.path]
            do {
                let summary = try paired[path]?.1 ?? managedImageHeader(file)
                for key in ["width", "height", "channels", "source_stat", "source_verification"] { item[key] = summary[key] }
                item["bits"] = summary["sample_bits"]
                if let match = paired[path] { item["state"] = "paired"; item["material_id"] = match.0 }
                else if let issue = warnings.first(where: { $0.contains(file.path) || $0.contains(path) }) {
                    item["state"] = issue.contains("Multiple") || issue.contains("multiple") || issue.contains("same color variant") ? "ambiguous" : "needs_assignment"
                    item["issue"] = issue
                }
                if role == "ancillary", item["issue"] == nil { item["issue"] = "Supplemental map retained with its source folder." }
                if role == "unknown", item["issue"] == nil { item["issue"] = "Choose a map role when this image is used in a material." }
            } catch is CancellationError { throw CancellationError() }
            catch { item["state"] = "header_error"; item["issue"] = error.localizedDescription }
            catalog[path] = item
        }
        index["managed_source_folders"] = true
        index["source_folders"] = Array(Set((index["source_folders"] as? [String] ?? []) + [folder.path])).sorted()
        index["source_inventory"] = catalog.keys.sorted().compactMap { catalog[$0] }
        index["updated_utc"] = timestamp()
        var response = try register(root, index: index, records: records, materials: materials, options: options)
        response["added_file_count"] = max(0, catalog.count - existingCount)
        response["imported_file_count"] = files.count
        response["warnings"] = warnings
        return response
    }
    private static func inferredSourceRole(_ file: URL) -> String {
        if let role = NativeSourceFileTags.read(file)?.role { return role }
        if let (_, suffix, _) = sourceMapFilename(file.lastPathComponent), let role = providerRole(suffix) { return role }
        if let role = mapAlias(file.deletingPathExtension().lastPathComponent) { return role }
        return isAncillarySourceFilename(file.lastPathComponent) ? "ancillary" : "unknown"
    }
    private static func standaloneMapRole(_ file: URL) -> String? {
        guard sourceMapFilename(file.lastPathComponent) == nil else { return nil }
        let role = NativeSourceFileTags.read(file)?.role ?? mapAlias(file.deletingPathExtension().lastPathComponent)
        return role.flatMap { ["input", "height", "roughness", "normal"].contains($0) ? $0 : nil }
    }
    private static func isManagedImageCandidate(_ file: URL) -> Bool {
        isSourceImageCandidate(file) || ["jpg", "jpeg", "jpe", "heic", "heif", "tif", "tiff", "webp", "bmp", "gif", "exr"].contains(file.pathExtension.lowercased())
    }
    private static func managedImageHeader(_ requested: URL) throws -> Object {
        let file = requested.resolvingSymlinksInPath().standardizedFileURL
        if let header = try? previewSourceSummary(file, cached: nil) { return header }
        let before = try fileState(file)
        let settings = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(file as CFURL, settings),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, settings) as? [CFString: Any],
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              width > 0, height > 0 else { throw StudioError("This image's header could not be read. The original file is retained.") }
        guard NSDictionary(dictionary: before).isEqual(to: try fileState(file)) else { throw StudioError("Image changed while inspecting its header.") }
        var summary: Object = ["width": width, "height": height, "source_stat": before, "source_verification": "header_inspected"]
        if let bits = (properties[kCGImagePropertyDepth] as? NSNumber)?.intValue { summary["sample_bits"] = bits }
        if let model = properties[kCGImagePropertyColorModel] as? String {
            if let channels = ["RGB": 3, "CMYK": 4, "Gray": 1][model] { summary["channels"] = channels }
        }
        return summary
    }
    private static func visibleSourceInventory(index: Object, records: [Record]) -> [Object] {
        var catalog = Dictionary((index["source_inventory"] as? [Object] ?? []).compactMap { item -> (String, Object)? in
            guard let path = item["path"] as? String else { return nil }; return (path, item)
        }, uniquingKeysWith: { _, new in new })
        for record in records { for source in sources(record.sample) { if let path = source["path"] as? String {
            var item = catalog[path] ?? ["id": hash(Data(path.utf8)), "path": path,
                "filename": URL(fileURLWithPath: path).lastPathComponent, "directory": URL(fileURLWithPath: path).deletingLastPathComponent().path,
                "role": inferredSourceRole(URL(fileURLWithPath: path)), "state": "paired"]
            item["material_id"] = record.sample["material_id"]
            item["role"] = providerRole(source["suffix"] as? String ?? "") ?? inferredSourceRole(URL(fileURLWithPath: path))
            if let issue = (record.sample["source_verification_issues"] as? [String: String])?[path] {
                item["state"] = "header_error"; item["issue"] = issue
            } else { item["state"] = "paired"; item.removeValue(forKey: "issue") }
            item["width"] = source["width"]; item["height"] = source["height"]
            item["bits"] = source["sample_bits"]; item["channels"] = source["channels"]
            if FileManager.default.fileExists(atPath: path), item["state"] as? String == "missing" { item["state"] = "paired"; item.removeValue(forKey: "issue") }
            catalog[path] = item
        } } }
        return catalog.keys.sorted().compactMap { path in
            var item = catalog[path]!
            if !FileManager.default.fileExists(atPath: path) { item["state"] = "missing"; item["issue"] = "Original file is missing from its recorded location. Its dataset entry is retained." }
            return item
        }
    }

    private static func inventory(_ folder: URL, dataset: URL, managedImages: Bool = false) throws -> [[Any]] {
        var result: [[Any]] = [], failed: Error?
        guard let enumerator = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles], errorHandler: { _, error in failed = error; return false }) else { throw StudioError("Choose an existing folder of material maps.") }
        for case let file as URL in enumerator {
            try Task.checkCancellation()
            let values = try file.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
            if values.isDirectory == true {
                if file.resolvingSymlinksInPath() == dataset || (!managedImages && FileManager.default.fileExists(atPath: file.appendingPathComponent("dataset.json").path)) { enumerator.skipDescendants() }
                continue
            }
            guard values.isRegularFile == true, managedImages ? isManagedImageCandidate(file) : isSourceImageCandidate(file) else { continue }
            let state = try fileState(file)
            result.append([file.path, file.resolvingSymlinksInPath().path, state["size"]!, state["mtime_ns"]!, state["ctime_ns"]!])
        }
        if let failed { throw failed }
        return result.sorted { ($0[0] as? String ?? "") < ($1[0] as? String ?? "") }
    }
    private static let providerFilenameExpressions: [NSRegularExpression] = {
        let known = #"nor_gl|nor_dx|disp_gl|rough_ao|translucent|(?:diffuse|diff|color|col|albedo)(?:_?\d+)?|coll\d+|displacement|roughness|disp|rough|anisotropy_rotation|anisotropy_strength|spec_ior|spec|bump|metal|ao|arm"#
        let patterns = ["^(.+?)_(" + known + #")_(\d+k)(?: \(\d+\))?$"#, #"^([A-Za-z][A-Za-z0-9]*)_([1248]K)-PNG_(Color|Displacement|NormalGL|NormalDX|Roughness)$"#]
        return patterns.map { try! NSRegularExpression(pattern: $0, options: [.caseInsensitive]) }
    }()
    private static func parsedProvider(_ filename: String) -> (String, String, String)? {
        let filename = URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent
        for (offset, expression) in providerFilenameExpressions.enumerated() {
            guard let match = expression.firstMatch(in: filename, range: NSRange(filename.startIndex..., in: filename)) else { continue }
            let fields = (1...3).map { String(filename[Range(match.range(at: $0), in: filename)!]).lowercased() }
            if offset == 1 { return (fields[0], ["color": "diff", "displacement": "disp", "normalgl": "nor_gl", "normaldx": "nor_dx", "roughness": "rough"][fields[2]]!, fields[1]) }
            return (fields[0], fields[1].hasPrefix("coll") ? "col_" + fields[1].dropFirst(4) : fields[1], fields[2])
        }
        return nil
    }
    private static func providerRole(_ suffix: String) -> String? {
        if suffix.range(of: #"^(?:diffuse|diff|color|col|albedo)(?:_?\d+)?$"#, options: .regularExpression) != nil { return "input" }
        return mapAlias(suffix)
    }
    private static let authoredFilenameExpressions: [NSRegularExpression] = {
        let explicit = mapAliases.keys.sorted { $0.count == $1.count ? $0 < $1 : $0.count > $1.count }
            .map { NSRegularExpression.escapedPattern(for: $0).replacingOccurrences(of: "_", with: "[_ -]") }.joined(separator: "|")
        let roles = #"(?:diffuse|diff|color|col|albedo)(?:_?\d+)?|coll\d+|"# + explicit
        return authoredFilenameExpressions(roles: roles)
    }()
    private static func authoredFilenameExpressions(roles: String) -> [NSRegularExpression] {
        // A size label is descriptive only. Actual PNG dimensions decide which
        // maps pair, including non-power-of-two authored sets such as 515 px.
        // Require at least three digits for bare sizes so diff_2 stays a color
        // variant rather than being reinterpreted as a two-pixel resolution.
        let size = #"(?:\d+k|[1-9]\d{2,5}(?:px)?|[1-9]\d{1,5}x[1-9]\d{1,5})"#
        let patterns = ["^(.+?)[_ -](" + roles + ")[_ -](" + size + #")(?: \(\d+\))?$"#,
                        "^(.+?)[_ -](" + size + #")(?:-PNG)?[_ -]("# + roles + #")(?: \(\d+\))?$"#,
                        "^(.+?)[_ -](" + roles + #")(?: \(\d+\))?$"#]
        return patterns.map { try! NSRegularExpression(pattern: $0, options: [.caseInsensitive]) }
    }
    // Supplemental PBR maps are recognizable source files, but none supplies
    // diffuse input or one of this trainer's targets. Keep this list separate
    // from mapAliases so AO and packed maps cannot become color variants.
    private static let ancillaryMapAliases: Set<String> = [
        "ao", "ambientocclusion", "ambient_occlusion", "ambientocclusionmap", "ambient_occlusion_map", "occlusion",
        "rough_ao", "orm", "arm", "rma", "occlusionroughnessmetallic", "occlusion_roughness_metallic",
        "metal", "metallic", "metalness", "bump", "gloss", "glossiness", "smoothness",
        "spec", "specular", "spec_ior", "anisotropy_rotation", "anisotropy_strength",
        "emission", "emissive", "opacity", "translucent"
    ]
    private static let ancillaryFilenameExpressions: [NSRegularExpression] = {
        let roles = ancillaryMapAliases.sorted { $0.count == $1.count ? $0 < $1 : $0.count > $1.count }
            .map { NSRegularExpression.escapedPattern(for: $0).replacingOccurrences(of: "_", with: "[_ -]") }.joined(separator: "|")
        return authoredFilenameExpressions(roles: roles)
    }()
    static func isAncillarySourceFilename(_ filename: String) -> Bool {
        let name = URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent
        if ancillaryMapAliases.contains(normalizedMapAlias(name)) { return true }
        if let (_, suffix, _) = parsedProvider(filename), providerRole(suffix) == nil { return true }
        return ancillaryFilenameExpressions.contains { $0.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)) != nil }
    }
    private static func sameSourceContents(_ left: Object, _ right: Object) -> Bool {
        if let leftPath = left["path"] as? String, let rightPath = right["path"] as? String, leftPath == rightPath { return true }
        guard let hash = left["file_sha256"] as? String, hash.count == 64,
              let other = right["file_sha256"] as? String, other.count == 64 else { return false }
        return hash == other
    }
    /// Explicit material roles only. This recognizes local authored maps as
    /// well as provider names, without assigning arbitrary photos a role or
    /// claiming that a generic filename proves provider provenance.
    static func sourceMapFilename(_ filename: String) -> (String, String, String)? {
        let name = URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent
        // A standalone Base_Color or Diffuse_Color is one role, not a family
        // named Base or Diffuse. Resolve those through their enclosing folder.
        if mapAlias(name) != nil { return nil }
        if let provider = parsedProvider(filename) { return provider }
        for (offset, expression) in authoredFilenameExpressions.enumerated() {
            guard let match = expression.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)),
                  let familyRange = Range(match.range(at: 1), in: name),
                  let roleRange = Range(match.range(at: offset == 1 ? 3 : 2), in: name) else { continue }
            let suffix = canonicalMapSuffix(String(name[roleRange]))
            let resolution = offset == 2 ? "native" : Range(match.range(at: offset == 1 ? 2 : 3), in: name).map { String(name[$0]).lowercased() } ?? "native"
            return (String(name[familyRange]).lowercased(), suffix, resolution)
        }
        return nil
    }
    static func diffuseSourceFilename(_ file: URL) -> (String, String)? {
        if let (family, suffix, _) = sourceMapFilename(file.lastPathComponent), providerRole(suffix) == "input" {
            let digits = suffix.filter(\.isNumber)
            return (identifier(family), digits.isEmpty ? "color_1" : "color_\(Int(digits) ?? 0)")
        }
        if mapAlias(file.deletingPathExtension().lastPathComponent) == "input" {
            return (identifier(file.deletingLastPathComponent().lastPathComponent), "color_1")
        }
        return nil
    }
    private static let mapAliases: [String: String] = {
        let aliases = [
            "input": ["input", "color", "colour", "col", "albedo", "albedocolor", "albedo_color", "basecolor", "base_color", "basecolour", "base_colour", "diffuse", "diff", "dif", "diffusecolor", "diffuse_color", "diffusecolour", "diffuse_colour"],
            "height": ["height", "heightmap", "height_map", "displacement", "displacementmap", "displacement_map", "displace", "displ", "disp", "disp_gl"],
            "roughness": ["roughness", "rough", "roughnessmap", "roughness_map"],
            "normal": ["normal", "normalmap", "normal_map", "nor", "norm", "nrm", "normalgl", "normal_gl", "nor_gl", "norm_gl", "nrm_gl", "normalopengl", "normal_opengl", "normal_open_gl", "normaldx", "normal_dx", "nor_dx", "norm_dx", "nrm_dx", "normaldirectx", "normal_directx"]
        ]
        // Gloss/smoothness, bump and packed ORM/ARM are deliberately absent:
        // they need an explicit conversion or interpretation, not a rename.
        return Dictionary(uniqueKeysWithValues: aliases.flatMap { role, names in names.map { ($0, role) } })
    }()
    private static func normalizedMapAlias(_ name: String) -> String {
        name.lowercased().replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: " ", with: "_")
    }
    private static func mapAlias(_ name: String) -> String? { mapAliases[normalizedMapAlias(name)] }
    private static func canonicalMapSuffix(_ name: String) -> String {
        let raw = normalizedMapAlias(name)
        if ["normalgl", "normal_gl", "nor_gl", "norm_gl", "nrm_gl", "normalopengl", "normal_opengl", "normal_open_gl"].contains(raw) { return "nor_gl" }
        if ["normaldx", "normal_dx", "nor_dx", "norm_dx", "nrm_dx", "normaldirectx", "normal_directx"].contains(raw) { return "nor_dx" }
        return raw.hasPrefix("coll") ? "col_" + raw.dropFirst(4) : raw
    }
    /// Downloads can contain PNG bytes under a different extension. Map names
    /// identify candidates; sourceSummary validates their actual PNG contents
    /// and reports invalid files instead of silently skipping them.
    private static func isSourceImageCandidate(_ file: URL) -> Bool {
        if file.pathExtension.lowercased() == "png" { return true }
        return sourceMapFilename(file.lastPathComponent) != nil || mapAlias(file.deletingPathExtension().lastPathComponent) != nil
    }
    private static func sourceFormatNotice(_ path: URL) -> String? {
        guard path.pathExtension.lowercased() != "png" else { return nil }
        let extensionDescription = path.pathExtension.isEmpty ? "no filename extension" : "a .\(path.pathExtension) extension"
        return "\(path.path): PNG image with \(extensionDescription); detected and read as PNG."
    }

    private static func discover(_ folder: URL, dataset: URL, records: [Record] = [], inventoriedFiles: [URL]? = nil, headersOnly: Bool = false) throws -> ([Object], [String]) {
        var cache: [String: Object] = [:]
        for record in records { for source in sources(record.sample) { if let path = source["path"] as? String { cache[path] = source } } }
        if FileManager.default.fileExists(atPath: folder.appendingPathComponent("dataset.json").path) {
            guard folder != dataset else { throw StudioError("This dataset is already open. Choose another folder to import.") }
            return try locked(folder) {
                let index = try object(folder.appendingPathComponent("dataset.json")), imported = try readRecords(folder, index: index)
                let paths = imported.flatMap { sources($0.sample).compactMap { ($0["path"] as? String).map { URL(fileURLWithPath: $0) } } }
                let summaries = try sourceSummaries(paths, cached: cache, headersOnly: headersOnly)
                let materials: [Object] = try imported.map { record in
                    var maps: [String: Object] = [:]
                    for (role, details) in record.sample["map_metadata"] as? [String: Object] ?? [:] {
                        guard ["input", "height", "roughness", "normal"].contains(role), var source = details["source"] as? Object, let path = source["path"] as? String else { continue }
                        let actual = try scannedSummary(URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL, summaries: summaries)
                        guard headersOnly || actual["file_sha256"] as? String == source["file_sha256"] as? String else { throw StudioError("Imported original source changed since registration.") }
                        source.merge(actual) { _, new in new }; maps[role] = source
                    }
                    guard let input = maps["input"] else { throw StudioError("Imported material has no diffuse source.") }
                    var variants: [Object] = []
                    for detail in record.sample["input_variants"] as? [Object] ?? [] {
                        guard var source = detail["source"] as? Object, let path = source["path"] as? String else { continue }
                        let actual = try scannedSummary(URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL, summaries: summaries)
                        guard headersOnly || actual["file_sha256"] as? String == source["file_sha256"] as? String else { throw StudioError("Imported original color changed since registration.") }
                        source.merge(actual) { _, new in new }; source["variant_id"] = detail["variant_id"] ?? "color_default"; variants.append(source)
                    }
                    if variants.isEmpty { var color = input; color["variant_id"] = "color_default"; variants = [color] }
                    return ["name": record.sample["name"] ?? record.sample["material_id"] ?? "Material", "material_id": record.sample["material_id"] ?? "", "source_family_id": record.sample["source_family_id"] ?? record.sample["material_id"] ?? "", "source_set_id": record.sample["source_set_id"] ?? record.sample["material_id"] ?? "", "source_directory": URL(fileURLWithPath: input["path"] as? String ?? "").deletingLastPathComponent().path, "common_pixel_dimensions": try commonNativeDimensions(Array(maps.values) + variants), "resolution_label": record.sample["source_resolution_label"] ?? "native", "maps": maps, "input_variants": variants]
                }
                return (materials, [])
            }
        }
        let files = try inventoriedFiles ?? inventory(folder, dataset: dataset).compactMap { ($0[0] as? String).map { URL(fileURLWithPath: $0) } }
        var warnings: [String] = [], groups: [String: Object] = [:]
        let byDirectory = Dictionary(grouping: files) { $0.deletingLastPathComponent() }
        let directories = byDirectory.keys.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
        let recognized = files.filter { path in
            ["input", "height", "roughness", "normal"].contains(inferredSourceRole(path))
        }
        var manifestFiles: [String: [URL]] = [:]
        for directory in directories where byDirectory[directory]!.contains(where: { sourceMapFilename($0.lastPathComponent).flatMap { providerRole($0.1) } != nil }) {
            let paths = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey, .isSymbolicLinkKey])) ?? [])
                .filter { $0.lastPathComponent.hasPrefix("material-source") && $0.pathExtension.lowercased() == "json" }
            manifestFiles[directory.resolvingSymlinksInPath().standardizedFileURL.path] = paths
        }
        let summaries = try sourceSummaries(recognized, cached: cache, manifests: manifestFiles, headersOnly: headersOnly)
        for directory in directories {
            try Task.checkCancellation()
            let local = byDirectory[directory]!.sorted { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() }
            let manifests: [(URL, Object)] = (manifestFiles[directory.resolvingSymlinksInPath().standardizedFileURL.path] ?? []).compactMap { path in
                guard let value = sourceManifest(path) else { return nil }; return (path, value)
            }
            for path in local {
                guard let (asset, suffix, label) = sourceMapFilename(path.lastPathComponent) else { continue }
                let role = inferredSourceRole(path)
                guard ["input", "height", "roughness", "normal"].contains(role) else { continue }
                var family = identifier(asset)
                if family.hasSuffix("_" + label) { family = String(family.dropLast(label.count + 1)) }
                let resolved = path.resolvingSymlinksInPath().standardizedFileURL
                var source: Object
                do { source = try scannedSummary(resolved, summaries: summaries) }
                catch is CancellationError { throw CancellationError() }
                catch { warnings.append("Skipped \(path.path): \(error.localizedDescription)"); continue }
                let verifiedMD5 = source.removeValue(forKey: "scan_verified_md5") as? String
                source["path"] = resolved.path; source["suffix"] = suffix; source["resolution_label"] = label; source["source_family_id"] = family; source["asset_family_id"] = family
                for (manifestPath, manifest) in manifests where manifest["material_id"] as? String == family && (manifest["resolution"] as? String)?.lowercased() == label {
                    for (downloadRole, downloaded) in manifest["downloaded_maps"] as? [String: Object] ?? [:] where URL(fileURLWithPath: downloaded["path"] as? String ?? "").lastPathComponent == path.lastPathComponent && downloaded["sha256"] as? String == source["file_sha256"] as? String && downloaded["bytes"] as? Int == source["file_bytes"] as? Int {
                        if manifest["provider"] as? String == "Poly Haven", manifest["license"] as? String == "CC0-1.0", let api = manifest["api"] as? Object, api["api_url"] as? String == "https://api.polyhaven.com/files/" + family, let published = (manifest["maps"] as? [String: Object])?[downloadRole], published["published_bytes"] as? Int == source["file_bytes"] as? Int, let verifiedMD5, verifiedMD5 == (published["published_md5"] as? String)?.lowercased(), (published["url"] as? String)?.hasPrefix("https://dl.polyhaven.org/file/ph-assets/Textures/png/") == true {
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
                let exactKey = "\(directory.path)|\(family)|\(dimensions[0])x\(dimensions[1])"
                // Authored maps can differ by a few edge pixels. A shared
                // explicit tier permits a crop-only intersection; never join
                // different resolution tiers or infer a match from an image
                // with no size label. Compare every source to prevent a chain
                // of small differences from admitting a large mismatch.
                let nearKeys = label == "native" || groups[exactKey] != nil ? [] : groups.keys.filter { candidate in
                    guard candidate.hasPrefix(directory.path + "|" + family + "|"),
                          let prior = groups[candidate],
                          (prior["resolution_labels"] as? [String] ?? []) == [label] else { return false }
                    return (prior["source_files"] as? [Object] ?? []).allSatisfy { existing in
                        abs((existing["width"] as? Int ?? 0) - dimensions[0]) <= 8 &&
                        abs((existing["height"] as? Int ?? 0) - dimensions[1]) <= 8
                    }
                }.sorted()
                let key = groups[exactKey] != nil ? exactKey : nearKeys.count == 1 ? nearKeys[0] : exactKey
                var group = groups[key] ?? ["source_family_id": family, "source_set_id": "\(family)_\(dimensions[0])x\(dimensions[1])", "source_directory": directory.resolvingSymlinksInPath().path, "common_pixel_dimensions": dimensions, "maps": [String: Object](), "input_variants": [Object](), "source_files": [Object](), "warnings": [String](), "ignored_files": [String](), "problems": [String](), "resolution_labels": [String]()]
                if let prior = group["common_pixel_dimensions"] as? [Int], prior.count == 2 {
                    group["common_pixel_dimensions"] = [min(prior[0], dimensions[0]), min(prior[1], dimensions[1])]
                }
                var labels = group["resolution_labels"] as? [String] ?? []; if !labels.contains(label) { labels.append(label) }; group["resolution_labels"] = labels
                var allSources = group["source_files"] as? [Object] ?? []; allSources.append(source); group["source_files"] = allSources
                var maps = group["maps"] as? [String: Object] ?? [:], variants = group["input_variants"] as? [Object] ?? [], problems = group["problems"] as? [String] ?? [], ignored = group["ignored_files"] as? [String] ?? [], issues = group["warnings"] as? [String] ?? []
                if let notice = sourceFormatNotice(path) { issues.append(notice) }
                if role == "input" {
                    let digits = suffix.filter(\.isNumber)
                    source["variant_id"] = digits.isEmpty ? "color_default" : "color_\(Int(digits) ?? 0)"
                    if let prior = variants.first(where: { $0["variant_id"] as? String == source["variant_id"] as? String }) {
                        if sameSourceContents(prior, source) { ignored.append(path.lastPathComponent) } else { problems.append("Different maps claim the same color variant; choose one explicitly.") }
                    } else { variants.append(source) }
                } else if let prior = maps[role] {
                    if role == "normal", prior["suffix"] as? String == "nor_dx", suffix == "nor_gl" { ignored.append(prior["filename"] as? String ?? ""); maps[role] = source }
                    else if role == "normal", prior["suffix"] as? String == "nor_gl", suffix == "nor_dx" { ignored.append(path.lastPathComponent) }
                    else if sameSourceContents(prior, source) { ignored.append(path.lastPathComponent) }
                    else { problems.append("Multiple source maps for \(role).") }
                } else { maps[role] = source }
                group["maps"] = maps; group["input_variants"] = variants; group["problems"] = problems; group["ignored_files"] = ignored; group["warnings"] = issues; groups[key] = group
            }
        }
        for directory in directories {
            let aliases = byDirectory[directory]!.filter { standaloneMapRole($0) != nil }
            guard !aliases.isEmpty else { continue }
            var paths: [String: URL] = [:], ambiguous = false, convention = "opengl"
            let sourceNotice = " Sources: " + aliases.map(\.path).joined(separator: ", ")
            for path in aliases {
                let role = standaloneMapRole(path)!
                if paths[role] != nil { ambiguous = true; warnings.append("Multiple \(role) maps need explicit selection." + sourceNotice) }
                paths[role] = path
                if role == "normal", canonicalMapSuffix(path.deletingPathExtension().lastPathComponent) == "nor_dx" { convention = "directx" }
            }
            guard !ambiguous else { continue }
            do {
                var standalone: [String: Object] = [:]
                for (role, path) in paths {
                    var source = try scannedSummary(path.resolvingSymlinksInPath().standardizedFileURL, summaries: summaries)
                    source["path"] = path.resolvingSymlinksInPath().standardizedFileURL.path
                    source["suffix"] = canonicalMapSuffix(path.deletingPathExtension().lastPathComponent)
                    if role == "input" { source["variant_id"] = "color_default" }
                    standalone[role] = source
                }
                let size = try commonNativeDimensions(Array(standalone.values))
                // A generic role may fill a unique named set at the same native
                // dimensions. Complete standalone sets stay separate from
                // already complete named sets instead of replacing their maps.
                let candidates = groups.keys.filter { key in
                    guard key.hasPrefix(directory.path + "|"), let group = groups[key] else { return false }
                    let maps = group["maps"] as? [String: Object] ?? [:]
                    let variants = group["input_variants"] as? [Object] ?? []
                    guard standalone.keys.allSatisfy({ $0 == "input" ? variants.isEmpty : maps[$0] == nil }) else { return false }
                    return ((group["source_files"] as? [Object] ?? []) + Array(standalone.values)).allSatisfy {
                        abs(($0["width"] as? Int ?? 0) - size[0]) <= 8 && abs(($0["height"] as? Int ?? 0) - size[1]) <= 8
                    }
                }
                if candidates.count == 1 {
                    let key = candidates[0]; var group = groups[key]!
                    var maps = group["maps"] as? [String: Object] ?? [:]
                    var variants = group["input_variants"] as? [Object] ?? []
                    for (role, source) in standalone { if role == "input" { variants.append(source) } else { maps[role] = source } }
                    let all = (group["source_files"] as? [Object] ?? []) + Array(standalone.values)
                    group["maps"] = maps; group["input_variants"] = variants; group["source_files"] = all
                    group["common_pixel_dimensions"] = try commonNativeDimensions(all); groups[key] = group
                } else if standalone["input"] != nil, ["height", "roughness", "normal"].contains(where: { standalone[$0] != nil }) {
                    groups["manual:" + directory.path] = try sourceMaterial(directory.lastPathComponent, paths: paths,
                                                                          convention: convention, summaries: summaries)
                } else {
                    warnings.append("Map roles were recognized but require a paired diffuse and surface map assignment." + sourceNotice)
                }
            } catch is CancellationError { throw CancellationError() }
            catch { warnings.append("Could not pair \(directory.lastPathComponent): \(error.localizedDescription)" + sourceNotice) }
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
            let id = group["material_id"] as! String
            // A recognized group can be rejected for identity, ambiguity or
            // native-map validation. Bind that specific diagnostic to all of
            // its files so the final inventory pass does not call them
            // unrecognized or suggest that their diffuse pairing is absent.
            let paths = (group["source_files"] as? [Object] ?? Array(maps.values))
                .compactMap { $0["path"] as? String }.sorted()
            let sourceNotice = " Sources: " + paths.joined(separator: ", ")
            let targetRoles = ["height", "normal", "roughness"].filter { maps[$0] != nil }
            if maps["input"] == nil || targetRoles.isEmpty {
                let dimensions = group["common_pixel_dimensions"] as? [Int] ?? []
                let size = dimensions.count == 2 ? " (\(dimensions[0]) × \(dimensions[1]))" : ""
                let reason = maps["input"] == nil
                    ? "The \(targetRoles.joined(separator: ", ")) maps were recognized, but a matching diffuse map at these pixel dimensions is missing."
                    : "The diffuse map was recognized, but matching height, normal or roughness maps at these pixel dimensions are missing."
                warnings.append("Not imported: \(id)\(size). \(reason)" + sourceNotice)
                continue
            }
            if seen.contains(id) { group = disambiguatedMaterial(group, occupied: seen) }
            seen.insert(group["material_id"] as! String)
            group["automatic_discovery"] = true
            let problems = group["problems"] as? [String] ?? []
            if !problems.isEmpty { warnings.append("Skipped \(id): " + problems.joined(separator: "; ") + sourceNotice); continue }
            do { _ = try sourceRecord(group); materials.append(group); warnings += (group["warnings"] as? [String] ?? []).map { id + ": " + $0 } }
            catch { warnings.append("Skipped \(id): \(error.localizedDescription)" + sourceNotice) }
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
        let files = before.compactMap { ($0[0] as? String).map { URL(fileURLWithPath: $0) } }
        let quickPreview = options["--quick-preview"] == "true"
        let (materials, discoveredWarnings) = try discover(folder, dataset: root, records: records, inventoriedFiles: files, headersOnly: quickPreview)
        guard NSArray(array: before).isEqual(to: try inventory(folder, dataset: root)) else { throw StudioError("Original folder changed while scanning. Scan it again.") }
        var identities = Set(records.map { quickPreview ? previewSourceIdentity($0.sample) : sourceIdentity($0.sample) }), names = Set(records.compactMap { $0.sample["material_id"] as? String }), accepted: [Object] = [], warnings = discoveredWarnings, duplicates = 0
        for requested in materials {
            var material = requested
            var sample = try sourceRecord(material)
            let identity = quickPreview ? previewSourceIdentity(sample) : sourceIdentity(sample)
            if identities.contains(identity) { duplicates += 1; continue }
            if names.contains(sample["material_id"] as! String) { material = disambiguatedMaterial(material, occupied: names); sample = try sourceRecord(material) }
            identities.insert(identity); names.insert(sample["material_id"] as! String); accepted.append(material)
        }
        let recognized = Set(materials.flatMap { ($0["source_files"] as? [Object] ?? Array(($0["maps"] as? [String: Object] ?? [:]).values)).compactMap { $0["path"] as? String } })
        let ignoredFiles = before.filter { !recognized.contains($0[1] as? String ?? "") }
        let ignored = ignoredFiles.count
        for file in ignoredFiles {
            let path = file[0] as? String ?? ""
            let resolvedPath = file[1] as? String ?? path
            // Providers include ancillary PBR maps that this trainer does not
            // consume. Their presence is expected, not a broken source pair.
            if isAncillarySourceFilename(URL(fileURLWithPath: path).lastPathComponent) { continue }
            if !warnings.contains(where: { $0.contains(path) || $0.contains(resolvedPath) }) {
                warnings.append("Not imported: \(path). No recognized paired diffuse and surface map set includes this file. Assign its maps with Add Materials.")
            }
        }
        if materials.isEmpty { warnings.append("No paired PNG material maps were recognized. Add maps manually or choose another folder.") }
        let reviews = try optionalObject(root.appendingPathComponent(".material-size-reviews.json")), settings = try validation(index["validation"] as? Object)
        let combined = try records + accepted.map { Record(entry: [:], path: root.appendingPathComponent("dataset.json"), sample: try sourceRecord($0)) }
        var plans: Object = [:]
        let previewSizes = quickPreview ? [try checkedSize(options["--size"] ?? options["--training-size"] ?? String(index["training_size"] as? Int ?? 1024))] : sizes
        for size in previewSizes {
            let shared = try regionPlan(combined, size: size, reviews: reviews, target: "height", settings: settings)
            var targets: Object = [:]
            for target in ["height", "roughness", "normal"] { var plan = target == "height" ? shared : planForTarget(shared, records: combined, target: target); plan.removeValue(forKey: "assignments"); targets[target] = plan }
            plans[String(size)] = targets
        }
        let indexHash = try digest(root.appendingPathComponent("dataset.json")), reviewHash = try optionalDigest(root.appendingPathComponent(".material-size-reviews.json")) ?? hash(Data())
        var plan: Object = ["schema": "texture-studio-folder-import-v1", "dataset_path": root.path, "folder_path": folder.path, "index_sha256": indexHash, "review_sha256": reviewHash, "inventory": before, "materials": accepted]
        if quickPreview { plan["source_verification"] = "headers-only-v1" }
        let output = URL(fileURLWithPath: planPath); try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true); try atomic(output, data: jsonData(plan))
        return ["folder_path": folder.path, "plan_path": output.path, "plan_sha256": try digest(output), "index_sha256": indexHash, "source_set_count": materials.count, "added_material_count": accepted.count, "duplicate_material_count": duplicates, "ignored_file_count": ignored, "warnings": warnings, "plans": plans]
    }
    private static func importFolder(_ root: URL, index: Object, records: [Record], options: [String: String]) throws -> Object {
        guard let requested = options["--folder"] else { throw StudioError("Choose a folder to import.") }
        let folder = URL(fileURLWithPath: requested).resolvingSymlinksInPath().standardizedFileURL
        if options["--managed-import"] == "true" { return try importManagedFolder(root, index: index, records: records, folder: folder, options: options) }
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
            if plan["source_verification"] as? String == "headers-only-v1" {
                // The quick preview is never durable evidence. Recheck complete
                // PNG chunks/checksums and published provenance at import.
                let files = currentInventory.compactMap { ($0[0] as? String).map { URL(fileURLWithPath: $0) } }
                let verified = try discover(folder, dataset: root, records: records, inventoriedFiles: files).0
                let wanted = Dictionary(try pending.map { (sourceLocationIdentity(try sourceRecord($0)), $0) }, uniquingKeysWith: { _, new in new })
                materials = try verified.compactMap { material -> Object? in
                    guard let planned = wanted[sourceLocationIdentity(try sourceRecord(material))] else { return nil }
                    var material = material
                    for key in ["material_id", "name", "source_family_id", "source_set_id"] { material[key] = planned[key] }
                    guard previewSourceIdentity(try sourceRecord(material)) == previewSourceIdentity(try sourceRecord(planned)) else { return nil }
                    return material
                }
                guard materials.count == pending.count,
                      NSArray(array: currentInventory).isEqual(to: try inventory(folder, dataset: root)) else {
                    throw StudioError("Original maps failed verification or changed since preview. Scan the folder again.")
                }
            } else { materials = pending }
        } else { materials = try discover(folder, dataset: root, records: records).0 }
        return try register(root, index: index, records: records, materials: materials, options: options)
    }
    private static func previewSourceIdentity(_ sample: Object) -> String {
        let maps = sample["map_metadata"] as? [String: Object] ?? [:]
        func identity(_ details: Object) -> Object {
            let source = details["source"] as? Object ?? [:]
            return ["path": source["path"] ?? "", "stat": source["source_stat"] ?? [:],
                    "width": source["width"] ?? 0, "height": source["height"] ?? 0,
                    "suffix": source["suffix"] ?? "", "variant_id": details["variant_id"] ?? source["variant_id"] ?? ""]
        }
        let value: Object = ["material_id": sample["material_id"] ?? "", "name": sample["name"] ?? "",
                             "source_family_id": sample["source_family_id"] ?? "", "source_set_id": sample["source_set_id"] ?? "",
                             "dimensions": sample["source_pixel_dimensions"] ?? [],
                             "maps": maps.mapValues(identity), "colors": (sample["input_variants"] as? [Object] ?? []).map(identity)]
        return hash((try? jsonData(value)) ?? Data())
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
        let removedFamilies = removedSourceFamilies(index: index, records: records)
        let fresh = materials.filter {
            !removed.contains($0["material_id"] as? String ?? "") && !currentIDs.contains($0["material_id"] as? String ?? "")
                && !removedFamilies.excludes(family: $0["source_family_id"] as? String ?? "", subject: $0["source_directory"] as? String ?? "")
        }
        guard try sourceInventoryDigest(root) == inventoryBefore else { throw StudioError("Original source inventory changed while it was inspected. Reload the dataset.") }
        if !fresh.isEmpty { _ = try register(root, index: index, records: records, materials: fresh, options: [:]) }
        var refreshed = try object(root.appendingPathComponent("dataset.json"))
        refreshed["source_discovery"] = "registered-resolution-and-color-sets-v2"
        refreshed["source_inventory_sha256"] = inventoryBefore
        try commit(root, updates: [(root.appendingPathComponent("dataset.json"), refreshed)])
        return refreshed
    }

    private static let preparationSchema = "texture-studio-native-crops-v1"
    /// Synchronous entry point for callers which have already recovered and
    /// registered source maps. It preserves source bytes and publishes a cache
    /// identity bound to the requested spatial selection.
    static func prepareTrainingDataset(datasetURL: URL, size: Int, target: String, cropSeed: UInt64 = 0,
                                       material: String? = nil, automaticValidation: Bool = true,
                                       preparationPolicy: PreparationPolicy? = nil,
                                       onEvent: @Sendable (String) -> Void = { _ in }) throws -> URL {
        var options = ["--size": String(size), "--target": target, "--crop-seed": String(cropSeed)]
        if let material { options["--material"] = material }
        if automaticValidation { options["--automatic-validation"] = "true" }
        let result = try prepare(canonical(datasetURL), options: options, policy: preparationPolicy ?? .current, onEvent: onEvent)
        guard let path = result["dataset_path"] as? String else { throw StudioError("Native preparation did not publish a dataset path.") }
        return URL(fileURLWithPath: path)
    }
    private static func verifyManagedSources(_ root: URL, index: inout Object, records: [Record],
                                             size: Int, target: String, selectedID: String?,
                                             onEvent: @Sendable (String) -> Void) throws -> [Record] {
        guard index["managed_source_folders"] as? Bool == true else { return records }
        let selected = selectedID.flatMap { id in records.first { $0.sample["material_id"] as? String == id } }
        let eligible = records.filter { record in
            if selectedID != nil, selected == nil || subjectID(record.sample) != subjectID(selected!.sample) || familyID(record.sample) != familyID(selected!.sample) { return false }
            let needed = sources(record.sample, target: target)
            return !["excluded", "rejected"].contains(record.sample["status"] as? String ?? "") &&
                (record.sample["map_metadata"] as? [String: Object])?[target] != nil &&
                (record.sample["available_targets"] as? [String] ?? [target]).contains(target) &&
                needed.allSatisfy { min($0["width"] as? Int ?? 0, $0["height"] as? Int ?? 0) >= size }
        }
        var needed: [String: Object] = [:]
        for record in eligible { for source in sources(record.sample, target: target) { if let path = source["path"] as? String {
            let unchanged = (try? fileState(URL(fileURLWithPath: path))).map { state in
                (source["source_stat"] as? Object).map { NSDictionary(dictionary: $0).isEqual(to: state) } ?? false
            } ?? false
            if (source["file_sha256"] as? String)?.count != 64 || !unchanged { needed[path] = source }
        } } }
        guard !needed.isEmpty else { return records }
        let paths = needed.keys.sorted()
        func event(_ name: String, completed: Int, path: String? = nil, issue: String? = nil) {
            var value: Object = ["event": name, "completed": completed, "total": paths.count, "training_size": size]
            value["source_path"] = path; value["issue"] = issue
            if let text = try? jsonString(value) { onEvent(text + "\n") }
        }
        event("source_verification_started", completed: 0)
        var verified: [String: Object] = [:], issues: [String: String] = [:]
        for (offset, path) in paths.enumerated() {
            try Task.checkCancellation()
            do {
                let before = needed[path]!, url = URL(fileURLWithPath: path), state = try fileState(url)
                let unchanged = (before["source_stat"] as? Object).map { recorded in
                    if NSDictionary(dictionary: recorded).isEqual(to: state) { return true }
                    guard NativeSourceFileTags.read(url) != nil else { return false }
                    var recorded = recorded, state = state
                    recorded["ctime_ns"] = nil; state["ctime_ns"] = nil
                    return NSDictionary(dictionary: recorded).isEqual(to: state)
                } ?? false
                let hadHash = (before["file_sha256"] as? String)?.count == 64
                guard hadHash || unchanged else { throw StudioError("Original changed after it was added. Reassign this file before preparing training crops.") }
                var actual = try sourceSummary(url, cached: before)
                for key in ["width", "height", "sample_bits", "channels"] {
                    guard (actual[key] as? Int) == (before[key] as? Int) else { throw StudioError("Original dimensions or precision changed after registration.") }
                }
                if hadHash, actual["file_sha256"] as? String != before["file_sha256"] as? String { throw StudioError("Original bytes changed after verification. Restore or reassign this file.") }
                try? NativeSourceFileTags.update(url, role: inferredSourceRole(url), verifiedSHA256: actual["file_sha256"] as? String,
                    verifiedMetadata: jsonString(compactMetadata(actual) as! Object))
                actual["source_stat"] = try fileState(url)
                var bound = before; bound.merge(actual) { _, new in new }; bound["path"] = path; verified[path] = bound
                event("source_verification_progress", completed: offset + 1, path: path)
            } catch is CancellationError { throw CancellationError() }
            catch { issues[path] = error.localizedDescription; event("source_verification_progress", completed: offset + 1, path: path, issue: error.localizedDescription) }
        }
        var updated = records, writes: [(URL, Object)] = [], reviews = try optionalObject(root.appendingPathComponent(".material-size-reviews.json"))
        var changedReviews = false
        for offset in updated.indices {
            let old = updated[offset].sample, oldBinding = sourceBinding(old)
            guard sources(old).contains(where: { source in
                let path = source["path"] as? String ?? ""
                return verified[path] != nil || issues[path] != nil
            }) else { continue }
            var sample = old, metadata = sample["map_metadata"] as? [String: Object] ?? [:]
            var variants = sample["input_variants"] as? [Object] ?? []
            func refresh(_ details: Object, role: String) -> Object {
                var details = details
                if let source = details["source"] as? Object, let path = source["path"] as? String, let bound = verified[path] {
                    details["source"] = bound; details["sample_sha256"] = bound["file_sha256"]
                    details["encoding"] = encoding(role, source: bound)
                }
                return details
            }
            for role in metadata.keys { metadata[role] = refresh(metadata[role]!, role: role) }
            for variant in variants.indices { variants[variant] = refresh(variants[variant], role: "input") }
            sample["map_metadata"] = metadata; sample["input_variants"] = variants
            var failures = sample["source_verification_issues"] as? [String: String] ?? [:]
            for source in sources(sample) { if let path = source["path"] as? String {
                if verified[path] != nil { failures.removeValue(forKey: path) }
                if let issue = issues[path] { failures[path] = issue }
            } }
            sample["source_verification_issues"] = failures
            let complete = failures.isEmpty && sources(sample).allSatisfy { ($0["file_sha256"] as? String)?.count == 64 }
            sample["source_precision_verified"] = complete; sample["crop_values_verified"] = complete
            sample["source_verification"] = complete ? "verified" : "header_inspected"
            guard !NSDictionary(dictionary: old).isEqual(to: sample) else { continue }
            let newBinding = sourceBinding(sample)
            if oldBinding != newBinding { for key in reviews.keys {
                guard var review = reviews[key] as? Object, review["source_binding_sha256"] as? String == oldBinding else { continue }
                review["source_binding_sha256"] = newBinding; reviews[key] = review; changedReviews = true
            } }
            updated[offset].sample = sample; writes.append((updated[offset].path, sample))
        }
        if var catalog = index["source_inventory"] as? [Object] { for offset in catalog.indices {
            let path = catalog[offset]["path"] as? String ?? ""
            if verified[path] != nil { catalog[offset]["source_verification"] = "verified"; catalog[offset]["state"] = "paired"; catalog[offset].removeValue(forKey: "issue") }
            if let issue = issues[path] { catalog[offset]["state"] = FileManager.default.fileExists(atPath: path) ? "header_error" : "missing"; catalog[offset]["issue"] = issue }
        }; index["source_inventory"] = catalog }
        if changedReviews { writes.append((root.appendingPathComponent(".material-size-reviews.json"), reviews)) }
        index["updated_utc"] = timestamp(); writes.append((root.appendingPathComponent("dataset.json"), index))
        try commit(root, updates: writes)
        event("source_verification_completed", completed: paths.count)
        return updated
    }
    private static func prepare(_ supplied: URL, options: [String: String], policy: PreparationPolicy, onEvent: @Sendable (String) -> Void) throws -> Object {
        let size = try checkedSize(options["--size"] ?? ""), target = options["--target"] ?? "height"
        guard let cropSeed = UInt64(options["--crop-seed"] ?? "0") else { throw StudioError("Crop selection seed must be a nonnegative whole number.") }
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
            var sourceIndex = try object(original.appendingPathComponent("dataset.json"))
            let current = try object(supplied.appendingPathComponent("dataset.json"))
            guard NSDictionary(dictionary: current["native_size_preparation"] as? Object ?? [:]).isEqual(to: lineage ?? [:]) else { throw StudioError("Native source lineage changed; reopen the dataset.") }
            let reviewPath = original.appendingPathComponent(".material-size-reviews.json")
            if let expected = options["--expected-review-sha256"], try optionalDigest(reviewPath) ?? hash(Data()) != expected { throw StudioError("Dataset reviews changed; reload before preparing native crops.") }
            let selectedID = options["--material"]
            let allRecords = try verifyManagedSources(original, index: &sourceIndex,
                records: readRecords(original, index: sourceIndex), size: size, target: target,
                selectedID: selectedID, onEvent: onEvent)
            let selectedSource = selectedID.flatMap { id in allRecords.first { $0.sample["material_id"] as? String == id } }
            let selected = allRecords.filter { record in
                guard selectedID != nil else { return true }
                guard let selectedSource else { return false }
                return subjectID(record.sample) == subjectID(selectedSource.sample) && familyID(record.sample) == familyID(selectedSource.sample)
            }
            guard !selected.isEmpty else { throw StudioError("Dataset has no selected original material maps.") }
            let records = selected.filter {
                let sample = $0.sample, selectedSources = sources(sample, target: target)
                return !["excluded", "rejected"].contains(sample["status"] as? String ?? "") &&
                    (sample["map_metadata"] as? [String: Object])?[target] != nil &&
                    (sample["available_targets"] as? [String] ?? [target]).contains(target) &&
                    !selectedSources.isEmpty && selectedSources.allSatisfy {
                        min($0["width"] as? Int ?? 0, $0["height"] as? Int ?? 0) >= size &&
                            ($0["file_sha256"] as? String)?.count == 64 &&
                            (sample["source_verification_issues"] as? [String: String])?[$0["path"] as? String ?? ""] == nil
                    }
            }
            guard !records.isEmpty else { throw StudioError("No original source set supplies the selected native training size.") }
            var reviews = try optionalObject(reviewPath)
            if supplied != original {
                let cached = try readRecords(supplied, index: current)
                for record in cached {
                    // Recovery can append a target without changing reviewed
                    // pixels. Its rebound original review is authoritative.
                    guard let sourceRecord = allRecords.first(where: { $0.sample["material_id"] as? String == record.sample["material_id"] as? String }),
                          sourceBinding(sourceRecord.sample) == sourceBinding(record.sample) else { continue }
                    let key = "\(current["crop_size"] as? Int ?? size):\(record.sample["sample_id"] as? String ?? "")", review = currentReview(reviews, key: key, sample: record.sample)
                    if let snapshot = record.sample["source_review_snapshot"] as? Object, NSDictionary(dictionary: snapshot).isEqual(to: review) || review.isEmpty { reviews[key] = savedReview(record.sample) }
                }
            }
            var settings = try validation(sourceIndex["validation"] as? Object)
            if options["--automatic-validation"] != "true" { settings["enabled"] = false }
            // Folder percentages and geometry remain common to all targets;
            // only target-eligible assignments become preparation jobs.
            let usable = Set(records.compactMap { $0.sample["material_id"] as? String })
            let plan = try regionPlan(selected.filter { usable.contains($0.sample["material_id"] as? String ?? "") }, size: size, reviews: reviews, target: target, settings: settings, cropSeed: cropSeed)
            let assignments = (plan["assignments"] as? [String: Object] ?? [:]).filter { $0.value["eligible"] as? Bool == true }
            guard !assignments.isEmpty else { throw StudioError("No included crops supply the selected training target.") }
            var metadataProof: [String: String] = [:]
            for record in allRecords { metadataProof[String(record.path.path.dropFirst(original.path.count + 1))] = try digest(record.path) }
            let proof: Object = ["dataset_path": original.path, "index_sha256": try digest(original.appendingPathComponent("dataset.json")), "sample_metadata_sha256": metadataProof]
            let selection: Object = ["policy": cropSelectionPolicy, "seed": String(cropSeed)]
            let signature: Object = ["schema": preparationSchema, "map_scope": "selected-target-v2", "size": size, "target": target, "snapshot": proof, "reviews": reviews, "validation": settings, "material": selectedID as Any? ?? NSNull(), "crop_selection": selection]
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
                    let sourceDimensions = [source["width"] as? Int ?? 0, source["height"] as? Int ?? 0]
                    let regionCount = assignments.values.filter {
                        $0["material_id"] as? String == record.sample["material_id"] as? String && $0["crop_rectangle"] as? [Int] != [0, 0] + sourceDimensions
                    }.count
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
            let binding: Object = ["schema": preparationSchema, "ephemeral": true, "source_dataset_path": original.path, "source_snapshot": proof, "target": target, "map_scope": "selected-target-v2", "target_resized": false, "target_cropped": cropped, "estimated_staging_bytes": bytesRequired, "preparation_workers": workerCount, "estimated_worker_peak_bytes": peakBytes, "preparation_memory_budget_bytes": policy.memoryBudgetBytes, "automatic_validation": automatic, "crop_selection": selection]
            let derived: Object = ["schema_version": 2, "generator": "ipde-material-dataset-v2", "crop_size": size, "source_tier_count": sourceIndex["source_tier_count"] ?? 1, "training_pixel_dimensions": [size, size], "split_strategy": "subject-extra-crops-v2", "validation": settings, "validation_scope": "known_subject_diagnostic", "original_sources_required_for_training": true, "source_images_modified": false, "native_size_preparation": binding, "automatic_validation": automatic, "samples": prepared.map { ["sample_id": $0["sample_id"]!, "material_id": $0["material_id"]!, "status": $0["status"]!, "split": $0["split"]!, "asset_family_id": familyID($0), "source_set_id": $0["source_set_id"] ?? NSNull(), "path": "samples/" + ($0["sample_id"] as! String)] }]
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
        let size = job.size
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
            sample["crop_kind"] = assignment["crop_kind"] ?? region
            sample["source_square_rectangle_top_left_xywh"] = assignment["source_square_rectangle_top_left_xywh"]
            sample["crop_selection_policy"] = assignment["crop_selection_policy"] ?? cropSelectionPolicy
            sample["crop_selection_seed"] = assignment["crop_selection_seed"] ?? "0"
            sample["maps"] = [String: String](); sample["map_metadata"] = [String: Object](); sample["input_variants"] = [Object]()
            sample["available_targets"] = [job.target]
            sample.removeValue(forKey: "source_files")
            sample.removeValue(forKey: "source_normalized_rectangle_xywh")
            sample["split"] = assignment["split"]; sample["split_assignment"] = assignment["split_assignment"] ?? "automatic"; sample["source_region_role"] = assignment["split"]; sample["source_region_id"] = region
            sample["source_review_snapshot"] = review; sample["source_binding_sha256"] = binding
            sample["split_strategy"] = "subject-extra-crops-v2"; sample["validation_scope"] = "known_subject_diagnostic"; sample["source_precision_verified"] = true; sample["crop_values_verified"] = true
            let changed = maps.contains { _, details, _ in
                let source = details["source"] as? Object ?? [:]
                return rectangle != [0, 0, source["width"] as? Int ?? 0, source["height"] as? Int ?? 0]
            }
            cropped = cropped || changed
            sample["native_size_preparation"] = ["schema": preparationSchema, "source_dataset_path": job.original.path, "target_resized": false, "target_cropped": changed, "crop_selection_policy": sample["crop_selection_policy"]!, "crop_selection_seed": sample["crop_selection_seed"]!]
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
            let sourceDimensions = [source["width"] as? Int ?? 0, source["height"] as? Int ?? 0]
            let cropOffsets = active.indices.filter { active[$0].2 != [0, 0] + sourceDimensions }
            let crops = cropOffsets.isEmpty ? [] : try NativePNG.crops(sourceURL, rectangles: cropOffsets.map { active[$0].2 }, flipGreen: flip)
            let selectedCrops = Dictionary(uniqueKeysWithValues: zip(cropOffsets, crops))
            guard NSDictionary(dictionary: before).isEqual(to: try fileState(sourceURL)) else { throw StudioError("Original source changed during preparation.") }
            for offset in active.indices {
                try Task.checkCancellation()
                let rectangle = active[offset].2, output = active[offset].1.appendingPathComponent(filename)
                let isCrop = rectangle != [0, 0] + sourceDimensions
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
        guard index["crop_size"] as? Int == size, let binding = index["native_size_preparation"] as? Object, binding["schema"] as? String == preparationSchema,
              binding["target_resized"] as? Bool == false,
              let snapshot = binding["source_snapshot"] as? Object, NSDictionary(dictionary: snapshot).isEqual(to: proof) else { return false }
        let verifyFile = checksumFile ?? digest
        var verified: [String: (String, Object)] = [:]
        for record in try readRecords(root, index: index) {
            try Task.checkCancellation()
            guard record.sample["sample_pixel_dimensions"] as? [Int] == [size, size] else { return false }
            var changed = false
            func validateMap(_ map: inout Object) throws -> Bool {
                try Task.checkCancellation()
                guard trainingMapIsExact(map, prepared: true), let name = map["filename"] as? String, let expected = map["sample_sha256"] as? String else { return false }
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
        if let folder = index["source_folder_dataset_path"] as? String {
            preparation["source_dataset_path"] = folder
            preparation["source_index_sha256"] = proof["index_sha256"] ?? ""
        }
        for key in ["preparation_workers", "estimated_worker_peak_bytes", "preparation_memory_budget_bytes"] { preparation[key] = binding[key] }
        result["preparation"] = preparation
        return result
    }
    private static func savedReview(_ sample: Object) -> Object {
        var result: Object = [:]
        for key in ["status", "split", "split_assignment", "review_status", "curation_note"] { result[key] = sample[key] ?? NSNull() }
        result["source_binding_sha256"] = sourceBinding(sample); return result
    }
    static func cleanup(_ supplied: URL, beforeDirectoryRemoval: (URL) throws -> Void = { _ in }) throws -> [String: Any] {
        let receiptURL = supplied.appendingPathComponent(".native-crop-cleanup.json")
        // Prefer the durable inventory even when dataset.json is still present:
        // a previous attempt may already have removed some sample manifests.
        if FileManager.default.fileExists(atPath: receiptURL.path) {
            let receipt = try object(receiptURL)
            guard receipt["schema"] as? String == "texture-studio-native-crop-cleanup-v2",
                  receipt["prepared_dataset_path"] as? String == supplied.path,
                  let path = receipt["source_dataset_path"] as? String else {
                throw StudioError("Temporary training cleanup ownership changed.")
            }
            let original = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
            try validateCleanupLocation(supplied, original: original)
            return try locked(original) { try locked(supplied) {
                let currentReceipt = try object(receiptURL)
                let inventory = try validateCleanupReceipt(currentReceipt, supplied: supplied, original: original)
                let receiptBytes = try jsonData(currentReceipt)
                do {
                    try removePreparedTrainingDirectory(supplied,
                        ownedFiles: Set(inventory.files.keys).union([receiptURL.lastPathComponent]), ownedDirectories: inventory.directories,
                        expectedFileHashes: inventory.files, beforeDirectoryRemoval: beforeDirectoryRemoval)
                } catch {
                    if FileManager.default.fileExists(atPath: supplied.path),
                       (try? supplied.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true {
                        try? atomic(receiptURL, data: receiptBytes)
                    }
                    throw error
                }
                try sync(supplied.deletingLastPathComponent())
                return ["dataset_path": supplied.path, "source_dataset_path": original.path, "removed": true,
                        "source_images_modified": false, "reviews_preserved": true]
            } }
        }
        guard FileManager.default.fileExists(atPath: supplied.appendingPathComponent("dataset.json").path) else { return ["dataset_path": supplied.path, "removed": false] }
        let index = try object(supplied.appendingPathComponent("dataset.json")), binding = index["native_size_preparation"] as? Object ?? [:]
        guard binding["schema"] as? String == preparationSchema, binding["ephemeral"] as? Bool == true, let source = binding["source_dataset_path"] as? String else { return ["dataset_path": supplied.path, "removed": false] }
        let original = URL(fileURLWithPath: source).resolvingSymlinksInPath().standardizedFileURL
        return try locked(original) { try locked(supplied) {
            try cleanupLocked(supplied, original: original, beforeDirectoryRemoval: beforeDirectoryRemoval)
            return ["dataset_path": supplied.path, "source_dataset_path": original.path, "removed": true, "source_images_modified": false, "reviews_preserved": true]
        } }
    }
    private static func cleanupLocked(_ supplied: URL, original: URL, beforeDirectoryRemoval: (URL) throws -> Void = { _ in }) throws {
        let staging = original.appendingPathComponent(".training-data")
        try validateCleanupLocation(supplied, original: original)
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
        // Generated PNGs, manifests and recognized Finder directory metadata
        // may be removed with this owned cache. Unrelated files still prevent
        // cleanup rather than disappearing with the training data.
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
                guard try (ownedFiles.contains(relative) || isFinderDirectoryMetadata(checkedFile)) else {
                    throw StudioError("Temporary training data contains new unrelated files (\(relative)); originals are kept.")
                }
            } else if values.isDirectory == true {
                guard ownedDirectories.contains(relative) else { throw StudioError("Temporary training data contains an unrelated folder; originals are kept.") }
            } else { throw StudioError("Temporary training data contains an unrelated special file; originals are kept.") }
        }
        guard !failed else { throw StudioError("Temporary training data could not be fully inspected.") }
        try Task.checkCancellation(); try commit(original, updates: [(reviewPath, reviews)])
        if (try? object(original.appendingPathComponent("dataset.json"))["source_folder_dataset_path"] as? String) != nil {
            try atomic(original.appendingPathComponent("teaching.json"), data: jsonData(sourceFolderChoices(original)))
        }
        // Store the complete validated inventory before removing any generated
        // files. Recovery no longer depends on sample manifests surviving a
        // failed child-directory removal.
        let receiptURL = supplied.appendingPathComponent(".native-crop-cleanup.json")
        var fileHashes: [String: String] = [:]
        for relative in ownedFiles.sorted() {
            let file = try contained(supplied, relative)
            var state = stat()
            if lstat(file.path, &state) == 0 {
                guard state.st_mode & S_IFMT == S_IFREG else { throw StudioError("Temporary training file type changed during cleanup.") }
                fileHashes[relative] = try digest(file)
            } else if errno != ENOENT { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        }
        let receiptObject: Object = ["schema": "texture-studio-native-crop-cleanup-v2",
            "source_dataset_path": original.path, "prepared_dataset_path": supplied.path,
            "prepared_index_sha256": fileHashes["dataset.json"]!, "owned_files_sha256": fileHashes,
            "owned_directories": ownedDirectories.sorted()]
        let receipt = try jsonData(receiptObject)
        try atomic(receiptURL, data: receipt)
        let inventory = try validateCleanupReceipt(receiptObject, supplied: supplied, original: original)
        do {
            try removePreparedTrainingDirectory(supplied, ownedFiles: Set(inventory.files.keys).union([receiptURL.lastPathComponent]),
                ownedDirectories: inventory.directories, expectedFileHashes: inventory.files, beforeDirectoryRemoval: beforeDirectoryRemoval)
        } catch {
            if FileManager.default.fileExists(atPath: supplied.path),
               (try? supplied.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true {
                try? atomic(receiptURL, data: receipt)
            }
            throw error
        }
        try sync(staging)
    }

    private static func validateCleanupReceipt(_ receipt: Object, supplied: URL, original: URL) throws -> (files: [String: String], directories: Set<String>) {
        try validateCleanupLocation(supplied, original: original)
        guard receipt["schema"] as? String == "texture-studio-native-crop-cleanup-v2",
              receipt["prepared_dataset_path"] as? String == supplied.path,
              receipt["source_dataset_path"] as? String == original.path,
              let hashes = receipt["owned_files_sha256"] as? [String: String],
              let indexHash = hashes["dataset.json"], indexHash == receipt["prepared_index_sha256"] as? String,
              let directories = receipt["owned_directories"] as? [String], Set(directories).count == directories.count else {
            throw StudioError("Temporary training cleanup ownership changed.")
        }
        var expectedDirectories = Set<String>()
        for (relative, hash) in hashes {
            guard !relative.isEmpty, !relative.hasPrefix("/"),
                  !relative.split(separator: "/").contains(where: { $0 == "." || $0 == ".." }),
                  hash.count == 64, hash.allSatisfy({ $0.isHexDigit }),
                  relative != ".native-crop-cleanup.json" else { throw StudioError("Invalid temporary training cleanup inventory.") }
            _ = try contained(supplied, relative)
            var components = relative.split(separator: "/")
            while components.count > 1 { components.removeLast(); expectedDirectories.insert(components.joined(separator: "/")) }
        }
        guard expectedDirectories == Set(directories) else { throw StudioError("Temporary training cleanup directory inventory changed.") }
        var failed = false
        guard let entries = FileManager.default.enumerator(at: supplied,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
            errorHandler: { _, _ in failed = true; return false }) else { throw StudioError("Cannot inspect temporary training cleanup inventory.") }
        for case let file as URL in entries {
            try Task.checkCancellation()
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey])
            guard values.isSymbolicLink != true else { throw StudioError("Temporary training data contains an unexpected symbolic link.") }
            // Foundation enumeration may spell /var as /private/var. Use the
            // same canonical spelling as the supplied dataset before deriving
            // its relative path; otherwise a valid receipt looks unrelated.
            let checkedFile = file.resolvingSymlinksInPath().standardizedFileURL
            guard isInside(checkedFile, supplied) else { throw StudioError("Temporary training data contains an unexpected outside path.") }
            let relative = String(checkedFile.path.dropFirst(supplied.path.count + 1))
            if values.isDirectory == true {
                guard expectedDirectories.contains(relative) else { throw StudioError("Temporary training data contains an unrelated folder; originals are kept.") }
            } else if values.isRegularFile == true {
                if relative == ".native-crop-cleanup.json" { continue }
                if let expected = hashes[relative] {
                    guard try digest(checkedFile) == expected else { throw StudioError("Temporary training contents changed during cleanup; originals are kept.") }
                } else if try !isFinderDirectoryMetadata(checkedFile) {
                    throw StudioError("Temporary training data contains new unrelated files (\(relative)); originals are kept.")
                }
            } else { throw StudioError("Temporary training data contains an unrelated special file; originals are kept.") }
        }
        guard !failed else { throw StudioError("Temporary training data could not be fully inspected.") }
        return (hashes, expectedDirectories)
    }

    private static func validateCleanupLocation(_ supplied: URL, original: URL) throws {
        let staging = original.appendingPathComponent(".training-data")
        guard supplied.deletingLastPathComponent() == staging,
              (try? supplied.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true,
              (try? staging.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
            throw StudioError("Temporary cleanup scope differs from its owned source location.")
        }
    }

    /// Remove validated generated files, then each directory explicitly. Finder
    /// can recreate .DS_Store during recursive FileManager removal; bounded final
    /// retries accept only its regular-file signature, never arbitrary new data.
    static func removePreparedTrainingDirectory(_ supplied: URL, ownedFiles: Set<String>, ownedDirectories: Set<String>,
                                                expectedFileHashes: [String: String] = [:],
                                                beforeDirectoryRemoval: (URL) throws -> Void = { _ in },
                                                beforeFinalRemoval: () throws -> Void = {}) throws {
        let finalFiles: Set<String> = ["dataset.json", ".material-workbench.lock", ".native-crop-cleanup.json"]
        for relative in ownedFiles.subtracting(finalFiles).sorted() {
            try Task.checkCancellation()
            let file = try contained(supplied, relative)
            var state = stat()
            if lstat(file.path, &state) == 0 {
                guard state.st_mode & S_IFMT == S_IFREG else { throw StudioError("Temporary training file type changed during cleanup.") }
                if let expected = expectedFileHashes[relative] {
                    guard try digest(file) == expected else { throw StudioError("Temporary training contents changed during cleanup; originals are kept.") }
                }
                try FileManager.default.removeItem(at: file)
            } else if errno != ENOENT { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        }
        for relative in ownedDirectories.sorted(by: { $0.count > $1.count }) {
            let directory = try contained(supplied, relative)
            try removeEmptyPreparedDirectory(directory, finalFiles: [], beforeFirstRemoval: { try beforeDirectoryRemoval(directory) })
        }
        try removeEmptyPreparedDirectory(supplied, finalFiles: finalFiles.intersection(ownedFiles),
            expectedFileHashes: expectedFileHashes, beforeFirstRemoval: {
                try beforeDirectoryRemoval(supplied)
                try beforeFinalRemoval()
            })
    }

    private static func removeEmptyPreparedDirectory(_ directory: URL, finalFiles: Set<String>,
                                                    expectedFileHashes: [String: String] = [:],
                                                    beforeFirstRemoval: () throws -> Void = {}) throws {
        for attempt in 0..<4 {
            try Task.checkCancellation()
            var directoryState = stat()
            if lstat(directory.path, &directoryState) != 0 {
                if errno == ENOENT { return }
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            guard directoryState.st_mode & S_IFMT == S_IFDIR else {
                throw StudioError("Temporary training directory became a symbolic link during cleanup.")
            }
            for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
                var state = stat()
                guard lstat(file.path, &state) == 0, state.st_mode & S_IFMT == S_IFREG,
                      try (finalFiles.contains(file.lastPathComponent) || isFinderDirectoryMetadata(file)) else {
                    throw StudioError("Temporary training data contains new unrelated files (\(file.lastPathComponent)); originals are kept.")
                }
                if let expected = expectedFileHashes[file.lastPathComponent] {
                    guard try digest(file) == expected else { throw StudioError("Temporary training contents changed during cleanup; originals are kept.") }
                }
                try FileManager.default.removeItem(at: file)
            }
            if attempt == 0 { try beforeFirstRemoval() }
            if Darwin.rmdir(directory.path) == 0 { return }
            let code = errno
            if code == ENOENT { return }
            guard code == ENOTEMPTY || code == EEXIST else { throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO) }
        }
        throw StudioError("Finder is still updating temporary training metadata. The saved model is intact; retry cleanup after closing that folder.")
    }

    /// Finder may add this cache while a run is in progress. Match its exact
    /// name and storage signature; arbitrary hidden files remain protected.
    private static func isFinderDirectoryMetadata(_ file: URL) throws -> Bool {
        guard file.lastPathComponent == ".DS_Store" else { return false }
        let descriptor = Darwin.open(file.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(descriptor) }
        var state = stat()
        guard fstat(descriptor, &state) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        guard state.st_mode & S_IFMT == S_IFREG else { return false }
        var signature = [UInt8](repeating: 0, count: 8)
        let count = signature.withUnsafeMutableBytes { Darwin.read(descriptor, $0.baseAddress!, $0.count) }
        guard count >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        return count == signature.count && signature == [0, 0, 0, 1, 0x42, 0x75, 0x64, 0x31]
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
        var entries: [[Any]] = []
        for folder in try FileManager.default.contentsOfDirectory(at: sourceRoot, includingPropertiesForKeys: [.isDirectoryKey]).sorted(by: { $0.path < $1.path }) {
            try Task.checkCancellation()
            guard (try folder.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true else { continue }
            for file in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isRegularFileKey]).sorted(by: { $0.path < $1.path }) {
                try Task.checkCancellation()
                let name = file.lastPathComponent
                guard (try file.resourceValues(forKeys: [.isRegularFileKey])).isRegularFile == true else { continue }
                let recognized = (name.hasPrefix("material-source") && file.pathExtension.lowercased() == "json") || (parsedProvider(name) != nil && isSourceImageCandidate(file))
                guard recognized else { continue }
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
              let max = settings["max_crops"] as? NSNumber, CFGetTypeID(max) != CFBooleanGetTypeID(), let maxCount = settings["max_crops"] as? Int, maxCount >= 0,
              let quick = settings["quick_count"] as? NSNumber, CFGetTypeID(quick) != CFBooleanGetTypeID(), let quickCount = settings["quick_count"] as? Int, quickCount >= 0,
              let folders = settings["folders"] as? Object, folders.values.allSatisfy({ ($0 as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? false }) else { throw StudioError("Validation needs an enabled flag, a percentage from 0 to 100 and nonnegative crop counts.") }
        settings["enabled"] = enabled.boolValue; settings["percent"] = percent.doubleValue; settings["max_crops"] = maxCount; settings["quick_count"] = quickCount
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
        let size = try checkedSize(options["--training-size"] ?? "2048")
        guard !["dataset.json", "sources", ".training-data"].contains(root.lastPathComponent), (try? root.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { throw StudioError("Choose a new dataset folder.") }
        try FileManager.default.createDirectory(at: root.deletingLastPathComponent(), withIntermediateDirectories: true)
        var created = false
        if !FileManager.default.fileExists(atPath: root.path) { try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); created = true }
        do {
            try locked(root) {
                guard try FileManager.default.contentsOfDirectory(atPath: root.path).allSatisfy({ $0 == ".material-workbench.lock" }) else { throw StudioError("That folder already contains files. Choose a new or empty dataset folder.") }
                let now = timestamp()
                let index: Object = ["schema_version": 2, "generator": "ipde-material-dataset-v2", "dataset_id": UUID().uuidString.lowercased(), "name": name, "description": description, "created_utc": now, "updated_utc": now, "dataset_management": ["schema": management, "owns_directory": true], "storage_policy": "original-source-references-v1", "source_images_modified": false, "original_sources_required_for_training": true, "split_strategy": "whole-material-v1", "validation_scope": "known_subject_diagnostic", "validation": try editedValidation(validation(nil), options: options), "training_size": size, "source_tier_count": try checkedSourceTierCount(options["--source-tier-count"] ?? "1"), "samples": [Object]()]
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
    private static func checkedSourceTierCount(_ value: String) throws -> Int { guard let count = Int(value), (1...3).contains(count) else { throw StudioError("Choose 1, 2 or 3 source tiers to fetch.") }; return count }
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
                guard directory.deletingLastPathComponent().path == root.path, directory.lastPathComponent.hasPrefix(".material-workbench-transaction-"), staged.pathExtension == "json" else { throw StudioError("Invalid dataset recovery staging path.") }
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
