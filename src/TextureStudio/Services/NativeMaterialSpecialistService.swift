import CryptoKit
import Darwin
import Foundation
import ImageIO

/// Classification selects original material references. Specialist refreshes
/// never render maps or add classifier-only photos to reconstruction datasets.
enum NativeMaterialSpecialistService {
    static let schema = "texture-studio-material-specialists-v1"
    static let registryFilename = "specialists.json"
    static let commands: Set<String> = ["specialists", "refresh-specialists", "add-specialist", "record-specialist-model", "approve-specialist-model"]
    static var defaultWorkspace: URL {
        let preferences = UserDefaults(suiteName: "org.ipde.material-tools")!
        if let configured = preferences.string(forKey: "specialistWorkspace"), !configured.isEmpty {
            return URL(fileURLWithPath: configured).standardizedFileURL
        }
        let managed = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Texture Studio/Material Workspace", isDirectory: true)
        if let configured = preferences.string(forKey: "workspace"), !configured.isEmpty {
            let candidate = canonical(URL(fileURLWithPath: configured))
            var ancestor = candidate
            while ancestor != ancestor.deletingLastPathComponent() {
                if FileManager.default.fileExists(atPath: ancestor.appendingPathComponent(".git").path) {
                    return managed.appendingPathComponent("Specialist Models")
                }
                ancestor = ancestor.deletingLastPathComponent()
            }
            return candidate.appendingPathComponent("Specialist Models")
        }
        return managed.appendingPathComponent("Specialist Models")
    }

    struct Registry: Codable, Sendable {
        let schema: String
        let workspacePath: String
        var sourceDatasetPath: String
        var classifierDatasetPath: String?
        var sourceFolderPath: String? = nil
        var classifierPath: String
        var classifierSha256: String
        var classifierLabels: [String]
        var updatedAt: String
        var specialists: [Specialist]
        var classifierOnlyAssets: [ClassifierOnlyAsset]
        var classifierOnlyCount: Int
        var materialCount: Int
        var familyCount: Int
        var warnings: [String]
    }
    struct Specialist: Codable, Sendable, Identifiable {
        let id: String
        var name: String
        let classLabel: String
        let subtype: String?
        let isCustom: Bool
        let datasetPath: String
        var materialIds: [String]
        var familyIds: [String]
        var assignments: [Assignment]
        var targets: [TargetStatus]
        var models: [ModelRecord]
        var automaticallyManagedMaterialIds: [String]
        var automaticallyRemovedMaterialIds: [String]
    }
    struct Assignment: Codable, Sendable {
        let familyId: String
        let materialIds: [String]
        let inputPath: String
        let inputSha256: String
        let predictedLabel: String
        let classLabel: String
        let confidence: Double
        let probabilities: [String: Double]
        let classifierSha256: String
        let source: String
        let subtype: String?
        let semanticTerms: [String]
    }
    struct TargetStatus: Codable, Sendable {
        let target: String
        let materialCount: Int
        let status: String
        let membershipSha256: String
    }
    struct ModelRecord: Codable, Sendable, Identifiable {
        let id: String
        let target: String
        let modelFamily: String
        let checkpointPath: String
        let checkpointSha256: String
        let trainingSize: Int
        let completedUpdates: Int
        let completedAt: String
        let membershipSha256: String
        var approved: Bool
        var approvedAt: String?
    }
    struct ClassifierOnlyAsset: Codable, Sendable, Identifiable {
        let id: String
        let path: String
        let classLabel: String
        let sha256: String
        let width: Int
        let height: Int
        let status: String
    }
    /// Offline injection verifies dataset behavior independently of Core ML.
    struct Classifier: Sendable {
        let path: String
        let sha256: String
        let labels: [String]
        let predict: @Sendable (URL) throws -> NativeDiffuseTextureClassifier.Prediction
        init(_ model: NativeDiffuseTextureClassifier) {
            path = model.modelURL.path; sha256 = model.modelSHA256; labels = model.labels
            predict = { try model.predict(url: $0) }
        }
        init(path: String, sha256: String, labels: [String],
             predict: @escaping @Sendable (URL) throws -> NativeDiffuseTextureClassifier.Prediction) {
            self.path = path; self.sha256 = sha256; self.labels = labels; self.predict = predict
        }
    }
    private typealias Object = [String: Any]
    private struct Material {
        let id: String
        let family: String
        let entry: Object
        let sample: Object
        let input: URL
        let inputHash: String
        let area: Int
    }
    private struct Curation {
        let labels: [String]
        let labelsByHash: [String: Set<String>]
        let hints: [String: NativeDiffuseClassifierDatasetService.Entry]
        let assets: [ClassifierOnlyAsset]
        let warnings: [String]
    }

    static func run(arguments: [String], classifier supplied: Classifier? = nil,
                    onEvent: @escaping @Sendable (String) -> Void = { _ in }) async throws -> String? {
        guard let command = arguments.first, commands.contains(command) else { return nil }
        let options = try parse(arguments)
        let workspace = canonical(options["--workspace"].map { URL(fileURLWithPath: $0) } ?? defaultWorkspace)
        if command == "specialists" { return try encode(load(workspaceURL: workspace)) }
        if command == "record-specialist-model" {
            guard let id = options["--specialist"], let target = options["--target"],
                  let checkpoint = options["--checkpoint"], let size = (options["--training-size"] ?? options["--size"]).flatMap(Int.init) else {
                throw StudioError("Choose the specialist, target, saved checkpoint, training size and completed updates.")
            }
            let info = try object(try NativeMaterialCheckpoint.inspect(at: URL(fileURLWithPath: checkpoint)))
            let updates = options["--completed-updates"].flatMap(Int.init) ?? info["step"] as? Int ?? 0
            return try encode(recordCompletedModel(workspaceURL: workspace, specialistID: id, target: target,
                checkpointURL: URL(fileURLWithPath: checkpoint), trainingSize: size, completedUpdates: updates,
                expectedSHA256: options["--expected-sha256"], expectedMembershipSHA256: options["--membership-sha256"]))
        }
        if command == "approve-specialist-model" {
            guard let id = options["--specialist"], let model = options["--model"] else {
                throw StudioError("Choose the specialist and reviewed model.")
            }
            return try encode(approveModel(workspaceURL: workspace, specialistID: id, modelID: model,
                approved: options["--unapprove"] == nil && options["--approved"] != "false"))
        }
        if command == "add-specialist" {
            guard let name = options["--name"], let label = options["--class-label"] else {
                throw StudioError("Enter a specialty name and its texture class.")
            }
            return try encode(await addSpecialist(workspaceURL: workspace, name: name, classLabel: label,
                subtype: options["--subtype"], onEvent: onEvent))
        }
        guard let path = options["--dataset"] else { throw StudioError("Choose the original material dataset to classify.") }
        let model = try supplied ?? Classifier(NativeDiffuseTextureClassifier(modelURL: options["--classifier"].map { URL(fileURLWithPath: $0) }))
        return try encode(await refresh(datasetURL: URL(fileURLWithPath: path),
            classifierDatasetURL: options["--classifier-dataset"].map { URL(fileURLWithPath: $0) },
            sourceFolderURL: options["--source-folder"].map { URL(fileURLWithPath: $0) }, workspaceURL: workspace,
            classifier: model, onEvent: onEvent))
    }

    static func load(workspaceURL: URL) throws -> Registry {
        let root = canonical(workspaceURL), url = root.appendingPathComponent(registryFilename)
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Registry.self, from: Data(contentsOf: url))
        guard value.schema == schema, canonical(URL(fileURLWithPath: value.workspacePath)) == root else {
            throw StudioError("The specialist registry does not belong to this workspace.")
        }
        for specialist in value.specialists {
            guard canonical(URL(fileURLWithPath: specialist.datasetPath)) == root.appendingPathComponent("datasets/" + specialist.id),
                  !specialist.id.isEmpty, !specialist.id.contains("/"), specialist.id != ".", specialist.id != ".." else {
                throw StudioError("The specialist dataset path has lost its workspace binding.")
            }
        }
        return value
    }

    /// Routing reads current curation rather than trusting a previous refresh's
    /// target counts. A manual add, removal or exclusion invalidates old model
    /// membership immediately, without mutating the saved registry.
    static func routingRegistry(workspaceURL: URL, classLabel: String? = nil) throws -> Registry {
        var registry = try load(workspaceURL: workspaceURL)
        for offset in registry.specialists.indices {
            let specialist = registry.specialists[offset]
            if let classLabel, specialist.classLabel != classLabel || specialist.isCustom || !(specialist.subtype?.isEmpty ?? true) {
                continue
            }
            let materials = try readMaterials(URL(fileURLWithPath: specialist.datasetPath), includeExcluded: true)
            try validateCurrentSources(materials)
            registry.specialists[offset].targets = targetStatuses(materials, models: specialist.models)
            registry.specialists[offset].materialIds = materials.map(\.id).sorted()
            registry.specialists[offset].familyIds = Array(Set(materials.map(\.family))).sorted()
        }
        return registry
    }

    static func refresh(datasetURL suppliedDataset: URL, classifierDatasetURL: URL? = nil,
                        sourceFolderURL: URL? = nil, workspaceURL: URL, classifier: Classifier,
                        onEvent: @escaping @Sendable (String) -> Void = { _ in }) async throws -> Registry {
        let dataset = datasetRoot(suppliedDataset), workspace = canonical(workspaceURL)
        guard !classifier.labels.isEmpty, Set(classifier.labels).count == classifier.labels.count,
              classifier.labels.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw StudioError("The classifier needs distinct nonempty texture labels.")
        }
        try claimWorkspace(workspace, source: dataset)
        let lock = try workspaceLock(workspace); defer { flock(lock, LOCK_UN); close(lock) }
        try migrateWorkspaceSourceIfNeeded(workspace, source: dataset)
        var registry: Registry
        if FileManager.default.fileExists(atPath: workspace.appendingPathComponent(registryFilename).path) {
            registry = try load(workspaceURL: workspace)
            guard datasetRoot(URL(fileURLWithPath: registry.sourceDatasetPath)).path == dataset.path else {
                throw StudioError("This specialist workspace belongs to another original dataset. Choose a separate workspace.")
            }
        } else {
            registry = .init(schema: schema, workspacePath: workspace.path, sourceDatasetPath: dataset.path,
                classifierDatasetPath: nil, classifierPath: classifier.path, classifierSha256: classifier.sha256,
                classifierLabels: classifier.labels, updatedAt: timestamp(), specialists: [], classifierOnlyAssets: [],
                classifierOnlyCount: 0, materialCount: 0, familyCount: 0, warnings: [])
        }
        try Task.checkCancellation()
        onEvent(try event("Reading original material inventory…", completed: 0, total: 0))
        _ = try await NativeMaterialDatasetService.run(arguments: ["dataset", "--dataset", dataset.path, "--skip-source-recovery"])
        var materials = try readMaterials(dataset)
        var warnings: [String] = []
        let sourceFolder = sourceFolderURL.map(canonical) ?? registry.sourceFolderPath.map { canonical(URL(fileURLWithPath: $0)) }
        if let folder = sourceFolder, folder.path != dataset.path {
            let inventory = workspace.appendingPathComponent(".source-inventory-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: inventory) }
            do {
                _ = try await NativeMaterialDatasetService.run(arguments: ["create-dataset", "--dataset", inventory.path, "--name", "Specialist source inventory"])
                _ = try await NativeMaterialDatasetService.run(arguments: ["import-folder", "--dataset", inventory.path, "--folder", canonical(folder).path, "--managed-import", "true", "--skip-source-recovery"])
                let additional = try readMaterials(inventory)
                var identities = Set(materials.map(sourceIdentity)), ids = Set(materials.map(\.id))
                for material in additional where !identities.contains(sourceIdentity(material)) {
                    guard ids.insert(material.id).inserted else {
                        warnings.append("A new source set uses existing material ID \(material.id); its original references need review.")
                        continue
                    }
                    materials.append(material); identities.insert(sourceIdentity(material))
                }
            } catch is CancellationError { throw CancellationError() }
            catch { warnings.append("Additional paired source inventory: \(error.localizedDescription)") }
        }
        let curationURL = classifierDatasetURL.map(canonical) ?? registry.classifierDatasetPath.map { canonical(URL(fileURLWithPath: $0)) }
        let curation = try readCuration(curationURL, materials: materials)
        warnings.append(contentsOf: curation.warnings)
        func inputFiles(_ material: Material) -> [URL] {
            var paths = Set([material.input.path])
            for variant in material.sample["input_variants"] as? [Object] ?? [] {
                if let path = (variant["source"] as? Object)?["path"] as? String { paths.insert(path) }
            }
            return paths.sorted().map { URL(fileURLWithPath: $0) }
        }
        let taughtLabels = materials.flatMap(inputFiles).compactMap { file -> String? in
            guard let tags = NativeSourceFileTags.read(file), tags.classifierIdentity == "user",
                  let label = tags.classLabel, !label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            return label
        }
        let labels = Array(Set(classifier.labels + curation.labels + taughtLabels)).sorted()
        // An old taxonomy manifest may mention labels whose folders were
        // removed by the user. Retire only empty automatic registry entries;
        // their owned on-disk folders and every curated/model record remain.
        registry.specialists = try registry.specialists.filter { specialist in
            guard !labels.contains(specialist.classLabel), !specialist.isCustom, specialist.models.isEmpty else { return true }
            let path = URL(fileURLWithPath: specialist.datasetPath)
            guard FileManager.default.fileExists(atPath: path.appendingPathComponent("dataset.json").path) else { return false }
            return !(try readMaterials(path, includeExcluded: true)).isEmpty
        }
        for label in labels where !registry.specialists.contains(where: { !$0.isCustom && $0.classLabel == label }) {
            registry.specialists.append(newSpecialist(workspace, name: label, label: label, subtype: nil, custom: false))
        }
        let groups = Dictionary(grouping: materials, by: \.family)
        let priorAssignments = Dictionary(registry.specialists.filter { !$0.isCustom }.flatMap(\.assignments)
            .map { ($0.familyId, $0) }, uniquingKeysWith: { first, _ in first })
        var assignments: [Assignment] = []
        for (offset, family) in groups.keys.sorted().enumerated() {
            try Task.checkCancellation()
            let group = groups[family]!
            let sourceFolder = !FileManager.default.fileExists(atPath: dataset.appendingPathComponent("dataset.json").path)
            let eligible = sourceFolder ? group.filter {
                let dimensions = $0.sample["source_pixel_dimensions"] as? [Int] ?? []
                return dimensions.count == 2 && dimensions.allSatisfy { $0 >= 2048 }
            } : group
            guard let chosen = eligible.sorted(by: { $0.area == $1.area ? $0.input.path < $1.input.path : $0.area < $1.area }).first else {
                warnings.append("\(family): no original diffuse image supplies the 2048 classifier grid. Its source folder is retained.")
                continue
            }
            let tags = NativeSourceFileTags.read(chosen.input)
            let familyTags = group.flatMap(inputFiles).compactMap { NativeSourceFileTags.read($0) }
                .filter { $0.classifierIdentity == "user" }
            let familyLabels = Set(familyTags.compactMap { tag -> String? in
                guard let label = tag.classLabel, !label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
                return label
            })
            if familyTags.contains(where: { $0.classLabel?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true }) {
                warnings.append("\(family): an original was explicitly left unlabeled; automatic classification was skipped for review.")
                continue
            }
            if familyLabels.count > 1 { warnings.append("\(family): original file tags teach conflicting classes; the selected file's teaching takes precedence, otherwise classification remains available for review.") }
            let identity = try NativeSourceFileTags.identity(chosen.input)
            let tagHash = tags?.verifiedSHA256.flatMap { $0.count == 64 ? $0 : nil }
            let chosenHash = try tagHash ?? fileHash(chosen.input)
            guard chosen.inputHash.isEmpty || chosenHash == chosen.inputHash else {
                warnings.append("\(family): original diffuse changed since registration; classification was skipped.")
                continue
            }
            let manualLabel = tags?.classifierIdentity == "user" ? tags?.classLabel : nil
            let chosenTeaching = manualLabel.flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
            let taught = chosenTeaching ?? (familyLabels.count == 1 ? familyLabels.first : nil)
            let taggedPrediction = tags?.classifierIdentity == classifier.sha256 ? tags?.classLabel : nil
            let cached = priorAssignments[family]
            let reusable = cached.flatMap { assignment -> Assignment? in
                guard assignment.inputSha256 == chosenHash, assignment.classifierSha256 == classifier.sha256,
                      classifier.labels.contains(assignment.predictedLabel),
                      Set(assignment.probabilities.keys).isSubset(of: Set(classifier.labels)) else { return nil }
                return assignment
            }
            let prediction: NativeDiffuseTextureClassifier.Prediction
            var performedPrediction = false
            if let taught {
                prediction = .init(label: taught, confidence: 1, probabilities: [taught: 1], modelSHA256: classifier.sha256)
            } else if let taggedPrediction, classifier.labels.contains(taggedPrediction) {
                if let reusable, reusable.predictedLabel == taggedPrediction {
                    prediction = .init(label: reusable.predictedLabel, confidence: reusable.confidence,
                        probabilities: reusable.probabilities, modelSHA256: classifier.sha256)
                } else {
                    // File tags persist the class, not a calibrated confidence.
                    prediction = .init(label: taggedPrediction, confidence: 0, probabilities: [:], modelSHA256: classifier.sha256)
                }
            } else if let reusable {
                prediction = .init(label: reusable.predictedLabel, confidence: reusable.confidence,
                    probabilities: reusable.probabilities, modelSHA256: classifier.sha256)
            } else {
                do { prediction = try classifier.predict(chosen.input); performedPrediction = true }
                catch is CancellationError { throw CancellationError() }
                catch { warnings.append("\(family): classification failed: \(error.localizedDescription)"); continue }
            }
            let stablePredictionBytes = performedPrediction ? try fileHash(chosen.input) == chosenHash : true
            guard (taught != nil || classifier.labels.contains(prediction.label)), prediction.modelSHA256 == classifier.sha256,
                  prediction.confidence.isFinite, (0...1).contains(prediction.confidence),
                  prediction.probabilities.allSatisfy({ (taught != nil || classifier.labels.contains($0.key)) && $0.value.isFinite && (0...1).contains($0.value) }),
                  try NativeSourceFileTags.identity(chosen.input) == identity, stablePredictionBytes else {
                throw StudioError("The classifier returned an invalid prediction or the original diffuse changed during classification.")
            }
            if performedPrediction, NativeSourceFileTags.read(chosen.input)?.classifierIdentity != "user" {
                do { try NativeSourceFileTags.update(chosen.input, classLabel: prediction.label,
                                                     classifierIdentity: classifier.sha256, verifiedSHA256: chosenHash) }
                catch { warnings.append("\(family): classification was saved in the specialist workspace, but its optional file tag could not be saved: \(error.localizedDescription)") }
            }
            let curatedLabels = Set(group.flatMap { inputHashes($0.sample).flatMap { curation.labelsByHash[$0] ?? [] } })
            var label = prediction.label, source = taught == nil ? "classifier" : "user_file_tag"
            if taught == nil, curatedLabels.count == 1, let curated = curatedLabels.first { label = curated; source = "curated_classifier_dataset" }
            else if taught == nil, curatedLabels.count > 1 { warnings.append("\(family): curated color variants have different labels; retained classifier prediction for review.") }
            let hint = curation.hints[family]
            let semantic = semanticTerms(chosen, hint: hint)
            let classID = NativeDiffuseClassifierDatasetService.classes.first { $0.name == label }?.id
            let subtype = hint?.subtype ?? NativeDiffuseClassifierDatasetService.subtype(family: family,
                filename: chosen.input.lastPathComponent, metadata: hint?.providerMetadata, classID: classID)
            assignments.append(.init(familyId: family, materialIds: group.map(\.id).sorted(), inputPath: chosen.input.path,
                inputSha256: chosenHash, predictedLabel: prediction.label, classLabel: label,
                confidence: prediction.confidence, probabilities: prediction.probabilities,
                classifierSha256: classifier.sha256, source: source, subtype: subtype, semanticTerms: semantic))
            onEvent(try event("Classified \(family) as \(label)", completed: offset + 1, total: groups.count))
        }
        let byID = Dictionary(uniqueKeysWithValues: materials.map { ($0.id, $0) })
        for offset in registry.specialists.indices {
            try Task.checkCancellation()
            var specialist = registry.specialists[offset]
            let selected = assignments.filter { assignment in
                assignment.classLabel == specialist.classLabel && matchesSubtype(specialist.subtype, assignment: assignment)
            }
            specialist.assignments = selected
            let selectedIDs = Set(selected.flatMap(\.materialIds))
            do {
                try await synchronizeDataset(&specialist, desired: selectedIDs.compactMap { byID[$0] }, workspace: workspace,
                    warnings: &warnings)
                registry.specialists[offset] = specialist
            } catch is CancellationError { throw CancellationError() }
            catch {
                warnings.append("\(specialist.name): specialist references could not refresh: \(error.localizedDescription)")
                // Previously completed models and dataset curation stay saved.
                registry.specialists[offset].assignments = selected
            }
        }
        registry.classifierDatasetPath = curationURL?.path; registry.classifierPath = classifier.path
        registry.sourceFolderPath = sourceFolder?.path
        registry.classifierSha256 = classifier.sha256; registry.classifierLabels = classifier.labels
        registry.classifierOnlyAssets = curation.assets; registry.classifierOnlyCount = curation.assets.count
        registry.materialCount = materials.count; registry.familyCount = groups.count
        registry.updatedAt = timestamp(); registry.warnings = warnings
        registry.specialists.sort { $0.isCustom == $1.isCustom ? $0.name.localizedStandardCompare($1.name) == .orderedAscending : !$0.isCustom }
        try save(registry, workspace: workspace)
        return registry
    }

    static func addSpecialist(workspaceURL: URL, name: String, classLabel: String, subtype: String? = nil,
                              onEvent: @escaping @Sendable (String) -> Void = { _ in }) async throws -> Registry {
        let workspace = canonical(workspaceURL), lock = try workspaceLock(workspace)
        defer { flock(lock, LOCK_UN); close(lock) }
        var registry = try load(workspaceURL: workspace)
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let subtype = subtype.flatMap { value -> String? in
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        guard !name.isEmpty, registry.specialists.contains(where: { !$0.isCustom && $0.classLabel == classLabel }) else {
            throw StudioError("Enter a specialty name and choose an existing classifier class.")
        }
        var specialist = newSpecialist(workspace, name: name, label: classLabel, subtype: subtype, custom: true)
        if registry.specialists.contains(where: { $0.id == specialist.id }) { return registry }
        let assignments = registry.specialists.filter { !$0.isCustom && $0.classLabel == classLabel }.flatMap(\.assignments)
            .filter { matchesSubtype(subtype, assignment: $0) }
        specialist.assignments = assignments
        // Use already registered original references, including additional
        // source sets which were deliberately kept outside the main database.
        var originals: [Material] = []
        for base in registry.specialists where !base.isCustom && base.classLabel == classLabel {
            originals.append(contentsOf: try readMaterials(URL(fileURLWithPath: base.datasetPath)))
        }
        let ids = Set(assignments.flatMap(\.materialIds)); var warnings = registry.warnings
        try await synchronizeDataset(&specialist, desired: originals.filter { ids.contains($0.id) }, workspace: workspace, warnings: &warnings)
        registry.specialists.append(specialist); registry.updatedAt = timestamp(); registry.warnings = warnings
        try save(registry, workspace: workspace)
        onEvent(try event("Created \(name) from original material references", completed: 1, total: 1))
        return registry
    }

    /// Called only after a successful native save. Approval remains a separate
    /// user review action; zero-update initialization never becomes a model.
    static func recordCompletedModel(workspaceURL: URL, specialistID: String, target: String,
                                     checkpointURL: URL, trainingSize: Int, completedUpdates: Int,
                                     expectedSHA256: String? = nil, expectedMembershipSHA256: String? = nil) throws -> Registry {
        let workspace = canonical(workspaceURL), lock = try workspaceLock(workspace)
        defer { flock(lock, LOCK_UN); close(lock) }
        var registry = try load(workspaceURL: workspace)
        guard completedUpdates > 0, let offset = registry.specialists.firstIndex(where: { $0.id == specialistID }) else {
            throw StudioError("A specialist model needs a saved checkpoint with completed training updates.")
        }
        var specialist = registry.specialists[offset]
        let records = try readMaterials(URL(fileURLWithPath: specialist.datasetPath), includeExcluded: true)
        try validateCurrentSources(records)
        specialist.targets = targetStatuses(records, models: specialist.models)
        guard let membership = specialist.targets.first(where: { $0.target == target }), membership.materialCount > 0,
              expectedMembershipSHA256 == nil || expectedMembershipSHA256 == membership.membershipSha256 else {
            throw StudioError("The specialist's eligible originals changed during training. Keep the saved checkpoint and refresh its membership before registering it.")
        }
        let inspected = try object(try NativeMaterialCheckpoint.inspect(at: checkpointURL, expectedSHA256: expectedSHA256))
        guard inspected["variant"] as? String == "full", inspected["schema"] as? String == NativeCompactMaterialModel.schema,
              inspected["compatible"] as? Bool == true, inspected["target"] as? String == target,
              inspected["training_size"] as? Int == trainingSize, let step = inspected["step"] as? Int,
              step >= completedUpdates, let hash = inspected["sha256"] as? String,
              let path = inspected["checkpoint_path"] as? String, let family = inspected["model_family"] as? String else {
            throw StudioError("The saved checkpoint is not a trained full native model matching this target and training size.")
        }
        let snapshot = try NativeSafetensors(contentsOf: URL(fileURLWithPath: path), expectedSHA256: hash)
        guard let configurationText = snapshot.metadata["configuration"], let configuration = try? object(configurationText),
              let training = configuration["training_configuration"] as? Object,
              let cachePath = training["training_cache_root"] as? String,
              canonical(URL(fileURLWithPath: cachePath)) == canonical(URL(fileURLWithPath: specialist.datasetPath)).appendingPathComponent(".training-data"),
              let initialStep = training["initial_step"] as? Int, step > initialStep,
              step - initialStep >= completedUpdates,
              let manifestHash = training["dataset_manifest_sha256"] as? String, manifestHash.count == 64 else {
            throw StudioError("The checkpoint was not trained from this specialist's original dataset. Its saved artifact was preserved.")
        }
        if !specialist.models.contains(where: { $0.checkpointSha256 == hash && $0.membershipSha256 == membership.membershipSha256 }) {
            specialist.models.append(.init(id: "model-" + hash.prefix(16), target: target, modelFamily: family,
                checkpointPath: path, checkpointSha256: hash, trainingSize: trainingSize, completedUpdates: completedUpdates,
                completedAt: timestamp(), membershipSha256: membership.membershipSha256, approved: false, approvedAt: nil))
        }
        specialist.targets = targetStatuses(records, models: specialist.models)
        registry.specialists[offset] = specialist; registry.updatedAt = timestamp()
        try save(registry, workspace: workspace)
        return registry
    }

    static func approveModel(workspaceURL: URL, specialistID: String, modelID: String, approved: Bool = true) throws -> Registry {
        let workspace = canonical(workspaceURL), lock = try workspaceLock(workspace)
        defer { flock(lock, LOCK_UN); close(lock) }
        var registry = try load(workspaceURL: workspace)
        guard let specialistOffset = registry.specialists.firstIndex(where: { $0.id == specialistID }),
              let modelOffset = registry.specialists[specialistOffset].models.firstIndex(where: { $0.id == modelID }) else {
            throw StudioError("Choose a saved specialist model to review.")
        }
        var specialist = registry.specialists[specialistOffset], model = specialist.models[modelOffset]
        let records = try readMaterials(URL(fileURLWithPath: specialist.datasetPath), includeExcluded: true)
        if approved {
            try validateCurrentSources(records)
            let inspected = try object(try NativeMaterialCheckpoint.inspect(at: URL(fileURLWithPath: model.checkpointPath), expectedSHA256: model.checkpointSha256))
            let current = targetStatuses(records, models: specialist.models).first { $0.target == model.target }
            guard model.completedUpdates > 0, inspected["variant"] as? String == "full",
                  ["compact-scalar", "compact-normal"].contains(model.modelFamily),
                  inspected["model_family"] as? String == model.modelFamily, inspected["target"] as? String == model.target,
                  (inspected["step"] as? Int ?? 0) >= model.completedUpdates,
                  current?.membershipSha256 == model.membershipSha256, (current?.materialCount ?? 0) > 0 else {
                throw StudioError("Only a current, trained diffuse specialist checkpoint can be approved for automatic routing.")
            }
        }
        model.approved = approved; model.approvedAt = approved ? timestamp() : nil
        specialist.models[modelOffset] = model; specialist.targets = targetStatuses(records, models: specialist.models)
        registry.specialists[specialistOffset] = specialist; registry.updatedAt = timestamp()
        try save(registry, workspace: workspace)
        return registry
    }

    private static func synchronizeDataset(_ specialist: inout Specialist, desired: [Material], workspace: URL,
                                            warnings: inout [String]) async throws {
        let root = URL(fileURLWithPath: specialist.datasetPath), manifest = root.appendingPathComponent("dataset.json")
        let bindingURL = root.appendingPathComponent(".specialist-binding.json")
        if !FileManager.default.fileExists(atPath: manifest.path) {
            _ = try await NativeMaterialDatasetService.run(arguments: ["create-dataset", "--dataset", root.path,
                "--name", specialist.name, "--description", "Original material references for \(specialist.classLabel) specialization."])
            try write(["schema": schema, "specialist_id": specialist.id, "workspace_path": workspace.path], to: bindingURL)
        } else {
            let binding = try object(bindingURL)
            guard binding["schema"] as? String == schema, binding["specialist_id"] as? String == specialist.id,
                  binding["workspace_path"] as? String == workspace.path else {
                throw StudioError("An unrelated dataset occupies the specialist folder; it was preserved.")
            }
        }
        var current = try readMaterials(root, includeExcluded: true)
        var automaticRemovals = Set(specialist.automaticallyRemovedMaterialIds)
        let desiredIDs = Set(desired.map(\.id)), priorIDs = Set(specialist.automaticallyManagedMaterialIds)
        let manualIDs = Set(current.map(\.id)).subtracting(priorIDs)
        try refreshManagedSourceReferences(root, current: current, desired: desired, managedIDs: priorIDs, warnings: &warnings)
        current = try readMaterials(root, includeExcluded: true)
        for material in current where priorIDs.contains(material.id) && !desiredIDs.contains(material.id) {
            let status = material.sample["status"] as? String ?? "unreviewed"
            if status == "unreviewed", (material.sample["curation_note"] as? String ?? "").isEmpty {
                _ = try await NativeMaterialDatasetService.run(arguments: ["remove-material", "--dataset", root.path, "--material", material.id])
                automaticRemovals.insert(material.id)
            } else {
                warnings.append("\(specialist.name): retained curated material \(material.id) after its classifier assignment changed.")
            }
        }
        current = try readMaterials(root, includeExcluded: true)
        let index = try object(root.appendingPathComponent("dataset.json"))
        let tombstones = Set((index["removed_materials"] as? [Object] ?? []).compactMap { $0["material_id"] as? String })
        let present = Set(current.map(\.id))
        let additions = desired.filter { !present.contains($0.id) && (!tombstones.contains($0.id) || automaticRemovals.contains($0.id)) }
        if !additions.isEmpty {
            let stage = workspace.appendingPathComponent(".specialist-import-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
            defer { try? FileManager.default.removeItem(at: stage) }
            var entries: [Object] = []
            for material in additions {
                let sampleID = material.sample["sample_id"] as? String ?? material.id + "_full"
                let relative = "samples/" + safeIdentifier(sampleID)
                let folder = stage.appendingPathComponent(relative)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                try write(material.sample, to: folder.appendingPathComponent("sample.json"))
                var entry = material.entry; entry["path"] = relative; entry["sample_id"] = sampleID
                entries.append(entry)
            }
            try write(["schema_version": 2, "storage_policy": "original-source-references-v1", "samples": entries], to: stage.appendingPathComponent("dataset.json"))
            let deferred = additions.contains { $0.inputHash.isEmpty }
            _ = try await NativeMaterialDatasetService.run(arguments: ["import-folder", "--dataset", root.path, "--folder", stage.path]
                + (deferred ? ["--managed-import", "true", "--skip-source-recovery"] : []))
            for material in additions { automaticRemovals.remove(material.id) }
        }
        let final = try readMaterials(root, includeExcluded: true)
        specialist.materialIds = final.map(\.id).sorted(); specialist.familyIds = Array(Set(final.map(\.family))).sorted()
        specialist.targets = targetStatuses(final, models: specialist.models)
        specialist.automaticallyManagedMaterialIds = Array(priorIDs.union(desiredIDs.subtracting(manualIDs))).sorted()
        specialist.automaticallyRemovedMaterialIds = automaticRemovals.sorted()
    }

    /// A folder rename changes authoritative paths while retaining a material
    /// ID. Refresh only app-managed references; reviews and manual additions
    /// remain owned by the user, and pending new proofs stay pending.
    private static func refreshManagedSourceReferences(_ root: URL, current: [Material], desired: [Material],
                                                        managedIDs: Set<String>, warnings: inout [String]) throws {
        func locations(_ sample: Object) -> [String] {
            let maps = sample["map_metadata"] as? [String: Object] ?? [:]
            var paths = maps.keys.sorted().map { role in
                role + "|" + (((maps[role]?["source"] as? Object)?["path"] as? String).map { canonical(URL(fileURLWithPath: $0)).path } ?? "")
            }
            paths += (sample["input_variants"] as? [Object] ?? []).compactMap { ($0["source"] as? Object)?["path"] as? String }
                .map { "color|" + canonical(URL(fileURLWithPath: $0)).path }.sorted()
            return paths
        }
        let wanted = Dictionary(desired.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        let reviewURL = root.appendingPathComponent(".material-size-reviews.json")
        var reviews = FileManager.default.fileExists(atPath: reviewURL.path) ? try object(reviewURL) : [:]
        var index = try object(root.appendingPathComponent("dataset.json")), updates: [(URL, Object)] = []
        var changedReviews = false, changedIndex = false
        for material in current where managedIDs.contains(material.id) {
            guard let replacement = wanted[material.id], locations(material.sample) != locations(replacement.sample),
                  let relative = material.entry["path"] as? String else { continue }
            try Task.checkCancellation()
            var sample = material.sample
            for key in ["maps", "map_metadata", "input_variants", "source_directory", "source_pixel_dimensions", "sample_pixel_dimensions",
                        "crop_rectangle_top_left_xywh", "source_resolution_label", "source_discovery_warnings", "ignored_source_files",
                        "available_targets", "source_verification", "source_precision_verified", "crop_values_verified", "source_verification_issues"] {
                sample[key] = replacement.sample[key]
            }
            sample.removeValue(forKey: "source_files"); sample.removeValue(forKey: "source_binding_sha256")
            if sample["source_verification"] as? String == "header_inspected" {
                index["managed_source_folders"] = true; changedIndex = true
            }
            let oldBinding = NativeMaterialDatasetService.sourceBinding(material.sample)
            let newBinding = NativeMaterialDatasetService.sourceBinding(sample)
            for key in reviews.keys {
                let parts = key.split(separator: ":", maxSplits: 1)
                guard parts.count == 2, parts[1].hasPrefix(material.id + "_"),
                      var review = reviews[key] as? Object, review["source_binding_sha256"] as? String == oldBinding else { continue }
                review["source_binding_sha256"] = newBinding; reviews[key] = review; changedReviews = true
            }
            if let old = material.sample["source_directory"] as? String, let new = sample["source_directory"] as? String,
               canonical(URL(fileURLWithPath: old)).path != canonical(URL(fileURLWithPath: new)).path,
               var settings = index["validation"] as? Object, var flags = settings["folders"] as? [String: Bool],
               let flag = flags.removeValue(forKey: canonical(URL(fileURLWithPath: old)).path) {
                let newPath = canonical(URL(fileURLWithPath: new)).path
                if flags[newPath] == nil { flags[newPath] = flag }
                settings["folders"] = flags; index["validation"] = settings; changedIndex = true
            }
            if var catalog = index["source_inventory"] as? [Object] {
                let oldSources = material.sample["map_metadata"] as? [String: Object] ?? [:]
                let newSources = sample["map_metadata"] as? [String: Object] ?? [:]
                for offset in catalog.indices where catalog[offset]["material_id"] as? String == material.id {
                    for role in oldSources.keys {
                        let oldPath = (oldSources[role]?["source"] as? Object)?["path"] as? String
                        guard catalog[offset]["path"] as? String == oldPath,
                              let source = newSources[role]?["source"] as? Object, let path = source["path"] as? String else { continue }
                        catalog[offset]["path"] = path; catalog[offset]["id"] = hash(Data(path.utf8))
                        catalog[offset]["filename"] = URL(fileURLWithPath: path).lastPathComponent
                        catalog[offset]["directory"] = URL(fileURLWithPath: path).deletingLastPathComponent().path
                        catalog[offset]["source_verification"] = source["source_verification"] ?? "header_inspected"
                        catalog[offset]["state"] = "paired"; catalog[offset].removeValue(forKey: "issue")
                    }
                }
                index["source_inventory"] = catalog; changedIndex = true
            }
            let folder = canonical(root.appendingPathComponent(relative))
            guard folder.path.hasPrefix(root.path + "/") else { throw StudioError("Original sample metadata escapes its specialist dataset.") }
            updates.append((folder.appendingPathComponent("sample.json"), sample))
        }
        guard !updates.isEmpty else { return }
        if changedReviews { updates.append((reviewURL, reviews)) }
        if changedIndex { updates.append((root.appendingPathComponent("dataset.json"), index)) }
        let previous = try updates.map { (url, _) in (url, FileManager.default.fileExists(atPath: url.path) ? try Data(contentsOf: url) : nil) }
        do {
            for (url, value) in updates { try Task.checkCancellation(); try write(value, to: url) }
        } catch {
            for (url, data) in previous {
                if let data { try? data.write(to: url, options: .atomic) }
                else { try? FileManager.default.removeItem(at: url) }
            }
            throw error
        }
    }

    /// The exact input variants and target hashes bind completion and routing to
    /// current active originals; excluded entries and classifier photos never
    /// contribute a fictional target.
    private static func targetStatuses(_ records: [Material], models: [ModelRecord]) -> [TargetStatus] {
        ["height", "roughness", "normal"].map { target in
            let active = records.filter { material in
                !["excluded", "rejected"].contains(material.sample["status"] as? String ?? "unreviewed") &&
                    (material.sample["available_targets"] as? [String] ?? []).contains(target)
            }
            let identities: [[String]] = active.sorted { $0.id < $1.id }.map { material in
                let metadata = material.sample["map_metadata"] as? [String: Object] ?? [:]
                let source = metadata[target]?["source"] as? Object ?? [:]
                return [material.id, material.family, source["path"] as? String ?? "", source["file_sha256"] as? String ?? ""] + inputHashes(material.sample).sorted()
            }
            let membership = hash((try? JSONSerialization.data(withJSONObject: identities, options: [.sortedKeys])) ?? Data())
            let saved = models.filter { $0.target == target && $0.completedUpdates > 0 }
            let current = saved.filter { $0.membershipSha256 == membership }
            let status = active.isEmpty ? "no_targets" : current.contains(where: \.approved) ? "approved" : !current.isEmpty ? "awaiting_review" : saved.isEmpty ? "ready" : "needs_update"
            return .init(target: target, materialCount: active.count, status: status, membershipSha256: membership)
        }
    }

    private static func readMaterials(_ supplied: URL, includeExcluded: Bool = false) throws -> [Material] {
        let root = datasetRoot(supplied)
        let registrations: [(entry: Object, sample: Object)]
        if !FileManager.default.fileExists(atPath: root.appendingPathComponent("dataset.json").path) {
            registrations = try NativeMaterialDatasetService.sourceFolderRecords(root)
        } else {
            let index = try object(root.appendingPathComponent("dataset.json"))
            guard index["schema_version"] as? Int == 2, index["storage_policy"] as? String == "original-source-references-v1",
                  index["native_size_preparation"] == nil else {
                throw StudioError("Specialization needs original sources, not prepared crops.")
            }
            registrations = try (index["samples"] as? [Object] ?? []).map { entry in
                guard let path = entry["path"] as? String else { throw StudioError("Original dataset sample path is missing.") }
                let folder = canonical(root.appendingPathComponent(path))
                guard folder.path.hasPrefix(root.path + "/") else { throw StudioError("Original sample metadata escapes its dataset.") }
                return (entry, try object(folder.appendingPathComponent("sample.json")))
            }
        }
        var result: [Material] = [], seen = Set<String>()
        for registration in registrations {
            let entry = registration.entry, sample = registration.sample
            if !includeExcluded && ["excluded", "rejected"].contains(sample["status"] as? String ?? "unreviewed") { continue }
            guard let id = sample["material_id"] as? String, seen.insert(id).inserted,
                  let metadata = sample["map_metadata"] as? [String: Object], let details = metadata["input"],
                  details["storage"] as? String == "source_reference", details["exact_source_crop"] as? Bool != false,
                  let source = details["source"] as? Object, let path = source["path"] as? String, path.hasPrefix("/"),
                  let width = source["width"] as? Int,
                  let height = source["height"] as? Int, width > 0, height > 0,
                  ((source["file_sha256"] as? String)?.count == 64 && details["sample_sha256"] as? String == source["file_sha256"] as? String
                    || source["source_verification"] as? String == "header_inspected" && source["source_stat"] is Object),
                  (sample["maps"] as? [String: String])?["input"] == path,
                  let common = sample["sample_pixel_dimensions"] as? [Int], common.count == 2,
                  common[0] <= width, common[1] <= height, width - common[0] <= 8, height - common[1] <= 8 else {
                throw StudioError("Original material reference or diffuse dimensions are invalid.")
            }
            let area = width.multipliedReportingOverflow(by: height)
            guard !area.overflow else { throw StudioError("Diffuse dimensions overflow.") }
            result.append(.init(id: id, family: sample["source_family_id"] as? String ?? sample["asset_family_id"] as? String ?? id,
                entry: entry, sample: sample, input: URL(fileURLWithPath: path), inputHash: source["file_sha256"] as? String ?? "", area: area.partialValue))
        }
        return result
    }

    private static func readCuration(_ root: URL?, materials: [Material]) throws -> Curation {
        guard let root else { return .init(labels: [], labelsByHash: [:], hints: [:], assets: [], warnings: []) }
        guard FileManager.default.fileExists(atPath: root.path), (try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])).isSymbolicLink != true else {
            throw StudioError("Choose the existing classifier curation folder, not a symbolic link.")
        }
        let manifestURL = root.appendingPathComponent(".classifier-manifest.json")
        let manifest: NativeDiffuseClassifierDatasetService.Manifest?
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            manifest = try JSONDecoder().decode(NativeDiffuseClassifierDatasetService.Manifest.self, from: Data(contentsOf: manifestURL))
        } else { manifest = nil }
        let known = Set(materials.flatMap { inputHashes($0.sample) })
        var labels = Set<String>(), labelsByHash: [String: Set<String>] = [:]
        var assets: [ClassifierOnlyAsset] = [], warnings: [String] = []
        let folders = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles])
        for folder in folders.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let values = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true else { continue }
            let label = folder.lastPathComponent; labels.insert(label)
            guard let enumerator = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles]) else { continue }
            for case let file as URL in enumerator {
                try Task.checkCancellation()
                let attributes = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                guard attributes.isRegularFile == true, attributes.isSymbolicLink != true,
                      ["png", "jpg", "jpeg", "tif", "tiff", "heic", "bmp", "webp"].contains(file.pathExtension.lowercased()) else { continue }
                guard let image = CGImageSourceCreateWithURL(file as CFURL, nil),
                      let properties = CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any],
                      let width = properties[kCGImagePropertyPixelWidth] as? Int,
                      let height = properties[kCGImagePropertyPixelHeight] as? Int, width > 0, height > 0 else {
                    warnings.append("Classifier-only image could not be inspected: \(file.path)"); continue
                }
                let digest = try fileHash(file); labelsByHash[digest, default: []].insert(label)
                if !known.contains(digest) {
                    assets.append(.init(id: "classifier-only-" + hash(Data(file.path.utf8)).prefix(20), path: file.path,
                        classLabel: label, sha256: digest, width: width, height: height, status: "classifier_only_no_targets"))
                }
            }
        }
        let hints = Dictionary((manifest?.entries ?? []).map { ($0.family, $0) }, uniquingKeysWith: { first, _ in first })
        return .init(labels: labels.sorted(), labelsByHash: labelsByHash, hints: hints,
            assets: assets.sorted { $0.path < $1.path }, warnings: warnings)
    }

    private static func semanticTerms(_ material: Material, hint: NativeDiffuseClassifierDatasetService.Entry?) -> [String] {
        var terms = [material.family, material.sample["name"] as? String ?? "", material.input.deletingPathExtension().lastPathComponent]
        if let hint {
            terms += hint.providerMetadata?.tags ?? []; terms += hint.providerMetadata?.categories ?? []
            terms += [hint.providerMetadata?.name ?? "", hint.subtype ?? ""]
        }
        terms += material.sample["tags"] as? [String] ?? []
        if let provider = material.sample["provider_metadata"] as? Object { terms += provider["tags"] as? [String] ?? [] }
        return Array(Set(terms.map(normalized).filter { !$0.isEmpty })).sorted()
    }
    private static func matchesSubtype(_ subtype: String?, assignment: Assignment) -> Bool {
        guard let subtype else { return true }
        let needle = normalized(subtype)
        if assignment.subtype.map(normalized) == needle { return true }
        return assignment.semanticTerms.contains { (" " + $0 + " ").contains(" " + needle + " ") }
    }
    private static func normalized(_ text: String) -> String {
        text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).joined(separator: " ")
    }
    private static func inputHashes(_ sample: Object) -> [String] {
        let variants = sample["input_variants"] as? [Object] ?? []
        let metadata = sample["map_metadata"] as? [String: Object] ?? [:]
        return Array(Set((variants + [metadata["input"] ?? [:]]).compactMap {
            ($0["source"] as? Object)?["file_sha256"] as? String
        })).sorted()
    }
    private static func sourceIdentity(_ material: Material) -> String {
        let maps = material.sample["map_metadata"] as? [String: Object] ?? [:]
        let identities = maps.keys.sorted().map { role -> [String] in
            let source = maps[role]?["source"] as? Object ?? [:]
            return [role, source["path"] as? String ?? "", source["file_sha256"] as? String ?? ""]
        } + inputHashes(material.sample).map { ["input_variant", $0] }
        return hash((try? JSONSerialization.data(withJSONObject: identities, options: [.sortedKeys])) ?? Data())
    }
    private static func validateCurrentSources(_ materials: [Material]) throws {
        var checked = Set<String>()
        for material in materials where !["excluded", "rejected"].contains(material.sample["status"] as? String ?? "unreviewed") {
            let metadata = material.sample["map_metadata"] as? [String: Object] ?? [:]
            let details = Array(metadata.values) + (material.sample["input_variants"] as? [Object] ?? [])
            for detail in details {
                guard let source = detail["source"] as? Object, let path = source["path"] as? String,
                      let expected = source["file_sha256"] as? String else { throw StudioError("A specialist original source binding is missing.") }
                guard checked.insert(path).inserted else { continue }
                var actual = stat()
                guard stat(path, &actual) == 0 else { throw StudioError("A specialist original map is missing: \(path)") }
                let proof = source["source_stat"] as? Object ?? [:]
                let size = (proof["size"] as? NSNumber)?.int64Value
                let mtime = (proof["mtime_ns"] as? NSNumber)?.int64Value
                let ctime = (proof["ctime_ns"] as? NSNumber)?.int64Value
                let same = size == Int64(actual.st_size) &&
                    mtime == Int64(actual.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(actual.st_mtimespec.tv_nsec) &&
                    ctime == Int64(actual.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(actual.st_ctimespec.tv_nsec)
                guard try same || fileHash(URL(fileURLWithPath: path)) == expected else {
                    throw StudioError("A specialist original map changed after training: \(path)")
                }
            }
        }
    }
    private static func newSpecialist(_ workspace: URL, name: String, label: String, subtype: String?, custom: Bool) -> Specialist {
        let identity = custom ? label + "|" + (subtype ?? "") + "|" + name : label
        let id = (custom ? "custom-" : "class-") + safeIdentifier(name).prefix(48) + "-" + hash(Data(identity.utf8)).prefix(12)
        return .init(id: id, name: name, classLabel: label, subtype: subtype, isCustom: custom,
            datasetPath: workspace.appendingPathComponent("datasets/" + id).path, materialIds: [], familyIds: [],
            assignments: [], targets: [], models: [], automaticallyManagedMaterialIds: [], automaticallyRemovedMaterialIds: [])
    }
    private static func safeIdentifier(_ text: String) -> String {
        let value = normalized(text).replacingOccurrences(of: " ", with: "_")
        return value.isEmpty ? "specialist" : value
    }
    /// Metadata-only eligibility lets the browser retain an existing workspace
    /// when the same originals move from a legacy reference dataset to a folder.
    /// It never creates folders, writes markers or reads compressed image bytes.
    static func canUseWorkspace(workspaceURL: URL, datasetURL: URL) throws -> Bool {
        let root = canonical(workspaceURL), source = datasetRoot(datasetURL)
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &directory) else { return true }
        guard directory.boolValue, (try root.resourceValues(forKeys: [.isSymbolicLinkKey])).isSymbolicLink != true else { return false }
        let marker = root.appendingPathComponent(".specialist-workspace.json")
        guard FileManager.default.fileExists(atPath: marker.path) else {
            return try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).isEmpty
        }
        let value = try object(marker)
        guard value["schema"] as? String == schema,
              let workspacePath = value["workspace_path"] as? String, canonical(URL(fileURLWithPath: workspacePath)).path == root.path,
              let oldPath = value["source_dataset_path"] as? String, oldPath.hasPrefix("/") else { return false }
        let oldSource = datasetRoot(URL(fileURLWithPath: oldPath))
        let registryURL = root.appendingPathComponent(registryFilename)
        let registry = FileManager.default.fileExists(atPath: registryURL.path) ? try load(workspaceURL: root) : nil
        if let registry, datasetRoot(URL(fileURLWithPath: registry.sourceDatasetPath)).path != oldSource.path { return false }
        if oldSource.path == source.path { return true }
        guard let registry else { return false }
        try validateMigrationBindings(registry, workspace: root)
        return try legacyDatasetUsesFolder(oldSource, folder: source)
    }
    private static func legacyDatasetUsesFolder(_ legacy: URL, folder: URL) throws -> Bool {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: legacy.appendingPathComponent("dataset.json").path),
              !FileManager.default.fileExists(atPath: folder.appendingPathComponent("dataset.json").path),
              FileManager.default.fileExists(atPath: folder.path, isDirectory: &directory), directory.boolValue else { return false }
        let records = try readMaterials(legacy, includeExcluded: true)
        guard !records.isEmpty else { return false }
        let prefix = folder.path + "/"
        for record in records {
            let details = Array((record.sample["map_metadata"] as? [String: Object] ?? [:]).values)
                + (record.sample["input_variants"] as? [Object] ?? [])
            guard !details.isEmpty else { return false }
            for map in details {
                guard map["storage"] as? String == "source_reference",
                      let path = (map["source"] as? Object)?["path"] as? String, path.hasPrefix("/") else { return false }
                let file = canonical(URL(fileURLWithPath: path))
                guard file.path.hasPrefix(prefix), FileManager.default.fileExists(atPath: file.path),
                      try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { return false }
            }
        }
        return true
    }
    private static func validateMigrationBindings(_ registry: Registry, workspace: URL) throws {
        for specialist in registry.specialists {
            let dataset = canonical(URL(fileURLWithPath: specialist.datasetPath))
            let bindingURL = dataset.appendingPathComponent(".specialist-binding.json")
            guard FileManager.default.fileExists(atPath: dataset.appendingPathComponent("dataset.json").path),
                  FileManager.default.fileExists(atPath: bindingURL.path) else {
                throw StudioError("A specialist dataset is missing its workspace binding; it was preserved.")
            }
            let binding = try object(bindingURL)
            guard binding["schema"] as? String == schema, binding["specialist_id"] as? String == specialist.id,
                  let path = binding["workspace_path"] as? String, canonical(URL(fileURLWithPath: path)).path == workspace.path else {
                throw StudioError("An unrelated dataset occupies the specialist folder; it was preserved.")
            }
        }
    }
    private static func migrateWorkspaceSourceIfNeeded(_ root: URL, source: URL) throws {
        let markerURL = root.appendingPathComponent(".specialist-workspace.json")
        let marker = try object(markerURL)
        guard let path = marker["source_dataset_path"] as? String else { throw StudioError("The specialist workspace has no source dataset binding.") }
        guard datasetRoot(URL(fileURLWithPath: path)).path != source.path else { return }
        guard try canUseWorkspace(workspaceURL: root, datasetURL: source) else {
            throw StudioError("The specialist workspace belongs to another source dataset. Choose a separate workspace.")
        }
        let registryURL = root.appendingPathComponent(registryFilename)
        let priorRegistry = try Data(contentsOf: registryURL), priorMarker = try Data(contentsOf: markerURL)
        var registry = try load(workspaceURL: root), revisedMarker = marker
        registry.sourceDatasetPath = source.path; revisedMarker["source_dataset_path"] = source.path
        // Rebind only ownership metadata. Existing category datasets, reviews,
        // classifier curation paths and completed model histories stay intact.
        do {
            try Task.checkCancellation()
            try save(registry, workspace: root)
            try Task.checkCancellation()
            try write(revisedMarker, to: markerURL)
        } catch {
            try? priorRegistry.write(to: registryURL, options: .atomic)
            try? priorMarker.write(to: markerURL, options: .atomic)
            throw error
        }
    }
    private static func claimWorkspace(_ root: URL, source: URL) throws {
        let marker = root.appendingPathComponent(".specialist-workspace.json")
        if FileManager.default.fileExists(atPath: root.path) {
            guard (try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])).isSymbolicLink != true else {
                throw StudioError("Specialist workspaces cannot be symbolic links.")
            }
            let contents = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            if !contents.isEmpty, !FileManager.default.fileExists(atPath: marker.path) {
                throw StudioError("The chosen specialist folder contains unrelated files. Choose an empty workspace.")
            }
        } else { try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true) }
        if FileManager.default.fileExists(atPath: marker.path) {
            guard try canUseWorkspace(workspaceURL: root, datasetURL: source) else {
                throw StudioError("The specialist workspace belongs to another source dataset.")
            }
        } else {
            try write(["schema": schema, "workspace_path": root.path, "source_dataset_path": source.path], to: marker)
        }
    }
    private static func workspaceLock(_ root: URL) throws -> Int32 {
        let path = root.appendingPathComponent(".specialist-lock").path
        let descriptor = open(path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw StudioError("The specialist workspace cannot be locked.") }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor); throw StudioError("Another specialist operation is running. Wait for it to finish.")
        }
        return descriptor
    }
    private static func parse(_ arguments: [String]) throws -> [String: String] {
        var result: [String: String] = [:], offset = 1
        while offset < arguments.count {
            let flag = arguments[offset]
            guard flag.hasPrefix("--") else { throw StudioError("Unexpected specialist argument: \(flag)") }
            if ["--approve", "--unapprove"].contains(flag) { result[flag] = "true"; offset += 1; continue }
            guard offset + 1 < arguments.count, !arguments[offset + 1].hasPrefix("--") else {
                throw StudioError("A value is required for \(flag).")
            }
            result[flag] = arguments[offset + 1]; offset += 2
        }
        return result
    }
    private static func save(_ value: Registry, workspace: URL) throws {
        let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase; encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(value).write(to: workspace.appendingPathComponent(registryFilename), options: .atomic)
    }
    private static func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase; encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
    private static func write(_ value: Object, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]).write(to: url, options: .atomic)
    }
    private static func object(_ url: URL) throws -> Object {
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? Object else { throw StudioError("Expected a native JSON manifest.") }
        return value
    }
    private static func object(_ text: String) throws -> Object {
        guard let value = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? Object else { throw StudioError("Expected a native JSON object.") }
        return value
    }
    private static func canonical(_ url: URL) -> URL { url.resolvingSymlinksInPath().standardizedFileURL }
    private static func datasetRoot(_ url: URL) -> URL {
        let root = canonical(url); return root.lastPathComponent == "dataset.json" ? root.deletingLastPathComponent() : root
    }
    private static func timestamp() -> String { ISO8601DateFormatter().string(from: Date()) }
    private static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private static func fileHash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        var value = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            try Task.checkCancellation(); value.update(data: data)
        }
        return value.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private static func event(_ message: String, completed: Int, total: Int) throws -> String {
        let value: Object = ["event": "specialist_progress", "message": message, "completed": completed, "total": total]
        return String(decoding: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]), as: UTF8.self) + "\n"
    }
}
