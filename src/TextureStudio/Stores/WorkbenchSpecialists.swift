import AppKit

struct WorkbenchSpecialistRegistry: Decodable, Sendable {
    struct Assignment: Decodable, Sendable, Identifiable {
        let familyId: String
        let inputPath: String
        let classLabel: String
        var subtype: String? = nil
        var semanticTerms: [String]? = nil
        var confidence: Double? = nil
        var source: String? = nil
        var id: String { inputPath }
        var tags: [String] { [classLabel] + (subtype.map { [$0] } ?? []) }
    }
    struct Target: Decodable, Sendable {
        let target: String
        let materialCount: Int
        let status: String
        let membershipSha256: String
    }
    struct Model: Decodable, Sendable, Identifiable {
        let id: String
        let target: String
        let checkpointPath: String
        let checkpointSha256: String
        let trainingSize: Int
        let completedAt: String
        let membershipSha256: String
        var approved: Bool? = nil
    }
    struct Specialist: Decodable, Sendable, Identifiable {
        let id: String
        let name: String
        let classLabel: String
        let subtype: String?
        let isCustom: Bool
        let datasetPath: String
        let materialIds: [String]
        let familyIds: [String]
        let targets: [Target]
        let models: [Model]
        var assignments: [Assignment]? = nil
        func latestModel(for target: String) -> Model? { models.last(where: { $0.target == target }) }
        func targetInfo(_ target: String) -> Target? { targets.first(where: { $0.target == target }) }
    }
    let workspacePath: String
    let sourceDatasetPath: String
    let classifierDatasetPath: String?
    let classifierPath: String?
    let classifierSha256: String
    let classifierLabels: [String]
    let updatedAt: String
    let specialists: [Specialist]
    let classifierOnlyCount: Int
    let materialCount: Int
    let familyCount: Int
    let warnings: [String]
}

struct WorkbenchSpecialistTrainingContext: Sendable {
    let workspacePath: String
    let specialistID: String
    let name: String
    let target: String
    let datasetPath: String
    let membershipSHA256: String
}

extension WorkbenchStore {
    var modelWorkshopSourcePath: String {
        let categorySelected = dataset.map { current in
            specialistRegistry?.specialists.contains { URL(fileURLWithPath: $0.datasetPath).standardizedFileURL.path == URL(fileURLWithPath: current.datasetPath).standardizedFileURL.path } ?? false
        } ?? false
        if specialistTrainingContext == nil, !categorySelected, let dataset {
            return dataset.sourceFolderPath ?? dataset.preparation?.sourceDatasetPath ?? dataset.datasetPath
        }
        let remembered = specialistSourceDatasetPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return remembered.isEmpty ? specialistRegistry?.sourceDatasetPath ?? "" : remembered
    }

    var modelWorkshopNeedsOriginalSources: Bool {
        guard let dataset, !modelWorkshopSourcePath.isEmpty else { return false }
        return specialistTrainingContext != nil || URL(fileURLWithPath: dataset.datasetPath).standardizedFileURL.path != URL(fileURLWithPath: modelWorkshopSourcePath).standardizedFileURL.path
    }

    func restoreModelWorkshopSourcesIfNeeded() async throws {
        let source = modelWorkshopSourcePath
        guard !source.isEmpty else { return }
        if modelWorkshopNeedsOriginalSources {
            try await loadDataset(URL(fileURLWithPath: source), recoverSources: false)
        }
        specialistTrainingContext = nil
        specialistSourceDatasetPath = source
        datasetBrowserMode = .originals
    }

    func returnToModelWorkshop() {
        guard !isBusy else { return }
        guard modelWorkshopNeedsOriginalSources else { specialistNavigationRequest = UUID(); return }
        operation("Opening original source folders…") {
            try await self.restoreModelWorkshopSourcesIfNeeded()
            self.specialistNavigationRequest = UUID()
            self.activity = "Original folders are ready in Loom Room."
        }
    }

    func addModelWorkshopSourceFiles(_ urls: [URL]) {
        guard !isBusy, !urls.isEmpty else { return }
        guard dataset != nil else { addMaterialFolders(urls); return }
        operation("Adding original images and folders…") {
            try await self.restoreModelWorkshopSourcesIfNeeded()
            if self.dataset?.isSourceFolder == true { try await self.registerSourceFiles(urls) }
            else { try await self.registerSourceFolders(urls) }
        }
    }

    var modelWorkshopSourceFolders: [String] {
        let materials = dataset?.originalMaterials ?? dataset?.materials ?? []
        let paths = materials.compactMap(\.sourceDirectory) + (dataset?.sourceInventory ?? []).map(\.folder)
        return Array(Set(paths)).sorted { $0.localizedStandardCompare($1) == .orderedAscending }
    }

    var modelWorkshopImageTags: [WorkbenchSpecialistRegistry.Assignment] {
        var seen = Set<String>()
        return (specialistRegistry?.specialists ?? []).flatMap { $0.assignments ?? [] }
            .filter { seen.insert($0.inputPath).inserted }
            .sorted { $0.familyId.localizedStandardCompare($1.familyId) == .orderedAscending }
    }

    var modelWorkshopTaggedOriginals: [WorkbenchSourceInventoryItem] {
        (dataset?.sourceInventory ?? []).filter { !($0.classLabel?.isEmpty ?? true) }
            .sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
    }

    func useOpenSourcesInModelWorkshop() {
        guard let dataset else { return }
        specialistSourceDatasetPath = dataset.sourceFolderPath ?? dataset.preparation?.sourceDatasetPath ?? dataset.datasetPath
    }

    func openModelWorkshopSources() {
        guard !isBusy else { return }
        let source = modelWorkshopSourcePath
        guard !source.isEmpty else { chooseDataset(); return }
        if modelWorkshopNeedsOriginalSources {
            operation("Opening original source folders…") {
                try await self.restoreModelWorkshopSourcesIfNeeded()
                self.datasetNavigationRequest = UUID()
                self.activity = "Original images are ready for review and tagging."
            }
            return
        }
        specialistTrainingContext = nil
        datasetBrowserMode = .originals
        datasetNavigationRequest = UUID()
    }

    var selectedSpecialist: WorkbenchSpecialistRegistry.Specialist? {
        specialistRegistry?.specialists.first(where: { $0.id == selectedSpecialistID })
            ?? specialistRegistry?.specialists.first
    }

    func loadSpecialistRegistry() {
        let url = URL(fileURLWithPath: specialistWorkspacePath).appendingPathComponent("specialists.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self,
                output: String(contentsOf: url, encoding: .utf8))
        } catch { self.error = "Could not open the model workspace: \(error.localizedDescription)" }
    }

    func refreshSpecialists() {
        guard !isBusy else { return }
        let source = modelWorkshopSourcePath
        guard !source.isEmpty else { error = "Open your original source folder in Loom Room first."; return }
        specialistSourceDatasetPath = source
        var args = ["refresh-specialists", "--dataset", source, "--workspace", specialistWorkspacePath,
                    "--classifier-dataset", classifierDatasetPath]
        let classifier = specialistClassifierPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !classifier.isEmpty { args += ["--classifier", classifier] }
        let originals = classifierSourceFolderPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if classifierUseSourceFolder, !originals.isEmpty { args += ["--source-folder", originals] }
        operation("Organizing image tags and map training datasets…") {
            self.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self,
                output: await self.worker(args))
            if self.selectedSpecialistID == nil { self.selectedSpecialistID = self.specialistRegistry?.specialists.first?.id }
            let registry = self.specialistRegistry!
            self.activity = "\(registry.specialists.count) model categories · \(registry.familyCount) source families. Choose a category and map to train."
            if !registry.warnings.isEmpty { self.logText += "\n" + registry.warnings.joined(separator: "\n") }
        }
    }

    func openSpecialistTrainer(continuing: Bool = false) {
        guard let specialist = selectedSpecialist,
              (specialist.targetInfo(specialistTarget)?.materialCount ?? 0) > 0 else {
            error = "This category needs paired \(specialistTarget) maps before training."; return
        }
        specialistSourceDatasetPath = modelWorkshopSourcePath
        let target = specialistTarget, workspace = specialistWorkspacePath
        operation("Opening \(specialist.name) training dataset…") {
            let family: MaterialTrainingModelFamily = target == "normal" ? .compactNormal : .compactScalar
            if self.training.modelFamily != family { self.training.learningRate = 0.0003 }
            self.training.modelFamily = family
            self.training.target = target
            self.training.scope = "full-model"
            self.training.useSelectedMaterialOnly = false
            self.training.useWarmStart = false
            self.training.modelName = "\(specialist.name) \(Self.specialistMapTitle(target))"
            try await self.loadTrainingCapabilities()
            try await self.loadDataset(URL(fileURLWithPath: specialist.datasetPath))
            self.selectedRole = target
            if continuing, let saved = specialist.latestModel(for: target) {
                try await self.loadCheckpoint(URL(fileURLWithPath: saved.checkpointPath))
                if self.selectedCheckpoint?.matchesTraining(self.training) == true { self.training.useWarmStart = true }
                else { throw StudioError("The saved specialist checkpoint cannot continue this model family and map.") }
            }
            self.specialistTrainingContext = .init(workspacePath: workspace, specialistID: specialist.id,
                name: specialist.name, target: target, datasetPath: specialist.datasetPath,
                membershipSHA256: specialist.targetInfo(target)?.membershipSha256 ?? "")
            self.trainingNavigationRequest = UUID()
            self.activity = "\(specialist.name) is ready. Review the training settings, then choose Start Training."
        }
    }

    func recordSpecialistCheckpoint(context: WorkbenchSpecialistTrainingContext,
                                   checkpointPath: String, trainingSize: Int, completedUpdates: Int) async throws {
        specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self,
            output: await worker(["record-specialist-model", "--workspace", context.workspacePath,
                "--specialist", context.specialistID, "--target", context.target,
                "--checkpoint", checkpointPath, "--size", String(trainingSize),
                "--completed-updates", String(completedUpdates), "--membership-sha256", context.membershipSHA256]))
    }

    func compareSpecialistModel() {
        guard let specialist = selectedSpecialist, let saved = specialist.latestModel(for: specialistTarget) else { return }
        specialistSourceDatasetPath = modelWorkshopSourcePath
        let target = specialistTarget
        operation("Opening \(specialist.name) comparison…") {
            self.training.target = target
            self.training.modelFamily = target == "normal" ? .compactNormal : .compactScalar
            try await self.loadDataset(URL(fileURLWithPath: specialist.datasetPath))
            try await self.loadCheckpoint(URL(fileURLWithPath: saved.checkpointPath))
            self.comparisonCheckpointIds = [self.selectedCheckpointId!]
            self.comparisonIncludesBase = false
            self.comparisonNavigationRequest = UUID()
            self.activity = "Compare the saved \(specialist.name) maps with their paired references."
        }
    }

    func approveSpecialistModel(approved: Bool) {
        guard let specialist = selectedSpecialist, let saved = specialist.latestModel(for: specialistTarget) else { return }
        operation(approved ? "Approving specialist model…" : "Removing specialist approval…") {
            self.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self,
                output: await self.worker(["approve-specialist-model", "--workspace", self.specialistWorkspacePath,
                    "--specialist", specialist.id, "--target", saved.target,
                    "--model", saved.id, approved ? "--approve" : "--unapprove"]))
            self.activity = approved ? "\(specialist.name) \(Self.specialistMapTitle(saved.target)) is available for automatic generation." : "Specialist approval removed."
        }
    }

    func addSpecialist(name: String, classLabel: String, subtype: String) {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { error = "Enter a name for the specialist."; return }
        var args = ["add-specialist", "--workspace", specialistWorkspacePath, "--name", name, "--class-label", classLabel]
        let subtype = subtype.trimmingCharacters(in: .whitespacesAndNewlines)
        if !subtype.isEmpty { args += ["--subtype", subtype] }
        operation("Adding \(name) specialist…") {
            self.specialistRegistry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self, output: await self.worker(args))
            self.selectedSpecialistID = self.specialistRegistry?.specialists.last(where: { $0.name == name })?.id
            self.activity = "\(name) added. Refresh specialist datasets to update its matching materials."
        }
    }

    func selectNextSpecialist() {
        guard let items = specialistRegistry?.specialists, !items.isEmpty else { return }
        let index = items.firstIndex(where: { $0.id == selectedSpecialist?.id }) ?? -1
        let ordered = Array(items.dropFirst(index + 1)) + Array(items.prefix(index + 1))
        selectedSpecialistID = ordered.first(where: {
            let model = $0.latestModel(for: specialistTarget)
            let ready = model?.approved == true && model?.membershipSha256 == $0.targetInfo(specialistTarget)?.membershipSha256
            return ($0.targetInfo(specialistTarget)?.materialCount ?? 0) > 0 && !ready
        })?.id ?? ordered.first?.id
    }

    func chooseSpecialistPath(_ kind: String) {
        let panel = NSOpenPanel()
        panel.title = kind == "classifier" ? "Choose exported diffuse classifier" : kind == "source" ? "Choose original material dataset" : "Choose specialist workspace"
        panel.canChooseFiles = kind == "classifier"
        panel.canChooseDirectories = true
        panel.canCreateDirectories = kind == "workspace"
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        switch kind {
        case "classifier": specialistClassifierPath = url.path
        case "source": specialistSourceDatasetPath = url.path
        default: specialistWorkspacePath = url.path; loadSpecialistRegistry()
        }
    }

    func recordSpecialistProgress(_ text: String) {
        for line in text.split(separator: "\n") {
            guard let event = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  ["specialist_progress", "material_specialist_progress"].contains(event["event"] as? String ?? ""), let message = event["message"] as? String else { continue }
            activity = message
        }
    }

    static func specialistMapTitle(_ target: String) -> String {
        ["height": "Displacement", "roughness": "Roughness", "normal": "Normals"][target] ?? target.capitalized
    }
}
