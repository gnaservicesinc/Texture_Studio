import AppKit
import UniformTypeIdentifiers

extension WorkbenchStore {
    var datasetFolderURL: URL? {
        guard let dataset else { return nil }
        let url = URL(fileURLWithPath: dataset.sourceFolderPath ?? dataset.preparation?.sourceDatasetPath ?? dataset.datasetPath)
        return url.lastPathComponent == "dataset.json" ? url.deletingLastPathComponent() : url
    }
    var datasetName: String {
        dataset?.name.flatMap { $0.isEmpty ? nil : $0 } ?? datasetFolderURL?.lastPathComponent ?? "Dataset"
    }
    var datasetDescription: String { dataset?.description ?? "" }
    var datasetResolution: Int { dataset?.trainingSize ?? 2048 }
    var datasetSourceTierCount: Int { dataset?.sourceTierCount ?? 1 }

    func rememberDataset(_ result: WorkbenchDataset) {
        let source = URL(fileURLWithPath: result.sourceFolderPath ?? result.preparation?.sourceDatasetPath ?? result.datasetPath)
        let folder = source.lastPathComponent == "dataset.json" ? source.deletingLastPathComponent() : source
        let path = folder.resolvingSymlinksInPath().standardizedFileURL.path
        recentDatasets.removeAll { $0.path == path }
        recentDatasets.insert(WorkbenchDatasetLocation(name: result.name ?? folder.lastPathComponent, path: path), at: 0)
        recentDatasets = Array(recentDatasets.prefix(30))
        saveDatasetLibrary()
    }
    func forgetDataset(_ location: WorkbenchDatasetLocation) {
        recentDatasets.removeAll { $0.id == location.id }
        saveDatasetLibrary()
    }
    private func saveDatasetLibrary() {
        if let data = try? JSONEncoder().encode(recentDatasets) { preferences.set(data, forKey: "recentDatasets.v1") }
    }

    func chooseDatasetParent(selected: URL? = nil, completion: @escaping (URL) -> Void) {
        guard !isBusy else { return }
        let panel = NSOpenPanel()
        panel.title = "Choose where to save your new dataset"
        panel.prompt = "Choose Location"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = selected
        panel.begin { response in
            if response == .OK, let url = panel.url { completion(url) }
        }
    }
    func chooseMaterialMap(title: String, selected: URL? = nil, completion: @escaping (URL) -> Void) {
        guard !isBusy else { return }
        let panel = NSOpenPanel()
        panel.title = title
        panel.message = "Choose an original PNG map. Files are referenced at their original dimensions and precision."
        panel.allowedContentTypes = [.png]
        panel.directoryURL = selected?.deletingLastPathComponent()
        panel.begin { response in
            if response == .OK, let url = panel.url { completion(url) }
        }
    }
    func createDataset(name: String, description: String, parentURL: URL, size: Int? = nil, sourceTierCount: Int = 1, validation: WorkbenchValidationSettings = .init(), sourceFolder: Bool = true) {
        guard !isBusy else { return }
        if sourceFolder, let existing = pendingSourceFolder {
            pendingSourceFolder = nil; pendingAdditionalSourceFolders = []
            showNewDatasetSheet = false
            openDataset(existing)
            return
        }
        let title = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { error = "Give your dataset a name."; return }
        let folderName = title.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
        guard folderName != ".", folderName != ".." else { error = "Choose a descriptive dataset name."; return }
        let destination = parentURL.appendingPathComponent(folderName, isDirectory: true)
        operation("Creating \(title)…") {
            let result = try WorkbenchResult.decode(WorkbenchDataset.self, output: await self.worker([
                "create-dataset", "--dataset", destination.path, "--name", title, "--description", description,
                "--training-size", String(size ?? 2048), "--source-tier-count", String(sourceTierCount)]
                + (sourceFolder ? ["--source-folder", "true"] : []) + validation.arguments))
            self.selectedSampleId = nil
            self.adoptDataset(result)
            if sourceFolder {
                self.classifierUseSourceFolder = false
                self.specialistTrainingContext = nil
                self.useOpenSourcesInModelWorkshop()
            }
            self.saveDatasetLocation(result)
            self.datasetPreparationSummary = ""
            self.showNewDatasetSheet = false
            self.activity = "Created \(self.datasetName). Add a source folder to get started."
            if let folder = self.pendingSourceFolder {
                self.pendingSourceFolder = nil
                let folders = [folder] + self.pendingAdditionalSourceFolders
                self.pendingAdditionalSourceFolders = []
                try await self.registerSourceFolders(folders)
            } else if !self.pendingAdditionalSourceFolders.isEmpty {
                let files = self.pendingAdditionalSourceFolders
                self.pendingAdditionalSourceFolders = []
                try await self.registerSourceFiles(files)
            }
        }
    }

    /// Lifecycle changes always apply to the durable source dataset, even when
    /// the trainer has selected a temporary prepared view.
    private func sourceDatasetForManagement() async throws -> WorkbenchDataset {
        guard let current = dataset else { throw StudioError("Create or open a dataset first.") }
        if let preparation = current.preparation {
            let sourceURL = URL(fileURLWithPath: preparation.sourceDatasetPath)
            if current.isSourceFolder == true {
                _ = try await worker(["cleanup-size", "--dataset", current.datasetPath])
                try await loadDataset(sourceURL, recoverSources: false)
                return dataset!
            }
            let indexURL = sourceURL.lastPathComponent == "dataset.json" ? sourceURL : sourceURL.appendingPathComponent("dataset.json")
            guard NativeMaterialTrainer.checksum(try Data(contentsOf: indexURL)) == preparation.sourceIndexSha256 else {
                throw StudioError("The original dataset changed since preparation. Review its current information before editing.")
            }
            _ = try await worker(["cleanup-size", "--dataset", current.datasetPath])
            try await loadDataset(sourceURL, recoverSources: false)
            guard dataset?.indexSha256 == preparation.sourceIndexSha256 else {
                throw StudioError("The original dataset changed since preparation. Review its current information before editing.")
            }
        }
        return dataset!
    }
    @discardableResult private func adoptManagedDataset(_ output: String) throws -> WorkbenchDataset {
        let result = try WorkbenchResult.decode(WorkbenchDataset.self, output: output)
        adoptDataset(result)
        saveDatasetLocation(result)
        datasetPreparationSummary = ""
        return result
    }
    func updateDatasetInfo(name: String, description: String, size: Int? = nil, sourceTierCount: Int? = nil, validation: WorkbenchValidationSettings? = nil) {
        guard !isBusy else { return }
        let title = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { error = "Give your dataset a name."; return }
        operation("Saving dataset info…") {
            let current = try await self.sourceDatasetForManagement()
            try self.adoptManagedDataset(await self.worker(["edit-dataset", "--dataset", current.datasetPath,
                "--name", title, "--description", description, "--expected-index-sha256", current.indexSha256,
                "--training-size", String(size ?? self.datasetResolution), "--review-size", String(size ?? self.datasetResolution),
                "--source-tier-count", String(sourceTierCount ?? self.datasetSourceTierCount)] + (validation ?? self.dataset?.validation ?? .init()).arguments))
            self.showDatasetInfoSheet = false
            self.activity = "Saved \(self.datasetName)."
        }
    }
    func addMaterial(name: String, input: URL, height: URL?, roughness: URL?, normal: URL?, normalConvention: String = "opengl") {
        guard !isBusy else { return }
        operation("Adding material maps…") {
            let current = try await self.sourceDatasetForManagement()
            var chosen = ["input": input]
            chosen["height"] = height; chosen["roughness"] = roughness; chosen["normal"] = normal
            if current.isSourceFolder == true {
                let root = URL(fileURLWithPath: current.sourceFolderPath ?? current.datasetPath)
                var folderName = name.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
                if FileManager.default.fileExists(atPath: root.appendingPathComponent(folderName).path), chosen.values.contains(where: { !$0.standardizedFileURL.path.hasPrefix(root.path + "/") }) {
                    folderName += " " + String(UUID().uuidString.prefix(6))
                }
                for (role, original) in chosen where !original.standardizedFileURL.path.hasPrefix(root.path + "/") {
                    let output = try await self.worker(["add-source", "--dataset", root.path, "--source", original.path, "--folder-name", folderName, "--defer-refresh", "true"])
                    guard let reply = try JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any],
                          let path = reply["added_source_path"] as? String else { throw StudioError("The added map's location was not returned.") }
                    chosen[role] = URL(fileURLWithPath: path)
                }
            }
            var args = ["add-material", "--dataset", current.datasetPath, "--name", name, "--input", chosen["input"]!.path,
                        "--normal-convention", normalConvention, "--expected-index-sha256", current.indexSha256,
                        "--review-size", String(self.training.size)]
            for role in ["height", "roughness", "normal"] { if let file = chosen[role] { args += ["--" + role, file.path] } }
            let previous = Set(self.samples.map(\.id))
            let result = try self.adoptManagedDataset(await self.worker(args))
            self.selectedSampleId = self.samples.first { !previous.contains($0.id) }?.id ?? self.selectedSampleId
            self.showAddMaterialSheet = false
            self.activity = result.addedMaterialCount == 0 && (result.duplicateMaterialCount ?? 0) > 0
                ? "These original maps are already in this dataset."
                : "Added material. Original maps are referenced without copying or rescaling."
        }
    }
    func importMaterialFolder() {
        guard !isBusy, dataset != nil else { return }
        let panel = NSOpenPanel()
        panel.title = "Import Material Folder"
        panel.message = "Choose a folder containing paired diffuse and target PNG maps. Original files stay in this folder."
        panel.prompt = "Import Materials"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.begin { response in
            if response == .OK, let url = panel.url { self.importMaterialFolder(url) }
        }
    }
    func addMaterialFolder(workshop: Bool = false) {
        guard !isBusy else { return }
        let panel = NSOpenPanel()
        panel.title = "Add Images or Folders"
        panel.message = "Add photos or material folders. Copies keep their original pixels and precision."
        panel.prompt = "Add Sources"
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = true
        panel.begin { response in
            if response == .OK {
                if workshop { self.addModelWorkshopSourceFiles(panel.urls) }
                else { self.addMaterialFolders(panel.urls) }
            }
        }
    }
    func addMaterialFolders(_ urls: [URL]) {
        guard !isBusy, !urls.isEmpty else { return }
        if dataset == nil, urls.count == 1, let folder = urls.first,
           (try? folder.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true,
           !FileManager.default.fileExists(atPath: folder.appendingPathComponent("dataset.json").path) {
            openDataset(folder)
            return
        }
        if dataset == nil {
            pendingSourceFolder = nil
            pendingAdditionalSourceFolders = urls
            showNewDatasetSheet = true
            return
        }
        if dataset?.isSourceFolder == true { addSourceFiles(urls); return }
        operation("Adding source folders…") { try await self.registerSourceFolders(urls) }
    }
    func registerSourceFolders(_ urls: [URL]) async throws {
        let previous = Set(datasetBrowserSamples.map(\.id))
        var addedFiles = 0
        var addedSets = 0
        for url in urls {
            try Task.checkCancellation()
            let current = try await sourceDatasetForManagement()
            let result = try adoptManagedDataset(await worker([
                "import-folder", "--dataset", current.datasetPath, "--folder", url.path,
                "--expected-index-sha256", current.indexSha256, "--managed-import", "true",
                "--review-size", String(datasetResolution), "--skip-source-recovery"]))
            addedFiles += result.addedFileCount ?? 0
            addedSets += result.addedMaterialCount ?? 0
        }
        datasetBrowserMode = .originals
        selectedSampleId = datasetBrowserSamples.first { !previous.contains($0.id) }?.id ?? selectedSampleId
        showAddMaterialSheet = false
        showImportFolderSheet = false
        activity = "Added \(addedFiles) original files and \(addedSets) paired sets. Every original remains available in Originals."
    }
    func importMaterialFolder(_ url: URL) {
        guard !isBusy else { return }
        if dataset == nil { pendingSourceFolder = url; showNewDatasetSheet = true; return }
        operation("Scanning material folders…") { try await self.scanMaterialFolder(url) }
    }
    private func scanMaterialFolder(_ url: URL, size: Int? = nil) async throws {
        let continuingImport = showImportFolderSheet
        let source = try await sourceDatasetForManagement()
        // A rescan is also recovery from another window changing membership or
        // reviews. Reconnect the durable index before binding a new preview.
        if continuingImport { try await loadDataset(URL(fileURLWithPath: source.datasetPath), recoverSources: false) }
        guard let current = dataset else { throw StudioError("Open a dataset before importing material maps.") }
        clearFolderImport()
        if !continuingImport { folderImportSize = datasetResolution }
        if let size { folderImportSize = size }
        let scannedSize = folderImportSize
        sourceRecoverySummary = ""
        sourceRecoveryResults = []
        folderImportURL = url
        isScanningFolder = true
        defer { isScanningFolder = false }
        showAddMaterialSheet = false
        showImportFolderSheet = true
        let planURL = FileManager.default.temporaryDirectory.appendingPathComponent("material-import-\(UUID().uuidString).json")
        folderImportPlanURL = planURL
        folderImport = try WorkbenchResult.decode(WorkbenchFolderImport.self, output: await worker([
            "scan-folder", "--dataset", current.datasetPath, "--folder", url.path, "--size", String(scannedSize), "--target", training.target,
            "--expected-index-sha256", current.indexSha256, "--plan", planURL.path, "--quick-preview", "true"]))
        folderImportScannedSize = scannedSize
        sourceRecoverySummary = folderImport?.sourceRecovery?.summary ?? ""
        sourceRecoveryResults = folderImport?.sourceRecoveryResults ?? []
        activity = "Folder scan complete at \(scannedSize) × \(scannedSize). Review the materials, then import."
    }
    func clearFolderImport() {
        if let folderImportPlanURL { try? FileManager.default.removeItem(at: folderImportPlanURL) }
        folderImportPlanURL = nil
        folderImport = nil
        folderImportScannedSize = nil
    }
    func commitFolderImport(size: Int) {
        guard !isBusy, let preview = folderImport, dataset != nil else { return }
        guard preview.addedMaterialCount > 0 else { return }
        operation("Verifying and importing original material maps…") {
            if self.folderImportScannedSize != size {
                try await self.scanMaterialFolder(URL(fileURLWithPath: preview.folderPath), size: size)
            }
            guard let preview = self.folderImport, let current = self.dataset else { throw StudioError("Scan the material folder before importing.") }
            let previous = Set(self.samples.map(\.id))
            let result = try self.adoptManagedDataset(await self.worker(["import-folder", "--dataset", current.datasetPath,
                "--folder", preview.folderPath, "--expected-index-sha256", preview.indexSha256,
                "--plan", preview.planPath, "--expected-plan-sha256", preview.planSha256,
                "--training-size", String(size), "--review-size", String(size)]))
            self.selectedSampleId = self.samples.first { !previous.contains($0.id) }?.id ?? self.selectedSampleId
            self.showImportFolderSheet = false
            self.sourceRecoverySummary = preview.sourceRecovery?.summary ?? ""
            self.sourceRecoveryResults = preview.sourceRecoveryResults ?? []
            self.clearFolderImport()
            self.activity = "Imported \(result.addedMaterialCount ?? 0) source sets at \(size) × \(size); \(preview.duplicateMaterialCount) already present. Original maps remain in their source folder."
        }
    }
    func setSubjectValidation(_ subject: WorkbenchSubject, enabled: Bool) {
        guard !isBusy else { return }
        operation("Saving folder validation…") {
            let current = try await self.sourceDatasetForManagement()
            try self.adoptManagedDataset(await self.worker(["edit-dataset", "--dataset", current.datasetPath,
                "--expected-index-sha256", current.indexSha256, "--validation-subject", subject.id,
                "--subject-validation", enabled ? "enabled" : "disabled"]))
            self.activity = "Saved shared folder validation for every material target."
        }
    }
    func removeSelectedMaterial() {
        guard !isBusy, let material = selectedMaterialId else { return }
        operation("Removing \(material) from dataset…") {
            let current = try await self.sourceDatasetForManagement()
            try self.adoptManagedDataset(await self.worker(["remove-material", "--dataset", current.datasetPath,
                "--material", material, "--expected-index-sha256", current.indexSha256, "--review-size", String(self.training.size)]))
            self.activity = "Removed material from dataset. Original maps are kept."
        }
    }
    func trashDataset() {
        guard !isBusy, dataset != nil else { return }
        operation("Moving dataset metadata to Trash…") {
            let current = try await self.sourceDatasetForManagement()
            let plan = try WorkbenchResult.decode(WorkbenchDatasetDeletion.self, output: await self.worker([
                "validate-delete", "--dataset", current.datasetPath, "--expected-index-sha256", current.indexSha256]))
            try DatasetFileOperations.trash(plan, expectedDataset: URL(fileURLWithPath: current.datasetPath), handler: self.trashHandler)
            let folder = URL(fileURLWithPath: current.datasetPath).resolvingSymlinksInPath().standardizedFileURL.path
            self.recentDatasets.removeAll { $0.path == folder }
            self.saveDatasetLibrary()
            self.closeDataset()
            self.showTrashDatasetConfirmation = false
            self.activity = "Dataset metadata moved to Trash. Original source maps are kept."
        }
    }
    func revealDataset() {
        if let folder = datasetFolderURL { NSWorkspace.shared.activateFileViewerSelecting([folder]) }
    }
    func closeDataset() {
        // A lifecycle operation can close its own dataset after a successful
        // Trash transaction; commands disable this action while work runs.
        dataset = nil
        sourceFolderObserver = nil
        datasetBrowserMode = .originals
        selectedSampleId = nil
        selectedInputVariantId = nil
        datasetPreparationSummary = ""
        preferences.removeObject(forKey: "dataset")
    }
}
