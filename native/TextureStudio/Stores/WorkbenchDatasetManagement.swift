import AppKit
import UniformTypeIdentifiers

extension WorkbenchStore {
    var datasetFolderURL: URL? {
        guard let dataset else { return nil }
        let url = URL(fileURLWithPath: dataset.preparation?.sourceDatasetPath ?? dataset.datasetPath)
        return url.lastPathComponent == "dataset.json" ? url.deletingLastPathComponent() : url
    }
    var datasetName: String {
        dataset?.name.flatMap { $0.isEmpty ? nil : $0 } ?? datasetFolderURL?.lastPathComponent ?? "Dataset"
    }
    var datasetDescription: String { dataset?.description ?? "" }
    var datasetResolution: Int { dataset?.trainingSize ?? training.size }

    func rememberDataset(_ result: WorkbenchDataset) {
        let source = URL(fileURLWithPath: result.preparation?.sourceDatasetPath ?? result.datasetPath)
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
    func createDataset(name: String, description: String, parentURL: URL, size: Int? = nil) {
        guard !isBusy else { return }
        let title = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { error = "Give your dataset a name."; return }
        let folderName = title.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
        guard folderName != ".", folderName != ".." else { error = "Choose a descriptive dataset name."; return }
        let destination = parentURL.appendingPathComponent(folderName, isDirectory: true)
        operation("Creating \(title)…") {
            let result = try WorkbenchProcess.decode(WorkbenchDataset.self, output: await self.worker([
                "create-dataset", "--dataset", destination.path, "--name", title, "--description", description,
                "--training-size", String(size ?? self.training.size)]))
            self.selectedSampleId = nil
            self.adoptDataset(result)
            self.saveDatasetLocation(result)
            self.datasetPreparationSummary = ""
            self.showNewDatasetSheet = false
            self.activity = "Created \(self.datasetName). Add material maps to get started."
            if let folder = self.pendingSourceFolder {
                self.pendingSourceFolder = nil
                try await self.scanMaterialFolder(folder)
            } else { self.showAddMaterialSheet = true }
        }
    }

    /// Lifecycle changes always apply to the durable source dataset, even when
    /// the trainer has selected a temporary prepared view.
    private func sourceDatasetForManagement() async throws -> WorkbenchDataset {
        guard let current = dataset else { throw StudioError("Create or open a dataset first.") }
        if let preparation = current.preparation {
            _ = try await worker(["cleanup-size", "--dataset", current.datasetPath])
            try await loadDataset(URL(fileURLWithPath: preparation.sourceDatasetPath))
            guard dataset?.indexSha256 == preparation.sourceIndexSha256 else {
                throw StudioError("The original dataset changed since preparation. Review its current information before editing.")
            }
        }
        return dataset!
    }
    @discardableResult private func adoptManagedDataset(_ output: String) throws -> WorkbenchDataset {
        let result = try WorkbenchProcess.decode(WorkbenchDataset.self, output: output)
        adoptDataset(result)
        saveDatasetLocation(result)
        datasetPreparationSummary = ""
        return result
    }
    func updateDatasetInfo(name: String, description: String, size: Int? = nil) {
        guard !isBusy else { return }
        let title = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { error = "Give your dataset a name."; return }
        operation("Saving dataset info…") {
            let current = try await self.sourceDatasetForManagement()
            try self.adoptManagedDataset(await self.worker(["edit-dataset", "--dataset", current.datasetPath,
                "--name", title, "--description", description, "--expected-index-sha256", current.indexSha256,
                "--training-size", String(size ?? self.datasetResolution), "--review-size", String(size ?? self.datasetResolution)]))
            self.showDatasetInfoSheet = false
            self.activity = "Saved \(self.datasetName)."
        }
    }
    func addMaterial(name: String, input: URL, height: URL?, roughness: URL?, normal: URL?, normalConvention: String = "opengl") {
        guard !isBusy else { return }
        operation("Adding material maps…") {
            let current = try await self.sourceDatasetForManagement()
            var args = ["add-material", "--dataset", current.datasetPath, "--name", name, "--input", input.path,
                        "--normal-convention", normalConvention, "--expected-index-sha256", current.indexSha256,
                        "--review-size", String(self.training.size)]
            if let height { args += ["--height", height.path] }
            if let roughness { args += ["--roughness", roughness.path] }
            if let normal { args += ["--normal", normal.path] }
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
    func importMaterialFolder(_ url: URL) {
        guard !isBusy else { return }
        if dataset == nil { pendingSourceFolder = url; showNewDatasetSheet = true; return }
        operation("Scanning material folders and verifying original maps…") { try await self.scanMaterialFolder(url) }
    }
    private func scanMaterialFolder(_ url: URL) async throws {
        let source = try await sourceDatasetForManagement()
        // A rescan is also recovery from another window changing membership or
        // reviews. Reconnect the durable index before binding a new preview.
        try await loadDataset(URL(fileURLWithPath: source.datasetPath))
        guard let current = dataset else { throw StudioError("Open a dataset before importing material maps.") }
        clearFolderImport()
        folderImportURL = url
        showAddMaterialSheet = false
        showImportFolderSheet = true
        let planURL = FileManager.default.temporaryDirectory.appendingPathComponent("material-import-\(UUID().uuidString).json")
        folderImportPlanURL = planURL
        folderImport = try WorkbenchProcess.decode(WorkbenchFolderImport.self, output: await worker([
            "scan-folder", "--dataset", current.datasetPath, "--folder", url.path,
            "--expected-index-sha256", current.indexSha256, "--plan", planURL.path]))
        activity = "Folder scan complete. Review the resolution and split counts, then import."
    }
    func clearFolderImport() {
        if let folderImportPlanURL { try? FileManager.default.removeItem(at: folderImportPlanURL) }
        folderImportPlanURL = nil
        folderImport = nil
    }
    func commitFolderImport(size: Int) {
        guard !isBusy, let preview = folderImport, let current = dataset else { return }
        guard preview.addedMaterialCount > 0 else { return }
        operation("Registering verified original material maps…") {
            guard current.indexSha256 == preview.indexSha256 else { throw StudioError("The dataset changed. Scan the folder again before importing.") }
            let previous = Set(self.samples.map(\.id))
            let result = try self.adoptManagedDataset(await self.worker(["import-folder", "--dataset", current.datasetPath,
                "--folder", preview.folderPath, "--expected-index-sha256", preview.indexSha256,
                "--plan", preview.planPath, "--expected-plan-sha256", preview.planSha256,
                "--training-size", String(size), "--review-size", String(size)]))
            self.selectedSampleId = self.samples.first { !previous.contains($0.id) }?.id ?? self.selectedSampleId
            self.showImportFolderSheet = false
            self.clearFolderImport()
            self.activity = "Imported \(result.addedMaterialCount ?? 0) source sets at \(size) × \(size); \(preview.duplicateMaterialCount) already present. Original maps remain in their source folder."
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
            let plan = try WorkbenchProcess.decode(WorkbenchDatasetDeletion.self, output: await self.worker([
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
        selectedSampleId = nil
        selectedInputVariantId = nil
        datasetPreparationSummary = ""
        preferences.removeObject(forKey: "dataset")
    }
}
