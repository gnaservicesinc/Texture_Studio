import AppKit
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class WorkbenchStore {
    var dataset: WorkbenchDataset?
    var datasetSheet: DatasetSheetRoute?
    var showNewDatasetSheet: Bool {
        get { datasetSheet == .new }
        set { if newValue { datasetSheet = .new } else if datasetSheet == .new { datasetSheet = nil } }
    }
    var showAddMaterialSheet: Bool {
        get { datasetSheet == .add }
        set { if newValue { datasetSheet = .add } else if datasetSheet == .add { datasetSheet = nil } }
    }
    var showDatasetInfoSheet: Bool {
        get { datasetSheet == .info }
        set { if newValue { datasetSheet = .info } else if datasetSheet == .info { datasetSheet = nil } }
    }
    var showImportFolderSheet: Bool {
        get { datasetSheet == .folder }
        set { if newValue { datasetSheet = .folder } else if datasetSheet == .folder { datasetSheet = nil } }
    }
    var pendingSourceFolder: URL?
    var folderImport: WorkbenchFolderImport?
    var folderImportURL: URL?
    @ObservationIgnored var folderImportPlanURL: URL?
    @ObservationIgnored private var queuedDatasetURL: URL?
    var showTrashDatasetConfirmation = false
    var recentDatasets: [WorkbenchDatasetLocation] = []
    var selectedSampleId: String? { didSet { saveUserSettings() } }
    var selectedRole = "height" { didSet { saveUserSettings() } }
    var selectedInputVariantId: String? { didSet { saveUserSettings() } }
    var checkpoints: [WorkbenchCheckpoint] = []
    var selectedCheckpointId: String? { didSet { saveUserSettings() } }
    var comparisonCheckpointIds: Set<String> = [] { didSet { saveUserSettings() } }
    var comparisonIncludesBase = true { didSet { saveUserSettings() } }
    var training = MaterialTrainingOptions() { didSet { saveUserSettings() } }
    var developerMode: Bool {
        get { StudioPreferences.defaults.bool(forKey: StudioPreferences.developerModeKey) }
        set { StudioPreferences.defaults.set(newValue, forKey: StudioPreferences.developerModeKey) }
    }
    var uploadAfterTraining = true { didSet { preferences.set(uploadAfterTraining, forKey: "uploadAfterTraining") } }
    private(set) var supportedTrainingSizes: [Int] = []
    private var backendTrainingSizes: [Int] = []
    private var memoryPlans: [String: WorkbenchMemoryPlan] = [:]
    private(set) var hubModels: [WorkbenchHubModel] = []
    var adapterMix: [WorkbenchAdapterWeight] = []
    var activity = ""
    var logText = ""
    var error: String?
    private(set) var isBusy = false
    private(set) var isTraining = false
    private(set) var isResumingTraining = false
    private(set) var isStopping = false
    private(set) var hasTrainingStarted = false
    private(set) var isSavingTraining = false
    @ObservationIgnored private var trainingEventBuffer = ""
    private(set) var isPreparingDataset = false
    var datasetPreparationSummary = ""
    private(set) var trainingPreferenceNotice: String?
    var lastOutputURL: URL? { didSet { saveUserSettings() } }
    var lastLogURL: URL? { didSet { saveUserSettings() } }
    var lastPackageURL: URL? { didSet { saveUserSettings() } }
    var lastPackageCheckpointId: String? { didSet { saveUserSettings() } }
    var sourceImageURL: URL? { didSet { saveUserSettings() } }
    var testPhotoSettings = TextureSettings()
    var comparisonCandidates: [MapReviewCandidate] = []
    var uploadRepo = "" { didSet { saveUploadConfiguration() } }
    var uploadPublic = false { didSet { saveUploadConfiguration() } }
    private(set) var uploadAccount: String?
    private(set) var uploadAccountMessage = "Checking your saved Hugging Face login…"
    private(set) var uploadAccountChecked = false
    private(set) var lastUploadURL: URL?
    var workspacePath: String { didSet { preferences.set(workspacePath, forKey: "workspace") } }
    var pythonPath: String { didSet { preferences.set(pythonPath, forKey: "python") } }
    var modelDirectory: String { didSet { preferences.set(modelDirectory, forKey: "materialModelDirectory") } }
    var codeDirectory: String { didSet { preferences.set(codeDirectory, forKey: "materialCodeDirectory") } }
    let resources: MachineResources
    @ObservationIgnored private var runner: WorkbenchProcess?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var activeWorkerId: UUID?
    @ObservationIgnored let preferences: UserDefaults
    @ObservationIgnored let trashHandler: (URL) throws -> Void
    @ObservationIgnored private let workerOverride: (@MainActor ([String], String) async throws -> String)?
    @ObservationIgnored private let managedWorkspaceURL: URL
    @ObservationIgnored private let selectedCheckpointRegistryURL: URL
    @ObservationIgnored private var hasRestored = false
    @ObservationIgnored private var isRestoringPreferences = false

    var samples: [WorkbenchSample] { dataset?.samples ?? [] }
    var canStopAndSave: Bool { isTraining && hasTrainingStarted && !isStopping }
    var selectedSample: WorkbenchSample? { samples.first { $0.id == selectedSampleId } }
    var selectedMaterialId: String? { dataset?.materials.first { $0.samples.contains { $0.id == selectedSampleId } }?.materialId }
    var selectedMaterialName: String? {
        dataset?.materials.first { $0.samples.contains { $0.id == selectedSampleId } }.map { $0.name ?? $0.materialId.replacingOccurrences(of: "_", with: " ") }
    }
    var selectedCheckpoint: WorkbenchCheckpoint? { checkpoints.first { $0.id == selectedCheckpointId } }
    var selectedDiffuseMap: WorkbenchMap? {
        selectedSample?.inputVariants?.first { $0.variantId == selectedInputVariantId } ?? selectedSample?.inputVariants?.first ?? selectedSample?.maps["input"]
    }
    var selectedMap: WorkbenchMap? { selectedRole == "input" ? selectedDiffuseMap : selectedSample?.maps[selectedRole] }
    func datasetReviewURL(_ map: WorkbenchMap) -> URL {
        map.originalSourcePath.map { URL(fileURLWithPath: $0) } ?? map.url
    }
    func datasetReviewSHA256(_ map: WorkbenchMap) -> String? { map.originalSourceSha256 ?? map.sha256 }
    func datasetDisplayTransform(_ map: WorkbenchMap, role: String) -> MapReviewDisplayTransform? {
        let width = map.originalSourceWidth ?? map.width
        let height = map.originalSourceHeight ?? map.height
        // Sources too small for the chosen training grid remain manageable and
        // display at their original size instead of requesting an invalid crop.
        if let width, let height, min(width, height) < training.size { return nil }
        let convention = map.originalNormalConvention ?? map.sourceNormalConvention ?? "opengl"
        guard let hash = datasetReviewSHA256(map),
              map.cropRectangle != nil || width != training.size || height != training.size || convention == "directx" else { return nil }
        let rectangle = map.cropRectangle ?? width.flatMap { w in height.map { h in
            [max(0, (w - training.size) / 2), max(0, (h - training.size) / 2), training.size, training.size]
        } }
        return MapReviewDisplayTransform(size: training.size, sourceSHA256: hash,
            algorithm: MapReviewDisplayTransform.exactCrop, mapType: role,
            normalConvention: convention, cropRectangle: rectangle)
    }

    var datasetURL: URL? { dataset.map { URL(fileURLWithPath: $0.datasetPath) } }
    var datasetDisplayURL: URL? {
        guard let canonical = datasetURL else { return nil }
        guard let saved = preferences.string(forKey: "dataset") else { return canonical }
        let located = URL(fileURLWithPath: saved)
        return located.resolvingSymlinksInPath().standardizedFileURL == canonical.resolvingSymlinksInPath().standardizedFileURL
            ? located : canonical
    }
    var datasetNativeSizeLabel: String {
        guard let dataset, let first = dataset.samples.first?.maps.values.first,
              let width = first.width, let height = first.height else { return "Map dimensions not verified" }
        let matching = dataset.samples.allSatisfy { !$0.maps.isEmpty && $0.maps.values.allSatisfy { $0.width == width && $0.height == height } }
        return matching ? "\(width.formatted()) × \(height.formatted()) native maps" : "Mixed native map sizes"
    }
    var workspaceURL: URL { URL(fileURLWithPath: workspacePath).standardizedFileURL }
    var backendDirectory: URL { Bundle.main.resourceURL!.appendingPathComponent("MaterialBackend") }
    var dependencyArguments: [String] { ["--model-directory", modelDirectory, "--code-directory", codeDirectory] }

    init(preferences defaults: UserDefaults = UserDefaults(suiteName: "org.ipde.material-tools")!,
         managedWorkspaceURL: URL? = nil,
         resources: MachineResources = .current,
         selectedCheckpointRegistryURL: URL = SelectedMaterialCheckpoint.registryURL,
         trashHandler: @escaping (URL) throws -> Void = { url in try FileManager.default.trashItem(at: url, resultingItemURL: nil) },
         workerOverride: (@MainActor ([String], String) async throws -> String)? = nil) {
        preferences = defaults
        self.trashHandler = trashHandler
        if let data = defaults.data(forKey: "recentDatasets.v1"),
           let locations = try? JSONDecoder().decode([WorkbenchDatasetLocation].self, from: data) {
            recentDatasets = locations
        }
        uploadRepo = defaults.string(forKey: "uploadRepository") ?? ""
        uploadPublic = defaults.object(forKey: "uploadPublic") == nil ? StudioPreferences.defaults.bool(forKey: StudioPreferences.developerModeKey) : defaults.bool(forKey: "uploadPublic")
        uploadAfterTraining = defaults.object(forKey: "uploadAfterTraining") == nil ? true : defaults.bool(forKey: "uploadAfterTraining")
        self.resources = resources
        self.selectedCheckpointRegistryURL = selectedCheckpointRegistryURL
        self.workerOverride = workerOverride
        self.managedWorkspaceURL = (managedWorkspaceURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Material Workspace")).standardizedFileURL
        let local = Bundle.main.object(forInfoDictionaryKey: "IPDEWorkspace") as? String ?? ""
        let configuredWorkspace = defaults.string(forKey: "workspace") ?? local
        let workspace = configuredWorkspace.isEmpty ? self.managedWorkspaceURL.path : URL(fileURLWithPath: configuredWorkspace).standardizedFileURL.path
        workspacePath = workspace
        pythonPath = defaults.string(forKey: "python") ?? MaterialWorkbenchRuntime.defaultPython(workspace: URL(fileURLWithPath: workspace))
        let cache = URL(fileURLWithPath: workspace).appendingPathComponent("out/material-training/transfer-models")
        let materialBasePath = defaults.string(forKey: "materialModelDirectory") ?? cache.appendingPathComponent("pbrnxt-base").path
        modelDirectory = materialBasePath
        codeDirectory = defaults.string(forKey: "materialCodeDirectory") ?? URL(fileURLWithPath: materialBasePath).appendingPathComponent("source").path
        let saved = WorkbenchPreferences.load(from: defaults)
        if let options = saved.training {
            training = options.restored(for: resources)
            if training != options {
                trainingPreferenceNotice = "Saved training settings were adapted to the supported limits of this Mac. Review the resource limit before training."
                activity = trainingPreferenceNotice!
            }
        } else { training.memoryGB = resources.defaultTrainingGiB }
        selectedSampleId = saved.selectedSampleId
        selectedInputVariantId = saved.selectedInputVariantId
        if let role = saved.selectedRole, ["input", "height", "roughness", "normal"].contains(role) { selectedRole = role }
        selectedCheckpointId = saved.selectedCheckpointId
        comparisonCheckpointIds = saved.comparisonCheckpointIds ?? []
        comparisonIncludesBase = saved.comparisonIncludesBase ?? true
        sourceImageURL = saved.sourceImagePath.map { URL(fileURLWithPath: $0) }
        lastOutputURL = saved.lastOutputPath.map { URL(fileURLWithPath: $0) }
        lastLogURL = saved.lastLogPath.map { URL(fileURLWithPath: $0) }
        lastPackageURL = saved.lastPackagePath.map { URL(fileURLWithPath: $0) }
        lastPackageCheckpointId = saved.lastPackageCheckpointId
    }

    /// Save changes as they happen; closing a window or the app is not a save
    /// boundary, and starting a worker must not be required to retain a choice.
    private func saveUserSettings() {
        guard !isRestoringPreferences else { return }
        WorkbenchPreferences(training: training, selectedSampleId: selectedSampleId, selectedRole: selectedRole,
            selectedInputVariantId: selectedInputVariantId,
            selectedCheckpointId: selectedCheckpointId, comparisonCheckpointIds: comparisonCheckpointIds,
            comparisonIncludesBase: comparisonIncludesBase, sourceImagePath: sourceImageURL?.path,
            lastOutputPath: lastOutputURL?.path, lastLogPath: lastLogURL?.path,
            lastPackagePath: lastPackageURL?.path, lastPackageCheckpointId: lastPackageCheckpointId).save(to: preferences)
    }

    func restore() {
        guard !isBusy, !hasRestored else { return }
        hasRestored = true
        let args = CommandLine.arguments
        var datasetPath = preferences.string(forKey: "dataset")
        if datasetPath == nil {
            let localDataset = workspaceURL.deletingLastPathComponent().appendingPathComponent("material-dataset/dataset.json")
            if FileManager.default.fileExists(atPath: localDataset.path) { datasetPath = localDataset.path }
        }
        if let i = args.firstIndex(of: "--dataset"), args.indices.contains(i + 1) { datasetPath = args[i + 1] }
        let paths = preferences.stringArray(forKey: "checkpoints") ?? []
        let saved = WorkbenchPreferences.load(from: preferences)
        operation("Opening workspace…") {
            do { try await self.loadTrainingCapabilities() }
            catch { self.logText += "Training setup is unavailable: \(error.localizedDescription)\n" }
            self.isRestoringPreferences = true
            defer { self.isRestoringPreferences = false; self.saveUserSettings() }
            if let datasetPath, FileManager.default.fileExists(atPath: datasetPath) {
                let url = URL(fileURLWithPath: datasetPath)
                var directory: ObjCBool = false
                if FileManager.default.fileExists(atPath: datasetPath, isDirectory: &directory), directory.boolValue,
                   !FileManager.default.fileExists(atPath: url.appendingPathComponent("dataset.json").path),
                   !(url.lastPathComponent == "sources" && FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("dataset.json").path)) {
                    self.pendingSourceFolder = url
                    self.showNewDatasetSheet = true
                } else { try await self.loadDataset(url) }
            }
            for path in paths where FileManager.default.fileExists(atPath: path) {
                do { try await self.loadCheckpoint(URL(fileURLWithPath: path), select: false) }
                catch { self.logText += "Could not reconnect \(path): \(error.localizedDescription)\n" }
            }
            if !self.checkpoints.contains(where: { $0.id == self.selectedCheckpointId }) {
                self.selectedCheckpointId = self.checkpoints.first?.id
            }
            let available = Set(self.checkpoints.map(\.id))
            self.comparisonCheckpointIds = saved.comparisonCheckpointIds.map { $0.intersection(available) } ?? available
        }
    }

    func saveConfiguration(refreshSelectedRuntime: Bool = true) {
        preferences.set(workspacePath, forKey: "workspace")
        preferences.set(pythonPath, forKey: "python")
        preferences.set(modelDirectory, forKey: "materialModelDirectory")
        preferences.set(codeDirectory, forKey: "materialCodeDirectory")
        saveUploadConfiguration()
        if refreshSelectedRuntime {
            do {
                for target in ["height", "roughness", "normal"] {
                    try SelectedMaterialCheckpoint.refreshRuntime(pythonPath: pythonPath, workspacePath: workspacePath,
                        modelDirectory: modelDirectory, codeDirectory: codeDirectory,
                        at: SelectedMaterialCheckpoint.registryURL(for: target, heightRegistryURL: selectedCheckpointRegistryURL))
                }
            } catch {
                self.error = "Runtime settings were saved, but Texture Studio could not reconnect its selected checkpoint: \(error.localizedDescription)"
            }
        }
    }

    func chooseDataset() {
        guard !isBusy else { return }
        let panel = NSOpenPanel()
        panel.title = "Open Dataset"
        panel.message = "Open a saved dataset or choose a folder of material maps to set up and import. Subfolders are scanned automatically."
        panel.prompt = "Open Dataset"
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.folder, .json]
        panel.directoryURL = datasetFolderURL?.deletingLastPathComponent()
        panel.begin { response in
            if response == .OK, let url = panel.url { self.openDataset(url) }
        }
    }
    func openDataset(_ url: URL) {
        guard !isBusy else { error = "Stop the current operation before changing datasets."; return }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue,
           !FileManager.default.fileExists(atPath: url.appendingPathComponent("dataset.json").path),
           !(url.lastPathComponent == "sources" && FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("dataset.json").path)) {
            if dataset != nil { importMaterialFolder(url) }
            else { pendingSourceFolder = url; showNewDatasetSheet = true }
            return
        }
        operation("Reading dataset…") {
            try await self.loadDataset(url)
            self.activity = "Opened \(self.datasetName)."
        }
    }
    func receiveDataset(_ url: URL) {
        if isBusy { queuedDatasetURL = url }
        else { openDataset(url) }
    }
    func openCheckpoint(_ url: URL) {
        guard !isBusy else { error = "Stop the current operation before changing checkpoints."; return }
        operation("Inspecting checkpoint…") { try await self.loadCheckpoint(url) }
    }
    func chooseCheckpoint() {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = true
        panel.title = "Choose material checkpoints"; panel.prompt = "Inspect Checkpoints"
        panel.begin { response in
            guard response == .OK else { return }
            self.operation("Inspecting checkpoints…") {
                for url in panel.urls { try await self.loadCheckpoint(url) }
            }
        }
    }
    func chooseSourceImage() {
        chooseFile(types: [.image], title: "Choose a surface photo to prepare as diffuse for every checkpoint") { self.sourceImageURL = $0 }
    }
    func choosePython() {
        chooseFile(title: "Locate Python with PyTorch, OpenImageIO and MPS") { url in
            self.pythonPath = url.path; self.saveConfiguration()
        }
    }
    func chooseWorkspace() {
        chooseFolder(title: "Choose a working folder for material runs") { url in self.workspacePath = url.path; self.saveConfiguration() }
    }
    func chooseEncoder() {
        chooseFolder(title: "Locate the material base model folder") { url in self.modelDirectory = url.path; self.saveConfiguration() }
    }
    func chooseEncoderCode() {
        chooseFolder(title: "Locate the material architecture source folder") { url in self.codeDirectory = url.path; self.saveConfiguration() }
    }

    func loadDataset(_ url: URL) async throws {
        let result: WorkbenchDataset = try WorkbenchProcess.decode(WorkbenchDataset.self,
            output: await worker(["dataset", "--dataset", url.path, "--target", training.target,
                                  "--default-review-size", String(training.size)]))
        adoptDataset(result)
        datasetPreparationSummary = ""
        saveDatasetLocation(result, requested: url)
    }

    func saveDatasetLocation(_ result: WorkbenchDataset, requested: URL? = nil) {
        let canonical = URL(fileURLWithPath: result.datasetPath)
        let located = requested ?? preferences.string(forKey: "dataset").map { URL(fileURLWithPath: $0) }
        let selected = located.flatMap {
            $0.resolvingSymlinksInPath().standardizedFileURL == canonical.resolvingSymlinksInPath().standardizedFileURL ? $0 : nil
        } ?? canonical
        preferences.set(selected.path, forKey: "dataset")
        rememberDataset(result)
    }

    func adoptDataset(_ result: WorkbenchDataset, preferredMaterial: String? = nil) {
        let material = preferredMaterial ?? selectedMaterialId
        dataset = result
        if let size = result.trainingSize {
            training.size = size
            applyAutomaticMemoryBudget()
        }
        supportedTrainingSizes = backendTrainingSizes.filter { result.supportedTrainingSizes?.contains($0) ?? true }
        if !result.samples.contains(where: { $0.id == selectedSampleId }) {
            selectedSampleId = result.materials.first(where: { $0.id == material })?.samples.first?.id ?? result.samples.first?.id
        }
    }

    func selectTrainingSize(_ size: Int) {
        guard !isBusy else { return }
        guard supportedTrainingSizes.contains(size) else { error = "This size does not fit the current training budget."; return }
        guard dataset != nil else { training.size = size; applyAutomaticMemoryBudget(); return }
        updateDatasetInfo(name: datasetName, description: datasetDescription, size: size)
    }

    func selectTrainingTarget(_ target: String) {
        guard !isBusy, ["height", "roughness", "normal"].contains(target) else { return }
        training.target = target
        guard let datasetURL else { return }
        operation("Updating \(target == "height" ? "displacement" : target) splits…") { try await self.loadDataset(datasetURL) }
    }

    func prepareTrainingDataset() {
        guard !isBusy, dataset != nil else { return }
        let size = training.size
        let material = training.useSelectedMaterialOnly ? selectedMaterialId : nil
        guard dataset?.readyForTraining(size: size, material: material, target: training.target) != true else { return }
        operation("Preparing \(size) × \(size) complete training maps…") {
            _ = try await self.ensureTrainingDataset(size: size)
            self.activity = self.datasetPreparationSummary
        }
    }

    private func ensureTrainingDataset(size: Int) async throws -> WorkbenchDataset {
        guard let current = dataset else { throw StudioError("Open a material dataset first.") }
        guard supportedTrainingSizes.contains(size) else { throw StudioError("Choose a size supported by the current training budget.") }
        let material = selectedMaterialId
        let checkMaterial = training.useSelectedMaterialOnly ? material : nil
        if current.readyForTraining(size: size, material: checkMaterial, target: training.target) { return current }
        isPreparingDataset = true
        activity = "Preparing \(size) × \(size) complete training maps from the original materials…"
        defer { isPreparingDataset = false }
        var arguments = [
            "prepare-size", "--dataset", current.datasetPath, "--size", String(size), "--target", training.target,
            "--automatic-validation", "--expected-index-sha256", current.indexSha256]
        if let reviewHash = current.reviewSha256 { arguments += ["--expected-review-sha256", reviewHash] }
        if let checkMaterial { arguments += ["--material", checkMaterial] }
        let result: WorkbenchDataset = try WorkbenchProcess.decode(WorkbenchDataset.self, output: await worker(arguments))
        guard result.readyForTraining(size: size, material: checkMaterial, target: training.target), let preparation = result.preparation,
              preparation.cropSize == size, !preparation.originalDatasetModified,
              URL(fileURLWithPath: preparation.preparedDatasetPath).standardizedFileURL == URL(fileURLWithPath: result.datasetPath).standardizedFileURL else {
            throw StudioError("Dataset preparation did not verify matching map dimensions and preserved originals.")
        }
        try Task.checkCancellation()
        adoptDataset(result, preferredMaterial: material)
        preferences.set(preparation.sourceDatasetPath, forKey: "dataset")
        datasetPreparationSummary = "\(size) × \(size) native pixel maps: matching originals are referenced directly; each crop is saved once in temporary training storage."
        return result
    }
    func loadCheckpoint(_ url: URL, select: Bool = true) async throws {
        let output = try await worker(["checkpoint", "--checkpoint", url.path])
        let checkpoint = try WorkbenchProcess.decode(WorkbenchCheckpoint.self, output: output)
        let runtime = try WorkbenchProcess.decode(CheckpointRuntimeLocation.self, output: output)
        if let source = runtime.codeDirectory { codeDirectory = source }
        guard checkpoint.compatible else { throw StudioError("This checkpoint is not supported by the material backend.") }
        if let i = checkpoints.firstIndex(where: { $0.id == checkpoint.id }) { checkpoints[i] = checkpoint }
        else { checkpoints.append(checkpoint) }
        if select {
            selectedCheckpointId = checkpoint.id
            comparisonCheckpointIds.insert(checkpoint.id)
        }
        if !isRestoringPreferences { preferences.set(checkpoints.map(\.checkpointPath), forKey: "checkpoints") }
    }

    func curateSelected(status: String, split: String? = nil, note: String? = nil) {
        guard let dataset, let sample = selectedSample else { return }
        operation("Saving material review…") {
            var args = ["curate", "--dataset", dataset.datasetPath, "--sample", sample.id,
                        "--status", status, "--expected-index-sha256", dataset.indexSha256, "--review-size", String(self.training.size)]
            if let reviewHash = dataset.reviewSha256 { args += ["--expected-review-sha256", reviewHash] }
            if let split { args += ["--split", split] }
            if let note { args += ["--note", note] }
            _ = try await self.worker(args)
            try await self.loadDataset(URL(fileURLWithPath: dataset.datasetPath))
        }
    }
    func inspectSelectedMap() {
        guard let map = selectedMap else { return }
        ReviewWindowController.shared.open(candidates: [MapReviewCandidate(id: map.path, label: "\(selectedSampleId ?? "Material") · \(selectedRole)",
            mapURL: datasetReviewURL(map), numeric: selectedRole != "input",
            sourceIdentity: MapReviewSourceIdentity(path: datasetReviewURL(map).path, sha256: datasetReviewSHA256(map)),
            displayTransform: datasetDisplayTransform(map, role: selectedRole))])
    }

    func compare() {
        let diffuseMap = sourceImageURL == nil ? selectedDiffuseMap : nil
        guard let image = sourceImageURL ?? diffuseMap.map(datasetReviewURL) else {
            error = "Choose a surface photo or select a dataset diffuse map first."; return
        }
        let selected = checkpoints.filter { comparisonCheckpointIds.contains($0.id) }
        let includeBase = comparisonIncludesBase
        guard !selected.isEmpty, selected.count + (includeBase ? 1 : 0) >= 2, Set(selected.map(\.target)).count == 1 else {
            error = "Select one checkpoint plus its base, or two checkpoints predicting the same map type."; return
        }
        let target = selected[0].target
        let sampleLabel = sourceImageURL == nil ? selectedSampleId ?? image.lastPathComponent : image.lastPathComponent
        let referenceMap = sourceImageURL == nil ? selectedSample?.maps[target] : nil
        let referenceSampleID = sourceImageURL == nil ? selectedSampleId : nil
        let diffuseTransform = diffuseMap.flatMap { datasetDisplayTransform($0, role: "input") }
        let referenceTransform = referenceMap.flatMap { datasetDisplayTransform($0, role: target) }
        let referenceURL = referenceMap.map(datasetReviewURL)
        let photoSettings = testPhotoSettings
        let comparisonSize = training.size
        let inputIsPhoto = sourceImageURL != nil
        let targetLabel = target == "height" ? "Displacement" : target.capitalized
        operation("Comparing \(selected.count + (includeBase ? 1 : 0)) models on the \(comparisonSize) × \(comparisonSize) diffuse grid…") {
            let parent = try self.newOutputURL(prefix: "comparison")
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
            self.lastOutputURL = parent
            self.comparisonCandidates = []
            var modelImage = image
            var reviewedDiffuse = image
            var temporaryInput: URL?
            defer { if let temporaryInput { try? FileManager.default.removeItem(at: temporaryInput) } }
            if inputIsPhoto {
                let engine = TextureEngine()
                let photo = try await engine.importPhoto(image)
                var settings = photoSettings
                settings.outputSize = comparisonSize
                let prepared = try await engine.prepareDiffuse(source: photo, settings: settings)
                modelImage = parent.appendingPathComponent("diffuse.png")
                try await engine.writeDiffuse(prepared.diffuse, to: modelImage)
                reviewedDiffuse = modelImage
            } else if let diffuseTransform {
                let directory = FileManager.default.temporaryDirectory.appendingPathComponent("material-comparison-input-\(UUID().uuidString)")
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
                temporaryInput = directory
                modelImage = directory.appendingPathComponent("diffuse.png")
                var arguments = ["review-source", "--image", image.path, "--expected-sha256", diffuseTransform.sourceSHA256,
                    "--size", String(diffuseTransform.size), "--map-type", "input", "--output", modelImage.path]
                if let rectangle = diffuseTransform.cropRectangle { arguments += ["--source-rectangle"] + rectangle.map(String.init) }
                _ = try await self.worker(arguments)
            }
            var candidates = [MapReviewCandidate(id: "diffuse|" + reviewedDiffuse.path, label: "Diffuse · model input", mapURL: reviewedDiffuse, numeric: false,
                sampleLabel: sampleLabel, detail: reviewedDiffuse.path, role: "diffuse",
                modelIdentity: MapReviewModelIdentity(mapType: "input"), displayTransform: diffuseTransform)]
            if let referenceMap {
                let sourceIdentity: MapReviewSourceIdentity
                if let original = referenceMap.originalSourcePath {
                    sourceIdentity = MapReviewSourceIdentity(path: original, sha256: referenceMap.originalSourceSha256,
                        bits: referenceMap.sourceBits, cropRectangle: referenceMap.cropRectangle, pixelDimensions: referenceMap.originalSourceWidth.flatMap { width in
                            referenceMap.originalSourceHeight.map { [width, $0] } })
                } else {
                    sourceIdentity = await Task.detached {
                        MapReviewSourceIdentity.fromDatasetMap(referenceMap, target: target, sampleID: referenceSampleID)
                    }.value
                }
                candidates.append(MapReviewCandidate(id: "target|" + referenceMap.path, label: "Source \(targetLabel.lowercased()) · reference",
                    mapURL: referenceURL ?? referenceMap.url, numeric: true, sampleLabel: sampleLabel,
                    detail: (["Real source map · not a model output"] + sourceIdentity.recordedDetails).joined(separator: " · "), role: "target",
                    modelIdentity: MapReviewModelIdentity(mapType: target, modelName: "Dataset reference"), sourceIdentity: sourceIdentity,
                    displayTransform: referenceTransform))
            }
            if includeBase, let reference = selected.first {
                self.activity = "Running the material base model"
                let result: MaterialInferenceResponse = try WorkbenchProcess.decode(MaterialInferenceResponse.self, output: await self.worker([
                    "infer", "--baseline", "--checkpoint", reference.checkpointPath, "--expected-sha256", reference.sha256,
                    "--image", modelImage.path, "--output", parent.appendingPathComponent("base-untrained").path, "--device", "mps"] + self.dependencyArguments))
                guard result.checkpointSha256 == reference.sha256, let map = result.outputs[target] else {
                    throw StudioError("The base comparison did not match the selected checkpoint architecture and map type.")
                }
                candidates.append(MapReviewCandidate(id: "base|" + reference.id, label: "Base · \(targetLabel)",
                    mapURL: URL(fileURLWithPath: map.path), numeric: true, sampleLabel: sampleLabel,
                    detail: "Material base prediction before refinement", role: "base",
                    modelIdentity: MapReviewModelIdentity(architecture: reference.trainingBaseLabel,
                        mapType: target, modelName: "Material base")))
            }
            for (i, checkpoint) in selected.enumerated() {
                self.activity = "Running checkpoint \(i + 1)/\(selected.count): \(checkpoint.title)"
                let child = parent.appendingPathComponent("candidate-\(i + 1)")
                let result: MaterialInferenceResponse = try WorkbenchProcess.decode(MaterialInferenceResponse.self, output: await self.worker([
                    "infer", "--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256,
                    "--image", modelImage.path, "--output", child.path, "--device", "mps"] + self.dependencyArguments))
                guard result.checkpointSha256 == checkpoint.sha256, let map = result.outputs[checkpoint.target] else {
                    throw StudioError("The comparison did not use the selected checkpoint or map type.")
                }
                candidates.append(MapReviewCandidate(id: checkpoint.id, label: "\(checkpoint.url.deletingLastPathComponent().lastPathComponent) · \(targetLabel)",
                    mapURL: URL(fileURLWithPath: map.path), numeric: true, sampleLabel: sampleLabel,
                    detail: "Trained model prediction · not the real source map · \(checkpoint.trainingBaseLabel) · \(checkpoint.url.lastPathComponent) · step \(checkpoint.step.formatted()) · SHA256 \(checkpoint.sha256.prefix(12))", role: "checkpoint",
                    modelIdentity: MapReviewModelIdentity(checkpointPath: checkpoint.checkpointPath,
                        checkpointSHA256: checkpoint.sha256, checkpointStep: checkpoint.step,
                        architecture: checkpoint.trainingBaseLabel, mapType: checkpoint.target)))
            }
            self.comparisonCandidates = candidates
            self.lastOutputURL = parent
            var material: [String: Any] = ["material_id": sampleLabel, "diffuse": reviewedDiffuse.path,
                    "variants": candidates.filter { $0.role != "diffuse" }.map { candidate in
                        var item: [String: Any] = ["name": candidate.label, "candidate_id": candidate.id, target: candidate.mapURL.path,
                            "sample_label": sampleLabel, "detail": candidate.detail ?? "", "role": candidate.role, "numeric": candidate.numeric]
                        if let identity = candidate.modelIdentity {
                            item.merge(identity.manifestFields) { _, recorded in recorded }
                        }
                        if let identity = candidate.sourceIdentity {
                            item.merge(identity.manifestFields) { _, recorded in recorded }
                        }
                        if let transform = candidate.displayTransform {
                            item.merge(transform.manifestFields) { _, recorded in recorded }
                        }
                        if candidate.role == "base" { item["baseline"] = true }
                        return item
                    }]
            material["diffuse_variant_id"] = diffuseMap?.variantId
            if let diffuseTransform {
                material["diffuse_native_size"] = diffuseTransform.size
                material["diffuse_source_sha256"] = diffuseTransform.sourceSHA256
                material["diffuse_resize_algorithm"] = diffuseTransform.algorithm
                material["diffuse_source_crop_rectangle"] = diffuseTransform.cropRectangle
            }
            let manifest: [String: Any] = ["schema": "texture-studio-material-quality-review-v1", "comparison_target": target,
                "materials": [material]]
            try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
                .write(to: parent.appendingPathComponent("review-manifest.json"), options: .atomic)
        }
    }

    func loadTrainingCapabilities() async throws {
        let result = try WorkbenchProcess.decode(WorkbenchTrainingCapabilities.self,
            output: await worker(["capabilities", "--memory-gib", String(training.automaticMemory ? resources.maximumTrainingGiB : training.memoryGB), "--cache-gib", String(training.cacheGB), "--scope", training.scope]))
        memoryPlans = result.memoryPlans ?? [:]
        backendTrainingSizes = result.trainingSizes.filter { [256, 512, 1024, 2048].contains($0) }.sorted()
        supportedTrainingSizes = backendTrainingSizes.filter { dataset?.supportedTrainingSizes?.contains($0) ?? true }
        if dataset?.trainingSize == nil, !supportedTrainingSizes.contains(training.size), let size = supportedTrainingSizes.last {
            training.size = size
        }
        applyAutomaticMemoryBudget()
    }

    private func applyAutomaticMemoryBudget() {
        guard training.automaticMemory, let plan = memoryPlans[String(training.size)] else { return }
        training.memoryGB = min(resources.maximumTrainingGiB, max(resources.trainingMemoryRange.lowerBound, plan.recommendedMemoryGib))
    }

    func refreshTrainingCapabilities() {
        guard !isBusy else { return }
        operation("Checking supported training sizes…") {
            let previousSize = self.training.size
            try await self.loadTrainingCapabilities()
            if previousSize != self.training.size, let datasetURL = self.datasetURL { try await self.loadDataset(datasetURL) }
        }
    }

    var trainingConfigurationIssue: String? {
        if dataset == nil { return "Open your source dataset first." }
        if let issue = resources.trainingMemoryIssue(training.memoryGB) { return issue }
        if !supportedTrainingSizes.contains(training.size) {
            if dataset?.supportedTrainingSizes?.contains(training.size) == false { return "Choose a training size supplied by the original source maps." }
            return "No selected training size fits the current memory budget."
        }
        if training.useSelectedMaterialOnly && selectedMaterialId == nil { return "Select a material first." }
        let selected = training.useSelectedMaterialOnly ? dataset?.materials.first { $0.id == selectedMaterialId }?.samples ?? [] : samples
        let eligible = selected.filter { self.sampleIsTrainable($0) }
        if eligible.isEmpty { return "No included material provides a registered \(training.target == "height" ? "16-bit displacement" : training.target) target at this size." }
        if eligible.allSatisfy({ $0.split == "validation" && $0.splitAssignment == "manual" }) {
            return "Assign at least one included material to Training in Dataset. All available materials are assigned to Validation."
        }
        if training.useWarmStart && selectedCheckpoint?.supportsTrainingWarmStart != true { return "Select a material checkpoint to refine." }
        return nil
    }

    func sampleIsTrainable(_ sample: WorkbenchSample) -> Bool {
        !["excluded", "rejected"].contains(sample.status) && min(sample.width, sample.height) >= training.size &&
            (sample.availableTargets?.contains(training.target) ?? (sample.maps[training.target] != nil))
    }

    func startTraining() {
        if let issue = trainingConfigurationIssue { error = issue; return }
        runTraining(checkpoint: training.useWarmStart ? selectedCheckpoint : nil)
    }

    private func runTraining(checkpoint: WorkbenchCheckpoint?) {
        operation(checkpoint == nil ? "Training material LoRA…" : "Refining material LoRA…", training: true) {
            try await self.loadTrainingCapabilities()
            let prepared = try await self.ensureTrainingDataset(size: self.training.size)
            let output = try self.newOutputURL(prefix: "material-\(self.training.target)")
            self.lastOutputURL = output
            var args = [checkpoint == nil ? "train" : "refine", "--dataset", prepared.datasetPath,
                "--output", output.path, "--size", String(self.training.size), "--whole-maps",
                "--target", self.training.target, "--scope", self.training.scope, "--memory-gib", String(self.training.memoryGB),
                "--cache-gib", String(self.training.cacheGB), "--max-minutes", String(self.training.maxMinutes),
                "--updates-per-map", String(self.training.updatesPerCrop),
                "--lora-rank", String(self.training.loraRank), "--lora-alpha", String(self.training.loraAlpha)] + self.dependencyArguments
            if self.developerMode { args += ["--developer-mode"] }
            if self.training.useSelectedMaterialOnly, let id = self.selectedMaterialId { args += ["--material", id] }
            if let checkpoint { args += ["--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256] }
            self.isResumingTraining = checkpoint != nil
            do {
                let result = try WorkbenchProcess.decode(WorkbenchTrainingResponse.self, output: await self.worker(args))
                try await self.loadCheckpoint(URL(fileURLWithPath: result.checkpointPath))
                self.lastPackageURL = result.packagePath.map { URL(fileURLWithPath: $0) }
                self.lastPackageCheckpointId = self.selectedCheckpointId
                self.activity = self.isSavingTraining ? "Stopped and saved material LoRA." : "Training finished. Review the material maps before using this model."
                try await self.cleanupTrainingDataset(prepared)
                if self.developerMode && self.uploadAfterTraining && !self.isStopping {
                    let account = try WorkbenchProcess.decode(HuggingFaceAccountResponse.self, output: await self.worker(["hub-account"]))
                    self.uploadAccount = account.authenticated ? account.username : nil
                    self.uploadAccountChecked = true
                    if let checkpoint = self.selectedCheckpoint, self.canUploadSelectedCheckpoint {
                        try await self.performUpload(checkpoint)
                    } else {
                        self.activity += " Saved locally; sign in to Hugging Face to publish it."
                    }
                }
            } catch {
                // Cancellation must not prevent cleanup of a positively owned stage.
                await Task { @MainActor in try? await self.cleanupTrainingDataset(prepared) }.value
                throw error
            }
        }
    }

    private func cleanupTrainingDataset(_ prepared: WorkbenchDataset) async throws {
        guard prepared.preparation != nil else { return }
        let output = try await worker(["cleanup-size", "--dataset", prepared.datasetPath])
        _ = try WorkbenchProcess.decode(WorkbenchDatasetCleanup.self, output: output)
        try await loadDataset(URL(fileURLWithPath: prepared.preparation!.sourceDatasetPath))
        datasetPreparationSummary = "Training files cleared. Original source maps are selected."
    }

    func chooseResumeCheckpoint() {
        chooseFile(title: "Choose material safetensors to refine") { self.resumeTraining(from: $0) }
    }

    func resumeTraining(from url: URL) {
        operation("Opening material checkpoint…") {
            try await self.loadCheckpoint(url)
            self.training.useWarmStart = true
            if let checkpoint = self.selectedCheckpoint {
                self.training.target = checkpoint.target
                self.training.scope = checkpoint.scope ?? "final-map"
            }
        }
    }

    func removeMissingSource(_ url: URL, sampleID: String) {
        guard !FileManager.default.fileExists(atPath: url.path), let dataset, !isBusy else { return }
        operation("Updating missing source reference…") {
            _ = try await self.worker([
                "remove-missing", "--dataset", dataset.datasetPath, "--sample", sampleID,
                "--path", url.path, "--expected-index-sha256", dataset.indexSha256, "--review-size", String(self.training.size)])
            try await self.loadDataset(URL(fileURLWithPath: dataset.datasetPath))
        }
    }

    func stop() {
        guard isBusy else { return }
        isStopping = true
        isSavingTraining = false
        activity = isTraining ? (hasTrainingStarted ? "Aborting training…" : "Aborting training setup…") : "Stopping the operation…"
        runner?.stop()
        task?.cancel()
    }

    func stopAndSave() {
        guard canStopAndSave else { stop(); return }
        isStopping = true
        isSavingTraining = true
        activity = "Finishing the current update and saving the material LoRA…"
        runner?.stopAndSave()
    }

    func recordTrainingProgress(_ chunk: String) {
        trainingEventBuffer += chunk
        while let newline = trainingEventBuffer.firstIndex(of: "\n") {
            let line = String(trainingEventBuffer[..<newline])
            trainingEventBuffer.removeSubrange(...newline)
            guard let data = line.data(using: .utf8),
                  let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  event["event"] as? String == "training_started", isTraining, !isStopping else { continue }
            hasTrainingStarted = true
            activity = "Training material LoRA…"
        }
        if trainingEventBuffer.count > 100000 { trainingEventBuffer = String(trainingEventBuffer.suffix(100000)) }
    }

    func useSelectedInStudio() {
        guard let checkpoint = selectedCheckpoint, checkpoint.supportsStudioInference else { return }
        do {
            try SelectedMaterialCheckpoint(checkpointPath: checkpoint.checkpointPath, sha256: checkpoint.sha256,
                target: checkpoint.target, pythonPath: pythonPath, workspacePath: workspacePath,
                modelDirectory: modelDirectory, codeDirectory: codeDirectory,
                displayName: checkpoint.title, modelSummary: checkpoint.modelSummary)
                .save(to: SelectedMaterialCheckpoint.registryURL(for: checkpoint.target, heightRegistryURL: selectedCheckpointRegistryURL))
            activity = "Selected \(checkpoint.title) for Texture Studio."
        } catch { self.error = error.localizedDescription }
    }
    func exportSelectedCheckpoint() {
        guard let checkpoint = selectedCheckpoint else { return }
        chooseFolder(title: "Choose a folder for a new model package") { parent in
            self.operation("Exporting selected model package…") {
                let destination = parent.appendingPathComponent("material-\(checkpoint.target)-\(UUID().uuidString.prefix(8))")
                var args = ["package", "--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256, "--output", destination.path] + self.dependencyArguments
                if self.developerMode { args += ["--developer-mode"] }
                for adapter in self.adapterMix { args += ["--adapter", "\(adapter.path)=\(adapter.weight)"] }
                _ = try await self.worker(args)
                self.lastPackageURL = destination
                self.lastPackageCheckpointId = checkpoint.id
                NSWorkspace.shared.activateFileViewerSelecting([destination])
            }
        }
    }
    var suggestedUploadRepo: String {
        guard let uploadAccount, let checkpoint = selectedCheckpoint else { return "" }
        return HuggingFaceUpload.repository(account: uploadAccount, checkpoint: checkpoint)
    }
    var effectiveUploadRepo: String {
        let configured = uploadRepo.trimmingCharacters(in: .whitespacesAndNewlines)
        return configured.isEmpty ? suggestedUploadRepo : configured
    }
    var canUploadSelectedCheckpoint: Bool {
        selectedCheckpoint?.compatible == true && uploadAccount != nil && HuggingFaceUpload.validRepository(effectiveUploadRepo)
    }
    func saveUploadConfiguration() {
        preferences.set(uploadRepo, forKey: "uploadRepository")
        preferences.set(uploadPublic, forKey: "uploadPublic")
    }
    func refreshUploadAccount() {
        guard !isBusy else { return }
        operation("Checking saved Hugging Face account…") {
            let result = try WorkbenchProcess.decode(HuggingFaceAccountResponse.self,
                output: await self.worker(["hub-account"]))
            self.uploadAccount = result.authenticated ? result.username : nil
            self.uploadAccountChecked = true
            self.uploadAccountMessage = result.message
            self.activity = result.message
            if result.authenticated {
                let models = try WorkbenchProcess.decode(WorkbenchHubModels.self, output: await self.worker(["hub-models"]))
                self.hubModels = models.models
            }
        }
    }
    func uploadPackage() {
        guard let checkpoint = selectedCheckpoint else { error = "Select a model checkpoint to upload."; return }
        let repository = effectiveUploadRepo
        guard uploadAccount != nil else {
            uploadAccountMessage = "Run hf auth login in Terminal, then click Refresh Account."
            return
        }
        guard HuggingFaceUpload.validRepository(repository) else {
            error = "Enter a Hugging Face repository as owner/model-name."; return
        }
        saveUploadConfiguration()
        operation("Packaging and uploading \(checkpoint.title)…") { try await self.performUpload(checkpoint) }
    }
    private func performUpload(_ checkpoint: WorkbenchCheckpoint) async throws {
        let repository = effectiveUploadRepo, isPublic = uploadPublic
        let package = try newOutputURL(prefix: "upload-\(checkpoint.target)")
        defer { try? FileManager.default.removeItem(at: package) }
        var args = ["upload-selected", "--checkpoint", checkpoint.checkpointPath,
            "--expected-sha256", checkpoint.sha256, "--output", package.path, "--repo", repository]
        if isPublic { args += ["--public"] }
        if developerMode { args += ["--developer-mode"] }
        args += dependencyArguments
        let result = try WorkbenchProcess.decode(HuggingFaceUploadResponse.self, output: await worker(args))
        guard result.sourceCheckpointSha256 == checkpoint.sha256, result.repository == repository,
              result.private == !isPublic else {
            throw StudioError("The upload response did not match the selected checkpoint and destination. See the operation log.")
        }
        let existingPackage = checkpoint.url.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: existingPackage.appendingPathComponent("config.json").path) {
            lastPackageURL = existingPackage
            lastPackageCheckpointId = checkpoint.id
        }
        lastUploadURL = URL(string: result.commitUrl ?? result.url)
        activity = "Uploaded \(checkpoint.title) to \(repository) (\(isPublic ? "public" : "private"))."
        let models = try WorkbenchProcess.decode(WorkbenchHubModels.self, output: await worker(["hub-models"]))
        hubModels = models.models
    }
    var managedEncoderURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Material Models/pbrnxt-base")
    }
    func installEncoder() {
        operation("Downloading the material base…") {
            _ = try await self.worker(["install-base", "--destination", self.managedEncoderURL.path])
            self.modelDirectory = self.managedEncoderURL.path
            self.codeDirectory = self.managedEncoderURL.appendingPathComponent("source").path
            self.saveConfiguration()
            self.activity = "Material base installed."
        }
    }
    func removeDownloadedEncoder() {
        operation("Removing downloaded base weights…") {
            _ = try await self.worker(["remove-base", "--directory", self.managedEncoderURL.path])
            self.activity = "Base weights removed. Your full checkpoint remains usable; the base can be downloaded again."
        }
    }
    func refreshHubModels() {
        operation("Finding your Hugging Face material models…") {
            let result = try WorkbenchProcess.decode(WorkbenchHubModels.self, output: await self.worker(["hub-models"]))
            self.hubModels = result.models
        }
    }
    func downloadHubModel(_ model: WorkbenchHubModel) {
        operation("Downloading \(model.repository)…") {
            let destination = self.managedEncoderURL.deletingLastPathComponent().appendingPathComponent(model.repository.replacingOccurrences(of: "/", with: "--"))
            var args = ["download-model", "--repo", model.repository, "--destination", destination.path]
            if let revision = model.revision { args += ["--revision", revision] }
            let result = try WorkbenchProcess.decode(WorkbenchCheckpoint.self, output: await self.worker(args))
            self.codeDirectory = destination.appendingPathComponent("source").path
            self.saveConfiguration()
            try await self.loadCheckpoint(result.url)
        }
    }
    func chooseMixAdapter() {
        chooseFile(title: "Choose a compatible material LoRA") { url in
            self.adapterMix.append(WorkbenchAdapterWeight(path: url.path))
        }
    }
    func forgetSelectedCheckpoint() {
        guard let checkpoint = selectedCheckpoint else { return }
        checkpoints.removeAll { $0.id == checkpoint.id }
        comparisonCheckpointIds.remove(checkpoint.id)
        selectedCheckpointId = checkpoints.first?.id
        preferences.set(checkpoints.map(\.checkpointPath), forKey: "checkpoints")
    }

    func worker(_ args: [String], script requestedScript: String? = nil) async throws -> String {
        let datasetCommands: Set<String> = ["dataset", "prepare-size", "cleanup-size", "curate", "remove-missing", "create-dataset", "edit-dataset", "add-material", "import-folder", "scan-folder", "remove-material", "validate-delete"]
        let includesTarget = ["create-dataset", "edit-dataset", "add-material", "import-folder", "remove-material"].contains(args.first ?? "") && !args.contains("--target")
        let args = args + (includesTarget ? ["--target", training.target] : [])
        let script = requestedScript ?? (datasetCommands.contains(args.first ?? "") ? "material_workbench.py" : "material_model_workbench.py")
        try Task.checkCancellation()
        guard !WorkbenchLifecycle.shared.isTerminating else { throw CancellationError() }
        let trainingWorker = ["train", "refine"].contains(args.first ?? "")
        if trainingWorker { trainingEventBuffer = ""; hasTrainingStarted = false }
        defer { if trainingWorker { hasTrainingStarted = false } }
        if let workerOverride {
            let output = try await workerOverride(args, script)
            if trainingWorker { recordTrainingProgress(output) }
            try Task.checkCancellation()
            return output
        }
        guard FileManager.default.isExecutableFile(atPath: pythonPath) else { throw StudioError("Python is missing. Locate your PyTorch environment in Runtime settings.") }
        let worker = backendDirectory.appendingPathComponent(script)
        guard FileManager.default.fileExists(atPath: worker.path) else { throw StudioError("The bundled material backend is missing. Rebuild the apps.") }
        saveConfiguration(refreshSelectedRuntime: false)
        let logs = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Worklogs")
        let log = logs.appendingPathComponent("\(UUID().uuidString).log")
        lastLogURL = log
        let process = WorkbenchProcess(); runner = process
        let workerId = UUID(); activeWorkerId = workerId
        WorkbenchLifecycle.shared.add(process)
        defer {
            runner = nil; activeWorkerId = nil
            if let completed = try? String(contentsOf: log, encoding: .utf8) { logText = String(completed.suffix(100000)) }
            WorkbenchLifecycle.shared.remove(process)
        }
        let output = try await process.run(executable: URL(fileURLWithPath: pythonPath), arguments: ["-B", worker.path] + args,
            directory: FileManager.default.fileExists(atPath: workspacePath) ? workspaceURL : nil, log: log) { [weak self] chunk in
                Task { @MainActor in
                    guard self?.activeWorkerId == workerId else { return }
                    self?.logText += chunk
                    if let text = self?.logText, text.count > 100000 { self?.logText = String(text.suffix(100000)) }
                    if trainingWorker { self?.recordTrainingProgress(chunk) }
                    if args.first == "scan-folder", let progress = self?.logText.components(separatedBy: "\n").last(where: { $0.hasPrefix("IPDE_PROGRESS:") }) {
                        self?.activity = String(progress.dropFirst("IPDE_PROGRESS:".count))
                    }
                }
            }
        try Task.checkCancellation()
        return output
    }
    func operation(_ label: String, training: Bool = false, body: @escaping @MainActor () async throws -> Void) {
        guard !isBusy else { return }
        isBusy = true; isTraining = training; isStopping = false; hasTrainingStarted = false; isSavingTraining = false
        error = nil; activity = label; logText = ""; trainingEventBuffer = ""
        task = Task {
            defer {
                isBusy = false; isTraining = false; isResumingTraining = false; isStopping = false
                hasTrainingStarted = false; isSavingTraining = false; isPreparingDataset = false; task = nil
                if let url = queuedDatasetURL {
                    queuedDatasetURL = nil
                    openDataset(url)
                }
            }
            do {
                try await body()
                if activity == label { activity = "Ready." }
            }
            catch {
                if Task.isCancelled || error is CancellationError {
                    activity = "Operation stopped. See the log and output folder."
                } else { self.error = error.localizedDescription; activity = "Operation stopped. See the error and log." }
            }
        }
    }
    private func newOutputURL(prefix: String) throws -> URL {
        if workspaceURL == managedWorkspaceURL, !FileManager.default.fileExists(atPath: workspacePath) {
            try FileManager.default.createDirectory(at: managedWorkspaceURL, withIntermediateDirectories: true)
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: workspacePath, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw StudioError("The chosen working folder is missing. Locate it in Local runtime settings.")
        }
        let root = workspaceURL.appendingPathComponent("out/material-training")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("\(prefix)-\(UUID().uuidString.prefix(8))")
    }
    private func chooseFile(types: [UTType] = [], title: String, selected: @escaping (URL) -> Void) {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.title = title; panel.allowedContentTypes = types
        panel.begin { response in if response == .OK, let url = panel.url { selected(url) } }
    }
    private func chooseFolder(title: String, selected: @escaping (URL) -> Void) {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.title = title; panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.begin { response in if response == .OK, let url = panel.url { selected(url) } }
    }
}

private struct CheckpointRuntimeLocation: Decodable {
    let codeDirectory: String?
}
