import AppKit
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class WorkbenchStore {
    var dataset: WorkbenchDataset?
    @ObservationIgnored var sourceFolderObserver: SourceFolderObserver?
    @ObservationIgnored var sourceFolderNeedsRefresh = false
    @ObservationIgnored var isRefreshingSourceFolder = false
    var datasetBrowserMode: DatasetBrowserMode = .originals {
        didSet {
            if oldValue != datasetBrowserMode, !datasetBrowserSamples.contains(where: { $0.id == selectedSampleId }) {
                selectedSampleId = datasetBrowserSamples.first?.id
            }
        }
    }
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
    var pendingAdditionalSourceFolders: [URL] = []
    var pendingMapAssignment: WorkbenchSourceInventoryItem?
    var folderImport: WorkbenchFolderImport?
    var folderImportSize = 2048
    var folderImportScannedSize: Int?
    var isScanningFolder = false
    var folderImportURL: URL?
    @ObservationIgnored var folderImportPlanURL: URL?
    @ObservationIgnored private var queuedDatasetURL: URL?
    @ObservationIgnored private var queuedTrainingHandoff: MaterialTrainingHandoff?
    @ObservationIgnored private var queuedResumeURL: URL?
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
    var trainingNavigationRequest = UUID()
    var datasetNavigationRequest = UUID()
    var developerMode: Bool {
        get { StudioPreferences.defaults.bool(forKey: StudioPreferences.developerModeKey) }
        set { StudioPreferences.defaults.set(newValue, forKey: StudioPreferences.developerModeKey) }
    }
    var uploadAfterTraining = true { didSet { preferences.set(uploadAfterTraining, forKey: "uploadAfterTraining") } }
    private var configuredPreparationWorkers: Int
    var preparationWorkers: Int {
        get { configuredPreparationWorkers }
        set {
            configuredPreparationWorkers = min(resources.availableProcessorCount, max(1, newValue))
            preferences.set(configuredPreparationWorkers, forKey: "preparationWorkers")
        }
    }
    private(set) var supportedTrainingSizes: [Int] = [256, 512, 1024, 2048]
    private var backendTrainingSizes: [Int] = [256, 512, 1024, 2048]
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
    private(set) var isCheckpointPending = false
    private(set) var validationSummary = ""
    private(set) var trainingProgress: WorkbenchTrainingProgress?
    @ObservationIgnored private var trainingEventBuffer = ""
    private(set) var isPreparingDataset = false
    var datasetPreparationSummary = ""
    var showClassifierDatasetSettings = false
    var classifierSourceFolderPath: String { didSet { preferences.set(classifierSourceFolderPath, forKey: "classifierSourceFolder") } }
    var classifierUseSourceFolder: Bool { didSet { preferences.set(classifierUseSourceFolder, forKey: "classifierUseSourceFolder") } }
    var classifierSourceMinimumSize: Int { didSet { preferences.set(classifierSourceMinimumSize, forKey: "classifierSourceMinimumSize") } }
    var classifierDatasetPath: String { didSet { preferences.set(classifierDatasetPath, forKey: "classifierDatasetPath") } }
    var classifierDatasetResult: WorkbenchClassifierDatasetResult?
    var specialistWorkspacePath: String { didSet { preferences.set(specialistWorkspacePath, forKey: "specialistWorkspace") } }
    var specialistClassifierPath: String { didSet { preferences.set(specialistClassifierPath, forKey: "specialistClassifier") } }
    var specialistSourceDatasetPath: String { didSet { preferences.set(specialistSourceDatasetPath, forKey: "specialistSourceDataset") } }
    var specialistRegistry: WorkbenchSpecialistRegistry?
    var selectedSpecialistID: String? { didSet { preferences.set(selectedSpecialistID, forKey: "selectedSpecialist") } }
    var specialistTarget: String { didSet { preferences.set(specialistTarget, forKey: "specialistTarget") } }
    var specialistTrainingContext: WorkbenchSpecialistTrainingContext?
    var specialistNavigationRequest = UUID()
    var comparisonNavigationRequest = UUID()
    var sourceRecoverySummary = ""
    var sourceRecoveryResults: [WorkbenchSourceRecoveryResult] = []
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
    var modelDirectory: String { didSet { preferences.set(modelDirectory, forKey: "materialModelDirectory") } }
    var depthProModelPath: String { didSet { depthProPreferences.set(depthProModelPath, forKey: NativeDepthProContract.modelPathKey) } }
    let resources: MachineResources
    @ObservationIgnored private var runner: NativeMaterialTrainingControl?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var activeWorkerId: UUID?
    @ObservationIgnored let preferences: UserDefaults
    @ObservationIgnored private let depthProPreferences: UserDefaults
    @ObservationIgnored let trashHandler: (URL) throws -> Void
    @ObservationIgnored private let workerOverride: (@MainActor ([String], String) async throws -> String)?
    @ObservationIgnored private let managedWorkspaceURL: URL
    @ObservationIgnored private let selectedCheckpointRegistryURL: URL
    @ObservationIgnored private var hasRestored = false
    @ObservationIgnored private var deferredCheckpointPaths: [String] = []
    @ObservationIgnored private var deferredCheckpointsSelectAll = false
    @ObservationIgnored private var materialBaseLocations: [String: String] {
        didSet { preferences.set(materialBaseLocations, forKey: "nativeMaterialBaseLocations") }
    }
    @ObservationIgnored private var isRestoringPreferences = false

    var samples: [WorkbenchSample] { dataset?.samples ?? [] }
    var datasetBrowserMaterials: [WorkbenchMaterial] { dataset?.browserMaterials(datasetBrowserMode) ?? [] }
    var datasetBrowserSamples: [WorkbenchSample] { datasetBrowserMaterials.flatMap(\.samples) }
    var canStopAndSave: Bool { isTraining && hasTrainingStarted && !isStopping }
    var canAbort: Bool { isBusy && (!isStopping || isSavingTraining) }
    var selectedSample: WorkbenchSample? { datasetBrowserSamples.first { $0.id == selectedSampleId } ?? samples.first { $0.id == selectedSampleId } }
    var selectedMaterialId: String? { datasetBrowserMaterials.first { $0.samples.contains { $0.id == selectedSampleId } }?.materialId
        ?? dataset?.materials.first { $0.samples.contains { $0.id == selectedSampleId } }?.materialId }
    var selectedMaterialName: String? {
        (datasetBrowserMaterials.first { $0.samples.contains { $0.id == selectedSampleId } }
            ?? dataset?.materials.first { $0.samples.contains { $0.id == selectedSampleId } })
            .map { $0.name ?? $0.materialId.replacingOccurrences(of: "_", with: " ") }
    }
    var selectedCheckpoint: WorkbenchCheckpoint? { checkpoints.first { $0.id == selectedCheckpointId } }
    var comparisonBaselineCheckpoint: WorkbenchCheckpoint? {
        checkpoints.first { comparisonCheckpointIds.contains($0.id) }
            ?? selectedCheckpoint.flatMap { $0.supportsStudioInference ? $0 : nil }
            ?? checkpoints.first { $0.supportsStudioInference && $0.modelFamily == training.modelFamily && $0.target == training.target }
            ?? checkpoints.first { $0.supportsStudioInference }
    }
    var comparisonBaselineLabel: String {
        (comparisonBaselineCheckpoint?.modelFamily ?? training.modelFamily).isCompact ?
            "Include untrained compact initialization" : "Include the material base before refinement"
    }
    var selectedDiffuseMap: WorkbenchMap? {
        selectedSample?.inputVariants?.first { $0.variantId == selectedInputVariantId } ?? selectedSample?.inputVariants?.first ?? selectedSample?.maps["input"]
    }
    var selectedMap: WorkbenchMap? { selectedRole == "input" ? selectedDiffuseMap : selectedSample?.maps[selectedRole] }
    func datasetReviewURL(_ map: WorkbenchMap) -> URL {
        map.originalSourcePath.map { URL(fileURLWithPath: $0) } ?? map.url
    }
    func datasetReviewSHA256(_ map: WorkbenchMap) -> String? { map.originalSourceSha256 ?? map.sha256 }
    func datasetDisplayTransform(_ map: WorkbenchMap, role: String) -> MapReviewDisplayTransform? {
        if selectedSample?.split == "original" { return nil }
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

    var datasetSourceFolderCount: Int {
        guard let dataset else { return 0 }
        if let count = dataset.registeredFolderCount { return count }
        return Set((dataset.originalMaterials ?? dataset.materials).compactMap { sourceFolder(for: $0) }
            + (dataset.sourceInventory ?? []).map(\.folder)).count
    }
    var datasetInventorySummary: String {
        guard let dataset else { return "No dataset selected" }
        let folders = datasetSourceFolderCount
        let sets = dataset.preparation == nil ? dataset.sourceSetCount ?? dataset.materials.count : dataset.materials.count
        let files = dataset.sourceInventory?.count ?? (dataset.originalMaterials ?? dataset.materials).flatMap(\.samples).reduce(0) { $0 + $1.maps.count }
        return "\(folders) \(folders == 1 ? "folder" : "folders") · \(files) original \(files == 1 ? "file" : "files") · \(sets) paired \(sets == 1 ? "set" : "sets")"
    }
    var trainingScopeSummary: String {
        training.useSelectedMaterialOnly ? selectedMaterialName.map { "Selected source folder: \($0) · includes all resolution tiers" }
            ?? "Select a source folder in Dataset." : "Training scope: all \(datasetSourceFolderCount) material folders"
    }
    var trainingCurrentCropSummary: String {
        guard let dataset else { return "No training crops registered" }
        let scoped = training.useSelectedMaterialOnly ? trainingFolderSamples : samples
        let count: Int
        if !training.useSelectedMaterialOnly, let plan = (dataset.resolutionPlans?[String(training.size)] ?? dataset.trainingPlans)?[training.target], plan.size == training.size {
            count = plan.trainCount
        } else { count = scoped.filter { $0.split == "train" && sampleIsTrainable($0) }.count }
        let stage = dataset.preparation == nil ? "Currently available" : "Prepared"
        return "\(stage): \(count) \(training.target) training \(count == 1 ? "crop" : "crops") at \(training.size) × \(training.size)"
    }

    private func sourceFolder(for material: WorkbenchMaterial) -> String? {
        if let path = material.sourceDirectory ?? material.subjectId {
            return URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
        }
        guard let input = material.samples.first?.maps["input"] else { return nil }
        return URL(fileURLWithPath: input.originalSourcePath ?? input.path).deletingLastPathComponent()
            .standardizedFileURL.resolvingSymlinksInPath().path
    }
    var workspaceURL: URL { URL(fileURLWithPath: workspacePath).standardizedFileURL }
    var dependencyArguments: [String] { ["--model-directory", modelDirectory] }
    func dependencyArguments(for checkpoint: WorkbenchCheckpoint) -> [String] {
        if checkpoint.modelFamily == .depthProHeightRefiner { return ["--depthpro-model", depthProModelPath] }
        if checkpoint.modelFamily?.isCompact == true { return [] }
        return ["--model-directory", baseDirectory(for: checkpoint)]
    }

    init(preferences suppliedDefaults: UserDefaults? = nil,
         depthProPreferences suppliedDepthProDefaults: UserDefaults? = nil,
         managedWorkspaceURL: URL? = nil,
         resources: MachineResources = .current,
         selectedCheckpointRegistryURL: URL = SelectedMaterialCheckpoint.registryURL,
         trashHandler: @escaping (URL) throws -> Void = { url in try FileManager.default.trashItem(at: url, resultingItemURL: nil) },
         workerOverride: (@MainActor ([String], String) async throws -> String)? = nil) {
        let defaults = suppliedDefaults ?? UserDefaults(suiteName: "org.ipde.material-tools")!
        preferences = defaults
        // A test or isolated workspace that injects its preferences keeps every
        // setting in that suite unless it explicitly supplies a shared suite.
        depthProPreferences = suppliedDepthProDefaults ?? suppliedDefaults ?? StudioPreferences.defaults
        materialBaseLocations = defaults.dictionary(forKey: "nativeMaterialBaseLocations") as? [String: String] ?? [:]
        self.trashHandler = trashHandler
        if let data = defaults.data(forKey: "recentDatasets.v1"),
           let locations = try? JSONDecoder().decode([WorkbenchDatasetLocation].self, from: data) {
            recentDatasets = locations
        }
        uploadRepo = defaults.string(forKey: "uploadRepository") ?? ""
        uploadPublic = defaults.bool(forKey: "uploadPublic")
        uploadAfterTraining = defaults.object(forKey: "uploadAfterTraining") == nil ? true : defaults.bool(forKey: "uploadAfterTraining")
        self.resources = resources
        configuredPreparationWorkers = defaults.object(forKey: "preparationWorkers") == nil
            ? resources.availableProcessorCount
            : min(resources.availableProcessorCount, max(1, defaults.integer(forKey: "preparationWorkers")))
        self.selectedCheckpointRegistryURL = selectedCheckpointRegistryURL
        self.workerOverride = workerOverride
        self.managedWorkspaceURL = (managedWorkspaceURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Material Workspace")).standardizedFileURL
        let developmentCorpus = "/opt/ipde/sources_mats"
        let hasDevelopmentCorpus = FileManager.default.fileExists(atPath: developmentCorpus)
        classifierSourceFolderPath = defaults.string(forKey: "classifierSourceFolder") ?? (hasDevelopmentCorpus ? developmentCorpus : "")
        classifierUseSourceFolder = defaults.bool(forKey: "classifierUseSourceFolder")
        let classifierSize = defaults.integer(forKey: "classifierSourceMinimumSize")
        classifierSourceMinimumSize = classifierSize > 0 ? classifierSize : 2048
        classifierDatasetPath = defaults.string(forKey: "classifierDatasetPath") ?? (hasDevelopmentCorpus ? "/opt/ipde/classifer_ds" : self.managedWorkspaceURL.appendingPathComponent("Diffuse Classifier Dataset").path)
        let workspace = Self.userWorkspaceURL(configured: defaults.string(forKey: "workspace"), managed: self.managedWorkspaceURL).path
        workspacePath = workspace
        let specialistWorkspace = defaults.string(forKey: "specialistWorkspace") ?? URL(fileURLWithPath: workspace).appendingPathComponent("Specialist Models").path
        specialistWorkspacePath = specialistWorkspace
        specialistClassifierPath = defaults.string(forKey: "specialistClassifier") ?? ""
        let specialistSourceDataset = defaults.string(forKey: "specialistSourceDataset") ?? defaults.string(forKey: "dataset") ?? ""
        specialistSourceDatasetPath = specialistSourceDataset
        selectedSpecialistID = defaults.string(forKey: "selectedSpecialist")
        specialistTarget = defaults.string(forKey: "specialistTarget") ?? "height"
        defaults.set(specialistWorkspace, forKey: "specialistWorkspace")
        defaults.set(specialistSourceDataset, forKey: "specialistSourceDataset")
        defaults.set(workspace, forKey: "workspace")
        let cache = URL(fileURLWithPath: workspace).appendingPathComponent("out/material-training/transfer-models")
        let configuredBase = defaults.string(forKey: "materialModelDirectory")
        let materialBasePath: String
        if let configuredBase, !FileManager.default.fileExists(atPath: configuredBase),
           Self.userWorkspaceURL(configured: configuredBase, managed: self.managedWorkspaceURL).path != URL(fileURLWithPath: configuredBase).standardizedFileURL.path {
            materialBasePath = cache.appendingPathComponent("pbrnxt-base").path
        } else { materialBasePath = configuredBase ?? cache.appendingPathComponent("pbrnxt-base").path }
        modelDirectory = materialBasePath
        defaults.set(materialBasePath, forKey: "materialModelDirectory")
        depthProModelPath = NativeDepthProContract.configuredModelURL(in: depthProPreferences, migrating: defaults).path
        let saved = WorkbenchPreferences.load(from: defaults)
        if let options = saved.training {
            training = options.restored(for: resources)
            if training != options {
                trainingPreferenceNotice = "Saved training settings were adapted to supported values. Review the training settings before starting."
                activity = trainingPreferenceNotice!
            }
        }
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

    func restore(arguments: [String] = CommandLine.arguments, sourceBrowsing: Bool? = nil) {
        guard !isBusy, !hasRestored else { return }
        let freshLaunch: FreshMaterialTrainingLaunch?
        do { freshLaunch = try FreshMaterialTrainingLaunch.parse(arguments: arguments) }
        catch { self.error = error.localizedDescription; return }
        if let freshLaunch { configureFreshTraining(freshLaunch) }
        let freshTrainingSize = freshLaunch.map { _ in training.size }
        hasRestored = true
        var datasetPath = preferences.string(forKey: "dataset")
        if datasetPath == nil {
            let localDataset = workspaceURL.deletingLastPathComponent().appendingPathComponent("material-dataset/dataset.json")
            if FileManager.default.fileExists(atPath: localDataset.path) { datasetPath = localDataset.path }
        }
        if let i = arguments.firstIndex(of: "--dataset"), arguments.indices.contains(i + 1) { datasetPath = arguments[i + 1] }
        if let freshLaunch { datasetPath = freshLaunch.datasetURL.path }
        let paths = preferences.stringArray(forKey: "checkpoints") ?? []
        let saved = WorkbenchPreferences.load(from: preferences)
        operation("Opening workspace…") {
            do { try await self.loadTrainingCapabilities() }
            catch { self.logText += "Training setup is unavailable: \(error.localizedDescription)\n" }
            self.isRestoringPreferences = true
            defer { self.isRestoringPreferences = false; self.saveUserSettings() }
            if let datasetPath, FileManager.default.fileExists(atPath: datasetPath) {
                let url = URL(fileURLWithPath: datasetPath)
                try await self.loadDataset(url)
                if self.dataset?.isSourceFolder == true {
                    self.specialistTrainingContext = nil
                    self.useOpenSourcesInModelWorkshop()
                    try await self.alignModelWorkshopWorkspace(with: self.modelWorkshopSourcePath)
                }
            }
            let browsing = sourceBrowsing ?? (MaterialTool.launchRole == nil && freshLaunch == nil)
            if MaterialTool.launchRole == .dataset || browsing && self.dataset?.isSourceFolder == true {
                self.deferredCheckpointPaths = paths
                self.deferredCheckpointsSelectAll = saved.comparisonCheckpointIds == nil
            } else {
                try await self.reconnectCheckpointPaths(paths, comparisonSelection: saved.comparisonCheckpointIds)
            }
            if let freshTrainingSize {
                // Dataset inspection can select a row or recover its recorded
                // review size. Fresh launch keeps the explicit training request
                // and starts with no old source or comparison selection.
                self.training.size = freshTrainingSize
                self.selectedSampleId = nil
                self.selectedInputVariantId = nil
                self.restoreFreshSpecialistTrainingContext()
                self.activity = "Ready to train a fresh RGB height model on all material folders."
            }
        }
    }

    /// Inspect saved model files only when a model workspace is requested.
    /// Source browsing retains their paths and selections without opening them.
    func reconnectSavedCheckpoints() {
        guard !isBusy, !deferredCheckpointPaths.isEmpty else { return }
        let paths = deferredCheckpointPaths
        operation("Opening saved models…") {
            self.isRestoringPreferences = true
            defer { self.isRestoringPreferences = false; self.saveUserSettings() }
            let selected = self.deferredCheckpointsSelectAll && self.comparisonCheckpointIds.isEmpty ? nil : self.comparisonCheckpointIds
            try await self.reconnectCheckpointPaths(paths, comparisonSelection: selected)
            self.deferredCheckpointPaths = []
            self.deferredCheckpointsSelectAll = false
        }
    }

    private func reconnectCheckpointPaths(_ paths: [String], comparisonSelection: Set<String>?) async throws {
        for path in paths where FileManager.default.fileExists(atPath: path) {
            try Task.checkCancellation()
            if checkpoints.contains(where: { $0.checkpointPath == path }) { continue }
            do { try await loadCheckpoint(URL(fileURLWithPath: path), select: false) }
            catch is CancellationError { throw CancellationError() }
            catch { logText += "Could not reconnect \(path): \(error.localizedDescription)\n" }
        }
        if !checkpoints.contains(where: { $0.id == selectedCheckpointId }) { selectedCheckpointId = checkpoints.first?.id }
        let available = Set(checkpoints.map(\.id))
        comparisonCheckpointIds = comparisonSelection.map { $0.intersection(available) } ?? available
    }

    private func saveCheckpointPaths() {
        var seen = Set<String>()
        let paths = (deferredCheckpointPaths + checkpoints.map(\.checkpointPath)).filter { seen.insert($0).inserted }
        preferences.set(paths, forKey: "checkpoints")
    }

    private func configureFreshTraining(_ launch: FreshMaterialTrainingLaunch) {
        isRestoringPreferences = true
        var options = training.restored(for: resources)
        options.modelFamily = .compactScalar
        options.target = "height"
        options.scope = "full-model"
        options.useWarmStart = false
        options.useSelectedMaterialOnly = false
        if let size = launch.trainingSize { options.size = size }
        if let rate = launch.learningRate { options.learningRate = rate }
        if let updates = launch.updatesPerMap { options.updatesPerCrop = updates }
        training = options
        dataset = nil
        selectedSampleId = nil
        selectedInputVariantId = nil
        selectedRole = "height"
        checkpoints = []
        deferredCheckpointPaths = []
        deferredCheckpointsSelectAll = false
        selectedCheckpointId = nil
        comparisonCheckpointIds = []
        comparisonIncludesBase = false
        comparisonCandidates = []
        sourceImageURL = nil
        lastOutputURL = nil
        lastLogURL = nil
        lastPackageURL = nil
        lastPackageCheckpointId = nil
        lastUploadURL = nil
        specialistTrainingContext = nil
        adapterMix = []
        datasetPreparationSummary = ""
        sourceRecoverySummary = ""
        sourceRecoveryResults = []
        trainingPreferenceNotice = nil
        preferences.set([String](), forKey: "checkpoints")
        preferences.set(launch.datasetURL.path, forKey: "dataset")
        isRestoringPreferences = false
        saveUserSettings()
    }

    private func restoreFreshSpecialistTrainingContext() {
        guard let folder = datasetFolderURL, let specialistID = selectedSpecialistID else { return }
        let bindingURL = folder.appendingPathComponent(".specialist-binding.json")
        guard let data = try? Data(contentsOf: bindingURL),
              let binding = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              binding["schema"] as? String == NativeMaterialSpecialistService.schema,
              binding["specialist_id"] as? String == specialistID,
              let boundWorkspace = binding["workspace_path"] as? String else { return }
        func canonical(_ path: String) -> URL {
            URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        }
        guard canonical(boundWorkspace) == canonical(specialistWorkspacePath) else { return }
        let registryURL = canonical(specialistWorkspacePath).appendingPathComponent("specialists.json")
        guard let registryText = try? String(contentsOf: registryURL, encoding: .utf8),
              let registry = try? WorkbenchResult.decode(WorkbenchSpecialistRegistry.self, output: registryText),
              canonical(registry.workspacePath) == canonical(specialistWorkspacePath),
              let specialist = registry.specialists.first(where: {
                  $0.id == specialistID && canonical($0.datasetPath) == folder.standardizedFileURL.resolvingSymlinksInPath()
              }),
              let height = specialist.targetInfo("height"), height.materialCount > 0,
              !height.membershipSha256.isEmpty, let datasetPath = dataset?.datasetPath else { return }
        specialistRegistry = registry
        specialistTarget = "height"
        specialistTrainingContext = .init(workspacePath: registry.workspacePath, specialistID: specialist.id,
            name: specialist.name, target: "height", datasetPath: datasetPath, membershipSHA256: height.membershipSha256)
        training.modelName = "\(specialist.name) Displacement"
    }

    func saveConfiguration(refreshSelectedRuntime: Bool = true) {
        preferences.set(workspacePath, forKey: "workspace")
        preferences.set(modelDirectory, forKey: "materialModelDirectory")
        saveUploadConfiguration()
        if refreshSelectedRuntime {
            do {
                for target in ["height", "roughness", "normal"] {
                    let registry = SelectedMaterialCheckpoint.registryURL(for: target, heightRegistryURL: selectedCheckpointRegistryURL)
                    let selectedBase = (try? SelectedMaterialCheckpoint.read(from: registry))?.modelDirectory
                    let base = selectedBase.flatMap { materialBaseLocations.values.contains($0) ? $0 : nil } ?? modelDirectory
                    try SelectedMaterialCheckpoint.refreshRuntime(workspacePath: workspacePath,
                        modelDirectory: base, at: registry)
                }
            } catch {
                self.error = "Runtime settings were saved, but Texture Studio could not reconnect its selected checkpoint: \(error.localizedDescription)"
            }
        }
    }

    func chooseDataset() {
        guard !isBusy else { return }
        let panel = NSOpenPanel()
        panel.title = "Open Source Folder"
        panel.message = "Choose a folder of original images. Its current files appear automatically."
        panel.prompt = "Open Folder"
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
        operation("Opening source folder…") {
            try await self.loadDataset(url)
            self.specialistTrainingContext = nil
            self.useOpenSourcesInModelWorkshop()
            if self.dataset?.isSourceFolder == true { self.classifierUseSourceFolder = false }
            try await self.alignModelWorkshopWorkspace(with: self.modelWorkshopSourcePath)
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
    func chooseWorkspace() {
        chooseFolder(title: "Choose a working folder for material runs") { url in self.workspacePath = url.path; self.saveConfiguration() }
    }
    func chooseEncoder() {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.title = "Locate material weights or their folder"
        panel.canChooseFiles = true; panel.canChooseDirectories = true
        panel.begin { response in
            if response == .OK, let url = panel.url { self.modelDirectory = url.path; self.saveConfiguration() }
        }
    }

    func chooseDepthProModel() {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.title = "Locate the DepthPro Core ML package"
        panel.canChooseFiles = true; panel.canChooseDirectories = true
        panel.treatsFilePackagesAsDirectories = false
        panel.begin { response in
            if response == .OK, let url = panel.url { self.depthProModelPath = url.path }
        }
    }

    func loadDataset(_ url: URL, recoverSources: Bool = false, reviewSize: Int? = nil) async throws {
        var result: WorkbenchDataset = try WorkbenchResult.decode(WorkbenchDataset.self,
            output: await worker(["dataset", "--dataset", url.path, "--target", training.target,
                                  "--default-review-size", String(training.size)]
                + (reviewSize.map { ["--review-size", String($0)] } ?? [])
                + (recoverSources ? [] : ["--skip-source-recovery"])))
        if let reviewSize {
            result.trainingSize = reviewSize
            result.reviewSize = reviewSize
        }
        adoptDataset(result)
        datasetPreparationSummary = ""
        sourceRecoverySummary = result.sourceRecovery?.summary ?? ""
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
        observeSourceFolder(result.isSourceFolder == true ? URL(fileURLWithPath: result.sourceFolderPath ?? result.datasetPath) : nil)
        sourceRecoverySummary = result.sourceRecovery?.summary ?? ""
        sourceRecoveryResults = result.sourceRecoveryResults ?? []
        if let size = result.trainingSize {
            training.size = size
        }
        supportedTrainingSizes = backendTrainingSizes.filter { result.supportedTrainingSizes?.contains($0) ?? true || result.samples.contains(where: canRecoverCropTiers) }
        let browserMaterials = result.browserMaterials(datasetBrowserMode)
        let browserSamples = browserMaterials.flatMap(\.samples)
        if !browserSamples.contains(where: { $0.id == selectedSampleId }) {
            selectedSampleId = browserMaterials.first(where: { $0.id == material })?.samples.first?.id ?? browserSamples.first?.id
        }
    }

    func selectTrainingSize(_ size: Int) {
        guard !isBusy else { return }
        guard supportedTrainingSizes.contains(size) else { error = "Choose a training size supplied by the original source maps."; return }
        guard dataset != nil else { training.size = size; return }
        updateDatasetInfo(name: datasetName, description: datasetDescription, size: size)
    }

    func selectTrainingTarget(_ target: String) {
        guard !isBusy, ["height", "roughness", "normal"].contains(target) else { return }
        training.target = target
        if training.modelFamily.isCompact && !training.modelFamily.supports(target: target) {
            training.modelFamily = target == "normal" ? .compactNormal : .compactScalar
        }
        clearIncompatibleTrainingCheckpoint()
    }

    func selectTrainingModelFamily(_ family: MaterialTrainingModelFamily) {
        guard !isBusy, family != training.modelFamily else { return }
        let previous = training.modelFamily
        training.modelFamily = family
        if previous.isCompact != family.isCompact { training.learningRate = family.defaultLearningRate }
        if !family.supports(target: training.target) { training.target = family == .compactNormal ? "normal" : "height" }
        if !family.isCompact, !["final-map", "map-decoder"].contains(training.scope) { training.scope = "final-map" }
        clearIncompatibleTrainingCheckpoint()
    }

    private func clearIncompatibleTrainingCheckpoint() {
        if let checkpoint = selectedCheckpoint, !checkpoint.matchesTraining(training) {
            selectedCheckpointId = nil
            training.useWarmStart = false
        } else if selectedCheckpoint == nil { training.useWarmStart = false }
    }

    func prepareTrainingDataset() {
        guard !isBusy, dataset != nil else { return }
        let options = training
        let size = options.size
        let selectedMaterial = selectedMaterialId
        let material = options.useSelectedMaterialOnly ? selectedMaterial : nil
        guard dataset?.readyForTraining(size: size, material: material, target: options.target) != true else { return }
        operation("Preparing \(size) × \(size) complete training maps…") {
            _ = try await self.ensureTrainingDataset(options: options, selectedMaterial: selectedMaterial)
            self.activity = self.datasetPreparationSummary
        }
    }

    func refreshTrainingTiers() {
        guard !isBusy, let source = datasetFolderURL else { return }
        let size = training.size, target = training.target
        operation("Refreshing balanced source tiers…") {
            let result = try WorkbenchResult.decode(WorkbenchDataset.self, output: await self.worker([
                "dataset", "--dataset", source.path, "--size", String(size), "--review-size", String(size), "--target", target]))
            self.adoptDataset(result)
            self.training.size = size
            self.sourceRecoverySummary = result.sourceRecovery?.summary ?? "Source tier check completed."
            self.datasetPreparationSummary = ""
            self.activity = self.sourceRecoverySummary
        }
    }

    private func ensureTrainingDataset(options: MaterialTrainingOptions, selectedMaterial: String?) async throws -> WorkbenchDataset {
        let size = options.size
        let target = options.target
        guard let current = dataset else { throw StudioError("Open a material dataset first.") }
        guard supportedTrainingSizes.contains(size) else { throw StudioError("Choose a training size supported by the model and original source maps.") }
        let checkMaterial = options.useSelectedMaterialOnly ? selectedMaterial : nil
        isPreparingDataset = true
        activity = "Preparing \(size) × \(size) complete training maps from the original materials…"
        defer { isPreparingDataset = false }
        var arguments = [
            "prepare-size", "--dataset", current.isSourceFolder == true ? current.sourceFolderPath ?? current.datasetPath : current.datasetPath, "--size", String(size), "--target", target,
            "--automatic-validation", "--expected-index-sha256", current.indexSha256]
        arguments += ["--crop-seed", String(options.randomizeCrops == .never ? 0 : UInt64.random(in: 1...UInt64.max))]
        if let reviewHash = current.reviewSha256 { arguments += ["--expected-review-sha256", reviewHash] }
        if let checkMaterial { arguments += ["--material", checkMaterial] }
        let result: WorkbenchDataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: await worker(arguments))
        guard result.readyForTraining(size: size, material: checkMaterial, target: target), let preparation = result.preparation,
              preparation.cropSize == size, !preparation.originalDatasetModified,
              URL(fileURLWithPath: preparation.preparedDatasetPath).standardizedFileURL == URL(fileURLWithPath: result.datasetPath).standardizedFileURL else {
            throw StudioError("Dataset preparation did not verify matching map dimensions and preserved originals.")
        }
        try Task.checkCancellation()
        adoptDataset(result, preferredMaterial: selectedMaterial)
        preferences.set(preparation.sourceDatasetPath, forKey: "dataset")
        datasetPreparationSummary = "\(size) × \(size) native pixel maps: matching originals are referenced directly; each crop is saved once in temporary training storage."
        return result
    }
    func loadCheckpoint(_ url: URL, select: Bool = true) async throws {
        let output = try await worker(["checkpoint", "--checkpoint", url.path])
        let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: output)
        if checkpoint.variant == "full" { materialBaseLocations[checkpoint.sha256] = checkpoint.checkpointPath }
        guard checkpoint.compatible else { throw StudioError("This checkpoint is not supported by the material backend.") }
        if let i = checkpoints.firstIndex(where: { $0.id == checkpoint.id }) { checkpoints[i] = checkpoint }
        else { checkpoints.append(checkpoint) }
        if select {
            selectedCheckpointId = checkpoint.id
            comparisonCheckpointIds.insert(checkpoint.id)
        }
        if !isRestoringPreferences { saveCheckpointPaths() }
    }

    func baseDirectory(for checkpoint: WorkbenchCheckpoint) -> String {
        checkpoint.base.flatMap { materialBaseLocations[$0.sha256] } ?? modelDirectory
    }

    func curateSelected(status: String, split: String? = nil, note: String? = nil) {
        guard let dataset, let sample = selectedSample else { return }
        operation("Saving material review…") {
            var args = ["curate", "--dataset", dataset.datasetPath, "--sample", sample.id,
                        "--status", status, "--expected-index-sha256", dataset.indexSha256, "--review-size", String(self.training.size)]
            if let reviewHash = dataset.reviewSha256 { args += ["--expected-review-sha256", reviewHash] }
            if let split { args += ["--split", split] }
            if let note { args += ["--note", note] }
            if sample.split == "original" { args += ["--original-source", "true"] }
            if dataset.isSourceFolder == true, let folder = dataset.sourceFolderPath {
                if let index = args.firstIndex(of: "--dataset") { args[index + 1] = folder }
            }
            _ = try await self.worker(args)
            try await self.loadDataset(URL(fileURLWithPath: dataset.sourceFolderPath ?? dataset.datasetPath))
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
        if let issue = comparisonConfigurationIssue { error = issue; return }
        let diffuseMap = sourceImageURL == nil ? selectedDiffuseMap : nil
        guard let image = sourceImageURL ?? diffuseMap.map(datasetReviewURL) else {
            error = "Choose a surface photo or select a dataset diffuse map first."; return
        }
        let selected = checkpoints.filter { comparisonCheckpointIds.contains($0.id) }
        let includeBase = comparisonIncludesBase
        let baselineCheckpoint = comparisonBaselineCheckpoint
        let baselineFamily = baselineCheckpoint?.modelFamily ?? training.modelFamily
        let target = selected.first?.target ?? baselineCheckpoint?.target ??
            (baselineFamily.supports(target: training.target) ? training.target : baselineFamily == .compactNormal ? "normal" : "height")
        let baselineSeed = training.seed
        let baselineWidth = training.compactWidth
        let baselineDependencies = baselineCheckpoint.map { dependencyArguments(for: $0) } ??
            (baselineFamily == .depthProHeightRefiner ? ["--depthpro-model", depthProModelPath] : baselineFamily.isCompact ? [] : dependencyArguments)
        let sampleLabel = sourceImageURL == nil ? selectedSampleId ?? image.lastPathComponent : image.lastPathComponent
        let referenceMap = sourceImageURL == nil ? selectedSample?.maps[target] : nil
        let referenceSampleID = sourceImageURL == nil ? selectedSampleId : nil
        let diffuseTransform = diffuseMap.flatMap { datasetDisplayTransform($0, role: "input") }
        let referenceTransform = referenceMap.flatMap { datasetDisplayTransform($0, role: target) }
        let referenceURL = referenceMap.map(datasetReviewURL)
        let photoSettings = testPhotoSettings
        let comparisonSize = training.size
        let inputIsPhoto = sourceImageURL != nil
        let inputEncoding = diffuseMap?.encoding ?? "srgb"
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
            let usesDepthPro = selected.contains { $0.modelFamily == .depthProHeightRefiner } || includeBase && baselineFamily == .depthProHeightRefiner
            var candidates = [MapReviewCandidate(id: "diffuse|" + reviewedDiffuse.path, label: usesDepthPro ? "Diffuse · DepthPro source" : "Diffuse · model input", mapURL: reviewedDiffuse, numeric: false,
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
            if includeBase {
                let compact = baselineFamily.isCompact
                self.activity = compact ? "Running untrained compact initialization" : "Running the material base model"
                var arguments = ["infer", "--baseline", "--image", modelImage.path,
                    "--input-encoding", inputEncoding,
                    "--output", parent.appendingPathComponent("base-untrained").path] + baselineDependencies
                if let reference = baselineCheckpoint {
                    arguments += ["--checkpoint", reference.checkpointPath, "--expected-sha256", reference.sha256]
                } else {
                    arguments += ["--model-family", baselineFamily.rawValue, "--target", target, "--seed", String(baselineSeed)]
                    if compact { arguments += ["--compact-width", String(baselineWidth)] }
                }
                let result: MaterialInferenceResponse = try WorkbenchResult.decode(MaterialInferenceResponse.self, output: await self.worker(arguments))
                if let path = result.preparedInputPath {
                    candidates.append(MapReviewCandidate(id: "depthpro|" + path, label: "DepthPro · refiner input", mapURL: URL(fileURLWithPath: path), numeric: true,
                        sampleLabel: sampleLabel, detail: "One-channel normalized inverse DepthPro depth before refinement", role: "model-input",
                        modelIdentity: MapReviewModelIdentity(mapType: target, modelName: "DepthPro refiner input")))
                }
                guard baselineCheckpoint.map({ result.checkpointSha256 == $0.sha256 }) ?? true,
                      let map = result.outputs[target] else {
                    throw StudioError("The baseline comparison did not match the requested model architecture and map type.")
                }
                let initializationDetail = baselineCheckpoint == nil ? "seed \(baselineSeed) · SHA256 \(result.checkpointSha256)" : "its recorded random initialization"
                candidates.append(MapReviewCandidate(id: "base|" + (baselineCheckpoint?.id ?? result.checkpointSha256), label: "\(compact ? "Untrained initialization" : "Base") · \(targetLabel)",
                    mapURL: URL(fileURLWithPath: map.path), numeric: true, sampleLabel: sampleLabel,
                    detail: compact ? "Compact model prediction from \(initializationDetail) before training" : "Material base prediction before refinement", role: "base",
                    modelIdentity: MapReviewModelIdentity(architecture: baselineCheckpoint?.trainingBaseLabel ?? baselineFamily.architecture ?? "PBRnxt material model",
                        mapType: target, modelName: compact ? "Untrained compact initialization" : "Material base")))
            }
            for (i, checkpoint) in selected.enumerated() {
                self.activity = "Running checkpoint \(i + 1)/\(selected.count): \(checkpoint.title)"
                let child = parent.appendingPathComponent("candidate-\(i + 1)")
                let result: MaterialInferenceResponse = try WorkbenchResult.decode(MaterialInferenceResponse.self, output: await self.worker([
                    "infer", "--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256,
                    "--image", modelImage.path, "--input-encoding", inputEncoding,
                    "--output", child.path] + self.dependencyArguments(for: checkpoint)))
                if let path = result.preparedInputPath, !candidates.contains(where: { $0.role == "model-input" }) {
                    candidates.append(MapReviewCandidate(id: "depthpro|" + path, label: "DepthPro · refiner input", mapURL: URL(fileURLWithPath: path), numeric: true,
                        sampleLabel: sampleLabel, detail: "One-channel normalized inverse DepthPro depth before refinement", role: "model-input",
                        modelIdentity: MapReviewModelIdentity(mapType: target, modelName: "DepthPro refiner input")))
                }
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
            material["diffuse_source_encoding"] = inputEncoding
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
        let result = try WorkbenchResult.decode(WorkbenchTrainingCapabilities.self,
            output: await worker(["capabilities", "--scope", training.effectiveScope, "--model-family", training.modelFamily.rawValue]))
        backendTrainingSizes = result.trainingSizes.filter { [256, 512, 1024, 2048].contains($0) }.sorted()
        supportedTrainingSizes = backendTrainingSizes.filter { dataset?.supportedTrainingSizes?.contains($0) ?? true || samples.contains(where: canRecoverCropTiers) }
        if dataset?.trainingSize == nil, !supportedTrainingSizes.contains(training.size), let size = supportedTrainingSizes.last {
            training.size = size
        }
    }

    func refreshTrainingCapabilities() {
        guard !isBusy else { return }
        operation("Checking supported training sizes…") {
            let previousSize = self.training.size
            try await self.loadTrainingCapabilities()
            if previousSize != self.training.size, let datasetURL = self.datasetURL { try await self.loadDataset(datasetURL) }
        }
    }

    var suggestedTrainingModelName: String {
        if training.useWarmStart,
           let name = selectedCheckpoint?.modelName?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty {
            return name
        }
        let map = ["height": "Displacement", "roughness": "Roughness", "normal": "Normals"][training.target] ?? training.target.capitalized
        return training.modelFamily == .depthProHeightRefiner ? "\(datasetName) DepthPro Height Refiner" : "\(datasetName) \(map)"
    }

    var effectiveTrainingModelName: String {
        let name = training.modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        return name.isEmpty ? suggestedTrainingModelName : name
    }

    var trainingConfigurationIssue: String? {
        if let issue = training.configurationIssue { return issue }
        if dataset == nil { return "Open your source dataset first." }
        if !supportedTrainingSizes.contains(training.size) {
            if dataset?.supportedTrainingSizes?.contains(training.size) == false { return "Choose a training size supplied by the original source maps." }
            return "Choose a supported training size."
        }
        if training.useSelectedMaterialOnly && selectedMaterialId == nil { return "Select a material first." }
        let selected = training.useSelectedMaterialOnly ? trainingFolderSamples : samples
        let eligible = selected.filter { self.sampleIsTrainable($0) || !["excluded", "rejected"].contains($0.status) && canRecoverCropTiers($0) }
        if eligible.isEmpty { return "No included material provides a registered \(training.target == "height" ? "16-bit displacement" : training.target) target at this size." }
        if eligible.allSatisfy({ $0.split == "validation" && $0.splitAssignment == "manual" }) {
            return "Assign at least one included material to Training in Dataset. All available materials are assigned to Validation."
        }
        if training.useWarmStart {
            guard let checkpoint = selectedCheckpoint, checkpoint.supportsTrainingWarmStart else { return "Select a material checkpoint to refine." }
            if checkpoint.modelFamily != training.modelFamily { return "Choose a starting checkpoint from the selected model family." }
            if training.modelFamily.isCompact, checkpoint.target != training.target { return "Use the compact checkpoint's recorded \(checkpoint.target) target when continuing training." }
            if checkpoint.schema == "texture-studio-material-lora-v1", checkpoint.target != training.target || (checkpoint.scope ?? "final-map") != training.scope {
                return "Use the selected LoRA's \(checkpoint.target) target and \(checkpoint.scope ?? "final-map") scope when refining it."
            }
        }
        return nil
    }

    func sampleIsTrainable(_ sample: WorkbenchSample) -> Bool {
        !["excluded", "rejected"].contains(sample.status) && min(sample.width, sample.height) >= training.size &&
            (sample.availableTargets?.contains(training.target) ?? (sample.maps[training.target] != nil))
    }

    /// Training selects a folder's balanced zoom tiers, including originals
    /// which preparation will obtain from the registered provider when absent.
    var trainingFolderSamples: [WorkbenchSample] {
        guard let dataset else { return [] }
        let materials = dataset.materials
        guard let selected = (dataset.originalMaterials ?? materials).first(where: { $0.id == selectedMaterialId })
                ?? materials.first(where: { $0.id == selectedMaterialId }) else { return [] }
        guard let selectedFolder = sourceFolder(for: selected) else { return selected.samples }
        let family = selected.samples.first?.sourceFamilyId
        return materials.filter { material in
            sourceFolder(for: material) == selectedFolder && (family == nil || material.samples.first?.sourceFamilyId == nil || material.samples.first?.sourceFamilyId == family)
        }.flatMap(\.samples)
    }

    private func canRecoverCropTiers(_ sample: WorkbenchSample) -> Bool {
        guard let input = sample.maps["input"] else { return false }
        let source = URL(fileURLWithPath: input.originalSourcePath ?? input.path)
        let filename = source.lastPathComponent.lowercased()
        let scale: Int?
        if filename.range(of: #"_(?:diff|col(?:_.+)?)_(?:1|2|4|8|16)k\.png$"#, options: .regularExpression) != nil {
            scale = source.deletingPathExtension().lastPathComponent.split(separator: "_").last.flatMap { Int($0.dropLast()) }
        } else if filename.range(of: #"^[a-z][a-z0-9]*_[1248]k-png_color\.png$"#, options: .regularExpression) != nil,
                  let tier = filename.range(of: #"(?<=_)[1248](?=k-png_color\.png$)"#, options: .regularExpression) {
            scale = Int(filename[tier])
        } else { return false }
        guard let scale else { return false }
        let pixels = scale * 1024, tolerance = max(2, Int(ceil(Double(pixels) * 0.02)))
        let width = input.originalSourceWidth ?? sample.width, height = input.originalSourceHeight ?? sample.height
        // Published source fields can be rectangular. A verified nominal axis
        // identifies their tier; preparation checks the final crop's short edge.
        return width > 0 && height > 0 &&
            (abs(width - pixels) <= tolerance || abs(height - pixels) <= tolerance)
    }

    func startTraining() {
        if let issue = trainingConfigurationIssue { error = issue; return }
        runTraining(checkpoint: training.useWarmStart ? selectedCheckpoint : nil)
    }

    private func runTraining(checkpoint: WorkbenchCheckpoint?) {
        let options = training
        let specialistContext = specialistTrainingContext.flatMap { context in
            !options.useSelectedMaterialOnly && context.datasetPath == (dataset?.preparation?.sourceDatasetPath ?? dataset?.datasetPath) && context.target == options.target ? context : nil
        }
        let modelName = effectiveTrainingModelName
        let selectedMaterial = selectedMaterialId
        let dependencies = options.modelFamily.isCompact ? [] : checkpoint.map { dependencyArguments(for: $0) } ?? dependencyArguments
        let depthProArguments = options.modelFamily == .depthProHeightRefiner ? ["--depthpro-model", depthProModelPath] : []
        let developer = developerMode
        let publishAfterTraining = developer && uploadAfterTraining
        operation(options.modelFamily == .depthProHeightRefiner ? "Training DepthPro Height Refiner…" : options.modelFamily.isCompact ? "Training compact material model…" : checkpoint == nil ? "Training material LoRA…" : "Refining material LoRA…", training: true) {
            let prepared = try await self.ensureTrainingDataset(options: options, selectedMaterial: selectedMaterial)
            let output = try self.newOutputURL(prefix: "material-\(options.target)")
            self.lastOutputURL = output
            var args = [checkpoint == nil ? "train" : "refine", "--dataset", prepared.datasetPath,
                "--output", output.path, "--size", String(options.size), "--whole-maps",
                "--model-name", modelName,
                "--target", options.target, "--scope", options.effectiveScope, "--model-family", options.modelFamily.rawValue,
                "--max-minutes", String(options.maxMinutes),
                "--updates-per-map", String(options.updatesPerCrop),
                "--validation-every", String(options.validationEvery), "--checkpoint-every", String(options.checkpointEvery),
                "--validation-unit", options.validationUnit.rawValue, "--checkpoint-unit", options.checkpointUnit.rawValue,
                "--learning-rate", String(options.learningRate),
                "--gradient-accumulation-steps", String(options.gradientAccumulationSteps),
                "--optimizer", options.optimizer,
                "--optimizer-beta1", String(options.optimizerBeta1), "--optimizer-beta2", String(options.optimizerBeta2),
                "--optimizer-epsilon", String(options.optimizerEpsilon), "--weight-decay", String(options.weightDecay),
                "--max-gradient-norm", String(options.maxGradientNorm),
                "--learning-rate-schedule", options.learningRateSchedule,
                "--minimum-learning-rate-ratio", String(options.minimumLearningRateRatio),
                "--warmup-updates", String(options.warmupUpdates), "--seed", String(options.seed)] + dependencies
            args += depthProArguments
            args += options.cropArguments
            args += ["--crop-selection-prepared"]
            if options.modelFamily.isCompact && checkpoint == nil { args += ["--compact-width", String(options.compactWidth)] }
            if !options.modelFamily.isCompact { args += ["--lora-rank", String(options.loraRank), "--lora-alpha", String(options.loraAlpha)] }
            if developer { args += ["--developer-mode"] }
            // Preparation already restricted the selected folder across all zoom
            // tiers; filtering its original resolution ID would discard the rest.
            if let checkpoint { args += ["--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256] }
            self.isResumingTraining = checkpoint != nil
            let result: WorkbenchTrainingResponse
            do {
                result = try WorkbenchResult.decode(WorkbenchTrainingResponse.self, output: await self.worker(args))
            } catch {
                // Cancellation must not prevent cleanup of a positively owned stage.
                await Task { @MainActor in try? await self.cleanupTrainingDataset(prepared) }.value
                throw error
            }
            // The worker has published the final weights. Library refresh, temporary
            // file cleanup and optional Hub upload cannot undo this completed run.
            var notices: [String] = []
            self.trainingProgress?.finish(state: result.status == "stopped" ? .stopped : .completed,
                completed: result.completedUpdates, total: result.requestedUpdates)
            self.lastPackageURL = result.packagePath.map { URL(fileURLWithPath: $0) }
            do {
                try await self.loadCheckpoint(URL(fileURLWithPath: result.checkpointPath))
                self.lastPackageCheckpointId = self.selectedCheckpointId
            } catch {
                self.lastPackageCheckpointId = nil
                notices.append("Model saved at \(result.checkpointPath). Could not refresh the model library: \(error.localizedDescription)")
            }
            if let specialistContext, (result.completedUpdates ?? 0) > 0 {
                do {
                    try await self.recordSpecialistCheckpoint(context: specialistContext,
                        checkpointPath: result.checkpointPath, trainingSize: options.size,
                        completedUpdates: result.completedUpdates ?? 0)
                    notices.append("Saved in \(specialistContext.name). Return to Loom Room to compare and approve it for automatic generation.")
                } catch {
                    notices.append("Model saved; could not update its specialist record: \(error.localizedDescription)")
                }
            }
            if result.stoppedReason == "time_limit" {
                let counts = result.completedUpdates.flatMap { completed in
                    result.requestedUpdates.map { " Saved \(completed.formatted()) of \($0.formatted()) updates." }
                } ?? " Saved the completed updates."
                self.activity = "Training time limit reached." + counts
            } else if result.stoppedReason == "no_valid_training_samples" {
                self.activity = result.completedUpdates == 0 ?
                    "No valid training samples remain. Saved the current weights without training." :
                    "No valid training samples remain. Saved \((result.completedUpdates ?? 0).formatted()) completed steps."
            } else if self.isSavingTraining || result.status == "stopped" {
                self.activity = options.modelFamily.isCompact ? "Stopped and saved compact material model." : "Stopped and saved material LoRA."
            } else {
                self.activity = "Training finished. Review the material maps before using this model."
            }
            let completionActivity = self.activity
            do {
                try await Task { @MainActor in try await self.cleanupTrainingDataset(prepared) }.value
            } catch {
                notices.append("Model saved; \(error.localizedDescription)")
            }
            if publishAfterTraining && !self.isStopping && !Task.isCancelled {
                do {
                    let account = try WorkbenchResult.decode(HuggingFaceAccountResponse.self, output: await self.worker(["hub-account"]))
                    self.uploadAccount = account.authenticated ? account.username : nil
                    self.uploadAccountChecked = true
                    self.uploadAccountMessage = account.message
                    if let checkpoint = self.selectedCheckpoint,
                       checkpoint.url.standardizedFileURL == URL(fileURLWithPath: result.checkpointPath).standardizedFileURL,
                       self.canUploadSelectedCheckpoint {
                        try await self.performUpload(checkpoint)
                        notices.append(self.activity)
                    } else {
                        notices.append(account.authenticated ? "Saved locally; locate the saved model in Saved Models to upload it." :
                            "Saved locally; sign in to Hugging Face to publish it.")
                    }
                } catch {
                    notices.append("Model saved locally; Hugging Face upload did not finish: \(error.localizedDescription)")
                }
            }
            self.activity = completionActivity + (notices.isEmpty ? "" : " " + notices.joined(separator: " "))
            if !notices.isEmpty {
                self.logText += "\n" + notices.joined(separator: "\n") + "\n"
            }
        }
    }

    private func cleanupTrainingDataset(_ prepared: WorkbenchDataset) async throws {
        guard let preparation = prepared.preparation else { return }
        var notices: [String] = []
        do {
            let output = try await worker(["cleanup-size", "--dataset", prepared.datasetPath])
            _ = try WorkbenchResult.decode(WorkbenchDatasetCleanup.self, output: output)
        } catch {
            notices.append("could not finish training-file cleanup at \(prepared.datasetPath): \(error.localizedDescription)")
        }
        // Reopening the durable source is independent of removing temporary
        // files. A cleanup failure must not strand the UI on a partial stage.
        do {
            try await loadDataset(URL(fileURLWithPath: preparation.sourceDatasetPath))
            datasetPreparationSummary = notices.isEmpty ?
                "Training files cleared. Original source maps are selected." :
                "Original source maps are selected. Temporary training files remain at \(prepared.datasetPath)."
        } catch {
            notices.append("could not reopen the source dataset at \(preparation.sourceDatasetPath): \(error.localizedDescription)")
        }
        if !notices.isEmpty { throw StudioError(notices.joined(separator: "; ")) }
    }

    func chooseResumeCheckpoint() {
        chooseFile(title: "Choose material safetensors to refine") { self.resumeTraining(from: $0) }
    }

    func resumeTraining(from url: URL) {
        operation("Opening material checkpoint…") {
            try await self.loadCheckpoint(url)
            if let checkpoint = self.selectedCheckpoint {
                guard checkpoint.supportsTrainingWarmStart else { throw StudioError("This checkpoint does not support material refinement.") }
                self.configureCheckpointForRefinement(checkpoint)
            }
        }
    }

    func receiveResumeCheckpoint(_ url: URL) {
        if isBusy { queuedResumeURL = url }
        else { resumeTraining(from: url) }
    }

    func receiveTrainingHandoff(_ handoff: MaterialTrainingHandoff) {
        if isBusy { queuedTrainingHandoff = handoff; return }
        operation("Opening the selected model in Trainer…") {
            try await self.loadCheckpoint(handoff.checkpointURL, select: false)
            guard let checkpoint = self.checkpoints.first(where: { $0.url.standardizedFileURL == handoff.checkpointURL.standardizedFileURL }),
                  checkpoint.sha256.lowercased() == handoff.checkpointSHA256,
                  checkpoint.supportsTrainingWarmStart else {
                throw StudioError("The requested checkpoint changed or cannot be refined. Choose the model again.")
            }
            guard checkpoint.modelFamily == handoff.training.modelFamily,
                  !handoff.training.modelFamily.isCompact || checkpoint.target == handoff.training.target else {
                throw StudioError("The requested training family or target does not match this compact checkpoint.")
            }
            self.training = handoff.training
            if let dataset = handoff.datasetURL { try await self.loadDataset(dataset) }
            self.training = handoff.training
            if let sample = handoff.sampleID {
                guard self.samples.contains(where: { $0.id == sample }) else {
                    throw StudioError("The requested material is no longer in this dataset. Select it in Dataset before training.")
                }
                self.selectedSampleId = sample
            }
            self.selectedInputVariantId = handoff.inputVariantID
            self.configureCheckpointForRefinement(checkpoint)
            self.activity = "Ready to refine \(checkpoint.title)."
            self.trainingNavigationRequest = UUID()
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

    @discardableResult func reportMissingDatasetSource(_ url: URL) -> String {
        if dataset?.isSourceFolder == true {
            sourceFolderNeedsRefresh = true
            refreshSourceFolderIfNeeded()
            let notice = "Original moved or removed: \(url.path). The source folder list will refresh."
            activity = notice
            return notice
        }
        let notice = "Original file is unavailable at \(url.path). Its reference is kept in this dataset."
        activity = notice
        return notice
    }

    func stop() {
        if isTraining { stopAndSave() }
        else { abort() }
    }

    func abort() {
        guard canAbort else { return }
        isStopping = true
        isSavingTraining = false
        activity = isTraining ? (hasTrainingStarted ? "Aborting training…" : "Aborting training setup…") : "Stopping the operation…"
        runner?.stop()
        task?.cancel()
    }

    func stopAndSave() {
        guard canStopAndSave else {
            if !isStopping { abort() }
            return
        }
        isStopping = true
        isSavingTraining = true
        activity = training.modelFamily.isCompact ? "Finishing the current step and saving the compact material model…" : "Finishing the current update and saving the material LoRA…"
        runner?.stopAndSave()
    }

    func saveCheckpointNow() {
        guard canStopAndSave, !isCheckpointPending else { return }
        isCheckpointPending = true
        activity = "Full validation and checkpoint queued after this update…"
        runner?.saveCheckpoint()
    }

    func recordTrainingProgress(_ chunk: String) {
        trainingEventBuffer += chunk
        while let newline = trainingEventBuffer.firstIndex(of: "\n") {
            let line = String(trainingEventBuffer[..<newline])
            trainingEventBuffer.removeSubrange(...newline)
            guard let data = line.data(using: .utf8),
                  let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let kind = event["event"] as? String else { continue }
            if isTraining {
                if trainingProgress == nil { trainingProgress = WorkbenchTrainingProgress() }
                trainingProgress?.consume(event)
            }
            if kind == "depthpro_model_setup" {
                if event["ready"] as? Bool == true, let path = event["model_path"] as? String {
                    depthProModelPath = path
                }
                if isBusy, !isStopping {
                    activity = [event["operation"] as? String, trainingProgress?.operationDetail].compactMap { $0 }.joined(separator: " · ")
                }
                continue
            }
            if ["source_verification_started", "source_verification_progress", "source_verification_completed"].contains(kind) {
                if isBusy, !isStopping, let completed = event["completed"] as? Int, let total = event["total"] as? Int,
                   completed >= 0, total >= completed {
                    let filename = (event["source_path"] as? String).map { URL(fileURLWithPath: $0).lastPathComponent }
                    activity = "Verifying originals for training: \(completed) / \(total) files"
                        + (filename.map { " · \($0)" } ?? "")
                    if let issue = event["issue"] as? String { logText += "\n\(filename ?? "Original"): \(issue)\n" }
                }
                continue
            }
            if ["source_recovery_started", "source_recovery_progress", "source_recovery_completed"].contains(kind) {
                if let completed = event["completed"] as? Int, let total = event["total"] as? Int {
                    sourceRecoverySummary = "Tier setup: \(completed) / \(total) maps checked"
                    if kind == "source_recovery_completed" {
                        sourceRecoverySummary += " · \(event["recovered"] as? Int ?? 0) fetched · \(event["not_published"] as? Int ?? 0) unavailable · \(event["failed"] as? Int ?? 0) failed"
                    }
                    if isBusy, !isStopping { activity = sourceRecoverySummary }
                }
                if isBusy, !isStopping, let message = event["message"] as? String { activity = message }
                continue
            }
            if ["preparation_started", "preparation_progress", "preparation_completed"].contains(kind) {
                guard isPreparingDataset, !isStopping,
                      let completed = event["completed"] as? Int, let total = event["total"] as? Int,
                      let size = event["training_size"] as? Int, let workers = event["worker_count"] as? Int,
                      total >= 0, completed >= 0, completed <= total, size > 0, workers > 0 else { continue }
                activity = "Preparing \(size) × \(size) maps: \(completed)/\(total) materials · \(workers) \(workers == 1 ? "worker" : "workers")"
                continue
            }
            guard isTraining else { continue }
            switch kind {
            case "depthpro_preparation", "crop_preparation":
                if !isStopping, let progress = trainingProgress {
                    activity = [progress.operationLabel, progress.operationDetail, progress.sampleID]
                        .compactMap { $0 }.joined(separator: " · ")
                }
            case "training_started":
                hasTrainingStarted = true
                if !isStopping { activity = training.modelFamily == .depthProHeightRefiner ? "Training DepthPro Height Refiner…" : training.modelFamily.isCompact ? "Training compact material model…" : "Training material LoRA…" }
            case "training_stopped":
                guard let reason = event["stopped_reason"] as? String,
                      ["time_limit", "no_valid_training_samples"].contains(reason), !isStopping else { continue }
                isStopping = true
                isSavingTraining = true
                activity = reason == "time_limit" ? "Training time limit reached. Validating and saving completed updates…" :
                    "No valid training samples remain. Saving current weights…"
            case "training_setup", "update_started", "operation_progress", "update", "validation_started", "validation_sample", "checkpoint_started", "export_started":
                if let progress = trainingProgress {
                    activity = [progress.currentUpdateSummary, progress.operationDetail.isEmpty ? nil : progress.operationDetail,
                        progress.operationLabel, progress.stageSummary].compactMap { $0 }.joined(separator: " · ")
                }
            case "sample_skipped":
                if !isStopping {
                    let sample = event["sample_id"] as? String ?? "sample"
                    let reason = event["error"] as? String
                    activity = "Skipped \(sample)" + (reason.map { ": \($0)" } ?? "") + " · Training continues…"
                }
            case "checkpoint_saved":
                if let checkpoint = try? WorkbenchResult.decode(WorkbenchCheckpoint.self, output: line) {
                    if !checkpoints.contains(where: { $0.id == checkpoint.id }) { checkpoints.append(checkpoint) }
                    saveCheckpointPaths()
                }
                isCheckpointPending = false
                if !isStopping { activity = "Checkpoint saved. Training continues…" }
            case "validation":
                let count = event["sample_count"] as? Int ?? 0
                let pool = event["pool_count"] as? Int ?? 0
                let skipped = max(0, event["validation_skipped_sample_count"] as? Int ?? 0)
                let label = event["scope"] as? String == "full" ? "Full validation" : "Quick check"
                let skippedSummary = skipped > 0 ? " · \(skipped) skipped" : ""
                if let error = event["mae"] as? Double {
                    validationSummary = "\(label): \(count)/\(pool) crops · error \(error.formatted(.number.precision(.fractionLength(6))))\(skippedSummary)"
                } else if event["status"] as? String == "unavailable" {
                    validationSummary = "\(label) unavailable: \(count)/\(pool) crops\(skippedSummary). Training and saving continue."
                } else { validationSummary = "" }
            default: break
            }
        }
        if trainingEventBuffer.count > 100000 { trainingEventBuffer = String(trainingEventBuffer.suffix(100000)) }
    }

    func useSelectedInStudio() {
        guard let checkpoint = selectedCheckpoint, checkpoint.supportsStudioInference else { return }
        do {
            try SelectedMaterialCheckpoint(checkpointPath: checkpoint.checkpointPath, sha256: checkpoint.sha256,
                target: checkpoint.target, workspacePath: workspacePath,
                modelDirectory: baseDirectory(for: checkpoint),
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
                let args = self.checkpointPackageArguments(for: checkpoint, output: destination, developer: self.developerMode)
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
            let result = try WorkbenchResult.decode(HuggingFaceAccountResponse.self,
                output: await self.worker(["hub-account"]))
            self.uploadAccount = result.authenticated ? result.username : nil
            self.uploadAccountChecked = true
            self.uploadAccountMessage = result.message
            self.activity = result.message
            if result.authenticated {
                let models = try WorkbenchResult.decode(WorkbenchHubModels.self, output: await self.worker(["hub-models"]))
                self.hubModels = models.models
            }
        }
    }
    func uploadPackage() {
        guard let checkpoint = selectedCheckpoint else { error = "Select a model checkpoint to upload."; return }
        let repository = effectiveUploadRepo
        guard uploadAccount != nil else {
            uploadAccountMessage = "Sign in with hf auth login or save a token in settings, then click Refresh Account."
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
        args += dependencyArguments(for: checkpoint)
        let result = try WorkbenchResult.decode(HuggingFaceUploadResponse.self, output: await worker(args))
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
        if let warning = result.catalogWarning { activity += " " + warning }
        do {
            let models = try WorkbenchResult.decode(WorkbenchHubModels.self, output: await worker(["hub-models"]))
            hubModels = models.models
        } catch {
            // The upload is already committed; discovery can be refreshed later.
            activity += " Could not refresh the Hub model list: \(error.localizedDescription)"
        }
    }
    var managedEncoderURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Material Models/pbrnxt-base")
    }
    func installEncoder() {
        operation("Downloading the material base…") {
            _ = try await self.worker(["install-base", "--destination", self.managedEncoderURL.path])
            self.modelDirectory = self.managedEncoderURL.path
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
            let result = try WorkbenchResult.decode(WorkbenchHubModels.self, output: await self.worker(["hub-models"]))
            self.hubModels = result.models
        }
    }
    func downloadHubModel(_ model: WorkbenchHubModel) {
        operation("Downloading \(model.repository)…") {
            let destination = self.managedEncoderURL.deletingLastPathComponent().appendingPathComponent(model.repository.replacingOccurrences(of: "/", with: "--"))
            var args = ["download-model", "--repo", model.repository, "--destination", destination.path]
            if let revision = model.revision { args += ["--revision", revision] }
            let result = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: await self.worker(args))
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
        deferredCheckpointPaths.removeAll { $0 == checkpoint.checkpointPath }
        saveCheckpointPaths()
    }

    func worker(_ args: [String]) async throws -> String {
        let includesTarget = ["create-dataset", "edit-dataset", "add-material", "import-folder", "remove-material"].contains(args.first ?? "") && !args.contains("--target")
        let includesPreparationWorkers = args.first == "prepare-size" && !args.contains("--preparation-workers")
        let args = args + (includesTarget ? ["--target", training.target] : [])
            + (includesPreparationWorkers ? ["--preparation-workers", String(preparationWorkers)] : [])
        try Task.checkCancellation()
        guard !WorkbenchLifecycle.shared.isTerminating else { throw CancellationError() }
        let trainingWorker = ["train", "refine"].contains(args.first ?? "")
        let classifierWorker = args.first == "refresh-classifier-dataset"
        let specialistWorker = args.first == "refresh-specialists"
        let progressWorker = trainingWorker || ["dataset", "prepare-size", "scan-folder", "import-folder", "edit-dataset"].contains(args.first ?? "")
        if progressWorker { trainingEventBuffer = "" }
        if trainingWorker { hasTrainingStarted = false }
        defer { if trainingWorker { hasTrainingStarted = false } }
        if let workerOverride {
            let output = try await workerOverride(args, args.first ?? "")
            if progressWorker { recordTrainingProgress(output) }
            if !trainingWorker || !Self.hasSavedTrainingResult(output) { try Task.checkCancellation() }
            return output
        }
        saveConfiguration(refreshSelectedRuntime: false)
        let logs = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Worklogs")
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let log = logs.appendingPathComponent("\(UUID().uuidString).log")
        lastLogURL = log
        let events = NativeWorkbenchLog(url: log)
        let control = NativeMaterialTrainingControl(); runner = control
        let workerId = UUID(); activeWorkerId = workerId
        let event: @Sendable (String) -> Void = { [weak self] chunk in
            guard events.append(chunk) else { return }
            Task { @MainActor in
                // Deliver progress in order at most ten times per second. The
                // trainer never waits for UI updates, and its disk log is complete.
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, self.activeWorkerId == workerId else { return }
                let pending = events.drainPending()
                self.logText = events.text
                if progressWorker, !pending.isEmpty { self.recordTrainingProgress(pending) }
                if classifierWorker, !pending.isEmpty { self.recordClassifierDatasetProgress(pending) }
                if specialistWorker, !pending.isEmpty { self.recordSpecialistProgress(pending) }
            }
        }
        let operation = Task { try await NativeMaterialCommands.run(arguments: args, onEvent: event, control: control) }
        WorkbenchLifecycle.shared.add(control, cancel: { operation.cancel() })
        defer {
            // Flush any final events once before retiring the worker. A queued
            // UI delivery then fails its worker identity check and cannot replay them.
            let pending = events.drainPending()
            if progressWorker, !pending.isEmpty { recordTrainingProgress(pending) }
            if classifierWorker, !pending.isEmpty { recordClassifierDatasetProgress(pending) }
            if specialistWorker, !pending.isEmpty { recordSpecialistProgress(pending) }
            runner = nil; activeWorkerId = nil
            logText = events.text
            WorkbenchLifecycle.shared.remove(control)
        }
        let output: String
        do {
            output = try await withTaskCancellationHandler { try await operation.value } onCancel: { operation.cancel(); control.stop() }
        } catch {
            if let bytes = try? JSONSerialization.data(withJSONObject: ["event": "operation_failed", "operation": args.first ?? "", "error": error.localizedDescription], options: [.sortedKeys]) {
                events.append(String(decoding: bytes, as: UTF8.self) + "\n")
            }
            throw error
        }
        events.append(output + "\n")
        if !trainingWorker || !Self.hasSavedTrainingResult(output) { try Task.checkCancellation() }
        return output
    }

    private static func hasSavedTrainingResult(_ output: String) -> Bool {
        guard let result = try? WorkbenchResult.decode(WorkbenchTrainingResponse.self, output: output) else { return false }
        return !result.checkpointPath.isEmpty && ["completed", "stopped"].contains(result.status ?? "")
    }
    func operation(_ label: String, training: Bool = false, body: @escaping @MainActor () async throws -> Void) {
        guard !isBusy else { return }
        isCheckpointPending = false; validationSummary = ""
        isBusy = true; isTraining = training; isStopping = false; hasTrainingStarted = false; isSavingTraining = false
        error = nil; activity = label; logText = ""; trainingEventBuffer = ""
        trainingProgress = training ? WorkbenchTrainingProgress() : nil
        task = Task {
            defer {
                isBusy = false; isTraining = false; isResumingTraining = false; isStopping = false
                hasTrainingStarted = false; isSavingTraining = false; isCheckpointPending = false; isPreparingDataset = false; task = nil
                if let url = queuedDatasetURL {
                    queuedDatasetURL = nil
                    openDataset(url)
                } else if let handoff = queuedTrainingHandoff {
                    queuedTrainingHandoff = nil
                    receiveTrainingHandoff(handoff)
                } else if let url = queuedResumeURL {
                    queuedResumeURL = nil
                    receiveResumeCheckpoint(url)
                }
                refreshSourceFolderIfNeeded()
            }
            do {
                try await body()
                if activity == label { activity = "Ready." }
            }
            catch {
                if Task.isCancelled || error is CancellationError {
                    if training { trainingProgress?.finish(state: .aborted) }
                    activity = training ? "Training aborted. Previously saved checkpoints are kept." : "Operation stopped. See the log and output folder."
                } else {
                    if training { trainingProgress?.finish(state: .failed) }
                    self.error = error.localizedDescription; activity = "Operation stopped. See the error and log."
                }
            }
        }
    }
    /// Installed and development app builds share a user-owned output default.
    /// Older builds saved the bundled source checkout as their working folder.
    private static func userWorkspaceURL(configured: String?, managed: URL) -> URL {
        guard let configured, !configured.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return managed }
        let candidate = URL(fileURLWithPath: configured).standardizedFileURL
        let resolved = candidate.resolvingSymlinksInPath()
        if let bundled = Bundle.main.object(forInfoDictionaryKey: "IPDEWorkspace") as? String, bundled.hasPrefix("/") {
            let source = URL(fileURLWithPath: bundled).standardizedFileURL.resolvingSymlinksInPath()
            if resolved == source || resolved.path.hasPrefix(source.path + "/") { return managed }
        }
        var ancestor = resolved
        while true {
            if FileManager.default.fileExists(atPath: ancestor.appendingPathComponent(".git").path) { return managed }
            let parent = ancestor.deletingLastPathComponent()
            if parent == ancestor { break }
            ancestor = parent
        }
        return candidate
    }

    private func newOutputURL(prefix: String) throws -> URL {
        let outputWorkspace = Self.userWorkspaceURL(configured: workspacePath, managed: managedWorkspaceURL)
        if outputWorkspace != workspaceURL { workspacePath = outputWorkspace.path }
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
