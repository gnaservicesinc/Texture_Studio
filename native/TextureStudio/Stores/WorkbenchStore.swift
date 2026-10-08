import AppKit
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class WorkbenchStore {
    var dataset: WorkbenchDataset?
    var selectedSampleId: String?
    var selectedRole = "height"
    var checkpoints: [WorkbenchCheckpoint] = []
    var selectedCheckpointId: String?
    var comparisonCheckpointIds: Set<String> = []
    var training = MaterialTrainingOptions()
    var activity = ""
    var logText = ""
    var error: String?
    private(set) var isBusy = false
    private(set) var isTraining = false
    private(set) var isResumingTraining = false
    private(set) var isStopping = false
    private(set) var isPreparingDataset = false
    private(set) var datasetPreparationSummary = ""
    var lastOutputURL: URL?
    var lastLogURL: URL?
    var lastPackageURL: URL?
    var lastPackageCheckpointId: String?
    var sourceImageURL: URL?
    var comparisonCandidates: [MapReviewCandidate] = []
    var uploadRepo = ""
    var uploadPublic = false
    var workspacePath: String
    var pythonPath: String
    var modelDirectory: String
    var codeDirectory: String
    @ObservationIgnored private var runner: WorkbenchProcess?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var activeWorkerId: UUID?
    @ObservationIgnored private let preferences: UserDefaults
    @ObservationIgnored private let workerOverride: (@MainActor ([String], String) async throws -> String)?
    @ObservationIgnored private let managedWorkspaceURL: URL
    @ObservationIgnored private var hasRestored = false

    var samples: [WorkbenchSample] { dataset?.samples ?? [] }
    var selectedSample: WorkbenchSample? { samples.first { $0.id == selectedSampleId } }
    var selectedMaterialId: String? { dataset?.materials.first { $0.samples.contains { $0.id == selectedSampleId } }?.materialId }
    var selectedCheckpoint: WorkbenchCheckpoint? { checkpoints.first { $0.id == selectedCheckpointId } }
    var selectedMap: WorkbenchMap? { selectedSample?.maps[selectedRole] }
    var datasetURL: URL? { dataset.map { URL(fileURLWithPath: $0.datasetPath) } }
    var workspaceURL: URL { URL(fileURLWithPath: workspacePath).standardizedFileURL }
    var backendDirectory: URL { Bundle.main.resourceURL!.appendingPathComponent("MaterialBackend") }
    var dependencyArguments: [String] { ["--model-directory", modelDirectory, "--code-directory", codeDirectory] }

    init(preferences defaults: UserDefaults = UserDefaults(suiteName: "org.ipde.material-tools")!,
         managedWorkspaceURL: URL? = nil,
         workerOverride: (@MainActor ([String], String) async throws -> String)? = nil) {
        preferences = defaults
        self.workerOverride = workerOverride
        self.managedWorkspaceURL = (managedWorkspaceURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Material Workspace")).standardizedFileURL
        let local = Bundle.main.object(forInfoDictionaryKey: "IPDEWorkspace") as? String ?? ""
        let configuredWorkspace = defaults.string(forKey: "workspace") ?? local
        let workspace = configuredWorkspace.isEmpty ? self.managedWorkspaceURL.path : URL(fileURLWithPath: configuredWorkspace).standardizedFileURL.path
        workspacePath = workspace
        pythonPath = defaults.string(forKey: "python") ?? MaterialWorkbenchRuntime.defaultPython(workspace: URL(fileURLWithPath: workspace))
        let cache = URL(fileURLWithPath: workspace).appendingPathComponent("out/material-training/transfer-models")
        modelDirectory = defaults.string(forKey: "encoder") ?? cache.appendingPathComponent("dinov2-base-f9e44c814b77").path
        codeDirectory = defaults.string(forKey: "encoderCode") ?? cache.appendingPathComponent("dinov2-code-7764ea0f912e").path
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
        operation("Opening workspace…") {
            if let datasetPath, FileManager.default.fileExists(atPath: datasetPath) {
                try await self.loadDataset(URL(fileURLWithPath: datasetPath))
            }
            for path in paths where FileManager.default.fileExists(atPath: path) {
                do { try await self.loadCheckpoint(URL(fileURLWithPath: path)) }
                catch { self.logText += "Could not reconnect \(path): \(error.localizedDescription)\n" }
            }
        }
    }

    func saveConfiguration(refreshSelectedRuntime: Bool = true) {
        preferences.set(workspacePath, forKey: "workspace")
        preferences.set(pythonPath, forKey: "python")
        preferences.set(modelDirectory, forKey: "encoder")
        preferences.set(codeDirectory, forKey: "encoderCode")
        if refreshSelectedRuntime {
            do {
                try SelectedMaterialCheckpoint.refreshRuntime(pythonPath: pythonPath, workspacePath: workspacePath,
                    modelDirectory: modelDirectory, codeDirectory: codeDirectory)
            } catch {
                self.error = "Runtime settings were saved, but Texture Studio could not reconnect its selected checkpoint: \(error.localizedDescription)"
            }
        }
    }

    func chooseDataset() {
        chooseFile(types: [.json], title: "Open material dataset.json") { url in
            self.openDataset(url)
        }
    }
    func openDataset(_ url: URL) {
        guard !isBusy else { error = "Stop the current operation before changing datasets."; return }
        operation("Reading dataset…") { try await self.loadDataset(url) }
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
        chooseFile(types: [.image], title: "Choose the same source photo for all checkpoints") { self.sourceImageURL = $0 }
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
        chooseFolder(title: "Locate the pinned DINOv2 encoder folder") { url in self.modelDirectory = url.path; self.saveConfiguration() }
    }
    func chooseEncoderCode() {
        chooseFolder(title: "Locate the pinned DINOv2 source folder") { url in self.codeDirectory = url.path; self.saveConfiguration() }
    }

    func loadDataset(_ url: URL) async throws {
        let result: WorkbenchDataset = try WorkbenchProcess.decode(WorkbenchDataset.self, output: await worker(["dataset", "--dataset", url.path]))
        adoptDataset(result)
        datasetPreparationSummary = ""
        preferences.set(result.datasetPath, forKey: "dataset")
    }

    private func adoptDataset(_ result: WorkbenchDataset, preferredMaterial: String? = nil) {
        let material = preferredMaterial ?? selectedMaterialId
        dataset = result
        if !result.samples.contains(where: { $0.id == selectedSampleId }) {
            selectedSampleId = result.materials.first(where: { $0.id == material })?.samples.first?.id ?? result.samples.first?.id
        }
    }

    func selectTrainingSize(_ size: Int) {
        guard !isBusy else { return }
        guard [1024, 2048].contains(size) else { error = "Choose 1024 or 2048 native pixels."; return }
        training.size = size
        prepareTrainingDataset()
    }

    func prepareTrainingDataset() {
        guard !isBusy, dataset != nil else { return }
        let size = training.size
        guard dataset?.hasNativeSize(size) != true else { return }
        operation("Preparing \(size) × \(size) native crops…") {
            _ = try await self.ensureTrainingDataset(size: size)
            self.activity = self.datasetPreparationSummary
        }
    }

    private func ensureTrainingDataset(size: Int) async throws -> WorkbenchDataset {
        guard let current = dataset else { throw StudioError("Open a material dataset first.") }
        guard [1024, 2048].contains(size) else { throw StudioError("Choose 1024 or 2048 native pixels.") }
        if current.hasNativeSize(size) { return current }
        let material = selectedMaterialId
        isPreparingDataset = true
        activity = "Preparing \(size) × \(size) native crops from the original materials…"
        defer { isPreparingDataset = false }
        let result: WorkbenchDataset = try WorkbenchProcess.decode(WorkbenchDataset.self, output: await worker([
            "prepare-size", "--dataset", current.datasetPath, "--size", String(size),
            "--expected-index-sha256", current.indexSha256]))
        guard result.hasNativeSize(size), let preparation = result.preparation,
              preparation.cropSize == size, !preparation.targetResized, !preparation.originalDatasetModified,
              URL(fileURLWithPath: preparation.preparedDatasetPath).standardizedFileURL == URL(fileURLWithPath: result.datasetPath).standardizedFileURL else {
            throw StudioError("Crop preparation did not verify the requested native size and preserved originals. The previous dataset remains selected.")
        }
        try Task.checkCancellation()
        adoptDataset(result, preferredMaterial: material)
        preferences.set(result.datasetPath, forKey: "dataset")
        datasetPreparationSummary = "\(preparation.reused ? "Opened existing" : "Prepared") \(size) × \(size) native crops. Original maps remain unchanged."
        return result
    }
    func loadCheckpoint(_ url: URL) async throws {
        let checkpoint: WorkbenchCheckpoint = try WorkbenchProcess.decode(WorkbenchCheckpoint.self, output: await worker(["checkpoint", "--checkpoint", url.path]))
        guard checkpoint.compatible else { throw StudioError("This checkpoint is not supported by the material backend.") }
        if let i = checkpoints.firstIndex(where: { $0.id == checkpoint.id }) { checkpoints[i] = checkpoint }
        else { checkpoints.append(checkpoint) }
        selectedCheckpointId = checkpoint.id
        comparisonCheckpointIds.insert(checkpoint.id)
        preferences.set(checkpoints.map(\.checkpointPath), forKey: "checkpoints")
    }

    func curateSelected(status: String, split: String? = nil, note: String? = nil) {
        guard let dataset, let sample = selectedSample else { return }
        operation("Saving crop review…") {
            var args = ["curate", "--dataset", dataset.datasetPath, "--sample", sample.id,
                        "--status", status, "--expected-index-sha256", dataset.indexSha256]
            if let split { args += ["--split", split] }
            if let note { args += ["--note", note] }
            _ = try await self.worker(args)
            try await self.loadDataset(URL(fileURLWithPath: dataset.datasetPath))
        }
    }
    func inspectSelectedMap() {
        guard let map = selectedMap else { return }
        ReviewWindowController.shared.open(candidates: [MapReviewCandidate(id: map.path, label: "\(selectedSampleId ?? "Crop") · \(selectedRole)", mapURL: map.url, numeric: selectedRole != "input")])
    }

    func compare() {
        guard let image = sourceImageURL ?? selectedSample?.maps["input"]?.url else {
            error = "Choose a source photo or select a dataset crop first."; return
        }
        let selected = checkpoints.filter { comparisonCheckpointIds.contains($0.id) }
        guard selected.count >= 2, Set(selected.map(\.target)).count == 1 else {
            error = "Choose at least two checkpoints predicting the same map type."; return
        }
        operation("Comparing \(selected.count) checkpoints at the source resolution…") {
            let parent = try self.newOutputURL(prefix: "comparison")
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
            self.lastOutputURL = parent
            self.comparisonCandidates = []
            var candidates: [MapReviewCandidate] = []
            for (i, checkpoint) in selected.enumerated() {
                self.activity = "Running checkpoint \(i + 1)/\(selected.count): \(checkpoint.title)"
                let child = parent.appendingPathComponent("candidate-\(i + 1)")
                let result: MaterialInferenceResponse = try WorkbenchProcess.decode(MaterialInferenceResponse.self, output: await self.worker([
                    "infer", "--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256,
                    "--image", image.path, "--output", child.path, "--device", "mps"] + self.dependencyArguments))
                guard result.checkpointSha256 == checkpoint.sha256, let map = result.outputs[checkpoint.target] else {
                    throw StudioError("The comparison did not use the selected checkpoint or map type.")
                }
                candidates.append(MapReviewCandidate(id: checkpoint.id, label: "\(checkpoint.title) · step \(checkpoint.step)", mapURL: URL(fileURLWithPath: map.path), numeric: true))
            }
            self.comparisonCandidates = candidates
            self.lastOutputURL = parent
            let manifest: [String: Any] = ["schema": "texture-studio-material-quality-review-v1", "comparison_target": selected[0].target,
                "materials": [["material_id": image.lastPathComponent, "diffuse": image.path,
                    "variants": zip(selected, candidates).map { checkpoint, candidate in
                        ["name": candidate.label, selected[0].target: candidate.mapURL.path,
                         "checkpoint": checkpoint.checkpointPath, "checkpoint_sha256": checkpoint.sha256] as [String: Any]
                    }]]]
            try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
                .write(to: parent.appendingPathComponent("review-manifest.json"), options: .atomic)
        }
    }

    func startTraining() {
        guard dataset != nil else { error = "Open a dataset first."; return }
        let options = training, material = selectedMaterialId, checkpoint = selectedCheckpoint
        if let issue = configurationIssue(options: options, checkpoint: checkpoint) { error = issue; return }
        operation("Preparing the training run…") {
            let prepared = try await self.ensureTrainingDataset(size: options.size)
            if let issue = self.sampleIssue(dataset: prepared, options: options, material: material) { throw StudioError(issue) }
            var args = ["train", "--dataset", prepared.datasetPath, "--target", options.target,
                        "--expected-size", String(options.size), "--updates-per-crop", String(options.updatesPerCrop),
                        "--max-minutes", String(options.maxMinutes), "--max-driver-bytes", String(Int64(options.memoryGB * 1_000_000_000)),
                        "--selection", "final", "--checkpoint-every", "100", "--evaluate-every", "400", "--prediction-limit", "2", "--device", "mps"]
            if options.allowUnreviewed { args += ["--allow-unreviewed"] }
            if options.maskTransparency { args += ["--mask-transparent-input"] }
            if options.useSelectedMaterialOnly, let material { args += ["--material", material] }
            if options.useWarmStart, let checkpoint {
                args += ["--warm-start", checkpoint.checkpointPath, "--warm-start-sha256", checkpoint.sha256]
            }
            try await self.runTraining(args, options: options)
        }
    }

    var trainingConfigurationIssue: String? {
        if let issue = configurationIssue(options: training, checkpoint: selectedCheckpoint) { return issue }
        guard let dataset else { return "Open a prepared dataset to begin." }
        // A different requested size is prepared from parents before training;
        // it is an action the app handles, rather than a manual setup error.
        return sampleIssue(dataset: dataset, options: training, material: selectedMaterialId)
    }

    private func configurationIssue(options: MaterialTrainingOptions, checkpoint: WorkbenchCheckpoint?) -> String? {
        guard ["height", "roughness", "normal"].contains(options.target), [1024, 2048].contains(options.size) else { return "Choose a supported map target and native size." }
        if let issue = resourceIssue(options: options) { return issue }
        if options.useWarmStart {
            guard let checkpoint else { return "Locate a checkpoint to refine, or choose Start from Base DINOv2." }
            if !checkpoint.supportsTrainingWarmStart { return "This checkpoint cannot be refined by the selected backend. Choose a supported material checkpoint, or start from Base DINOv2." }
        }
        return nil
    }

    private func resourceIssue(options: MaterialTrainingOptions) -> String? {
        guard options.memoryGB.isFinite, (2...30).contains(options.memoryGB), options.maxMinutes.isFinite,
              (1...240).contains(options.maxMinutes), (1...10000).contains(options.updatesPerCrop) else { return "Choose 2–30 GB of memory, 1–240 minutes and 1–10,000 updates per crop." }
        return nil
    }

    private func sampleIssue(dataset: WorkbenchDataset, options: MaterialTrainingOptions, material: String?) -> String? {
        if options.useSelectedMaterialOnly && material == nil { return "Select the material to train." }
        let candidates = options.useSelectedMaterialOnly ? dataset.materials.first(where: { $0.id == material })?.samples ?? [] : dataset.samples
        if !options.allowUnreviewed && candidates.contains(where: { ["prepared", "unreviewed"].contains($0.status) }) {
            return "Approve or exclude the crops awaiting review, or allow crops awaiting approval."
        }
        let eligible = candidates.filter { ["approved", "accepted"].contains($0.status) || (options.allowUnreviewed && ["prepared", "unreviewed"].contains($0.status)) }
        if eligible.isEmpty { return "Approve crops or include prepared crops awaiting review." }
        if eligible.contains(where: { $0.maps[options.target] == nil }) { return "Each selected crop needs its own \(options.target) target map." }
        if !eligible.contains(where: { $0.split == "train" }) || !eligible.contains(where: { $0.split == "validation" }) { return "The selection needs training and validation crops." }
        return nil
    }
    func chooseResumeCheckpoint() {
        if let issue = resourceIssue(options: training) { error = issue; return }
        chooseFile(title: "Choose checkpoint.latest.pt with optimizer state", selected: resumeTraining)
    }
    func resumeTraining(from url: URL) {
        let options = training
        if let issue = resourceIssue(options: options) { error = issue; return }
        launchTraining(["resume", "--resume-checkpoint", url.path, "--updates-per-crop", String(options.updatesPerCrop),
                        "--max-minutes", String(options.maxMinutes), "--max-driver-bytes", String(Int64(options.memoryGB * 1_000_000_000))])
    }
    private func launchTraining(_ args: [String]) {
        let options = training
        operation("Training native material maps…", training: true) {
            try await self.runTraining(args, options: options)
        }
    }
    private func runTraining(_ args: [String], options: MaterialTrainingOptions) async throws {
        try Task.checkCancellation()
        isTraining = true
        isResumingTraining = args.first == "resume"
        activity = isResumingTraining ? "Resuming the saved run’s crop selection and map target…" : "Training native material maps…"
        let output = try self.newOutputURL(prefix: isResumingTraining ? "resumed-material-run" : "native-\(options.size)-\(options.target)")
        self.lastOutputURL = output
        let text = try await self.worker(args + ["--output", output.path] + self.dependencyArguments, script: "material_training_cycle.py")
        let trainingLogURL = self.lastLogURL
        defer { self.lastLogURL = trainingLogURL; self.logText = String(text.suffix(100000)) }
        let checkpoint = output.appendingPathComponent("checkpoint.selected.pt")
        if FileManager.default.fileExists(atPath: checkpoint.path) { try await self.loadCheckpoint(checkpoint) }
        self.activity = "Training stopped or finished. Checkpoints and summary are in the run folder."
    }
    func stop() {
        guard isBusy else { return }
        isStopping = true
        activity = isTraining ? "Stopping after saving the latest training checkpoint…" : "Stopping the operation…"
        runner?.stop()
        task?.cancel()
    }

    func useSelectedInStudio() {
        guard let checkpoint = selectedCheckpoint, checkpoint.target == "height" else { error = "Select a height checkpoint for Texture Studio."; return }
        do {
            let record = SelectedMaterialCheckpoint(checkpointPath: checkpoint.checkpointPath, sha256: checkpoint.sha256, target: checkpoint.target,
                pythonPath: pythonPath, workspacePath: workspacePath, modelDirectory: modelDirectory, codeDirectory: codeDirectory)
            try record.save()
            activity = "Selected for Texture Studio: \(checkpoint.title). Choose Material checkpoint in its depth controls."
        } catch { self.error = error.localizedDescription }
    }
    func exportSelectedCheckpoint() {
        guard let checkpoint = selectedCheckpoint else { return }
        chooseFolder(title: "Choose a folder for a new model package") { parent in
            self.operation("Exporting selected model package…") {
                let destination = parent.appendingPathComponent("material-\(checkpoint.target)-\(UUID().uuidString.prefix(8))")
                _ = try await self.worker(["package", "--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256, "--output", destination.path])
                self.lastPackageURL = destination
                self.lastPackageCheckpointId = checkpoint.id
                NSWorkspace.shared.activateFileViewerSelecting([destination])
            }
        }
    }
    func uploadPackage() {
        guard let package = lastPackageURL else { error = "Export a model package before uploading it."; return }
        guard uploadRepo.split(separator: "/").count == 2 else { error = "Enter a Hugging Face repository as owner/model-name."; return }
        operation("Uploading the exported package to Hugging Face…") {
            var args = ["upload", "--package", package.path, "--repo", self.uploadRepo]
            if self.uploadPublic { args += ["--public"] }
            _ = try await self.worker(args)
            self.activity = "Upload completed: \(self.uploadRepo)"
        }
    }
    var managedEncoderURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/Material Models/dinov2-pinned")
    }
    func installEncoder() {
        operation("Downloading and checking the pinned encoder…") {
            _ = try await self.worker(["install-encoder", "--destination", self.managedEncoderURL.path])
            self.modelDirectory = self.managedEncoderURL.appendingPathComponent("model").path
            self.codeDirectory = self.managedEncoderURL.appendingPathComponent("code").path
            self.saveConfiguration()
            self.activity = "Pinned encoder installed. It can be removed from Runtime settings."
        }
    }
    func removeDownloadedEncoder() {
        operation("Removing the app-managed encoder…") {
            _ = try await self.worker(["remove-encoder", "--directory", self.managedEncoderURL.path])
            self.activity = "Downloaded encoder removed. Locate a copy or download it again before inference."
        }
    }
    func forgetSelectedCheckpoint() {
        guard let checkpoint = selectedCheckpoint else { return }
        checkpoints.removeAll { $0.id == checkpoint.id }
        comparisonCheckpointIds.remove(checkpoint.id)
        selectedCheckpointId = checkpoints.first?.id
        preferences.set(checkpoints.map(\.checkpointPath), forKey: "checkpoints")
    }

    func worker(_ args: [String], script: String = "material_workbench.py") async throws -> String {
        try Task.checkCancellation()
        guard !WorkbenchLifecycle.shared.isTerminating else { throw CancellationError() }
        if let workerOverride {
            let output = try await workerOverride(args, script)
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
                }
            }
        try Task.checkCancellation()
        return output
    }
    private func operation(_ label: String, training: Bool = false, body: @escaping @MainActor () async throws -> Void) {
        guard !isBusy else { return }
        isBusy = true; isTraining = training; isStopping = false; error = nil; activity = label; logText = ""
        task = Task {
            defer { isBusy = false; isTraining = false; isResumingTraining = false; isStopping = false; isPreparingDataset = false; task = nil }
            do { try await body() }
            catch {
                if Task.isCancelled || error is CancellationError {
                    let saved = lastOutputURL.map { FileManager.default.fileExists(atPath: $0.appendingPathComponent("checkpoint.latest.pt").path) } ?? false
                    activity = isTraining && saved ? "Stopped. Latest checkpoint and optimizer state were saved in the run folder." : "Operation stopped. See the log and output folder."
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
