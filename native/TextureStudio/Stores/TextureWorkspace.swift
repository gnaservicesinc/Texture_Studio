import AppKit
import CoreImage
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class TextureWorkspace {
    var source: TextureSource?
    var attachedDepth: TextureDepth?
    var generatedDepth: TextureDepth?
    var generatedProvenance: ModelDepthProvenance?
    var depthURL: URL?
    var settings = TextureSettings() { didSet { if settings != oldValue { savePreferences(); markEdited() } } }
    var depthChoice = DepthChoice.photoDetail {
        didSet {
            if depthChoice != oldValue { savePreferences(); markEdited() }
        }
    }
    var heightSourceNotice: String?
    private(set) var selectedMaterialCheckpoint: SelectedMaterialCheckpoint?
    private(set) var selectedMaterialMaps: [String: SelectedMaterialCheckpoint] = [:]
    var modelID = LocalModelDescriptor.customDepthID { didSet { if modelID != oldValue { savePreferences(); invalidateModelDepth() } } }
    var customInverseDepth = true { didSet { if customInverseDepth != oldValue { savePreferences(); invalidateModelDepth() } } }
    var selectedPreview = MaterialPreview.diffuse { didSet { savePreferences() } }
    var renderedPreview = MaterialPreview.source
    var preview: CGImage?
    var result: MaterialResult?
    var isBusy = false
    var activity = ""
    var notice: StudioNotice?
    var showModels = false
    var showAdvice = false
    var decision: MaterialDecision?
    var showModelRecovery = false
    var showInspector = true { didSet { savePreferences() } }
    var warnings: [String] = []
    var exportURL: URL?
    var recipeURL: URL?
    var hasEdits = false
    private let engine = TextureEngine()
    private let modelService = ModelDepthService()
    private let materialCheckpointService = MaterialCheckpointService()
    private var operation: Task<Void, Never>?
    private var generatedModelID: String?
    private var generatedResolution: Int?
    private var generatedModelPath: String?
    private var generatedFileSignature: String?
    private var sourcePreview: CGImage?
    private var recipeCheckpoint: MaterialCheckpointIdentity?
    private var recipeMapCheckpoints: [String: MaterialCheckpointIdentity]?
    private var lastRegistrySelectionIdentity: String?
    private var generatedCheckpointKey: String?
    private var materialCache: MaterialRenderCache?
    @ObservationIgnored private weak var renderModels: ModelManager?
    @ObservationIgnored private let preferences: UserDefaults?
    @ObservationIgnored private let checkpointPredictor: (@MainActor (TextureSource, SelectedMaterialCheckpoint, Int) async throws -> TextureDepth)?
    @ObservationIgnored private let materialProcessor: (@MainActor (TextureSource, TextureSettings, TextureDepth?) async throws -> MaterialResult)?
    @ObservationIgnored private let mapPredictor: (@MainActor (CIImage, SelectedMaterialCheckpoint, Int) async throws -> MaterialModelMap)?
    private var exportDirectory: String?
    private let checkpointRegistryURL: URL
    @ObservationIgnored private var selectionObserver: MaterialSelectionObserver?
    @ObservationIgnored private var pendingCheckpointActivation = false

    init(checkpointRegistryURL: URL = SelectedMaterialCheckpoint.registryURL,
         preferences: UserDefaults? = nil,
         checkpointPredictor: (@MainActor (TextureSource, SelectedMaterialCheckpoint, Int) async throws -> TextureDepth)? = nil,
         materialProcessor: (@MainActor (TextureSource, TextureSettings, TextureDepth?) async throws -> MaterialResult)? = nil,
         mapPredictor: (@MainActor (CIImage, SelectedMaterialCheckpoint, Int) async throws -> MaterialModelMap)? = nil) {
        self.checkpointRegistryURL = checkpointRegistryURL
        self.preferences = preferences
        self.checkpointPredictor = checkpointPredictor
        self.materialProcessor = materialProcessor
        self.mapPredictor = mapPredictor
        selectedMaterialCheckpoint = try? SelectedMaterialCheckpoint.read(from: checkpointRegistryURL)
        selectedMaterialMaps = SelectedMaterialCheckpoint.readAll(heightRegistryURL: checkpointRegistryURL)
        lastRegistrySelectionIdentity = selectedMaterialCheckpoint?.selectionIdentity
        if selectedMaterialCheckpoint?.supportsStudioInference == true, selectedMaterialCheckpoint?.target == "height" {
            depthChoice = .materialCheckpoint
        }
        if let preferences {
            let saved = StudioPreferences.load(from: preferences)
            settings = saved.settings ?? settings
            depthChoice = saved.depthChoice ?? depthChoice
            modelID = saved.modelID ?? modelID
            customInverseDepth = saved.customInverseDepth ?? customInverseDepth
            selectedPreview = saved.selectedPreview.flatMap(MaterialPreview.init(rawValue:)) ?? selectedPreview
            showInspector = saved.showInspector ?? showInspector
            exportDirectory = saved.exportDirectory
        }
        selectionObserver = MaterialSelectionObserver { [weak self] in
            self?.reloadSelectedCheckpoint(activate: true, target: self?.selectionObserver?.changedTarget)
        }
        if preferences != nil { savePreferences() }
    }

    private func savePreferences() {
        guard let preferences else { return }
        StudioPreferences(settings: settings, depthChoice: depthChoice, modelID: modelID,
            customInverseDepth: customInverseDepth, selectedPreview: selectedPreview.rawValue,
            showInspector: showInspector, exportDirectory: exportDirectory).save(to: preferences)
    }

    private func currentRenderKey() -> MaterialRenderKey? {
        guard let source else { return nil }
        let depthIdentity: String
        switch depthChoice {
        case .materialCheckpoint:
            guard let checkpoint = try? currentMaterialCheckpoint() else { return nil }
            depthIdentity = checkpointCacheKey(checkpoint)
        case .model:
            let record = renderModels?.records[modelID]
            let path = record?.path ?? generatedModelPath ?? ""
            let signature = path.isEmpty ? "" : modelFileSignature(URL(fileURLWithPath: path), record: record)
            depthIdentity = "\(modelID)|\(settings.modelProcessResolution)|\(customInverseDepth)|\(path)|\(signature)"
        case .attached:
            guard let attachedDepth else { return nil }
            depthIdentity = "attached|\(ObjectIdentifier(attachedDepth.image))|\(settings.attachedMapIsHeight)"
        case .photoDetail: depthIdentity = "flat"
        }
        guard let auxiliary = try? activeAuxiliaryCheckpoints() else { return nil }
        let auxiliaryKey = auxiliary.keys.sorted().compactMap { auxiliary[$0].map(checkpointCacheKey) }.joined(separator: "|")
        return MaterialRenderKey(sourceIdentity: "\(source.url.path)|\(ObjectIdentifier(source.orientedImage))", depthIdentity: depthIdentity + "|material-maps|" + auxiliaryKey, settings: settings)
    }

    private var currentCache: MaterialRenderCache? {
        guard let materialCache, materialCache.key == currentRenderKey() else { return nil }
        return materialCache
    }

    var materialNeedsUpdate: Bool { currentCache == nil }
    var fullQualityAvailable: Bool { source != nil && (selectedPreview == .source || currentCache != nil) }
    var materialStatusText: String {
        if let cache = currentCache { return "\(cache.material.outputSize) × \(cache.material.outputSize) maps ready · Full Quality and export reuse these maps" }
        return source == nil ? "Import a surface photo to begin" : "Generate material at \(settings.outputSize) × \(settings.outputSize) to inspect or export its maps"
    }
    func cachedMaterialMapURL(_ selection: MaterialPreview) -> URL? { currentCache?.mapURL(selection) }

    private func checkpointCacheKey(_ checkpoint: SelectedMaterialCheckpoint) -> String {
        let file = URL(fileURLWithPath: checkpoint.checkpointPath)
        let values = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        return "\(checkpoint.selectionIdentity)|\(checkpoint.pythonPath)|\(checkpoint.modelDirectory)|\(checkpoint.codeDirectory)|\(settings.outputSize)|\(source.map { ObjectIdentifier($0.orientedImage).debugDescription } ?? "")|\(values?.fileSize ?? -1)|\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
    }

    func reloadSelectedCheckpoint(activate: Bool = false, target: String? = nil) {
        selectedMaterialMaps = SelectedMaterialCheckpoint.readAll(heightRegistryURL: checkpointRegistryURL)
        let activateHeight = activate && (target == nil || target == "height")
        if activate, target != "height" { recipeMapCheckpoints = nil; markEdited() }
        let selected = try? SelectedMaterialCheckpoint.read(from: checkpointRegistryURL)
        let changed = selected?.selectionIdentity != lastRegistrySelectionIdentity
        lastRegistrySelectionIdentity = selected?.selectionIdentity
        if activateHeight || changed { recipeCheckpoint = nil }
        if let recipeCheckpoint, let selected { selectedMaterialCheckpoint = recipeCheckpoint.resolve(using: selected) }
        else { selectedMaterialCheckpoint = selected }
        guard (activateHeight || changed), selected?.target == "height" else { return }
        guard selected?.supportsStudioInference == true else { return }
        heightSourceNotice = nil
        depthChoice = .materialCheckpoint
        invalidateModelDepth()
    }

    var activeHeightSourceLabel: String {
        switch depthChoice {
        case .materialCheckpoint: selectedMaterialCheckpoint?.title ?? "Material checkpoint"
        case .model: "Custom local depth model"
        case .attached: depthURL?.lastPathComponent ?? "Attached height / depth map"
        case .photoDetail: "Flat surface"
        }
    }
    var activeMaterialModels: [String: SelectedMaterialCheckpoint] {
        var selected = (try? activeAuxiliaryCheckpoints()) ?? [:]
        if depthChoice == .materialCheckpoint, let height = try? currentMaterialCheckpoint() { selected["height"] = height }
        return selected
    }

    func choosePhoto() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = "Import Photo"
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            self?.importPhoto(url)
        }
    }

    func importPhoto(_ url: URL) {
        var directory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &directory), directory.boolValue {
            MaterialToolLauncher.open(.dataset, document: url)
            return
        }
        guard !isBusy else { return }
        run("Reading photo and camera information…") {
            let imported = try await self.engine.importPhoto(url)
            let preview = try await self.engine.preview(imported.orientedImage)
            try Task.checkCancellation()
            self.source = imported
            self.sourcePreview = preview
            self.materialCache = nil
            self.recipeCheckpoint = nil
            self.recipeMapCheckpoints = nil
            self.attachedDepth = nil
            self.generatedDepth = nil
            self.generatedProvenance = nil
            self.generatedModelID = nil
            self.generatedCheckpointKey = nil
            self.generatedResolution = nil
            self.generatedModelPath = nil
            self.generatedFileSignature = nil
            self.decision = nil
            self.depthURL = nil
            self.result = nil
            self.recipeURL = nil
            self.exportURL = nil
            self.selectedMaterialCheckpoint = try? SelectedMaterialCheckpoint.read(from: self.checkpointRegistryURL)
            self.selectedMaterialMaps = SelectedMaterialCheckpoint.readAll(heightRegistryURL: self.checkpointRegistryURL)
            self.lastRegistrySelectionIdentity = self.selectedMaterialCheckpoint?.selectionIdentity
            if self.depthChoice == .attached {
                self.depthChoice = .photoDetail
            }
            if self.depthChoice == .materialCheckpoint, self.selectedMaterialCheckpoint?.supportsStudioInference != true {
                self.depthChoice = .photoDetail
            }
            self.hasEdits = false
            self.preview = preview
            self.renderedPreview = .source
            self.warnings = []
        }
    }

    func chooseDepth() {
        guard source != nil, !isBusy else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = "Attach Depth Map"
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            self?.attachDepth(url)
        }
    }

    func attachDepth(_ url: URL) {
        guard let source, !isBusy else { return }
        run("Reading the attached depth map…") {
            let depth = try await self.engine.importDepth(url, matching: source)
            try Task.checkCancellation()
            self.attachedDepth = depth
            self.depthURL = url
            self.depthChoice = .attached
            self.markEdited()
        }
    }

    func markEdited() {
        hasEdits = true
        result = currentCache?.material
        if result == nil, renderedPreview != .source {
            preview = sourcePreview
            renderedPreview = .source
        }
    }

    func updatePreview(models: ModelManager) {
        guard source != nil, !isBusy else { return }
        if depthChoice == .model, models.availableURL(for: modelID) == nil {
            models.refresh()
            showModelRecovery = true
            return
        }
        run("Generating \(settings.outputSize) × \(settings.outputSize) material maps…") {
            let material = try await self.ensureMaterial(models: models).material
            let selection: MaterialPreview = self.selectedPreview == .source ? .diffuse : self.selectedPreview
            let (preview, rendered) = try await self.preparePreview(source: self.source!, result: material, selection: selection)
            try Task.checkCancellation()
            self.result = material
            self.warnings = material.warnings
            self.selectedPreview = selection
            self.preview = preview
            self.renderedPreview = rendered
        }
    }

    private func ensureMaterial(models: ModelManager) async throws -> MaterialRenderCache {
        renderModels = models
        if let currentCache { return currentCache }
        guard let source else { throw StudioError("Import a surface photo to generate its material.") }
        var preparedDiffuse: TextureEngine.PreparedDiffuse?
        let auxiliary = try activeAuxiliaryCheckpoints()
        if depthChoice == .materialCheckpoint || !auxiliary.isEmpty {
            activity = "Preparing the diffuse input for the material models…"
            preparedDiffuse = try await engine.prepareDiffuse(source: source, settings: settings)
        }
        let depth: TextureDepth?
        if depthChoice == .materialCheckpoint {
            let checkpoint = try currentMaterialCheckpoint()
            let prepared = preparedDiffuse!
            if let checkpointPredictor {
                let input = TextureSource(url: source.url, orientedImage: prepared.diffuse, camera: source.camera,
                    pixelWidth: settings.outputSize, pixelHeight: settings.outputSize)
                depth = try await checkpointPredictor(input, checkpoint, settings.outputSize)
            } else {
                depth = try await materialCheckpointService.predict(diffuse: prepared.diffuse, checkpoint: checkpoint, size: settings.outputSize)
            }
        } else { depth = try await selectedDepth(models: models) }
        var modelMaps: [String: MaterialModelMap] = [:]
        for target in ["roughness", "normal"] {
            guard let checkpoint = auxiliary[target], let preparedDiffuse else { continue }
            activity = "Generating \(target) with \(checkpoint.title)…"
            if let mapPredictor { modelMaps[target] = try await mapPredictor(preparedDiffuse.diffuse, checkpoint, settings.outputSize) }
            else { modelMaps[target] = try await materialCheckpointService.predictMap(diffuse: preparedDiffuse.diffuse, checkpoint: checkpoint, size: settings.outputSize) }
            try Task.checkCancellation()
        }
        guard let key = currentRenderKey() else { throw StudioError("Choose a height source before generating the material.") }
        let renderSettings = settings
        activity = "Generating \(renderSettings.outputSize) × \(renderSettings.outputSize) maps…"
        let material: MaterialResult
        if let materialProcessor { material = try await materialProcessor(source, renderSettings, depth) }
        else { material = try await engine.process(source: source, settings: renderSettings, attachedDepth: depth, preparedDiffuse: preparedDiffuse, modelMaps: modelMaps) }
        activity = "Retaining native maps for immediate inspection and export…"
        let cache = try await MaterialRenderCache.make(material: material, key: key, engine: engine)
        try Task.checkCancellation()
        guard currentRenderKey() == key else { throw CancellationError() }
        materialCache = cache
        result = cache.material
        warnings = cache.material.warnings
        return cache
    }

    func selectPreview(_ selection: MaterialPreview) {
        selectedPreview = selection
        guard !isBusy, source != nil else { return }
        run("Updating preview…") { try await self.renderSelectedPreview() }
    }

    private func renderSelectedPreview() async throws {
        guard let source else { return }
        let (preview, rendered) = try await preparePreview(source: source, result: result, selection: selectedPreview)
        try Task.checkCancellation()
        self.preview = preview
        renderedPreview = rendered
    }

    private func preparePreview(source: TextureSource, result: MaterialResult?, selection: MaterialPreview) async throws -> (CGImage, MaterialPreview) {
        let image: CIImage
        if selection == .source { image = source.orientedImage }
        else if let result {
            image = switch selection {
            case .source: source.orientedImage
            case .diffuse: result.diffuse
            case .roughness: result.roughness
            case .normal: result.normal
            case .height: result.height
            }
        } else { image = source.orientedImage }
        let rendered = result == nil ? MaterialPreview.source : selection
        if rendered == .source || rendered == .diffuse {
            return (try await engine.preview(image), rendered)
        } else {
            return (try await engine.mapPreview(image), rendered)
        }
    }

    private func selectedDepth(models: ModelManager) async throws -> TextureDepth? {
        switch depthChoice {
        case .materialCheckpoint:
            throw StudioError("Material inference requires the prepared diffuse input.")
        case .photoDetail: return nil
        case .attached:
            guard let attachedDepth else { throw StudioError("Attach a height/depth map, or choose Flat surface.") }
            return TextureDepth(image: attachedDepth.image, sourceLabel: attachedDepth.sourceLabel,
                interpretation: settings.attachedMapIsHeight ? .surfaceHeight : .distance)
        case .model:
            guard let modelURL = models.availableURL(for: modelID) else {
                showModelRecovery = true
                throw StudioError("The depth model is missing. Locate it or download it in Models.")
            }
            guard let descriptor = models.catalog.first(where: { $0.id == modelID }) else {
                throw StudioError("Choose an available depth model in Local Models.")
            }
            let signature = modelFileSignature(modelURL, record: models.records[modelID])
            if let generatedDepth, generatedModelID == modelID,
               generatedModelPath == modelURL.path, generatedResolution == settings.modelProcessResolution,
               generatedFileSignature == signature {
                return generatedDepth
            }
            activity = "Generating depth with \(descriptor.name)…"
            let input = try await engine.preview(source!.orientedImage, maxDimension: settings.modelProcessResolution)
            let prediction: ModelDepthResult
            do {
                prediction = try await modelService.predict(image: input, modelURL: modelURL,
                    outputName: models.records[modelID]?.selectedOutput)
            } catch LocalModelError.missingModel {
                models.refresh()
                showModelRecovery = true
                throw StudioError("The model moved while loading. Locate it or download it in Models.")
            }
            let inverse = customInverseDepth
            let depth = try TextureDepth(width: prediction.width, height: prediction.height,
                                         values: prediction.values, sourceLabel: "\(descriptor.name) · \(prediction.width) × \(prediction.height)",
                                         interpretation: inverse ? .inverseDepth : .distance)
            try Task.checkCancellation()
            generatedDepth = depth
            generatedCheckpointKey = nil
            generatedProvenance = prediction.provenance
            generatedModelID = modelID
            generatedResolution = settings.modelProcessResolution
            generatedModelPath = modelURL.path
            generatedFileSignature = signature
            return depth
        }
    }

    func invalidateModelDepth() {
        generatedDepth = nil
        generatedCheckpointKey = nil
        generatedProvenance = nil
        generatedModelID = nil
        generatedResolution = nil
        generatedModelPath = nil
        generatedFileSignature = nil
        markEdited()
    }

    func chooseExport(models: ModelManager) {
        guard source != nil, !isBusy else { return }
        if depthChoice == .model, models.availableURL(for: modelID) == nil {
            showModelRecovery = true
            return
        }
        let panel = NSSavePanel()
        panel.title = "Export Blender Material"
        panel.message = "Create a new material folder containing all four maps and a Blender setup script."
        panel.nameFieldStringValue = "\(source!.url.deletingPathExtension().lastPathComponent)-material"
        panel.canCreateDirectories = true
        panel.prompt = "Export Material"
        panel.directoryURL = exportDirectory.map { URL(fileURLWithPath: $0, isDirectory: true) }
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            self?.export(to: url, models: models)
        }
    }

    func export(to url: URL, models: ModelManager, reveal: Bool = true) {
        guard source != nil, !isBusy else { return }
        run("Exporting material maps…") {
            guard !FileManager.default.fileExists(atPath: url.path) else {
                throw StudioError("Choose a new material folder to preserve the previous export.")
            }
            let staging = url.deletingLastPathComponent().appendingPathComponent(".texture-export-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: staging) }
            let cache = try await self.ensureMaterial(models: models)
            try await self.renderSelectedPreview()
            let exportSettings = self.settings
            self.activity = "Writing PNG, linear EXR maps and material metadata…"
            try await cache.export(to: staging, precision: self.settings.exrPrecision, settings: exportSettings, engine: self.engine)
            try BlenderMaterialScript.write(to: staging, settings: exportSettings)
            if self.depthChoice == .model, let provenance = self.generatedProvenance {
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(provenance).write(to: staging.appendingPathComponent("depth-source.json"), options: .atomic)
            }
            if self.depthChoice == .materialCheckpoint {
                try self.materialCheckpointProvenance().write(to: staging.appendingPathComponent("depth-source.json"), options: .atomic)
            }
            let modelSelections = try self.activeAuxiliaryCheckpoints()
            if !modelSelections.isEmpty {
                let identity = modelSelections.mapValues { ["checkpoint_path": $0.checkpointPath, "checkpoint_sha256": $0.sha256, "target": $0.target, "model_name": $0.title] }
                try JSONSerialization.data(withJSONObject: ["input": "Shared prepared diffuse map", "native_size": self.settings.outputSize,
                    "numeric_values_preserved": true, "models": identity], options: [.prettyPrinted, .sortedKeys])
                    .write(to: staging.appendingPathComponent("material-models.json"), options: .atomic)
            }
            if let decision = self.decision {
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(decision).write(to: staging.appendingPathComponent("photo-review.json"), options: .atomic)
            }
            try Task.checkCancellation()
            try FileManager.default.moveItem(at: staging, to: url)
            self.exportURL = url
            self.exportDirectory = url.deletingLastPathComponent().path
            self.savePreferences()
            self.warnings = cache.material.warnings
            if reveal { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        }
    }

    func saveRecipe() {
        guard let source, !isBusy else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = source.url.deletingPathExtension().lastPathComponent + ".texture.json"
        panel.begin { [weak self] response in
            guard let self, response == .OK, let url = panel.url else { return }
            do {
                let recipe = try self.makeRecipe()
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(recipe).write(to: url, options: .atomic)
                self.recipeURL = url
                self.hasEdits = false
            } catch { self.report(error) }
        }
    }

    private func currentMaterialCheckpoint() throws -> SelectedMaterialCheckpoint {
        let runtime = try SelectedMaterialCheckpoint.read(from: checkpointRegistryURL)
        let checkpoint = recipeCheckpoint?.resolve(using: runtime) ?? runtime
        guard checkpoint.supportsStudioInference else { throw StudioError("Select a supported material safetensors checkpoint in Model Training.") }
        return checkpoint
    }
    private func activeAuxiliaryCheckpoints() throws -> [String: SelectedMaterialCheckpoint] {
        let selected = SelectedMaterialCheckpoint.readAll(heightRegistryURL: checkpointRegistryURL)
        guard let pins = recipeMapCheckpoints else { return selected.filter { ["roughness", "normal"].contains($0.key) } }
        var resolved: [String: SelectedMaterialCheckpoint] = [:]
        for (target, identity) in pins {
            guard ["roughness", "normal"].contains(target), identity.target == target,
                  let runtime = selected[target] ?? selected["height"] ?? selected.values.first else {
                throw StudioError("Reconnect the material runtime in Model Training to use this recipe's \(target) model.")
            }
            resolved[target] = identity.resolve(using: runtime)
        }
        return resolved
    }

    func makeRecipe() throws -> TextureRecipe {
        guard let source else { throw StudioError("Import a photo before saving a recipe.") }
        var recipe = TextureRecipe(photoPath: source.url.path, depthPath: depthURL?.path,
            depthChoice: depthChoice, modelID: modelID, customInverseDepth: customInverseDepth, settings: settings)
        if depthChoice == .materialCheckpoint { recipe.materialCheckpoint = MaterialCheckpointIdentity(try currentMaterialCheckpoint()) }
        recipe.materialMapCheckpoints = try activeAuxiliaryCheckpoints().mapValues(MaterialCheckpointIdentity.init)
        return recipe
    }

    func materialCheckpointProvenance() throws -> Data {
        let checkpoint = try currentMaterialCheckpoint()
        return try JSONSerialization.data(withJSONObject: [
            "backend": "PyTorch / Metal", "model_type": "PBRnxt material refinement",
            "input": "Prepared diffuse map shared with material rendering", "checkpoint_path": checkpoint.checkpointPath,
            "checkpoint_sha256": checkpoint.sha256, "target": checkpoint.target,
            "model_name": checkpoint.title, "height_interpretation": "surface height; no camera-depth normalization"
        ], options: [.prettyPrinted, .sortedKeys])
    }

    func chooseRecipe() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            self?.openRecipe(url)
        }
    }

    func openRecipe(_ url: URL) {
        guard !isBusy else { return }
        run("Opening texture recipe…") {
            let recipe = try JSONDecoder().decode(TextureRecipe.self, from: Data(contentsOf: url))
            guard recipe.version == 3 else { throw StudioError("This recipe version is not supported.") }
            let runtime = try? SelectedMaterialCheckpoint.read(from: self.checkpointRegistryURL)
            let pinnedCheckpoint = recipe.depthChoice == .materialCheckpoint ? recipe.materialCheckpoint : nil
            let photoURL = URL(fileURLWithPath: recipe.photoPath)
            guard FileManager.default.fileExists(atPath: photoURL.path) else {
                throw StudioError("The original photo has moved. Import its new location, then reapply the saved settings.")
            }
            let source = try await self.engine.importPhoto(photoURL)
            var depth: TextureDepth?
            var depthURL: URL?
            if (recipe.depthChoice == .attached || recipe.depthChoice == .materialCheckpoint), let path = recipe.depthPath {
                let candidate = URL(fileURLWithPath: path)
                if FileManager.default.fileExists(atPath: path) {
                    depth = try await self.engine.importDepth(candidate, matching: source)
                    depthURL = candidate
                }
            }
            let preview = try await self.engine.preview(source.orientedImage)
            try Task.checkCancellation()
            self.source = source
            self.sourcePreview = preview
            self.materialCache = nil
            self.recipeCheckpoint = pinnedCheckpoint
            self.recipeMapCheckpoints = recipe.materialMapCheckpoints
            self.selectedMaterialMaps = SelectedMaterialCheckpoint.readAll(heightRegistryURL: self.checkpointRegistryURL)
            self.lastRegistrySelectionIdentity = runtime?.selectionIdentity
            self.selectedMaterialCheckpoint = pinnedCheckpoint.flatMap { pin in runtime.map { pin.resolve(using: $0) } } ?? runtime
            self.settings = recipe.settings
            self.attachedDepth = depth
            self.depthURL = depthURL
            self.generatedDepth = nil
            self.generatedCheckpointKey = nil
            self.generatedProvenance = nil
            self.generatedModelID = nil
            self.generatedResolution = nil
            self.generatedModelPath = nil
            self.generatedFileSignature = nil
            if recipe.depthChoice == .materialCheckpoint {
                self.depthChoice = .materialCheckpoint
            } else {
                self.depthChoice = recipe.depthChoice == .attached && depth == nil ? .photoDetail : recipe.depthChoice
            }
            self.modelID = recipe.modelID
            self.customInverseDepth = recipe.customInverseDepth
            self.recipeURL = url
            self.decision = nil
            self.exportURL = nil
            self.warnings = []
            self.hasEdits = false
            self.result = nil
            self.preview = preview
            self.renderedPreview = .source
        }
    }

    func cancel() { materialCheckpointService.cancel(); operation?.cancel() }

    func inspectFullQuality(models: ModelManager) {
        guard let source, !isBusy else { return }
        if selectedPreview == .source {
            ReviewWindowController.shared.open(candidates: [MapReviewCandidate(id: source.url.path, label: source.url.lastPathComponent, mapURL: source.url, numeric: false)])
            return
        }
        guard let cache = currentCache, let url = cache.mapURL(selectedPreview) else { return }
        let choice = selectedPreview
        let target = choice == .height ? "height" : choice == .normal ? "normal" : choice == .roughness ? "roughness" : "input"
        let checkpoint = activeMaterialModels[target]
        var candidates = [MapReviewCandidate(id: url.path,
            label: "\(source.url.deletingPathExtension().lastPathComponent) · \(choice.rawValue) · \(cache.material.outputSize) × \(cache.material.outputSize)",
            mapURL: url, numeric: choice != .diffuse, role: choice == .diffuse ? "diffuse" : checkpoint == nil ? "map" : "checkpoint",
            modelIdentity: checkpoint.map { MapReviewModelIdentity(checkpointPath: $0.checkpointPath,
                checkpointSHA256: $0.sha256, architecture: $0.modelSummary, mapType: target, modelName: $0.title) })]
        if choice != .diffuse, let diffuse = cache.mapURL(.diffuse) {
            candidates.insert(MapReviewCandidate(label: "Diffuse · model input", mapURL: diffuse, numeric: false, role: "diffuse"), at: 0)
        }
        ReviewWindowController.shared.open(candidates: candidates, retaining: cache)
    }

    func requestAdvice(adviser: OllamaDecisionService) {
        guard let source, !isBusy, !adviser.isBusy else { return }
        run("Reviewing the photo with local Clef…") {
            let image = try await self.engine.preview(source.orientedImage, maxDimension: 768)
            let decision = try await adviser.propose(image: image)
            try Task.checkCancellation()
            self.decision = decision
        }
    }

    func applyDecision() {
        guard let decision, !isBusy else { return }
        settings.lightingStrength = switch decision.lighting { case .none: 0; case .mild: 0.35; case .strong: 0.7 }
        settings.heightStrength = switch decision.relief { case .subtle: 0.3; case .medium: 0.65; case .strong: 1 }
        settings.roughnessBase = switch decision.roughness { case .matte: 0.8; case .mixed: 0.55; case .glossy: 0.25 }
        markEdited()
    }

    private func run(_ label: String, work: @escaping @MainActor () async throws -> Void) {
        guard !isBusy else { return }
        isBusy = true
        activity = label
        let previousDepth = generatedDepth
        let previousProvenance = generatedProvenance
        let previousModel = generatedModelID
        let previousResolution = generatedResolution
        let previousPath = generatedModelPath
        let previousSignature = generatedFileSignature
        let previousCheckpointKey = generatedCheckpointKey
        operation = Task {
            defer {
                self.isBusy = false; self.activity = ""; self.operation = nil
                if self.pendingCheckpointActivation {
                    self.pendingCheckpointActivation = false
                    self.reloadSelectedCheckpoint(activate: true)
                }
            }
            do {
                try Task.checkCancellation()
                try await work()
            }
            catch is CancellationError {
                self.generatedDepth = previousDepth
                self.generatedProvenance = previousProvenance
                self.generatedModelID = previousModel
                self.generatedResolution = previousResolution
                self.generatedModelPath = previousPath
                self.generatedFileSignature = previousSignature
                self.generatedCheckpointKey = previousCheckpointKey
            }
            catch { self.report(error) }
        }
    }

    private func report(_ error: Error) {
        notice = StudioNotice(title: "Texture Studio", message: error.localizedDescription)
    }

    private func modelFileSignature(_ url: URL, record: InstalledLocalModel?) -> String {
        let weight = url.appendingPathComponent("model.safetensors")
        let config = url.appendingPathComponent("config.json")
        let files = [url, weight, config].map { file in
            let values = try? file.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            return "\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0):\(values?.fileSize ?? -1)"
        }
        return files.joined(separator: "|") + "|\(record?.installedAt.timeIntervalSince1970 ?? 0)|\(record?.selectedOutput ?? "")"
    }
}

struct StudioError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
