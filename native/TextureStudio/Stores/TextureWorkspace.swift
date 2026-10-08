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
    var settings = TextureSettings()
    var depthChoice = DepthChoice.model
    var modelID = LocalModelDescriptor.da3GiantID
    var customInverseDepth = true
    var selectedPreview = MaterialPreview.diffuse
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
    var showMemoryWarning = false
    var showInspector = true
    var warnings: [String] = []
    var exportURL: URL?
    var recipeURL: URL?
    var hasEdits = false
    private let engine = TextureEngine()
    private let modelService = ModelDepthService()
    private let materialCheckpointService = MaterialCheckpointService()
    let pythonDepthService: PythonDepthService
    private var operation: Task<Void, Never>?
    private var generatedModelID: String?
    private var generatedResolution: Int?
    private var generatedModelPath: String?
    private var generatedFileSignature: String?

    init(pythonDepthService: PythonDepthService = PythonDepthService()) {
        self.pythonDepthService = pythonDepthService
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
        guard !isBusy else { return }
        run("Reading photo and camera information…") {
            let imported = try await self.engine.importPhoto(url)
            let preview = try await self.engine.preview(imported.orientedImage)
            try Task.checkCancellation()
            self.source = imported
            self.attachedDepth = nil
            self.generatedDepth = nil
            self.generatedProvenance = nil
            self.generatedModelID = nil
            self.generatedFileSignature = nil
            self.decision = nil
            self.depthURL = nil
            self.result = nil
            self.recipeURL = nil
            self.exportURL = nil
            self.depthChoice = (try? SelectedMaterialCheckpoint.read()) == nil ? .model : .materialCheckpoint
            self.modelID = LocalModelDescriptor.da3GiantID
            self.settings = TextureSettings()
            self.hasEdits = false
            self.selectedPreview = .source
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
        result = nil
    }

    func updatePreview(models: ModelManager) {
        guard source != nil, !isBusy else { return }
        if depthChoice == .model, models.availableURL(for: modelID) == nil {
            models.refresh()
            showModelRecovery = true
            return
        }
        run("Preparing the material preview…") {
            let depth = try await self.selectedDepth(models: models)
            var previewSettings = self.settings
            previewSettings.outputSize = 1024
            previewSettings.useEmbeddedDepth = false
            let material = try await self.engine.process(source: self.source!, settings: previewSettings, attachedDepth: depth)
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
            let record: SelectedMaterialCheckpoint
            do { record = try SelectedMaterialCheckpoint.read() }
            catch { throw StudioError("No material checkpoint is selected. Open Material Trainer → Checkpoints, locate a height checkpoint, then choose Use in Texture Studio.") }
            activity = "Predicting native surface height with the selected material checkpoint…"
            return try await materialCheckpointService.predict(source: source!, checkpoint: record, size: settings.outputSize)
        case .photoDetail: return nil
        case .attached:
            guard let attachedDepth else { throw StudioError("Attach a height/depth map, or choose Flat surface.") }
            return TextureDepth(image: attachedDepth.image, sourceLabel: attachedDepth.sourceLabel,
                interpretation: settings.attachedMapIsHeight ? .inverseDepth : .distance)
        case .model:
            guard let modelURL = models.availableURL(for: modelID) else {
                showModelRecovery = true
                throw StudioError("The depth model is missing. Locate it or download it in Models.")
            }
            guard let descriptor = models.catalog.first(where: { $0.id == modelID }) else {
                throw StudioError("Choose an available depth model in Local Models.")
            }
            let signature = modelFileSignature(modelURL, record: models.records[modelID])
            if descriptor.backend == .pytorchDA3 && !pythonDepthService.runtimeStatus.isReady {
                showModels = true
                throw StudioError("The local PyTorch runtime is missing or unavailable. Install it or locate its Python executable in Local Models.")
            }
            if descriptor.backend == .pytorchDA3, let generatedDepth, generatedModelID == modelID,
               generatedModelPath == modelURL.path, generatedResolution == settings.modelProcessResolution,
               generatedFileSignature == signature {
                return generatedDepth
            }
            activity = "Generating depth with \(descriptor.name)…"
            let input = try await engine.preview(source!.orientedImage, maxDimension: settings.modelProcessResolution)
            let prediction: ModelDepthResult
            do {
                switch descriptor.backend {
                case .coreML:
                    prediction = try await modelService.predict(image: input, modelURL: modelURL,
                        outputName: models.records[modelID]?.selectedOutput)
                case .pytorchDA3:
                    prediction = try await pythonDepthService.predict(image: input, modelURL: modelURL,
                        processResolution: settings.modelProcessResolution)
                }
            } catch LocalModelError.missingModel {
                models.refresh()
                showModelRecovery = true
                throw StudioError("The model moved while loading. Locate it or download it in Models.")
            }
            let inverse = descriptor.backend == .coreML && customInverseDepth
            let depth = try TextureDepth(width: prediction.width, height: prediction.height,
                                         values: prediction.values, sourceLabel: "\(descriptor.name) · \(prediction.width) × \(prediction.height)",
                                         interpretation: inverse ? .inverseDepth : .distance)
            try Task.checkCancellation()
            generatedDepth = depth
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
        generatedProvenance = nil
        generatedModelID = nil
        generatedResolution = nil
        generatedModelPath = nil
        generatedFileSignature = nil
        markEdited()
    }

    func chooseExport(models: ModelManager, memoryApproved: Bool = false) {
        guard source != nil, !isBusy else { return }
        if depthChoice == .model, models.availableURL(for: modelID) == nil {
            showModelRecovery = true
            return
        }
        if settings.outputSize >= 4098, !memoryApproved {
            showMemoryWarning = true
            return
        }
        let panel = NSSavePanel()
        panel.title = "Export Blender Material"
        panel.message = "Create a new material folder containing all four maps and a Blender setup script."
        panel.nameFieldStringValue = "\(source!.url.deletingPathExtension().lastPathComponent)-material"
        panel.canCreateDirectories = true
        panel.prompt = "Export Material"
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            self?.export(to: url, models: models)
        }
    }

    func export(to url: URL, models: ModelManager, reveal: Bool = true) {
        guard let source, !isBusy else { return }
        run("Rendering \(self.settings.outputSize) × \(self.settings.outputSize) material maps…") {
            guard !FileManager.default.fileExists(atPath: url.path) else {
                throw StudioError("Choose a new material folder to preserve the previous export.")
            }
            let staging = url.deletingLastPathComponent().appendingPathComponent(".texture-export-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: staging) }
            let depth = try await self.selectedDepth(models: models)
            var exportSettings = self.settings
            exportSettings.useEmbeddedDepth = false
            let material = try await self.engine.process(source: source, settings: exportSettings, attachedDepth: depth)
            self.activity = "Writing PNG, linear EXR maps and material metadata…"
            _ = try await self.engine.export(material, to: staging, precision: self.settings.exrPrecision)
            try BlenderMaterialScript.write(to: staging, settings: exportSettings)
            if self.depthChoice == .model, let provenance = self.generatedProvenance {
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(provenance).write(to: staging.appendingPathComponent("depth-source.json"), options: .atomic)
            }
            if let decision = self.decision {
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(decision).write(to: staging.appendingPathComponent("photo-review.json"), options: .atomic)
            }
            try Task.checkCancellation()
            try FileManager.default.moveItem(at: staging, to: url)
            self.exportURL = url
            self.warnings = material.warnings
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
                let recipe = TextureRecipe(photoPath: source.url.path, depthPath: self.depthURL?.path,
                    depthChoice: self.depthChoice, modelID: self.modelID, customInverseDepth: self.customInverseDepth, settings: self.settings)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(recipe).write(to: url, options: .atomic)
                self.recipeURL = url
                self.hasEdits = false
            } catch { self.report(error) }
        }
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
            guard [1, 2].contains(recipe.version) else { throw StudioError("This recipe version is not supported.") }
            let photoURL = URL(fileURLWithPath: recipe.photoPath)
            guard FileManager.default.fileExists(atPath: photoURL.path) else {
                throw StudioError("The original photo has moved. Import its new location, then reapply the saved settings.")
            }
            let source = try await self.engine.importPhoto(photoURL)
            var depth: TextureDepth?
            var depthURL: URL?
            if let path = recipe.depthPath {
                let candidate = URL(fileURLWithPath: path)
                if FileManager.default.fileExists(atPath: path) {
                    depth = try await self.engine.importDepth(candidate, matching: source)
                    depthURL = candidate
                }
            }
            let preview = try await self.engine.preview(source.orientedImage)
            try Task.checkCancellation()
            self.source = source
            self.settings = recipe.settings
            self.settings.useEmbeddedDepth = false
            if recipe.version == 1 {
                self.settings.heightDetail = 0
                self.settings.attachedMapIsHeight = false
            }
            self.attachedDepth = depth
            self.depthURL = depthURL
            self.generatedDepth = nil
            self.generatedProvenance = nil
            self.generatedModelID = nil
            self.generatedFileSignature = nil
            self.depthChoice = recipe.depthChoice == .attached && depth == nil ? .photoDetail : recipe.depthChoice
            self.modelID = recipe.modelID == "depth-anything-v2-small" ? LocalModelDescriptor.da3GiantID : recipe.modelID
            self.customInverseDepth = recipe.customInverseDepth
            self.recipeURL = url
            self.decision = nil
            self.exportURL = nil
            self.warnings = recipe.version == 1 ? ["Recipe upgraded: portrait depth and automatic brightness bumps are disabled. Review the new DA3 surface-height settings."] : []
            self.hasEdits = false
            self.result = nil
            self.selectedPreview = .source
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
        run("Rendering \(settings.outputSize) × \(settings.outputSize) maps for full-quality inspection…") {
            let depth = try await self.selectedDepth(models: models)
            let material = try await self.engine.process(source: source, settings: self.settings, attachedDepth: depth)
            let folder = FileManager.default.temporaryDirectory.appendingPathComponent("material-review-\(UUID().uuidString)")
            _ = try await self.engine.export(material, to: folder, precision: .float32)
            let names: [MaterialPreview: String] = [.diffuse: "diffuse.png", .roughness: "roughness.exr", .normal: "normal.exr", .height: "displacement.exr"]
            let choice = self.selectedPreview
            guard let name = names[choice] else { return }
            let url = folder.appendingPathComponent(name)
            ReviewWindowController.shared.open(candidates: [MapReviewCandidate(id: url.path, label: "\(choice.rawValue) · \(material.outputSize) × \(material.outputSize)", mapURL: url, numeric: choice != .diffuse)])
        }
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
        settings.noiseReduction = switch decision.noise { case .none: 0; case .mild: 0.01; case .moderate: 0.03 }
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
        operation = Task {
            defer { self.isBusy = false; self.activity = ""; self.operation = nil }
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
        let files = [weight, config].map { file in
            let values = try? file.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            return "\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0):\(values?.fileSize ?? -1)"
        }
        return files.joined(separator: "|") + "|\(record?.installedAt.timeIntervalSince1970 ?? 0)"
    }
}

struct StudioError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
