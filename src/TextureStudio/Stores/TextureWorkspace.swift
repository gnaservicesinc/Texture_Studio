import AppKit
import CoreImage
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class TextureWorkspace {
    var source: TextureSource?
    var attachedDepth: TextureDepth?
    var depthURL: URL?
    var settings = TextureSettings() { didSet { if settings != oldValue { savePreferences(); markEdited() } } }
    var depthChoice = DepthChoice.textureLoom { didSet { if depthChoice != oldValue { savePreferences(); markEdited() } } }
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
    var showInspector = true { didSet { savePreferences() } }
    var warnings: [String] = []
    var exportURL: URL?
    var recipeURL: URL?
    var hasEdits = false
    let textureLoom: TextureLoomStore
    private(set) var classification: String?
    private let engine = TextureEngine()
    private let loomService = TextureLoomService()
    private var operation: Task<Void, Never>?
    private var sourcePreview: CGImage?
    private var materialCache: MaterialRenderCache?
    @ObservationIgnored private let preferences: UserDefaults?
    @ObservationIgnored private let materialProcessor: (@MainActor (TextureSource, TextureSettings, TextureDepth?) async throws -> MaterialResult)?
    private var exportDirectory: String?

    init(preferences: UserDefaults? = nil, textureLoom: TextureLoomStore? = nil,
         materialProcessor: (@MainActor (TextureSource, TextureSettings, TextureDepth?) async throws -> MaterialResult)? = nil) {
        self.preferences = preferences
        self.textureLoom = textureLoom ?? TextureLoomStore()
        self.materialProcessor = materialProcessor
        if let preferences {
            let saved = StudioPreferences.load(from: preferences)
            settings = saved.settings ?? settings
            depthChoice = saved.depthChoice ?? depthChoice
            selectedPreview = saved.selectedPreview.flatMap(MaterialPreview.init(rawValue:)) ?? selectedPreview
            showInspector = saved.showInspector ?? showInspector
            exportDirectory = saved.exportDirectory
        }
    }
    private func savePreferences() {
        guard let preferences else { return }
        StudioPreferences(settings: settings, depthChoice: depthChoice, selectedPreview: selectedPreview.rawValue,
            showInspector: showInspector, exportDirectory: exportDirectory).save(to: preferences)
    }
    private func currentRenderKey() -> MaterialRenderKey? {
        guard let source else { return nil }
        let identity: String
        switch depthChoice {
        case .textureLoom: identity = "\(textureLoom.package?.manifest.id ?? "missing")|\(textureLoom.package?.manifest.version ?? "")|\(textureLoom.revision)"
        case .attached:
            guard let attachedDepth else { return nil }
            identity = "attached|\(ObjectIdentifier(attachedDepth.image))|\(settings.attachedMapIsHeight)"
        case .photoDetail: identity = "flat"
        }
        return MaterialRenderKey(sourceIdentity: "\(source.url.path)|\(ObjectIdentifier(source.orientedImage))", depthIdentity: identity, settings: settings)
    }
    private var currentCache: MaterialRenderCache? {
        guard let materialCache, materialCache.key == currentRenderKey() else { return nil }
        return materialCache
    }
    var materialNeedsUpdate: Bool { currentCache == nil }
    var fullQualityAvailable: Bool { source != nil && (selectedPreview == .source || currentCache != nil) }
    var materialStatusText: String {
        if let cache = currentCache { return "\(cache.material.outputSize) × \(cache.material.outputSize) maps ready" }
        return source == nil ? "Import a surface photo to begin" : "Generate material at \(settings.outputSize) × \(settings.outputSize)"
    }
    var activeHeightSourceLabel: String {
        switch depthChoice {
        case .textureLoom: classification.map { "Texture Loom · \($0)" } ?? "Texture Loom"
        case .attached: depthURL?.lastPathComponent ?? "Attached height / depth map"
        case .photoDetail: "Flat surface"
        }
    }
    func cachedMaterialMapURL(_ selection: MaterialPreview) -> URL? { currentCache?.mapURL(selection) }
    func openDocument(_ url: URL) {
        switch url.pathExtension.lowercased() {
        case "loomlora":
            showModels = true
            do {
                let id = try textureLoom.importAdapter(url)
                try textureLoom.setEnabled(id, true)
            } catch {
                notice = StudioNotice(title: "Import LoRA", message: error.localizedDescription)
            }
        case "textureloom":
            notice = StudioNotice(title: "Texture Loom build package", message: "Include this Loom Factory package when building Texture Studio.")
        case "json": openRecipe(url)
        default: importPhoto(url)
        }
    }
    func choosePhoto() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.prompt = "Import Photo"
        panel.begin { [weak self] response in if response == .OK, let url = panel.url { self?.importPhoto(url) } }
    }
    func importPhoto(_ url: URL) {
        guard !isBusy else { return }
        run("Reading photo…") {
            let imported = try await self.engine.importPhoto(url)
            let preview = try await self.engine.preview(imported.orientedImage)
            try Task.checkCancellation()
            self.source = imported; self.sourcePreview = preview; self.materialCache = nil
            self.attachedDepth = nil; self.depthURL = nil; self.classification = nil
            self.decision = nil; self.result = nil; self.recipeURL = nil; self.exportURL = nil
            if self.depthChoice == .attached { self.depthChoice = .textureLoom }
            self.hasEdits = false; self.preview = preview; self.renderedPreview = .source; self.warnings = []
        }
    }
    func chooseDepth() {
        guard source != nil, !isBusy else { return }
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.prompt = "Attach Height Map"
        panel.begin { [weak self] response in if response == .OK, let url = panel.url { self?.attachDepth(url) } }
    }
    func attachDepth(_ url: URL) {
        guard let source, !isBusy else { return }
        run("Reading attached map…") {
            let depth = try await self.engine.importDepth(url, matching: source)
            try Task.checkCancellation()
            self.attachedDepth = depth; self.depthURL = url; self.depthChoice = .attached; self.markEdited()
        }
    }
    func markEdited() {
        hasEdits = true; result = currentCache?.material
        if result == nil, renderedPreview != .source { preview = sourcePreview; renderedPreview = .source }
    }
    func updatePreview() {
        guard source != nil, !isBusy else { return }
        run("Generating material maps…") {
            let material = try await self.ensureMaterial().material
            if self.selectedPreview == .source { self.selectedPreview = .diffuse }
            try await self.renderSelectedPreview(material: material)
            self.result = material; self.warnings = material.warnings
        }
    }
    private func ensureMaterial() async throws -> MaterialRenderCache {
        if let currentCache { return currentCache }
        guard let source, let key = currentRenderKey() else { throw StudioError("Import a photo and choose a height source.") }
        var depth: TextureDepth?
        var prepared: TextureEngine.PreparedDiffuse?
        var maps: [String: MaterialModelMap] = [:]
        switch depthChoice {
        case .textureLoom:
            guard let package = textureLoom.package else { throw StudioError(textureLoom.statusDescription) }
            activity = "Preparing diffuse and classifying the surface…"
            prepared = try await engine.prepareDiffuse(source: source, settings: settings)
            let prediction = try await loomService.predict(diffuse: prepared!.diffuse, package: package, overrides: textureLoom.activeOverrides())
            classification = prediction.categoryName
            maps = prediction.maps
            if let height = maps.removeValue(forKey: "height") {
                depth = TextureDepth(image: height.image, sourceLabel: height.sourceLabel, interpretation: .surfaceHeight)
                depth?.alignedToOutput = true
            }
        case .attached:
            guard let attachedDepth else { throw StudioError("Attach a height map before generating the material.") }
            depth = TextureDepth(image: attachedDepth.image, sourceLabel: attachedDepth.sourceLabel,
                interpretation: settings.attachedMapIsHeight ? .surfaceHeight : .distance)
        case .photoDetail: depth = nil
        }
        let material: MaterialResult
        if let materialProcessor { material = try await materialProcessor(source, settings, depth) }
        else { material = try await engine.process(source: source, settings: settings, attachedDepth: depth, preparedDiffuse: prepared, modelMaps: maps) }
        let cache = try await MaterialRenderCache.make(material: material, key: key, engine: engine)
        try Task.checkCancellation()
        guard currentRenderKey() == key else { throw CancellationError() }
        materialCache = cache; result = cache.material; warnings = cache.material.warnings
        return cache
    }
    func selectPreview(_ selection: MaterialPreview) {
        guard !isBusy else { return }
        selectedPreview = selection
        run("Updating preview…") { try await self.renderSelectedPreview(material: self.currentCache?.material) }
    }
    private func renderSelectedPreview(material: MaterialResult?) async throws {
        guard let source else { return }
        let image: CIImage
        let choice = material == nil ? MaterialPreview.source : selectedPreview
        switch choice {
        case .source: image = source.orientedImage
        case .diffuse: image = material!.diffuse
        case .roughness: image = material!.roughness
        case .normal: image = material!.normal
        case .height: image = material!.height
        }
        let preview = try await (choice == .source || choice == .diffuse ? engine.preview(image) : engine.mapPreview(image))
        try Task.checkCancellation(); self.preview = preview; renderedPreview = choice
    }
    func chooseExport() {
        guard let source, !isBusy else { return }
        let panel = NSSavePanel(); panel.title = "Export Blender Material"
        panel.message = "Save all four maps and a Blender setup script in a new material folder."
        panel.nameFieldStringValue = source.url.deletingPathExtension().lastPathComponent + "-material"
        panel.canCreateDirectories = true; panel.prompt = "Export Material"
        panel.directoryURL = exportDirectory.map { URL(fileURLWithPath: $0, isDirectory: true) }
        panel.begin { [weak self] response in if response == .OK, let url = panel.url { self?.export(to: url) } }
    }
    func export(to url: URL, reveal: Bool = true) {
        guard source != nil, !isBusy else { return }
        run("Exporting material maps…") {
            guard !FileManager.default.fileExists(atPath: url.path) else { throw StudioError("Choose a new folder to preserve the previous export.") }
            let staging = url.deletingLastPathComponent().appendingPathComponent(".texture-export-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: staging) }
            let cache = try await self.ensureMaterial()
            try await cache.export(to: staging, precision: self.settings.exrPrecision, settings: self.settings, engine: self.engine)
            try BlenderMaterialSetup.write(to: staging, settings: self.settings)
            if self.depthChoice == .textureLoom, let package = self.textureLoom.package {
                let record: [String: Any] = ["model_id": package.manifest.id, "model_version": package.manifest.version,
                    "dataset": self.classification ?? "", "loras": self.textureLoom.adapters.filter(\.enabled).map(\.id)]
                try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
                    .write(to: staging.appendingPathComponent("texture-loom.json"))
            }
            if let decision = self.decision { try JSONEncoder().encode(decision).write(to: staging.appendingPathComponent("photo-review.json")) }
            try Task.checkCancellation(); try FileManager.default.moveItem(at: staging, to: url)
            self.exportURL = url; self.exportDirectory = url.deletingLastPathComponent().path; self.savePreferences()
            if reveal { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        }
    }
    func chooseAuxiliaryExport() {
        guard let source, !isBusy else { return }
        let panel = NSSavePanel(); panel.title = "Export Original Auxiliary Data"
        panel.nameFieldStringValue = source.url.deletingPathExtension().lastPathComponent + "-auxiliary"
        panel.begin { [weak self] response in if response == .OK, let url = panel.url { self?.exportAuxiliary(to: url) } }
    }
    func exportAuxiliary(to url: URL, reveal: Bool = true) {
        guard let source, !isBusy else { return }
        run("Copying original auxiliary data…") {
            _ = try await NativeAuxiliaryExporter.export(sourceURL: source.url, to: url)
            self.exportURL = url
            if reveal { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        }
    }
    func makeRecipe() throws -> TextureRecipe {
        guard let source else { throw StudioError("Import a photo before saving a recipe.") }
        return TextureRecipe(photoPath: source.url.path, depthPath: depthURL?.path, depthChoice: depthChoice, settings: settings,
            textureLoomID: textureLoom.package?.manifest.id, textureLoomVersion: textureLoom.package?.manifest.version,
            enabledLoRAs: textureLoom.adapters.filter(\.enabled).map(\.id))
    }
    func saveRecipe() {
        guard source != nil, !isBusy else { return }
        let panel = NSSavePanel(); panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = source!.url.deletingPathExtension().lastPathComponent + ".texture.json"
        panel.begin { [weak self] response in
            guard let self, response == .OK, let url = panel.url else { return }
            do { try JSONEncoder().encode(self.makeRecipe()).write(to: url, options: .atomic); self.recipeURL = url; self.hasEdits = false }
            catch { self.report(error) }
        }
    }
    func chooseRecipe() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        panel.begin { [weak self] response in if response == .OK, let url = panel.url { self?.openRecipe(url) } }
    }
    func openRecipe(_ url: URL) {
        guard !isBusy else { return }
        run("Opening texture recipe…") {
            let recipe = try JSONDecoder().decode(TextureRecipe.self, from: Data(contentsOf: url))
            guard recipe.version == 4 else { throw StudioError("This recipe belongs to the previous model architecture. Import its photo to create a new recipe.") }
            if recipe.depthChoice == .textureLoom {
                guard self.textureLoom.package?.manifest.id == recipe.textureLoomID,
                      self.textureLoom.package?.manifest.version == recipe.textureLoomVersion else { throw StudioError("This recipe uses another Texture Loom build.") }
                guard Set(recipe.enabledLoRAs).isSubset(of: Set(self.textureLoom.adapters.map(\.id))) else { throw StudioError("Import this recipe's LoRAs before opening it.") }
                for adapter in self.textureLoom.adapters { try self.textureLoom.setEnabled(adapter.id, recipe.enabledLoRAs.contains(adapter.id)) }
            }
            let source = try await self.engine.importPhoto(URL(fileURLWithPath: recipe.photoPath))
            var depth: TextureDepth?
            if recipe.depthChoice == .attached {
                guard let path = recipe.depthPath else { throw StudioError("The recipe has no attached height map.") }
                depth = try await self.engine.importDepth(URL(fileURLWithPath: path), matching: source)
            }
            let preview = try await self.engine.preview(source.orientedImage); try Task.checkCancellation()
            self.source = source; self.attachedDepth = depth; self.depthURL = recipe.depthPath.map { URL(fileURLWithPath: $0) }
            self.settings = recipe.settings; self.depthChoice = recipe.depthChoice; self.recipeURL = url
            self.sourcePreview = preview; self.preview = preview; self.renderedPreview = .source
            self.materialCache = nil; self.result = nil; self.classification = nil; self.warnings = []; self.hasEdits = false
        }
    }
    func inspectFullQuality() {
        guard let source, !isBusy else { return }
        if selectedPreview == .source {
            ReviewWindowController.shared.open(candidates: [MapReviewCandidate(label: source.url.lastPathComponent, mapURL: source.url, numeric: false)])
            return
        }
        guard let cache = currentCache, let url = cache.mapURL(selectedPreview) else { return }
        ReviewWindowController.shared.open(candidates: [MapReviewCandidate(label: selectedPreview.rawValue, mapURL: url, numeric: selectedPreview != .diffuse)], retaining: cache)
    }
    func requestAdvice(adviser: OllamaDecisionService) {
        guard let source, !isBusy, !adviser.isBusy else { return }
        run("Reviewing the photo with Clef…") {
            let image = try await self.engine.preview(source.orientedImage, maxDimension: 768)
            self.decision = try await adviser.propose(image: image)
        }
    }
    func applyDecision() {
        guard let decision, !isBusy else { return }
        settings.lightingStrength = switch decision.lighting { case .none: 0; case .mild: 0.35; case .strong: 0.7 }
        settings.heightStrength = switch decision.relief { case .subtle: 0.3; case .medium: 0.65; case .strong: 1 }
        settings.roughnessBase = switch decision.roughness { case .matte: 0.8; case .mixed: 0.55; case .glossy: 0.25 }
    }
    func cancel() { operation?.cancel() }
    private func run(_ label: String, work: @escaping @MainActor () async throws -> Void) {
        guard !isBusy else { return }
        isBusy = true; activity = label
        operation = Task {
            defer { self.isBusy = false; self.activity = ""; self.operation = nil }
            do { try Task.checkCancellation(); try await work() }
            catch is CancellationError { }
            catch { self.report(error) }
        }
    }
    private func report(_ error: Error) { notice = StudioNotice(title: "Texture Studio", message: error.localizedDescription) }
}
