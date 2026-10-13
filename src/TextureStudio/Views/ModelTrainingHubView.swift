import SwiftUI

private enum TrainingDestination: String, CaseIterable, Identifiable {
    case specialists, dataset, train, compare, review, checkpoints, overview
    var id: String { rawValue }
    var title: String {
        switch self {
        case .overview: "Getting Started"
        case .dataset: "Dataset"
        case .specialists: "Loom Room"
        case .train: "Train Material"
        case .compare: "Compare Checkpoints"
        case .review: "Review Details"
        case .checkpoints: "Saved Models & Export"
        }
    }
    var symbol: String {
        switch self {
        case .overview: "sparkles"
        case .dataset: "square.stack.3d.up"
        case .specialists: "arrow.triangle.branch"
        case .train: "graduationcap"
        case .compare: "rectangle.split.2x1"
        case .review: "viewfinder"
        case .checkpoints: "shippingbox"
        }
    }
    var tool: MaterialTool? {
        switch self {
        case .dataset: .dataset
        case .train: .train
        case .compare: .compare
        case .review: .review
        default: nil
        }
    }
    var guidance: String {
        switch self {
        case .overview: "Open your source folder, tag your materials and train a model for each surface map."
        case .dataset: "Inspect the images in your source folders and edit their material and map tags."
        case .specialists: "Organize source folders and image tags, teach the classifier, then train and review each material map."
        case .train: "Train displacement, roughness or normal models from matching tagged materials."
        case .compare: "Choose a photo and models, then inspect their outputs side by side."
        case .review: "Inspect original pixels, pan together and pop maps out for a closer look."
        case .checkpoints: "Use, export and recover your material models."
        }
    }
}

struct ModelTrainingHubView: View {
    @Bindable var store: WorkbenchStore
    @AppStorage("trainingDestination", store: UserDefaults(suiteName: "org.ipde.material-tools"))
    private var destinationName = TrainingDestination.specialists.rawValue
    @State private var review = ReviewSessionStore()

    private var destination: TrainingDestination? {
        get { TrainingDestination(rawValue: destinationName) ?? .overview }
        nonmutating set { destinationName = (newValue ?? .overview).rawValue }
    }

    var body: some View {
        NavigationSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Model Training").font(.caption.bold()).foregroundStyle(.secondary).padding(10)
                    ForEach(TrainingDestination.allCases) { item in
                        MaterialSidebarRow(selected: destination == item, action: { destination = item }) {
                            Label(item.title, systemImage: item.symbol)
                        }.help(item.guidance)
                    }
                }.padding(8)
            }
            .focusable().focusEffectDisabled()
            .onKeyPress(.downArrow) { destination = MaterialSidebarSelection.next(destination, in: TrainingDestination.allCases, direction: 1); return .handled }
            .onKeyPress(.upArrow) { destination = MaterialSidebarSelection.next(destination, in: TrainingDestination.allCases, direction: -1); return .handled }
            .navigationSplitViewColumnWidth(min: 190, ideal: 225, max: 270)
            .safeAreaInset(edge: .bottom) {
                VStack(alignment: .leading, spacing: 6) {
                    Label("Local on your Mac", systemImage: "desktopcomputer")
                    Text("Photos and training stay local. Export and upload are choices you make.")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding().frame(maxWidth: .infinity, alignment: .leading)
            }
        } detail: {
            VStack(spacing: 0) {
                if let item = destination, item != .overview {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.title).font(.title3.bold())
                            Text(item.guidance).font(.callout).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if [.train, .compare].contains(item), store.specialistTrainingContext != nil {
                            Button("Loom Room") { destination = .specialists }
                        }
                        if let role = item.tool {
                            Button("Separate Window", systemImage: "macwindow.on.rectangle") { MaterialToolLauncher.open(role) }
                                .help("Open the bundled tool as its own app. Finish saving dataset edits before using two copies at once.")
                        }
                    }.padding(16)
                    Divider()
                }
                switch destination ?? .overview {
                case .overview: overview.modifier(DatasetManagementPresentation(store: store))
                case .dataset: MaterialToolRootView(role: .dataset, store: store, review: review, embeddedInHub: true)
                case .specialists: MaterialSpecialistsView(store: store).modifier(DatasetManagementPresentation(store: store))
                case .train: MaterialToolRootView(role: .train, store: store, review: review, embeddedInHub: true)
                case .compare: MaterialToolRootView(role: .compare, store: store, review: review, embeddedInHub: true)
                case .review: MaterialToolRootView(role: .review, store: store, review: review, embeddedInHub: true)
                case .checkpoints: CheckpointLibraryView(store: store, showsDismissButton: false)
                }
            }
        }
        .navigationTitle("Model Training")
        .frame(minWidth: 1280, minHeight: 780)
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if destination == .overview || destination == .specialists || destination == .checkpoints || (destination == .review && store.isBusy) {
                WorkbenchActivityView(store: store)
            }
        }
        .alert("Model Training", isPresented: Binding(get: {
            store.error != nil && (destination == .overview || destination == .specialists) && store.datasetSheet == nil
        }, set: { if !$0 { store.error = nil } })) {
            Button("OK") { store.error = nil }
        } message: { Text(store.error ?? "") }
        .onChange(of: store.trainingNavigationRequest) { _, _ in destination = .train }
        .onChange(of: store.datasetNavigationRequest) { _, _ in destination = .dataset }
        .onChange(of: store.specialistNavigationRequest) { _, _ in destination = .specialists }
        .onChange(of: store.comparisonNavigationRequest) { _, _ in destination = .compare }
        .onChange(of: destinationName) { _, name in
            if name == TrainingDestination.specialists.rawValue { store.returnToModelWorkshop() }
            reconnectModelsForDestination()
        }
        .onChange(of: store.isBusy) { _, busy in
            if !busy { reconnectModelsForDestination() }
        }
        .task {
            if store.developerMode, destination == .overview { destination = .specialists }
            let modelDestination = destination == .train || destination == .compare || destination == .checkpoints
            store.restore(sourceBrowsing: !modelDestination)
        }
    }

    private func reconnectModelsForDestination() {
        if destination == .dataset, store.modelWorkshopNeedsOriginalSources { store.openModelWorkshopSources() }
        if destination == .train { store.openActiveTrainerDataset() }
        let modelDestination = destination == .train || destination == .compare
        if destination == .checkpoints || (modelDestination && store.specialistTrainingContext == nil && !store.modelWorkshopNeedsOriginalSources) {
            store.reconnectSavedCheckpoints()
        }
    }

    private var overview: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 10) {
                    Label("Train your material models", systemImage: "graduationcap")
                        .font(.largeTitle.bold())
                    Text("Your source folder is your dataset. Keep each material's original images and maps in its own subfolder, tag them by material and map type, then train displacement, roughness or normal models.")
                        .font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                startCard("Organize materials by their tags", symbol: "arrow.triangle.branch", description: "Loom Room groups your tagged materials into categories. Choose a category and surface map to open its training setup.", button: "Open Loom Room") {
                    destination = .specialists
                }
                HStack(alignment: .top, spacing: 18) {
                    startCard("Inspect images and map tags", symbol: "square.stack.3d.up", description: "Review all files in your source folders. Correct material categories and diffuse, displacement, roughness or normal tags before training.", button: "Open Dataset") {
                        destination = .dataset
                    }
                    startCard("Train a surface map", symbol: "graduationcap", description: "Choose the model family, crop size and training settings. Training pairs matching diffuse images with the selected surface-map type.", button: "Open Trainer") {
                        destination = .train
                    }
                }
                GroupBox("Your workspace") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Label(store.dataset == nil ? "Choose a source folder to begin" : store.datasetInventorySummary, systemImage: "square.stack.3d.up")
                            Spacer()
                            Button("Inspect Images & Tags") { destination = .dataset }.disabled(store.isBusy)
                            Button("Choose Source Folder…") { store.chooseDataset() }.disabled(store.isBusy)
                                .help("Select the folder containing your original images and material maps.")
                        }
                        if let dataset = store.dataset {
                            Text(dataset.datasetPath).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                        Divider()
                        Label(store.selectedCheckpoint.map { "Saved checkpoint: \($0.title)" } ?? "No saved checkpoint selected", systemImage: "shippingbox")
                        if store.isBusy { HStack { ProgressView().controlSize(.small); Text(store.activity).font(.caption) } }
                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                }
                VStack(alignment: .leading, spacing: 14) {
                    Text("A workflow you can return to").font(.title2.bold())
                    workflowRow("1", .dataset, "Open your folder and review tags", "Choose your existing source folder. Its current images appear here; correct each material category and map type as needed.")
                    workflowRow("2", .specialists, "Choose a material category", "Refresh categories in Loom Room. Select the category and displacement, roughness or normal map you want to train.")
                    workflowRow("3", .train, "Set the crop size and train", "Use a size that your paired maps can supply. Crop position, flips and rotations are shared by the diffuse and target maps; originals keep their pixels.")
                    workflowRow("4", .compare, "Compare saved models", "Run models on the same photo and compare their outputs side by side. Open Review Details to inspect matching areas at full quality.")
                    workflowRow("5", .checkpoints, "Choose and keep a model", "Name, select and export your saved checkpoints. Approve a category model in Loom Room when you want automatic material generation to use it.")
                }
            }.padding(30).frame(maxWidth: 1100, alignment: .leading).frame(maxWidth: .infinity)
        }
    }

    private func startCard(_ title: String, symbol: String, description: String, button: String, action: @escaping () -> Void) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Label(title, systemImage: symbol).font(.title3.bold())
                Text(description).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 4)
                Button(button, action: action).buttonStyle(.glassProminent).disabled(store.isBusy)
                    .help(description)
            }.padding(12).frame(maxWidth: .infinity, minHeight: 155, alignment: .leading)
        }
    }

    private func workflowRow(_ number: String, _ item: TrainingDestination, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            Text(number).font(.headline).foregroundStyle(.secondary).frame(width: 24)
            VStack(alignment: .leading, spacing: 4) {
                Button(title) { destination = item }.buttonStyle(.link).font(.headline)
                Text(detail).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
    }
}
