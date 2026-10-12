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
        case .overview: "A practical path from photographed surfaces to a reusable material model."
        case .dataset: "Create, rename and manage datasets, then inspect their original material maps."
        case .specialists: "Organize source folders and image tags, teach the classifier, then train and review each material map."
        case .train: "Refine displacement, roughness or normals at the selected native pixel grid."
        case .compare: "Prepare diffuse once, then compare matching model outputs."
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
        let modelDestination = destination == .train || destination == .compare
        if destination == .checkpoints || (modelDestination && store.specialistTrainingContext == nil && !store.modelWorkshopNeedsOriginalSources) {
            store.reconnectSavedCheckpoints()
        }
    }

    private var overview: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 10) {
                    Label("MatterLoom texture pipeline", systemImage: "graduationcap")
                        .font(.largeTitle.bold())
                    Text("Build your MatterLoom pipeline with models for surface height, roughness and normals. Train with paired diffuse and surface maps, then inspect the exact inputs and outputs.")
                        .font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                startCard("Build your material models", symbol: "arrow.triangle.branch", description: "Organize your original folders and image tags, teach the classifier, then train displacement, roughness and normal models.", button: "Open Loom Room") {
                    destination = .specialists
                }
                HStack(alignment: .top, spacing: 18) {
                    startCard("Create and manage datasets", symbol: "square.stack.3d.up", description: "Give your dataset a name, add paired material maps, and edit or remove datasets from your library.", button: "New Dataset…") {
                        destination = .dataset
                        store.showNewDatasetSheet = true
                    }
                    startCard("Train your material model", symbol: "shippingbox", description: "Refine the selected material base and save a separate LoRA. Developer mode also saves a full fused checkpoint.", button: "Open Trainer") {
                        destination = .train
                    }
                }
                GroupBox("Your workspace") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Label(store.dataset.map { "\($0.materials.count) materials · \($0.samples.count) material sets" } ?? "Create or open a dataset", systemImage: "square.stack.3d.up")
                            Spacer()
                            Button("Manage Datasets") { destination = .dataset }.disabled(store.isBusy)
                            Button("New Dataset…") {
                                destination = .dataset
                                store.showNewDatasetSheet = true
                            }.disabled(store.isBusy)
                            Button("Open Dataset…") { store.chooseDataset() }.disabled(store.isBusy)
                                .help("Choose an existing dataset folder. Open it in Dataset to rename it, edit its info, add materials or move it to Trash.")
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
                    workflowRow("1", .dataset, "Set up your dataset", "Choose rendering and training resolution, import a material folder, then review the planned native crops and validation checks.")
                    workflowRow("2", .train, "Train material detail", "Choose your map and type the training settings directly. Complete registered maps share the exact grid shown in Dataset.")
                    workflowRow("3", .compare, "Compare what changed", "Use the same prepared diffuse for each checkpoint. Inspect detail, noise, inversion and relief.")
                    workflowRow("4", .review, "Inspect at full quality", "Use 100% zoom, linked dragging and pop-out windows. Export the untouched map or open an editable copy in GIMP.")
                    workflowRow("5", .checkpoints, "Keep the model", "Save the LoRA or full checkpoint, upload it to Hugging Face, and download it again when needed.")
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
