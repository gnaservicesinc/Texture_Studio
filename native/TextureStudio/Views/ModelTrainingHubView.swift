import SwiftUI

private enum TrainingDestination: String, CaseIterable, Identifiable {
    case overview, dataset, train, compare, review, checkpoints
    var id: String { rawValue }
    var title: String {
        switch self {
        case .overview: "Getting Started"
        case .dataset: "Dataset"
        case .train: "Train & Refine"
        case .compare: "Compare Checkpoints"
        case .review: "Review Details"
        case .checkpoints: "Export & Use Model"
        }
    }
    var symbol: String {
        switch self {
        case .overview: "sparkles"
        case .dataset: "square.stack.3d.up"
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
        case .dataset: "Inspect paired photos and maps. Exclude problem crops without deleting originals."
        case .train: "Choose a starting point, map type and native crop size, then run a bounded experiment."
        case .compare: "Run the same photo through two or more checkpoints and compare matching details."
        case .review: "Inspect original pixels, pan together and pop maps out for a closer look."
        case .checkpoints: "Select the model Studio uses, or export a portable package for another workflow."
        }
    }
}

struct ModelTrainingHubView: View {
    @Bindable var store: WorkbenchStore
    @AppStorage("trainingDestination", store: UserDefaults(suiteName: "org.ipde.material-tools"))
    private var destinationName = TrainingDestination.overview.rawValue
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
                        if let role = item.tool {
                            Button("Separate Window", systemImage: "macwindow.on.rectangle") { MaterialToolLauncher.open(role) }
                                .help("Open the bundled tool as its own app. Finish saving dataset edits before using two copies at once.")
                        }
                    }.padding(16)
                    Divider()
                }
                switch destination ?? .overview {
                case .overview: overview
                case .dataset: MaterialToolRootView(role: .dataset, store: store, review: review)
                case .train: MaterialToolRootView(role: .train, store: store, review: review)
                case .compare: MaterialToolRootView(role: .compare, store: store, review: review)
                case .review: MaterialToolRootView(role: .review, store: store, review: review)
                case .checkpoints: CheckpointLibraryView(store: store, showsDismissButton: false)
                }
            }
        }
        .navigationTitle("Model Training")
        .frame(minWidth: 1280, minHeight: 780)
        .task { store.restore() }
    }

    private var overview: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 10) {
                    Label("Build a model for your materials", systemImage: "graduationcap")
                        .font(.largeTitle.bold())
                    Text("Teach a model useful relief and surface detail from matching photographs and maps. Start with a small experiment, inspect what it actually does, then keep refining the version you like.")
                        .font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                HStack(alignment: .top, spacing: 18) {
                    startCard("Refine a checkpoint", symbol: "arrow.trianglehead.2.clockwise", description: "Continue from a material model you have already trained. A new run keeps the starting checkpoint intact.", button: "Choose & Refine…") {
                        store.training.useWarmStart = true
                        destination = .train
                        if store.selectedCheckpoint == nil { store.chooseCheckpoint() }
                    }
                    startCard("Start from the base", symbol: "leaf", description: "Use the pretrained DINOv2 Base encoder with a fresh material head. Your paired maps teach it height, roughness or normals.", button: "Set Up New Model") {
                        store.training.useWarmStart = false
                        destination = .train
                    }
                }
                GroupBox("Your workspace") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Label(store.dataset.map { "\($0.materials.count) materials · \($0.samples.count) crops" } ?? "Choose a prepared dataset", systemImage: "square.stack.3d.up")
                            Spacer()
                            Button("Open Dataset…") { store.chooseDataset() }.disabled(store.isBusy)
                                .help("Choose dataset.json from your prepared materials. Original source maps allow automatic recropping at another native size.")
                        }
                        if let dataset = store.dataset {
                            Text(dataset.datasetPath).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                        Divider()
                        Label(store.selectedCheckpoint.map { "Starting checkpoint: \($0.title)" } ?? "No starting checkpoint selected", systemImage: "shippingbox")
                        if store.isBusy { HStack { ProgressView().controlSize(.small); Text(store.activity).font(.caption) } }
                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                }
                VStack(alignment: .leading, spacing: 14) {
                    Text("A workflow you can return to").font(.title2.bold())
                    workflowRow("1", .dataset, "Check your source maps", "Review matching diffuse, displacement, roughness and OpenGL normal crops. Keep original high-bit-depth data.")
                    workflowRow("2", .train, "Fit or refine", "Choose 1K or 2K. Matching native crops are prepared automatically from the original sources. Set a time and memory budget.")
                    workflowRow("3", .compare, "Compare what changed", "Use the same photo for each checkpoint. Look for useful detail, noise, inversion and exaggerated relief.")
                    workflowRow("4", .review, "Inspect at full quality", "Use 100% zoom, linked dragging and pop-out windows. Export the untouched map or open an editable copy in GIMP.")
                    workflowRow("5", .checkpoints, "Keep and use the result", "Choose the exact height checkpoint for Texture Studio. Export a package when you want to share or move it.")
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
