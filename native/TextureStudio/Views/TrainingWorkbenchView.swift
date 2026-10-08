import SwiftUI
import AppKit

struct TrainingWorkbenchView: View {
    @Bindable var store: WorkbenchStore
    @State private var showCheckpoints = false

    var body: some View {
        HSplitView {
            VStack(alignment: .leading, spacing: 18) {
                Label("Material training is paused", systemImage: "pause.circle")
                    .font(.title2.bold())
                Text("The previous material models are retired. Your datasets and saved checkpoints remain intact. A replacement texture-height training backend is being connected.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Text("Preparing and reviewing paired source maps still works. No replacement model is advertised as ready for training or production use.")
                    .font(.callout).foregroundStyle(.secondary)
                GroupBox("Prepare your dataset") {
                    VStack(alignment: .leading, spacing: 12) {
                        if let dataset = store.dataset {
                            Text((store.datasetDisplayURL ?? URL(fileURLWithPath: dataset.datasetPath)).lastPathComponent).font(.headline)
                            Text("\(dataset.materials.count) materials · \(dataset.samples.count) crops")
                                .font(.caption).foregroundStyle(.secondary)
                            Text(store.datasetNativeSizeLabel).font(.caption).foregroundStyle(.secondary)
                            Text(store.datasetDisplayURL?.path ?? dataset.datasetPath)
                                .font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                        Button("Open Dataset…") { store.chooseDataset() }.disabled(store.isBusy)
                        Picker("Native crop size", selection: Binding(get: { store.training.size }, set: store.selectTrainingSize)) {
                            Text("1024 × 1024").tag(1024)
                            Text("2048 × 2048").tag(2048)
                        }.disabled(store.isBusy)
                            .help("Prepare matching photo and map crops from the original sources. This does not start a model run.")
                        Text("Changing size prepares native crops automatically. Original maps and the existing dataset stay intact.")
                            .font(.caption).foregroundStyle(.secondary)
                        if !store.datasetPreparationSummary.isEmpty {
                            Text(store.datasetPreparationSummary).font(.caption).foregroundStyle(.secondary)
                        }
                    }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
                }
                Button("Review Saved Checkpoints…", systemImage: "shippingbox") { showCheckpoints = true }
                    .help("Inspect, compare or export the older experimental models. They cannot be reactivated in Texture Studio.")
                Spacer()
            }.padding(20).frame(minWidth: 360, idealWidth: 420, maxWidth: 520)
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(store.isPreparingDataset ? "Preparing native crops" : "Saved operation log").font(.headline)
                    Spacer()
                    if store.isBusy {
                        ProgressView().controlSize(.small)
                        Button(store.isStopping ? "Stopping…" : "Stop") { store.stop() }.disabled(store.isStopping)
                    }
                }
                Text("Earlier logs and checkpoints are retained for inspection. Archived models are available through Review Saved Checkpoints.")
                    .font(.caption).foregroundStyle(.secondary)
                ScrollView {
                    Text(store.logText.isEmpty ? "No saved log. Dataset preparation and legacy comparison events appear here." : store.logText)
                        .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }.frame(maxHeight: .infinity)
                if !store.activity.isEmpty { Text(store.activity).font(.caption).foregroundStyle(.secondary) }
                if let error = store.error {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
                HStack {
                    if let output = store.lastOutputURL {
                        Button("Show Run Folder") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
                    }
                    if let log = store.lastLogURL {
                        Button("Show Log") { NSWorkspace.shared.activateFileViewerSelecting([log]) }
                    }
                }
            }.padding(20).frame(minWidth: 420)
        }
        .sheet(isPresented: $showCheckpoints) {
            CheckpointLibraryView(store: store).frame(minWidth: 780, minHeight: 560)
        }
    }
}
