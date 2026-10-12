import SwiftUI

struct ClassifierDatasetSettingsSheet: View {
    @Bindable var store: WorkbenchStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Diffuse classifier dataset").font(.title2.bold())
            Text("Classifier preparation runs only when you choose Create / Refresh. Move images between class folders to correct labels; refresh preserves your review. Create ML receives images that supply the selected resolution, copied without resizing. Smaller originals stay in your folders and remain reviewable.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            VStack(alignment: .leading, spacing: 6) {
                if store.dataset != nil {
                    Toggle("Use another original source folder", isOn: $store.classifierUseSourceFolder)
                    if !store.classifierUseSourceFolder {
                        Text("Open dataset: \(store.datasetName) · \(store.datasetResolution) × \(store.datasetResolution)")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
                Text("Original material folder").font(.headline)
                HStack {
                    TextField("Original diffuse source folder", text: $store.classifierSourceFolderPath)
                        .disabled(store.dataset != nil && !store.classifierUseSourceFolder)
                    Button("Choose…") { store.chooseClassifierFolder(source: true) }
                }
                if store.dataset == nil || store.classifierUseSourceFolder {
                    DatasetResolutionPicker(size: $store.classifierSourceMinimumSize)
                }
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Classifier dataset folder").font(.headline)
                HStack {
                    TextField("Dataset destination", text: $store.classifierDatasetPath)
                    Button("Choose…") { store.chooseClassifierFolder(source: false) }
                }
            }
            if let result = store.classifierDatasetResult {
                Text(result.summary).font(.callout)
                if let path = result.trainingPath {
                    Text("Create ML training data: \(path)").font(.callout).textSelection(.enabled)
                    Button("Show Training Data") { store.revealClassifierTrainingDataset() }
                        .disabled(result.trainingReady != true)
                }
                Button("Show Label Review File") { store.revealClassifierReview() }
            }
            HStack {
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Create / Refresh") { dismiss(); store.refreshClassifierDataset() }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
            }
        }
        .padding(24).frame(width: 660).disabled(store.isBusy)
    }
}
