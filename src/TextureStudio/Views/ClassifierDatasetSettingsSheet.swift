import SwiftUI

struct ClassifierDatasetSettingsSheet: View {
    @Bindable var store: WorkbenchStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Diffuse classifier dataset").font(.title2.bold())
            Text("Move images between class folders to correct labels; refresh preserves your review. The generated Create ML training folder contains only populated classes, so empty folders can stay available for new examples. Images are copied without resizing.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            VStack(alignment: .leading, spacing: 6) {
                Text("Original material folder").font(.headline)
                HStack {
                    TextField("Leave empty to use the open material dataset", text: $store.classifierSourceFolderPath)
                    Button("Choose…") { store.chooseClassifierFolder(source: true) }
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
