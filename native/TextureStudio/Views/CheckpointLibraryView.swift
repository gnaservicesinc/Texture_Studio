import SwiftUI
import AppKit

struct CheckpointLibraryView: View {
    @Bindable var store: WorkbenchStore
    var showsDismissButton = true
    @Environment(\.dismiss) private var dismiss
    @State private var showUpload = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Material Checkpoints").font(.title2.bold())
                    Text("Choose the exact saved model for training, comparison or Texture Studio.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                if showsDismissButton { Button("Done") { dismiss() }.keyboardShortcut(.cancelAction) }
            }
            .padding(20)
            Divider()
            HSplitView {
                VStack(spacing: 0) {
                    if store.checkpoints.isEmpty {
                        ContentUnavailableView {
                            Label("Add a checkpoint", systemImage: "shippingbox")
                        } description: {
                            Text("Inspect a saved material checkpoint to view its target and training step.")
                        } actions: {
                            Button("Locate Checkpoints…") { store.chooseCheckpoint() }
                                .buttonStyle(.glassProminent)
                                .disabled(store.isBusy)
                        }
                    } else {
                        List(selection: $store.selectedCheckpointId) {
                            ForEach(store.checkpoints) { checkpoint in
                                HStack(spacing: 9) {
                                    Image(systemName: "shippingbox").foregroundStyle(.secondary)
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(checkpoint.title).lineLimit(2)
                                        Text("\(targetTitle(checkpoint.target)) · step \(checkpoint.step)")
                                            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                                    }
                                }
                                .tag(checkpoint.id)
                            }
                        }
                        .listStyle(.sidebar)
                        .disabled(store.isBusy)
                    }
                    Divider()
                    HStack {
                        Button { store.chooseCheckpoint() } label: { Label("Locate…", systemImage: "plus") }
                            .disabled(store.isBusy)
                        Spacer()
                        Button { store.forgetSelectedCheckpoint() } label: { Label("Unlink", systemImage: "minus") }
                            .disabled(store.isBusy || store.selectedCheckpoint == nil)
                            .help("Remove the checkpoint from this library. Its file remains on disk.")
                    }
                    .padding(12)
                }
                .frame(minWidth: 260, idealWidth: 310, maxWidth: 390)
                if let checkpoint = store.selectedCheckpoint {
                    checkpointDetail(checkpoint)
                        .frame(minWidth: 390)
                } else {
                    ContentUnavailableView("Select a checkpoint", systemImage: "shippingbox", description: Text("Select a saved model to inspect or export it."))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                if store.isBusy {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text(store.activity).font(.caption)
                    }
                } else if !store.activity.isEmpty {
                    Text(store.activity).font(.caption).foregroundStyle(.secondary)
                }
                if let error = store.error {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
                Text("Unlinking keeps checkpoint files. Export creates a new package; uploading is an explicit action.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
        }
        .sheet(isPresented: $showUpload) {
            CheckpointUploadSheet(store: store)
                .frame(width: 560)
        }
    }

    private func checkpointDetail(_ checkpoint: WorkbenchCheckpoint) -> some View {
        Form {
            Section("Selected model") {
                Text(checkpoint.title).font(.headline).textSelection(.enabled)
                LabeledContent("Target", value: targetTitle(checkpoint.target))
                LabeledContent("Saved step", value: checkpoint.step.formatted())
                LabeledContent("Interface", value: checkpoint.compatible ? "Compatible" : "Unsupported")
                Text(checkpoint.schema).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            Section("Use this checkpoint") {
                Button("Use in Texture Studio") { store.useSelectedInStudio() }
                    .disabled(store.isBusy || checkpoint.target != "height")
                    .help("Save this exact checkpoint as Studio’s material-height source. In Studio choose Material checkpoint; its file is never replaced by this action.")
                if checkpoint.target != "height" {
                    Text("Texture Studio currently uses displacement checkpoints. This model can be selected for its own training target.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Text("Choose Refine a checkpoint in Train & Refine to start a new run from this model. Resume Saved Run instead restores the optimizer and exact data from an interrupted run.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Model package") {
                Button("Export Package…") { store.exportSelectedCheckpoint() }
                    .disabled(store.isBusy)
                    .help("Create a new portable package containing learned weights, a model card and dependency identities. Training photos and optimizer state stay local.")
                if let package = store.lastPackageURL {
                    LabeledContent("Last exported package", value: package.lastPathComponent)
                    HStack {
                        Button("Show Package") { NSWorkspace.shared.activateFileViewerSelecting([package]) }
                        Button("Upload Exported Package…") { showUpload = true }.disabled(store.isBusy)
                    }
                } else {
                    Text("Export the checkpoint and model card before uploading to Hugging Face.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("File identity") {
                Text(checkpoint.checkpointPath).textSelection(.enabled)
                Text("SHA256: \(checkpoint.sha256)").textSelection(.enabled)
                Button("Show Checkpoint") { NSWorkspace.shared.activateFileViewerSelecting([checkpoint.url]) }
            }
            .font(.caption)
        }
        .formStyle(.grouped)
    }

    private func targetTitle(_ value: String) -> String {
        switch value {
        case "height": "Displacement"
        case "roughness": "Roughness"
        case "normal": "OpenGL Normal"
        default: value.capitalized
        }
    }
}

private struct CheckpointUploadSheet: View {
    @Bindable var store: WorkbenchStore
    @Environment(\.dismiss) private var dismiss

    private var validRepository: Bool {
        let pieces = store.uploadRepo.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "/", omittingEmptySubsequences: false)
        return pieces.count == 2 && pieces.allSatisfy { part in
            !part.isEmpty && part.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == ".") }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Upload model package", systemImage: "square.and.arrow.up").font(.title2.bold())
            if let package = store.lastPackageURL {
                LabeledContent("Package", value: package.lastPathComponent)
                Text(package.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            TextField("Hugging Face repository: owner/model-name", text: $store.uploadRepo)
                .textFieldStyle(.roundedBorder)
            Toggle("Make the repository public", isOn: $store.uploadPublic)
            Text(store.uploadPublic
                 ? "The exported model package will be uploaded to the public repository you enter."
                 : "The exported model package will be uploaded to a private repository. Authenticate your local Hugging Face CLI first.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Spacer()
                Button("Upload Package") {
                    store.uploadRepo = store.uploadRepo.trimmingCharacters(in: .whitespacesAndNewlines)
                    store.uploadPackage()
                    dismiss()
                }
                .buttonStyle(.glassProminent)
                .disabled(store.lastPackageURL == nil || !validRepository || store.isBusy)
            }
        }
        .padding(24)
    }
}
