import SwiftUI
import AppKit

struct CheckpointLibraryView: View {
    @Bindable var store: WorkbenchStore
    var showsDismissButton = true
    @Environment(\.dismiss) private var dismiss

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
                        ScrollView {
                          LazyVStack(spacing: 4) {
                            ForEach(store.checkpoints) { checkpoint in
                              MaterialSidebarRow(selected: store.selectedCheckpointId == checkpoint.id, action: { store.selectedCheckpointId = checkpoint.id }) {
                                HStack(spacing: 9) {
                                    Image(systemName: "shippingbox").foregroundStyle(.secondary)
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(checkpoint.title).lineLimit(2)
                                        Text("\(targetTitle(checkpoint.target)) · step \(checkpoint.step)")
                                            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                                    }
                                }
                              }
                            }
                          }.padding(8)
                        }
                        .focusable().focusEffectDisabled()
                        .onKeyPress(.downArrow) { store.selectedCheckpointId = MaterialSidebarSelection.next(store.selectedCheckpointId, in: store.checkpoints.map(\.id), direction: 1); return .handled }
                        .onKeyPress(.upArrow) { store.selectedCheckpointId = MaterialSidebarSelection.next(store.selectedCheckpointId, in: store.checkpoints.map(\.id), direction: -1); return .handled }
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
        .task {
            // Workspace restoration may still be reconnecting checkpoints.
            // Account discovery is read-only and never uploads anything.
            while store.isBusy {
                do { try await Task.sleep(for: .milliseconds(100)) }
                catch { return }
            }
            if !store.uploadAccountChecked { store.refreshUploadAccount() }
        }
        .onChange(of: store.uploadRepo) { _, _ in store.saveUploadConfiguration() }
        .onChange(of: store.uploadPublic) { _, _ in store.saveUploadConfiguration() }
    }

    private func checkpointDetail(_ checkpoint: WorkbenchCheckpoint) -> some View {
        Form {
            Section("Selected model") {
                Text(checkpoint.title).font(.headline).textSelection(.enabled)
                LabeledContent("Target", value: targetTitle(checkpoint.target))
                LabeledContent("Training base", value: checkpoint.trainingBaseLabel)
                LabeledContent("Saved step", value: checkpoint.step.formatted())
                LabeledContent("Interface", value: checkpoint.compatible ? "Compatible" : "Unsupported")
                Text(checkpoint.schema).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            Section("Use this checkpoint") {
                Button("Use in Texture Studio") { store.useSelectedInStudio() }
                    .disabled(store.isBusy || checkpoint.target != "height")
                    .help("Make this exact trained model the active surface-height source in Texture Studio.")
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
                if store.lastPackageCheckpointId == checkpoint.id, let package = store.lastPackageURL {
                    LabeledContent("Export of this model", value: package.lastPathComponent)
                    Button("Show Package") { NSWorkspace.shared.activateFileViewerSelecting([package]) }
                }
            }
            Section("Upload to Hugging Face") {
                HStack(alignment: .firstTextBaseline) {
                    LabeledContent("Account", value: store.uploadAccount ?? "Not signed in")
                    Button("Refresh Account") { store.refreshUploadAccount() }.disabled(store.isBusy)
                        .help("Check the saved Hugging Face CLI login in this Python environment. No files are uploaded.")
                }
                if store.uploadAccount == nil {
                    Text(store.uploadAccountMessage).font(.caption).foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    Button("Copy Login Command") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString("hf auth login", forType: .string)
                    }.help("Paste this command into Terminal to sign in, then click Refresh Account.")
                }
                TextField("Repository (automatic when blank)", text: $store.uploadRepo,
                          prompt: Text(store.suggestedUploadRepo.isEmpty ? "owner/model-name" : store.suggestedUploadRepo))
                    .help("Leave blank to create a separate repository for this selected model in your account, or enter owner/model-name to use a particular repository. This choice is saved.")
                    .disabled(store.isBusy)
                if !store.effectiveUploadRepo.isEmpty {
                    LabeledContent("Destination", value: store.effectiveUploadRepo).textSelection(.enabled)
                    if !HuggingFaceUpload.validRepository(store.effectiveUploadRepo) {
                        Text("Use owner/model-name with letters, numbers, underscores, dots or single hyphens.")
                            .font(.caption).foregroundStyle(.red)
                    }
                }
                Toggle("Public repository", isOn: $store.uploadPublic).disabled(store.isBusy)
                    .help("Off creates a private repository. Existing repository visibility must match this choice; the app never changes its visibility silently.")
                Text("Upload packages the selected checkpoint and its model card automatically. Source photos, optimizer state and pretrained DINOv2 weights stay local.")
                    .font(.caption).foregroundStyle(.secondary)
                Button { store.uploadPackage() } label: {
                    Label("Upload Selected Model", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.glassProminent)
                .disabled(store.isBusy || !store.canUploadSelectedCheckpoint)
                .help("Upload this selected model directly to the destination shown above. A fresh, checksum-verified package is created first. No additional confirmation is shown.")
                if let uploaded = store.lastUploadURL {
                    Link("Open Last Uploaded Model", destination: uploaded)
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
        case "height": "Surface height / displacement"
        case "roughness": "Roughness"
        case "normal": "OpenGL Normal"
        default: value.capitalized
        }
    }
}
