import SwiftUI
import AppKit

struct AdviceReviewView: View {
    @Bindable var workspace: TextureWorkspace
    @Bindable var adviser: OllamaDecisionService
    let models: ModelManager
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("Photo Review", systemImage: "eye").font(.title2.bold())
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            Text("Clef can choose from fixed processing options. Review and apply the proposal, then inspect the actual material preview.")
                .foregroundStyle(.secondary)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    OllamaModelControls(adviser: adviser)
                    Divider()
                    if let decision = workspace.decision {
                        Text("Suggested settings").font(.headline)
                        Grid(alignment: .leading, horizontalSpacing: 22, verticalSpacing: 8) {
                            GridRow { Text("Lighting balance"); Text(decision.lighting.rawValue) }
                            GridRow { Text("Noise reduction"); Text(decision.noise.rawValue) }
                            GridRow { Text("Surface relief"); Text(decision.relief.rawValue) }
                            GridRow { Text("Roughness"); Text(decision.roughness.rawValue) }
                            GridRow { Text("Confidence"); Text(decision.confidence.rawValue) }
                        }
                        Text(decision.rationale).font(.callout).foregroundStyle(.secondary)
                        Button("Apply Suggestions & Preview") {
                            workspace.applyDecision()
                            dismiss()
                            workspace.updatePreview(models: models)
                        }
                        .buttonStyle(.glassProminent).disabled(workspace.isBusy || adviser.isBusy)
                    }
                    if workspace.source == nil {
                        Text("Import a surface photo to ask for suggestions.").foregroundStyle(.secondary)
                    }
                }
            }
            if workspace.isBusy {
                HStack { ProgressView().controlSize(.small); Text(workspace.activity); Spacer(); Button("Cancel") { adviser.cancel(); workspace.cancel() } }
            } else {
                Button("Ask Clef to Review Photo") { workspace.requestAdvice(adviser: adviser) }
                    .buttonStyle(.borderedProminent)
                    .disabled(workspace.source == nil || adviser.isBusy || adviser.modelInfo == nil)
            }
            Text("Fixed choices reduce variation; model judgments still need review. The image is sent only to Ollama on this Mac.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(24)
        .task { await adviser.refresh() }
    }
}

struct OllamaModelControls: View {
    @Bindable var adviser: OllamaDecisionService
    @State private var confirmDownload = false
    @State private var confirmRemoval = false
    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("Clef · MLX NVFP4").font(.headline)
            Text(OllamaDecisionService.model).font(.caption.monospaced())
            Text(OllamaDecisionService.compatibilityNotice).font(.caption).foregroundStyle(.secondary)
            Text("Optional vision decision model, about 18 GB. Runs through local Ollama; it never generates replacement textures.")
                .font(.callout).foregroundStyle(.secondary)
            Text(adviser.status.message).font(.caption).foregroundStyle(.secondary)
            if let info = adviser.modelInfo {
                Text("\(info.format) · \(info.quantization) · \(info.capabilities.joined(separator: ", "))")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let progress = adviser.progress { ProgressView(value: progress) }
            HStack {
                if adviser.isBusy {
                    Button("Cancel") { adviser.cancel() }
                } else {
                    Button("Refresh") { Task { await adviser.refresh() } }
                    Button("Open Ollama") { NSWorkspace.shared.openApplication(at: URL(fileURLWithPath: "/Applications/Ollama.app"), configuration: NSWorkspace.OpenConfiguration()) }
                    if adviser.modelInfo == nil {
                        Button("Download 18 GB Model…") { confirmDownload = true }
                    } else {
                        Button("Delete Model…", role: .destructive) { confirmRemoval = true }
                    }
                }
            }
            Link("Model details and requirements", destination: URL(string: "https://ollama.com/library/clef:27b-nvfp4")!)
                .font(.caption)
            if let error = adviser.lastError { Text(error).font(.caption).foregroundStyle(.red) }
        }
        .confirmationDialog("Download Clef MLX NVFP4?", isPresented: $confirmDownload, titleVisibility: .visible) {
            Button("Download clef:27b-nvfp4 (18 GB)") { adviser.pull() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Ollama will store this model in its own library. It can require substantially more unified memory while running. Delete it here when you no longer need it; the model may be shared by other apps.")
        }
        .confirmationDialog("Delete Clef from Ollama?", isPresented: $confirmRemoval, titleVisibility: .visible) {
            Button("Delete clef:27b-nvfp4", role: .destructive) {
                Task { do { try await adviser.remove() } catch { adviser.lastError = error.localizedDescription } }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This removes the exact model from Ollama’s shared library. Other apps using it will also need to download it again.")
        }
    }
}
