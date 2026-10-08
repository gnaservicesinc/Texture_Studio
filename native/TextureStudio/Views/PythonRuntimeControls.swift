import AppKit
import SwiftUI

struct PythonRuntimeControls: View {
    @Bindable var runtime: PythonDepthService
    @State private var confirmRemoval = false

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("PyTorch · Metal runtime").font(.headline)
            Text("DA3 camera depth uses this local PyTorch / Metal runtime. Trained material-height checkpoints use the Python environment selected in Model Training → Runtime. Neither requires Core ML conversion.")
                .font(.callout).foregroundStyle(.secondary)
            Text(runtime.runtimeStatus.message).font(.caption).foregroundStyle(.secondary)
            if let python = runtime.pythonURL {
                Text(python.path).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
            }
            if let progress = runtime.progress { ProgressView(value: progress) }
            HStack {
                if runtime.isBusy {
                    Button("Cancel") { runtime.cancel() }
                } else {
                    Button("Install Managed Runtime…") { choosePython(install: true) }
                    Button("Locate Python Environment…") { choosePython(install: false) }
                    if runtime.pythonURL != nil {
                        Button("Remove Runtime…", role: .destructive) { confirmRemoval = true }
                    }
                }
            }
            if let error = runtime.lastError { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
        }
        .confirmationDialog("Remove this runtime?", isPresented: $confirmRemoval, titleVisibility: .visible) {
            Button("Remove Runtime", role: .destructive) {
                do { try runtime.removeManagedRuntime() }
                catch { runtime.lastError = error.localizedDescription }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Only an environment installed by Texture Studio is deleted. A located external environment is unlinked. Model weights and photographs remain in place.")
        }
    }

    private func choosePython(install: Bool) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = install ? "Install Runtime" : "Use Environment"
        panel.message = install
            ? "Choose an installed Python 3 executable. Texture Studio will create its own environment and install the pinned PyTorch/DA3 dependencies. Downloads can take several minutes."
            : "Choose the Python executable inside an environment containing PyTorch/MPS and the pinned inference dependencies. DA3 source is bundled with this app. Use Command-Shift-G to enter its full path."
        panel.directoryURL = URL(fileURLWithPath: "/Library/Frameworks/Python.framework/Versions")
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task {
                do {
                    if install { try await runtime.setupRuntime(using: url) }
                    else { try await runtime.locatePython(at: url) }
                } catch { runtime.lastError = error.localizedDescription }
            }
        }
    }
}
