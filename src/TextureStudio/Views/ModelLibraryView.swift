import SwiftUI
import AppKit

struct ModelLibraryView: View {
    @Bindable var textureLoom: TextureLoomStore
    let adviser: OllamaDecisionService?
    @Environment(\.dismiss) private var dismiss
    @State private var notice: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Texture Loom & LoRAs").font(.title2.bold())
                    Text("Customize the built-in model with compatible LoRAs.").foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    GroupBox("Built-in Texture Loom") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(textureLoom.statusDescription).font(.headline)
                            if let package = textureLoom.package {
                                ForEach(package.manifest.categories) { category in
                                    LabeledContent(category.name) { Text(category.models.map { $0.map.capitalized }.joined(separator: ", ")) }
                                }
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(6)
                    }
                    HStack {
                        Text("Installed LoRAs").font(.headline)
                        Spacer()
                        Button("Import LoRA…", systemImage: "square.and.arrow.down") { chooseAdapter() }
                            .disabled(textureLoom.package == nil)
                    }
                    if textureLoom.adapters.isEmpty {
                        ContentUnavailableView("No LoRAs installed", systemImage: "shippingbox", description: Text("Import a Loom Workshop export made for this Texture Loom version."))
                    } else {
                        ForEach(textureLoom.adapters) { adapter in
                            Toggle(isOn: Binding(get: { adapter.enabled }, set: { value in
                                do { try textureLoom.setEnabled(adapter.id, value) } catch { notice = error.localizedDescription }
                            })) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(adapter.manifest.name).font(.headline)
                                    Text(adapter.manifest.categories.compactMap { category in textureLoom.package?.manifest.categories.first(where: { $0.id == category.id })?.name }.joined(separator: ", "))
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                            }.toggleStyle(.switch).padding(.vertical, 6)
                        }
                    }
                    if let adviser { Divider(); OllamaModelControls(adviser: adviser) }
                }
            }
            if let error = notice ?? textureLoom.error { Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled) }
        }.padding(24)
    }
    private func chooseAdapter() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = true
        panel.message = "Choose a Loom Workshop LoRA package for this Texture Loom build."
        panel.prompt = "Import LoRA"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do { try textureLoom.importAdapter(url); notice = nil } catch { notice = error.localizedDescription }
        }
    }
}
