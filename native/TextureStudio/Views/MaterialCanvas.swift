import SwiftUI
import UniformTypeIdentifiers

struct MaterialCanvas: View {
    @Bindable var workspace: TextureWorkspace
    let models: ModelManager

    var body: some View {
        VStack(spacing: 0) {
            if let image = workspace.preview {
                StudioPreviewInspection(image: image)
                .overlay(alignment: .topLeading) {
                    Text(workspace.renderedPreview.rawValue)
                        .font(.headline)
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .glassEffect()
                        .padding(16)
                }
            } else {
                ContentUnavailableView {
                    Label("Start with a surface photo", systemImage: "square.3.layers.3d")
                } description: {
                    Text("Straighten the surface, balance lighting, and export a material for Blender Cycles.")
                } actions: {
                    Button("Import Photo…") { workspace.choosePhoto() }
                        .buttonStyle(.glassProminent)
                    Button("Open Recipe…") { workspace.chooseRecipe() }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                if workspace.isBusy {
                    HStack(spacing: 12) {
                        ProgressView().controlSize(.small)
                        Text(workspace.activity).lineLimit(2)
                        Spacer()
                        Button("Cancel") { workspace.cancel() }
                    }
                } else {
                    HStack(alignment: .top) {
                        Text(workspace.materialStatusText).foregroundStyle(.secondary)
                        Spacer(minLength: 8)
                        if let output = workspace.exportURL {
                            Button("Show Export in Finder", systemImage: "folder") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
                                .help("Open the folder containing your diffuse, roughness, normal and displacement maps.")
                        }
                    }
                    if workspace.source != nil {
                        ViewThatFits(in: .horizontal) {
                            HStack(spacing: 10) {
                                generateButton
                                inspectButton
                                Spacer(minLength: 0)
                                exportButton
                            }
                            VStack(alignment: .leading, spacing: 10) {
                                HStack(spacing: 10) { generateButton; inspectButton }
                                exportButton
                            }
                        }
                    }
                }
            }
            .font(.caption)
            .padding(12)
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first, !workspace.isBusy else { return false }
            workspace.importPhoto(url)
            return true
        }
    }

    private var generateButton: some View {
        Button(workspace.materialNeedsUpdate ? (workspace.hasEdits ? "Update Material" : "Generate Material") : "Material Is Current",
               systemImage: workspace.materialNeedsUpdate ? "arrow.trianglehead.2.clockwise" : "checkmark.circle") {
            workspace.updatePreview(models: models)
        }
        .disabled(!workspace.materialNeedsUpdate)
        .help(workspace.materialNeedsUpdate
            ? "Generate all four maps at \(workspace.settings.outputSize) × \(workspace.settings.outputSize). Open Full Quality and export then reuse those maps. ⌘R"
            : "Your maps are current. Open Full Quality or export them now. Change a setting to generate a different result. ⌘R")
    }

    private var inspectButton: some View {
        Button("Open Full Quality", systemImage: "arrow.up.left.and.arrow.down.right") {
            workspace.inspectFullQuality(models: models)
        }
        .disabled(!workspace.fullQualityAvailable)
        .help(workspace.fullQualityAvailable
            ? "Open the selected photo or finished map at its original resolution. Pan, zoom or save a copy for GIMP. No model rerun. ⇧⌘F"
            : "Generate or update the material first. Full Quality opens an existing map without rerunning the model. ⇧⌘F")
    }

    private var exportButton: some View {
        Button("Export Material…", systemImage: "square.and.arrow.up") { workspace.chooseExport(models: models) }
            .buttonStyle(.glassProminent)
            .help("Choose a folder for all four maps and a Blender setup script. Generates any pending changes automatically. ⌘E")
    }
}

private struct StudioPreviewInspection: View {
    let image: CGImage
    @State private var viewport = InspectionViewport()
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button("Fit") { viewport.fit() }.help("Fit the whole image in the canvas.")
                Button("100%") { viewport.setActualSize() }.help("Show one image pixel per screen point. Open Full Quality for the final-resolution map.")
                Button("200%") { viewport.setZoom(2) }.help("Magnify the displayed image to inspect individual pixels.")
                Text("Drag to pan · scroll to move · pinch or ⌘-scroll to zoom").font(.caption).foregroundStyle(.secondary)
                Spacer()
            }.padding(8)
            InspectionCanvas(image: image, viewport: viewport)
        }
    }
}
