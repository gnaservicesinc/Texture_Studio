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
                .overlay(alignment: .bottom) {
                    if workspace.hasEdits && workspace.result == nil {
                        Button("Update material preview") { workspace.updatePreview(models: models) }
                            .buttonStyle(.glassProminent)
                            .padding(18)
                            .disabled(workspace.isBusy)
                    }
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
            HStack(spacing: 12) {
                if workspace.isBusy {
                    ProgressView().controlSize(.small)
                    Text(workspace.activity).lineLimit(2)
                    Spacer()
                    Button("Cancel") { workspace.cancel() }
                } else {
                    Text(workspace.result == nil ? "Original photo • update preview to see your material" : "1024 px preview • exports use the selected resolution")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Full Quality…", systemImage: "arrow.up.left.and.arrow.down.right") { workspace.inspectFullQuality(models: models) }
                        .help("Inspect the original photo or final-size map with pan, zoom, pop-out and original-file export")
                    if let output = workspace.exportURL {
                        Button("Show Export") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
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
}

private struct StudioPreviewInspection: View {
    let image: CGImage
    @State private var viewport = InspectionViewport()
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button("Fit") { viewport.fit() }
                Button("100%") { viewport.setActualSize() }
                Button("200%") { viewport.setZoom(2) }
                Text("Drag to pan · scroll to move · pinch or ⌘-scroll to zoom").font(.caption).foregroundStyle(.secondary)
                Spacer()
            }.padding(8)
            InspectionCanvas(image: image, viewport: viewport)
        }
    }
}
