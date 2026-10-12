import SwiftUI

struct SourceImageTagsSheet: View {
    @Bindable var store: WorkbenchStore
    let url: URL
    @Environment(\.dismiss) private var dismiss
    @State private var role = "unknown"
    @State private var classLabel = ""
    @State private var folderName = ""
    @State private var verifiedSHA256: String?
    @State private var loaded = false
    private let roles = ["input", "height", "roughness", "normal", "ancillary", "unknown"]
    private var parent: URL { url.deletingLastPathComponent() }
    private var canRenameFolder: Bool {
        guard let root = store.datasetFolderURL else { return false }
        return parent.resolvingSymlinksInPath().standardizedFileURL.path != root.resolvingSymlinksInPath().standardizedFileURL.path
    }
    private var cleanFolderName: String { folderName.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var classNames: [String] {
        Array(Set(NativeDiffuseClassifierDatasetService.classes.map(\.name)
            + (store.classifierDatasetResult?.classes.map(\.name) ?? [])
            + (store.specialistRegistry?.classifierLabels ?? []))).sorted()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Image Tags", systemImage: "tag").font(.title2.bold())
            Text(url.lastPathComponent).font(.headline)
            Text(url.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            Text("Choose this image's map role and material type. Tags stay with the original file; image pixels remain unchanged.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Form {
                Picker("Map role", selection: $role) {
                    ForEach(roles, id: \.self) { Text(roleTitle($0)).tag($0) }
                }
                LabeledContent("Material type") {
                    HStack {
                        TextField("Unlabeled or custom type", text: $classLabel).textFieldStyle(.roundedBorder)
                        Menu("Choose Type") {
                            Button("Unlabeled") { classLabel = "" }
                            Divider()
                            ForEach(classNames, id: \.self) { label in
                                Button(label) { classLabel = label }
                            }
                        }
                    }
                }
                TextField("Folder name", text: $folderName)
                    .disabled(!canRenameFolder)
                if !canRenameFolder {
                    Text("This image is in the open source root. Rename individual material folders inside it; the root stays in place.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                LabeledContent("Original verification", value: verifiedSHA256 == nil ? "Checked when needed for training" : "Verified original bytes")
            }.formStyle(.grouped).disabled(store.isBusy || !loaded)
            DatasetSheetOperationNotice(store: store)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction).disabled(store.isBusy)
                Button("Save Tags") { save() }
                    .buttonStyle(.glassProminent).keyboardShortcut(.defaultAction)
                    .disabled(store.isBusy || !loaded || cleanFolderName.isEmpty)
            }
        }.padding(24).frame(width: 620)
            .interactiveDismissDisabled(store.isBusy)
            .task {
                let inventory = store.dataset?.sourceInventory?.first { $0.path == url.path }
                role = inventory?.role.flatMap { roles.contains($0) ? $0 : nil } ?? "unknown"
                classLabel = inventory?.classLabel ?? ""
                folderName = parent.lastPathComponent
                let sourceURL = url
                let tags = await Task.detached(priority: .userInitiated) { NativeSourceFileTags.read(sourceURL) }.value
                if let taggedRole = tags?.role, roles.contains(taggedRole) { role = taggedRole }
                if let taggedClass = tags?.classLabel { classLabel = taggedClass }
                verifiedSHA256 = tags?.verifiedSHA256
                loaded = true
            }
    }

    private func save() {
        guard !store.isBusy else { return }
        let renamedFolder = canRenameFolder && cleanFolderName != parent.lastPathComponent ? cleanFolderName : nil
        store.saveSourceImageTags(url, role: role, classLabel: classLabel.trimmingCharacters(in: .whitespacesAndNewlines),
                                  folderName: renamedFolder)
        dismiss()
    }

    private func roleTitle(_ value: String) -> String {
        switch value {
        case "input": "Diffuse / Base Color"
        case "height": "Displacement / Height"
        case "roughness": "Roughness"
        case "normal": "Normal"
        case "ancillary": "Additional Map (AO, masks, other)"
        default: "Unassigned Image"
        }
    }
}
