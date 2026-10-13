import SwiftUI
import AppKit

struct MaterialSpecialistsView: View {
    @Bindable var store: WorkbenchStore
    @State private var showAdd = false
    @State private var newName = ""
    @State private var newClass = "Wood"
    @State private var newSubtype = ""
    @AppStorage("specialistMinimumConfidence", store: UserDefaults(suiteName: "org.ipde.material-tools"))
    private var minimumConfidence = 0.7

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                workshopSources
                workshopClassifier
                GroupBox("3. Organize material categories") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Use image labels to organize one training dataset per material category. Original files, review notes and saved models remain available.")
                            .foregroundStyle(.secondary)
                        DisclosureGroup("Classifier and source settings") {
                            VStack(alignment: .leading, spacing: 10) {
                                pathField("Classifier export", value: $store.specialistClassifierPath, kind: "classifier", prompt: "Bundled \(NativeDiffuseTextureClassifier.bundledDisplayName) diffuse classifier")
                                pathField("Model workspace", value: $store.specialistWorkspacePath, kind: "workspace")
                                Text("Active originals: \(store.modelWorkshopSourcePath.isEmpty ? "Open a source folder above" : store.modelWorkshopSourcePath)")
                                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                                Text("Classifier-only diffuse examples remain in \(store.classifierDatasetPath). Add matching surface maps to the source dataset when you want to use them for material training.")
                                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                            }.padding(.top, 8)
                        }
                        HStack {
                            Button("Refresh Model Categories", systemImage: "arrow.triangle.branch") { store.refreshSpecialists() }
                                .buttonStyle(.glassProminent).disabled(store.isBusy)
                            if let registry = store.specialistRegistry {
                                Text("\(registry.specialists.count) categories · \(registry.familyCount) paired families · \(registry.classifierOnlyCount) classifier examples")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                }
                if let registry = store.specialistRegistry {
                    HStack(alignment: .top, spacing: 20) {
                        GroupBox("Material categories") {
                            VStack(alignment: .leading, spacing: 4) {
                                ForEach(registry.specialists) { item in
                                    Button {
                                        store.selectedSpecialistID = item.id
                                    } label: {
                                        HStack {
                                            VStack(alignment: .leading, spacing: 3) {
                                                Text(item.name).font(.headline)
                                                Text("\(item.familyIds.count) families · \(item.models.count) saved models")
                                                    .font(.caption).foregroundStyle(.secondary)
                                            }
                                            Spacer()
                                            if store.selectedSpecialist?.id == item.id { Image(systemName: "checkmark.circle.fill") }
                                        }.padding(8).contentShape(Rectangle())
                                    }.buttonStyle(.plain)
                                        .background(store.selectedSpecialist?.id == item.id ? Color.accentColor.opacity(0.12) : Color.clear, in: RoundedRectangle(cornerRadius: 8))
                                }
                                Divider().padding(.vertical, 8)
                                Button("Add Category / Subtype…", systemImage: "plus") {
                                    newClass = store.selectedSpecialist?.classLabel ?? registry.classifierLabels.first ?? "Wood"
                                    showAdd = true
                                }.disabled(store.isBusy)
                            }.padding(8)
                        }.frame(width: 285)
                        if let specialist = store.selectedSpecialist {
                            VStack(alignment: .leading, spacing: 18) {
                                GroupBox("4. Train each map for \(specialist.name)") {
                                    VStack(alignment: .leading, spacing: 12) {
                                        if let subtype = specialist.subtype {
                                            Text("\(specialist.classLabel) · subtype: \(subtype)").foregroundStyle(.secondary)
                                        }
                                        ForEach(["height", "roughness", "normal"], id: \.self) { target in
                                            HStack {
                                                VStack(alignment: .leading, spacing: 3) {
                                                    Text(WorkbenchStore.specialistMapTitle(target)).font(.callout.bold())
                                                    Text(mapTrainingSummary(specialist, target: target)).font(.caption).foregroundStyle(.secondary)
                                                }
                                                Spacer()
                                                Button(specialist.latestModel(for: target) == nil ? "Set Up Training" : "Continue Training") {
                                                    store.specialistTarget = target
                                                    store.openSpecialistTrainer(continuing: specialist.latestModel(for: target) != nil)
                                                }.disabled(store.isBusy)
                                                    .help("Check this category's current image tags and paired maps, then open Trainer.")
                                            }
                                        }
                                        Divider()
                                        Picker("Surface map", selection: $store.specialistTarget) {
                                            ForEach(["height", "roughness", "normal"], id: \.self) { target in
                                                Text("\(WorkbenchStore.specialistMapTitle(target)) (\(specialist.targetInfo(target)?.materialCount ?? 0) at last check)").tag(target)
                                            }
                                        }
                                        let count = specialist.targetInfo(store.specialistTarget)?.materialCount ?? 0
                                        if count == 0 {
                                            Text("Open training setup to check the current folder for paired \(WorkbenchStore.specialistMapTitle(store.specialistTarget).lowercased()) maps tagged for \(specialist.classLabel).")
                                                .font(.callout).foregroundStyle(.secondary)
                                        }
                                        HStack {
                                            Button("Set Up Training", systemImage: "graduationcap") { store.openSpecialistTrainer() }
                                                .buttonStyle(.glassProminent).disabled(store.isBusy)
                                            if specialist.latestModel(for: store.specialistTarget) != nil {
                                                Button("Continue Saved Model") { store.openSpecialistTrainer(continuing: true) }
                                                    .disabled(store.isBusy)
                                            }
                                        }
                                        Text("Setup checks current files and tags, then opens Trainer with this category, map and model name at your chosen resolution. Crops retain their original pixel grid.")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                                }
                                GroupBox("5. Compare and approve") {
                                    VStack(alignment: .leading, spacing: 12) {
                                        if let model = specialist.latestModel(for: store.specialistTarget) {
                                            let current = model.membershipSha256 == specialist.targetInfo(store.specialistTarget)?.membershipSha256
                                            Label(!current ? "Source membership changed — train an updated model" : model.approved == true ? "Approved for automatic generation" : "Saved — awaiting your review",
                                                systemImage: !current ? "arrow.clockwise" : model.approved == true ? "checkmark.seal" : "eye")
                                            Text("\(model.trainingSize) × \(model.trainingSize) · \(URL(fileURLWithPath: model.checkpointPath).lastPathComponent)")
                                                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                                            HStack {
                                                Button("Compare with References") { store.compareSpecialistModel() }
                                                Button(model.approved == true ? "Remove Approval" : "Approve for Generation") {
                                                    store.approveSpecialistModel(approved: model.approved != true)
                                                }.disabled(!current && model.approved != true)
                                            }.disabled(store.isBusy)
                                        } else {
                                            Text("Training saves this category's checkpoint here. Inspect detail and artifacts before approving it.")
                                                .foregroundStyle(.secondary)
                                        }
                                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                                }
                                GroupBox("6. Generate materials") {
                                    VStack(alignment: .leading, spacing: 12) {
                                        Text("In Texture Studio, choose Automatic specialist models as the height source. The diffuse classifier selects approved category models; missing or uncertain categories use the chosen fallback.")
                                            .foregroundStyle(.secondary)
                                        NumericField("Minimum classifier confidence", value: $minimumConfidence, in: 0...1)
                                        if specialist.isCustom {
                                            Text("Subtype models can be trained and used directly from Saved Models. Automatic routing follows the classifier's categories; add a subtype to the classifier when you want it recognized automatically.")
                                                .font(.caption).foregroundStyle(.secondary)
                                        }
                                        Button("Next Category", systemImage: "arrow.right") { store.selectNextSpecialist() }
                                            .disabled(store.isBusy)
                                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                                }
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    if !registry.warnings.isEmpty {
                        DisclosureGroup("Source and classification notices (\(registry.warnings.count))") {
                            ForEach(Array(registry.warnings.enumerated()), id: \.offset) { _, warning in
                                Text(warning).font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                } else {
                    ContentUnavailableView("Organize your model categories", systemImage: "arrow.triangle.branch",
                        description: Text("Start with your source folders and image tags above, then refresh categories to train the available map models."))
                }
            }.padding(24).frame(maxWidth: 1200).frame(maxWidth: .infinity)
        }
        .task { await store.loadSpecialistRegistry() }
        .sheet(isPresented: $showAdd) { addSheet }
        .sheet(isPresented: $store.showClassifierDatasetSettings) { ClassifierDatasetSettingsSheet(store: store) }
    }

    private var workshopSources: some View {
        GroupBox("1. Organize your original folders and image tags") {
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.dataset == nil ? "Open a source folder to begin" : store.datasetName).font(.headline)
                    Text(store.dataset == nil ? "Images and maps stay at their original dimensions. Add, inspect and label them inside this app." : store.datasetInventorySummary)
                        .font(.callout).foregroundStyle(.secondary)
                    if !store.modelWorkshopSourcePath.isEmpty {
                        Text(store.modelWorkshopSourcePath)
                            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
                HStack {
                    Button("Open Source Folder", systemImage: "folder") { openSourceFolder() }
                        .disabled(store.modelWorkshopSourcePath.isEmpty)
                        .help("Open the current original source folder in Finder.")
                    Button("Choose Source Folder…") { store.chooseDataset() }.disabled(store.isBusy)
                        .help("Select a different folder of original images and material maps.")
                    Button("Add Images or Folders…", systemImage: "photo.badge.plus") { store.addMaterialFolder(workshop: true) }.disabled(store.isBusy)
                    Button("Manage Images & Tags", systemImage: "tag") { store.openModelWorkshopSources() }
                        .disabled(store.isBusy || store.dataset == nil)
                    Spacer(minLength: 0)
                }
                if !store.modelWorkshopSourceFolders.isEmpty {
                    DisclosureGroup("Current source folders (\(store.modelWorkshopSourceFolders.count))") {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 6) {
                                ForEach(store.modelWorkshopSourceFolders, id: \.self) { folder in
                                    Label(folder, systemImage: "folder").font(.caption).textSelection(.enabled)
                                }
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }.frame(maxHeight: 130)
                    }
                }
                if !store.modelWorkshopImageTags.isEmpty {
                    DisclosureGroup("Current classified image tags (\(store.modelWorkshopImageTags.count))") {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(store.modelWorkshopImageTags) { image in
                                    HStack(alignment: .top) {
                                        VStack(alignment: .leading, spacing: 3) {
                                            Text(image.familyId).font(.callout.bold())
                                            Text(URL(fileURLWithPath: image.inputPath).lastPathComponent).font(.caption).foregroundStyle(.secondary)
                                        }
                                        Spacer()
                                        Text(image.tags.joined(separator: " · ")).font(.callout).textSelection(.enabled)
                                    }
                                }
                            }
                        }.frame(maxHeight: 180)
                    }
                } else {
                    Text("Review material tags in Manage Images & Tags. Category refresh shows the classifier's current labels here.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !store.modelWorkshopTaggedOriginals.isEmpty {
                    DisclosureGroup("Saved original image tags (\(store.modelWorkshopTaggedOriginals.count))") {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(store.modelWorkshopTaggedOriginals) { image in
                                    HStack(alignment: .top) {
                                        VStack(alignment: .leading, spacing: 3) {
                                            Text(image.name).font(.callout.bold())
                                            Text(image.folder).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                                        }
                                        Spacer()
                                        VStack(alignment: .trailing, spacing: 3) {
                                            Text(image.classLabel ?? "").font(.callout)
                                            Text(originalTagSummary(image)).font(.caption).foregroundStyle(.secondary)
                                        }
                                    }
                                }
                            }
                        }.frame(maxHeight: 180)
                    }
                }
            }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func openSourceFolder() {
        let path = store.modelWorkshopSourcePath
        guard !path.isEmpty else { return }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
            store.error = "The source folder is no longer available at \(path). Choose its current location."
            return
        }
        guard NSWorkspace.shared.open(URL(fileURLWithPath: path, isDirectory: true)) else {
            store.error = "Finder could not open the source folder at \(path)."
            return
        }
    }

    private var workshopClassifier: some View {
        GroupBox("2. Teach the diffuse classifier") {
            VStack(alignment: .leading, spacing: 12) {
                Text("Label original diffuse images by material type, prepare the labeled examples, train the classifier, then choose the exported model here. Classifier examples and surface-map training crops have separate jobs.")
                    .foregroundStyle(.secondary)
                HStack {
                    Button("Review Image Tags", systemImage: "tag") { store.openModelWorkshopSources() }
                        .disabled(store.isBusy || store.dataset == nil)
                    Button("Prepare Classifier Examples", systemImage: "arrow.clockwise") { store.refreshClassifierDataset() }
                        .disabled(store.isBusy || store.dataset == nil && !store.classifierUseSourceFolder)
                    Button("Classifier Settings…", systemImage: "gearshape") { store.showClassifierDatasetSettings = true }
                        .disabled(store.isBusy)
                    Button("Choose Trained Classifier…", systemImage: "shippingbox") { store.chooseSpecialistPath("classifier") }
                        .disabled(store.isBusy)
                }
                if let result = store.classifierDatasetResult {
                    Text(result.summary).font(.caption).foregroundStyle(.secondary)
                }
                Text(store.specialistClassifierPath.isEmpty ? "Using the bundled diffuse classifier until you choose a trained export." : "Classifier: \(store.specialistClassifierPath)")
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func mapTrainingSummary(_ specialist: WorkbenchSpecialistRegistry.Specialist, target: String) -> String {
        let count = specialist.targetInfo(target)?.materialCount ?? 0
        guard count > 0 else { return "Last check found no paired maps · Setup checks current tags" }
        if let model = specialist.latestModel(for: target) {
            let current = model.membershipSha256 == specialist.targetInfo(target)?.membershipSha256
            return "Last check: \(count) paired sets · " + (!current ? "Sources changed" : model.approved == true ? "Approved model" : "Saved model needs review")
        }
        return "Last check: \(count) paired sets · Setup checks current tags"
    }

    private func originalTagSummary(_ image: WorkbenchSourceInventoryItem) -> String {
        let role: String
        switch image.role {
        case "input": role = "Diffuse"
        case "height": role = "Displacement"
        case "roughness": role = "Roughness"
        case "normal": role = "Normal"
        case "ancillary": role = "Additional map"
        default: role = "Unassigned image"
        }
        return role + (image.sourceVerification == "verified" ? " · Verified original" : "")
    }

    private func pathField(_ title: String, value: Binding<String>, kind: String, prompt: String = "") -> some View {
        HStack {
            Text(title).frame(width: 135, alignment: .leading)
            TextField(prompt, text: value).textFieldStyle(.roundedBorder)
            Button("Choose…") { store.chooseSpecialistPath(kind) }.disabled(store.isBusy)
        }
    }

    private var addSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Add a model category").font(.title2.bold())
            TextField("Name, e.g. Wood Planks", text: $newName)
            Picker("Classifier category", selection: $newClass) {
                ForEach(store.specialistRegistry?.classifierLabels ?? [], id: \.self) { Text($0).tag($0) }
            }
            TextField("Subtype or filename text, e.g. planks (optional)", text: $newSubtype)
            Text("Keep descriptive filenames. The subtype selects matching materials within the chosen category. Leave it blank to train the whole category.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Cancel") { showAdd = false }.keyboardShortcut(.cancelAction)
                Button("Add Category") {
                    store.addSpecialist(name: newName, classLabel: newClass, subtype: newSubtype)
                    showAdd = false
                }.keyboardShortcut(.defaultAction).disabled(newName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }.textFieldStyle(.roundedBorder).padding(24).frame(width: 530)
    }
}
