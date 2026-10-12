import SwiftUI

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
                GroupBox("1. Prepare the specialist library") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Classify original diffuse maps and create one paired training dataset per category. Refresh keeps curated examples, dataset reviews and saved models.")
                            .foregroundStyle(.secondary)
                        DisclosureGroup("Classifier and source settings") {
                            VStack(alignment: .leading, spacing: 10) {
                                pathField("Original dataset", value: $store.specialistSourceDatasetPath, kind: "source")
                                pathField("Classifier export", value: $store.specialistClassifierPath, kind: "classifier", prompt: "Bundled Diffuse Texture Type Classifier IFPv2")
                                pathField("Specialist workspace", value: $store.specialistWorkspacePath, kind: "workspace")
                                Text("Additional paired maps are discovered in the classifier source folder: \(store.classifierSourceFolderPath)")
                                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                                Text("Classifier-only diffuse examples remain in \(store.classifierDatasetPath). Add matching surface maps to the source dataset when you want to use them for material training.")
                                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                            }.padding(.top, 8)
                        }
                        HStack {
                            Button("Create / Refresh Specialists", systemImage: "arrow.triangle.branch") { store.refreshSpecialists() }
                                .buttonStyle(.glassProminent).disabled(store.isBusy)
                            if let registry = store.specialistRegistry {
                                Text("\(registry.specialists.count) specialists · \(registry.familyCount) paired families · \(registry.classifierOnlyCount) classifier-only examples")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                }
                if let registry = store.specialistRegistry {
                    HStack(alignment: .top, spacing: 20) {
                        GroupBox("2. Choose one specialist") {
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
                                GroupBox("3. Train \(specialist.name)") {
                                    VStack(alignment: .leading, spacing: 12) {
                                        if let subtype = specialist.subtype {
                                            Text("\(specialist.classLabel) · subtype: \(subtype)").foregroundStyle(.secondary)
                                        }
                                        Picker("Surface map", selection: $store.specialistTarget) {
                                            ForEach(["height", "roughness", "normal"], id: \.self) { target in
                                                Text("\(WorkbenchStore.specialistMapTitle(target)) (\(specialist.targetInfo(target)?.materialCount ?? 0) paired sets)").tag(target)
                                            }
                                        }
                                        let count = specialist.targetInfo(store.specialistTarget)?.materialCount ?? 0
                                        if count == 0 {
                                            Text("Add paired \(WorkbenchStore.specialistMapTitle(store.specialistTarget).lowercased()) maps, then refresh this library.")
                                                .font(.callout).foregroundStyle(.secondary)
                                        }
                                        HStack {
                                            Button("Set Up Training", systemImage: "graduationcap") { store.openSpecialistTrainer() }
                                                .buttonStyle(.glassProminent).disabled(store.isBusy || count == 0)
                                            if specialist.latestModel(for: store.specialistTarget) != nil {
                                                Button("Continue Saved Model") { store.openSpecialistTrainer(continuing: true) }
                                                    .disabled(store.isBusy || count == 0)
                                            }
                                        }
                                        Text("The Trainer opens with this category, map and model name. Set resolution and training settings there. Crops retain their original pixel grid.")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                                }
                                GroupBox("4. Compare and approve") {
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
                                            Text("Training saves this specialist's checkpoint here. Inspect detail and artifacts before approving it.")
                                                .foregroundStyle(.secondary)
                                        }
                                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                                }
                                GroupBox("5. Generate automatically or continue") {
                                    VStack(alignment: .leading, spacing: 12) {
                                        Text("In Texture Studio, choose Automatic specialist models as the height source. The diffuse classifier selects approved category models; missing or uncertain categories use the chosen fallback.")
                                            .foregroundStyle(.secondary)
                                        NumericField("Minimum classifier confidence", value: $minimumConfidence, in: 0...1)
                                        if specialist.isCustom {
                                            Text("Subtype specialists can be trained and used directly from Saved Models. Automatic routing follows the classifier's categories; add a subtype to the classifier when you want it recognized automatically.")
                                                .font(.caption).foregroundStyle(.secondary)
                                        }
                                        Button("Next Specialist", systemImage: "arrow.right") { store.selectNextSpecialist() }
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
                    ContentUnavailableView("Create your specialist library", systemImage: "arrow.triangle.branch",
                        description: Text("The bundled classifier is ready. Refresh to organize paired materials and set up each model."))
                }
            }.padding(24).frame(maxWidth: 1200).frame(maxWidth: .infinity)
        }
        .task { store.loadSpecialistRegistry() }
        .sheet(isPresented: $showAdd) { addSheet }
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
            Text("Add a specialist").font(.title2.bold())
            TextField("Name, e.g. Wood Planks", text: $newName)
            Picker("Classifier category", selection: $newClass) {
                ForEach(store.specialistRegistry?.classifierLabels ?? [], id: \.self) { Text($0).tag($0) }
            }
            TextField("Subtype or filename text, e.g. planks (optional)", text: $newSubtype)
            Text("Keep descriptive filenames. The subtype selects matching materials within the chosen category. Leave it blank to create a specialist for the whole category.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Cancel") { showAdd = false }.keyboardShortcut(.cancelAction)
                Button("Add Specialist") {
                    store.addSpecialist(name: newName, classLabel: newClass, subtype: newSubtype)
                    showAdd = false
                }.keyboardShortcut(.defaultAction).disabled(newName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }.textFieldStyle(.roundedBorder).padding(24).frame(width: 530)
    }
}
