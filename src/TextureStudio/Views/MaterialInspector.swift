import SwiftUI

struct MaterialInspector: View {
    @Bindable var workspace: TextureWorkspace
    let models: ModelManager

    var body: some View {
        Form {
            Section("Material workflow") {
                Text("Adjust the photo and surface settings, then choose Generate Material. Open Full Quality inspects the finished map; Export Material saves all four maps for Blender.")
                    .font(.caption).foregroundStyle(.secondary)
                LabeledContent("Height source") {
                    Text(workspace.activeHeightSourceLabel).multilineTextAlignment(.trailing).textSelection(.enabled)
                }
            }
            Section("Photo") {
                if let source = workspace.source {
                    Text(source.url.lastPathComponent).font(.headline).lineLimit(2)
                    Text("\(source.pixelWidth) × \(source.pixelHeight) pixels").foregroundStyle(.secondary)
                    CameraInfoView(source: source)
                    if source.hdrImage != nil {
                        Toggle("Use HDR gain map detail", isOn: $workspace.settings.useHDRGainMap)
                        Text("HDR detail is balanced in float before the 8-bit diffuse export.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if !source.supportingViews.isEmpty {
                        Toggle("Use spatial companion views", isOn: $workspace.settings.useSupportingViews)
                        Text("\(source.supportingViews.count) supporting view(s). Only matching surface regions contribute.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                } else {
                    Text("Import a photo to begin.").foregroundStyle(.secondary)
                }
            }
            Section("Straighten & crop") {
                DoubleControl(title: "Tilt X", value: $workspace.settings.rotationX, range: -70...70, suffix: "°")
                    .help("Straighten a surface that tilts away vertically. Empty edges are cropped automatically.")
                DoubleControl(title: "Tilt Y", value: $workspace.settings.rotationY, range: -70...70, suffix: "°")
                    .help("Straighten a surface that tilts away horizontally. Empty edges are cropped automatically.")
                DoubleControl(title: "Rotate Z", value: $workspace.settings.rotationZ, range: -180...180, suffix: "°", enforcesSliderRange: false)
                    .help("Rotate within the photo to level horizontal or vertical surface features.")
                DoubleControl(title: "Lens correction", value: $workspace.settings.lensDistortion, range: -0.15...0.15)
                    .help("Correct barrel or pincushion curvature. Leave at zero if straight features already look straight.")
                LabeledContent("Focal length (px)") {
                    OptionalNumericTextField(title: "Focal length in pixels", value: $workspace.settings.focalLengthPixels, greaterThan: 0)
                        .frame(width: 120)
                }
                Text("Leave focal length blank to use the photo's camera information.")
                    .font(.caption).foregroundStyle(.secondary)
                DoubleControl(title: "Crop scale", value: $workspace.settings.cropScale, range: 0.2...1, entryRange: 0...1, greaterThan: 0)
                DoubleControl(title: "Crop horizontal", value: $workspace.settings.cropOffsetX, range: -1...1)
                DoubleControl(title: "Crop vertical", value: $workspace.settings.cropOffsetY, range: -1...1)
                Text("The crop stays inside the transformed photo. Smaller crops let you choose a tighter surface area.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Reset Transform") {
                    workspace.settings.rotationX = 0; workspace.settings.rotationY = 0; workspace.settings.rotationZ = 0
                    workspace.settings.cropScale = 1; workspace.settings.cropOffsetX = 0; workspace.settings.cropOffsetY = 0
                    workspace.settings.lensDistortion = 0; workspace.settings.focalLengthPixels = nil
                }
                .help("Reset tilt, rotation, lens correction and crop. Focal length returns to the photo's camera information.")
            }
            Section("Balance the photo") {
                FloatControl(title: "Lighting balance", value: $workspace.settings.lightingStrength, range: 0...1)
                FloatControl(title: "Lighting scale", value: $workspace.settings.lightingRadius, range: 0...1)
                Text("Registered companion views can reduce capture noise while retaining source detail. The photo is never passed through a smoothing denoiser.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Broad illumination is reduced while retaining photo detail. Clipped highlights and hidden shadow detail need review.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Height / displacement source") {
                Picker("Source", selection: $workspace.depthChoice) {
                    ForEach(DepthChoice.studioChoices) { choice in
                        Text(choice.title).tag(choice)
                    }
                }
                if workspace.depthChoice == .attached {
                    Button(workspace.depthURL == nil ? "Attach Depth Map…" : "Replace Depth Map…") { workspace.chooseDepth() }
                    if let url = workspace.depthURL { Text(url.lastPathComponent).font(.caption).foregroundStyle(.secondary) }
                    Toggle("Higher values mean raised surface", isOn: $workspace.settings.attachedMapIsHeight)
                    Text("Use a map registered to this photo. Turn this off for camera-distance maps, where larger values mean farther away.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if workspace.depthChoice == .model {
                    Picker("Model", selection: $workspace.modelID) {
                        ForEach(models.catalog) { descriptor in Text(descriptor.name).tag(descriptor.id) }
                    }
                    Text(models.status(for: workspace.modelID).message).font(.caption).foregroundStyle(.secondary)
                    Button("Manage / Locate Model…") { workspace.showModels = true }
                    if workspace.modelID == LocalModelDescriptor.customDepthID {
                        Toggle("Higher values mean nearer", isOn: $workspace.customInverseDepth)
                    }
                    Text("Camera-depth models estimate scene structure. Inspect their fine surface detail before using the result for material displacement.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if workspace.depthChoice == .materialCheckpoint {
                    Text(workspace.selectedMaterialCheckpoint?.modelSummary ?? "Choose a trained material checkpoint in Model Training → Saved Models.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("The model receives the prepared diffuse map with this same crop and lighting balance.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if workspace.depthChoice == .photoDetail {
                    Text("No inferred relief: displacement stays neutral and normals stay flat unless you explicitly add brightness relief below.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("Surface maps") {
                FloatControl(title: "Relief contrast", value: $workspace.settings.heightStrength, range: 0...1)
                FloatControl(title: "Remove surface slope", value: $workspace.settings.surfacePlaneRemoval, range: 0...1)
                    .disabled(workspace.depthChoice == .materialCheckpoint)
                FloatControl(title: "Depth artifact cleanup", value: $workspace.settings.depthCleanup, range: 0...1)
                    .disabled(workspace.depthChoice == .materialCheckpoint)
                Toggle("Invert height", isOn: $workspace.settings.heightInvert)
                Toggle("Protect near-flat surfaces", isOn: $workspace.settings.adaptiveRelief)
                    .disabled(workspace.depthChoice == .materialCheckpoint)
                Text("Reduce amplification when relative depth has little surface contrast. This artistic protection can be disabled for deliberate exaggeration.")
                    .font(.caption).foregroundStyle(.secondary)
                FloatControl(title: "Artistic brightness relief", value: $workspace.settings.heightDetail, range: 0...1)
                Text("Brightness relief is off by default: printed color and shadows can otherwise become false bumps. Normals use OpenGL +Y and the final cleaned height.")
                    .font(.caption).foregroundStyle(.secondary)
                FloatControl(title: "Base roughness", value: $workspace.settings.roughnessBase, range: 0...1)
                FloatControl(title: "Roughness detail", value: $workspace.settings.roughnessDetail, range: 0...1)
                NumericField("Surface width", value: $workspace.settings.materialWidthMeters, greaterThan: 0, unit: "m")
                .help("The real or intended width of the material tile in Blender. Used with relief scale to calculate normal strength.")
                NumericField("Relief scale", value: $workspace.settings.displacementScaleMeters, atLeast: 0, unit: "m")
                .help("The displacement amount for Blender. Adjust for useful visual relief; inferred height is not a calibrated physical measurement.")
                Text("Depth cleanup targets isolated artifacts while retaining coherent edges. Roughness remains an editable estimate.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Export material") {
                Picker("Map size", selection: $workspace.settings.outputSize) {
                    ForEach(Array(Set(TextureSettings.outputSizes + [workspace.settings.outputSize])).sorted(), id: \.self) { size in
                        Text("\(size) × \(size)").tag(size)
                    }
                }
                Picker("EXR storage", selection: $workspace.settings.exrPrecision) {
                    Text("16-bit float").tag(EXRPrecision.float16)
                    Text("32-bit float").tag(EXRPrecision.float32)
                }
                .help("16-bit float uses less disk space; 32-bit float preserves the pipeline's numeric precision. Both store linear map data.")
                Text("Diffuse: 16-bit sRGB PNG. Roughness, normal and displacement: linear EXR.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Camera-depth model resolution is set separately. Larger output maps do not add detail absent from the height prediction.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Export Material…", systemImage: "square.and.arrow.up") { workspace.chooseExport(models: models) }
                    .buttonStyle(.borderedProminent)
                    .disabled(workspace.source == nil)
                    .help("Save all four maps and a Blender setup script in one folder. Any pending changes are generated automatically. ⌘E")
                Text("Choose a new material folder. Export includes every map, regardless of which one you are viewing.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !workspace.warnings.isEmpty {
                Section("Review before use") {
                    ForEach(workspace.warnings, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
                }
            }
        }
        .formStyle(.grouped)
        .disabled(workspace.isBusy)
    }
}

struct FloatControl: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                NumericTextField(title: title, value: $value, in: range).frame(width: 120)
            }
            Slider(value: $value, in: range).labelsHidden().accessibilityLabel(title)
        }
    }
}

struct DoubleControl: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var suffix = ""
    var enforcesSliderRange = true
    var entryRange: ClosedRange<Double>? = nil
    var greaterThan: Double? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                NumericTextField(title: title, value: $value, in: entryRange ?? (enforcesSliderRange ? range : nil),
                                 greaterThan: greaterThan).frame(width: 120)
                if !suffix.isEmpty { Text(suffix).foregroundStyle(.secondary) }
            }
            // The thumb uses the suggested slider range without replacing a
            // valid number typed outside it. Only a slider action writes back.
            Slider(value: Binding(get: { min(range.upperBound, max(range.lowerBound, value)) }, set: { value = $0 }), in: range)
                .labelsHidden().accessibilityLabel(title)
        }
    }
}

struct CameraInfoView: View {
    let source: TextureSource
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(source.camera.summary).textSelection(.enabled)
            if let profile = source.camera.colorProfile { Text(profile) }
            if let iso = source.camera.iso { Text("ISO \(Int(iso))") }
            if let aperture = source.camera.aperture { Text("f/\(aperture, specifier: "%.1f")") }
            if !source.camera.auxiliaryTypes.isEmpty {
                Text("Available: \(source.camera.auxiliaryTypes.joined(separator: ", "))")
            }
        }
        .font(.caption).foregroundStyle(.secondary)
    }
}
