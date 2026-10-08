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
                DoubleControl(title: "Tilt X", value: $workspace.settings.rotationX, range: -45...45, suffix: "°")
                    .help("Straighten a surface that tilts away vertically. Empty edges are cropped automatically.")
                DoubleControl(title: "Tilt Y", value: $workspace.settings.rotationY, range: -45...45, suffix: "°")
                    .help("Straighten a surface that tilts away horizontally. Empty edges are cropped automatically.")
                DoubleControl(title: "Rotate Z", value: $workspace.settings.rotationZ, range: -45...45, suffix: "°")
                    .help("Rotate within the photo to level horizontal or vertical surface features.")
                DoubleControl(title: "Lens correction", value: $workspace.settings.lensDistortion, range: -0.15...0.15)
                    .help("Correct barrel or pincushion curvature. Leave at zero if straight features already look straight.")
                LabeledContent("Focal length (px)") {
                    TextField("Auto", value: $workspace.settings.focalLengthPixels, format: .number)
                        .frame(width: 90)
                }
                DoubleControl(title: "Crop scale", value: $workspace.settings.cropScale, range: 0.2...1)
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
                FloatControl(title: "Lighting scale", value: $workspace.settings.lightingRadius, range: 0.01...0.5)
                FloatControl(title: "Noise reduction", value: $workspace.settings.noiseReduction, range: 0...0.1)
                    .help("Use the lowest value that removes visible sensor noise. Inspect at full quality to preserve small surface detail.")
                Text("Broad illumination is reduced while retaining photo detail. Clipped highlights and hidden shadow detail need review.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Height / displacement source") {
                Picker("Source", selection: $workspace.depthChoice) {
                    ForEach(DepthChoice.allCases) { choice in
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
                    if workspace.modelID == LocalModelDescriptor.da3GiantID {
                        Picker("Inference edge", selection: $workspace.settings.modelProcessResolution) {
                            Text("1036 px").tag(1036)
                            Text("1540 px").tag(1540)
                            Text("2044 px").tag(2044)
                        }
                        Text("DA3-GIANT-1.1 runs locally through PyTorch/MPS. Larger inference uses more memory and time; final map size is independent. Weights are non-commercial.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Model depth is converted to relative surface height. Embedded portrait depth never supplies displacement detail.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if workspace.depthChoice == .photoDetail {
                    Text("No inferred relief: displacement stays neutral and normals stay flat unless you explicitly add brightness relief below.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if workspace.depthChoice == .materialCheckpoint {
                    if let checkpoint = workspace.selectedMaterialCheckpoint {
                        Text(checkpoint.title).font(.headline).textSelection(.enabled)
                        Text(checkpoint.modelSummary ?? "DINOv2 Base features + trained material-height head")
                            .font(.caption).foregroundStyle(.secondary)
                        Text("SHA256 \(checkpoint.sha256.prefix(12)) · native height, no range normalization").font(.caption).foregroundStyle(.secondary)
                    } else { Text("Choose a height checkpoint in Material Trainer.").foregroundStyle(.secondary) }
                    Button("Choose Material Checkpoint…") { MaterialToolLauncher.open(.train) }
                    Text("Predicts surface height using your trained material head and DINOv2 features. DA3 is a separate source. Native inference supports 1024 or 2048 output.").font(.caption).foregroundStyle(.secondary)
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
                LabeledContent("Surface width (m)") {
                    TextField("Meters", value: $workspace.settings.materialWidthMeters, format: .number)
                        .frame(width: 90)
                }
                .help("The real or intended width of the material tile in Blender. Used with relief scale to calculate normal strength.")
                LabeledContent("Relief scale (m)") {
                    TextField("Meters", value: $workspace.settings.displacementScaleMeters, format: .number)
                        .frame(width: 90)
                }
                .help("The displacement amount for Blender. Adjust for useful visual relief; inferred height is not a calibrated physical measurement.")
                Text("Depth cleanup targets isolated artifacts while retaining coherent edges. Roughness remains an editable estimate.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Export material") {
                Picker("Map size", selection: $workspace.settings.outputSize) {
                    Text("1024 × 1024").tag(1024)
                    Text("2048 × 2048").tag(2048)
                    Text("4098 × 4098").tag(4098)
                    Text("8K · 8192 × 8192").tag(8192)
                }
                Picker("EXR storage", selection: $workspace.settings.exrPrecision) {
                    Text("16-bit float").tag(EXRPrecision.float16)
                    Text("32-bit float").tag(EXRPrecision.float32)
                }
                .help("16-bit float uses less disk space; 32-bit float preserves the pipeline's numeric precision. Both store linear map data.")
                Text("Diffuse: 8-bit sRGB PNG. Roughness, normal and displacement: linear EXR.")
                    .font(.caption).foregroundStyle(.secondary)
                Text(workspace.depthChoice == .materialCheckpoint
                    ? "This material model supports native 1024 or 2048 output. Choose either size for this checkpoint."
                    : "Camera-depth model resolution is set separately. Larger output maps do not add detail absent from the height prediction.")
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
            HStack { Text(title); Spacer(); Text(value, format: .number.precision(.fractionLength(2))).monospacedDigit().foregroundStyle(.secondary) }
            Slider(value: $value, in: range).labelsHidden().accessibilityLabel(title)
        }
    }
}

struct DoubleControl: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var suffix = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                TextField(title, value: $value, format: .number.precision(.fractionLength(2)))
                    .multilineTextAlignment(.trailing).frame(width: 68)
                    .labelsHidden()
                    .accessibilityLabel(title)
                if !suffix.isEmpty { Text(suffix).foregroundStyle(.secondary) }
            }
            Slider(value: $value, in: range).labelsHidden().accessibilityLabel(title)
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
