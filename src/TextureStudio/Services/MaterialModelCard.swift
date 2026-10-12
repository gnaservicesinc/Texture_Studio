import Foundation

/// Describes recorded settings without inferring learned quality or physical scale.
enum MaterialModelCard {
    static let pipelineName = "MatterLoom"

    static func targetTitle(_ target: String) -> String {
        ["height": "Surface height / displacement", "roughness": "Surface roughness", "normal": "Surface normals"][target]
            ?? target.capitalized
    }

    static func readme(configuration: [String: Any], familyDescription: String) -> String {
        let settings = configuration["training_configuration"] as? [String: Any] ?? [:]
        let recordedName = clean(configuration["model_name"] as? String ?? "")
        let name = recordedName.isEmpty ? "Texture Studio material model" : recordedName
        let pipeline = clean(configuration["texture_pipeline"] as? String ?? pipelineName)
        let target = targetTitle(configuration["target"] as? String ?? "material")
        let subjects = configuration["training_subjects"] as? [String] ?? []
        let dataset = configuration["training_dataset_name"] as? String
        let size = configuration["training_size"] ?? settings["training_size"]
        let parameters = configuration["parameter_count"] ?? settings["trainable_parameter_count"]
        let input = configuration["input_kind"] as? String == "depthpro"
            ? "Cached normalized inverse DepthPro depth"
            : "Diffuse / base-color image"
        var rows = [("Target", target), ("Input", input)]
        if let dataset, !dataset.isEmpty { rows.append(("Dataset", clean(dataset))) }
        if !subjects.isEmpty { rows.append(("Configured training subjects", subjects.map(clean).joined(separator: ", "))) }
        if let size { rows.append(("Native training grid", "\(size) × \(size) pixels")) }
        if let parameters { rows.append(("Trainable parameters", String(describing: parameters))) }
        if let family = configuration["model_family"] { rows.append(("Model family", clean(String(describing: family)))) }
        if let width = configuration["network_width"] { rows.append(("Network width", String(describing: width))) }
        if let step = configuration["step"] { rows.append(("Saved training step", String(describing: step))) }
        for (key, title) in [("optimizer", "Optimizer"), ("learning_rate", "Learning rate"),
                             ("optimizer_beta1", "Optimizer beta 1"), ("optimizer_beta2", "Optimizer beta 2"),
                             ("weight_decay", "Weight decay"), ("max_gradient_norm", "Gradient norm limit"),
                             ("learning_rate_schedule", "Learning rate schedule"), ("warmup_updates", "Warmup steps"),
                             ("gradient_accumulation_steps", "Gradient accumulation"), ("randomize_crops", "Crop randomization"),
                             ("random_flip", "Random flips"), ("random_spin_centercut", "Center spins"),
                             ("value_loss_weight", "Value loss weight"), ("detail_loss_weight", "Detail loss weight"),
                             ("source_tier_count", "Fetched source tiers"),
                             ("training_map_count", "Training maps"), ("validation_map_count", "Validation maps")]
            where settings[key] != nil {
            rows.append((title, clean(String(describing: settings[key]!))))
        }
        if let transfer = configuration["input_transfer"] { rows.append(("Input transfer", clean(String(describing: transfer)))) }
        let material = !subjects.isEmpty && subjects.count <= 8
            ? " for " + subjects.map(clean).joined(separator: ", ")
            : dataset.map { " for the “\(clean($0))” dataset" } ?? ""
        let summary = "\(name) is the \(target.lowercased()) specialist\(material) in the \(pipeline) texture pipeline. It takes \(input.lowercased()) and predicts the selected material map."
        let yamlName = String(data: try! JSONEncoder().encode(name), encoding: .utf8)!
        let table = rows.map { "| \($0.0) | \($0.1) |" }.joined(separator: "\n")
        return """
        ---
        license: gpl-3.0
        model_name: \(yamlName)
        tags:
        - texture-studio-material
        - matterloom
        - material-maps
        - safetensors
        ---

        # \(name)

        \(summary)

        | Setting | Recorded value |
        | --- | --- |
        \(table)

        Every training map uses its declared native grid without padding or resizing. Texture Studio evaluates this numeric checkpoint with Apple MPSGraph in Float32. \(familyDescription)

        The quality goal is a highly detailed, visually pleasing texture that passes practical sanity checks. Review detail, useful relief, map consistency and repeating seams on held-out materials before approving this specialist. Training settings and parameter count describe this checkpoint; they do not establish that it has met those quality goals.

        Source images and an interpreter are not included.
        """
    }

    private static func clean(_ value: String) -> String {
        let line = value.components(separatedBy: .newlines).joined(separator: " ")
            .components(separatedBy: .controlCharacters).joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return line.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "[", with: "\\[")
            .replacingOccurrences(of: "]", with: "\\]")
    }
}
