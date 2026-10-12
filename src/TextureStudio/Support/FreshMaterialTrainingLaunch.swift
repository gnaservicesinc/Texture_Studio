import Foundation

/// An explicit launch request chooses a new RGB model without reconnecting
/// saved checkpoint weights. It prepares settings only; training starts in UI.
struct FreshMaterialTrainingLaunch {
    let datasetURL: URL
    let trainingSize: Int?
    let learningRate: Double?
    let updatesPerMap: Int?

    static func parse(arguments: [String]) throws -> Self? {
        guard arguments.contains("--fresh-training") else { return nil }
        guard arguments.filter({ $0 == "--fresh-training" }).count == 1 else {
            throw StudioError("Pass --fresh-training only once.")
        }
        func value(for flag: String, required: Bool = false) throws -> String? {
            let indices = arguments.indices.filter { arguments[$0] == flag }
            guard indices.count <= 1 else { throw StudioError("Pass \(flag) only once.") }
            guard let index = indices.first else {
                if required { throw StudioError("Fresh training requires --dataset followed by a saved dataset path.") }
                return nil
            }
            guard arguments.indices.contains(index + 1), !arguments[index + 1].hasPrefix("--"),
                  !arguments[index + 1].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw StudioError("Pass a value after \(flag).")
            }
            return arguments[index + 1]
        }
        let datasetPath = try value(for: "--dataset", required: true)!
        let sizeText = try value(for: "--training-size")
        let rateText = try value(for: "--learning-rate")
        let updatesText = try value(for: "--updates-per-map")
        var size: Int?
        if let sizeText {
            guard let parsed = Int(sizeText), [256, 512, 1024, 2048].contains(parsed) else {
                throw StudioError("--training-size must be 256, 512, 1024, or 2048.")
            }
            size = parsed
        }
        var rate: Double?
        if let rateText {
            guard let parsed = Double(rateText), parsed.isFinite, Float(parsed).isFinite, Float(parsed) > 0 else {
                throw StudioError("--learning-rate must be a positive finite Float32 value.")
            }
            rate = parsed
        }
        var updates: Int?
        if let updatesText {
            guard let parsed = Int(updatesText), parsed > 0 else {
                throw StudioError("--updates-per-map must be a positive integer.")
            }
            updates = parsed
        }
        let url = URL(fileURLWithPath: datasetPath).standardizedFileURL
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) else {
            throw StudioError("The fresh training dataset does not exist at \(url.path).")
        }
        if !directory.boolValue {
            let resolved = url.resolvingSymlinksInPath().standardizedFileURL
            guard resolved.lastPathComponent == "dataset.json",
                  (try? resolved.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else {
                throw StudioError("Fresh training requires a saved dataset. Choose its dataset.json or dataset folder.")
            }
            return Self(datasetURL: resolved, trainingSize: size, learningRate: rate, updatesPerMap: updates)
        }
        if directory.boolValue,
           !FileManager.default.fileExists(atPath: url.appendingPathComponent("dataset.json").path),
           !(url.lastPathComponent == "sources" && FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("dataset.json").path)) {
            throw StudioError("Fresh training requires a saved dataset. Choose its dataset.json or dataset folder.")
        }
        return Self(datasetURL: url, trainingSize: size, learningRate: rate, updatesPerMap: updates)
    }
}
