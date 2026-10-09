import CryptoKit
import Darwin
import Foundation

final class NativeMaterialTrainingControl: @unchecked Sendable {
    private let lock = NSLock()
    private var aborted = false, finalSave = false, checkpoint = false
    func stop() { lock.withLock { aborted = true } }
    func stopAndSave() { lock.withLock { finalSave = true } }
    func saveCheckpoint() { lock.withLock { checkpoint = true } }
    var shouldStopAndSave: Bool { lock.withLock { finalSave } }
    func check() throws { try Task.checkCancellation(); if lock.withLock({ aborted }) { throw CancellationError() } }
    func consumeCheckpoint() -> Bool { lock.withLock { let requested = checkpoint; checkpoint = false; return requested } }
}

/// Native whole-grid Float32 material training. MPSGraph differentiates the
/// recorded LoRA factors and performs clipping/Adam; Swift owns dataset locks,
/// iteration, validation and bit-preserving safetensors checkpoint publication.
enum NativeMaterialTrainer {
    typealias Event = @Sendable (String) -> Void
    static func run(arguments: [String], onEvent: @escaping Event = { _ in }, control: NativeMaterialTrainingControl = .init()) async throws -> String {
        try Task.checkCancellation()
        let job = Task.detached(priority: .userInitiated) {
            let options = try Options(arguments)
            if options.command == "infer" { return try infer(options, control: control) }
            return try train(options, onEvent: onEvent, control: control)
        }
        return try await withTaskCancellationHandler { try await job.value } onCancel: { control.stop(); job.cancel() }
    }
    struct Options: Sendable {
        let command: String, target: String, scope: String
        let dataset: URL?, output: URL, image: URL?, checkpoint: URL?, expectedSHA256: String?, base: URL
        let size: Int, rank: Int, updatesPerMap: Int, validationEvery: Int, checkpointEvery: Int
        let alpha: Float, learningRate: Float, maxMinutes: Double
        let seed: UInt64, materials: [String], inputEncoding: String, baseline: Bool, developerMode: Bool
        init(_ arguments: [String]) throws {
            guard let command = arguments.first, ["train", "refine", "infer"].contains(command) else { throw StudioError("Unsupported native material operation.") }
            self.command = command
            func value(_ flag: String) -> String? { guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }; return arguments[index + 1] }
            func url(_ flag: String) -> URL? { value(flag).map { URL(fileURLWithPath: $0).standardizedFileURL } }
            guard let output = url("--output") else { throw StudioError("Choose a new material operation output folder.") }
            self.output = output; dataset = url("--dataset"); image = url("--image"); checkpoint = url("--checkpoint")
            expectedSHA256 = value("--expected-sha256")
            var recorded: [String: Any] = [:]
            if let checkpoint {
                let snapshot = try NativeSafetensors(contentsOf: NativeMaterialModel.resolvedCheckpoint(checkpoint), expectedSHA256: expectedSHA256)
                if let text = snapshot.metadata["configuration"] { recorded = (try JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:] }
            }
            target = value("--target") ?? recorded["target"] as? String ?? "height"
            scope = value("--scope") ?? recorded["scope"] as? String ?? "final-map"
            guard ["height", "roughness", "normal"].contains(target), ["final-map", "map-decoder"].contains(scope) else { throw StudioError("Choose a supported material target and layer scope.") }
            let directory = url("--model-directory") ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!.appendingPathComponent("Texture Studio/Material Models/pbrnxt-base")
            base = url("--base-checkpoint") ?? NativeMaterialPackage.baseURL(directory, configuration: recorded)
            size = Int(value("--size") ?? "1024") ?? 0
            rank = Int(value("--lora-rank") ?? "8") ?? 0
            alpha = Float(value("--lora-alpha") ?? "8") ?? .nan
            learningRate = Float(value("--learning-rate") ?? "0.00001") ?? .nan
            updatesPerMap = Int(value("--updates-per-map") ?? "100") ?? 0
            maxMinutes = Double(value("--max-minutes") ?? "30") ?? .nan
            validationEvery = Int(value("--validation-every") ?? "20") ?? 0
            checkpointEvery = Int(value("--checkpoint-every") ?? "0") ?? -1
            seed = UInt64(value("--seed") ?? "17") ?? 17
            materials = arguments.indices.filter { arguments[$0] == "--material" && arguments.indices.contains($0 + 1) }.map { arguments[$0 + 1] }
            inputEncoding = value("--input-encoding") ?? "srgb"
            baseline = arguments.contains("--baseline"); developerMode = arguments.contains("--developer-mode")
            guard size >= 256, size <= 8192, size % 64 == 0, rank > 0, alpha.isFinite, alpha > 0,
                  learningRate.isFinite, learningRate > 0, updatesPerMap > 0, maxMinutes.isFinite, maxMinutes > 0,
                  validationEvery > 0, checkpointEvery >= 0 else { throw StudioError("Material training sizes, rates and update counts are invalid.") }
            guard command == "infer" ? image != nil : dataset != nil else { throw StudioError("The native material operation needs its input image or dataset.") }
        }
    }
    private static func infer(_ options: Options, control: NativeMaterialTrainingControl) throws -> String {
        try control.check()
        let imageURL = options.image!, snapshot = try Data(contentsOf: imageURL)
        let photo = try NativePNG.decode(snapshot)
        let rgb = try photo.modelFloatSamples(role: "input", encoding: options.inputEncoding)
        let model = try NativeMaterialModel.load(checkpointURL: options.baseline ? nil : options.checkpoint,
            expectedSHA256: options.expectedSHA256, baseURL: options.base, target: options.target, scope: options.scope)
        let prediction = try model.predict(rgb: rgb, width: photo.header.width, height: photo.header.height, target: options.target)
        try control.check()
        guard try checksum(Data(contentsOf: imageURL)) == checksum(snapshot) else { throw StudioError("The prepared diffuse changed during inference.") }
        guard !FileManager.default.fileExists(atPath: options.output.path) else { throw StudioError("Choose a new inference output folder.") }
        let stage = options.output.deletingLastPathComponent().appendingPathComponent(".native-material-inference-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: stage) }
        let filename = options.target + ".float32.exr", final = options.output.appendingPathComponent(filename)
        try NativeMaterialNumericExporter.writeEXR(prediction, to: stage.appendingPathComponent(filename))
        let digest = try checksum(Data(contentsOf: stage.appendingPathComponent(filename)))
        let checkpointHash = try options.checkpoint.map { checksum(try Data(contentsOf: NativeMaterialModel.resolvedCheckpoint($0))) }
        let result: [String: Any] = ["checkpoint_sha256": checkpointHash ?? model.baseSHA256,
            "checkpoint_step": options.baseline ? 0 : model.configuration["step"] ?? 0,
            "target": options.target, "input_kind": "diffuse", "diffuse_path": imageURL.path,
            "image_sha256": checksum(snapshot), "native_dimensions": [prediction.width, prediction.height],
            "source_bits": photo.header.bits, "source_bytes_modified": false,
            "generation": ["tiled": false, "model_input_dimensions": [prediction.width, prediction.height],
                "source_pixels_resized": false, "source_pixels_discarded": false, "runtime": "Apple MPSGraph Float32"],
            "outputs": [options.target: ["path": final.path, "sha256": digest, "encoding": "linear_data", "storage": "FLOAT32", "blender_color_space": "Non-Color"]]]
        try writeJSON(result, to: stage.appendingPathComponent("inference.json"))
        try control.check()
        try FileManager.default.moveItem(at: stage, to: options.output)
        return try json(result)
    }
    static func train(_ options: Options, onEvent: Event, control: NativeMaterialTrainingControl, model suppliedModel: NativeMaterialModel? = nil) throws -> String {
        try control.check()
        let dataset = options.dataset!
        var directories = [dataset]
        let manifestBytes = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let manifest = try JSONSerialization.jsonObject(with: manifestBytes) as? [String: Any]
        if let lineage = manifest?["native_size_preparation"] as? [String: Any],
           let source = lineage["source_dataset_path"] as? String {
            let original = URL(fileURLWithPath: source).resolvingSymlinksInPath()
            if original != dataset.resolvingSymlinksInPath() { directories.append(original) }
        }
        let locks = try directories.sorted { $0.path < $1.path }.map(DatasetReadLock.init)
        defer { locks.forEach { $0.close() } }
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: dataset, size: options.size, target: options.target, materials: options.materials)
        let training = descriptors.filter { $0.split == "train" }, validation = descriptors.filter { $0.split == "validation" }
        guard !training.isEmpty else { throw StudioError("The selected native dataset has no included training maps.") }
        let sourceHash = checksum(manifestBytes)
        guard checksum(try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))) == sourceHash else {
            throw StudioError("The prepared dataset changed before native training obtained its read locks.")
        }
        let model = try suppliedModel ?? NativeMaterialModel.load(checkpointURL: options.checkpoint, expectedSHA256: options.expectedSHA256,
            baseURL: options.base, target: options.target, scope: options.scope, rank: options.rank, alpha: options.alpha, training: true, seed: options.seed)
        let program = try model.program(width: options.size, height: options.size, target: options.target)
        try control.check()
        guard !FileManager.default.fileExists(atPath: options.output.path) else { throw StudioError("Choose a new training output directory.") }
        try FileManager.default.createDirectory(at: options.output, withIntermediateDirectories: true)
        let started = ContinuousClock.now
        var completed = 0, random = NativeMaterialRandom(seed: options.seed)
        let initialStep = model.configuration["step"] as? Int ?? 0
        var optimizer: [String: NativeTensor] = [:], checkpoints: [[String: Any]] = [], history: [[String: Any]] = []
        var lastSavedStep = -1, lastValidation: [String: Any] = [:]
        func emit(_ event: [String: Any]) throws { let line = try json(event) + "\n"; onEvent(line) }
        func pair(_ descriptor: NativeMaterialDatasetService.TrainingSample) throws -> ([Float], [Float]) {
            try control.check()
            let inputBytes = try Data(contentsOf: descriptor.inputURL), targetBytes = try Data(contentsOf: descriptor.targetURL)
            guard checksum(inputBytes) == descriptor.inputSHA256, checksum(targetBytes) == descriptor.targetSHA256 else {
                throw StudioError("A native training map changed after the dataset was prepared.")
            }
            let input = try NativePNG.decode(inputBytes), target = try NativePNG.decode(targetBytes)
            guard input.header.width == options.size, input.header.height == options.size,
                  target.header.width == input.header.width, target.header.height == input.header.height else { throw StudioError("Every native training pair must exactly match the prepared model grid.") }
            let rgb = try input.modelFloatSamples(role: "input", encoding: descriptor.inputEncoding)
            let reference = try target.modelFloatSamples(role: options.target, normalConvention: descriptor.targetConvention)
            return (rgb, reference)
        }
        func check(full: Bool) throws -> [String: Any] {
            var errors: [[String: Any]] = [], sum = Double(0)
            let selected = full ? validation : Array(validation.prefix(3))
            for sample in selected {
                let data = try pair(sample)
                let result = try program.execute(rgb: data.0, adapters: model.adapterWeights, reference: data.1)
                let mae = Double(result.valueLoss!)
                errors.append(["sample_id": sample.id, "mae": mae]); sum += mae
            }
            let result: [String: Any] = ["event": "validation", "status": errors.isEmpty ? "unavailable" : "checked",
                "scope": full ? "full" : "quick", "sample_count": errors.count, "pool_count": validation.count,
                "mae": errors.isEmpty ? NSNull() : sum / Double(errors.count), "samples": errors,
                "step": initialStep + completed, "reference": "Teacher/source agreement, not measured material accuracy"]
            history.append(result); if history.count > 200 { history.removeFirst() }
            try emit(result)
            return result
        }
        func save() throws -> URL {
            if lastSavedStep == completed { return options.output.appendingPathComponent(String(format: "checkpoint-step-%08d.safetensors", initialStep + completed)) }
            lastValidation = try check(full: true)
            let config = try model.checkpointConfiguration(size: options.size, step: initialStep + completed, validation: lastValidation)
            let configText = try json(config)
            let destination = options.output.appendingPathComponent(String(format: "checkpoint-step-%08d.safetensors", initialStep + completed))
            try NativeSafetensors.write(tensors: model.adapterWeights, metadata: ["configuration": configText], to: destination)
            var information = try JSONSerialization.jsonObject(with: Data(NativeMaterialCheckpoint.inspect(at: destination).utf8)) as! [String: Any]
            checkpoints.append(information)
            information["event"] = "checkpoint_saved"; information["validation"] = lastValidation
            try emit(information)
            lastSavedStep = completed
            return destination
        }
        do {
            let baseline = try check(full: true)
            try emit(["event": "training_started", "runtime": "Apple MPSGraph", "precision": "Float32", "training_size": options.size])
            for _ in 0..<options.updatesPerMap {
                var order = Array(training.indices); random.shuffle(&order)
                for index in order {
                    try control.check()
                    if control.shouldStopAndSave { break }
                    let sample = training[index], data = try pair(sample)
                    let update = try program.execute(rgb: data.0, adapters: model.adapterWeights, reference: data.1,
                        learningRate: options.learningRate, step: completed + 1, optimizerState: optimizer)
                    try control.check()
                    model.updateAdapters(update.updated); optimizer = update.optimizerState
                    completed += 1
                    try emit(["event": "update", "step": initialStep + completed, "sample_id": sample.id,
                        "value_l1": update.valueLoss!, "detail_l1": update.gradientLoss!, "total": update.loss!,
                        "native_dimensions": [options.size, options.size]])
                    if control.consumeCheckpoint() || options.checkpointEvery > 0 && completed % options.checkpointEvery == 0 { _ = try save() }
                    else if completed % options.validationEvery == 0 { _ = try check(full: false) }
                    if control.shouldStopAndSave || started.duration(to: .now) >= .seconds(options.maxMinutes * 60) { break }
                }
                if control.shouldStopAndSave || started.duration(to: .now) >= .seconds(options.maxMinutes * 60) { break }
            }
            try control.check()
            let checkpoint = try save()
            let export = options.output.appendingPathComponent("export")
            let configuration = try model.checkpointConfiguration(size: options.size, step: initialStep + completed, validation: lastValidation)
            _ = try NativeMaterialPackage.export(model: model, configuration: configuration, to: export, developer: options.developerMode)
            let result: [String: Any] = ["status": control.shouldStopAndSave ? "stopped" : "completed",
                "checkpoint_path": export.appendingPathComponent(options.developerMode ? "model.safetensors" : "adapter.safetensors").path,
                "package_path": export.path, "completed_updates": completed, "training_performed": completed > 0,
                "baseline_validation": baseline, "final_validation": lastValidation, "validation_history": history,
                "checkpoints": checkpoints, "last_checkpoint": checkpoint.path,
                "dataset_manifest_sha256": sourceHash, "native_dimensions": [options.size, options.size],
                "image_padding": false, "image_resizing": false, "runtime": "Apple MPSGraph Float32"]
            try writeJSON(result, to: options.output.appendingPathComponent("run.json"))
            return try json(result)
        } catch {
            try? writeJSON(["status": error is CancellationError ? "aborted" : "failed", "completed_updates": completed,
                "error": error.localizedDescription, "training_performed": completed > 0], to: options.output.appendingPathComponent("run.json"))
            throw error
        }
    }
    static func checksum(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func json(_ object: [String: Any]) throws -> String { String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self) }
    private static func writeJSON(_ object: [String: Any], to url: URL) throws { try Data(json(object).utf8).write(to: url, options: .atomic) }
    private final class DatasetReadLock {
        var descriptor: Int32
        init(_ directory: URL) throws {
            descriptor = Darwin.open(directory.appendingPathComponent(".material-workbench.lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
            guard descriptor >= 0 else { throw StudioError("The training dataset cannot be locked.") }
            guard flock(descriptor, LOCK_SH | LOCK_NB) == 0 else { Darwin.close(descriptor); descriptor = -1; throw StudioError("The dataset is being modified in another window.") }
        }
        func close() { if descriptor >= 0 { flock(descriptor, LOCK_UN); Darwin.close(descriptor); descriptor = -1 } }
        deinit { close() }
    }
}

/// Direct standard FLOAT32 EXR scanlines avoid Core Image's HALF export and
/// any color-management/renderer conversion on a numeric model prediction.
enum NativeMaterialNumericExporter {
    static func writeEXR(_ prediction: NativeMaterialPrediction, to url: URL) throws {
        let width = prediction.width, height = prediction.height, channelCount = prediction.channels
        guard [1, 3].contains(channelCount), width > 0, height > 0,
              prediction.values.count == width * height * channelCount, prediction.values.allSatisfy(\.isFinite) else { throw StudioError("Invalid native material FLOAT32 image.") }
        let channels = channelCount == 1 ? ["R"] : ["B", "G", "R"]
        func u32(_ value: UInt32) -> Data { var value = value.littleEndian; return withUnsafeBytes(of: &value) { Data($0) } }
        func u64(_ value: UInt64) -> Data { var value = value.littleEndian; return withUnsafeBytes(of: &value) { Data($0) } }
        func text(_ value: String) -> Data { Data(value.utf8) + Data([0]) }
        func attribute(_ name: String, _ type: String, _ value: Data) -> Data { text(name) + text(type) + u32(UInt32(value.count)) + value }
        var header = u32(20_000_630) + u32(2), list = Data()
        for channel in channels { list.append(text(channel) + u32(2) + Data([0, 0, 0, 0]) + u32(1) + u32(1)) }
        list.append(0)
        header.append(attribute("channels", "chlist", list)); header.append(attribute("compression", "compression", Data([0])))
        let window = u32(0) + u32(0) + u32(UInt32(width - 1)) + u32(UInt32(height - 1))
        header.append(attribute("dataWindow", "box2i", window)); header.append(attribute("displayWindow", "box2i", window))
        header.append(attribute("lineOrder", "lineOrder", Data([0])))
        header.append(attribute("pixelAspectRatio", "float", u32(Float(1).bitPattern)))
        header.append(attribute("screenWindowCenter", "v2f", u32(0) + u32(0)))
        header.append(attribute("screenWindowWidth", "float", u32(Float(1).bitPattern))); header.append(0)
        let rowBytes = width * channelCount * 4, first = header.count + height * 8
        for row in 0..<height { header.append(u64(UInt64(first + row * (rowBytes + 8)))) }
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw StudioError("The native EXR output could not be created.") }
        let handle = try FileHandle(forWritingTo: url); defer { try? handle.close() }
        try handle.write(contentsOf: header)
        for row in 0..<height {
            try Task.checkCancellation()
            var chunk = u32(UInt32(row)) + u32(UInt32(rowBytes))
            for channel in channels {
                let component = channel == "B" ? 2 : channel == "G" ? 1 : 0
                let start = component * width * height + row * width
                prediction.values[start..<start + width].withUnsafeBytes { chunk.append(contentsOf: $0) }
            }
            try handle.write(contentsOf: chunk)
        }
        try handle.synchronize()
    }
}
