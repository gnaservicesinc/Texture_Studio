import CoreImage
import Foundation

@MainActor
final class MaterialCheckpointService {
    private var runner: NativeMaterialTrainingControl?
    func cancel() { runner?.stop() }

    func predict(diffuse: CIImage, checkpoint: SelectedMaterialCheckpoint, size: Int) async throws -> TextureDepth {
        guard checkpoint.target == "height" else { throw StudioError("Select a height model for displacement.") }
        let prediction = try await predictMap(diffuse: diffuse, checkpoint: checkpoint, size: size)
        var depth = TextureDepth(image: prediction.image, sourceLabel: prediction.sourceLabel, interpretation: .surfaceHeight)
        depth.alignedToOutput = true
        return depth
    }

    func predictMap(diffuse: CIImage, checkpoint: SelectedMaterialCheckpoint, size: Int) async throws -> MaterialModelMap {
        guard checkpoint.supportsStudioInference else { throw StudioError("Select a supported material safetensors checkpoint.") }
        guard !WorkbenchLifecycle.shared.isTerminating else { throw CancellationError() }
        guard FileManager.default.fileExists(atPath: checkpoint.checkpointPath) else {
            throw StudioError("The selected checkpoint moved. Open Material Trainer → Checkpoints to reconnect it.")
        }
        let control = NativeMaterialTrainingControl(); runner = control
        let job = Task.detached(priority: .userInitiated) {
            try await Self.numericPrediction(diffuse: diffuse, checkpoint: checkpoint, size: size, control: control)
        }
        WorkbenchLifecycle.shared.add(control, cancel: { job.cancel() })
        defer { runner = nil; WorkbenchLifecycle.shared.remove(control) }
        return try await withTaskCancellationHandler {
            let prediction = try await job.value
            try Task.checkCancellation()
            return prediction
        } onCancel: { control.stop(); job.cancel() }
    }

    private nonisolated static func numericPrediction(diffuse: CIImage, checkpoint: SelectedMaterialCheckpoint, size: Int,
        control: NativeMaterialTrainingControl) async throws -> MaterialModelMap {
        try control.check()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-inference-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let imageURL = root.appendingPathComponent("source.png")
        try Self.writeInferencePhoto(diffuse, maximumSize: size, to: imageURL)
        let output = root.appendingPathComponent("prediction")
        let text = try await NativeMaterialTrainer.run(arguments: [
            "infer", "--checkpoint", checkpoint.checkpointPath,
            "--expected-sha256", checkpoint.sha256, "--image", imageURL.path, "--output", output.path,
            "--model-directory", checkpoint.modelDirectory,
            "--input-kind", "diffuse", "--input-encoding", "srgb"
        ], control: control)
        try control.check()
        let result = try WorkbenchResult.decode(MaterialInferenceResponse.self, output: text)
        guard result.checkpointSha256 == checkpoint.sha256, let output = result.outputs[checkpoint.target],
              let imported = CIImage(contentsOf: URL(fileURLWithPath: output.path), options: [.colorSpace: NSNull()]),
              imported.extent == CGRect(x: 0, y: 0, width: size, height: size) else {
            throw StudioError("The material model returned a map that does not match the prepared diffuse grid.")
        }
        // Realize numeric Float32 samples before purging the temporary files.
        // RGB normals keep their encoded OpenGL values; they are not renormalized.
        let components = checkpoint.target == "normal" ? 4 : 1
        var values = [Float](repeating: 0, count: size * size * components)
        let numeric = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        let format: CIFormat = components == 4 ? .RGBAf : .Rf
        values.withUnsafeMutableBytes { numeric.render(imported, toBitmap: $0.baseAddress!, rowBytes: size * components * 4,
            bounds: imported.extent, format: format, colorSpace: nil) }
        guard values.allSatisfy(\.isFinite) else { throw StudioError("The material model returned nonfinite map values.") }
        try control.check()
        let image = CIImage(bitmapData: values.withUnsafeBytes { Data($0) }, bytesPerRow: size * components * 4,
            size: CGSize(width: size, height: size), format: format, colorSpace: nil)
        return MaterialModelMap(target: checkpoint.target, image: image,
            sourceLabel: "\(checkpoint.target.capitalized) model · \(checkpoint.sha256.prefix(12)) · native \(size) × \(size)")
    }

    nonisolated static func writeInferencePhoto(_ image: CIImage, maximumSize: Int, to imageURL: URL) throws {
        try Task.checkCancellation()
        guard image.extent == CGRect(x: 0, y: 0, width: maximumSize, height: maximumSize) else {
            throw StudioError("Material inference needs the prepared diffuse map at the exact output resolution.")
        }
        let context = CIContext()
        // Keep photographed color above 8-bit precision when the source has it.
        // This storage does not create extra detail in an 8-bit original.
        try context.writePNGRepresentation(of: image, to: imageURL, format: .RGBA16,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        try Task.checkCancellation()
    }
}
