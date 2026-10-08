import CoreImage
import Foundation

@MainActor
final class MaterialCheckpointService {
    private var runner: WorkbenchProcess?
    func cancel() { runner?.stop() }

    func predict(source: TextureSource, checkpoint: SelectedMaterialCheckpoint, size: Int) async throws -> TextureDepth {
        guard !WorkbenchLifecycle.shared.isTerminating else { throw CancellationError() }
        guard checkpoint.target == "height" else { throw StudioError("Select a height checkpoint in Material Trainer.") }
        guard FileManager.default.fileExists(atPath: checkpoint.checkpointPath),
              FileManager.default.isExecutableFile(atPath: checkpoint.pythonPath) else {
            throw StudioError("The selected checkpoint or Python runtime moved. Open Material Trainer → Checkpoints to reconnect it.")
        }
        guard size <= 2048 else { throw StudioError("Material-checkpoint inference supports up to 2048 native pixels. Choose a 1024 or 2048 output for this model; larger training/inference has not been validated.") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("material-inference-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let imageURL = root.appendingPathComponent("source.png")
        try Self.writeInferencePhoto(source.orientedImage, maximumSize: size, to: imageURL)
        let backend = Bundle.main.resourceURL!.appendingPathComponent("MaterialBackend/material_workbench.py")
        let output = root.appendingPathComponent("prediction")
        let process = WorkbenchProcess(); runner = process
        WorkbenchLifecycle.shared.add(process)
        defer { runner = nil; WorkbenchLifecycle.shared.remove(process) }
        let text = try await process.run(executable: URL(fileURLWithPath: checkpoint.pythonPath), arguments: [
            "-B", backend.path, "infer", "--checkpoint", checkpoint.checkpointPath,
            "--expected-sha256", checkpoint.sha256, "--image", imageURL.path, "--output", output.path,
            "--device", "mps", "--model-directory", checkpoint.modelDirectory, "--code-directory", checkpoint.codeDirectory
        ], directory: URL(fileURLWithPath: checkpoint.workspacePath), log: root.appendingPathComponent("worker.log"), onLog: { _ in })
        let result = try WorkbenchProcess.decode(MaterialInferenceResponse.self, output: text)
        guard result.checkpointSha256 == checkpoint.sha256, let height = result.outputs["height"],
              let imported = CIImage(contentsOf: URL(fileURLWithPath: height.path), options: [.colorSpace: NSNull()]) else {
            throw StudioError("The selected material checkpoint returned inconsistent height output.")
        }
        // Materialize the Float32 data before deleting the worker directory.
        let width = Int(imported.extent.width), heightPixels = Int(imported.extent.height)
        var values = [Float](repeating: 0, count: width * heightPixels)
        let numeric = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        values.withUnsafeMutableBytes { numeric.render(imported, toBitmap: $0.baseAddress!, rowBytes: width * 4,
            bounds: imported.extent, format: .Rf, colorSpace: nil) }
        return try TextureDepth(width: width, height: heightPixels, values: values,
            sourceLabel: "Material head · \(checkpoint.sha256.prefix(12)) · native \(width) × \(heightPixels)", interpretation: .surfaceHeight)
    }

    static func writeInferencePhoto(_ image: CIImage, maximumSize: Int, to imageURL: URL) throws {
        // An explicit material output size bounds the photographed input. The
        // material head never reduces its native target to the encoder grid.
        let scale = min(1, Double(maximumSize) / max(image.extent.width, image.extent.height))
        let bounded = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let context = CIContext()
        // Keep photographed color above 8-bit precision when the source has it.
        // This storage does not create extra detail in an 8-bit original.
        try context.writePNGRepresentation(of: bounded, to: imageURL, format: .RGBA16,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
    }
}
