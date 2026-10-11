import Foundation

/// Material safetensors use the native trainer's checkpoint validator and
/// selection registry. They are independent of developer Core ML packages.
@MainActor
enum MaterialCheckpointImporter {
    @discardableResult
    static func activate(_ url: URL, using store: WorkbenchStore) async throws -> WorkbenchCheckpoint {
        store.error = nil
        try await store.loadCheckpoint(url)
        guard let checkpoint = store.selectedCheckpoint, checkpoint.supportsStudioInference else {
            throw StudioError("Choose a supported Texture Studio material safetensors model or its exported package.")
        }
        store.useSelectedInStudio()
        if let error = store.error { throw StudioError(error) }
        return checkpoint
    }
}
