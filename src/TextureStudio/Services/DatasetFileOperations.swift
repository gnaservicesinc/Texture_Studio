import Foundation
import CryptoKit
import Darwin

enum DatasetFileOperations {
    /// Validate the backend deletion scope again under the metadata lock before
    /// handing the one owned item to macOS Trash. Source maps are never paths
    /// in a deletion request.
    static func trash(_ plan: WorkbenchDatasetDeletion, expectedDataset: URL, handler: (URL) throws -> Void) throws {
        let root = expectedDataset.resolvingSymlinksInPath().standardizedFileURL
        guard root == URL(fileURLWithPath: plan.datasetPath).resolvingSymlinksInPath().standardizedFileURL,
              root.path != "/", root.path != NSHomeDirectory(), plan.trashPaths.count == 1 else {
            throw StudioError("Dataset deletion scope could not be verified.")
        }
        let allowed = plan.safeToTrashFolder ? root : root.appendingPathComponent("dataset.json")
        let target = URL(fileURLWithPath: plan.trashPaths[0]).standardizedFileURL
        guard target == allowed, target.resolvingSymlinksInPath().standardizedFileURL == allowed else {
            throw StudioError("Dataset deletion scope changed. Reopen the dataset and try again.")
        }
        let descriptor = Darwin.open(root.appendingPathComponent(".material-workbench.lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(descriptor) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            throw StudioError("This dataset is being edited in another window. Try again when it finishes.")
        }
        defer { flock(descriptor, LOCK_UN) }
        let index = root.appendingPathComponent("dataset.json")
        let bytes = try Data(contentsOf: index)
        let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        guard hash == plan.indexSha256 else { throw StudioError("Dataset changed since selection. Reopen it before deleting.") }
        if plan.safeToTrashFolder { try verifyOwnedFolder(root) }
        try handler(target)
    }

    private static func verifyOwnedFolder(_ root: URL) throws {
        let metadataFiles: Set<String> = ["dataset.json", ".material-workbench.lock", ".material-size-reviews.json", ".DS_Store"]
        var enumerationFailed = false
        guard let entries = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isSymbolicLinkKey, .isRegularFileKey], options: [], errorHandler: { _, _ in enumerationFailed = true; return false }) else {
            throw StudioError("Dataset folder could not be checked before moving it to Trash.")
        }
        for case let entry as URL in entries {
            let values = try entry.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey])
            let canonicalEntry = entry.resolvingSymlinksInPath().standardizedFileURL
            guard canonicalEntry.path.hasPrefix(root.path + "/") else { throw changedFolder() }
            let relative = String(canonicalEntry.path.dropFirst(root.path.count + 1))
            let parts = relative.split(separator: "/")
            guard values.isSymbolicLink != true else { throw changedFolder() }
            if values.isRegularFile == true {
                let sampleMetadata = parts.count == 3 && parts.first == "samples" && parts.last == "sample.json"
                guard metadataFiles.contains(relative) || sampleMetadata else { throw changedFolder() }
                if sampleMetadata {
                    let document = try JSONSerialization.jsonObject(with: Data(contentsOf: entry))
                    guard !referencesInsideFolder(document, root: root) else { throw changedFolder() }
                }
            }
        }
        guard !enumerationFailed else { throw changedFolder() }
    }

    private static func referencesInsideFolder(_ value: Any, root: URL) -> Bool {
        if let string = value as? String, string.hasPrefix("/") {
            let path = URL(fileURLWithPath: string).resolvingSymlinksInPath().standardizedFileURL.path
            return path == root.path || path.hasPrefix(root.path + "/")
        }
        if let dictionary = value as? [String: Any] { return dictionary.values.contains { referencesInsideFolder($0, root: root) } }
        if let array = value as? [Any] { return array.contains { referencesInsideFolder($0, root: root) } }
        return false
    }
    private static func changedFolder() -> StudioError {
        StudioError("Dataset folder contents changed. Reopen it before deleting; original maps will be kept.")
    }
}
