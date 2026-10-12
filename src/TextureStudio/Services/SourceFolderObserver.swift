import Foundation
import Darwin

/// Event driven membership updates. Directory enumeration and file-descriptor
/// work run on a utility queue, never on the UI thread. Images are not decoded.
final class SourceFolderObserver: @unchecked Sendable {
    let root: URL
    private let queue = DispatchQueue(label: "TextureStudio.source-folder", qos: .utility)
    private let lock = NSLock()
    private var sources: [String: DispatchSourceFileSystemObject] = [:]
    private var stopped = false
    private var pending: DispatchWorkItem?
    private let onChange: @Sendable () -> Void

    init(root: URL, onChange: @escaping @Sendable () -> Void) {
        self.root = root.resolvingSymlinksInPath().standardizedFileURL
        self.onChange = onChange
        queue.async { [weak self] in self?.updateWatches() }
    }
    deinit {
        lock.lock(); stopped = true
        pending?.cancel()
        let active = Array(sources.values); sources.removeAll()
        lock.unlock()
        active.forEach { $0.cancel() }
    }
    private func updateWatches() {
        var directories = Set([root.path])
        if let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles]) {
            for case let folder as URL in files {
                if let values = try? folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]), values.isDirectory == true, values.isSymbolicLink != true {
                    directories.insert(folder.path)
                }
            }
        }
        lock.lock(); defer { lock.unlock() }
        guard !stopped else { return }
        for path in Array(sources.keys) where !directories.contains(path) { sources.removeValue(forKey: path)?.cancel() }
        for path in directories where sources[path] == nil {
            let descriptor = Darwin.open(path, O_EVTONLY | O_NOFOLLOW)
            guard descriptor >= 0 else { continue }
            let source = DispatchSource.makeFileSystemObjectSource(fileDescriptor: descriptor,
                eventMask: [.write, .rename, .delete, .attrib, .extend], queue: queue)
            source.setEventHandler { [weak self] in self?.changed() }
            source.setCancelHandler { Darwin.close(descriptor) }
            sources[path] = source; source.resume()
        }
    }
    private func changed() {
        lock.lock(); defer { lock.unlock() }
        guard !stopped else { return }
        pending?.cancel()
        let item = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.updateWatches()
            self.onChange()
        }
        pending = item
        queue.asyncAfter(deadline: .now() + .milliseconds(150), execute: item)
    }
}
