import Foundation
import AppKit

extension WorkbenchStore {
    func addSourceFiles(_ urls: [URL]) {
        guard !isBusy else { return }
        operation("Adding original images and folders…") { try await self.registerSourceFiles(urls) }
    }
    func registerSourceFiles(_ urls: [URL]) async throws {
        guard let root = datasetFolderURL else { throw StudioError("Create or open a source folder first.") }
        let loose = urls.filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) != true }
        let supplementalRole = #"ambient[_ -]?occlusion(?:[_ -]?map)?|occlusion[_ -]?roughness[_ -]?metallic|anisotropy[_ -](?:rotation|strength)|rough[_ -]ao|spec[_ -]ior|ao|occlusion|orm|arm|rma|metallic|metalness|metal|bump|glossiness|gloss|smoothness|specular|spec|emissive|emission|opacity|translucent"#
        let size = #"\d+k|[1-9]\d{2,5}(?:px)?|[1-9]\d{1,5}x[1-9]\d{1,5}"#
        let supplementalPatterns = [
            "^(.+?)[_ -](?:" + supplementalRole + ")[_ -](" + size + ")$",
            "^(.+?)[_ -](" + size + ")(?:-PNG)?[_ -](?:" + supplementalRole + ")$",
            "^(.+?)[_ -](?:" + supplementalRole + ")$"
        ].compactMap { try? NSRegularExpression(pattern: $0, options: .caseInsensitive) }
        func familyAndTier(_ url: URL) -> (String, String)? {
            if let (family, _, tier) = NativeMaterialDatasetService.sourceMapFilename(url.lastPathComponent) { return (family, tier) }
            guard NativeMaterialDatasetService.isAncillarySourceFilename(url.lastPathComponent) else { return nil }
            let stem = url.deletingPathExtension().lastPathComponent
            for pattern in supplementalPatterns {
                guard let match = pattern.firstMatch(in: stem, range: NSRange(stem.startIndex..., in: stem)),
                      let family = Range(match.range(at: 1), in: stem) else { continue }
                let tier = match.numberOfRanges > 2 ? Range(match.range(at: 2), in: stem).map { String(stem[$0]).lowercased() } : nil
                return (String(stem[family]).lowercased(), tier ?? "native")
            }
            return nil
        }
        var groupKeys: [String: String] = [:], preferredNames: [String: String] = [:]
        for url in loose {
            let parent = url.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL
            let family = familyAndTier(url)
            let key = parent.path + "\n" + (family?.0 ?? "") + "\n" + (family?.1 ?? "")
            groupKeys[url.path] = key
            preferredNames[key] = family.map { $0.0 + ($0.1 == "native" ? "" : " " + $0.1.uppercased()) } ?? parent.lastPathComponent
        }
        let loosePaths = Set(loose.map(\.path))
        var reserved = Set(urls.filter { !loosePaths.contains($0.path) }.map(\.lastPathComponent))
        var groupNames: [String: String] = [:]
        for key in preferredNames.keys.sorted() {
            let base = (preferredNames[key] ?? "Images").replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
            let cleanName = ["", ".", ".."].contains(base) ? "Images" : base
            var name = cleanName, suffix = 2
            while reserved.contains(name) || FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path) {
                name = cleanName + " \(suffix)"; suffix += 1
            }
            reserved.insert(name); groupNames[key] = name
        }
        var issues: [String] = [], added = 0
        for url in urls {
            try Task.checkCancellation()
            do {
                var args = ["add-source", "--dataset", root.path, "--source", url.path, "--defer-refresh", "true"]
                if let key = groupKeys[url.path], let name = groupNames[key] { args += ["--folder-name", name] }
                _ = try await worker(args); added += 1
            } catch is CancellationError { throw CancellationError() }
            catch { issues.append(url.lastPathComponent + ": " + error.localizedDescription) }
        }
        try await loadDataset(root, recoverSources: false)
        datasetBrowserMode = .originals
        activity = "Added \(added) sources. " + issues.joined(separator: "; ")
    }
    func saveSourceImageTags(_ url: URL, role: String, classLabel: String, folderName: String? = nil) {
        guard !isBusy, let root = datasetFolderURL else { return }
        operation("Saving image tags…") {
            _ = try await self.worker(["tag-source", "--dataset", root.path, "--source", url.path, "--role", role, "--class-label", classLabel, "--defer-refresh", "true"])
            if let folderName, folderName != url.deletingLastPathComponent().lastPathComponent,
               url.deletingLastPathComponent().resolvingSymlinksInPath().path != root.resolvingSymlinksInPath().path {
                _ = try await self.worker(["rename-source", "--dataset", root.path, "--source", url.deletingLastPathComponent().path, "--name", folderName, "--defer-refresh", "true"])
            }
            try await self.loadDataset(root, recoverSources: false)
            self.activity = "Image tags saved. Changed source folders are shown with their current names."
        }
    }
    func renameSourceFolder(_ folder: URL, name: String) {
        guard !isBusy, let root = datasetFolderURL else { return }
        operation("Renaming source folder…") {
            _ = try await self.worker(["rename-source", "--dataset", root.path, "--source", folder.path, "--name", name, "--defer-refresh", "true"])
            try await self.loadDataset(root, recoverSources: false)
        }
    }
    func repairSourceTags() {
        guard !isBusy, let root = datasetFolderURL else { return }
        operation("Remembering recognized map roles…") {
            _ = try await self.worker(["repair-source-tags", "--dataset", root.path, "--defer-refresh", "true"])
            try await self.loadDataset(root, recoverSources: false)
            self.activity = "Recognized map roles saved with the images. Unassigned images remain visible for teaching."
        }
    }
    func moveSourceFolderOut(_ folder: URL) {
        guard !isBusy, let root = datasetFolderURL else { return }
        let panel = NSOpenPanel()
        panel.title = "Move Material Out"
        panel.message = "Choose a folder outside this source folder. The material and every original map move together."
        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let destination = panel.url else { return }
            self.operation("Moving material folder…") {
                _ = try await self.worker(["move-source", "--dataset", root.path, "--source", folder.path, "--destination", destination.path, "--defer-refresh", "true"])
                try await self.loadDataset(root, recoverSources: false)
                self.activity = "Moved material out of the source folder. Its original images are kept together."
            }
        }
    }
    func revealSelectedOriginalSource() {
        if let map = selectedMap {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: map.originalSourcePath ?? map.path)])
        } else { revealDataset() }
    }
    func observeSourceFolder(_ folder: URL?) {
        let canonical = folder?.resolvingSymlinksInPath().standardizedFileURL
        guard sourceFolderObserver?.root != canonical else { return }
        sourceFolderObserver = canonical.map { root in
            SourceFolderObserver(root: root) { [weak self] in
                Task { @MainActor [weak self] in
                    self?.sourceFolderNeedsRefresh = true
                    self?.refreshSourceFolderIfNeeded()
                }
            }
        }
    }
    func refreshSourceFolderIfNeeded() {
        guard sourceFolderNeedsRefresh, !isBusy, !isRefreshingSourceFolder,
              let current = dataset, sourceFolderObserver != nil else { return }
        sourceFolderNeedsRefresh = false
        isRefreshingSourceFolder = true
        let selectedPath = current.datasetPath
        let root = current.preparation == nil ? selectedPath :
            specialistTrainingContext?.datasetPath ?? current.sourceFolderPath ?? current.preparation!.sourceDatasetPath
        let size = training.size
        Task { @MainActor in
            defer {
                isRefreshingSourceFolder = false
                refreshSourceFolderIfNeeded()
            }
            do {
                let output = try await worker(["dataset", "--dataset", root, "--review-size", String(size), "--skip-source-recovery"])
                var updated = try WorkbenchResult.decode(WorkbenchDataset.self, output: output)
                updated.trainingSize = size
                updated.reviewSize = size
                guard !isBusy, dataset?.datasetPath == selectedPath, training.size == size else {
                    sourceFolderNeedsRefresh = true
                    return
                }
                adoptDataset(updated)
            } catch {
                if !isBusy { activity = "Source folder update: \(error.localizedDescription)" }
            }
        }
    }
}
