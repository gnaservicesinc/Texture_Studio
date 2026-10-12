import CryptoKit
import Foundation

extension WorkbenchStore {
    /// Select a source-bound workspace using metadata only. Model files stay closed.
    func alignModelWorkshopWorkspace(with sourcePath: String) async throws {
        let current = specialistWorkspacePath
        let remembered = preferences.dictionary(forKey: "specialistWorkspacesBySource.v1") as? [String: String] ?? [:]
        let base = workspaceURL.appendingPathComponent("Loom Room", isDirectory: true)
        let selection = try await Task.detached(priority: .userInitiated) {
            let source = URL(fileURLWithPath: sourcePath).resolvingSymlinksInPath().standardizedFileURL
            let key = source.path
            let suffix = SHA256.hash(data: Data(key.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
            let folderName = source.lastPathComponent.replacingOccurrences(of: ":", with: "-")
            let preferred = base.appendingPathComponent(folderName + "-" + suffix, isDirectory: true)
            var candidates = [current, remembered[key]].compactMap { $0 }.filter { !$0.isEmpty }
            candidates.append(preferred.path)
            var seen = Set<String>()
            for index in 0..<1000 {
                let path = index < candidates.count ? candidates[index] : preferred.path + " \(index - candidates.count + 2)"
                let workspace = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
                guard seen.insert(workspace.path).inserted else { continue }
                guard (try? NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: source)) == true else { continue }
                let registryURL = workspace.appendingPathComponent("specialists.json")
                let registry: WorkbenchSpecialistRegistry?
                if FileManager.default.fileExists(atPath: registryURL.path) {
                    registry = try WorkbenchResult.decode(WorkbenchSpecialistRegistry.self,
                        output: String(contentsOf: registryURL, encoding: .utf8))
                } else { registry = nil }
                return (key, workspace.path, registry)
            }
            throw StudioError("Could not find an available model workspace for this source folder.")
        }.value
        guard URL(fileURLWithPath: modelWorkshopSourcePath).resolvingSymlinksInPath().standardizedFileURL.path == selection.0 else { return }
        specialistWorkspacePath = selection.1
        specialistRegistry = selection.2
        if !(specialistRegistry?.specialists.contains(where: { $0.id == selectedSpecialistID }) ?? false) {
            selectedSpecialistID = specialistRegistry?.specialists.first?.id
        }
        var updated = remembered
        updated[selection.0] = selection.1
        preferences.set(updated, forKey: "specialistWorkspacesBySource.v1")
    }
}
