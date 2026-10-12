import AppKit

struct WorkbenchClassifierDatasetResult: Decodable {
    struct ClassCount: Decodable, Identifiable {
        let id: String
        let name: String
        let count: Int
    }
    let datasetPath: String
    let manifestPath: String
    let reviewPath: String
    let trainingPath: String?
    let trainingReady: Bool?
    let total: Int
    let added: Int
    let preserved: Int
    let reclassified: Int?
    let needsReview: Int
    let classes: [ClassCount]
    let warnings: [String]
    let trainingCount: Int?
    let minimumNativeSize: Int?
    let eligibleCount: Int?
    let undersizedCount: Int?
    let unreadableCount: Int?
    let trainingUndersizedCount: Int?
    let unclassified: Int?

    var summary: String {
        let moved = (reclassified ?? 0) > 0 ? " · \(reclassified!) reclassified" : ""
        let count = trainingCount.map { "\($0) Create ML images · \(total) curated images" } ?? "\(total) diffuse images"
        let skipped = (undersizedCount ?? 0) > 0 ? " · \(undersizedCount!) smaller originals preserved" : ""
        let unreadable = (unreadableCount ?? 0) > 0 ? " · \(unreadableCount!) unreadable originals" : ""
        let pending = (unclassified ?? 0) > 0 ? " · \(unclassified!) images awaiting a class" : ""
        return count + " · \(added) added\(moved) · \(needsReview) labels to review" + skipped + unreadable + pending
    }
}

extension WorkbenchStore {
    func recordClassifierDatasetProgress(_ text: String) {
        for line in text.split(separator: "\n") {
            guard let event = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  event["event"] as? String == "classifier_dataset_progress",
                  let message = event["message"] as? String else { continue }
            let completed = event["completed"] as? Int ?? 0, total = event["total"] as? Int ?? 0
            activity = total > 0 ? "\(message) · \(completed) / \(total)" : message
        }
    }

    func refreshClassifierDataset() {
        guard !isBusy else { return }
        let folder = classifierSourceFolderPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let destination = classifierDatasetPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let openDataset = dataset.map { _ in modelWorkshopSourcePath }
        let useSourceFolder = classifierUseSourceFolder || openDataset == nil
        guard !destination.isEmpty, (useSourceFolder ? !folder.isEmpty : openDataset != nil) else {
            showClassifierDatasetSettings = true
            return
        }
        let source = useSourceFolder ? ["--source-folder", folder] : ["--dataset", openDataset!]
        operation("Refreshing diffuse classifier dataset…") {
            if !useSourceFolder { try await self.restoreModelWorkshopSourcesIfNeeded() }
            let size = useSourceFolder ? self.classifierSourceMinimumSize : self.datasetResolution
            let result = try WorkbenchResult.decode(WorkbenchClassifierDatasetResult.self,
                output: await self.worker(["refresh-classifier-dataset"] + source + ["--destination", destination, "--size", String(size)]))
            self.classifierDatasetResult = result
            self.activity = result.summary + (result.trainingReady == false
                ? ". Labels saved. Review the notices for images omitted from the Create ML folder."
                : ". Review labels, then use the Create ML training folder shown in Settings.")
            if !result.warnings.isEmpty { self.logText += "\n" + result.warnings.joined(separator: "\n") }
        }
    }

    func chooseClassifierFolder(source: Bool) {
        let panel = NSOpenPanel()
        panel.title = source ? "Choose original diffuse map source folder" : "Choose classifier dataset folder"
        panel.message = source ? "Scan original material folders for diffuse maps." : "Create ML reads one subfolder per class inside this folder."
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = !source
        panel.allowsMultipleSelection = false
        let path = source ? classifierSourceFolderPath : classifierDatasetPath
        if !path.isEmpty { panel.directoryURL = URL(fileURLWithPath: path) }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if source { classifierSourceFolderPath = url.path; classifierUseSourceFolder = true } else { classifierDatasetPath = url.path }
    }

    func revealClassifierDataset() {
        let url = URL(fileURLWithPath: classifierDatasetPath)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func revealClassifierReview() {
        guard let result = classifierDatasetResult else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: result.reviewPath)])
    }

    func revealClassifierTrainingDataset() {
        guard let result = classifierDatasetResult, result.trainingReady == true,
              let path = result.trainingPath else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }
}
