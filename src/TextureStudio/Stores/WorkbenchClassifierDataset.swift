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

    var summary: String {
        let moved = (reclassified ?? 0) > 0 ? " · \(reclassified!) reclassified" : ""
        return "\(total) diffuse images · \(added) added\(moved) · \(needsReview) labels to review"
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
        let originalDataset = dataset?.preparation?.sourceDatasetPath ?? dataset?.datasetPath
        guard !destination.isEmpty, !folder.isEmpty || originalDataset != nil else {
            showClassifierDatasetSettings = true
            return
        }
        let source = folder.isEmpty ? ["--dataset", originalDataset!] : ["--source-folder", folder]
        operation("Refreshing diffuse classifier dataset…") {
            let result = try WorkbenchResult.decode(WorkbenchClassifierDatasetResult.self,
                output: await self.worker(["refresh-classifier-dataset"] + source + ["--destination", destination]))
            self.classifierDatasetResult = result
            self.activity = result.summary + (result.trainingReady == false
                ? ". Labels saved. Could not refresh the Create ML training folder; see the activity log."
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
        if source { classifierSourceFolderPath = url.path } else { classifierDatasetPath = url.path }
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
