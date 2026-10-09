import Foundation
import Observation

@MainActor @Observable
final class ModelManager {
    let catalog: [LocalModelDescriptor]
    private(set) var records: [String: InstalledLocalModel] = [:]
    private(set) var statuses: [String: LocalModelStatus] = [:]
    var lastError: String?
    let storageDirectory: URL

    @ObservationIgnored private let fileManager = FileManager.default
    @ObservationIgnored private let transfer: any ModelArtifactTransferring
    @ObservationIgnored private let validator: any ModelValidating
    @ObservationIgnored private var downloadTasks: [String: Task<Void, Never>] = [:]
    @ObservationIgnored private var operationIDs: [String: UUID] = [:]
    private var registryURL: URL { storageDirectory.appendingPathComponent("model-records.json") }

    init(storageDirectory: URL? = nil, catalog: [LocalModelDescriptor] = LocalModelDescriptor.catalog,
        transfer: any ModelArtifactTransferring = ModelDownloadService(),
        validator: any ModelValidating = LocalModelValidationService()) {
        self.catalog = catalog
        self.transfer = transfer
        self.validator = validator
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        self.storageDirectory = (storageDirectory ?? support.appendingPathComponent("Texture Studio/Models", isDirectory: true))
            .standardizedFileURL.resolvingSymlinksInPath()
        do {
            if fileManager.fileExists(atPath: registryURL.path) {
                records = try JSONDecoder().decode([String: InstalledLocalModel].self, from: Data(contentsOf: registryURL))
                records = records.filter { key, _ in catalog.contains { $0.id == key } }
            }
        } catch { lastError = "Could not read the model library: \(error.localizedDescription). Locate a model to reconnect it." }
        refresh()
    }

    func status(for id: String) -> LocalModelStatus { statuses[id] ?? .missing }

    func availableURL(for id: String) -> URL? {
        guard let record = records[id], modelFilesExist(id: id, record: record), !status(for: id).isBusy else { return nil }
        return URL(fileURLWithPath: record.path)
    }

    func refresh() {
        for descriptor in catalog where !status(for: descriptor.id).isBusy {
            statuses[descriptor.id] = records[descriptor.id].map {
                modelFilesExist(id: descriptor.id, record: $0) ? .ready : .missing
            } ?? .missing
        }
    }

    /// Only an explicit UI action starts a network download. A missing model never
    /// changes the photo workflow or automatically triggers a download.
    func download(id: String) {
        guard let descriptor = catalog.first(where: { $0.id == id }), descriptor.downloadable else {
            lastError = LocalModelError.unknownModel.localizedDescription
            return
        }
        guard !status(for: id).isBusy else { return }
        let operationID = UUID()
        operationIDs[id] = operationID
        statuses[id] = .downloading(0)
        lastError = nil
        downloadTasks[id] = Task { [weak self] in
            guard let self else { return }
            await self.install(descriptor, operationID: operationID)
        }
    }

    func cancelDownload(id: String) { downloadTasks[id]?.cancel() }

    func locate(id: String, url: URL) async throws {
        guard let descriptor = catalog.first(where: { $0.id == id }) else { throw LocalModelError.unknownModel }
        guard !status(for: id).isBusy else { throw LocalModelError.busy }
        if let existing = records[id], existing.isManaged,
           fileManager.fileExists(atPath: existing.path) {
            throw LocalModelError.unsupported("remove the downloaded copy before locating a replacement")
        }
        let previous = records[id]
        let resolved = url.standardizedFileURL.resolvingSymlinksInPath()
        // An external link never inherits deletion rights over managed storage.
        guard !Self.contains(resolved, in: storageDirectory) else { throw LocalModelError.unsafeRemoval }
        statuses[id] = .validating
        do {
            try Self.requireBackend(descriptor.backend, at: resolved)
            let interface = try await validator.inspect(at: resolved)
            try Task.checkCancellation()
            let record = InstalledLocalModel(path: resolved.path, isManaged: false, interface: interface,
                selectedOutput: interface.unambiguousOutput, installedAt: Date())
            try updateRecord(id: id, record: record)
            // Moving an app-managed package away can leave an empty install
            // directory. Reconnecting its new external path never deletes the
            // package, and removes the old directory only when proven empty.
            if let previous, previous.isManaged,
               let directory = try? managedDirectory(for: id, record: previous),
               (try? fileManager.contentsOfDirectory(atPath: directory.path).isEmpty) == true {
                try? fileManager.removeItem(at: directory)
            }
            statuses[id] = .ready
            lastError = nil
        } catch {
            statuses[id] = error is CancellationError
                ? (records[id].map { fileManager.fileExists(atPath: $0.path) ? .ready : .missing } ?? .missing)
                : .failed(error.localizedDescription)
            lastError = error is CancellationError ? nil : error.localizedDescription
            throw error
        }
    }

    func chooseOutput(id: String, name: String) throws {
        guard var record = records[id] else { throw LocalModelError.missingModel }
        guard record.interface.outputs.contains(where: { $0.name == name }) else {
            throw LocalModelError.unsupported("the chosen output is not a grayscale float map")
        }
        record.selectedOutput = name
        try updateRecord(id: id, record: record)
    }

    /// Managed downloads are deleted. Located external files are only unlinked.
    func remove(id: String) throws {
        guard !status(for: id).isBusy else { throw LocalModelError.busy }
        guard let record = records[id] else { statuses[id] = .missing; return }
        if record.isManaged, fileManager.fileExists(atPath: URL(fileURLWithPath: record.path).deletingLastPathComponent().path) {
            let directory = try managedDirectory(for: id, record: record)
            try fileManager.removeItem(at: directory)
        }
        try updateRecord(id: id, record: nil)
        statuses[id] = .missing
        lastError = nil
    }

    private func modelFilesExist(id: String, record: InstalledLocalModel) -> Bool {
        fileManager.fileExists(atPath: record.path)
    }

    private func install(_ descriptor: LocalModelDescriptor, operationID: UUID) async {
        let id = descriptor.id
        let stagingRoot = storageDirectory.appendingPathComponent(".staging", isDirectory: true)
            .appendingPathComponent(operationID.uuidString, isDirectory: true)
        let stagedPackage = stagingRoot.appendingPathComponent(descriptor.packageName, isDirectory: true)
        var publishedDirectory: URL?
        defer {
            try? fileManager.removeItem(at: stagingRoot)
            if operationIDs[id] == operationID {
                operationIDs[id] = nil
                downloadTasks[id] = nil
            }
        }
        do {
            guard Self.safeComponent(id), Self.safeComponent(descriptor.packageName) else {
                throw LocalModelError.invalidDownload("invalid package path")
            }
            try fileManager.createDirectory(at: stagedPackage, withIntermediateDirectories: true)
            let finalDirectory = storageDirectory.appendingPathComponent(id, isDirectory: true)
            // Do not overwrite an installed model or unrelated storage.
            guard !fileManager.fileExists(atPath: finalDirectory.path) else {
                throw LocalModelError.invalidDownload("a managed copy already exists; remove it before installing again")
            }
            let capacity = try storageDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage
            if let capacity, capacity < descriptor.downloadBytes + 1_000_000_000 {
                throw LocalModelError.invalidDownload("not enough disk space for this model and installation staging")
            }
            var completedBytes: Int64 = 0
            for artifact in descriptor.artifacts {
                try Task.checkCancellation()
                let destination = stagedPackage.appendingPathComponent(artifact.relativePath)
                guard Self.contains(destination.standardizedFileURL, in: stagedPackage),
                      !artifact.relativePath.split(separator: "/").contains("..") else {
                    throw LocalModelError.invalidDownload("invalid artifact path")
                }
                let bytesBeforeArtifact = completedBytes
                try await transfer.fetch(artifact, to: destination) { [weak self] fraction in
                    Task { @MainActor in
                        guard let self, self.operationIDs[id] == operationID,
                              case .downloading = self.status(for: id) else { return }
                        let total = Double(bytesBeforeArtifact) + Double(artifact.byteCount) * fraction
                        self.statuses[id] = .downloading(min(1, total / Double(max(1, descriptor.downloadBytes))))
                    }
                }
                try await Task.detached { try ModelFileValidation.validate(destination, artifact: artifact) }.value
                completedBytes += artifact.byteCount
            }
            try Task.checkCancellation()
            statuses[id] = .validating
            let interface = try await validator.inspect(at: stagedPackage)
            try Task.checkCancellation()
            // Publish only after every hash passes and the backend accepts the package.
            try fileManager.moveItem(at: stagingRoot, to: finalDirectory)
            publishedDirectory = finalDirectory
            let package = finalDirectory.appendingPathComponent(descriptor.packageName, isDirectory: true)
            try updateRecord(id: id, record: InstalledLocalModel(path: package.path, isManaged: true,
                interface: interface, selectedOutput: interface.unambiguousOutput, installedAt: Date()))
            publishedDirectory = nil
            statuses[id] = .ready
            lastError = nil
        } catch {
            if let publishedDirectory { try? fileManager.removeItem(at: publishedDirectory) }
            let cancelled = error is CancellationError || (error as? URLError)?.code == .cancelled
            statuses[id] = cancelled ? (records[id].map { fileManager.fileExists(atPath: $0.path) ? .ready : .missing } ?? .missing)
                : .failed(error.localizedDescription)
            lastError = cancelled ? nil : error.localizedDescription
        }
    }

    private func managedDirectory(for id: String, record: InstalledLocalModel) throws -> URL {
        guard let descriptor = catalog.first(where: { $0.id == id }), Self.safeComponent(id) else { throw LocalModelError.unsafeRemoval }
        let expectedDirectory = storageDirectory.appendingPathComponent(id, isDirectory: true)
        let expectedPackage = expectedDirectory.appendingPathComponent(descriptor.packageName)
        let actual = URL(fileURLWithPath: record.path).standardizedFileURL
        let directoryValues = try expectedDirectory.resourceValues(forKeys: [.isSymbolicLinkKey])
        guard actual.path == expectedPackage.standardizedFileURL.path, directoryValues.isSymbolicLink != true,
              expectedDirectory.resolvingSymlinksInPath().deletingLastPathComponent().path == storageDirectory.resolvingSymlinksInPath().path else {
            throw LocalModelError.unsafeRemoval
        }
        return expectedDirectory
    }

    private func updateRecord(id: String, record: InstalledLocalModel?) throws {
        let previous = records[id]
        records[id] = record
        do {
            try fileManager.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(records)
            try data.write(to: registryURL, options: .atomic)
        } catch { records[id] = previous; throw error }
    }

    private static func requireBackend(_ backend: LocalModelBackend, at url: URL) throws {
        guard ["mlpackage", "mlmodel", "mlmodelc"].contains(url.pathExtension.lowercased()) else {
            throw LocalModelError.unsupported("select a Core ML model package")
        }
    }

    private static func safeComponent(_ value: String) -> Bool {
        !value.isEmpty && !value.contains("/") && value != "." && value != ".."
    }
    private static func contains(_ child: URL, in parent: URL) -> Bool {
        child.path.hasPrefix(parent.path.hasSuffix("/") ? parent.path : parent.path + "/")
    }
}
