#!/usr/bin/env swift
import Foundation
import Darwin

struct InstallFailure: Error, CustomStringConvertible {
    let description: String
}
struct RunningApplication: Error, CustomStringConvertible {
    let description: String
}

let filesystem = FileManager.default
let materialTools = [("review", "Material Review"), ("compare", "Checkpoint Compare"),
                     ("dataset", "Material Dataset"), ("train", "Material Trainer")]

func exists(_ url: URL) -> Bool {
    (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true ||
        filesystem.fileExists(atPath: url.path)
}

func command(_ executable: String, _ arguments: [String]) throws -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    let output = Pipe()
    process.standardOutput = output
    process.standardError = output
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let text = String(decoding: data, as: UTF8.self)
    guard process.terminationStatus == 0 else {
        throw InstallFailure(description: "\(executable) failed (\(process.terminationStatus)): \(text.trimmingCharacters(in: .whitespacesAndNewlines))")
    }
    return text
}

func readInfo(_ app: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: app.appendingPathComponent("Contents/Info.plist"))
    guard let info = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
        throw InstallFailure(description: "Invalid application Info.plist: \(app.path)")
    }
    return info
}

func validateBundle(_ source: URL) throws {
    var applications: [(URL, String, String, String?)] = [(source, "org.ipde.texture-studio", "Texture Studio", nil)]
    applications += materialTools.map { role, name in
        (source.appendingPathComponent("Contents/Applications/\(name).app"), "org.ipde.material-\(role)", name, role)
    }
    for (app, identifier, name, role) in applications {
        let info = try readInfo(app)
        guard info["IPDEBuildConfiguration"] as? String == "Release" else {
            throw InstallFailure(description: "Install requires an optimized Release build: \(app.path)")
        }
        guard info["CFBundleIdentifier"] as? String == identifier, info["CFBundleExecutable"] as? String == name else {
            throw InstallFailure(description: "Unexpected application identity: \(app.path)")
        }
        let executable = app.appendingPathComponent("Contents/MacOS/\(name)")
        let values = try executable.resourceValues(forKeys: [.isRegularFileKey])
        guard values.isRegularFile == true, filesystem.isExecutableFile(atPath: executable.path) else {
            throw InstallFailure(description: "Application executable is missing: \(executable.path)")
        }
        if let role, info["MaterialToolRole"] as? String != role || exists(app.appendingPathComponent("Contents/Applications")) {
            throw InstallFailure(description: "Invalid or recursively embedded material tool: \(app.path)")
        }
    }
}

struct InstallerIO {
    var processes: () throws -> String = { try command("/bin/ps", ["-axo", "pid=,comm="]) }
    var verifySignature: (URL) throws -> Void = { _ = try command("/usr/bin/codesign", ["--verify", "--deep", "--strict", $0.path]) }
    var copy: (URL, URL) throws -> Void = { _ = try command("/usr/bin/ditto", [$0.path, $1.path]) }
    var move: (URL, URL) throws -> Void = { try filesystem.moveItem(at: $0, to: $1) }
}

func ensureClosed(_ destination: URL, processes: String) throws {
    for line in processes.split(separator: "\n") {
        let fields = line.split(maxSplits: 1, omittingEmptySubsequences: true, whereSeparator: { $0.isWhitespace })
        if fields.count == 2, fields[1].drop(while: { $0.isWhitespace }).hasPrefix(destination.path + "/") {
            throw RunningApplication(description: "Close Texture Studio and its material tools before installing (PID \(fields[0]))")
        }
    }
}

func sameSignedBuild(_ source: URL, _ destination: URL, io: InstallerIO) -> Bool {
    do {
        guard let executable = try readInfo(source)["CFBundleExecutable"] as? String else { return false }
        for name in ["Contents/Info.plist", "Contents/_CodeSignature/CodeResources", "Contents/MacOS/\(executable)"] {
            guard try Data(contentsOf: source.appendingPathComponent(name)) == Data(contentsOf: destination.appendingPathComponent(name)) else { return false }
        }
        try io.verifySignature(source)
        try io.verifySignature(destination)
        return true
    } catch { return false }
}

// Preflight every old file before cleanup: a failed recursive deletion must not
// destroy part of a retained, recoverable previous application.
func removableTree(_ path: URL) -> Bool {
    var info = stat()
    guard lstat(path.path, &info) == 0 else { return false }
    if (info.st_mode & S_IFMT) == S_IFLNK { return true }
    var enumerationFailed = false
    guard let entries = filesystem.enumerator(at: path, includingPropertiesForKeys: [.isDirectoryKey], options: [], errorHandler: { _, _ in
        enumerationFailed = true
        return false
    }) else { return false }
    var paths = [path]
    while let entry = entries.nextObject() as? URL { paths.append(entry) }
    guard !enumerationFailed else { return false }
    let blocked = UInt32(UF_IMMUTABLE | SF_IMMUTABLE | UF_APPEND | SF_APPEND)
    for entry in paths {
        guard lstat(entry.path, &info) == 0, info.st_flags & blocked == 0 else { return false }
        if (info.st_mode & S_IFMT) == S_IFDIR {
            guard access(entry.path, R_OK | W_OK | X_OK) == 0 else { return false }
            if info.st_mode & S_ISVTX != 0, info.st_uid != geteuid(), geteuid() != 0 { return false }
        }
    }
    return true
}

func install(_ source: URL, _ destination: URL, io: InstallerIO = InstallerIO()) throws {
    if sameSignedBuild(source, destination, io: io) {
        print("Already current: \(destination.path)")
        return
    }
    try ensureClosed(destination, processes: io.processes())
    let parent = destination.deletingLastPathComponent()
    try filesystem.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appendingPathComponent(".texture-studio-install-\(UUID().uuidString)")
    try filesystem.createDirectory(at: staging, withIntermediateDirectories: false,
                                   attributes: [.posixPermissions: 0o700])
    let backup = staging.appendingPathComponent("previous.app")
    var preserveBackup = false
    defer {
        if exists(backup), preserveBackup {
            print("Previous app retained at \(backup.path); installation could not restore it.")
        } else if exists(backup), !removableTree(backup) {
            print("Previous app retained at \(backup.path); this account cannot remove its files.")
        } else {
            do { try filesystem.removeItem(at: staging) }
            catch { print("Installation temporary files retained at \(staging.path); cleanup failed: \(error)") }
        }
    }
    let staged = staging.appendingPathComponent(destination.lastPathComponent)
    try io.copy(source, staged)
    try io.verifySignature(staged)
    try ensureClosed(destination, processes: io.processes())
    if exists(destination) {
        preserveBackup = true
        try io.move(destination, backup)
    }
    do { try io.move(staged, destination) }
    catch let swapError {
        if exists(backup) {
            do {
                try io.move(backup, destination)
                preserveBackup = false
            } catch let rollbackError {
                throw InstallFailure(description: "Installation failed (\(swapError)); restoring the previous app also failed (\(rollbackError)). Previous app retained at \(backup.path)")
            }
        }
        throw swapError
    }
    preserveBackup = false
    print("Installed \(destination.path)")
}

func isInside(_ path: URL, _ ancestor: URL) -> Bool {
    path.path == ancestor.path || path.path.hasPrefix(ancestor.path + "/")
}

// These checks exercise replacement failure and rollback with disposable files.
func selfTest() throws {
    func require(_ value: Bool, _ message: String) throws {
        guard value else { throw InstallFailure(description: "Installer regression failed: \(message)") }
    }
    func expectsFailure(_ body: () throws -> Void) throws {
        var failed = false
        do { try body() } catch { failed = true }
        try require(failed, "expected failure")
    }
    let root = filesystem.temporaryDirectory.appendingPathComponent("native-install-test-\(UUID().uuidString)")
    try filesystem.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? filesystem.removeItem(at: root) }
    let source = root.appendingPathComponent("Source.app")
    let destination = root.appendingPathComponent("Applications/Texture Studio.app")
    try filesystem.createDirectory(at: source, withIntermediateDirectories: true)
    for (role, name) in [("", "Texture Studio")] + materialTools {
        let app = role.isEmpty ? source : source.appendingPathComponent("Contents/Applications/\(name).app")
        let executable = app.appendingPathComponent("Contents/MacOS/\(name)")
        try filesystem.createDirectory(at: executable.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("fixture".utf8).write(to: executable)
        try filesystem.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        var info = ["CFBundleIdentifier": role.isEmpty ? "org.ipde.texture-studio" : "org.ipde.material-\(role)",
                    "CFBundleExecutable": name, "IPDEBuildConfiguration": "Release"]
        if !role.isEmpty { info["MaterialToolRole"] = role }
        let data = try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
        try data.write(to: app.appendingPathComponent("Contents/Info.plist"))
    }
    try validateBundle(source)
    let recursive = source.appendingPathComponent("Contents/Applications/Material Trainer.app/Contents/Applications")
    try filesystem.createDirectory(at: recursive, withIntermediateDirectories: true)
    try expectsFailure { try validateBundle(source) }
    try filesystem.removeItem(at: recursive)
    let infoURL = source.appendingPathComponent("Contents/Info.plist")
    let releaseInfo = try Data(contentsOf: infoURL)
    var info = try readInfo(source)
    info["IPDEBuildConfiguration"] = "Debug"
    try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0).write(to: infoURL)
    try expectsFailure { try validateBundle(source) }
    try releaseInfo.write(to: infoURL)
    try expectsFailure { try ensureClosed(destination, processes: "  124   \(destination.path)/Contents/Applications/Material Trainer.app/Contents/MacOS/Material Trainer\n") }
    try ensureClosed(destination, processes: "124 \(destination.path).backup/Contents/MacOS/Texture Studio\n")
    try require(!isInside(source, destination) && isInside(destination.appendingPathComponent("child"), destination), "path boundaries")
    try filesystem.createDirectory(at: destination, withIntermediateDirectories: true)
    try Data("previous".utf8).write(to: destination.appendingPathComponent("marker"))
    var io = InstallerIO(processes: { "" }, verifySignature: { _ in }, copy: { try filesystem.copyItem(at: $0, to: $1) })
    io.verifySignature = { _ in throw InstallFailure(description: "fixture signature failure") }
    try expectsFailure { try install(source, destination, io: io) }
    try require(try String(contentsOf: destination.appendingPathComponent("marker"), encoding: .utf8) == "previous", "signature failure preserves old app")
    try require(try filesystem.contentsOfDirectory(atPath: destination.deletingLastPathComponent().path).count == 1, "signature failure cleans staging")
    io.verifySignature = { _ in }
    io.move = { from, to in
        if from != destination, from.lastPathComponent == destination.lastPathComponent { throw InstallFailure(description: "fixture swap failure") }
        try filesystem.moveItem(at: from, to: to)
    }
    try expectsFailure { try install(source, destination, io: io) }
    try require(try String(contentsOf: destination.appendingPathComponent("marker"), encoding: .utf8) == "previous", "failed replacement restores old app")
    try require(try filesystem.contentsOfDirectory(atPath: destination.deletingLastPathComponent().path).count == 1, "successful rollback cleans staging")
    io.move = { from, to in
        if from != destination { throw InstallFailure(description: "fixture replacement and rollback failure") }
        try filesystem.moveItem(at: from, to: to)
    }
    try expectsFailure { try install(source, destination, io: io) }
    try require(!exists(destination), "failed rollback leaves destination absent")
    let retained = try filesystem.contentsOfDirectory(at: destination.deletingLastPathComponent(), includingPropertiesForKeys: nil)
    try require(retained.count == 1, "failed rollback retains one recovery directory")
    try require(try String(contentsOf: retained[0].appendingPathComponent("previous.app/marker"), encoding: .utf8) == "previous", "failed rollback retains previous bytes")
    print("Native installer checks passed")
}

func main() throws {
    var arguments = Array(CommandLine.arguments.dropFirst())
    if arguments == ["--self-test"] { try selfTest(); return }
    guard let sourceArgument = arguments.first, sourceArgument != "--help" else {
        print("usage: install_macos.swift /path/to/Texture Studio.app [--destdir ROOT] [--if-closed]")
        if arguments.isEmpty { throw InstallFailure(description: "A source application is required") }
        return
    }
    arguments.removeFirst()
    var destdir = URL(fileURLWithPath: "/", isDirectory: true)
    var ifClosed = false
    while !arguments.isEmpty {
        let option = arguments.removeFirst()
        if option == "--if-closed" { ifClosed = true }
        else if option == "--destdir", !arguments.isEmpty { destdir = URL(fileURLWithPath: arguments.removeFirst(), isDirectory: true) }
        else { throw InstallFailure(description: "Unknown or incomplete install option: \(option)") }
    }
    let source = URL(fileURLWithPath: sourceArgument, isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
    let destination = destdir.standardizedFileURL.resolvingSymlinksInPath().appendingPathComponent("Applications/Texture Studio.app")
    guard !isInside(source, destination), !isInside(destination, source) else {
        throw InstallFailure(description: "Source and installed application must be separate")
    }
    try validateBundle(source)
    do { try install(source, destination) }
    catch let error as RunningApplication {
        if ifClosed { print("Installation skipped; running application unchanged: \(error)") }
        else { throw error }
    }
}

do { try main() }
catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
