#!/usr/bin/env swift
import AppKit
import Foundation
import Darwin

struct AppControlFailure: Error, CustomStringConvertible {
    let description: String
}

struct SuiteApplication {
    let name: String
    let identifier: String
    let executable: String
    var bundle: URL { URL(fileURLWithPath: "/Applications/\(name).app", isDirectory: true) }
    var binary: URL { bundle.appendingPathComponent("Contents/MacOS/\(executable)") }
}

let applications = [
    SuiteApplication(name: "Texture Studio", identifier: "org.ipde.texture-studio", executable: "Texture Studio"),
    SuiteApplication(name: "Loom Factory", identifier: "com.gnaservicesinc.LoomFactory", executable: "LoomFactory"),
    SuiteApplication(name: "Loom Workshop", identifier: "com.gnaservicesinc.LoomWorkshop", executable: "LoomWorkshop")
]

func waitFor(_ timeout: TimeInterval, until complete: () -> Bool) -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while !complete(), Date() < deadline {
        RunLoop.current.run(until: Date().addingTimeInterval(0.05))
    }
    return complete()
}

func samePath(_ actual: URL?, _ expected: URL) -> Bool {
    actual?.standardizedFileURL.resolvingSymlinksInPath() == expected.standardizedFileURL.resolvingSymlinksInPath()
}

func sameExecutable(_ running: NSRunningApplication, _ expected: URL) -> Bool {
    samePath(running.executableURL, expected)
}

func descendants(of pid: pid_t, in processList: String) -> [pid_t] {
    let relationships = processList.split(separator: "\n").compactMap { line -> (pid_t, pid_t)? in
        let fields = line.split(maxSplits: 2, whereSeparator: { $0.isWhitespace })
        guard fields.count >= 2, let child = pid_t(fields[0]), let parent = pid_t(fields[1]) else { return nil }
        return (child, parent)
    }
    var parents = Set([pid])
    var result: [pid_t] = []
    while true {
        let children = relationships.filter { parents.contains($0.1) && !parents.contains($0.0) }.map(\.0)
        if children.isEmpty { return result }
        result.append(contentsOf: children)
        parents.formUnion(children)
    }
}

func processList() throws -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/ps")
    process.arguments = ["-axo", "pid=,ppid=,comm="]
    let output = Pipe()
    process.standardOutput = output
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw AppControlFailure(description: "Could not check running project operations; installed apps were left unchanged.")
    }
    return String(decoding: data, as: UTF8.self)
}

func closeInstalledApplications() throws {
    let runningApplications = applications.flatMap { application in
        NSRunningApplication.runningApplications(withBundleIdentifier: application.identifier)
            .filter { sameExecutable($0, application.binary) && !$0.isTerminated }
            .map { (application, $0) }
    }
    // Protect training even when replacing an older app that predates its Cocoa
    // quit guard. Check all apps before sending the first termination request.
    let processes = try processList()
    for (application, running) in runningApplications {
        guard descendants(of: running.processIdentifier, in: processes).isEmpty else {
            throw AppControlFailure(description: "\(application.name) has an active training or project operation. Finish or cancel it, then run make install again. Installed applications were left unchanged.")
        }
    }
    for (application, running) in runningApplications {
        print("Closing \(application.name) for installation.")
        guard running.terminate(), waitFor(15, until: { running.isTerminated }) else {
            throw AppControlFailure(description: "\(application.name) did not quit. Finish or cancel its current work, then run make install again. No application was force-quit or replaced.")
        }
    }
}

func launchInstalledApplications() throws {
    for application in applications {
        guard let bundle = Bundle(url: application.bundle), bundle.bundleIdentifier == application.identifier,
              FileManager.default.isExecutableFile(atPath: application.binary.path) else {
            throw AppControlFailure(description: "The installed \(application.name) application could not be verified at \(application.bundle.path).")
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = application.name == "Loom Factory"
        configuration.allowsRunningApplicationSubstitution = false
        var completed = false
        var launchError: Error?
        var launched: NSRunningApplication?
        NSWorkspace.shared.openApplication(at: application.bundle, configuration: configuration) { running, error in
            // NSWorkspace invokes this completion on a concurrent queue. Keep
            // all state used by the waiting run loop on its main queue.
            DispatchQueue.main.async {
                launched = running
                launchError = error
                completed = true
            }
        }
        guard waitFor(30, until: { completed }) else {
            throw AppControlFailure(description: "Timed out opening \(application.name).")
        }
        if let launchError { throw launchError }
        guard let launched, sameExecutable(launched, application.binary), !launched.isTerminated else {
            throw AppControlFailure(description: "\(application.name) did not start from its installed application.")
        }
        print("Running \(application.name): \(application.bundle.path)")
    }
}

do {
    switch Array(CommandLine.arguments.dropFirst()) {
    case ["close"]: try closeInstalledApplications()
    case ["launch"]: try launchInstalledApplications()
    case ["--self-test"]:
        guard waitFor(0.1, until: { true }), !waitFor(0.01, until: { false }) else {
            throw AppControlFailure(description: "Application control wait checks failed.")
        }
        var callbackDelivered = false
        DispatchQueue.main.async { callbackDelivered = true }
        guard waitFor(1, until: { callbackDelivered }) else {
            throw AppControlFailure(description: "Main queue application launch callback checks failed.")
        }
        let executable = URL(fileURLWithPath: "/Applications/Loom Factory.app/Contents/MacOS/LoomFactory")
        guard samePath(executable, executable.deletingLastPathComponent().appendingPathComponent("./LoomFactory")),
              !samePath(executable, executable.appendingPathExtension("backup")), !samePath(nil, executable) else {
            throw AppControlFailure(description: "Application control path checks failed.")
        }
        let fixture = "100 1 /Applications/Loom Factory.app/Contents/MacOS/LoomFactory\n101 100 /project/runtime/bin/python3\n102 101 /project/runtime/helper\n200 1 /unrelated/python3\n"
        guard Set(descendants(of: 100, in: fixture)) == Set([101, 102]),
              descendants(of: 200, in: fixture).isEmpty, descendants(of: 999, in: fixture).isEmpty else {
            throw AppControlFailure(description: "Active project operation checks failed.")
        }
        print("Suite application control checks passed")
    default:
        throw AppControlFailure(description: "usage: suite_app_control.swift close|launch|--self-test")
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
