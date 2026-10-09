#!/usr/bin/env swift
import Foundation
import Darwin

struct VersionFailure: Error, CustomStringConvertible {
    let description: String
}

func matches(_ pattern: String, _ text: String) throws -> [String] {
    let expression = try NSRegularExpression(pattern: pattern)
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    return expression.matches(in: text, range: range).compactMap { match in
        guard let range = Range(match.range(at: 1), in: text) else { return nil }
        return String(text[range])
    }
}

func main() throws -> Int32 {
    var root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    var notesURL: URL?
    var isPrerelease = false
    var arguments = Array(CommandLine.arguments.dropFirst())
    while !arguments.isEmpty {
        let option = arguments.removeFirst()
        switch option {
        case "--is-prerelease": isPrerelease = true
        case "--root", "--release-notes":
            guard !arguments.isEmpty else { throw VersionFailure(description: "\(option) requires a path") }
            let path = URL(fileURLWithPath: arguments.removeFirst())
            if option == "--root" { root = path } else { notesURL = path }
        case "--help":
            print("usage: check_release_version.swift [--is-prerelease] [--release-notes FILE] [--root REPOSITORY]")
            return 0
        default: throw VersionFailure(description: "Unknown release option: \(option)")
        }
    }
    let version = try String(contentsOf: root.appendingPathComponent("VERSION"), encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines)
    guard version.range(of: #"^[0-9]+\.[0-9]+\.[0-9]+$"#, options: .regularExpression) != nil else {
        throw VersionFailure(description: "Source project must declare one numeric release version")
    }
    let project = try String(contentsOf: root.appendingPathComponent("native/TextureStudio/TextureStudio.xcodeproj/project.pbxproj"), encoding: .utf8)
    let nativeVersions = Set(try matches(#"MARKETING_VERSION = ([0-9.]+);"#, project))
    guard nativeVersions.count == 1, let native = nativeVersions.first else {
        throw VersionFailure(description: "Native target versions disagree: \(nativeVersions.sorted())")
    }
    guard version == native else {
        throw VersionFailure(description: "Version mismatch: VERSION=\(version), native=\(native)")
    }
    let environment = ProcessInfo.processInfo.environment
    let tag = environment["TAG"] ?? ""
    if environment["REF_TYPE"] == "tag" || notesURL != nil {
        guard tag == "v" + version else { throw VersionFailure(description: "Tag \(tag.debugDescription) must match source version v\(version)") }
    }
    let prerelease = version.split(separator: ".").compactMap { Int($0) }.lexicographicallyPrecedes([1, 0, 0])
    if let notesURL {
        var notes = "Texture Studio v\(version)\n\n"
        if prerelease {
            notes += "Development pre-release. Save files, datasets, projects and checkpoints may change incompatibly until 1.0.0. Preserve originals and exports before upgrading.\n\n"
        }
        notes += "Requires an Apple Silicon Mac with macOS 26 or later. Download the arm64 ZIP and move Texture Studio.app to Applications. This SwiftUI app uses Apple image, Metal, Accelerate and native model APIs. Optional models are managed from the Models window.\n\n"
        notes += "This release is ad-hoc signed and is not notarized. macOS may require explicit approval in Privacy & Security on first launch.\n\nSee the bundled manual and https://github.com/gnaservicesinc/ipde for setup and issue reporting.\n"
        try notes.write(to: notesURL, atomically: true, encoding: .utf8)
    }
    if isPrerelease { return prerelease ? 0 : 1 }
    print("Version \(version); \(prerelease ? "pre-release" : "stable release")")
    return 0
}

do { exit(try main()) }
catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
