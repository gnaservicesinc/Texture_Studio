import CoreGraphics
import Darwin
import Foundation
import ImageIO
import Observation
import UniformTypeIdentifiers

protocol OllamaHTTPTransport: Sendable {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
    func lines(for request: URLRequest) -> AsyncThrowingStream<String, any Error>
}

/// Talks only to the numeric loopback address. It never starts Ollama, selects a
/// substitute model, or applies a returned proposal to the document.
@MainActor @Observable
final class OllamaDecisionService {
    static let model = MaterialDecision.exactModel
    static let compatibilityNotice = "Requires Ollama 0.40.0 or later for MLX decision models on Apple Silicon. The app checks the exact clef:27b-nvfp4 tag and vision/decision capabilities; it never substitutes a GGUF model."
    private(set) var status: OllamaDecisionStatus = .offline
    private(set) var modelInfo: OllamaModelInfo?
    var lastError: String?
    var isBusy: Bool { status.isBusy }
    var progress: Double? { status.progress }
    var memoryAssessment: OllamaMemoryAssessment {
        memoryOverride ?? OllamaMemoryAssessment(physicalBytes: ProcessInfo.processInfo.physicalMemory,
            availableBytes: Self.availableMemory())
    }
    @ObservationIgnored private let transport: any OllamaHTTPTransport
    @ObservationIgnored private let memoryOverride: OllamaMemoryAssessment?
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var analysisTask: Task<MaterialDecision, any Error>?
    @ObservationIgnored private var qualityTask: Task<MaterialQualityDecision, any Error>?

    init(transport: any OllamaHTTPTransport = LoopbackOllamaTransport(), memoryAssessment: OllamaMemoryAssessment? = nil) {
        self.transport = transport
        self.memoryOverride = memoryAssessment
    }

    func refresh() async {
        guard !isBusy else { return }
        status = .checking
        lastError = nil
        do { try await checkModel() }
        catch { setFailure(error) }
    }

    /// The caller presents the 18 GB download and compatibility notice first.
    func pull() {
        guard !isBusy else { return }
        status = .checking
        lastError = nil
        operation = Task { [weak self] in
            guard let self else { return }
            defer { self.operation = nil }
            do {
                try await self.checkVersion()
                try Task.checkCancellation()
                self.status = .pulling(0)
                var layers: [String: (completed: Double, total: Double)] = [:]
                var succeeded = false
                let request = try Self.request("/api/pull", body: ["model": Self.model, "stream": true])
                for try await line in self.transport.lines(for: request) {
                    try Task.checkCancellation()
                    let object = try JSONDecoder().decode(PullProgress.self, from: Data(line.utf8))
                    if let error = object.error { throw OllamaDecisionError.server(String(error.prefix(2000))) }
                    if let digest = object.digest, let total = object.total, let completed = object.completed {
                        guard total > 0, completed >= 0, completed <= total else {
                            throw OllamaDecisionError.invalidResponse("invalid download progress")
                        }
                        layers[digest] = (Double(completed), Double(total))
                        let all = layers.values.reduce((0.0, 0.0)) { ($0.0 + $1.completed, $0.1 + $1.total) }
                        self.status = .pulling(min(0.99, all.0 / max(1, all.1)))
                    }
                    if object.status == "success" { succeeded = true }
                }
                try Task.checkCancellation()
                guard succeeded else { throw OllamaDecisionError.server("download stopped before completion; retry in Local Models") }
                self.status = .checking
                try await self.checkModel()
            } catch { self.setFailure(error) }
        }
    }

    func cancel() { operation?.cancel(); analysisTask?.cancel(); qualityTask?.cancel() }

    func remove() async throws {
        guard !isBusy else { throw OllamaDecisionError.busy }
        status = .checking
        do {
            _ = try await checkedData(Self.request("/api/delete", method: "DELETE", body: ["model": Self.model]))
            modelInfo = nil
            status = .missing
            lastError = nil
        } catch { setFailure(error); throw error }
    }

    func propose(image: CGImage) async throws -> MaterialDecision {
        guard !isBusy else { throw OllamaDecisionError.busy }
        status = .checking
        lastError = nil
        let task = Task { try await self.performProposal(image: image) }
        analysisTask = task
        defer { analysisTask = nil }
        return try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
    }

    private func performProposal(image: CGImage) async throws -> MaterialDecision {
        do {
            try await checkModel()
            guard case .ready = status else { throw OllamaDecisionError.missing }
            let memory = memoryAssessment
            guard memory.canRun else { throw OllamaDecisionError.memory(memory.message) }
            status = .analysing
            let bounded = try Self.boundedPNG(image)
            let request = try Self.decisionRequest(imageData: bounded)
            let data = try await checkedData(request, decisionEndpoint: true)
            try Task.checkCancellation()
            let decision = try MaterialDecision.decodeSystemOne(data)
            status = .ready
            return decision
        } catch { setFailure(error); throw error }
    }

    func reviewMaterial(diffuse: CGImage, map: CGImage, mapType: String,
                        purpose: MaterialQualityDecision.Purpose) async throws -> MaterialQualityDecision {
        guard !isBusy else { throw OllamaDecisionError.busy }
        guard diffuse.width == map.width, diffuse.height == map.height else {
            throw OllamaDecisionError.invalidResponse("diffuse and map dimensions must match for an aligned quality review")
        }
        status = .checking; lastError = nil
        let task = Task {
            do {
                try await self.checkModel()
                guard case .ready = self.status else { throw OllamaDecisionError.missing }
                guard self.memoryAssessment.canRun else { throw OllamaDecisionError.memory(self.memoryAssessment.message) }
                self.status = .analysing
                let images = try Self.reviewImages(diffuse: diffuse, map: map)
                let request = try Self.qualityReviewRequest(imageData: images, mapType: mapType, purpose: purpose)
                let data = try await self.checkedData(request, decisionEndpoint: true)
                try Task.checkCancellation()
                let decision = try MaterialQualityDecision.decodeSystemOne(data)
                self.status = .ready
                return decision
            } catch { self.setFailure(error); throw error }
        }
        qualityTask = task
        defer { qualityTask = nil }
        return try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
    }

    /// Two overview previews and two aligned native-pixel center crops. The
    /// adviser receives sampled display evidence; no training/source file is
    /// copied or quantized on disk and no full-resolution claim is made.
    static func reviewImages(diffuse: CGImage, map: CGImage) throws -> [Data] {
        guard diffuse.width == map.width, diffuse.height == map.height else {
            throw OllamaDecisionError.invalidResponse("quality review maps must have matching dimensions")
        }
        let width = min(768, diffuse.width), height = min(768, diffuse.height)
        let crop = CGRect(x: (diffuse.width - width) / 2, y: (diffuse.height - height) / 2, width: width, height: height)
        guard let diffuseCrop = diffuse.cropping(to: crop), let mapCrop = map.cropping(to: crop) else {
            throw OllamaDecisionError.invalidResponse("could not prepare aligned detail crops")
        }
        return try [boundedPNG(diffuse), boundedPNG(map), boundedPNG(diffuseCrop), boundedPNG(mapCrop)]
    }

    static func qualityReviewRequest(imageData: [Data], mapType: String,
                                    purpose: MaterialQualityDecision.Purpose) throws -> URLRequest {
        guard imageData.count == 4, imageData.allSatisfy({ !$0.isEmpty && $0.count <= 4_194_304 }),
              ["height", "depth", "roughness", "normal"].contains(mapType) else {
            throw OllamaDecisionError.invalidResponse("quality review requires an aligned diffuse/map pair and detail crops")
        }
        var questions: [String: Any] = [:]
        for (name, criteria) in MaterialQualityDecision.scoreCriteria {
            let instruction: String
            switch name {
            case "detail": instruction = "Rate visible surface detail in the supplied \(mapType) map, using the native-pixel crop. Reward coherent fine structure matching the diffuse texture, not random noise, sharpening halos, or invented detail."
            case "appeal": instruction = "Rate visual material-map quality and coherence, not the beauty or subject of the photographed surface. Correct normal-map colors and valid roughness variation are not artifacts."
            default: instruction = "Rate spatial correspondence between the diffuse map and the \(mapType) map. Material color changes need not imply height or roughness. Do not claim physically measured depth from visual evidence."
            }
            questions[name] = ["type": "score", "instructions": instruction, "criteria": criteria]
        }
        questions["artifact_free"] = ["type": "noul", "instructions": "Is the \(mapType) output free of visible major errors such as seams, double edges, blotches, ringing, invented structures, clipping, or broad lighting copied as false surface shape? Check overview and detail crop."]
        questions["diffuse_ready"] = ["type": "noul", "instructions": "Is the diffuse input sharp and suitable for training or deployed inference: flat surface view, neutral balanced color, even illumination, no broad shadows or specular hotspots? Preserve real material color and texture; do not treat fine surface detail as noise."]
        questions["recommendation"] = ["type": "choice", "instructions": "Recommend a human review action for this \(purpose.rawValue) pair. Prefer further review if any evidence is ambiguous. The recommendation never edits or discards files.", "criteria": MaterialQualityDecision.recommendationCriteria]
        return try request("/v1/systemone", body: ["model": model, "keep_alive": 0,
            "state": ["task": "Assess a diffuse surface texture and its aligned \(mapType) map for a high-detail material workflow.",
                      "purpose": purpose.rawValue,
                      "image_order": ["Diffuse overview (display preview)", "\(mapType) overview (display preview)", "Diffuse center crop at native pixels", "Aligned \(mapType) center crop at native pixels"],
                      "limits": "Overview images may be reduced to 768 pixels; detail crops cover only the center. These are display evidence, not numerical ground truth. Do not infer absolute metric depth, complete-file quality, or correctness of unseen pixels. No denoising, geometry edits, or training decisions are authorized."],
            "images": imageData.map { $0.base64EncodedString() }, "questions": questions])
    }

    private func checkModel() async throws {
        try await checkVersion()
        let tagsData = try await checkedData(Self.request("/api/tags", method: "GET"))
        let tags = try JSONDecoder().decode(Tags.self, from: tagsData)
        guard let installed = tags.models.first(where: { $0.name == Self.model || $0.model == Self.model }) else {
            modelInfo = nil
            status = .missing
            return
        }
        guard installed.remote_model?.isEmpty != false, installed.remote_host?.isEmpty != false else {
            throw OllamaDecisionError.unsupported("the exact tag is linked to a remote model; only local MLX weights are accepted")
        }
        let showData = try await checkedData(Self.request("/api/show", body: ["model": Self.model]))
        let show = try JSONDecoder().decode(Show.self, from: showData)
        let format = show.details?.format ?? "unknown"
        let quantization = show.details?.quantization_level ?? "unknown"
        let capabilities = show.capabilities ?? []
        modelInfo = OllamaModelInfo(name: Self.model, format: format, quantization: quantization,
            capabilities: capabilities, sizeBytes: installed.size ?? 18_000_000_000)
        try Self.validateMetadata(format: format, quantization: quantization, capabilities: capabilities)
        status = .ready
    }

    private func checkVersion() async throws {
        let versionData = try await checkedData(Self.request("/api/version", method: "GET"))
        let version = try JSONDecoder().decode(Version.self, from: versionData).version
        guard Self.supportedVersion(version) else {
            throw OllamaDecisionError.unsupported("MLX decision models require Ollama 0.40.0 or later; this server reports \(version)")
        }
    }

    static func supportedVersion(_ value: String) -> Bool {
        let values = value.split(separator: ".").prefix(3).map { Int($0.prefix(while: { $0.isNumber })) ?? -1 }
        guard values.count == 3, values.allSatisfy({ $0 >= 0 }) else { return false }
        return values.lexicographicallyPrecedes([0, 40, 0]) == false
    }

    static func validateMetadata(format: String, quantization: String, capabilities: [String]) throws {
        guard format.lowercased() == "safetensors", quantization.uppercased().replacingOccurrences(of: "-", with: "").contains("NVFP4"),
              capabilities.contains("vision"), capabilities.contains("decision") else {
            throw OllamaDecisionError.unsupported("the installed tag must report Safetensors, NVFP4, vision, and decision capabilities. Reported: \(format), \(quantization), \(capabilities.joined(separator: ", ")). \(compatibilityNotice)")
        }
    }

    private func checkedData(_ request: URLRequest, decisionEndpoint: Bool = false) async throws -> Data {
        let (data, response) = try await transport.data(for: request)
        guard data.count <= 2_097_152 else { throw OllamaDecisionError.invalidResponse("server response is too large") }
        guard (200..<300).contains(response.statusCode) else {
            let error = (try? JSONDecoder().decode(ServerError.self, from: data).error) ?? "HTTP \(response.statusCode)"
            if decisionEndpoint && [400, 404, 422, 501].contains(response.statusCode) {
                throw OllamaDecisionError.unsupported("\(String(error.prefix(2000))). \(Self.compatibilityNotice)")
            }
            throw OllamaDecisionError.server(String(error.prefix(2000)))
        }
        return data
    }

    private func setFailure(_ error: any Error) {
        if error is CancellationError || (error as? URLError)?.code == .cancelled {
            status = modelInfo == nil ? .missing : .ready
            lastError = nil
        } else if let network = error as? URLError, [.cannotConnectToHost, .networkConnectionLost, .timedOut, .notConnectedToInternet].contains(network.code) {
            status = .offline
            lastError = OllamaDecisionError.offline.localizedDescription
        } else if case OllamaDecisionError.unsupported(let message) = error {
            status = .unsupported(message)
            lastError = error.localizedDescription
        } else {
            status = .failed(error.localizedDescription)
            lastError = error.localizedDescription
        }
    }

    static func request(_ path: String, method: String = "POST", body: [String: Any]? = nil) throws -> URLRequest {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:11434\(path)")!)
        request.httpMethod = method
        request.timeoutInterval = path == "/api/pull" ? 3600 : path == "/v1/systemone" ? 240 : 15
        if let body { request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys]) }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return request
    }

    static func decisionRequest(imageData: Data) throws -> URLRequest {
        guard imageData.count <= 4_194_304 else { throw OllamaDecisionError.invalidResponse("adviser image exceeds its budget") }
        // Fixed JSON preserves option order for stable tie-breaking. System One
        // does not support chat generation controls; do not send fictional seed,
        // temperature, context, or token controls. Zero keep-alive unloads it.
        let body = """
        {"model":"clef:27b-nvfp4","keep_alive":0,"state":"Review this surface photograph as input to a Blender material workflow. Preserve the photographed texture. Choose conservative starting settings. Never infer rotation, perspective, crop, metric height, or a physically measured roughness from this photograph. Lighting means broad lighting imbalances; noise means visible capture noise; relief means apparent surface relief; roughness means a visual starting preset. Confidence means image suitability for this limited task, not probability of physical truth.","images":["\(imageData.base64EncodedString())"],"questions":{
          "lighting":{"type":"choice","instructions":"Which broad lighting correction strength is justified?","criteria":{"none":"Even illumination; avoid correcting material color.","mild":"Mild broad brightness imbalance.","strong":"Obvious broad shadows or hotspots requiring strong correction."}},
          "noise":{"type":"choice","instructions":"Flag visible capture noise for review. Do not recommend denoising, smoothing, or any loss of surface detail.","criteria":{"none":"No visible capture noise; preserve fine texture.","mild":"Some fine random camera noise; compare with native-pixel detail.","moderate":"Obvious capture noise; improve capture or review source quality while preserving real texture."}},
          "relief":{"type":"choice","instructions":"Choose a conservative apparent-relief starting preset, not measured depth.","criteria":{"subtle":"Flat or ambiguous surface; use subtle relief.","medium":"Clearly visible modest relief.","strong":"Pronounced visible surface relief."}},
          "roughness":{"type":"choice","instructions":"Choose an editable visual roughness preset, not a physical measurement.","criteria":{"mixed":"Ambiguous or varied finish.","matte":"Mostly diffuse matte surface.","glossy":"Clear smooth specular glossy finish."}},
          "confidence":{"type":"choice","instructions":"How suitable is this photograph for choosing these limited starting settings?","criteria":{"low":"Blur, strong lighting, mixed objects, or insufficient evidence.","medium":"Usable surface with some ambiguity.","high":"Sharp clearly visible uniform surface with useful lighting cues."}}
        }}
        """
        var request = try Self.request("/v1/systemone")
        request.httpBody = Data(body.utf8)
        return request
    }

    static func boundedPNG(_ image: CGImage) throws -> Data {
        let scale = min(1, 768 / Double(max(image.width, image.height)))
        let width = max(1, Int(Double(image.width) * scale)), height = max(1, Int(Double(image.height) * scale))
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw OllamaDecisionError.invalidResponse("could not prepare the bounded adviser image")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let thumbnail = context.makeImage() else { throw OllamaDecisionError.invalidResponse("could not prepare the adviser image") }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw OllamaDecisionError.invalidResponse("could not encode the adviser image")
        }
        CGImageDestinationAddImage(destination, thumbnail, nil)
        guard CGImageDestinationFinalize(destination) else { throw OllamaDecisionError.invalidResponse("adviser image encoding failed") }
        return data as Data
    }

    private nonisolated static func availableMemory() -> UInt64 {
        var statistics = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &statistics) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        let pages = UInt64(statistics.free_count) + UInt64(statistics.inactive_count) + UInt64(statistics.speculative_count)
        return min(ProcessInfo.processInfo.physicalMemory, pages * UInt64(max(1, sysconf(_SC_PAGESIZE))))
    }

    private struct Version: Decodable { let version: String }
    private struct Tags: Decodable {
        let models: [Entry]
        struct Entry: Decodable {
            let name: String?
            let model: String?
            let size: Int64?
            let remote_model: String?
            let remote_host: String?
        }
    }
    private struct Show: Decodable {
        let details: Details?
        let capabilities: [String]?
        struct Details: Decodable { let format: String?; let quantization_level: String? }
    }
    private struct PullProgress: Decodable {
        let status: String?
        let digest: String?
        let total: Int64?
        let completed: Int64?
        let error: String?
    }
    private struct ServerError: Decodable { let error: String }
}

private struct LoopbackOllamaTransport: OllamaHTTPTransport {
    private let session: URLSession
    init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = ["HTTPEnable": 0, "HTTPSEnable": 0, "SOCKSEnable": 0]
        session = URLSession(configuration: configuration, delegate: NoRedirectDelegate(), delegateQueue: nil)
    }
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        try validate(request)
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw OllamaDecisionError.invalidResponse("non-HTTP response") }
        return (data, response)
    }
    func lines(for request: URLRequest) -> AsyncThrowingStream<String, any Error> {
        AsyncThrowingStream { continuation in
            let worker = Task {
                do {
                    try validate(request)
                    let (bytes, response) = try await session.bytes(for: request)
                    guard let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode) else {
                        throw OllamaDecisionError.server("the local model download request was rejected")
                    }
                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard line.utf8.count <= 65_536 else { throw OllamaDecisionError.invalidResponse("download progress line too large") }
                        if !line.isEmpty { continuation.yield(line) }
                    }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in worker.cancel() }
        }
    }
    private func validate(_ request: URLRequest) throws {
        guard request.url?.scheme == "http", request.url?.host == "127.0.0.1", request.url?.port == 11434 else {
            throw OllamaDecisionError.invalidResponse("only local Ollama at 127.0.0.1:11434 is allowed")
        }
    }
}

private final class NoRedirectDelegate: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
