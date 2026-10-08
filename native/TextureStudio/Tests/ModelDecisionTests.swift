import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import TextureStudio

@MainActor
final class ModelDecisionTests: XCTestCase {
    private static func response(model: String = MaterialDecision.exactModel, change: (inout [String: Any]) -> Void = { _ in }) throws -> Data {
        var answers: [String: Any] = [:]
        for (key, choices) in MaterialDecision.allowedChoices {
            let choice = choices[0]
            answers[key] = ["type": "choice", "choice": choice,
                "probabilities": Dictionary(uniqueKeysWithValues: choices.map { ($0, $0 == choice ? 1.0 : 0.0) }), "confidence": 1.0]
        }
        change(&answers)
        return try JSONSerialization.data(withJSONObject: ["model": model, "answers": answers, "usage": ["input_tokens": 100, "output_tokens": 0]])
    }

    private func image(width: Int = 16, height: Int = 16) throws -> CGImage {
        let context = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 0.4, green: 0.3, blue: 0.2, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return try XCTUnwrap(context.makeImage())
    }

    func testFixedProposalSchemaKeepsReviewableChoices() throws {
        let decision = try MaterialDecision.decodeSystemOne(Self.response())
        XCTAssertEqual(decision.model, "clef:27b-nvfp4")
        XCTAssertEqual(decision.lighting, .none)
        XCTAssertEqual(decision.noise, .none)
        XCTAssertEqual(decision.relief, .subtle)
        XCTAssertTrue(decision.rationale.count <= 2000)
    }

    func testWrongModelAndUnknownActionsAreRejected() throws {
        XCTAssertThrowsError(try MaterialDecision.decodeSystemOne(Self.response(model: "clef:27b-q4_K_M")))
        let response = try Self.response { answers in
            answers["lighting"] = ["type": "choice", "choice": "run-shell", "probabilities": ["run-shell": 1.0], "confidence": 1.0]
        }
        XCTAssertThrowsError(try MaterialDecision.decodeSystemOne(response))
        XCTAssertThrowsError(try MaterialDecision.decodeSystemOne(Self.response { $0["rotation"] = ["choice": "7.93"] }))
        XCTAssertThrowsError(try MaterialDecision.decodeSystemOne(Self.response { $0.removeValue(forKey: "noise") }))
    }

    func testInvalidProbabilitiesAndOversizedResponseAreRejected() throws {
        let response = try Self.response { answers in
            answers["noise"] = ["type": "choice", "choice": "none", "probabilities": ["none": 1.7, "mild": -0.7, "moderate": 0.0], "confidence": 2.0]
        }
        XCTAssertThrowsError(try MaterialDecision.decodeSystemOne(response))
        XCTAssertThrowsError(try MaterialDecision.decodeSystemOne(Data(repeating: 32, count: 1_048_577)))
        var largeContext = try JSONSerialization.jsonObject(with: Self.response()) as! [String: Any]
        largeContext["usage"] = ["input_tokens": 4097, "output_tokens": 0]
        XCTAssertThrowsError(try MaterialDecision.decodeSystemOne(JSONSerialization.data(withJSONObject: largeContext)))
    }

    func testMLXRequires040AndRejectsGGUFFallbackMetadata() throws {
        XCTAssertFalse(OllamaDecisionService.supportedVersion("0.35.1"))
        XCTAssertFalse(OllamaDecisionService.supportedVersion("0.39.9"))
        XCTAssertFalse(OllamaDecisionService.supportedVersion("unknown"))
        XCTAssertTrue(OllamaDecisionService.supportedVersion("0.40.0"))
        XCTAssertTrue(OllamaDecisionService.supportedVersion("1.0.0"))
        XCTAssertThrowsError(try OllamaDecisionService.validateMetadata(format: "gguf", quantization: "Q4_K_M", capabilities: ["vision", "decision"]))
        XCTAssertThrowsError(try OllamaDecisionService.validateMetadata(format: "safetensors", quantization: "NVFP4", capabilities: ["vision", "completion"]))
        XCTAssertNoThrow(try OllamaDecisionService.validateMetadata(format: "safetensors", quantization: "NVFP4", capabilities: ["vision", "decision"]))
    }

    func testBoundedImageAndRepeatableSystemOneRequest() throws {
        let data = try OllamaDecisionService.boundedPNG(image(width: 2048, height: 1024))
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let bounded = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(bounded.width, 768); XCTAssertEqual(bounded.height, 384)
        let request = try OllamaDecisionService.decisionRequest(imageData: data)
        XCTAssertEqual(request.url?.host, "127.0.0.1")
        XCTAssertEqual(request.url?.path, "/v1/systemone")
        XCTAssertEqual(request.httpBody, try OllamaDecisionService.decisionRequest(imageData: data).httpBody)
        let body = try XCTUnwrap(try JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody)) as? [String: Any])
        XCTAssertEqual(body["model"] as? String, "clef:27b-nvfp4")
        XCTAssertEqual(body["keep_alive"] as? Int, 0)
        XCTAssertNil(body["options"])
        XCTAssertNil(body["tools"])
    }

    func testOfflineAndMissingExactModelAreRecoverable() async throws {
        let offline = OllamaDecisionService(transport: DecisionTransport(offline: true))
        await offline.refresh()
        XCTAssertEqual(offline.status, .offline)
        let missing = OllamaDecisionService(transport: DecisionTransport(missing: true))
        await missing.refresh()
        XCTAssertEqual(missing.status, .missing)
        XCTAssertNil(missing.modelInfo)
    }

    func testValidatedLocalDecisionAndInsufficientMemoryGuard() async throws {
        let transport = DecisionTransport(decision: try Self.response())
        let adequate = OllamaMemoryAssessment(physicalBytes: 64 * 1_073_741_824, availableBytes: 40 * 1_073_741_824)
        let service = OllamaDecisionService(transport: transport, memoryAssessment: adequate)
        let decision = try await service.propose(image: image())
        XCTAssertEqual(service.status, .ready)
        XCTAssertEqual(service.modelInfo?.format, "safetensors")
        XCTAssertEqual(decision.model, MaterialDecision.exactModel)
        let lowMemory = OllamaDecisionService(transport: transport,
            memoryAssessment: OllamaMemoryAssessment(physicalBytes: 16 * 1_073_741_824, availableBytes: 10 * 1_073_741_824))
        do { _ = try await lowMemory.propose(image: image()); XCTFail("Memory guard was bypassed") }
        catch { guard case OllamaDecisionError.memory = error else { return XCTFail("Wrong memory error: \(error)") } }
    }

    func testRunnerRejectionReportsUnsupportedAndNeverChangesModel() async throws {
        let service = OllamaDecisionService(transport: DecisionTransport(rejectRunner: true),
            memoryAssessment: OllamaMemoryAssessment(physicalBytes: 64 * 1_073_741_824, availableBytes: 40 * 1_073_741_824))
        do { _ = try await service.propose(image: image()); XCTFail("Runner rejection was accepted") }
        catch { guard case .unsupported = service.status else { return XCTFail("Expected incompatibility status") } }
        XCTAssertEqual(service.modelInfo?.name, MaterialDecision.exactModel)
    }

    func testRemoteModelAliasIsRejectedBeforeSendingAPhoto() async throws {
        let transport = DecisionTransport(remoteAlias: true)
        let service = OllamaDecisionService(transport: transport)
        await service.refresh()
        guard case .unsupported = service.status else { return XCTFail("A remote alias was accepted as local") }
        XCTAssertNil(service.modelInfo)
    }

    func testPullCancellationAndExactTagRemoval() async throws {
        let transport = DecisionTransport(slowPull: true)
        let service = OllamaDecisionService(transport: transport)
        service.pull()
        try await Task.sleep(for: .milliseconds(20))
        service.cancel()
        for _ in 0..<100 { if !service.isBusy { break }; try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(service.isBusy)
        XCTAssertNil(service.lastError)
        await service.refresh()
        try await service.remove()
        XCTAssertEqual(service.status, .missing)
        let deletion = await transport.lastDelete
        XCTAssertEqual(deletion, MaterialDecision.exactModel)
    }
}

private actor DecisionTransport: OllamaHTTPTransport {
    let offline: Bool
    let missing: Bool
    let decision: Data
    let rejectRunner: Bool
    let slowPull: Bool
    let remoteAlias: Bool
    private(set) var lastDelete: String?
    init(offline: Bool = false, missing: Bool = false, decision: Data = Data(), rejectRunner: Bool = false, slowPull: Bool = false, remoteAlias: Bool = false) {
        self.offline = offline; self.missing = missing; self.decision = decision; self.rejectRunner = rejectRunner; self.slowPull = slowPull; self.remoteAlias = remoteAlias
    }
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        if offline { throw URLError(.cannotConnectToHost) }
        let object: [String: Any]
        var status = 200
        switch request.url!.path {
        case "/api/version": object = ["version": "0.40.0"]
        case "/api/tags":
            var tag: [String: Any] = ["name": missing ? "clef:27b-q4_K_M" : MaterialDecision.exactModel, "size": 18_000_000_000]
            if remoteAlias { tag["remote_model"] = "clef-cloud"; tag["remote_host"] = "https://ollama.com" }
            object = ["models": [tag]]
        case "/api/show": object = ["details": ["format": "safetensors", "quantization_level": "NVFP4"], "capabilities": ["vision", "decision"]]
        case "/v1/systemone":
            if !rejectRunner { return (decision, HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!) }
            object = ["error": "runner does not support this model"]; status = 400
        case "/api/delete":
            let body = try JSONSerialization.jsonObject(with: request.httpBody!) as! [String: Any]
            lastDelete = body["model"] as? String
            object = [:]
        default: throw OllamaDecisionError.server("unexpected mock path")
        }
        return (try JSONSerialization.data(withJSONObject: object), HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!)
    }
    nonisolated func lines(for request: URLRequest) -> AsyncThrowingStream<String, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    if slowPull { try await Task.sleep(for: .seconds(3)) }
                    continuation.yield("{\"status\":\"pulling\",\"digest\":\"fixture\",\"total\":100,\"completed\":100}")
                    continuation.yield("{\"status\":\"success\"}")
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
