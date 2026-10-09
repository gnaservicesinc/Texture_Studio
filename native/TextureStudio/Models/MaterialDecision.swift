import Foundation

/// A fixed, reviewable proposal. It cannot carry commands, file paths, or edits.
struct MaterialDecision: Codable, Identifiable, Sendable, Equatable {
    enum Lighting: String, Codable, CaseIterable, Sendable { case none, mild, strong }
    enum Noise: String, Codable, CaseIterable, Sendable { case none, mild, moderate }
    enum Relief: String, Codable, CaseIterable, Sendable { case subtle, medium, strong }
    enum Roughness: String, Codable, CaseIterable, Sendable { case matte, mixed, glossy }
    enum Confidence: String, Codable, CaseIterable, Sendable { case low, medium, high }
    let id: UUID
    let model: String
    let lighting: Lighting
    let noise: Noise
    let relief: Relief
    let roughness: Roughness
    let confidence: Confidence
    let rationale: String

    static let exactModel = "clef:27b-nvfp4"
    static let allowedChoices: [String: [String]] = [
        "lighting": Lighting.allCases.map(\.rawValue), "noise": Noise.allCases.map(\.rawValue),
        "relief": Relief.allCases.map(\.rawValue), "roughness": Roughness.allCases.map(\.rawValue),
        "confidence": Confidence.allCases.map(\.rawValue)
    ]

    static func decodeSystemOne(_ data: Data) throws -> MaterialDecision {
        guard data.count <= 1_048_576 else { throw OllamaDecisionError.invalidResponse("response exceeds the decision budget") }
        let response = try JSONDecoder().decode(SystemOneResponse.self, from: data)
        guard response.model == exactModel else { throw OllamaDecisionError.invalidResponse("the server used a different model") }
        guard (0...4096).contains(response.usage.input_tokens), (0...1024).contains(response.usage.output_tokens) else {
            throw OllamaDecisionError.invalidResponse("the decision exceeded the bounded token budget")
        }
        guard Set(response.answers.keys) == Set(allowedChoices.keys) else {
            throw OllamaDecisionError.invalidResponse("the response did not answer exactly the material questions")
        }
        for (question, choices) in allowedChoices {
            guard let answer = response.answers[question], answer.type == "choice", choices.contains(answer.choice),
                  answer.confidence.isFinite, (0...1).contains(answer.confidence),
                  Set(answer.probabilities.keys) == Set(choices),
                  answer.probabilities.values.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
                  abs(answer.probabilities.values.reduce(0, +) - 1) <= 0.02 else {
                throw OllamaDecisionError.invalidResponse("invalid \(question) choice or probabilities")
            }
        }
        let lighting = Lighting(rawValue: response.answers["lighting"]!.choice)!
        let noise = Noise(rawValue: response.answers["noise"]!.choice)!
        let relief = Relief(rawValue: response.answers["relief"]!.choice)!
        let roughness = Roughness(rawValue: response.answers["roughness"]!.choice)!
        let confidence = Confidence(rawValue: response.answers["confidence"]!.choice)!
        let rationale = "The local model suggests \(lighting.rawValue) lighting correction, identifies \(noise.rawValue) capture noise, and proposes \(relief.rawValue) relief and \(roughness.rawValue) roughness. Image suitability: \(confidence.rawValue). Preserve fine texture; capture noise is a review flag, not permission to blur the image. A single image cannot determine physical roughness or relief."
        guard rationale.count <= 2000 else { throw OllamaDecisionError.invalidResponse("rationale exceeds its limit") }
        return MaterialDecision(id: UUID(), model: exactModel, lighting: lighting, noise: noise,
            relief: relief, roughness: roughness, confidence: confidence, rationale: rationale)
    }

    private struct SystemOneResponse: Decodable {
        let model: String
        let answers: [String: Answer]
        let usage: Usage
        struct Usage: Decodable { let input_tokens: Int; let output_tokens: Int }
        struct Answer: Decodable {
            let type: String
            let choice: String
            let probabilities: [String: Double]
            let confidence: Double
        }
    }
}

enum OllamaDecisionError: LocalizedError {
    case offline
    case missing
    case unsupported(String)
    case invalidResponse(String)
    case server(String)
    case busy
    var errorDescription: String? {
        switch self {
        case .offline: "Ollama is not running at 127.0.0.1:11434. Open Ollama and check again, or continue without the adviser."
        case .missing: "The optional clef:27b-nvfp4 model is missing. Download this exact MLX model in Local Models or continue without it."
        case .unsupported(let message): "This Ollama setup cannot run the requested MLX vision decision model: \(message)"
        case .invalidResponse(let message): "The model proposal was rejected: \(message). No settings changed."
        case .server(let message): "Ollama: \(message)"
        case .busy: "An Ollama operation is already running. Cancel or wait for it to finish."
        }
    }
}

enum OllamaDecisionStatus: Equatable {
    case offline, missing, ready, checking, analysing
    case pulling(Double)
    case unsupported(String), failed(String)
    var isBusy: Bool { switch self { case .checking, .analysing, .pulling: true; default: false } }
    var progress: Double? { if case .pulling(let value) = self { value } else { nil } }
    var message: String {
        switch self {
        case .offline: "Ollama is not running. Open it and check again."
        case .missing: "Exact MLX NVFP4 model missing. Optional download: about 18 GB."
        case .ready: "MLX NVFP4 vision decision model ready locally"
        case .checking: "Checking local Ollama and model format…"
        case .analysing: "Reviewing the supplied image evidence locally…"
        case .pulling(let value): "Downloading \(Int(value * 100))%"
        case .unsupported(let value), .failed(let value): value
        }
    }
}

struct OllamaModelInfo: Sendable, Equatable {
    let name: String
    let format: String
    let quantization: String
    let capabilities: [String]
    let sizeBytes: Int64
}
