import Foundation

/// An advisory visual assessment. The stored maps and curation state are never
/// changed by this response; the reviewer explicitly confirms a recommendation.
struct MaterialQualityDecision: Identifiable, Sendable, Equatable {
    enum Purpose: String, Sendable { case result, dataset }
    enum Recommendation: String, CaseIterable, Sendable {
        case review, approve, exclude
        var title: String {
            switch self {
            case .review: "Review further"
            case .approve: "Recommend approval"
            case .exclude: "Recommend exclusion"
            }
        }
    }
    let id: UUID
    let model: String
    let detail: Double
    let appeal: Double
    let alignment: Double
    let artifactFree: Double
    let diffuseReady: Double
    let recommendation: Recommendation
    let recommendationConfidence: Double

    static let scoreCriteria: [String: [String]] = [
        "detail": ["No usable surface detail", "Little detail; visibly blurred", "Moderate surface detail", "Sharp detail in most areas", "Very fine, sharp, coherent surface detail"],
        "appeal": ["Visually unusable material map", "Distracting visual defects", "Usable but uneven material quality", "Clean, convincing material texture", "Exceptionally clean and coherent material texture"],
        "alignment": ["Map contradicts the diffuse texture", "Frequent displaced or invented features", "Some alignment but ambiguous features", "Most map features follow the diffuse texture", "Fine map features consistently align with the diffuse texture"]
    ]
    static let recommendationCriteria: [String: String] = [
        "review": "Ambiguous evidence, lighting problems, or uncertain detail; ask the reviewer to inspect further.",
        "approve": "Sharp aligned surface detail, clean diffuse input, and no visible major artifacts; recommend manual approval.",
        "exclude": "Obvious major artifacts, severe blur, misalignment, or unusable diffuse input; recommend manual exclusion."
    ]

    static func decodeSystemOne(_ data: Data) throws -> MaterialQualityDecision {
        guard data.count <= 1_048_576 else { throw OllamaDecisionError.invalidResponse("quality response exceeds its budget") }
        let response = try JSONDecoder().decode(Response.self, from: data)
        guard response.model == MaterialDecision.exactModel,
              (0...16384).contains(response.usage.input_tokens), (0...1024).contains(response.usage.output_tokens),
              Set(response.answers.keys) == Set(["detail", "appeal", "alignment", "artifact_free", "diffuse_ready", "recommendation"]) else {
            throw OllamaDecisionError.invalidResponse("quality review model, usage, or questions do not match the request")
        }
        var scores: [String: Double] = [:]
        for (question, criteria) in scoreCriteria {
            guard let answer = response.answers[question], answer.type == "score", let score = answer.score,
                  score.isFinite, (0...4).contains(score), let probabilities = answer.probabilities,
                  validProbabilities(probabilities, keys: Set((0..<criteria.count).map(String.init))),
                  let confidence = answer.confidence, confidence.isFinite, (0...1).contains(confidence),
                  let legend = answer.legend,
                  legend == Dictionary(uniqueKeysWithValues: criteria.enumerated().map { (String($0.offset), $0.element) }) else {
                throw OllamaDecisionError.invalidResponse("invalid \(question) quality score")
            }
            let weighted = probabilities.reduce(0.0) { $0 + Double(Int($1.key)!) * $1.value }
            guard abs(score - weighted) <= 0.03 else { throw OllamaDecisionError.invalidResponse("inconsistent \(question) score") }
            scores[question] = score
        }
        func probability(_ question: String) throws -> Double {
            guard let answer = response.answers[question], answer.type == "noul", let value = answer.noul,
                  value.isFinite, (0...1).contains(value) else {
                throw OllamaDecisionError.invalidResponse("invalid \(question) probability")
            }
            return value
        }
        guard let answer = response.answers["recommendation"], answer.type == "choice",
              let choice = answer.choice, let recommendation = Recommendation(rawValue: choice),
              let probabilities = answer.probabilities,
              validProbabilities(probabilities, keys: Set(recommendationCriteria.keys)),
              let confidence = answer.confidence, confidence.isFinite, (0...1).contains(confidence) else {
            throw OllamaDecisionError.invalidResponse("invalid quality recommendation")
        }
        // A concentrated distribution is not a correctness guarantee. Ambiguous
        // recommendations remain in the review queue even if a tie picks approve.
        let ordered = probabilities.values.sorted(by: >)
        let proposed: Recommendation = ordered[0] - ordered[1] < 0.1 ? .review : recommendation
        return MaterialQualityDecision(id: UUID(), model: response.model, detail: scores["detail"]!,
            appeal: scores["appeal"]!, alignment: scores["alignment"]!,
            artifactFree: try probability("artifact_free"), diffuseReady: try probability("diffuse_ready"),
            recommendation: proposed, recommendationConfidence: confidence)
    }

    private static func validProbabilities(_ probabilities: [String: Double], keys: Set<String>) -> Bool {
        Set(probabilities.keys) == keys && probabilities.values.allSatisfy { $0.isFinite && (0...1).contains($0) }
            && abs(probabilities.values.reduce(0, +) - 1) <= 0.02
    }
    private struct Response: Decodable {
        let model: String
        let answers: [String: Answer]
        let usage: Usage
        struct Usage: Decodable { let input_tokens: Int; let output_tokens: Int }
        struct Answer: Decodable {
            let type: String
            let score: Double?
            let legend: [String: String]?
            let probabilities: [String: Double]?
            let confidence: Double?
            let noul: Double?
            let choice: String?
        }
    }
}
