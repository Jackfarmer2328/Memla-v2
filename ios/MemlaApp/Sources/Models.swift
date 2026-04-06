import Foundation

struct MemlaScoutRequest: Codable {
    let prompt: String
}

struct MemlaFollowupRequest: Codable {
    let prompt: String
    let model: String
    let provider: String
    let baseURL: String
    let heuristicOnly: Bool

    enum CodingKeys: String, CodingKey {
        case prompt
        case model
        case provider
        case baseURL = "base_url"
        case heuristicOnly = "heuristic_only"
    }
}

struct MemlaHealthResponse: Codable {
    let ok: Bool
    let service: String
    let runtimeDefaults: RuntimeDefaults

    enum CodingKeys: String, CodingKey {
        case ok
        case service
        case runtimeDefaults = "runtime_defaults"
    }
}

struct RuntimeDefaults: Codable {
    let model: String
    let provider: String
    let heuristicOnly: Bool

    enum CodingKeys: String, CodingKey {
        case model
        case provider
        case heuristicOnly = "heuristic_only"
    }
}

struct MemlaStateEnvelope: Codable {
    let ok: Bool
    let state: BrowserState
}

struct BrowserState: Codable {
    let currentURL: String?
    let pageKind: String?
    let searchEngine: String?
    let searchQuery: String?
    let subjectTitle: String?
    let researchSubjectTitle: String?

    enum CodingKeys: String, CodingKey {
        case currentURL = "current_url"
        case pageKind = "page_kind"
        case searchEngine = "search_engine"
        case searchQuery = "search_query"
        case subjectTitle = "subject_title"
        case researchSubjectTitle = "research_subject_title"
    }
}

struct MemlaScoutEnvelope: Codable {
    let ok: Bool
    let mode: String
    let result: ScoutResult
    let totalDurationSeconds: Double

    enum CodingKeys: String, CodingKey {
        case ok
        case mode
        case result
        case totalDurationSeconds = "total_duration_seconds"
    }
}

struct ScoutResult: Codable {
    let prompt: String
    let scoutKind: String
    let source: String
    let ok: Bool
    let query: String
    let goal: String
    let topResults: [ScoutCard]
    let bestMatch: ScoutCard?
    let summary: String

    enum CodingKeys: String, CodingKey {
        case prompt
        case scoutKind = "scout_kind"
        case source
        case ok
        case query
        case goal
        case topResults = "top_results"
        case bestMatch = "best_match"
        case summary
    }
}

struct ScoutCard: Codable, Identifiable {
    var id: String { url.isEmpty ? title : url }

    let title: String
    let url: String
    let summary: String?
    let stars: String?
    let language: String?
    let score: Double?
}

struct MemlaRunEnvelope: Codable {
    let ok: Bool
    let mode: String
    let prompt: String
    let plan: MemlaPlan
    let execution: MemlaExecution?
    let totalDurationSeconds: Double

    enum CodingKeys: String, CodingKey {
        case ok
        case mode
        case prompt
        case plan
        case execution
        case totalDurationSeconds = "total_duration_seconds"
    }
}

struct MemlaPlan: Codable {
    let source: String
    let clarification: String
    let residualConstraints: [String]

    enum CodingKeys: String, CodingKey {
        case source
        case clarification
        case residualConstraints = "residual_constraints"
    }
}

struct MemlaExecution: Codable {
    let ok: Bool
    let records: [MemlaExecutionRecord]
    let browserState: BrowserState?

    enum CodingKeys: String, CodingKey {
        case ok
        case records
        case browserState = "browser_state"
    }
}

struct MemlaExecutionRecord: Codable, Identifiable {
    var id: String { "\(kind)-\(target)-\(message)" }

    let kind: String
    let target: String
    let status: String
    let message: String
}
