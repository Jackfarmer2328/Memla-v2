import Foundation

enum MemlaClientError: Error, LocalizedError {
    case invalidBaseURL
    case invalidServerHost(String)
    case badResponse

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Memla base URL is invalid."
        case .invalidServerHost(let detail):
            return detail
        case .badResponse:
            return "Memla returned an unexpected response."
        }
    }
}

actor MemlaClient {
    static let shared = MemlaClient()

    func health(baseURL: String) async throws -> MemlaHealthResponse {
        try await get(path: "/health", baseURL: baseURL)
    }

    func state(baseURL: String) async throws -> MemlaStateEnvelope {
        try await get(path: "/state", baseURL: baseURL)
    }

    func memory(baseURL: String) async throws -> MemlaMemoryEnvelope {
        try await get(path: "/memory", baseURL: baseURL)
    }

    func actions(baseURL: String) async throws -> MemlaActionEnvelope {
        try await get(path: "/actions", baseURL: baseURL)
    }

    func missions(baseURL: String) async throws -> MemlaMissionsEnvelope {
        try await get(path: "/missions", baseURL: baseURL)
    }

    func createMission(prompt: String, baseURL: String) async throws -> MemlaMissionEnvelope {
        try await post(
            path: "/missions",
            baseURL: baseURL,
            payload: MemlaMissionRequest(prompt: prompt)
        )
    }

    func decideMission(missionID: String, decision: String, note: String, baseURL: String) async throws -> MemlaMissionEnvelope {
        let encodedMissionID = missionID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? missionID
        return try await post(
            path: "/missions/\(encodedMissionID)/decision",
            baseURL: baseURL,
            payload: MemlaMissionDecisionRequest(decision: decision, note: note)
        )
    }

    func actionDraft(prompt: String, baseURL: String) async throws -> MemlaActionDraftEnvelope {
        try await post(
            path: "/actions/draft",
            baseURL: baseURL,
            payload: MemlaActionDraftRequest(prompt: prompt)
        )
    }

    func actionCapsule(prompt: String, baseURL: String) async throws -> MemlaActionCapsuleEnvelope {
        try await post(
            path: "/actions/capsule",
            baseURL: baseURL,
            payload: MemlaActionDraftRequest(prompt: prompt)
        )
    }

    func scout(prompt: String, baseURL: String) async throws -> MemlaScoutEnvelope {
        try await post(
            path: "/scout",
            baseURL: baseURL,
            payload: MemlaScoutRequest(prompt: prompt)
        )
    }

    func run(prompt: String, baseURL: String, model: String = "", provider: String = "", heuristicOnly: Bool = false) async throws -> MemlaRunEnvelope {
        try await post(
            path: "/run",
            baseURL: baseURL,
            payload: MemlaFollowupRequest(
                prompt: prompt,
                model: model,
                provider: provider,
                baseURL: "",
                heuristicOnly: heuristicOnly
            )
        )
    }

    func followup(prompt: String, baseURL: String, model: String = "", provider: String = "", heuristicOnly: Bool = false) async throws -> MemlaRunEnvelope {
        try await post(
            path: "/followup",
            baseURL: baseURL,
            payload: MemlaFollowupRequest(
                prompt: prompt,
                model: model,
                provider: provider,
                baseURL: "",
                heuristicOnly: heuristicOnly
            )
        )
    }

    func debugBrowser(payload: MemlaBrowserDebugRequest, baseURL: String) async throws -> MemlaAckEnvelope {
        try await post(
            path: "/debug/browser",
            baseURL: baseURL,
            payload: payload
        )
    }

    private func get<T: Decodable>(path: String, baseURL: String) async throws -> T {
        let request = try URLRequest(url: endpoint(path: path, baseURL: baseURL))
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw MemlaClientError.badResponse
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func post<T: Decodable, Body: Encodable>(path: String, baseURL: String, payload: Body) async throws -> T {
        var request = try URLRequest(url: endpoint(path: path, baseURL: baseURL))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(payload)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw MemlaClientError.badResponse
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func endpoint(path: String, baseURL: String) throws -> URL {
        let trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let base = URL(string: trimmed), let host = base.host?.lowercased() else {
            throw MemlaClientError.invalidBaseURL
        }
        if host == "0.0.0.0" {
            throw MemlaClientError.invalidServerHost("0.0.0.0 is only the server bind address. In Memla on iPhone, use your PC's Wi-Fi IP instead, like http://192.168.1.23:8080.")
        }
        guard let url = URL(string: trimmed + path) else {
            throw MemlaClientError.invalidBaseURL
        }
        return url
    }
}
