import Foundation

enum MemlaClientError: Error, LocalizedError {
    case invalidBaseURL
    case badResponse

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Memla base URL is invalid."
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

    func scout(prompt: String, baseURL: String) async throws -> MemlaScoutEnvelope {
        try await post(
            path: "/scout",
            baseURL: baseURL,
            payload: MemlaScoutRequest(prompt: prompt)
        )
    }

    func followup(prompt: String, baseURL: String, model: String = "", provider: String = "ollama", heuristicOnly: Bool = false) async throws -> MemlaRunEnvelope {
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
        guard let url = URL(string: trimmed + path) else {
            throw MemlaClientError.invalidBaseURL
        }
        return url
    }
}
