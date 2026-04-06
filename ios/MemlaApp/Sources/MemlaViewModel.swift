import Foundation

@MainActor
final class MemlaViewModel: ObservableObject {
    @Published var baseURL: String = "http://192.168.1.10:8080"
    @Published var scoutPrompt: String = "find the top 10 github repos for local llms and tell me which best fits weak hardware"
    @Published var followupPrompt: String = "find a youtube video about it then open the first one and summarize it"
    @Published var healthStatus: String = "Unknown"
    @Published var currentState: BrowserState?
    @Published var scoutResult: ScoutResult?
    @Published var followupResult: MemlaRunEnvelope?
    @Published var errorMessage: String = ""
    @Published var isLoading: Bool = false

    func refreshHealth() async {
        await runTask { [self] in
            let response = try await MemlaClient.shared.health(baseURL: self.baseURL)
            self.healthStatus = response.ok ? "Connected" : "Unavailable"
        }
    }

    func refreshState() async {
        await runTask { [self] in
            let response = try await MemlaClient.shared.state(baseURL: self.baseURL)
            self.currentState = response.state
        }
    }

    func runScout() async {
        await runTask { [self] in
            let envelope = try await MemlaClient.shared.scout(prompt: self.scoutPrompt, baseURL: self.baseURL)
            self.scoutResult = envelope.result
            self.followupResult = nil
            self.currentState = try await MemlaClient.shared.state(baseURL: self.baseURL).state
        }
    }

    func runFollowup() async {
        await runTask { [self] in
            let envelope = try await MemlaClient.shared.followup(prompt: self.followupPrompt, baseURL: self.baseURL, heuristicOnly: false)
            self.followupResult = envelope
            self.currentState = try await MemlaClient.shared.state(baseURL: self.baseURL).state
        }
    }

    private func runTask(_ operation: @escaping () async throws -> Void) async {
        isLoading = true
        errorMessage = ""
        defer { isLoading = false }
        do {
            try await operation()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
