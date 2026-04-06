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
        await runTask {
            let response = try await MemlaClient.shared.health(baseURL: baseURL)
            healthStatus = response.ok ? "Connected" : "Unavailable"
        }
    }

    func refreshState() async {
        await runTask {
            let response = try await MemlaClient.shared.state(baseURL: baseURL)
            currentState = response.state
        }
    }

    func runScout() async {
        await runTask {
            let envelope = try await MemlaClient.shared.scout(prompt: scoutPrompt, baseURL: baseURL)
            scoutResult = envelope.result
            followupResult = nil
            currentState = try await MemlaClient.shared.state(baseURL: baseURL).state
        }
    }

    func runFollowup() async {
        await runTask {
            let envelope = try await MemlaClient.shared.followup(prompt: followupPrompt, baseURL: baseURL, heuristicOnly: false)
            followupResult = envelope
            currentState = try await MemlaClient.shared.state(baseURL: baseURL).state
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
