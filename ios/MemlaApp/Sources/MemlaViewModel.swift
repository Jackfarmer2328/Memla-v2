import Foundation

struct MemlaActivityItem: Identifiable {
    let id = UUID()
    let title: String
    let detail: String
    let timestamp: Date
}

@MainActor
final class MemlaViewModel: ObservableObject {
    @Published var baseURL: String {
        didSet {
            UserDefaults.standard.set(baseURL, forKey: "memlaBaseURL")
        }
    }
    @Published var scoutPrompt: String = "find the top 10 github repos for local llms and tell me which best fits weak hardware"
    @Published var followupPrompt: String = "find a youtube video about it then open the first one and summarize it"
    @Published var healthStatus: String = "Unknown"
    @Published var currentState: BrowserState?
    @Published var scoutResult: ScoutResult?
    @Published var followupResult: MemlaRunEnvelope?
    @Published var activityItems: [MemlaActivityItem] = []
    @Published var errorMessage: String = ""
    @Published var isLoading: Bool = false

    init() {
        self.baseURL = UserDefaults.standard.string(forKey: "memlaBaseURL") ?? "http://192.168.1.10:8080"
    }

    func refreshHealth() async {
        await runTask { [self] in
            let response = try await MemlaClient.shared.health(baseURL: self.baseURL)
            self.healthStatus = response.ok ? "Connected" : "Unavailable"
            self.appendActivity(title: "Health Check", detail: self.healthStatus)
        }
    }

    func refreshState() async {
        await runTask { [self] in
            let response = try await MemlaClient.shared.state(baseURL: self.baseURL)
            self.currentState = response.state
            self.appendActivity(title: "State Refresh", detail: self.currentState?.pageKind ?? "unknown")
        }
    }

    func runScout() async {
        await runTask { [self] in
            let envelope = try await MemlaClient.shared.scout(prompt: self.scoutPrompt, baseURL: self.baseURL)
            self.scoutResult = envelope.result
            self.followupResult = nil
            self.currentState = try await MemlaClient.shared.state(baseURL: self.baseURL).state
            self.appendActivity(title: "Scout", detail: envelope.result.summary)
        }
    }

    func runFollowup() async {
        await runTask { [self] in
            let envelope = try await MemlaClient.shared.followup(prompt: self.followupPrompt, baseURL: self.baseURL, heuristicOnly: false)
            self.followupResult = envelope
            self.currentState = try await MemlaClient.shared.state(baseURL: self.baseURL).state
            let detail = envelope.execution?.records.first?.message ?? envelope.plan.clarification
            self.appendActivity(title: "Follow-up", detail: detail)
        }
    }

    func applyScoutPreset(_ prompt: String) {
        scoutPrompt = prompt
    }

    func applyFollowupPreset(_ prompt: String) {
        followupPrompt = prompt
    }

    func useWinnerForYouTubeFollowup() {
        if let title = scoutResult?.bestMatch?.title, !title.isEmpty {
            followupPrompt = "find a youtube video about \(title) then open the first one and summarize it"
            return
        }
        followupPrompt = "find a youtube video about it then open the first one and summarize it"
    }

    func useWinnerForRedditFollowup() {
        if let title = scoutResult?.bestMatch?.title, !title.isEmpty {
            followupPrompt = "find a reddit post about \(title) then open the first one and explain it"
            return
        }
        followupPrompt = "find a reddit post about it then open the first one and explain it"
    }

    func clearActivity() {
        activityItems.removeAll()
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

    private func appendActivity(title: String, detail: String) {
        let cleanTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanDetail = detail.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanTitle.isEmpty || !cleanDetail.isEmpty else {
            return
        }
        activityItems.insert(
            MemlaActivityItem(
                title: cleanTitle.isEmpty ? "Memla" : cleanTitle,
                detail: cleanDetail,
                timestamp: Date()
            ),
            at: 0
        )
        if activityItems.count > 12 {
            activityItems = Array(activityItems.prefix(12))
        }
    }
}
