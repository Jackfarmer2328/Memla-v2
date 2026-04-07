import Foundation
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var viewModel: MemlaViewModel
    private let scoutPresets = [
        "find the top 10 github repos for local llms and tell me which best fits weak hardware",
        "find the top 10 github repos for browser agents and tell me which looks strongest",
        "find the top 10 github repos for local coding agents and tell me which is most active",
    ]
    private let followupPresets = [
        "find a youtube video about it then open the first one and summarize it",
        "find a reddit post about it then open the first one and explain it",
        "tell me which source best explains what this project is for",
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    connectionCard
                    missionControlCard
                    chatCard
                    scoutCard
                    stateCard
                    memoryCard
                    actionOntologyCard
                    followupCard
                    activityCard
                }
                .padding(20)
            }
            .navigationTitle("Memla")
            .task {
                await viewModel.refreshHealth()
                await viewModel.refreshState()
                await viewModel.refreshMemory()
                await viewModel.refreshActions()
            }
        }
    }

    private var connectionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Connection")
                .font(.headline)
            TextField("http://192.168.1.10:8080", text: $viewModel.baseURL)
                .textInputAutocapitalization(.never)
                .disableAutocorrection(true)
                .textFieldStyle(.roundedBorder)
            HStack {
                Text("Status: \(viewModel.healthStatus)")
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Use Default") {
                    viewModel.baseURL = "http://192.168.1.10:8080"
                }
                Button("Refresh") {
                    Task {
                        await viewModel.refreshHealth()
                        await viewModel.refreshState()
                        await viewModel.refreshMemory()
                        await viewModel.refreshActions()
                    }
                }
            }
            if !viewModel.errorMessage.isEmpty {
                Text(viewModel.errorMessage)
                    .foregroundStyle(.red)
                    .font(.footnote)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var missionControlCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Mission Control")
                .font(.headline)
            Text("Tap once and Memla will scout repos, follow the winner to YouTube and Reddit, then bring back a final comparison.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Button(viewModel.isLoading ? "Running Mission..." : "Run Local LLM Demo") {
                Task { await viewModel.runLocalLLMDemo() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.isLoading)
            Text("Voice note: iPhone keyboard dictation already works in the prompt fields. Siri can use the existing \"Scout with Memla\" App Intent after install/Shortcuts validation.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(
            LinearGradient(
                colors: [
                    Color.blue.opacity(0.18),
                    Color.teal.opacity(0.10),
                    Color(.secondarySystemBackground),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 20, style: .continuous)
        )
    }

    private var chatCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Conversation")
                    .font(.headline)
                Spacer()
                if !viewModel.chatItems.isEmpty {
                    Button("Clear") {
                        viewModel.clearChat()
                    }
                    .font(.caption)
                }
            }

            if viewModel.chatItems.isEmpty {
                Text("Run the demo or a scout and Memla will write the agent trace here.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                VStack(spacing: 10) {
                    ForEach(viewModel.chatItems) { message in
                        chatBubble(message)
                    }
                }
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var scoutCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Scout")
                .font(.headline)
            TextField("What should Memla scout?", text: $viewModel.scoutPrompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3...6)
            presetChips(items: scoutPresets) { preset in
                viewModel.applyScoutPreset(preset)
            }
            Button(viewModel.isLoading ? "Running..." : "Run Scout") {
                Task { await viewModel.runScout() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.isLoading)

            if let scout = viewModel.scoutResult {
                Text(scout.summary)
                    .font(.subheadline)
                if let best = scout.bestMatch {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Best Match")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(best.title)
                            .font(.headline)
                        HStack(spacing: 8) {
                            if let score = best.score {
                                chip(text: "score \(String(format: "%.2f", score))")
                            }
                            if let stars = best.stars, !stars.isEmpty {
                                chip(text: stars)
                            }
                            if let language = best.language, !language.isEmpty {
                                chip(text: language)
                            }
                        }
                        if let summary = best.summary, !summary.isEmpty {
                            Text(summary)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        Text(best.url)
                            .font(.caption2)
                            .foregroundStyle(.blue)
                    }
                    .padding()
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                }
                if !scout.topResults.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Top Results")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ForEach(scout.topResults.prefix(5)) { item in
                            VStack(alignment: .leading, spacing: 2) {
                                HStack(alignment: .top) {
                                    Text(item.title)
                                        .font(.subheadline.weight(.medium))
                                    Spacer(minLength: 8)
                                    if let score = item.score {
                                        Text(String(format: "%.2f", score))
                                            .font(.caption2.weight(.semibold))
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                HStack(spacing: 8) {
                                    if let stars = item.stars, !stars.isEmpty {
                                        chip(text: stars)
                                    }
                                    if let language = item.language, !language.isEmpty {
                                        chip(text: language)
                                    }
                                }
                                if let summary = item.summary, !summary.isEmpty {
                                    Text(summary)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var stateCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Current State")
                .font(.headline)
            if let state = viewModel.currentState {
                LabeledContent("Page", value: state.pageKind ?? "unknown")
                LabeledContent("Search", value: state.searchQuery ?? "none")
                LabeledContent("Subject", value: state.subjectTitle ?? state.researchSubjectTitle ?? "none")
                if let currentURL = state.currentURL, !currentURL.isEmpty {
                    Text(currentURL)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Text("No browser state yet.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var memoryCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Memory")
                        .font(.headline)
                    Text("Governed operational memory")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Refresh") {
                    Task { await viewModel.refreshMemory() }
                }
                .font(.caption)
            }

            if let memory = viewModel.memorySummary {
                HStack(spacing: 10) {
                    memoryMetric(title: "Episodic", value: "\(memory.episodicCount)")
                    memoryMetric(title: "Semantic", value: "\(memory.semanticCount)")
                    memoryMetric(title: "Rule", value: "\(memory.ruleCount)")
                }
                HStack(spacing: 10) {
                    memoryMetric(title: "Autonomy", value: "\(memory.autonomyCount ?? 0)")
                    memoryMetric(title: "Actions", value: "\(memory.actionCount ?? 0)")
                    memoryMetric(title: "Language", value: "\(memory.languageCount ?? 0)")
                }
                HStack(spacing: 10) {
                    memoryMetric(title: "Active", value: "\(memory.activeCount)")
                    memoryMetric(title: "Stale", value: "\(memory.staleCount)")
                    memoryMetric(title: "Invalid", value: "\(memory.invalidCount)")
                }
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Trust")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.2f", memory.avgTrust))
                            .font(.caption.weight(.semibold))
                    }
                    ProgressView(value: min(max(memory.avgTrust, 0.0), 1.0))
                }
                Text("Memla writes down successful structure, tests reuse, and promotes repeated wins into rules.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("No memory summary loaded yet. Refresh after a scout or follow-up to see the lifecycle state.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var actionOntologyCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Action Ontology")
                        .font(.headline)
                    Text("Safe app-action layer")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Refresh") {
                    Task { await viewModel.refreshActions() }
                }
                .font(.caption)
            }

            if let actions = viewModel.actionSummary {
                HStack(spacing: 10) {
                    memoryMetric(title: "Actions", value: "\(actions.actionCount)")
                    memoryMetric(title: "Ready", value: "\(actions.implementedCount)")
                    memoryMetric(title: "Confirm", value: "\(actions.confirmationRequiredCount)")
                }
                Text(actions.domains.joined(separator: "  "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ForEach(actions.capabilities.prefix(4)) { capability in
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(capability.title)
                                .font(.caption.weight(.semibold))
                            Text("\(capability.domain) - \(capability.status)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(capability.confirmationRequired ? "confirm" : "auto")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(capability.confirmationRequired ? .orange : .green)
                    }
                }
                Text("Money, rides, and messages stay confirmation-gated while Memla learns the action shape.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Divider()
                Text("Draft Action")
                    .font(.caption.weight(.semibold))
                TextField("Ask a contact or draft an email", text: $viewModel.actionPrompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                Button(viewModel.isLoading ? "Drafting..." : "Draft Safely") {
                    Task { await viewModel.draftAction() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isLoading)
                if let draft = viewModel.actionDraft {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(draft.title)
                            .font(.caption.weight(.semibold))
                        Text("\(draft.domain) - \(draft.status) - \(draft.safeNextStep)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        if !draft.recipients.isEmpty {
                            Text("To: \(draft.recipients.joined(separator: ", "))")
                                .font(.caption)
                        }
                        if !draft.subject.isEmpty {
                            Text("Subject: \(draft.subject)")
                                .font(.caption)
                        }
                        if !draft.draftText.isEmpty {
                            Text(draft.draftText)
                                .font(.subheadline)
                                .textSelection(.enabled)
                                .padding(10)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }
                        if !draft.residualConstraints.isEmpty {
                            Text(draft.residualConstraints.joined(separator: ", "))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            } else {
                Text("No action ontology summary loaded yet.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var followupCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Follow-up")
                .font(.headline)
            TextField("What should Memla do next?", text: $viewModel.followupPrompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...5)
            presetChips(items: followupPresets) { preset in
                viewModel.applyFollowupPreset(preset)
            }
            if viewModel.scoutResult?.bestMatch != nil {
                HStack {
                    Button("Use Winner -> YouTube") {
                        viewModel.useWinnerForYouTubeFollowup()
                    }
                    .buttonStyle(.bordered)
                    Button("Use Winner -> Reddit") {
                        viewModel.useWinnerForRedditFollowup()
                    }
                    .buttonStyle(.bordered)
                }
            }
            Button(viewModel.isLoading ? "Running..." : "Run Follow-up") {
                Task { await viewModel.runFollowup() }
            }
            .buttonStyle(.bordered)
            .disabled(viewModel.isLoading)

            if let result = viewModel.followupResult {
                Text("Plan Source: \(result.plan.source)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let execution = result.execution {
                    ForEach(execution.records) { record in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(record.kind)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(record.message)
                                .font(.subheadline)
                        }
                    }
                } else if !result.plan.clarification.isEmpty {
                    Text(result.plan.clarification)
                        .font(.subheadline)
                }
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var activityCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Activity")
                    .font(.headline)
                Spacer()
                if !viewModel.activityItems.isEmpty {
                    Button("Clear") {
                        viewModel.clearActivity()
                    }
                    .font(.caption)
                }
            }
            if viewModel.activityItems.isEmpty {
                Text("Your local scout and follow-up actions will show up here.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(viewModel.activityItems) { item in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(item.title)
                                .font(.subheadline.weight(.medium))
                            Spacer()
                            Text(item.timestamp.formatted(date: .omitted, time: .shortened))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Text(item.detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    @ViewBuilder
    private func presetChips(items: [String], action: @escaping (String) -> Void) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(items, id: \.self) { item in
                    Button {
                        action(item)
                    } label: {
                        Text(item)
                            .lineLimit(1)
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }

    private func chip(text: String) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(.tertiarySystemBackground), in: Capsule())
    }

    private func memoryMetric(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func chatBubble(_ message: MemlaChatMessage) -> some View {
        HStack(alignment: .top) {
            if message.isUser {
                Spacer(minLength: 42)
            }
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text(message.speaker)
                        .font(.caption.weight(.semibold))
                    Spacer(minLength: 8)
                    Text(message.timestamp.formatted(date: .omitted, time: .shortened))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text(message.title)
                    .font(.subheadline.weight(.semibold))
                if !message.detail.isEmpty {
                    Text(message.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .background(
                message.isUser ? Color.accentColor.opacity(0.14) : Color(.secondarySystemBackground),
                in: RoundedRectangle(cornerRadius: 16, style: .continuous)
            )
            if !message.isUser {
                Spacer(minLength: 42)
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(MemlaViewModel())
}
