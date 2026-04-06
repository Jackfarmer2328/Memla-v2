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
                    scoutCard
                    stateCard
                    followupCard
                    activityCard
                }
                .padding(20)
            }
            .navigationTitle("Memla")
            .task {
                await viewModel.refreshHealth()
                await viewModel.refreshState()
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
}

#Preview {
    ContentView()
        .environmentObject(MemlaViewModel())
}
