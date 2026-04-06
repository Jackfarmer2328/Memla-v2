import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var viewModel: MemlaViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    connectionCard
                    scoutCard
                    stateCard
                    followupCard
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
                                Text(item.title)
                                    .font(.subheadline.weight(.medium))
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
}

#Preview {
    ContentView()
        .environmentObject(MemlaViewModel())
}
