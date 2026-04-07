import Foundation
import SwiftUI
import WebKit

struct MemlaBrowserRoute: Identifiable {
    let id = UUID()
    let url: URL
    let capsule: ActionCapsule?
    let option: ActionBridgeOption?
}

struct WebsiteC2AState {
    let pageKind: String
    let summary: String
    let headings: [String]
    let inputs: [String]
    let buttons: [String]
    let links: [String]
    let safeActions: [String]
    let residuals: [String]
    let textSnippet: String
}

final class MemlaBrowserModel: NSObject, ObservableObject, WKNavigationDelegate {
    let webView: WKWebView

    @Published var pageTitle: String = "Loading"
    @Published var currentURL: String = ""
    @Published var isLoading: Bool = false
    @Published var canGoBack: Bool = false
    @Published var canGoForward: Bool = false
    @Published var websiteState: WebsiteC2AState?
    @Published var inspectionStatus: String = ""
    @Published var isInspecting: Bool = false

    override init() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        self.webView = WKWebView(frame: .zero, configuration: configuration)
        super.init()
        self.webView.navigationDelegate = self
    }

    func load(_ url: URL) {
        if webView.url == nil {
            webView.load(URLRequest(url: url))
        }
        syncState()
    }

    func goBack() {
        if webView.canGoBack {
            webView.goBack()
        }
        syncState()
    }

    func goForward() {
        if webView.canGoForward {
            webView.goForward()
        }
        syncState()
    }

    func reload() {
        webView.reload()
        syncState()
    }

    func inspectPage(capsule: ActionCapsule?) {
        isInspecting = true
        inspectionStatus = "Inspecting page..."
        webView.evaluateJavaScript(Self.pageInspectionScript) { [weak self] result, error in
            DispatchQueue.main.async {
                guard let self = self else {
                    return
                }
                self.isInspecting = false
                if let error = error {
                    self.inspectionStatus = error.localizedDescription
                    return
                }
                guard let payload = result as? [String: Any] else {
                    self.inspectionStatus = "Memla could not read this page yet."
                    return
                }
                self.websiteState = Self.buildWebsiteState(payload: payload, capsule: capsule)
                self.inspectionStatus = "Page inspected."
            }
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        syncState()
    }

    func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
        syncState()
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        syncState()
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        syncState()
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        syncState()
    }

    private func syncState() {
        let cleanTitle = webView.title?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let cleanTitle = cleanTitle, !cleanTitle.isEmpty {
            pageTitle = cleanTitle
        } else {
            pageTitle = "Untitled"
        }
        currentURL = webView.url?.absoluteString ?? ""
        isLoading = webView.isLoading
        canGoBack = webView.canGoBack
        canGoForward = webView.canGoForward
    }

    private static func buildWebsiteState(payload: [String: Any], capsule: ActionCapsule?) -> WebsiteC2AState {
        let title = stringValue(payload["title"])
        let url = stringValue(payload["url"])
        let textSnippet = stringValue(payload["text_snippet"])
        let headings = stringArray(payload["headings"])
        let inputs = stringArray(payload["inputs"])
        let buttons = stringArray(payload["buttons"])
        let links = stringArray(payload["links"])
        let combined = ([title, url, textSnippet] + headings + inputs + buttons + links).joined(separator: " ").lowercased()
        let pageKind = classifyPageKind(combined: combined, url: url, inputs: inputs)
        let safeActions = candidateActions(pageKind: pageKind, inputs: inputs, buttons: buttons, links: links)
        let residuals = residualsForPage(pageKind: pageKind, combined: combined, textSnippet: textSnippet, capsule: capsule)
        let summary = "Compiled \(pageKind.replacingOccurrences(of: "_", with: " ")) with \(inputs.count) inputs, \(buttons.count) buttons, and \(links.count) links."
        return WebsiteC2AState(
            pageKind: pageKind,
            summary: summary,
            headings: headings,
            inputs: inputs,
            buttons: buttons,
            links: links,
            safeActions: safeActions,
            residuals: residuals,
            textSnippet: textSnippet
        )
    }

    private static func classifyPageKind(combined: String, url: String, inputs: [String]) -> String {
        if combined.contains("captcha") || combined.contains("verify you are human") {
            return "blocked_or_bot_check"
        }
        if combined.contains("sign in") || combined.contains("log in") || combined.contains("login") {
            return "login"
        }
        if combined.contains("checkout") || combined.contains("place order") || combined.contains("payment") {
            return "checkout"
        }
        if combined.contains("cart") || combined.contains("subtotal") || combined.contains("tip") {
            return "cart"
        }
        if combined.contains("add to cart") || combined.contains("customize") || combined.contains("toppings") || combined.contains("menu") {
            return "menu_or_item"
        }
        if combined.contains("results for") || combined.contains("search results") || url.contains("/search") {
            return "search_results"
        }
        if inputs.contains(where: { $0.lowercased().contains("search") }) {
            return "search_form"
        }
        return "web_page"
    }

    private static func candidateActions(pageKind: String, inputs: [String], buttons: [String], links: [String]) -> [String] {
        var actions: [String] = []
        if pageKind == "search_form" || inputs.contains(where: { $0.lowercased().contains("search") }) {
            actions.append("fill_search_query")
        }
        if pageKind == "search_results" && !links.isEmpty {
            actions.append("select_matching_candidate")
        }
        if pageKind == "menu_or_item" {
            actions.append("review_item_and_modifiers")
        }
        if pageKind == "cart" {
            actions.append("verify_cart_against_capsule")
        }
        if pageKind == "checkout" {
            actions.append("stop_before_purchase")
        }
        if buttons.contains(where: { $0.lowercased().contains("filter") || $0.lowercased().contains("apply") }) {
            actions.append("use_filter_control")
        }
        if actions.isEmpty && (!buttons.isEmpty || !links.isEmpty) {
            actions.append("inspect_visible_candidates")
        }
        return Array(dictUnique(actions))
    }

    private static func residualsForPage(pageKind: String, combined: String, textSnippet: String, capsule: ActionCapsule?) -> [String] {
        var residuals: [String] = []
        if pageKind == "login" {
            residuals.append("login_required")
        }
        if pageKind == "blocked_or_bot_check" {
            residuals.append("bot_check_or_captcha")
        }
        if pageKind == "checkout" {
            residuals.append("irreversible_action_nearby")
        }
        if textSnippet.isEmpty {
            residuals.append("visible_text_empty")
        }
        if let capsule = capsule {
            let restaurant = capsule.slots["restaurant"]?.lowercased() ?? ""
            if !restaurant.isEmpty && !combined.contains(restaurant) {
                residuals.append("target_restaurant_not_visible")
            }
            let item = capsule.slots["item"]?.lowercased() ?? ""
            if !item.isEmpty && !combined.contains(item) {
                residuals.append("target_item_not_visible")
            }
        }
        return Array(dictUnique(residuals))
    }

    private static func stringValue(_ value: Any?) -> String {
        if let value = value as? String {
            return value.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }

    private static func stringArray(_ value: Any?) -> [String] {
        guard let items = value as? [Any] else {
            return []
        }
        return items.compactMap { item in
            guard let text = item as? String else {
                return nil
            }
            let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return clean.isEmpty ? nil : clean
        }
    }

    private static func dictUnique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { value in
            if seen.contains(value) {
                return false
            }
            seen.insert(value)
            return true
        }
    }

    private static let pageInspectionScript = """
    (() => {
      const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
      const visible = (el) => {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 1 && rect.height > 1;
      };
      const collectText = (selector, limit, mapper) => Array.from(document.querySelectorAll(selector))
        .filter(visible)
        .map(mapper)
        .map(clean)
        .filter(Boolean)
        .slice(0, limit);
      const headings = collectText('h1,h2,h3,[role="heading"]', 10, (el) => el.innerText || el.textContent);
      const inputs = collectText('input,textarea,select', 18, (el) => el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.name || el.id || el.type || el.tagName);
      const buttons = collectText('button,[role="button"],input[type="button"],input[type="submit"],a[role="button"]', 24, (el) => el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title'));
      const links = collectText('a[href]', 24, (el) => el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || el.href);
      const text = clean(document.body ? document.body.innerText : '');
      return {
        title: document.title || '',
        url: location.href,
        text_snippet: text.slice(0, 1000),
        headings,
        inputs,
        buttons,
        links
      };
    })();
    """
}

struct MemlaBrowserWebView: UIViewRepresentable {
    @ObservedObject var browser: MemlaBrowserModel

    func makeUIView(context: Context) -> WKWebView {
        browser.webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
    }
}

struct MemlaBrowserView: View {
    let route: MemlaBrowserRoute

    @Environment(\.dismiss) private var dismiss
    @StateObject private var browser = MemlaBrowserModel()
    @State private var verifiedItems: Set<String> = []
    @State private var isCapsuleExpanded = false

    private let commerceChecklist = [
        "restaurant_match",
        "item_match",
        "modifier_match",
        "tip_match",
        "delivery_address_match",
        "total_price_limit",
        "user_checkout_confirmation",
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                capsulePanel
                browserToolbar
                if browser.websiteState != nil || !browser.inspectionStatus.isEmpty {
                    websiteC2APanel
                }
                Divider()
                MemlaBrowserWebView(browser: browser)
                    .ignoresSafeArea(edges: .bottom)
            }
            .navigationTitle("Memla Browser")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                browser.load(route.url)
            }
        }
    }

    private var capsulePanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(route.capsule?.title ?? "Web Bridge")
                        .font(.headline)
                    Text(route.option?.label ?? "Memla-controlled web surface")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if browser.isLoading {
                    ProgressView()
                }
                if route.capsule != nil {
                    Button(isCapsuleExpanded ? "Hide" : "Plan") {
                        isCapsuleExpanded.toggle()
                    }
                    .buttonStyle(.bordered)
                    .font(.caption)
                }
            }

            if let capsule = route.capsule {
                if !capsule.slots.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(capsule.slots.keys.sorted(), id: \.self) { key in
                                if let value = capsule.slots[key], !value.isEmpty {
                                    capsuleChip(title: key.replacingOccurrences(of: "_", with: " ").capitalized, value: value)
                                }
                            }
                        }
                    }
                }
                if isCapsuleExpanded {
                    Text(capsule.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                    verificationChecklist(for: capsule)
                } else {
                    Text("Tap Plan for checklist and blockers. Stop before the final purchase/send action.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            } else {
                Text("Memla is keeping this web path inside its own browser surface so future page-state checks can attach here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(Color(.secondarySystemBackground))
    }

    private var browserToolbar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Button {
                    browser.goBack()
                } label: {
                    Image(systemName: "chevron.left")
                }
                .disabled(!browser.canGoBack)

                Button {
                    browser.goForward()
                } label: {
                    Image(systemName: "chevron.right")
                }
                .disabled(!browser.canGoForward)

                Button {
                    browser.reload()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }

                Button(browser.isInspecting ? "Inspecting..." : "Inspect") {
                    browser.inspectPage(capsule: route.capsule)
                }
                .disabled(browser.isInspecting || browser.isLoading)

                Spacer()

                if let url = URL(string: browser.currentURL), !browser.currentURL.isEmpty {
                    ShareLink(item: url) {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
            }
            Text(browser.pageTitle)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
            if !browser.currentURL.isEmpty {
                Text(browser.currentURL)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color(.systemBackground))
    }

    private var websiteC2APanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Website C2A")
                    .font(.caption.weight(.semibold))
                Spacer()
                if !browser.inspectionStatus.isEmpty {
                    Text(browser.inspectionStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            if let state = browser.websiteState {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        capsuleChip(title: "Page", value: readableRequirement(state.pageKind))
                        capsuleChip(title: "Inputs", value: "\(state.inputs.count)")
                        capsuleChip(title: "Buttons", value: "\(state.buttons.count)")
                        capsuleChip(title: "Links", value: "\(state.links.count)")
                    }
                }
                Text(state.summary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                if !state.safeActions.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(Array(state.safeActions.prefix(6)), id: \.self) { action in
                                Label(readableRequirement(action), systemImage: "checkmark.shield")
                                    .font(.caption2)
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 6)
                                    .background(Color.green.opacity(0.12), in: Capsule())
                            }
                        }
                    }
                }
                if !state.residuals.isEmpty {
                    Text("Residuals: \(state.residuals.map { readableRequirement($0) }.joined(separator: ", "))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                if !state.headings.isEmpty {
                    Text("Headings: \(state.headings.prefix(3).joined(separator: " / "))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
    }

    @ViewBuilder
    private func verificationChecklist(for capsule: ActionCapsule) -> some View {
        let requirements = capsule.verifierRequirements.isEmpty ? commerceChecklist : capsule.verifierRequirements
        VStack(alignment: .leading, spacing: 6) {
            Text("Verify Before Final Action")
                .font(.caption.weight(.semibold))
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(requirements.prefix(7), id: \.self) { item in
                        Button {
                            toggleVerified(item)
                        } label: {
                            Label(readableRequirement(item), systemImage: verifiedItems.contains(item) ? "checkmark.circle.fill" : "circle")
                                .font(.caption2)
                        }
                        .buttonStyle(.bordered)
                        .tint(verifiedItems.contains(item) ? .green : .gray)
                    }
                }
            }
            if !capsule.autoSubmitBlockers.isEmpty {
                Text("Auto-submit blocked: \(capsule.autoSubmitBlockers.prefix(3).joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
    }

    private func capsuleChip(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func toggleVerified(_ item: String) {
        if verifiedItems.contains(item) {
            verifiedItems.remove(item)
        } else {
            verifiedItems.insert(item)
        }
    }

    private func readableRequirement(_ value: String) -> String {
        value.replacingOccurrences(of: "_", with: " ").capitalized
    }
}
