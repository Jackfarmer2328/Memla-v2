import Foundation
import SwiftUI
import UIKit
import WebKit

struct MemlaBrowserRoute: Identifiable {
    let id = UUID()
    let url: URL
    let capsule: ActionCapsule?
    let option: ActionBridgeOption?
}

struct WebsiteC2ACandidate: Identifiable {
    let id: String
    let domIndex: Int
    let label: String
    let url: String
    let kind: String
    let score: Double
    let matchedTerms: [String]
    let blocked: Bool
    let tapSafety: String
    let tapReason: String
    let reason: String
}

struct WebsiteC2AState {
    let pageKind: String
    let summary: String
    let headings: [String]
    let inputs: [String]
    let buttons: [String]
    let links: [String]
    let candidates: [WebsiteC2ACandidate]
    let safeActions: [String]
    let residuals: [String]
    let textSnippet: String
}

struct WebsiteBridgeSuggestion: Identifiable {
    let id: String
    let option: ActionBridgeOption
    let reason: String
}

struct WebsiteGuidedStep {
    let title: String
    let detail: String
    let icon: String
    let tone: String
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
    @Published var searchActionStatus: String = ""
    @Published var isRunningSearchAction: Bool = false
    @Published var buttonActionStatus: String = ""
    @Published var isRunningButtonAction: Bool = false

    private var shouldAutoInspectAfterNavigation = false
    private var autoInspectCapsule: ActionCapsule?

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

    func navigate(to url: URL, autoInspect: Bool = false, capsule: ActionCapsule? = nil) {
        websiteState = nil
        inspectionStatus = ""
        searchActionStatus = ""
        buttonActionStatus = ""
        shouldAutoInspectAfterNavigation = autoInspect
        autoInspectCapsule = capsule
        webView.load(URLRequest(url: url))
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

    func fillSearchQuery(_ query: String, submit: Bool, capsule: ActionCapsule?) {
        let cleanQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanQuery.isEmpty else {
            searchActionStatus = "No capsule search query is available."
            return
        }
        isRunningSearchAction = true
        searchActionStatus = submit ? "Filling and submitting search..." : "Filling search..."
        webView.evaluateJavaScript(Self.searchFillScript(query: cleanQuery, submit: submit)) { [weak self] result, error in
            DispatchQueue.main.async {
                guard let self = self else {
                    return
                }
                self.isRunningSearchAction = false
                if let error = error {
                    self.searchActionStatus = error.localizedDescription
                    return
                }
                guard let payload = result as? [String: Any] else {
                    self.searchActionStatus = "Search action returned no page result."
                    return
                }
                let reason = Self.stringValue(payload["reason"]).replacingOccurrences(of: "_", with: " ")
                let ok = payload["ok"] as? Bool ?? false
                self.searchActionStatus = ok ? reason.capitalized : "Search action blocked: \(reason)"
                if submit {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        self.inspectPage(capsule: capsule)
                    }
                }
            }
        }
    }

    func tapButtonCandidate(_ candidate: WebsiteC2ACandidate, capsule: ActionCapsule?) {
        guard candidate.tapSafety == "safe", candidate.kind == "button", candidate.url.isEmpty else {
            buttonActionStatus = "Button tap blocked by policy: \(candidate.tapReason)"
            return
        }
        isRunningButtonAction = true
        buttonActionStatus = "Tapping safe button..."
        webView.evaluateJavaScript(Self.buttonTapScript(domIndex: candidate.domIndex)) { [weak self] result, error in
            DispatchQueue.main.async {
                guard let self = self else {
                    return
                }
                self.isRunningButtonAction = false
                if let error = error {
                    self.buttonActionStatus = error.localizedDescription
                    return
                }
                guard let payload = result as? [String: Any] else {
                    self.buttonActionStatus = "Button tap returned no page result."
                    return
                }
                let reason = Self.stringValue(payload["reason"]).replacingOccurrences(of: "_", with: " ")
                let ok = payload["ok"] as? Bool ?? false
                self.buttonActionStatus = ok ? reason.capitalized : "Button tap blocked: \(reason)"
                if ok {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.9) {
                        self.inspectPage(capsule: capsule)
                    }
                }
            }
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        websiteState = nil
        inspectionStatus = ""
        searchActionStatus = ""
        buttonActionStatus = ""
        syncState()
    }

    func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
        syncState()
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        syncState()
        if shouldAutoInspectAfterNavigation {
            shouldAutoInspectAfterNavigation = false
            let capsule = autoInspectCapsule
            autoInspectCapsule = nil
            inspectionStatus = "Auto-inspecting next page..."
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
                self?.inspectPage(capsule: capsule)
            }
        }
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
        let candidates = candidateArray(payload["candidates"], capsule: capsule)
        let combined = ([title, url, textSnippet] + headings + inputs + buttons + links).joined(separator: " ").lowercased()
        let pageKind = classifyPageKind(combined: combined, url: url, inputs: inputs)
        let safeActions = candidateActions(pageKind: pageKind, inputs: inputs, buttons: buttons, links: links, candidates: candidates)
        let residuals = residualsForPage(pageKind: pageKind, combined: combined, textSnippet: textSnippet, capsule: capsule)
        let summary = "Compiled \(pageKind.replacingOccurrences(of: "_", with: " ")) with \(inputs.count) inputs, \(buttons.count) buttons, \(links.count) links, and \(candidates.count) candidates."
        return WebsiteC2AState(
            pageKind: pageKind,
            summary: summary,
            headings: headings,
            inputs: inputs,
            buttons: buttons,
            links: links,
            candidates: candidates,
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

    private static func candidateActions(pageKind: String, inputs: [String], buttons: [String], links: [String], candidates: [WebsiteC2ACandidate]) -> [String] {
        var actions: [String] = []
        if pageKind == "search_form" || inputs.contains(where: { $0.lowercased().contains("search") }) {
            actions.append("fill_search_query")
        }
        if pageKind == "search_results" && !links.isEmpty {
            actions.append("select_matching_candidate")
        }
        if candidates.contains(where: { !$0.blocked && !$0.url.isEmpty && $0.score > 0 }) {
            actions.append("open_ranked_candidate")
        }
        if candidates.contains(where: { $0.kind == "button" && $0.url.isEmpty && $0.tapSafety == "safe" }) {
            actions.append("tap_safe_button_candidate")
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

    private static func candidateArray(_ value: Any?, capsule: ActionCapsule?) -> [WebsiteC2ACandidate] {
        guard let items = value as? [Any] else {
            return []
        }
        let targets = candidateTargets(from: capsule)
        let candidates = items.enumerated().compactMap { index, item -> WebsiteC2ACandidate? in
            guard let raw = item as? [String: Any] else {
                return nil
            }
            let label = stringValue(raw["label"])
            let url = stringValue(raw["url"])
            let kind = stringValue(raw["kind"])
            let context = stringValue(raw["context"])
            let domIndex = Int(stringValue(raw["id"])) ?? index
            guard !label.isEmpty || !url.isEmpty else {
                return nil
            }
            let candidateText = [label, url, context].joined(separator: " ")
            let ranking = rankCandidate(candidateText, targets: targets)
            let blocked = isIrreversibleCandidate(candidateText)
            let tapPolicy = buttonTapPolicy(kind: kind, url: url, candidateText: candidateText, blocked: blocked)
            let reason: String
            if blocked {
                reason = "Final or irreversible action nearby"
            } else if ranking.matchedTerms.isEmpty {
                reason = "Visible candidate with no capsule slot match yet"
            } else {
                reason = "Matched \(ranking.matchedTerms.joined(separator: ", "))"
            }
            return WebsiteC2ACandidate(
                id: "\(kind)-\(index)-\(url)-\(label)",
                domIndex: domIndex,
                label: label.isEmpty ? url : label,
                url: url,
                kind: kind.isEmpty ? "candidate" : kind,
                score: ranking.score,
                matchedTerms: ranking.matchedTerms,
                blocked: blocked,
                tapSafety: tapPolicy.safety,
                tapReason: tapPolicy.reason,
                reason: reason
            )
        }
        return Array(candidates.sorted { left, right in
            if left.blocked != right.blocked {
                return !left.blocked
            }
            if left.score == right.score {
                return left.label.localizedCaseInsensitiveCompare(right.label) == .orderedAscending
            }
            return left.score > right.score
        }.prefix(12))
    }

    private static func candidateTargets(from capsule: ActionCapsule?) -> [(term: String, weight: Double)] {
        guard let capsule = capsule else {
            return []
        }
        var targets: [(term: String, weight: Double)] = []
        for (key, value) in capsule.slots {
            let cleanValue = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleanValue.isEmpty else {
                continue
            }
            let weight: Double
            switch key {
            case "restaurant", "merchant", "destination":
                weight = 8.0
            case "item", "service_class":
                weight = 2.0
            case "modifiers":
                weight = 1.5
            case "service":
                continue
            default:
                weight = 1.0
            }
            let parts: [String]
            if key == "modifiers" {
                parts = cleanValue
                    .replacingOccurrences(of: " and ", with: ",")
                    .components(separatedBy: CharacterSet(charactersIn: ",/+&"))
            } else {
                parts = [cleanValue]
            }
            for part in parts {
                let normalized = normalizedText(part)
                if normalized.count > 1 {
                    targets.append((term: normalized, weight: weight))
                }
            }
        }
        return targets
    }

    private static func rankCandidate(_ candidateText: String, targets: [(term: String, weight: Double)]) -> (score: Double, matchedTerms: [String]) {
        let text = normalizedText(candidateText)
        var score = 0.0
        var matches: [String] = []
        for target in targets {
            let term = target.term
            guard !term.isEmpty else {
                continue
            }
            if text.contains(term) {
                score += target.weight
                matches.append(term)
                continue
            }
            let tokens = term.split(separator: " ").map(String.init).filter { $0.count > 2 }
            if !tokens.isEmpty && tokens.allSatisfy({ text.contains($0) }) {
                score += target.weight * 0.55
                matches.append(term)
            }
        }
        return (score: score, matchedTerms: Array(dictUnique(matches)))
    }

    private static func isIrreversibleCandidate(_ candidateText: String) -> Bool {
        let text = normalizedText(candidateText)
        let blockedTerms = [
            "checkout",
            "place order",
            "submit order",
            "complete order",
            "purchase",
            "payment",
            "pay now",
            "confirm order",
            "book ride",
            "reserve",
            "send message",
        ]
        return blockedTerms.contains { text.contains($0) }
    }

    private static func buttonTapPolicy(kind: String, url: String, candidateText: String, blocked: Bool) -> (safety: String, reason: String) {
        guard kind == "button", url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return ("not_applicable", "Link-backed candidates open by URL, not button tap")
        }
        if blocked {
            return ("blocked", "Final or irreversible action language")
        }
        let text = normalizedText(candidateText)
        let blockedTerms = [
            "checkout",
            "place order",
            "submit order",
            "complete order",
            "purchase",
            "payment",
            "pay now",
            "confirm order",
            "book ride",
            "reserve",
            "send",
            "delete",
        ]
        if blockedTerms.contains(where: { text.contains($0) }) {
            return ("blocked", "Button label is sensitive")
        }
        let safeTerms = [
            "search",
            "find",
            "go",
            "menu",
            "view",
            "filter",
            "apply",
            "show",
            "details",
            "results",
            "load more",
            "see more",
            "continue browsing",
            "close",
            "dismiss",
        ]
        if safeTerms.contains(where: { text.contains($0) }) {
            return ("safe", "Non-final navigation/control button")
        }
        let cautionTerms = [
            "add",
            "add to cart",
            "select",
            "customize",
            "choose",
            "continue",
            "next",
            "sign in",
            "log in",
            "accept",
        ]
        if cautionTerms.contains(where: { text.contains($0) }) {
            return ("caution", "Needs user review before Memla taps")
        }
        return ("caution", "Unknown button intent")
    }

    private static func normalizedText(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
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

    private static func javascriptStringLiteral(_ value: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: [value], options: []),
              let encodedArray = String(data: data, encoding: .utf8),
              encodedArray.hasPrefix("["),
              encodedArray.hasSuffix("]") else {
            return "\"\""
        }
        return String(encodedArray.dropFirst().dropLast())
    }

    private static func searchFillScript(query: String, submit: Bool) -> String {
        let queryLiteral = javascriptStringLiteral(query)
        let submitLiteral = submit ? "true" : "false"
        return """
        (() => {
          const query = \(queryLiteral);
          const shouldSubmit = \(submitLiteral);
          const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
          const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') return false;
            const rect = el.getBoundingClientRect();
            return rect.width > 1 && rect.height > 1;
          };
          const labelFor = (el) => clean([
            el.getAttribute('aria-label'),
            el.getAttribute('placeholder'),
            el.getAttribute('name'),
            el.getAttribute('id'),
            el.getAttribute('type'),
            el.getAttribute('title')
          ].join(' ')).toLowerCase();
          const fields = Array.from(document.querySelectorAll('input,textarea'))
            .filter(visible)
            .filter((el) => {
              const type = (el.getAttribute('type') || '').toLowerCase();
              return !['hidden', 'password', 'checkbox', 'radio', 'file'].includes(type);
            });
          const target = fields.find((el) => {
            const type = (el.getAttribute('type') || '').toLowerCase();
            const label = labelFor(el);
            return type === 'search' || label.includes('search') || label.includes('restaurant') || label.includes('store') || label.includes('food');
          }) || fields[0];
          if (!target) {
            return { ok: false, reason: 'search_input_not_found' };
          }
          target.focus();
          const valueSetter = Object.getOwnPropertyDescriptor(
            target.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype,
            'value'
          )?.set;
          if (valueSetter) {
            valueSetter.call(target, query);
          } else {
            target.value = query;
          }
          try {
            target.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: query }));
          } catch (_) {
            target.dispatchEvent(new Event('input', { bubbles: true }));
          }
          target.dispatchEvent(new Event('change', { bubbles: true }));
          if (!shouldSubmit) {
            return { ok: true, reason: 'filled_search_query', query };
          }
          const buttons = Array.from(document.querySelectorAll('button,input[type="submit"],[role="button"]'))
            .filter(visible);
          const searchButton = buttons.find((el) => /search|find|go|submit/i.test(clean([el.innerText, el.value, el.getAttribute('aria-label'), el.getAttribute('title')].join(' '))));
          if (searchButton && typeof searchButton.click === 'function') {
            searchButton.click();
            return { ok: true, reason: 'filled_and_clicked_search', query };
          }
          target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }));
          target.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }));
          if (target.form && typeof target.form.requestSubmit === 'function') {
            target.form.requestSubmit();
            return { ok: true, reason: 'filled_and_submitted_search', query };
          }
          return { ok: true, reason: 'filled_search_query_enter_sent', query };
        })();
        """
    }

    private static func buttonTapScript(domIndex: Int) -> String {
        """
        (() => {
          const targetIndex = \(domIndex);
          const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
          const normalized = (value) => clean(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ');
          const visible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') return false;
            const rect = el.getBoundingClientRect();
            return rect.width > 1 && rect.height > 1;
          };
          const elements = Array.from(document.querySelectorAll('a[href],button,[role="button"],input[type="button"],input[type="submit"]'))
            .filter(visible);
          const target = elements[targetIndex];
          if (!target) {
            return { ok: false, reason: 'button_candidate_not_found' };
          }
          if (target.matches('a[href]')) {
            return { ok: false, reason: 'link_candidate_not_tapped' };
          }
          const label = clean(target.innerText || target.value || target.getAttribute('aria-label') || target.getAttribute('title'));
          const context = clean(target.closest('article,li,section,div')?.innerText || '');
          const text = normalized([label, context].join(' '));
          const blockedTerms = [
            'checkout',
            'place order',
            'submit order',
            'complete order',
            'purchase',
            'payment',
            'pay now',
            'confirm order',
            'book ride',
            'reserve',
            'send',
            'delete'
          ];
          if (blockedTerms.some((term) => text.includes(term))) {
            return { ok: false, reason: 'sensitive_button_blocked', label };
          }
          const safeTerms = [
            'search',
            'find',
            'go',
            'menu',
            'view',
            'filter',
            'apply',
            'show',
            'details',
            'results',
            'load more',
            'see more',
            'continue browsing',
            'close',
            'dismiss'
          ];
          if (!safeTerms.some((term) => text.includes(term))) {
            return { ok: false, reason: 'button_not_policy_safe', label };
          }
          target.scrollIntoView({ block: 'center', inline: 'center' });
          target.click();
          return { ok: true, reason: 'tapped_safe_button', label };
        })();
        """
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
      const candidates = Array.from(document.querySelectorAll('a[href],button,[role="button"],input[type="button"],input[type="submit"]'))
        .filter(visible)
        .map((el, index) => {
          const rawHref = el.getAttribute('href') || '';
          let href = '';
          try {
            href = rawHref ? new URL(rawHref, location.href).href : '';
          } catch (_) {
            href = rawHref;
          }
          const label = clean(el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || rawHref);
          const context = clean(el.closest('article,li,section,div')?.innerText || '').slice(0, 280);
          return {
            id: String(index),
            kind: el.matches('a[href]') ? 'link' : 'button',
            label,
            url: href,
            context
          };
        })
        .filter((candidate) => candidate.label || candidate.url)
        .slice(0, 40);
      const text = clean(document.body ? document.body.innerText : '');
      return {
        title: document.title || '',
        url: location.href,
        text_snippet: text.slice(0, 1000),
        headings,
        inputs,
        buttons,
        links,
        candidates
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
    @State private var isC2AConsoleExpanded = false

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
                Divider()
                ZStack(alignment: .bottom) {
                    MemlaBrowserWebView(browser: browser)
                        .ignoresSafeArea(edges: .bottom)
                    if hasC2AConsole {
                        websiteC2AConsole
                    }
                }
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

    private var hasC2AConsole: Bool {
        browser.websiteState != nil || !browser.inspectionStatus.isEmpty
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

    private var websiteC2AConsole: some View {
        VStack(alignment: .leading, spacing: 8) {
            Capsule()
                .fill(Color.secondary.opacity(0.35))
                .frame(width: 36, height: 4)
                .frame(maxWidth: .infinity)
                .padding(.top, 2)

            HStack(spacing: 8) {
                Text("Website C2A")
                    .font(.caption.weight(.semibold))
                Spacer()
                if !browser.inspectionStatus.isEmpty {
                    Text(browser.inspectionStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Button(isC2AConsoleExpanded ? "Hide" : "Open") {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isC2AConsoleExpanded.toggle()
                    }
                }
                .buttonStyle(.bordered)
                .font(.caption2)
            }

            if let state = browser.websiteState {
                if isC2AConsoleExpanded {
                    ScrollView {
                        expandedWebsiteC2AContent(for: state)
                    }
                    .frame(maxHeight: 360)
                } else {
                    compactWebsiteC2AContent(for: state)
                }
            } else if !browser.inspectionStatus.isEmpty {
                Text(browser.inspectionStatus)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .shadow(color: Color.black.opacity(0.16), radius: 18, x: 0, y: -6)
        .padding(.horizontal, 10)
        .padding(.bottom, 10)
    }

    private func compactWebsiteC2AContent(for state: WebsiteC2AState) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    capsuleChip(title: "Page", value: readableRequirement(state.pageKind))
                    capsuleChip(title: "Inputs", value: "\(state.inputs.count)")
                    capsuleChip(title: "Candidates", value: "\(state.candidates.count)")
                    if let best = state.candidates.first(where: { !$0.blocked && !$0.url.isEmpty && $0.score > 0 }) {
                        capsuleChip(title: "Best", value: best.label)
                    }
                }
            }
            guidedStepControls(for: state)
        }
    }

    private func expandedWebsiteC2AContent(for state: WebsiteC2AState) -> some View {
        VStack(alignment: .leading, spacing: 8) {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        capsuleChip(title: "Page", value: readableRequirement(state.pageKind))
                        capsuleChip(title: "Inputs", value: "\(state.inputs.count)")
                        capsuleChip(title: "Buttons", value: "\(state.buttons.count)")
                        capsuleChip(title: "Links", value: "\(state.links.count)")
                        capsuleChip(title: "Candidates", value: "\(state.candidates.count)")
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
                guidedStepControls(for: state)
                candidateControls(for: state)
                if !state.residuals.isEmpty {
                    Text("Residuals: \(state.residuals.map { readableRequirement($0) }.joined(separator: ", "))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                searchActionControls(for: state)
                bridgeSuggestionControls(for: state)
                if !state.headings.isEmpty {
                    Text("Headings: \(state.headings.prefix(3).joined(separator: " / "))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
        }
    }

    private func guidedStepControls(for state: WebsiteC2AState) -> some View {
        let step = guidedStep(for: state)
        return HStack(alignment: .top, spacing: 8) {
            Image(systemName: step.icon)
                .font(.caption.weight(.semibold))
                .foregroundStyle(guidanceColor(step.tone))
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(step.title)
                    .font(.caption.weight(.semibold))
                Text(step.detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
            Spacer(minLength: 0)
        }
        .padding(8)
        .background(guidanceColor(step.tone).opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @ViewBuilder
    private func candidateControls(for state: WebsiteC2AState) -> some View {
        if !state.candidates.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("Ranked Candidates")
                    .font(.caption.weight(.semibold))
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(alignment: .top, spacing: 8) {
                        ForEach(Array(state.candidates.prefix(8))) { candidate in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text(candidate.kind.capitalized)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Text(String(format: "%.1f", candidate.score))
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(candidate.score > 0 ? .green : .secondary)
                                }
                                Text(candidate.label)
                                    .font(.caption.weight(.semibold))
                                    .lineLimit(2)
                                Text(candidate.reason)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                                if candidate.kind == "button" && candidate.url.isEmpty {
                                    Text(candidate.tapReason)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                                candidateActionButton(candidate)
                            }
                            .frame(width: 190, alignment: .leading)
                            .padding(8)
                            .background(candidate.score > 0 ? Color.green.opacity(0.10) : Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }
                    }
                }
                if !browser.buttonActionStatus.isEmpty {
                    Text(browser.buttonActionStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
        }
    }

    @ViewBuilder
    private func candidateActionButton(_ candidate: WebsiteC2ACandidate) -> some View {
        if candidate.blocked {
            Button("Blocked") {
            }
            .buttonStyle(.bordered)
            .disabled(true)
        } else if !candidate.url.isEmpty {
            Button("Open") {
                openCandidate(candidate)
            }
            .buttonStyle(.bordered)
        } else if candidate.kind == "button" && candidate.tapSafety == "safe" {
            Button(browser.isRunningButtonAction ? "Tapping..." : "Tap") {
                tapCandidate(candidate)
            }
            .buttonStyle(.borderedProminent)
            .disabled(browser.isRunningButtonAction)
        } else if candidate.kind == "button" && candidate.tapSafety == "caution" {
            Button("Review") {
            }
            .buttonStyle(.bordered)
            .disabled(true)
        } else {
            Button("No Action") {
            }
            .buttonStyle(.bordered)
            .disabled(true)
        }
    }

    @ViewBuilder
    private func searchActionControls(for state: WebsiteC2AState) -> some View {
        if let query = searchQueryIfAvailable(for: state) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Safe Search Primitive")
                    .font(.caption.weight(.semibold))
                HStack(spacing: 8) {
                    Button("Fill Search") {
                        browser.fillSearchQuery(query, submit: false, capsule: route.capsule)
                    }
                    .buttonStyle(.bordered)
                    .disabled(browser.isRunningSearchAction)

                    Button("Fill + Submit") {
                        browser.fillSearchQuery(query, submit: true, capsule: route.capsule)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(browser.isRunningSearchAction)
                }
                Text("Query: \(query)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                if !browser.searchActionStatus.isEmpty {
                    Text(browser.searchActionStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
        }
    }

    @ViewBuilder
    private func bridgeSuggestionControls(for state: WebsiteC2AState) -> some View {
        if !bridgeSuggestions(for: state).isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("Bridge Suggestions")
                    .font(.caption.weight(.semibold))
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(bridgeSuggestions(for: state)) { suggestion in
                            Button(suggestion.option.label) {
                                openBridgeOption(suggestion.option)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                Text(bridgeSuggestions(for: state).map { $0.reason }.joined(separator: " "))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
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

    private func guidedStep(for state: WebsiteC2AState) -> WebsiteGuidedStep {
        if state.pageKind == "checkout" || state.residuals.contains("irreversible_action_nearby") {
            return WebsiteGuidedStep(
                title: "Stop before final action",
                detail: "Checkout or payment language is visible. Verify the capsule checklist and let the user make the final purchase decision.",
                icon: "hand.raised.fill",
                tone: "danger"
            )
        }
        if state.pageKind == "login" || state.residuals.contains("login_required") {
            return WebsiteGuidedStep(
                title: "Recover session",
                detail: "This page looks like a login or footer state. Try the app bridge if you are logged in there, or use neutral web search to find a better landing page.",
                icon: "person.crop.circle.badge.exclamationmark",
                tone: "warning"
            )
        }
        if state.pageKind == "blocked_or_bot_check" || state.residuals.contains("bot_check_or_captcha") {
            return WebsiteGuidedStep(
                title: "Use another bridge",
                detail: "This looks like a blocker or bot-check. Memla should not bypass it; try app bridge or neutral web search.",
                icon: "exclamationmark.triangle.fill",
                tone: "warning"
            )
        }
        if state.safeActions.contains("fill_search_query") {
            return WebsiteGuidedStep(
                title: "Fill the search box",
                detail: "A search-like input is visible. Use the safe search primitive to fill the capsule query before selecting a result.",
                icon: "magnifyingglass",
                tone: "safe"
            )
        }
        if let best = state.candidates.first(where: { !$0.blocked && !$0.url.isEmpty && $0.score > 0 }) {
            return WebsiteGuidedStep(
                title: "Open ranked candidate",
                detail: "\(best.label) matches the capsule best right now. Open it, then Memla will inspect the next page.",
                icon: "link",
                tone: "safe"
            )
        }
        if let button = state.candidates.first(where: { $0.kind == "button" && $0.url.isEmpty && $0.tapSafety == "safe" }) {
            return WebsiteGuidedStep(
                title: "Tap safe button",
                detail: "\(button.label) is classified as a non-final control. Tap it only if it matches the current step.",
                icon: "hand.tap",
                tone: "safe"
            )
        }
        if state.pageKind == "cart" {
            return WebsiteGuidedStep(
                title: "Verify cart",
                detail: "Cart state is visible. Compare item, modifiers, tip, address, and total against the capsule before final checkout.",
                icon: "cart.badge.questionmark",
                tone: "warning"
            )
        }
        if state.pageKind == "menu_or_item" {
            return WebsiteGuidedStep(
                title: "Review item page",
                detail: "Menu or item controls are visible. Match the item and modifiers before adding anything to cart.",
                icon: "list.bullet.rectangle",
                tone: "safe"
            )
        }
        if state.residuals.contains("target_restaurant_not_visible") || state.residuals.contains("target_item_not_visible") {
            return WebsiteGuidedStep(
                title: "Find a better page",
                detail: "The capsule target is not visible here. Use the bridge suggestions or search again before opening candidates.",
                icon: "arrow.triangle.branch",
                tone: "warning"
            )
        }
        if !state.candidates.isEmpty {
            return WebsiteGuidedStep(
                title: "Review visible candidates",
                detail: "Memla found page candidates but none strongly match the capsule yet. Prefer search or user review before opening.",
                icon: "rectangle.stack.badge.person.crop",
                tone: "neutral"
            )
        }
        return WebsiteGuidedStep(
            title: "Continue inspection",
            detail: "This page does not expose a clear next action yet. Navigate or search, then inspect again.",
            icon: "scope",
            tone: "neutral"
        )
    }

    private func guidanceColor(_ tone: String) -> Color {
        switch tone {
        case "safe":
            return .green
        case "warning":
            return .orange
        case "danger":
            return .red
        default:
            return .blue
        }
    }

    private func searchQueryIfAvailable(for state: WebsiteC2AState) -> String? {
        let query = commerceSearchQuery(from: route.capsule)
        guard !query.isEmpty else {
            return nil
        }
        let exposesSearch = state.safeActions.contains("fill_search_query")
            || state.pageKind == "search_form"
            || state.inputs.contains { $0.lowercased().contains("search") }
        return exposesSearch ? query : nil
    }

    private func commerceSearchQuery(from capsule: ActionCapsule?) -> String {
        guard let capsule = capsule else {
            return ""
        }
        let restaurant = capsule.slots["restaurant"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let item = capsule.slots["item"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if restaurant.isEmpty {
            return item
        }
        if item.isEmpty {
            return restaurant
        }
        if restaurant.lowercased().contains(item.lowercased()) {
            return restaurant
        }
        return "\(restaurant) \(item)"
    }

    private func bridgeSuggestions(for state: WebsiteC2AState) -> [WebsiteBridgeSuggestion] {
        guard let capsule = route.capsule else {
            return []
        }
        var suggestions: [WebsiteBridgeSuggestion] = []
        var seen = Set<String>()

        func add(optionID: String, reason: String) {
            guard let option = capsule.bridgeOptions.first(where: { $0.optionID == optionID }),
                  !seen.contains(option.optionID) else {
                return
            }
            seen.insert(option.optionID)
            suggestions.append(WebsiteBridgeSuggestion(id: option.optionID, option: option, reason: reason))
        }

        if state.pageKind == "login" || state.residuals.contains("login_required") {
            add(optionID: "service_app", reason: "Login state detected, so the installed app may have the best session.")
            add(optionID: "generic_web_search", reason: "Neutral web search can recover when the service URL lands in login/footer state.")
        }
        if state.pageKind == "blocked_or_bot_check" || state.residuals.contains("bot_check_or_captcha") {
            add(optionID: "generic_web_search", reason: "Bot-check state detected, so try a neutral web search instead.")
            add(optionID: "service_app", reason: "The app bridge may avoid this web blocker.")
        }
        if state.residuals.contains("target_restaurant_not_visible") || state.residuals.contains("target_item_not_visible") {
            add(optionID: "generic_web_search", reason: "Target terms are not visible on this page, so search the open web.")
        }
        return suggestions
    }

    private func openCandidate(_ candidate: WebsiteC2ACandidate) {
        guard !candidate.blocked, let url = URL(string: candidate.url) else {
            return
        }
        let scheme = url.scheme?.lowercased() ?? ""
        if scheme == "http" || scheme == "https" {
            browser.navigate(to: url, autoInspect: true, capsule: route.capsule)
            return
        }
        UIApplication.shared.open(url)
    }

    private func tapCandidate(_ candidate: WebsiteC2ACandidate) {
        browser.tapButtonCandidate(candidate, capsule: route.capsule)
    }

    private func openBridgeOption(_ option: ActionBridgeOption) {
        guard let url = URL(string: option.url) else {
            return
        }
        if option.kind == "in_app_web" {
            browser.navigate(to: url, autoInspect: true, capsule: route.capsule)
            return
        }
        UIApplication.shared.open(url)
    }

    private func readableRequirement(_ value: String) -> String {
        value.replacingOccurrences(of: "_", with: " ").capitalized
    }
}
