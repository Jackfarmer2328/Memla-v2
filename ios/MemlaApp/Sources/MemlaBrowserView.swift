import Foundation
import SwiftUI
import UIKit
import WebKit
import Combine

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
    let role: String
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
    let authState: String
    let authDomain: String
    let textSnippet: String
    let capsuleVerification: WebsiteCapsuleVerification?
}

struct WebsiteCapsuleVerification {
    let matched: [String]
    let missing: [String]
    let warnings: [String]
    let summary: String
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

struct WebsiteMirrorFact: Identifiable {
    let id: String
    let title: String
    let value: String
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
    private var observationCapsule: ActionCapsule?
    private var pageObservationTimer: Timer?
    private var lastObservedFingerprint: String = ""
    private var isAutoInspectQueued = false
    private var lastDoorDashStorefrontWarmupURL: String = ""
    private var doorDashStorefrontWarmupAttempts = 0

    override init() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        self.webView = WKWebView(frame: .zero, configuration: configuration)
        super.init()
        self.webView.navigationDelegate = self
    }

    deinit {
        pageObservationTimer?.invalidate()
    }

    func load(_ url: URL) {
        if webView.url == nil {
            webView.load(URLRequest(url: url))
        }
        syncState()
    }

    func startGrounding(capsule: ActionCapsule?) {
        observationCapsule = capsule
        guard pageObservationTimer == nil else {
            return
        }
        let timer = Timer(timeInterval: 1.35, repeats: true) { [weak self] _ in
            self?.pollForPageChange()
        }
        pageObservationTimer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    func stopGrounding() {
        pageObservationTimer?.invalidate()
        pageObservationTimer = nil
        lastObservedFingerprint = ""
        isAutoInspectQueued = false
        lastDoorDashStorefrontWarmupURL = ""
        doorDashStorefrontWarmupAttempts = 0
    }

    func navigate(to url: URL, autoInspect: Bool = false, capsule: ActionCapsule? = nil) {
        websiteState = nil
        inspectionStatus = ""
        searchActionStatus = ""
        buttonActionStatus = ""
        shouldAutoInspectAfterNavigation = autoInspect
        autoInspectCapsule = capsule
        observationCapsule = capsule
        lastObservedFingerprint = ""
        lastDoorDashStorefrontWarmupURL = ""
        doorDashStorefrontWarmupAttempts = 0
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
        observationCapsule = capsule
        isInspecting = true
        inspectionStatus = "Inspecting page..."
        syncState()
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
                let state = Self.buildWebsiteState(payload: payload, capsule: capsule)
                self.websiteState = state
                self.inspectionStatus = "Page inspected."
                self.maybeWarmDoorDashStorefront(payload: payload, state: state, capsule: capsule)
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

    func tapButtonCandidate(_ candidate: WebsiteC2ACandidate, allowCaution: Bool = false, capsule: ActionCapsule?) {
        let canTap = candidate.tapSafety == "safe" || (allowCaution && candidate.tapSafety == "caution")
        guard canTap, candidate.kind == "button", candidate.url.isEmpty else {
            buttonActionStatus = "Button tap blocked by policy: \(candidate.tapReason)"
            return
        }
        isRunningButtonAction = true
        buttonActionStatus = candidate.tapSafety == "caution" ? "Tapping reviewed button..." : "Tapping safe button..."
        webView.evaluateJavaScript(Self.buttonTapScript(domIndex: candidate.domIndex, allowCaution: allowCaution)) { [weak self] result, error in
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
        lastObservedFingerprint = ""
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

    private func pollForPageChange() {
        guard !isInspecting, !isLoading, webView.url != nil else {
            return
        }
        webView.evaluateJavaScript(Self.pageGroundingProbeScript) { [weak self] result, error in
            DispatchQueue.main.async {
                guard let self = self, error == nil, let payload = result as? [String: Any] else {
                    return
                }
                let fingerprint = Self.stringValue(payload["fingerprint"])
                guard !fingerprint.isEmpty else {
                    return
                }
                if self.lastObservedFingerprint.isEmpty {
                    self.lastObservedFingerprint = fingerprint
                    return
                }
                guard fingerprint != self.lastObservedFingerprint else {
                    return
                }
                self.lastObservedFingerprint = fingerprint
                self.queueAutoInspect(reason: "Mirror detected a page change...")
            }
        }
    }

    private func queueAutoInspect(reason: String) {
        guard !isAutoInspectQueued, !isInspecting else {
            return
        }
        isAutoInspectQueued = true
        inspectionStatus = reason
        let capsule = observationCapsule
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) { [weak self] in
            guard let self = self else {
                return
            }
            self.isAutoInspectQueued = false
            self.inspectPage(capsule: capsule)
        }
    }

    private func maybeWarmDoorDashStorefront(payload: [String: Any], state: WebsiteC2AState, capsule: ActionCapsule?) {
        guard state.pageKind == "dd_storefront" else {
            return
        }
        let itemCount = payload["doordash_item_card_count"] as? Int ?? 0
        guard itemCount == 0 else {
            lastDoorDashStorefrontWarmupURL = currentURL
            doorDashStorefrontWarmupAttempts = 0
            return
        }
        let url = currentURL
        guard !url.isEmpty else {
            return
        }
        if lastDoorDashStorefrontWarmupURL != url {
            lastDoorDashStorefrontWarmupURL = url
            doorDashStorefrontWarmupAttempts = 0
        }
        guard doorDashStorefrontWarmupAttempts < 1 else {
            return
        }
        doorDashStorefrontWarmupAttempts += 1
        inspectionStatus = "Memla is peeking into the DoorDash menu..."
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.1) { [weak self] in
            guard let self = self else {
                return
            }
            guard self.currentURL == url, !self.isLoading else {
                return
            }
            self.webView.evaluateJavaScript(Self.doorDashStorefrontPeekScript) { [weak self] _, _ in
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                    self?.inspectPage(capsule: capsule)
                }
            }
        }
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
        let normalizedCombined = normalizedText(combined)
        let isDoorDash = isDoorDashDomain(url: url)
        let pageKind = isDoorDash
            ? classifyDoorDashPage(payload: payload, combined: combined, url: url, inputs: inputs)
            : classifyPageKind(combined: combined, url: url, inputs: inputs)
        let candidateSource = isDoorDash ? (payload["doordash_candidates"] ?? payload["candidates"]) : payload["candidates"]
        let candidates = candidateArray(candidateSource, capsule: capsule, pageKind: pageKind)
        let safeActions = candidateActions(pageKind: pageKind, inputs: inputs, buttons: buttons, links: links, candidates: candidates)
        let residuals = residualsForPage(pageKind: pageKind, combined: normalizedCombined, textSnippet: textSnippet, capsule: capsule)
        let authState = classifyAuthState(pageKind: pageKind, combined: combined)
        let authDomain = domainFrom(url: url)
        let capsuleVerification = capsuleVerificationForPage(pageKind: pageKind, combined: normalizedCombined, capsule: capsule)
        let summary = isDoorDash
            ? doorDashSummary(payload: payload, pageKind: pageKind, candidates: candidates)
            : "Compiled \(pageKind.replacingOccurrences(of: "_", with: " ")) with \(inputs.count) inputs, \(buttons.count) buttons, \(links.count) links, and \(candidates.count) candidates."
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
            authState: authState,
            authDomain: authDomain,
            textSnippet: textSnippet,
            capsuleVerification: capsuleVerification
        )
    }

    private static func isDoorDashDomain(url: String) -> Bool {
        domainFrom(url: url).contains("doordash.com")
    }

    private static func classifyDoorDashPage(payload: [String: Any], combined: String, url: String, inputs: [String]) -> String {
        let layerKind = stringValue(payload["doordash_active_layer"])
        let modalTitle = normalizedText(stringValue(payload["doordash_modal_title"]))
        let hasStoreCards = (payload["doordash_store_card_count"] as? Int ?? 0) > 0
        let hasCartCTA = payload["doordash_has_cart_cta"] as? Bool ?? false
        let hasContinueCTA = payload["doordash_has_continue_cta"] as? Bool ?? false
        let hasAddToCartCTA = payload["doordash_has_add_to_cart_cta"] as? Bool ?? false
        let tipCount = payload["doordash_tip_option_count"] as? Int ?? 0
        let hasPaymentSheet = payload["doordash_has_payment_sheet"] as? Bool ?? false
        let hasAddressCTA = payload["doordash_has_address_cta"] as? Bool ?? false
        let hasCartCloseCTA = payload["doordash_has_cart_close_cta"] as? Bool ?? false

        if hasPaymentSheet || layerKind == "payment_sheet" || combined.contains("select payment method") {
            return "dd_payment_sheet"
        }
        if url.contains("/consumer/checkout") {
            return "dd_checkout"
        }
        if layerKind == "cart_drawer" || hasContinueCTA || hasCartCloseCTA {
            return "dd_cart_drawer"
        }
        if layerKind == "item_modal" || (!modalTitle.isEmpty && layerKind != "page") {
            return "dd_item_modal"
        }
        if hasStoreCards || url.contains("/search/store/") {
            return "dd_search_results"
        }
        if url.contains("/store/") {
            return "dd_storefront"
        }
        if hasContinueCTA || tipCount > 0 || hasAddressCTA {
            return "dd_cart_page"
        }
        return classifyPageKind(combined: combined, url: url, inputs: inputs)
    }

    private static func doorDashSummary(payload: [String: Any], pageKind: String, candidates: [WebsiteC2ACandidate]) -> String {
        let storeCardCount = payload["doordash_store_card_count"] as? Int ?? 0
        let itemCardCount = payload["doordash_item_card_count"] as? Int ?? 0
        let tipCount = payload["doordash_tip_option_count"] as? Int ?? 0
        let layerKind = stringValue(payload["doordash_active_layer"])
        let modalTitle = stringValue(payload["doordash_modal_title"])
        switch pageKind {
        case "dd_search_results":
            return "DoorDash search results distilled into \(storeCardCount) store cards."
        case "dd_storefront":
            return "DoorDash storefront with \(itemCardCount) relevant item/menu cards."
        case "dd_item_modal":
            return "DoorDash item modal detected\(modalTitle.isEmpty ? "" : ": \(modalTitle)")."
        case "dd_cart_drawer":
            return "DoorDash cart drawer is active with a continue path."
        case "dd_cart_page":
            return "DoorDash cart page detected with \(tipCount) tip options."
        case "dd_checkout":
            return "DoorDash checkout detected. Mirror should stop before payment."
        case "dd_payment_sheet":
            return "DoorDash payment sheet is active. Human confirmation boundary reached."
        default:
            return "DoorDash \(pageKind.replacingOccurrences(of: "_", with: " ")) layer \(layerKind) with \(candidates.count) candidates."
        }
    }

    private static func classifyPageKind(combined: String, url: String, inputs: [String]) -> String {
        if combined.contains("captcha") || combined.contains("verify you are human") {
            return "blocked_or_bot_check"
        }
        if combined.contains("sign in") || combined.contains("log in") || combined.contains("login") {
            return "login"
        }
        if combined.contains("place order")
            || combined.contains("submit order")
            || combined.contains("complete order")
            || combined.contains("confirm order")
            || combined.contains("pay now")
            || combined.contains("card number")
            || combined.contains("cvv")
            || combined.contains("payment method")
            || combined.contains("payment information") {
            return "checkout"
        }
        if combined.contains("add to cart") || combined.contains("customize") || combined.contains("toppings") || combined.contains("menu") {
            return "menu_or_item"
        }
        if combined.contains("cart") || combined.contains("subtotal") || combined.contains("tip") || combined.contains("checkout") {
            return "cart"
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
        if pageKind == "dd_search_results" {
            actions.append("open_ranked_candidate")
        }
        if pageKind == "dd_storefront" {
            actions.append("open_ranked_candidate")
            actions.append("review_item_and_modifiers")
        }
        if pageKind == "dd_item_modal" {
            actions.append("review_item_and_modifiers")
            actions.append("review_then_tap_candidate")
        }
        if pageKind == "dd_cart_drawer" || pageKind == "dd_cart_page" {
            actions.append("verify_cart_against_capsule")
            actions.append("review_then_enter_checkout")
        }
        if pageKind == "dd_checkout" || pageKind == "dd_payment_sheet" {
            actions.append("stop_before_purchase")
        }
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
        if candidates.contains(where: { $0.kind == "button" && $0.url.isEmpty && $0.tapSafety == "caution" }) {
            actions.append("review_then_tap_candidate")
        }
        if pageKind == "menu_or_item" {
            actions.append("review_item_and_modifiers")
        }
        if pageKind == "cart" {
            actions.append("verify_cart_against_capsule")
            actions.append("review_then_enter_checkout")
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
        if pageKind == "dd_payment_sheet" {
            residuals.append("payment_sheet_active")
            residuals.append("final_confirmation_required")
            residuals.append("irreversible_action_nearby")
        }
        if pageKind == "dd_item_modal" {
            residuals.append("active_customizer_layer")
        }
        if pageKind == "dd_cart_drawer" {
            residuals.append("cart_drawer_active")
        }
        if pageKind == "login" {
            residuals.append("login_required")
        }
        if pageKind == "blocked_or_bot_check" {
            residuals.append("bot_check_or_captcha")
        }
        if pageKind == "checkout" {
            residuals.append("irreversible_action_nearby")
            residuals.append("final_confirmation_required")
        }
        if pageKind == "cart" {
            residuals.append("cart_verification_required")
        }
        if textSnippet.isEmpty {
            residuals.append("visible_text_empty")
        }
        if let capsule = capsule {
            let restaurant = normalizedText(capsule.slots["restaurant"] ?? "")
            if !restaurant.isEmpty && !combined.contains(restaurant) {
                residuals.append("target_restaurant_not_visible")
            }
            let item = normalizedText(capsule.slots["item"] ?? "")
            if !item.isEmpty && !combined.contains(item) {
                residuals.append("target_item_not_visible")
            }
        }
        return Array(dictUnique(residuals))
    }

    private static func capsuleVerificationForPage(pageKind: String, combined: String, capsule: ActionCapsule?) -> WebsiteCapsuleVerification? {
        guard let capsule = capsule else {
            return nil
        }
        let relevantPages = ["menu_or_item", "cart", "checkout", "dd_storefront", "dd_item_modal", "dd_cart_drawer", "dd_cart_page", "dd_checkout", "dd_payment_sheet"]
        guard relevantPages.contains(pageKind) else {
            return nil
        }
        var matched: [String] = []
        var missing: [String] = []
        var warnings: [String] = []

        func verifySlot(_ key: String, label: String? = nil) {
            let raw = capsule.slots[key]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !raw.isEmpty else {
                return
            }
            let readable = label ?? key.replacingOccurrences(of: "_", with: " ")
            let parts: [String]
            if key == "modifiers" {
                parts = raw
                    .replacingOccurrences(of: " and ", with: ",")
                    .components(separatedBy: CharacterSet(charactersIn: ",/+&"))
            } else {
                parts = [raw]
            }
            let normalizedParts = parts
                .map { normalizedText($0) }
                .filter { !$0.isEmpty }
            guard !normalizedParts.isEmpty else {
                return
            }
            let foundParts = normalizedParts.filter { combined.contains($0) }
            if foundParts.isEmpty {
                missing.append(readable)
            } else if foundParts.count == normalizedParts.count {
                matched.append(readable)
            } else {
                matched.append(readable)
                missing.append("\(readable) partial")
            }
        }

        verifySlot("restaurant")
        verifySlot("item")
        verifySlot("modifiers")
        verifySlot("tip")

        if pageKind == "menu_or_item" {
            warnings.append("review_item_options_before_cart")
        }
        if pageKind == "dd_storefront" || pageKind == "dd_item_modal" {
            warnings.append("review_item_options_before_cart")
        }
        if pageKind == "cart" {
            warnings.append("verify_cart_before_checkout")
            warnings.append("checkout_is_reviewed_navigation_only")
        }
        if pageKind == "dd_cart_drawer" || pageKind == "dd_cart_page" {
            warnings.append("verify_cart_before_checkout")
            warnings.append("checkout_is_reviewed_navigation_only")
        }
        if pageKind == "checkout" {
            warnings.append("human_must_complete_final_payment_or_place_order")
        }
        if pageKind == "dd_checkout" || pageKind == "dd_payment_sheet" {
            warnings.append("human_must_complete_final_payment_or_place_order")
        }

        let summary: String
        if missing.isEmpty {
            summary = "Capsule terms are visible. Continue only after reviewing item, modifiers, tip, address, and total."
        } else {
            summary = "Missing visible evidence for \(missing.joined(separator: ", ")). Review before continuing."
        }
        return WebsiteCapsuleVerification(
            matched: Array(dictUnique(matched)),
            missing: Array(dictUnique(missing)),
            warnings: Array(dictUnique(warnings)),
            summary: summary
        )
    }

    private static func classifyAuthState(pageKind: String, combined: String) -> String {
        if pageKind == "login" {
            return "login_required"
        }
        let signedInHints = [
            "account",
            "profile",
            "sign out",
            "log out",
            "orders",
            "your address",
            "deliver to",
            "cart",
            "checkout",
        ]
        if signedInHints.contains(where: { combined.contains($0) }) {
            return "likely_signed_in"
        }
        return "unknown"
    }

    private static func domainFrom(url: String) -> String {
        guard let host = URL(string: url)?.host else {
            return ""
        }
        return host.replacingOccurrences(of: "www.", with: "")
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

    private static func candidateArray(_ value: Any?, capsule: ActionCapsule?, pageKind: String) -> [WebsiteC2ACandidate] {
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
            let serviceRole = stringValue(raw["service_role"])
            guard !label.isEmpty || !url.isEmpty else {
                return nil
            }
            let candidateText = [label, url, context].joined(separator: " ")
            let ranking = rankCandidate(candidateText, targets: targets)
            let role = serviceRole.isEmpty ? candidateRole(pageKind: pageKind, kind: kind, url: url, label: label, context: context) : serviceRole
            let blockedText = kind == "button" ? label : candidateText
            let blocked = isIrreversibleCandidate(blockedText) || isBlockedServiceRole(role)
            let tapPolicy = buttonTapPolicy(kind: kind, url: url, label: label, context: context, blocked: blocked)
            let adjustedScore = ranking.score + roleScoreAdjustment(role: role) + serviceLabelAdjustment(role: role, label: label, context: context)
            let resolvedLabel = displayLabel(label: label, url: url, context: context, role: role)
            let reason: String
            if blocked {
                reason = "Final or irreversible action nearby"
            } else if ranking.matchedTerms.isEmpty {
                reason = roleReason(role: role)
            } else {
                reason = "Matched \(ranking.matchedTerms.joined(separator: ", ")) via \(role.replacingOccurrences(of: "_", with: " "))"
            }
            return WebsiteC2ACandidate(
                id: "\(kind)-\(index)-\(url)-\(label)",
                domIndex: domIndex,
                label: resolvedLabel,
                url: url,
                kind: kind.isEmpty ? "candidate" : kind,
                role: role,
                score: adjustedScore,
                matchedTerms: ranking.matchedTerms,
                blocked: blocked,
                tapSafety: tapPolicy.safety,
                tapReason: tapPolicy.reason,
                reason: reason
            )
        }
        var seenKeys = Set<String>()
        let deduped = candidates.filter { candidate in
            let key = [candidate.role, normalizedText(candidate.label), candidate.url].joined(separator: "|")
            if seenKeys.contains(key) {
                return false
            }
            seenKeys.insert(key)
            return true
        }
        return Array(deduped.sorted { left, right in
            if left.blocked != right.blocked {
                return !left.blocked
            }
            if left.score == right.score {
                return left.label.localizedCaseInsensitiveCompare(right.label) == .orderedAscending
            }
            return left.score > right.score
        }.prefix(12))
    }

    private static func isBlockedServiceRole(_ role: String) -> Bool {
        ["dd_payment_method", "dd_payment_sheet"].contains(role)
    }

    private static func displayLabel(label: String, url: String, context: String, role: String) -> String {
        let cleanLabel = label.trimmingCharacters(in: .whitespacesAndNewlines)
        let looksLikeURL = cleanLabel.hasPrefix("http") || cleanLabel.contains("/store/")
        guard cleanLabel.isEmpty || looksLikeURL else {
            return cleanLabel
        }
        let contextTokens = context
            .components(separatedBy: CharacterSet(charactersIn: "|•"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if let useful = contextTokens.first(where: { token in
            let lower = token.lowercased()
            return !lower.contains("delivery fee")
                && !lower.contains("dashpass")
                && !lower.contains("minutes")
                && !lower.contains("reviews")
                && !lower.contains("pickup")
        }) {
            let words = useful.split(separator: " ")
            if !words.isEmpty {
                return words.prefix(role == "store_link" ? 4 : 6).joined(separator: " ")
            }
        }
        return cleanLabel.isEmpty ? url : cleanLabel
    }

    private static func candidateRole(pageKind: String, kind: String, url: String, label: String, context: String) -> String {
        let text = normalizedText([label, context, url].joined(separator: " "))
        let labelText = normalizedText(label)
        if text.contains("skip to main content") || text.contains("accessibility") {
            return "accessibility_link"
        }
        if text.contains("become a dasher") || text.contains("merchant") || text.contains("doordash merchant") || text.contains("gift cards") {
            return "utility_link"
        }
        if text.contains("review") || text.contains("rating") || text.contains("ratings") {
            return "review_link"
        }
        if text.contains("privacy") || text.contains("terms") || text.contains("careers") || text.contains("gift cards") || text.contains("about") || text.contains("help") || text.contains("support") {
            return "utility_link"
        }
        if text.contains("login") || text.contains("sign in") || text.contains("log in") || text.contains("account") {
            return "auth_link"
        }
        if text.contains("checkout") || text.contains("place order") || text.contains("pay now") || text.contains("complete order") {
            return kind == "button" ? "checkout_button" : "checkout_link"
        }
        if text.contains("cart") || text.contains("bag") {
            return kind == "button" ? "cart_button" : "cart_link"
        }
        if pageKind == "menu_or_item" && kind == "link" && (url.contains("#") || text.contains("menu") || text.contains("featured items") || text.contains("specialty pizza") || text.contains("build your own")) && !context.contains("$") {
            return "nav_link"
        }
        if kind == "link" && url.contains("/store/") {
            return "store_link"
        }
        if pageKind == "menu_or_item" {
            if kind == "button" && ["add", "customize", "choose", "select", "continue", "modifier", "topping", "size", "crust"].contains(where: { labelText.contains($0) || text.contains($0) }) {
                return kind == "button" ? "item_action_button" : "menu_item"
            }
            if (label.contains("$") || context.contains("$")) && !text.contains("review") {
                return kind == "button" ? "item_action_button" : "menu_item"
            }
            if kind == "link" {
                return "menu_item"
            }
        }
        if pageKind == "search_results" || pageKind == "web_page" {
            if kind == "link" {
                return "store_link"
            }
        }
        if kind == "button" {
            return "control_button"
        }
        return "generic_candidate"
    }

    private static func roleScoreAdjustment(role: String) -> Double {
        switch role {
        case "store_link":
            return 2.2
        case "dd_store_card":
            return 5.2
        case "menu_item":
            return 1.8
        case "dd_item_card":
            return 3.4
        case "item_action_button":
            return 1.2
        case "dd_add_to_cart":
            return 4.4
        case "dd_continue_cta":
            return 5.0
        case "dd_cart_cta":
            return 3.6
        case "dd_tip_option":
            return 2.0
        case "dd_address_cta":
            return 1.0
        case "dd_modal_close":
            return -0.4
        case "cart_link", "cart_button":
            return 1.0
        case "control_button":
            return 0.5
        case "review_link":
            return -6.0
        case "nav_link":
            return -3.5
        case "dd_menu_nav":
            return -7.0
        case "accessibility_link":
            return -8.0
        case "utility_link", "auth_link":
            return -4.0
        case "checkout_link", "checkout_button":
            return -1.0
        case "dd_payment_method":
            return -10.0
        default:
            return 0.0
        }
    }

    private static func roleReason(role: String) -> String {
        switch role {
        case "store_link":
            return "Visible store/result candidate"
        case "dd_store_card":
            return "DoorDash merchant card"
        case "menu_item":
            return "Visible menu or item candidate"
        case "dd_item_card":
            return "DoorDash menu item card"
        case "item_action_button":
            return "Visible item funnel control"
        case "dd_add_to_cart":
            return "DoorDash add-to-cart control"
        case "dd_continue_cta":
            return "DoorDash continue-to-checkout control"
        case "dd_cart_cta":
            return "DoorDash cart CTA"
        case "dd_tip_option":
            return "DoorDash tip selector"
        case "dd_address_cta":
            return "DoorDash address/edit control"
        case "dd_modal_close":
            return "Dismiss the active sheet if needed"
        case "cart_link", "cart_button":
            return "Visible cart navigation control"
        case "nav_link":
            return "Category/navigation link inside the page"
        case "dd_menu_nav":
            return "DoorDash category navigation"
        case "review_link":
            return "Review content is usually not task-relevant"
        case "accessibility_link":
            return "Accessibility/navigation helper link"
        case "utility_link", "auth_link":
            return "Utility or account navigation"
        default:
            return "Visible candidate with no capsule slot match yet"
        }
    }

    private static func serviceLabelAdjustment(role: String, label: String, context: String) -> Double {
        let text = normalizedText([label, context].joined(separator: " "))
        switch role {
        case "dd_add_to_cart":
            if text.contains("add to cart") && !text.contains("add item to cart") {
                return 1.4
            }
            if text.contains("add item to cart") {
                return -0.6
            }
            return 0.0
        case "dd_cart_cta":
            if text.contains("view cart") {
                return 1.2
            }
            if text.contains("open cart") || text.contains("open order cart") {
                return 0.2
            }
            return 0.0
        case "dd_store_card":
            if text.contains("domino") || text.contains("pizza") || text.contains("taco") || text.contains("burger") || text.contains("johns") || text.contains("caesars") {
                return 0.5
            }
            return 0.0
        default:
            return 0.0
        }
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
        let textTokens = Set(text.split(separator: " ").map(String.init))
        var score = 0.0
        var matches: [String] = []
        for target in targets {
            let variants = targetVariants(for: target.term)
            guard !variants.isEmpty else { continue }
            if variants.contains(where: { text.contains($0) }) {
                score += target.weight
                matches.append(target.term)
                continue
            }
            let targetTokens = variants
                .flatMap { $0.split(separator: " ").map(String.init) }
                .filter { $0.count > 2 }
            guard !targetTokens.isEmpty else { continue }
            let exactTokenMatch = Set(targetTokens).allSatisfy { token in
                textTokens.contains(token)
            }
            if exactTokenMatch {
                score += target.weight * 0.60
                matches.append(target.term)
                continue
            }
            let fuzzyTokenMatch = Set(targetTokens).allSatisfy { token in
                textTokens.contains(where: { candidateToken in
                    fuzzyTokenEqual(token, candidateToken)
                })
            }
            if fuzzyTokenMatch {
                score += target.weight * 0.45
                matches.append(target.term)
            }
        }
        return (score: score, matchedTerms: Array(dictUnique(matches)))
    }

    private static func targetVariants(for term: String) -> [String] {
        guard !term.isEmpty else {
            return []
        }
        var variants = [term]
        let words = term.split(separator: " ").map(String.init)
        if !words.isEmpty {
            let strippedWords = words.map { stripTokenNoise($0) }.filter { !$0.isEmpty }
            if !strippedWords.isEmpty {
                variants.append(strippedWords.joined(separator: " "))
            }
        }
        return Array(dictUnique(variants.map { normalizedText($0) }.filter { !$0.isEmpty }))
    }

    private static func stripTokenNoise(_ token: String) -> String {
        var clean = token
        if clean.hasSuffix("oes") && clean.count > 5 {
            clean = String(clean.dropLast(2))
        }
        if clean.hasSuffix("s") && clean.count > 4 {
            clean.removeLast()
        }
        return clean
    }

    private static func fuzzyTokenEqual(_ left: String, _ right: String) -> Bool {
        if left == right {
            return true
        }
        let normalizedLeft = stripTokenNoise(left)
        let normalizedRight = stripTokenNoise(right)
        if normalizedLeft == normalizedRight {
            return true
        }
        let shorter = normalizedLeft.count <= normalizedRight.count ? normalizedLeft : normalizedRight
        let longer = normalizedLeft.count <= normalizedRight.count ? normalizedRight : normalizedLeft
        if shorter.count >= 5 && longer.hasPrefix(shorter) && (longer.count - shorter.count) <= 2 {
            return true
        }
        return false
    }

    private static func isIrreversibleCandidate(_ candidateText: String) -> Bool {
        let text = normalizedText(candidateText)
        let blockedTerms = [
            "place order",
            "submit order",
            "complete order",
            "purchase",
            "buy now",
            "payment",
            "pay now",
            "confirm order",
            "book ride",
            "reserve",
            "send message",
        ]
        return blockedTerms.contains { text.contains($0) }
    }

    private static func buttonTapPolicy(kind: String, url: String, label: String, context: String, blocked: Bool) -> (safety: String, reason: String) {
        guard kind == "button", url.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return ("not_applicable", "Link-backed candidates open by URL, not button tap")
        }
        if blocked {
            return ("blocked", "Final or irreversible action language")
        }
        let labelText = normalizedText(label)
        let fullText = normalizedText([label, context].joined(separator: " "))
        let text = labelText.isEmpty ? fullText : labelText
        let blockedTerms = [
            "place order",
            "submit order",
            "complete order",
            "purchase",
            "buy now",
            "payment",
            "pay now",
            "credit debit card",
            "paypal",
            "venmo",
            "cash app pay",
            "klarna",
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
            "checkout",
            "continue",
            "next",
            "tip",
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

    private static func buttonTapScript(domIndex: Int, allowCaution: Bool) -> String {
        let allowCautionLiteral = allowCaution ? "true" : "false"
        return """
        (() => {
          const targetIndex = \(domIndex);
          const allowCaution = \(allowCautionLiteral);
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
          const labelText = normalized(label);
          const text = normalized([label, context].join(' '));
          const sensitiveText = labelText || text;
          const blockedTerms = [
            'place order',
            'submit order',
            'complete order',
            'purchase',
            'buy now',
            'payment',
            'pay now',
            'confirm order',
            'book ride',
            'reserve',
            'send',
            'delete'
          ];
          if (blockedTerms.some((term) => sensitiveText.includes(term))) {
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
            if (!allowCaution) {
              return { ok: false, reason: 'button_not_policy_safe', label };
            }
            target.scrollIntoView({ block: 'center', inline: 'center' });
            target.click();
            return { ok: true, reason: 'tapped_reviewed_button', label };
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
      const absoluteHref = (el) => {
        const rawHref = el?.getAttribute?.('href') || '';
        try {
          return rawHref ? new URL(rawHref, location.href).href : '';
        } catch (_) {
          return rawHref;
        }
      };
      const interactiveElements = Array.from(document.querySelectorAll('a[href],button,[role="button"],input[type="button"],input[type="submit"]'))
        .filter(visible);
      interactiveElements.forEach((el, index) => {
        try {
          el.dataset.memlaIndex = String(index);
        } catch (_) {}
      });
      const elementIndex = (el) => Number(el?.dataset?.memlaIndex ?? interactiveElements.indexOf(el));
      const labelForElement = (el) => clean(el?.innerText || el?.value || el?.getAttribute?.('aria-label') || el?.getAttribute?.('title') || absoluteHref(el));
      const contextForElement = (el, rootOverride) => clean((rootOverride || el.closest('article,li,section,div,[role="dialog"]'))?.innerText || '').slice(0, 420);
      const candidateFromElement = (el, role, rootOverride, overrideLabel) => ({
        id: String(elementIndex(el)),
        kind: el.matches('a[href]') ? 'link' : 'button',
        label: clean(overrideLabel || labelForElement(el)),
        url: absoluteHref(el),
        context: contextForElement(el, rootOverride),
        service_role: role
      });
      const collectText = (selector, limit, mapper, root = document) => Array.from(root.querySelectorAll(selector))
        .filter(visible)
        .map(mapper)
        .map(clean)
        .filter(Boolean)
        .slice(0, limit);

      const headings = collectText('h1,h2,h3,[role="heading"]', 10, (el) => el.innerText || el.textContent);
      const inputs = collectText('input,textarea,select', 18, (el) => el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.name || el.id || el.type || el.tagName);
      const buttons = collectText('button,[role="button"],input[type="button"],input[type="submit"],a[role="button"]', 24, (el) => el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title'));
      const links = collectText('a[href]', 24, (el) => el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || el.href);
      const candidates = interactiveElements
        .map((el) => {
          return {
            id: String(elementIndex(el)),
            kind: el.matches('a[href]') ? 'link' : 'button',
            label: labelForElement(el),
            url: absoluteHref(el),
            context: contextForElement(el)
          };
        })
        .filter((candidate) => candidate.label || candidate.url)
        .slice(0, 40);

      let doordashCandidates = [];
      let doordashActiveLayer = 'page';
      let doordashModalTitle = '';
      let doordashStoreCardCount = 0;
      let doordashItemCardCount = 0;
      let doordashTipOptionCount = 0;
      let doordashHasCartCTA = false;
      let doordashHasContinueCTA = false;
      let doordashHasAddToCartCTA = false;
      let doordashHasAddressCTA = false;
      let doordashHasPaymentSheet = false;
      let doordashHasCartCloseCTA = false;

      if (/doordash\\.com$/i.test(location.hostname)) {
        const pushUnique = (collection, candidate) => {
          if (!candidate || (!candidate.label && !candidate.url)) return;
          const key = [candidate.service_role || '', candidate.label || '', candidate.url || '', candidate.id || ''].join('|');
          if (collection.some((existing) => [existing.service_role || '', existing.label || '', existing.url || '', existing.id || ''].join('|') === key)) {
            return;
          }
          collection.push(candidate);
        };
        const rawLines = (el) => String(el?.innerText || '')
          .split('\\n')
          .map(clean)
          .filter(Boolean);
        const firstMeaningfulTextLine = (el) => {
          const lines = rawLines(el);
          return lines.find((line) => {
            const lower = line.toLowerCase();
            return /[a-z]/i.test(line)
              && !/^(get to know us|let us help you|doing business|welcome back!?|convenience & drugstores|deals & benefits|featured items|most ordered|restaurant|grocery|all)$/i.test(lower)
              && !/lower fees|delivery fee|dashpass|pickup|minutes|mins| mi\\b|\\bmin\\b|reviews?|\\(\\d\\+\\)/i.test(lower);
          }) || lines[0] || '';
        };
        const closestWithText = (el, minLength = 20, maxLength = 260) => {
          let node = el;
          let fallback = el;
          while (node && node !== document.body) {
            const length = clean(node.innerText || '').length;
            if (length >= minLength && length <= maxLength) {
              return node;
            }
            if (length >= minLength) {
              fallback = node;
            }
            node = node.parentElement;
          }
          return fallback || el;
        };
        const ancestorSet = (el) => {
          const nodes = new Set();
          let node = el;
          while (node) {
            nodes.add(node);
            node = node.parentElement;
          }
          return nodes;
        };
        const sharedAncestor = (left, right) => {
          if (!left && !right) return null;
          if (!left) return closestWithText(right, 20);
          if (!right) return closestWithText(left, 20);
          const leftAncestors = ancestorSet(left);
          let node = right;
          while (node) {
            if (leftAncestors.has(node)) {
              return node;
            }
            node = node.parentElement;
          }
          return closestWithText(left, 20);
        };
        const isDoorDashMenuNavLabel = (label, text) => {
          const cleanLabel = clean(label).toLowerCase();
          const lowerText = clean(text).toLowerCase();
          if (/\\$\\d/.test(lowerText)) return false;
          if (/^(all|featured items|most ordered|breads|chicken & wings|chicken wings|oven-baked pastas|oven baked pastas|pastas|desserts|salads|drinks|sandwiches|wings|sides|beverages)$/.test(cleanLabel)) {
            return true;
          }
          if ((/specialty pizzas?|build your own(?: pizza)?/.test(cleanLabel) || /specialty pizzas?|build your own(?: pizza)?/.test(lowerText)) && !/\\$\\d/.test(lowerText)) {
            return true;
          }
          return false;
        };
        const isDoorDashNoise = (text) => /open app|login|log in|become a dasher|merchant|gift cards|notification bell|save$|see all$|pricing & fees|open menu/i.test(text);

        const continueAnchor = Array.from(document.querySelectorAll('a[href*="/consumer/checkout"]'))
          .filter(visible)
          .find((el) => /continue/i.test(labelForElement(el)));
        const cartCloseButton = Array.from(document.querySelectorAll('button[data-testid="dbd-pre-checkout-header-close-button"], button[aria-label="Close"]'))
          .filter(visible)[0] || null;

        const visibleDialogs = Array.from(document.querySelectorAll('[role="dialog"], [aria-modal="true"]')).filter(visible);
        const paymentSheet = visibleDialogs.find((el) => /select payment method|credit\\/debit card|paypal|venmo|cash app pay|klarna/i.test(clean(el.innerText)));
        const itemModal = visibleDialogs.find((el) => /add to cart|choose your size|recommended options|build your own pizza|custom pizza/i.test(clean(el.innerText)));
        const cartDrawer = continueAnchor || cartCloseButton ? sharedAncestor(continueAnchor, cartCloseButton) : null;
        const activeRoot = paymentSheet || cartDrawer || itemModal || null;
        doordashActiveLayer = paymentSheet ? 'payment_sheet' : cartDrawer ? 'cart_drawer' : itemModal ? 'item_modal' : 'page';
        doordashHasPaymentSheet = Boolean(paymentSheet);
        doordashHasCartCloseCTA = Boolean(cartCloseButton);
        doordashModalTitle = doordashActiveLayer === 'cart_drawer'
          ? 'Cart'
          : clean((activeRoot?.querySelector('h1,h2,h3,[role="heading"]')?.innerText) || '');

        const storeCardElements = Array.from(document.querySelectorAll('[data-anchor-id="StoreCard"]'))
          .filter(visible)
          .slice(0, 10);
        doordashStoreCardCount = storeCardElements.length;
        storeCardElements.forEach((anchor) => {
          const cardRoot = closestWithText(anchor.parentElement || anchor, 24, 220);
          const headingLabel = Array.from(cardRoot.querySelectorAll('h1,h2,h3,h4,[role="heading"]'))
            .map((node) => clean(node.innerText || ''))
            .find((line) => line && !/^(get to know us|let us help you|doing business|welcome back!?|convenience & drugstores|deals & benefits|featured items|most ordered)$/i.test(line));
          const label = headingLabel || firstMeaningfulTextLine(cardRoot) || labelForElement(anchor);
          pushUnique(doordashCandidates, candidateFromElement(anchor, 'dd_store_card', cardRoot, label));
        });

        const cartReviewButtons = Array.from(document.querySelectorAll('button,a[href],[role="button"]'))
          .filter(visible)
          .filter((el) => /view cart|items? in cart|open order cart/i.test(labelForElement(el)));
        cartReviewButtons.slice(0, 4).forEach((button) => {
          doordashHasCartCTA = true;
          const root = closestWithText(button, 12, 180);
          const label = /view cart/i.test(labelForElement(button)) ? 'View cart' : 'Open cart';
          pushUnique(doordashCandidates, candidateFromElement(button, 'dd_cart_cta', root, label));
        });

        const cartButton = Array.from(document.querySelectorAll('button[data-testid="OrderCartIconButton"], button[aria-controls="order-cart"]'))
          .filter(visible)[0];
        if (cartButton) {
          doordashHasCartCTA = true;
          pushUnique(doordashCandidates, candidateFromElement(cartButton, 'dd_cart_cta', cartButton.closest('section,div'), 'Open cart'));
        }

        if (continueAnchor) {
          doordashHasContinueCTA = true;
          pushUnique(doordashCandidates, candidateFromElement(continueAnchor, 'dd_continue_cta', cartDrawer || continueAnchor.closest('section,div'), 'Continue'));
        }

        if (cartCloseButton && cartDrawer) {
          pushUnique(doordashCandidates, candidateFromElement(cartCloseButton, 'dd_modal_close', cartDrawer, 'Close cart'));
        }

        if (itemModal) {
          const modalClose = Array.from(itemModal.querySelectorAll('button,[role="button"]'))
            .filter(visible)
            .find((el) => /^x$/i.test(labelForElement(el)) || /close|dismiss/i.test(labelForElement(el)));
          if (modalClose) {
            pushUnique(doordashCandidates, candidateFromElement(modalClose, 'dd_modal_close', itemModal, 'Close item'));
          }

          const addToCartButtons = Array.from(document.querySelectorAll('button,[role="button"],a[href]'))
            .filter(visible)
            .filter((el) => /add to cart/i.test(labelForElement(el)));
          addToCartButtons.forEach((button) => {
            doordashHasAddToCartCTA = true;
            const root = sharedAncestor(itemModal, button) || closestWithText(button, 18, 240);
            pushUnique(doordashCandidates, candidateFromElement(button, 'dd_add_to_cart', root, 'Add to cart'));
          });

          const modalOptionButtons = Array.from(itemModal.querySelectorAll('button,[role="button"],a[href]'))
            .filter(visible)
            .filter((el) => {
              const label = labelForElement(el);
              const text = clean([label, contextForElement(el, itemModal)].join(' '));
              if (!text || isDoorDashNoise(text) || /close|dismiss|add special instructions|add to cart/i.test(text)) {
                return false;
              }
              return /ordered recently|small|medium|large|hand tossed|crust|seasoning|pizza|\\$\\d|recommended/i.test(text);
            })
            .slice(0, 14);
          modalOptionButtons.forEach((el) => {
            doordashItemCardCount += 1;
            const root = closestWithText(el, 18);
            const label = firstMeaningfulTextLine(root) || labelForElement(el);
            pushUnique(doordashCandidates, candidateFromElement(el, 'dd_item_card', root, label));
          });
        } else if (!cartDrawer) {
          const quickAddButtons = Array.from(document.querySelectorAll('button[data-testid="quick-add-button"]'))
            .filter(visible)
            .slice(0, 8);
          quickAddButtons.forEach((button) => {
            doordashHasAddToCartCTA = true;
            pushUnique(doordashCandidates, candidateFromElement(button, 'dd_add_to_cart', closestWithText(button, 18), 'Add to cart'));
          });

          const explicitStoreItemCards = Array.from(document.querySelectorAll('[role="button"][aria-label], button[aria-label]'))
            .filter(visible)
            .filter((el) => {
              const label = labelForElement(el);
              const root = closestWithText(el, 18, 240);
              const text = clean([label, contextForElement(el, root)].join(' '));
              if (!label || isDoorDashNoise(text)) {
                return false;
              }
              if (/open menu|notification bell|group order|delivery|pickup|save|see more|search stores/i.test(text)) {
                return false;
              }
              return /build your own|specialty pizza|custom pizza|pizza|wings|bread|pasta|dessert|sandwich|salad|bites/i.test(label)
                && (/\\$\\d/.test(text) || label.length >= 6);
            })
            .slice(0, 18);
          explicitStoreItemCards.forEach((el) => {
            doordashItemCardCount += 1;
            const root = closestWithText(el, 18, 240);
            const itemLabel = firstMeaningfulTextLine(root) || clean(labelForElement(el));
            pushUnique(doordashCandidates, candidateFromElement(el, 'dd_item_card', root, itemLabel));
          });

          const storefrontElements = Array.from(document.querySelectorAll('button,a[href],[role="button"]'))
            .filter(visible)
            .slice(0, 140);
          storefrontElements.forEach((el) => {
            const label = labelForElement(el);
            const root = closestWithText(el, 18);
            const context = contextForElement(el, root);
            const text = clean([label, context].join(' '));
            if (!text || isDoorDashNoise(text)) return;
            if ((el.getAttribute('aria-controls') || '') === 'order-cart' || (el.getAttribute('data-testid') || '') === 'OrderCartIconButton') {
              return;
            }
            if (/continue/i.test(text) && !/continue browsing/i.test(text)) {
              doordashHasContinueCTA = true;
              pushUnique(doordashCandidates, candidateFromElement(el, 'dd_continue_cta', root, 'Continue'));
              return;
            }
            if (/add to cart/i.test(text)) {
              doordashHasAddToCartCTA = true;
              pushUnique(doordashCandidates, candidateFromElement(el, 'dd_add_to_cart', root, 'Add to cart'));
              return;
            }
            if (isDoorDashMenuNavLabel(label, text)) {
              pushUnique(doordashCandidates, candidateFromElement(el, 'dd_menu_nav', root, clean(label)));
              return;
            }
            if (/build your own|specialty pizza|custom pizza|pizza|wings|bread|pasta|dessert|sandwich|salad|bites/i.test(text) && (/\\$\\d/.test(text) || (el.getAttribute('role') || '') === 'button' || (el.getAttribute('aria-label') || ''))) {
              doordashItemCardCount += 1;
              const itemLabel = firstMeaningfulTextLine(root) || clean(label);
              pushUnique(doordashCandidates, candidateFromElement(el, 'dd_item_card', root, itemLabel));
            }
          });
        }

        const tipButtons = Array.from(document.querySelectorAll('button[data-anchor-id="TipPickerOption"]'))
          .filter(visible)
          .slice(0, 6);
        doordashTipOptionCount = tipButtons.length;
        tipButtons.forEach((button) => {
          pushUnique(doordashCandidates, candidateFromElement(button, 'dd_tip_option', button.closest('[role="radiogroup"],section,div'), labelForElement(button)));
        });

        const addressButtons = Array.from(document.querySelectorAll('button,[role="button"],a[href]'))
          .filter(visible)
          .filter((el) => /\\d{3,} .*\\b(st|street|rd|road|ct|court|ave|avenue|blvd|lane|ln|drive|dr)\\b/i.test(clean(el.innerText || '')))
          .slice(0, 2);
        doordashHasAddressCTA = addressButtons.length > 0;
        addressButtons.forEach((button) => {
          pushUnique(doordashCandidates, candidateFromElement(button, 'dd_address_cta', button.closest('section,div'), labelForElement(button)));
        });

        if (paymentSheet) {
          Array.from(paymentSheet.querySelectorAll('button,[role="button"]'))
            .filter(visible)
            .forEach((button) => {
              pushUnique(doordashCandidates, candidateFromElement(button, 'dd_payment_method', paymentSheet, labelForElement(button)));
            });
        }
      }

      const text = clean(document.body ? document.body.innerText : '');
      return {
        title: document.title || '',
        url: location.href,
        text_snippet: text.slice(0, 1000),
        headings,
        inputs,
        buttons,
        links,
        candidates,
        doordash_candidates: doordashCandidates.slice(0, 40),
        doordash_active_layer: doordashActiveLayer,
        doordash_modal_title: doordashModalTitle,
        doordash_store_card_count: doordashStoreCardCount,
        doordash_item_card_count: doordashItemCardCount,
        doordash_tip_option_count: doordashTipOptionCount,
        doordash_has_cart_cta: doordashHasCartCTA,
        doordash_has_continue_cta: doordashHasContinueCTA,
        doordash_has_add_to_cart_cta: doordashHasAddToCartCTA,
        doordash_has_address_cta: doordashHasAddressCTA,
        doordash_has_payment_sheet: doordashHasPaymentSheet,
        doordash_has_cart_close_cta: doordashHasCartCloseCTA
      };
    })();
    """

    private static let doorDashStorefrontPeekScript = """
    (() => {
      const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
      const visible = (el) => {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 1 && rect.height > 1;
      };
      const candidates = Array.from(document.querySelectorAll('[role="button"][aria-label], button[aria-label], h2, h3, [role="heading"]'))
        .filter(visible);
      const target = candidates.find((el) => {
        const text = clean(el.getAttribute('aria-label') || el.innerText || '');
        return /build your own|specialty pizza|featured items|most ordered|pizza|wings|breads|dessert|sandwich|salad|pasta/i.test(text);
      });
      if (target && typeof target.scrollIntoView === 'function') {
        target.scrollIntoView({ block: 'center', inline: 'nearest' });
        return { ok: true, reason: 'scrolled_to_menu_target' };
      }
      window.scrollBy(0, Math.min(Math.max(window.innerHeight * 0.9, 320), 720));
      return { ok: true, reason: 'scrolled_storefront_down' };
    })();
    """

    private static let pageGroundingProbeScript = """
    (() => {
      const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
      const visible = (el) => {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 1 && rect.height > 1;
      };
      const visibleControls = Array.from(document.querySelectorAll('h1,h2,h3,button,a[href],[role="dialog"],[aria-modal="true"]'))
        .filter(visible)
        .slice(0, 14)
        .map((el) => clean(el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('title')))
        .filter(Boolean)
        .join(' | ')
        .slice(0, 320);
      const bodyText = clean(document.body ? document.body.innerText : '').slice(0, 320);
      return {
        fingerprint: [location.href, document.title || '', visibleControls, bodyText].join(' || ')
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
    @State private var isC2AConsoleClosed = false
    @State private var isRawPageVisible = false
    @State private var authNotes: [String: String] = [:]
    @State private var autoDriveEnabled = true
    @State private var autoDriveStatus = "Memla auto-drive is ready."
    @State private var lastAutoDriveSignature = ""
    @State private var pendingDoorDashRole: String = ""
    @State private var pendingDoorDashLabel: String = ""
    @State private var addToCartRetryCount = 0
    @State private var preferCartProgress = false

    private struct MirrorAutoDriveAction {
        let candidate: WebsiteC2ACandidate
        let allowCaution: Bool
        let status: String
        let pendingRole: String
    }

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
                if let state = browser.websiteState {
                    mirrorPrimarySurface(for: state)
                } else {
                    mirrorLoadingSurface
                }
                rawPageSurface
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
                autoDriveEnabled = true
                autoDriveStatus = "Memla auto-drive is ready."
                lastAutoDriveSignature = ""
                pendingDoorDashRole = ""
                pendingDoorDashLabel = ""
                addToCartRetryCount = 0
                preferCartProgress = false
                browser.startGrounding(capsule: route.capsule)
                if browser.currentURL.isEmpty {
                    browser.navigate(to: route.url, autoInspect: true, capsule: route.capsule)
                }
            }
            .onDisappear {
                browser.stopGrounding()
            }
            .onReceive(browser.$websiteState.compactMap { $0 }) { state in
                handleAutoDriveUpdate(for: state)
            }
        }
    }

    private var hasC2AConsole: Bool {
        browser.websiteState != nil || !browser.inspectionStatus.isEmpty
    }

    private var mirrorLoadingSurface: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                mirrorSectionCard(title: "Memla's Mirror", systemImage: "sparkles.rectangle.stack") {
                    Text(browser.inspectionStatus.isEmpty ? "Memla is loading the page and preparing a distilled task surface." : browser.inspectionStatus)
                        .font(.subheadline)
                    Text("The raw website stays underneath as execution substrate, but the goal is for you to mostly interact with Memla's simplified view.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        Button(browser.isInspecting ? "Inspecting..." : "Inspect Page") {
                            browser.inspectPage(capsule: route.capsule)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(browser.isInspecting || browser.isLoading)

                        Button(isRawPageVisible ? "Hide Raw Page" : "Show Raw Page") {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                isRawPageVisible.toggle()
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding(12)
        }
        .background(Color(.systemGroupedBackground))
    }

    private var rawPageSurface: some View {
        VStack(spacing: 0) {
            Divider()
            ZStack(alignment: .top) {
                VStack(spacing: 0) {
                    if isRawPageVisible {
                        browserToolbar
                        Divider()
                    }
                    ZStack(alignment: .bottom) {
                        MemlaBrowserWebView(browser: browser)
                            .frame(height: isRawPageVisible ? 360 : 360)
                            .frame(height: isRawPageVisible ? 360 : 1, alignment: .top)
                            .opacity(isRawPageVisible ? 1 : 0.02)
                            .allowsHitTesting(isRawPageVisible)
                            .clipped()
                        if isRawPageVisible, hasC2AConsole, !isC2AConsoleClosed {
                            websiteC2AConsole
                        }
                    }
                }
                if !isRawPageVisible {
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Raw Page Hidden")
                                .font(.caption.weight(.semibold))
                            Text("Memla is keeping the website mounted underneath so the mirror can keep reading and steering it.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                        Spacer()
                        Button("Show Raw Page") {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                isRawPageVisible = true
                                isC2AConsoleClosed = false
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                    .background(Color(.secondarySystemBackground))
                }
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
                    isC2AConsoleClosed = false
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
                Text("Memla's Mirror")
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
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isC2AConsoleClosed = true
                    }
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.bordered)
                .font(.caption2)
            }

            if let state = browser.websiteState {
                if isC2AConsoleExpanded {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 10) {
                            mirrorCompactContent(for: state)
                            Divider()
                            Text("Raw Website C2A")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            expandedWebsiteC2AContent(for: state)
                        }
                    }
                    .frame(maxHeight: 360)
                } else {
                    mirrorCompactContent(for: state)
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

    private func mirrorCompactContent(for state: WebsiteC2AState) -> some View {
        let step = guidedStep(for: state)
        let candidates = mirrorCandidates(for: state)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: mirrorIcon(for: state))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(mirrorColor(for: state))
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 3) {
                    Text(mirrorTitle(for: state))
                        .font(.caption.weight(.semibold))
                    Text(mirrorSummary(for: state, step: step))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
                Spacer(minLength: 0)
            }
            if state.pageKind.hasPrefix("dd_"), !autoDriveStatus.isEmpty {
                Label(autoDriveStatus, systemImage: "bolt.fill")
                    .font(.caption2)
                    .foregroundStyle(.green)
                    .lineLimit(2)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(mirrorFacts(for: state)) { fact in
                        capsuleChip(title: fact.title, value: fact.value)
                    }
                }
            }
            if !candidates.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(alignment: .top, spacing: 8) {
                        ForEach(Array(candidates.prefix(3))) { candidate in
                            mirrorCandidateCard(candidate)
                        }
                    }
                }
            }
            if let verification = state.capsuleVerification, (state.pageKind == "cart" || state.pageKind == "checkout") {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Mirror Verification")
                        .font(.caption.weight(.semibold))
                    Text(verification.summary)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
                .padding(8)
                .background(Color.blue.opacity(0.10), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
        }
    }

    private func mirrorPrimarySurface(for state: WebsiteC2AState) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                mirrorSectionCard(title: "Memla's Mirror", systemImage: mirrorIcon(for: state)) {
                    mirrorCompactContent(for: state)
                }

                mirrorSectionCard(title: "Next Move", systemImage: "point.topleft.down.curvedto.point.bottomright.up") {
                    guidedStepControls(for: state)
                    HStack(spacing: 8) {
                        Button(browser.isInspecting ? "Inspecting..." : "Refresh Mirror") {
                            browser.inspectPage(capsule: route.capsule)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(browser.isInspecting || browser.isLoading)

                        Button(isRawPageVisible ? "Hide Raw Page" : "Show Raw Page") {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                isRawPageVisible.toggle()
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }

                if state.authState == "login_required" || state.authState == "likely_signed_in" {
                    mirrorSectionCard(title: "Session", systemImage: "person.badge.key") {
                        authBridgeControls(for: state)
                    }
                }

                if !mirrorItemCandidates(for: state).isEmpty {
                    mirrorSectionCard(title: mirrorItemSectionTitle(for: state), systemImage: "square.grid.2x2") {
                        mirrorCandidateScroller(mirrorItemCandidates(for: state))
                    }
                }

                if !mirrorControlCandidates(for: state).isEmpty {
                    mirrorSectionCard(title: "Mirror Controls", systemImage: "slider.horizontal.3") {
                        mirrorCandidateScroller(mirrorControlCandidates(for: state))
                    }
                }

                if let verification = state.capsuleVerification {
                    mirrorSectionCard(title: state.pageKind == "checkout" ? "Final Review" : "Checkpoint Verification", systemImage: state.pageKind == "checkout" ? "hand.raised.fill" : "checklist") {
                        capsuleVerificationControls(for: state)
                        if !verification.missing.isEmpty {
                            Text("Still missing: \(verification.missing.map { readableRequirement($0) }.joined(separator: ", "))")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                                .lineLimit(3)
                        }
                    }
                }

                if !bridgeSuggestions(for: state).isEmpty {
                    mirrorSectionCard(title: "Recovery Bridges", systemImage: "arrow.triangle.branch") {
                        bridgeSuggestionControls(for: state)
                    }
                }

                if searchQueryIfAvailable(for: state) != nil {
                    mirrorSectionCard(title: "Search Primitive", systemImage: "magnifyingglass") {
                        searchActionControls(for: state)
                    }
                }
            }
            .padding(12)
        }
        .background(Color(.systemGroupedBackground))
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
                        capsuleChip(title: "Auth", value: readableRequirement(state.authState))
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
                authBridgeControls(for: state)
                capsuleVerificationControls(for: state)
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

    private func mirrorCandidates(for state: WebsiteC2AState) -> [WebsiteC2ACandidate] {
        let deprioritizedRoles = Set(["review_link", "accessibility_link", "utility_link", "auth_link", "nav_link", "dd_menu_nav"])
        let preferredRoles = Set(["store_link", "menu_item", "item_action_button", "cart_link", "cart_button", "control_button", "checkout_button", "dd_store_card", "dd_item_card", "dd_add_to_cart", "dd_continue_cta", "dd_cart_cta", "dd_tip_option", "dd_address_cta", "dd_modal_close"])
        let visibleCandidates = state.candidates.filter { !deprioritizedRoles.contains($0.role) && !$0.blocked }
        let stateScopedRoles: Set<String>
        switch state.pageKind {
        case "dd_search_results":
            stateScopedRoles = ["dd_store_card", "dd_cart_cta"]
        case "dd_storefront":
            stateScopedRoles = ["dd_item_card", "dd_add_to_cart", "dd_cart_cta"]
        case "dd_item_modal":
            stateScopedRoles = ["dd_add_to_cart", "dd_item_card", "dd_cart_cta", "dd_modal_close"]
        case "dd_cart_drawer", "dd_cart_page":
            stateScopedRoles = ["dd_cart_cta", "dd_continue_cta", "dd_tip_option", "dd_address_cta", "dd_modal_close"]
        case "dd_checkout":
            stateScopedRoles = ["dd_tip_option", "dd_address_cta", "dd_modal_close"]
        case "dd_payment_sheet":
            stateScopedRoles = ["dd_modal_close"]
        default:
            stateScopedRoles = preferredRoles
        }
        let stateScoped = visibleCandidates.filter { candidate in
            stateScopedRoles.contains(candidate.role)
        }
        let candidatePool = stateScoped.isEmpty ? visibleCandidates : stateScoped
        let preferred = candidatePool.filter { candidate in
            candidate.score > 0 || preferredRoles.contains(candidate.role)
        }
        let effectivePool = preferred.isEmpty ? candidatePool : preferred
        let sorted = effectivePool.sorted { left, right in
            let leftPriority = mirrorRolePriority(pageKind: state.pageKind, role: left.role)
            let rightPriority = mirrorRolePriority(pageKind: state.pageKind, role: right.role)
            if leftPriority != rightPriority {
                return leftPriority > rightPriority
            }
            if left.score != right.score {
                return left.score > right.score
            }
            return left.label.localizedCaseInsensitiveCompare(right.label) == .orderedAscending
        }
        return Array(sorted.prefix(4))
    }

    private func mirrorRolePriority(pageKind: String, role: String) -> Int {
        switch pageKind {
        case "dd_search_results":
            switch role {
            case "dd_store_card":
                return 100
            case "dd_cart_cta":
                return 85
            default:
                return 10
            }
        case "dd_storefront":
            switch role {
            case "dd_item_card":
                return 100
            case "dd_add_to_cart":
                return 90
            case "dd_cart_cta":
                return 70
            case "dd_modal_close":
                return 5
            default:
                return 10
            }
        case "dd_item_modal":
            switch role {
            case "dd_add_to_cart":
                return 100
            case "dd_cart_cta":
                return 85
            case "dd_item_card":
                return 75
            case "dd_modal_close":
                return 10
            default:
                return 15
            }
        case "dd_cart_drawer", "dd_cart_page":
            switch role {
            case "dd_continue_cta":
                return 100
            case "dd_cart_cta":
                return 70
            case "dd_tip_option":
                return 60
            case "dd_address_cta":
                return 50
            case "dd_modal_close":
                return 5
            default:
                return 10
            }
        case "dd_checkout":
            switch role {
            case "dd_tip_option":
                return 100
            case "dd_address_cta":
                return 90
            case "dd_modal_close":
                return 10
            default:
                return 15
            }
        default:
            return 10
        }
    }

    private func mirrorFacts(for state: WebsiteC2AState) -> [WebsiteMirrorFact] {
        var facts: [WebsiteMirrorFact] = [
            WebsiteMirrorFact(id: "page", title: "Page", value: readableRequirement(state.pageKind)),
            WebsiteMirrorFact(id: "auth", title: "Auth", value: readableRequirement(state.authState)),
            WebsiteMirrorFact(id: "actions", title: "Moves", value: "\(state.safeActions.count)")
        ]
        if let verification = state.capsuleVerification {
            facts.append(WebsiteMirrorFact(id: "matched", title: "Matched", value: "\(verification.matched.count)"))
            if !verification.missing.isEmpty {
                facts.append(WebsiteMirrorFact(id: "missing", title: "Missing", value: "\(verification.missing.count)"))
            }
        }
        if let best = mirrorCandidates(for: state).first {
            facts.append(WebsiteMirrorFact(id: "best", title: "Best", value: best.label))
        }
        return facts
    }

    private func mirrorTitle(for state: WebsiteC2AState) -> String {
        switch state.pageKind {
        case "dd_search_results":
            return "DoorDash search results distilled"
        case "dd_storefront":
            return "DoorDash storefront distilled"
        case "dd_item_modal":
            return "DoorDash customizer is active"
        case "dd_cart_drawer":
            return "DoorDash cart drawer is active"
        case "dd_cart_page":
            return "DoorDash cart is ready"
        case "dd_checkout":
            return "DoorDash checkout detected"
        case "dd_payment_sheet":
            return "Payment sheet is active"
        case "login":
            return "Session needs recovery"
        case "search_results", "search_form":
            return "Mirror has a search surface"
        case "menu_or_item":
            return "Mirror distilled the item funnel"
        case "cart":
            return "Mirror sees a cart checkpoint"
        case "checkout":
            return "Mirror reached final review"
        default:
            return "Mirror distilled the current page"
        }
    }

    private func mirrorSummary(for state: WebsiteC2AState, step: WebsiteGuidedStep) -> String {
        if let best = mirrorCandidates(for: state).first {
            return "\(step.detail) Best distilled candidate: \(best.label)."
        }
        return step.detail
    }

    private func mirrorIcon(for state: WebsiteC2AState) -> String {
        switch state.pageKind {
        case "dd_search_results":
            return "building.2"
        case "dd_storefront":
            return "storefront"
        case "dd_item_modal":
            return "square.and.pencil"
        case "dd_cart_drawer", "dd_cart_page":
            return "cart"
        case "dd_checkout", "dd_payment_sheet":
            return "hand.raised.fill"
        case "login":
            return "person.badge.key"
        case "menu_or_item":
            return "square.grid.2x2"
        case "cart":
            return "cart"
        case "checkout":
            return "hand.raised.fill"
        default:
            return "sparkles.rectangle.stack"
        }
    }

    private func mirrorColor(for state: WebsiteC2AState) -> Color {
        switch state.pageKind {
        case "dd_search_results", "dd_storefront", "dd_item_modal", "dd_cart_drawer", "dd_cart_page":
            return .green
        case "dd_checkout", "dd_payment_sheet":
            return .red
        case "login":
            return .orange
        case "checkout":
            return .red
        case "menu_or_item", "cart":
            return .green
        default:
            return .blue
        }
    }

    private func mirrorCandidateCard(_ candidate: WebsiteC2ACandidate) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                Text(readableRequirement(candidate.role))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer(minLength: 8)
                Text(String(format: "%.1f", candidate.score))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(candidate.score > 0 ? .green : .secondary)
            }
            Text(candidate.label)
                .font(.caption.weight(.semibold))
                .lineLimit(2)
            if !candidate.matchedTerms.isEmpty {
                Text("Matched: \(candidate.matchedTerms.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            } else {
                Text(candidate.reason)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            candidateActionButton(candidate)
        }
        .frame(width: 205, alignment: .leading)
        .padding(8)
        .background(candidate.score > 0 ? Color.green.opacity(0.10) : Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func mirrorCandidateScroller(_ candidates: [WebsiteC2ACandidate]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(alignment: .top, spacing: 8) {
                ForEach(Array(candidates.prefix(6))) { candidate in
                    mirrorCandidateCard(candidate)
                }
            }
        }
    }

    private func mirrorItemCandidates(for state: WebsiteC2AState) -> [WebsiteC2ACandidate] {
        let itemRoles = Set(["store_link", "menu_item", "dd_store_card", "dd_item_card"])
        let filtered = mirrorCandidates(for: state).filter { itemRoles.contains($0.role) }
        return filtered.isEmpty ? mirrorCandidates(for: state).filter { !$0.url.isEmpty } : filtered
    }

    private func mirrorControlCandidates(for state: WebsiteC2AState) -> [WebsiteC2ACandidate] {
        let controlRoles = Set(["item_action_button", "cart_button", "cart_link", "control_button", "checkout_button", "dd_add_to_cart", "dd_continue_cta", "dd_cart_cta", "dd_tip_option", "dd_address_cta", "dd_modal_close"])
        return mirrorCandidates(for: state).filter { controlRoles.contains($0.role) }
    }

    private func mirrorItemSectionTitle(for state: WebsiteC2AState) -> String {
        switch state.pageKind {
        case "dd_search_results":
            return "Store Cards"
        case "dd_storefront":
            return "Menu Items"
        case "dd_item_modal":
            return "Customizer Items"
        case "dd_cart_drawer":
            return "Cart Signals"
        case "dd_checkout":
            return "Checkout Signals"
        case "search_results", "search_form":
            return "Relevant Results"
        case "menu_or_item":
            return "Relevant Items"
        case "cart":
            return "Cart Links"
        case "checkout":
            return "Checkout Signals"
        default:
            return "Relevant Candidates"
        }
    }

    private func mirrorSectionCard<Content: View>(title: String, systemImage: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
            content()
        }
        .padding(12)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
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
    private func authBridgeControls(for state: WebsiteC2AState) -> some View {
        if state.authState == "login_required" || state.authState == "likely_signed_in" {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Label("Auth Bridge", systemImage: "person.badge.key")
                        .font(.caption.weight(.semibold))
                    Spacer()
                    Text(state.authDomain.isEmpty ? "current site" : state.authDomain)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Text(authGuidance(for: state))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                HStack(spacing: 8) {
                    Button("I Signed In, Inspect Again") {
                        isC2AConsoleClosed = false
                        browser.inspectPage(capsule: route.capsule)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(browser.isInspecting || browser.isLoading)

                    Button("Mark Session Likely") {
                        rememberSessionLikely(for: state)
                    }
                    .buttonStyle(.bordered)
                    .disabled(state.authDomain.isEmpty)
                }
                if let note = rememberedAuthNote(for: state), !note.isEmpty {
                    Text(note)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            .padding(8)
            .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
    }

    @ViewBuilder
    private func capsuleVerificationControls(for state: WebsiteC2AState) -> some View {
        if let verification = state.capsuleVerification {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Label("Capsule Match", systemImage: state.pageKind == "checkout" ? "hand.raised.fill" : "checklist")
                        .font(.caption.weight(.semibold))
                    Spacer()
                    Text(readableRequirement(state.pageKind))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text(verification.summary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                if !verification.matched.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(verification.matched, id: \.self) { item in
                                Label(readableRequirement(item), systemImage: "checkmark.circle.fill")
                                    .font(.caption2)
                                    .foregroundStyle(.green)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 5)
                                    .background(Color.green.opacity(0.12), in: Capsule())
                            }
                        }
                    }
                }
                if !verification.missing.isEmpty {
                    Text("Missing: \(verification.missing.map { readableRequirement($0) }.joined(separator: ", "))")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .lineLimit(2)
                }
                if !verification.warnings.isEmpty {
                    Text("Policy: \(verification.warnings.map { readableRequirement($0) }.joined(separator: ", "))")
                        .font(.caption2)
                        .foregroundStyle(state.pageKind == "checkout" ? .red : .secondary)
                        .lineLimit(2)
                }
            }
            .padding(8)
            .background((state.pageKind == "checkout" ? Color.red : Color.blue).opacity(0.10), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
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
                                    Text(readableRequirement(candidate.role))
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
            Button(candidateOpenTitle(candidate)) {
                openCandidate(candidate)
            }
            .buttonStyle(.bordered)
        } else if candidate.kind == "button" && candidate.tapSafety == "safe" {
            Button(browser.isRunningButtonAction ? "Tapping..." : candidateTapTitle(candidate, allowCaution: false)) {
                tapCandidate(candidate)
            }
            .buttonStyle(.borderedProminent)
            .disabled(browser.isRunningButtonAction)
        } else if candidate.kind == "button" && candidate.tapSafety == "caution" {
            Button(browser.isRunningButtonAction ? "Tapping..." : candidateTapTitle(candidate, allowCaution: true)) {
                tapCandidate(candidate, allowCaution: true)
            }
            .buttonStyle(.bordered)
            .tint(.orange)
            .disabled(browser.isRunningButtonAction)
        } else {
            Button("No Action") {
            }
            .buttonStyle(.bordered)
            .disabled(true)
        }
    }

    private func candidateOpenTitle(_ candidate: WebsiteC2ACandidate) -> String {
        switch candidate.role {
        case "dd_store_card":
            return "Open Store"
        case "dd_item_card":
            return "Open Item"
        case "dd_continue_cta":
            return "Continue"
        case "cart_link", "dd_cart_cta":
            return "Open Cart"
        default:
            return "Open"
        }
    }

    private func candidateTapTitle(_ candidate: WebsiteC2ACandidate, allowCaution: Bool) -> String {
        switch candidate.role {
        case "dd_cart_cta":
            return "Open Cart"
        case "dd_continue_cta":
            return "Continue"
        case "dd_add_to_cart":
            return "Add To Cart"
        case "dd_tip_option":
            return "Set Tip"
        case "dd_modal_close":
            return "Dismiss"
        case "checkout_button":
            return "Enter Checkout"
        default:
            if allowCaution {
                return candidate.label.localizedCaseInsensitiveContains("checkout") ? "Enter Checkout" : "Tap Reviewed"
            }
            return "Tap"
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

    private func authGuidance(for state: WebsiteC2AState) -> String {
        if state.authState == "login_required" {
            return "Sign in manually inside this browser if you trust the site, then tap inspect again. Memla does not store passwords or bypass 2FA/captcha."
        }
        if state.authState == "likely_signed_in" {
            return "This page has account/cart/address hints, so the session may be active. Continue with capsule verification before final checkout."
        }
        return "Session status is unknown. Inspect after any manual sign-in to refresh Website C2A."
    }

    private func authMemoryKey(for state: WebsiteC2AState) -> String {
        "memla.auth.\(state.authDomain)"
    }

    private func rememberSessionLikely(for state: WebsiteC2AState) {
        guard !state.authDomain.isEmpty else {
            return
        }
        let note = "Session marked likely for \(state.authDomain). Memla stores no credentials."
        UserDefaults.standard.set(note, forKey: authMemoryKey(for: state))
        authNotes[state.authDomain] = note
    }

    private func rememberedAuthNote(for state: WebsiteC2AState) -> String? {
        guard !state.authDomain.isEmpty else {
            return nil
        }
        if let note = authNotes[state.authDomain] {
            return note
        }
        return UserDefaults.standard.string(forKey: authMemoryKey(for: state))
    }

    private func guidedStep(for state: WebsiteC2AState) -> WebsiteGuidedStep {
        if state.pageKind == "dd_payment_sheet" {
            return WebsiteGuidedStep(
                title: "Stop at payment sheet",
                detail: "DoorDash opened payment selection. This is a hard human boundary, so Memla should not pick a payment method or continue.",
                icon: "hand.raised.fill",
                tone: "danger"
            )
        }
        if state.pageKind == "dd_checkout" {
            return WebsiteGuidedStep(
                title: "Review checkout details",
                detail: "Address, tip, order summary, and total are visible. Verify them here, then leave payment and final order submission to the user.",
                icon: "checklist",
                tone: "warning"
            )
        }
        if state.pageKind == "dd_cart_drawer" || state.pageKind == "dd_cart_page" {
            return WebsiteGuidedStep(
                title: "Move from cart to checkout",
                detail: "DoorDash cart state is active. Suppress store browsing and focus on cart, continue, tip, and address controls.",
                icon: "cart.badge.questionmark",
                tone: "warning"
            )
        }
        if state.pageKind == "dd_item_modal" {
            return WebsiteGuidedStep(
                title: "Use the customizer layer",
                detail: "The active DoorDash item modal is open. Focus on size, quantity, modifiers, and Add to cart, not the background storefront.",
                icon: "slider.horizontal.3",
                tone: "safe"
            )
        }
        if state.pageKind == "dd_storefront" {
            return WebsiteGuidedStep(
                title: "Pick a menu item",
                detail: "DoorDash storefront detected. Prefer food item cards and suppress category or marketing links.",
                icon: "fork.knife",
                tone: "safe"
            )
        }
        if state.pageKind == "dd_search_results" {
            return WebsiteGuidedStep(
                title: "Pick the right store card",
                detail: "DoorDash search results are visible. Prefer merchant cards that match the requested restaurant, rating, and distance.",
                icon: "building.2",
                tone: "safe"
            )
        }
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
        if state.pageKind == "cart" {
            return WebsiteGuidedStep(
                title: "Verify cart, then enter checkout",
                detail: "Cart state is visible. Verify item, modifiers, tip, address, and total. If it matches, use Enter Checkout; final purchase still remains blocked.",
                icon: "cart.badge.questionmark",
                tone: "warning"
            )
        }
        if state.pageKind == "menu_or_item" {
            return WebsiteGuidedStep(
                title: "Review item funnel",
                detail: "Item/menu controls are visible. Match the item and modifiers, then use Tap Reviewed for add/customize/cart steps.",
                icon: "list.bullet.rectangle",
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
        if let button = state.candidates.first(where: { $0.kind == "button" && $0.url.isEmpty && $0.tapSafety == "caution" }) {
            return WebsiteGuidedStep(
                title: "Review candidate tap",
                detail: "\(button.label) needs human review first. If this is the intended item/control, use Tap Reviewed.",
                icon: "hand.tap",
                tone: "warning"
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

    private func handleAutoDriveUpdate(for state: WebsiteC2AState) {
        guard autoDriveEnabled, state.pageKind.hasPrefix("dd_") else {
            return
        }
        guard !browser.isLoading, !browser.isInspecting, !browser.isRunningButtonAction else {
            return
        }

        let signature = doorDashAutoDriveSignature(for: state)
        let candidates = mirrorCandidates(for: state)

        if state.pageKind == "dd_payment_sheet" || state.pageKind == "dd_checkout" {
            pendingDoorDashRole = ""
            pendingDoorDashLabel = ""
            addToCartRetryCount = 0
            autoDriveStatus = "Reached checkout. Waiting for your final confirmation."
            lastAutoDriveSignature = signature
            return
        }

        if state.authState == "login_required" {
            autoDriveStatus = "Waiting for you to sign in."
            lastAutoDriveSignature = signature
            return
        }

        if pendingDoorDashRole == "dd_add_to_cart" {
            if state.pageKind == "dd_item_modal",
               let retryCandidate = candidates.first(where: { $0.role == "dd_add_to_cart" && !$0.blocked }),
               addToCartRetryCount < 1 {
                addToCartRetryCount += 1
                autoDriveStatus = "Retrying Add to cart..."
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) {
                    performAutoDriveAction(retryCandidate, allowCaution: true)
                }
                return
            }
            if state.pageKind == "dd_cart_drawer"
                || state.pageKind == "dd_cart_page"
                || candidates.contains(where: { $0.role == "dd_continue_cta" || $0.role == "dd_cart_cta" }) {
                preferCartProgress = true
            }
            pendingDoorDashRole = ""
            pendingDoorDashLabel = ""
            addToCartRetryCount = 0
        }

        guard signature != lastAutoDriveSignature else {
            return
        }

        guard let action = doorDashAutoAction(for: state, candidates: candidates) else {
            lastAutoDriveSignature = signature
            if state.pageKind == "dd_storefront" {
                autoDriveStatus = "Waiting for DoorDash menu items..."
            }
            return
        }

        lastAutoDriveSignature = signature
        autoDriveStatus = action.status
        if action.pendingRole == "dd_add_to_cart" {
            pendingDoorDashRole = action.pendingRole
            pendingDoorDashLabel = action.candidate.label
            addToCartRetryCount = 0
            preferCartProgress = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) {
            performAutoDriveAction(action.candidate, allowCaution: action.allowCaution)
        }
    }

    private func doorDashAutoDriveSignature(for state: WebsiteC2AState) -> String {
        let candidateSlice = mirrorCandidates(for: state)
            .prefix(4)
            .map { "\($0.role):\($0.label)" }
            .joined(separator: "|")
        return [state.pageKind, browser.currentURL, candidateSlice].joined(separator: "||")
    }

    private func doorDashAutoAction(for state: WebsiteC2AState, candidates: [WebsiteC2ACandidate]) -> MirrorAutoDriveAction? {
        switch state.pageKind {
        case "dd_search_results":
            if let store = candidates.first(where: { $0.role == "dd_store_card" && !$0.blocked }) {
                return MirrorAutoDriveAction(
                    candidate: store,
                    allowCaution: false,
                    status: "Opening \(store.label)...",
                    pendingRole: ""
                )
            }
        case "dd_storefront":
            if preferCartProgress,
               let cart = candidates.first(where: { $0.role == "dd_cart_cta" && !$0.blocked }) {
                return MirrorAutoDriveAction(
                    candidate: cart,
                    allowCaution: cart.tapSafety == "caution",
                    status: "Opening cart...",
                    pendingRole: ""
                )
            }
            if let item = candidates.first(where: { $0.role == "dd_item_card" && !$0.blocked }) {
                return MirrorAutoDriveAction(
                    candidate: item,
                    allowCaution: item.tapSafety == "caution",
                    status: "Opening \(item.label)...",
                    pendingRole: ""
                )
            }
            if let add = candidates.first(where: { $0.role == "dd_add_to_cart" && !$0.blocked }) {
                return MirrorAutoDriveAction(
                    candidate: add,
                    allowCaution: true,
                    status: "Adding item to cart...",
                    pendingRole: "dd_add_to_cart"
                )
            }
        case "dd_item_modal":
            if let add = candidates.first(where: { $0.role == "dd_add_to_cart" && !$0.blocked }) {
                return MirrorAutoDriveAction(
                    candidate: add,
                    allowCaution: true,
                    status: "Adding item to cart...",
                    pendingRole: "dd_add_to_cart"
                )
            }
            if let option = candidates.first(where: { $0.role == "dd_item_card" && !$0.blocked }) {
                return MirrorAutoDriveAction(
                    candidate: option,
                    allowCaution: option.tapSafety == "caution",
                    status: "Selecting \(option.label)...",
                    pendingRole: ""
                )
            }
        case "dd_cart_drawer", "dd_cart_page":
            if let checkout = candidates.first(where: { $0.role == "dd_continue_cta" && !$0.blocked }) {
                return MirrorAutoDriveAction(
                    candidate: checkout,
                    allowCaution: true,
                    status: "Continuing to checkout...",
                    pendingRole: ""
                )
            }
            if let cart = candidates.first(where: { $0.role == "dd_cart_cta" && !$0.blocked }) {
                return MirrorAutoDriveAction(
                    candidate: cart,
                    allowCaution: cart.tapSafety == "caution",
                    status: "Opening cart...",
                    pendingRole: ""
                )
            }
        default:
            break
        }
        return nil
    }

    private func performAutoDriveAction(_ candidate: WebsiteC2ACandidate, allowCaution: Bool) {
        if !candidate.url.isEmpty {
            openCandidate(candidate)
        } else {
            tapCandidate(candidate, allowCaution: allowCaution)
        }
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

    private func tapCandidate(_ candidate: WebsiteC2ACandidate, allowCaution: Bool = false) {
        browser.tapButtonCandidate(candidate, allowCaution: allowCaution, capsule: route.capsule)
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
