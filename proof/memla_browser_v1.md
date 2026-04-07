# Memla Browser V1

Memla Browser V1 is the first capsule-aware in-app web surface.

It replaces the simple Safari sheet for `in_app_web` bridge options with a `WKWebView` screen that keeps the current action capsule visible while the user navigates.

V1 tracks:

- page title
- current URL
- loading state
- back / forward / reload controls
- read-only website C2A inspection
- residual-driven bridge suggestions
- conservative search fill / search submit primitives
- ranked page candidates extracted from visible links and buttons
- guided next-step planning after inspection
- auto-reinspection after opening an in-browser bridge or URL-backed candidate
- compact bottom console layout that preserves webview space by default
- policy-gated safe button candidate taps

V1 shows:

- capsule summary
- structured slots such as restaurant, item, modifiers, and tip
- verifier checklist items
- auto-submit blockers
- canonical page kind
- visible input / button / link counts
- safe next-action candidates
- page residuals such as login, checkout, bot-check, or target-not-visible
- recovery suggestions such as trying the installed app or neutral web search after bad landing states
- safe search controls when Website C2A sees a search-like input
- ranked candidates scored against capsule slots such as restaurant, item, modifiers, or destination
- guidance for the current safe next step, including hard stop guidance around checkout/payment states
- a compact console with page kind, best candidate, and guidance, expandable into full candidates/search/residuals/checklist context
- button candidates classified as safe, caution, or blocked

Commerce capsules can now route through service-specific app links, service web links, or a generic web-search bridge when a service URL drops part of the capsule intent.

V1 intentionally does not:

- auto-fill non-search forms
- auto-click caution, unknown, or sensitive page controls
- auto-click checkout
- auto-submit purchases
- bypass service login or payment confirmation

Candidate opening is intentionally limited to non-final candidates with usable URLs. Button-only candidates can be tapped only when their label is classified as a safe non-final control such as search, filter, view, menu, close, or load more. Caution and blocked buttons remain disabled.

The purpose is to create the controlled surface where verifier checks and safe fill/click primitives can attach after page-state extraction is visible and stable.
