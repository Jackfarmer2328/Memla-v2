# Memla Browser V1

Memla Browser V1 is the first capsule-aware in-app web surface.

It replaces the simple Safari sheet for `in_app_web` bridge options with a `WKWebView` screen that keeps the current action capsule visible while the user navigates.

V1 tracks:

- page title
- current URL
- loading state
- back / forward / reload controls

V1 shows:

- capsule summary
- structured slots such as restaurant, item, modifiers, and tip
- verifier checklist items
- auto-submit blockers

Commerce capsules can now route through service-specific app links, service web links, or a generic web-search bridge when a service URL drops part of the capsule intent.

V1 intentionally does not:

- auto-click checkout
- auto-submit purchases
- bypass service login or payment confirmation

The purpose is to create the controlled surface where future page-state extraction, verifier checks, and safe fill/click primitives can attach.
