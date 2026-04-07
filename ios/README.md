# Memla iOS Shell

This folder holds the thin iPhone client for Memla.

The app is intentionally small:
- SwiftUI front-end
- App Intent for Siri / Shortcuts
- local-network HTTP client to the Mac-hosted Memla API

The iOS project is defined with XcodeGen so it can live in git cleanly and be built on GitHub-hosted macOS runners without needing a checked-in `.xcodeproj`.

## Local runtime contract

The app talks to the Memla API routes exposed by:

```bash
memla serve --host 0.0.0.0 --port 8080 --model phi3:mini
```

Routes:
- `GET /health`
- `GET /state`
- `GET /memory`
- `GET /actions`
- `POST /actions/plan`
- `POST /actions/draft`
- `POST /actions/capsule`
- `POST /run`
- `POST /scout`
- `POST /followup`

## GitHub-only workflow

1. Push this repo to GitHub.
2. Run the `Build iOS App` workflow.
3. GitHub generates the Xcode project with XcodeGen and builds the app on a macOS runner.

The workflow now does two things:
- validates the app with an `iphonesimulator` build
- produces a device-targeted `Memla-AltStore.ipa` artifact for Windows + AltStore testing

## Memla Browser

Commerce/action capsules can open web bridge options in Memla Browser, a first-party `WKWebView` surface that keeps the capsule, slots, verifier checklist, URL, and page title visible while the user navigates. It also has an `Inspect` pass that compiles the page into a rough page kind, element counts, safe next actions, and residuals. When a page lands in login, bot-check, or target-not-visible state, Memla can suggest a safer bridge such as the installed app or neutral web search. If the page exposes a search-like input, Memla can fill or submit the capsule search query, but it still does not auto-click checkout or submit purchases.

## AltStore testing flow

1. Run the `Build iOS App` workflow on GitHub.
2. Download the `memla-ios-build` artifact bundle.
3. Extract `Memla-AltStore.ipa`.
4. In AltServer on Windows, hold `Shift` and choose `Sideload .ipa...`
5. Pick `Memla-AltStore.ipa` and install it to the iPhone.

The IPA is intentionally unsigned in CI. AltServer handles the Apple-ID signing step for sideloaded testing.

## Real device install later

To get an installable iPhone build later, you will still need:
- an Apple Developer signing setup
- App Store Connect / TestFlight or another valid signing path

That signing layer is intentionally not baked into V1.
