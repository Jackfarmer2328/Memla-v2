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
- `POST /run`
- `POST /scout`
- `POST /followup`

## GitHub-only workflow

1. Push this repo to GitHub.
2. Run the `Build iOS App` workflow.
3. GitHub generates the Xcode project with XcodeGen and builds the app on a macOS runner.

The default workflow validates the app with an `iphonesimulator` build, which avoids signing requirements.

## Real device install later

To get an installable iPhone build later, you will still need:
- an Apple Developer signing setup
- App Store Connect / TestFlight or another valid signing path

That signing layer is intentionally not baked into V1.
