# mccl — notes for Claude

- Build with `swift build` / `swift test` only. Do **not** use `xcodebuild` — it is broken
  machine-wide (Exec format error) as of 2026-08-22.
- This is an open-source infrastructure library (NCCL-equivalent for Apple Silicon
  clusters). API stability and a clean C ABI matter more than features; don't grow it
  into a training framework.
- No cloud AI dependencies anywhere in this project.
- Design doc: docs/ARCHITECTURE.md. Keep it updated when the plan changes.
