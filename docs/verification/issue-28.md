# Issue 28 verification

Implemented from [issue #28](https://github.com/jimdawdy-hub/Local-LLM-Launcher-GUI/issues/28) using the [CE plan](../plans/2026-10-13-feat-favorites-log-benchmarks-plan.md) and CE work workflow.

## Automated checks

- `PYTHONPATH=src <venv>/bin/python -m pytest -q`: **511 passed**, 15.95 seconds (final run).
- `npm run build` in `frontend`: passed; generated assets included in the commit.
- Changed frontend sources passed ESLint recommended rules, React hook rules, and unused-variable checks with a temporary flat config using the installed dependencies.
- Repository `npm run lint` cannot run because the existing project has no ESLint configuration. No project-wide lint success is claimed.
- `git diff --check`: passed.

Regression tests cover persistence, replay with current environment settings, missing GGUF files, credential exclusion, malformed arguments, legitimate token-count flags, implicit GGUF selection, concurrent saves, failed saves, legacy servers, log parsing, unavailable measurements, full-log reading, and separate logs for launches within the same second.

## Actual browser verification

Chrome 155 via agent-browser, serving the built application from the real FastAPI backend at `http://127.0.0.1:18768`. Viewports: 1440 × 1000 and 390 × 844. These screenshots are actual browser captures, not mockups.

The isolated [fixture](issue-28-fixture.py) replaces model discovery, hardware discovery, and the inference command with harmless local HTTP subprocesses. The launch lifecycle, API routes, disk persistence, chat proxy, log parsing, and UI are real. **The displayed performance numbers are fixture measurements, not GPU performance claims.**

Verified through browser actions:

1. Empty Favorites and Benchmarks states render clearly.
2. Save action appears beside Test chat; blank names cannot submit. Saved “Qwen3.8 Unsloth Best” appears under Favorites with model and settings.
3. Favorite Launch opens Servers, displays the matching settings, opens the log, and reserves another port when the saved port is occupied.
4. Test chat receives the fixture response. Benchmark records llama.cpp 1,000 prefill TPS / 25 generation TPS and vLLM 120 / 30 from known log samples, with their different measurement bases labeled.
5. A legacy server cannot save unknown settings. Benchmarking its unmeasured log shows “No supported timing measurements … Run a test chat”.
6. Favorites and benchmark rows survive browser reload and a backend restart; the saved favorite still launches.
7. Delaying a favorite launch response by five seconds, then navigating to Benchmarks, leaves the user on Benchmarks after completion.
8. Light and dark layouts remain usable at desktop and mobile sizes. At 390 px the document width is 390 px; only the 1,000 px benchmark table scrolls horizontally.
9. A zero-only vLLM log renders `0` TPS while missing token and time counts render `—`; the legacy server shows its explicit unavailable-settings message.
10. No uncaught browser errors were reported.

Visual inspection found and fixed a selected server heading hidden beneath the sticky topbar. The replacement desktop and mobile captures show the heading and settings below the topbar.

## Screenshots

- [Server controls and test chat, desktop](issue-28/servers-desktop.png)
- [Save form, mobile](issue-28/save-favorite-mobile.png)
- [Favorites, desktop](issue-28/favorites-desktop.png)
- [Favorites, dark mobile](issue-28/favorites-dark-mobile.png)
- [Favorite relaunch, desktop](issue-28/favorite-relaunch-desktop.png)
- [Favorite relaunch, mobile](issue-28/favorite-relaunch-mobile.png)
- [Servers, dark mobile](issue-28/servers-dark-mobile.png)
- [Benchmarks, desktop](issue-28/benchmarks-desktop.png)
- [Benchmarks, mobile](issue-28/benchmarks-mobile.png)
- [Benchmarks, dark desktop](issue-28/benchmarks-dark-desktop.png)
- [Benchmarks, dark mobile](issue-28/benchmarks-dark-mobile.png)
- [Measured zero versus unavailable, desktop](issue-28/benchmarks-zero-desktop.png)

## Reproduce

Build the frontend, create a disposable directory, and run from the repository root:

```sh
mkdir -p /tmp/launcher-28-verification
LOCAL_LLM_LAUNCHER_HOME=/tmp/launcher-28-verification PYTHONPATH=src \
  .venv/bin/python docs/verification/issue-28-fixture.py
```

Open `http://127.0.0.1:18768`, then follow the steps above. Stop with Ctrl-C; the fixture stops its child servers. Use a fresh disposable home for an empty-state run. Python dependencies must already be installed. This fixture must not use an existing personal launcher home.

## Limits

No real llama.cpp/vLLM GPU inference was run. Benchmarks summarize existing engine logs; the two engines' measurements are not equivalent workloads. Old server records have no reconstructed launch settings. Favorites with launch-specific API credentials or credential-bearing extra arguments are blocked to avoid silently changing authentication on replay.

## Code review

Nine review lenses covered correctness, security, project standards, testing, maintainability, API contracts, reliability, frontend races, and adversarial cases. Five initial findings were fixed, then independently checked against the resulting source:

- Recognize the `-hft` credential alias.
- Allow legitimate token-count tuning arguments.
- Save the actual implicitly selected GGUF file.
- Give each server a unique log filename to prevent mixed benchmark samples.
- Preserve later navigation when a favorite launch finishes asynchronously.

No actionable findings remained. Suggested credential and zero-display coverage was completed. Cross-model review transports were unavailable (adapter initialization/authentication failures); adversarial review ran locally.

The final CE simplification pass completed its reuse, quality, and efficiency reviews with no worthwhile changes.

Retrospective CE document review completed all six selected local lenses (coherence, feasibility, design, security, scope, adversarial) with no plan corrections. Its cross-model adapter also failed before review.
