# Open issue implementation record

Branch: `fix/open-issues`, based on `55c42b1`.

Scope: issues #7, #8, #9, #10, #11, #13, #15, #16, plus the requested NUMA controls. Stop before adding other engines or implementing the broader research recommendations.

## Progress

- [x] #9: isolate application state before pytest imports the API; subprocess regression proves the configured temporary directory is used.
- [x] #7: emit explicit negative vLLM flags for prefix caching and chunked prefill in native and Docker commands.
- [x] #8: preserve a hidden saved token on unrelated edits; support explicit replacement and removal.
- [x] #10: atomic private settings writes, JSON API misses, malformed chat response errors, retained failed-launch logs, synchronized server/download operations, accurate filtered download progress, source-build binary preference, live hardware/VRAM display, one scan per completed download, dead-code cleanup, and one canonical changelog.
- [x] #16: Refresh shows a busy state and a success or error notification.
- [x] #11 / U1: isolated source updater and managed engine selection.
- [x] #13 / U2: explicit multi-GPU placement and honest memory advice.
- [x] #15 / U3: upstream MoE, microbatch, loading compatibility, and NUMA controls.
- [ ] U4: documentation, browser acceptance, simplification, independent review, final verification, PR handoff.

## Evidence so far

- Bug regressions were observed failing before their fixes, then passing.
- Full Python suite after backend fixes: 143 passed; one existing Starlette/httpx deprecation warning.
- Frontend production build passed after refresh/settings changes.
- `frontend/tests/browser-smoke.mjs` passed against an isolated local application. It checks breadcrumb text, Refresh feedback, live hardware/VRAM, token preservation/replacement/removal, and download scan behavior. Browser mutations are mocked.
- Duplicate changelog comparison found the root copy contains all history plus newer entries; `docs/CHANGELOG.md` now links to it. Documentation-only change checked with `git diff --check`.

## Hardware verification limits

Development VM100 has no GPU and exposes one NUMA node (CPU and memory locality group), although its physical Proxmox host has two CPUs. NUMA controls target all supported users, including ordinary dual-CPU bare-metal machines. Tests must include multiple visible nodes; do not automatically enable NUMA options or infer host topology from virtualization.

Actual multi-GPU inference and CUDA source builds cannot be verified on this VM. Command construction, validation, failure handling, and browser behavior are verified here; hardware performance is not claimed.

### U1 verification

- Initial updater isolation tests failed before implementation, then passed.
- Full suite: 160 passed; controller reran 24 updater/registry tests successfully.
- Frontend production build and whitespace checks passed.
- Build failure/activation, immutable revision selection, resource limits, direct-local API checks, and managed native executable routing are covered. No upstream source was downloaded or compiled during verification.
- Browser acceptance after U1: existing smoke suite passed; real Settings preflight correctly blocked vLLM on this no-GPU machine and exposed no build button. Desktop view inspected.

### U2/U3 and simplification verification

- Missing placement behaviors were observed failing before implementation.
- Full suite: 186 passed; controller reran 92 placement/engine/advisor checks.
- Frontend build passed. Browser acceptance with synthetic three-GPU/two-NUMA-node hardware passed: device selection, proportions, expert count, optional interleaving, visible validation errors, and disabled invalid launch.
- Simplification: one reuse improvement and two code-quality improvements applied; 33 affected Python tests passed. Three efficiency suggestions were handed to code review because two affect concurrency/ordering and one is low-impact polling overhead.
- `npm run lint` cannot run: the existing project has an ESLint script/dependency but no ESLint configuration. No typecheck script is configured. This is reported rather than represented as a passing check.
