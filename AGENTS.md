# AGENTS.md - multi-agent concurrency model (vendor-neutral)

Not locked. Non-Claude agents: read this, then follow the read order in `CLAUDE.md`. IDs and read order live there only.

## Ownership (surfaces)
| Surface | Owner |
|---|---|
| LOCKED files | Main Agent only (`GOVERNANCE.md`) |
| `STATE.md` claims table | every agent, own rows only |
| `src/` or equivalent (after stack ADR) | TODO(<owner>) |
| `DEMO.md` | whoever last touched a demo path keeps it runnable |
| `worklog.md`, logs | append-only, anyone |

## Claiming work
1. Add a row to the claims table in `STATE.md`: surface/task, agent/human, start time (with timezone), related IDs.
2. Work only inside your claim. If your surface overlaps an existing row, stop and escalate.
3. **Release the row the moment you stop.** A stale claim blocks others worse than no claim.

## The one hard rule
Any agent finishing a unit of work updates `STATE.md` before yielding. "Finishing" includes blocking and running out of turn budget.

## Conflicting work
Never silently pick a winner. Record the conflict under **Blocked** in `STATE.md`, keep both versions discoverable (branch or file), and register a `Q-####`.
