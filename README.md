# National hackathon repo (event: TODO(<owner>))

Fixed-deadline build-and-demo repo for humans and AI agents working at once. Event facts: `COMPETITION.md`. Thesis: `VISION.md`.

## Find things in 15 seconds
| I need | Go to |
|---|---|
| deadline, rubric, rules | `COMPETITION.md` |
| what's happening right now | `STATE.md` |
| demo script and reset | `DEMO.md` |
| what judges score vs what we built | `notes/judging_alignment.md` |
| who may edit what | `GOVERNANCE.md`, `AGENTS.md` |
| deliverables staging | `submissions/` |
| picking an idea | `ideas/` |

## Cold-agent read order
`CLAUDE.md` (or `AGENTS.md` if not Claude Code) -> `STATE.md` -> `VISION.md` -> `COMPETITION.md` -> `GOVERNANCE.md` -> `AGENTS.md` -> recent `worklog.md` tail -> `experiments/experiment_log.md`

## End-of-session checklist
overwrite `STATE.md` | append dated+timestamped `worklog.md` entry | append any new `EXP-####` row | log any LOCKED-file edit in `GOVERNANCE.md` audit table | release any finished work claim

If a doc is still full of `TODO(<owner>)`, say so rather than invent. Find all blanks: `grep -rn "TODO(<owner>)" .`
