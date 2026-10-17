# ADR-0001: Scaffold choices

- Status: Accepted
- Date/time (TZ): 2026-10-16 (time TODO(<owner>))
- Deciders: hackathon-setup agent (initial authorship); Main Agent TODO(<owner>) to confirm
- Related: Q-0001, Q-0002, Q-0003

## Context
New hackathon repo; event facts, thesis, and stack were all unknown at scaffold time.

## Decision
- Three-piece cold-start (CLAUDE.md / STATE.md / worklog.md) plus hackathon layer (COMPETITION.md, DEMO.md, AGENTS.md, judging_alignment).
- Event facts, thesis: placeholders (TODO(<owner>)), not guesses (Q-0001, Q-0002).
- Stack undecided: no toolchain files, no src tree, no test baseline (ADR-0002, Q-0003).
- `ideas/` included because no idea is chosen; delete it once one is.
- Omitted as unjustified by the brief: `data/`, `models/`, `evals/`, `hardware/`, deployment notes. Add the module when the brief justifies it (agent/LLM subject -> `evals/`; hardware -> `hardware/SAFETY.md`; ML/data -> `data/`).

## Consequences
Nothing here constrains the eventual stack. First real work is filling COMPETITION.md and choosing the idea.
