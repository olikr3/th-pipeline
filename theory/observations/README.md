# Observations

A running log of what we learned the hard way while building the engine, so it isn't lost in commit messages or chat. Theory notes one level up explain how things work; these files record where reality differed from the plan.

| File | Records |
|---|---|
| [bugs.md](bugs.md) | defects found in code or tests: what broke, why, how it was fixed |
| [wrong-assumptions.md](wrong-assumptions.md) | assumptions (in the spec, the code or our reasoning) that turned out wrong or incomplete |
| [tradeoffs.md](tradeoffs.md) | choices between alternatives, and what was given up |

## Entry format

Add new entries at the top of the right file, as a `##` heading followed by these bullets:

- **Bugs:** Date, Where, What happened, Impact, Resolution.
- **Wrong assumptions:** Date, Source, Assumption, Why it's wrong, Impact, Status.
- **Tradeoffs:** Date, Where, Chosen, Alternative, Given up, Gained, and optionally Revisit if / Revisit when.

Use absolute dates. Link the code (`thp/src/...`), the spec, ADRs or theory notes involved. When a status changes (an open question gets decided, a tradeoff is revisited), update the entry rather than adding a new one.

Trivial slips (typos, lint fixes, a test expectation mistyped and fixed at once) don't belong here; record what someone working on the engine later would want to know.
