# Subagent orchestration policy

This project relies on Claude Code's real subagent mechanics (context
windows, the `Agent` tool, `.claude/agents/`, forked skills) — not on hoping
a prose rule gets followed. Where a rule below is "hard," it's implemented as
an actual tool restriction, not just a request.

## How many subagents to spawn (soft — use judgment, but default small)

- Claude Code's own session default is 200 subagent spawns before it refuses
  (raise-only via `CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION`, no upper bound).
  That number is not a budget to use — for this project, getting anywhere
  near it is itself a signal that something is spawning redundantly, not a
  sign that more parallelism is fine.
- **One research question = one subagent.** If a task is fundamentally one
  question — "what's the current status of X," "how does library Y handle
  Z" — dispatch exactly one `deep-research-agent` (via the `deep-research`
  skill, see below) for it. Do not spawn two or three subagents to
  redundantly cover the same question from slightly different angles.
- **Independent branches are the exception.** Spawn multiple subagents in
  parallel only when the branches are genuinely independent and don't need
  each other's results — e.g. investigating three unrelated roadmap items at
  once, or reviewing three unrelated modules. If branch B needs branch A's
  answer first, that's a sequential chain, not parallel research, and
  probably doesn't need more than one subagent at a time either.
- Before spawning anything, ask: would this task flood the main conversation
  with output you won't reference again (search results, log dumps, file
  contents)? If not, just do it directly instead of delegating.

## No nested subagent spawning (hard — enforced via tool restriction)

Claude Code's platform itself allows a subagent to spawn its own nested
subagents (up to a fixed depth of 5, as of recent Claude Code versions).
**This project overrides that down to zero extra levels.** A subagent
spawned here can never itself spawn another subagent — not "shouldn't,"
literally can't, because the mechanism is a tool restriction:

- Every subagent definition under `.claude/agents/` in this project must
  either omit `Agent` from an explicit `tools:` allowlist, or set
  `disallowedTools: Agent` explicitly. See `.claude/agents/deep-research-agent.md`
  for the reference implementation — its `tools:` list has no `Agent` entry,
  so per Claude Code's own documented behavior ("If Agent is omitted from
  the tools list entirely, the agent can't spawn any subagents") it is
  structurally incapable of spawning anything, regardless of what it's
  asked to do or what it reads.
- The same applies to any skill that forks into a subagent
  (`context: fork` in a `SKILL.md`'s frontmatter, like
  `.claude/skills/deep-research/SKILL.md`): the `agent:` it targets must be
  one that excludes `Agent` from its own tools, for the same reason.
- If you (Claude, working in the main session) are defining a *new* custom
  subagent or fork-skill for this project, apply this same restriction by
  default. Only give a subagent the `Agent` tool if there's a specific,
  deliberate reason it needs to delegate further — and if you do, that's a
  deviation from this project's default and should be called out explicitly
  when you do it, not done silently.

## The `deep-research` skill specifically (hard)

`.claude/skills/deep-research/SKILL.md` forks into
`.claude/agents/deep-research-agent.md` — a read-only subagent with no
`Agent` tool. Use it for genuine research questions (something needs
investigating across multiple sources or files), not for tasks you could
just do directly.

- Invoke it **at most once per distinct question.** If a task seems to need
  it multiple times, that's a sign the question should be reframed into one
  thorough invocation (the skill's own instructions ask it to answer
  completely in a single pass), not a reason to call it repeatedly.
- Never redefine `deep-research`'s `agent:` target to point at a subagent
  that has `Agent` in its `tools:` list — that would silently reopen the
  nesting this policy exists to prevent.
