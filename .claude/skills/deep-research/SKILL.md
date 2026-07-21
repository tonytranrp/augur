---
name: deep-research
description: Research a topic thoroughly via exactly one dispatch of the read-only deep-research-agent subagent. Use when a question needs real investigation (multiple sources, a genuine unknown) rather than something already answerable from context. Invoke once per distinct question — do not stack multiple calls for the same question.
context: fork
agent: deep-research-agent
---

Research $ARGUMENTS thoroughly:

1. Find relevant files, docs, or external references (local source, local
   docs, and — via WebFetch/WebSearch — external documentation, papers, or
   package registries as the question requires).
2. Read and analyze what you find.
3. Summarize findings with specific file references or citations.

Answer the question completely in this one pass rather than returning a
partial result that expects a follow-up call. This skill forks into
`deep-research-agent`, which has no `Agent` tool and cannot spawn further
subagents under any circumstances — that guarantee comes from the
subagent's tool list, not from this skill's wording.
