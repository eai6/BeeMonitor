---
name: pi-git-commit-and-push
description: "How to commit/push from the BeeMonitor Pi — no git identity configured, remote main is usually ahead"
metadata:
  node_type: memory
  type: feedback
  originSessionId: fa9e86bd-41ae-4282-80fc-0e0732432307
  modified: 2026-09-23T14:47:38.616Z
---

This Pi has no git user.name/user.email configured. Commit with a per-command identity matching the repo's existing author, without touching git config:
`git -c user.name=eai6 -c user.email=32150686+eai6@users.noreply.github.com commit ...`

origin/main usually has commits the Pi lacks (the user also pushes from elsewhere, e.g. enclosure changes). `git fetch` and `git rebase origin/main` (same `-c` identity) before `git push origin main`.

**Why:** the first commit attempt on 2026-09-23 failed on "Author identity unknown" and the push was rejected as non-fast-forward.
**How to apply:** whenever the user asks to commit/push from this device. Pushing directly to main is what they asked for; see [[pi-no-passwordless-sudo]].
