#!/bin/bash
# SessionStart hook — keep the Claude Code on the web clone synced to latest main.
#
# The web container re-provisions from an older snapshot after inactivity, which
# leaves the repo on a stale commit (merged work missing) with newer files left
# untracked. This re-syncs the working tree at session start/resume.
#
# Safety:
#   * Runs ONLY in the remote web env (CLAUDE_CODE_REMOTE) — never touches a
#     local checkout / a collaborator's machine.
#   * Only hard-resets when on the default branch (the stale-reset state); on a
#     feature branch it only fetches, so in-progress branch work is never lost.
#   * `git clean -fd` (no -x) removes stray untracked files but KEEPS gitignored
#     ones (venvs, __pycache__, caches).
#   * Committed work is always safe on the remote; only uncommitted scratch on
#     the default branch is discarded.
#   * Never blocks session start (always exit 0); the fetch is time-bounded.

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0   # web env only; no-op locally
cat >/dev/null 2>&1                                 # drain SessionStart JSON stdin

cd "${CLAUDE_PROJECT_DIR:-$PWD}" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0
[ -n "$(git remote 2>/dev/null)" ] || exit 0

if ! timeout 60 git fetch origin --prune >/dev/null 2>&1; then
  echo "[session-start] git fetch failed/timed out — leaving the tree as-is."
  exit 0
fi

default_branch=$(git remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')
[ -n "$default_branch" ] || default_branch=main
git rev-parse --verify "origin/$default_branch" >/dev/null 2>&1 || exit 0

current=$(git branch --show-current 2>/dev/null)
if [ -z "$current" ] || [ "$current" = "$default_branch" ]; then
  git checkout "$default_branch" >/dev/null 2>&1 \
    || git checkout -B "$default_branch" "origin/$default_branch" >/dev/null 2>&1
  git reset --hard "origin/$default_branch" >/dev/null 2>&1
  git clean -fd >/dev/null 2>&1
  echo "[session-start] synced $default_branch -> origin/$default_branch ($(git rev-parse --short HEAD 2>/dev/null))"
else
  echo "[session-start] on feature branch '$current' — fetched only (no reset)."
fi
exit 0
