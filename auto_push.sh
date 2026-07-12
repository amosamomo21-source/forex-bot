#!/bin/bash
# Auto-commit and push all changes to GitHub.
# Runs nightly via launchd. Token read from .env (gitignored — never committed).

set -euo pipefail

REPO="/Users/bamznizzy/forex-bot"
LOG="$REPO/autopush.log"
REMOTE="https://github.com/amosamomo21-source/forex-bot.git"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

cd "$REPO"

# Load GITHUB_TOKEN from .env
if [ ! -f .env ]; then
    log "ERROR: .env not found"; exit 1
fi
GITHUB_TOKEN=$(grep -E '^GITHUB_TOKEN=' .env | cut -d= -f2- | tr -d '"' | tr -d "'")
if [ -z "$GITHUB_TOKEN" ]; then
    log "ERROR: GITHUB_TOKEN not set in .env"; exit 1
fi

# Check for anything to commit
git add .
if git diff --cached --quiet; then
    log "No changes to commit — skipping push"
    exit 0
fi

MSG="Auto-commit $(date -u '+%Y-%m-%d %H:%M UTC')"
git commit -m "$MSG"
log "Committed: $MSG"

git push "https://${GITHUB_TOKEN}@${REMOTE#https://}" main
log "Pushed to GitHub successfully"
