# Git history and attribution

The 2026-11-17 review inspected both the original research checkout and the private publication repository. The research history has 13 commits under the configured identity Aditya Jain; the publication repository has four commits under the legitimate GitHub identity `adi271213`.

Neither history contains assistant coauthor trailers, generated-by footers, or bot contributor records. One research commit subject names an external audit; that factual provenance is not present in the publication history. No published history was rewritten and no contributor identity was changed.

## Inspect before changing history

Run these in the checkout with the intended remote:

```bash
git remote -v
git config --get user.name
git config --get user.email
git log --format=fuller
git log --all --format='%h %s%n%b' --grep='Co-authored-by:' --regexp-ignore-case
```

Review locally; do not paste personal metadata into an issue unnecessarily. An attribution record is not an error merely because it identifies a tool or contributor. Preserve genuine contributions, legal notices and required disclosures.

## Clean up future unpublished work

No rewrite is needed for the current published repository. For a future working branch whose commits have **never been published**, use interactive rebase to improve messages or squash related work:

```bash
git fetch origin main
git switch -c codex/history-cleanup
git log --oneline origin/main..HEAD
git rebase -i origin/main
```

In the editor, use `reword` for an inaccurate message and `squash` or `fixup` for related unpublished commits. Keep authentic attribution when combining commits. Prefer a subject such as `fix: reject conflicting processor aliases` that explains the final behavior. To leave the rebase without changing the branch, run `git rebase --abort` while it is in progress.

After reviewing the resulting tree and running checks, publish the new branch normally:

```bash
git push -u origin HEAD
```

Do not use this procedure on shared `main`, rewrite existing authors to imply different provenance, or force-push published history as cosmetic cleanup. Repository-local editor/agent scratch files are ignored prospectively; the active `AGENTS.md` engineering guidance remains part of the source.
