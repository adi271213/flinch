# Git maintenance

Inspect the target remote and branch before editing unpublished commits. Preserve published contribution records, authentic authorship, audit provenance and legal attribution.

## Inspect before changing history

Run these in the checkout with the intended remote:

```bash
git remote -v
git config --get user.name
git config --get user.email
git log --format=fuller
git log --all --format='%h %s%n%b' --grep='Co-authored-by:' --regexp-ignore-case
```

Review locally; do not paste personal metadata into an issue unnecessarily. Preserve genuine contributions, legal notices and required disclosures.

## Clean up future unpublished work

No rewrite is needed for the current published repository. For a future working branch whose commits have **never been published**, use interactive rebase to improve messages or squash related work:

```bash
git fetch origin main
git switch -c chore/history-cleanup
git log --oneline origin/main..HEAD
git rebase -i origin/main
```

In the editor, use `reword` for an inaccurate message and `squash` or `fixup` for related unpublished commits. Keep authentic attribution when combining commits. Prefer a subject such as `fix: reject conflicting processor aliases` that explains the final behavior. To leave the rebase without changing the branch, run `git rebase --abort` while it is in progress.

After reviewing the resulting tree and running checks, publish the new branch normally:

```bash
git push -u origin HEAD
```

Do not use this procedure on shared `main`, rewrite existing authors to imply different provenance, or force-push published history as cosmetic cleanup. Keep local scratch-file exclusions in `.git/info/exclude`; the active `AGENTS.md` engineering guidance remains part of the source.
