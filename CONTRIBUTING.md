# Working agreement (2-person team)

## Branches and pull requests

* `main` is protected: no direct pushes, CI must be green, 1 approving review from the other member.
* One branch per issue: `feat/<issue-no>-short-name`, `fix/<issue-no>-...`, `docs/...`, `test/...`.
* Small PRs (under ~400 changed lines). Link the issue (`Closes #12`).
* Squash-merge; the squash message follows Conventional Commits:
  `feat(models): add TSB routing for intermittent items`.

## PR checklist (copy into the PR description)

- [ ] Linked issue and what changed, in one paragraph
- [ ] `make lint test` passes locally
- [ ] New logic has a unit test; anything touching features has a leakage test
- [ ] No data, artifacts, `.env` or keys committed
- [ ] Numbers in docs regenerated with `make all` if results changed
- [ ] Work log updated (`docs/worklogs/<you>.md`)

## Code review guide

Reviewers check, in this order: correctness (does it do what the issue says), leakage (does any
feature read after the origin), tests, readability, then style. Leave at least one substantive
comment or an explicit "LGTM because ..." so the review is visible evidence.

## Ownership

`.github/CODEOWNERS` maps every folder to an owner. The owner writes the module and its tests; the
other member reviews. Both members must be able to explain every module in the viva.

## Setting up the GitHub side

1. Create the repo, push `main`, enable branch protection (require PR, 1 review, status check `ci / test`).
2. Import the backlog: create milestones M1 to M7, then create issues from `docs/01-brief/backlog.csv`
   (GitHub CLI: `gh issue create --title ... --body ... --label ... --milestone ...`).
3. Create a Project board (Todo / In progress / In review / Done) and add all issues.
4. Replace `@member-a` / `@member-b` in `.github/CODEOWNERS` with your usernames.
