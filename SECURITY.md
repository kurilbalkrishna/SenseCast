# Security

* **Secrets:** none are needed to run SenseCast. Configuration comes from environment variables;
  `.env` is git-ignored and `gitleaks` scans every push in CI.
* **Personal data:** none. Sales are item x store x day aggregates; there are no customer records.
* **Dependencies:** pinned in `requirements.txt`; `pip-audit` runs in CI. Upgrade and re-run tests
  when it flags a package.
* **Container:** slim base image, runs as a non-root user (`uid 10001`), health-checked.
* **API:** store-scoped RBAC, rate limiting (429), 64 KB payload cap, strict input validation,
  append-only audit log, CSV formula-injection escaping. Authentication is a hook
  (`src/sensecast/api/auth.py`): run `SENSECAST_AUTH_MODE=header` only behind a gateway that sets
  the identity headers, and never expose that mode directly to the internet.
* **Threat model:** `docs/02-design/architecture.md` section 10.

Report a problem by opening a private security advisory on the repository.
