# Privacy and security

This runtime is a local research tool, not an authorization boundary. Treat model outputs as suggestions. Enforce permissions, confirmation requirements and execution limits separately.

## Distribution policy

The repository includes reviewed source, synthetic examples, unit tests, aggregate research measurements and upstream license notices. It does not include model weights, private prompts, message or email exports, benchmark raw-output archives, credentials, workstation paths or process/environment dumps.

Avoid committing:

- `.env` files, access tokens, keychain exports or SSH/private keys;
- downloaded model/cache directories or virtual environments;
- real user requests, generated telemetry or complete local reports;
- personal commit email addresses.

Review `git diff --cached` and the commit identity before pushing. A secret scanner does not replace manual review.

## Downloads and inference

Obtain the original weights from Mapika and retain their licensing terms. Pin revisions rather than trusting a moving model tag. Do not enable remote model code from an unreviewed checkpoint.

Run one model worker at a time on constrained unified-memory systems. MLX allocator limits do not replace an external resource watchdog. Inputs may contain sensitive information; do not log them by default in an application wrapper. Any included local diagnostic CLI is not an authenticated multi-user server.

## Reporting

Do not post credentials or private prompts in public issues. If reporting a vulnerability, use GitHub's private vulnerability reporting when enabled; otherwise share only a non-sensitive description and request a private channel.
