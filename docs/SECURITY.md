# Pengu Security

## Permission System

Every tool call has a permission level:

| Level | Risk | Examples | Auto-Approve |
|-------|------|----------|-------------|
| 0 - SAFE | Read-only | read files, list dirs, screen capture, git status | Yes |
| 1 - LOW | Write/Launch | create files, open apps, browser navigation | Yes (configurable) |
| 2 - HIGH | Destructive | delete files, shell commands, git push | No — requires confirmation |
| 3 - CRITICAL | System-wide | disk ops, credential changes, admin | Always requires confirmation |

## What Pengu Will NOT Do

Without explicit user confirmation, Pengu will never:

- Delete entire directories recursively
- Force push to Git
- Execute unknown shell commands
- Access credentials or passwords
- Send data to cloud services (in FREE_ONLY mode)
- Modify system security settings
- Install software without asking

## What Pengu NEVER Does

Under any circumstances:

- Steals passwords
- Bypasses authentication
- Attacks networks
- Exfiltrates private files
- Logs sensitive data (API keys, Wi-Fi passwords)
- Modifies Git identity silently
- Commits with fake contributor names

## Audit Trail

Every tool execution is logged with:

- Timestamp
- Tool name
- Permission level
- Whether it was granted
- Parameters (keys only, not values)
- Result summary

Logs never contain:

- API keys
- Passwords
- Tokens
- Wi-Fi passwords
- SSH keys

## Secrets Management

- API keys stored in environment variables or `.env` file
- `.env` is in `.gitignore` — never committed
- `.env.example` contains placeholders only
- No secrets in source code, tests, or documentation
