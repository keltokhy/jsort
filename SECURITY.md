# Security

## Reporting a vulnerability

Please report security problems privately through GitHub:
[Report a vulnerability](https://github.com/keltokhy/jsort/security/advisories/new).
Do not open a public issue. This is maintained by one person in spare time; expect a first reply
within a week.

Only the latest release is supported. Fixes ship as a new release, not as patches to older ones.

## What to know when using it

- Text you pass in is sent to the model provider you configure (TypeSafe, OpenRouter, a gateway,
  or a server on your own machine). Do not send data you are not allowed to share with that provider.
- API keys are read from environment variables or `~/.config/jev/<provider>.key` and are sent only
  to the provider they belong to. They are never written to the answer cache or to logs.
- Answers are cached on disk in `~/.cache/jev`. Delete it to remove stored answers.
