# Security Policy

gHALO is an early-stage HPC benchmark and diagnostic suite. Security reports are
welcome, especially for issues involving unsafe file handling, command
execution, dependency handling, malformed input, CI configuration, or output
generation.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `main`  | Yes       |
| `< 0.1` | No        |

Until the first tagged release, security fixes target `main`.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting if it is enabled for the
repository. If private reporting is not available, contact the repository owner
or maintainers privately and include:

- affected version or commit;
- platform and build configuration;
- steps to reproduce;
- potential impact;
- suggested fix or mitigation, if known.

## Responsible Disclosure

Maintainers will acknowledge reports when possible, investigate in good faith,
and coordinate a fix before public disclosure. Reporters are asked to avoid
publicly disclosing details until maintainers have had reasonable time to
respond and prepare a fix.

For non-security bugs, use the normal issue tracker.
