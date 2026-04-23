# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.3.x   | :white_check_mark: |
| < 2.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within EnlightenLM, please follow responsible disclosure:

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Send a detailed description of the vulnerability to the maintainers
3. Allow time for the maintainers to assess and fix the vulnerability
4. Once fixed, you may publicly disclose the vulnerability

## Security Features

EnlightenLM implements several security features:

- **Input Validation**: All inputs are validated before processing
- **VAN Monitoring**: Variable Attractor Network monitors for malicious patterns
- **Self-loop Detection**: Prevents self-referential content generation
- **Entropy Monitoring**: Detects low-entropy (repetitive) outputs
- **Cooldown Mechanism**: Rate limiting to prevent abuse

## Best Practices

When using EnlightenLM:

1. Keep your API keys secure and never commit them to version control
2. Use environment variables for sensitive configuration
3. Regularly update to the latest version
4. Monitor security logs for suspicious activity
5. Implement additional security measures at the application level

## Security Updates

Security updates will be released as patch versions (e.g., 2.3.1) and announced via:

- GitHub Security Advisories
- Project releases page
