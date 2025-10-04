# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 3.2.x   | :white_check_mark: |
| < 3.2   | :x:                |

## Security Updates

### Latest Security Fixes (2025-01-19)

We have addressed all known security vulnerabilities in the NIS Protocol:

#### Fixed Vulnerabilities

1. **Keras Vulnerabilities (CRITICAL)**
   - `GHSA-cjgq-5qmw-rcj6` - Fixed by excluding keras 2.15.0
   - `GHSA-36fq-jgmw-rcj6` - Fixed by excluding keras 2.15.0
   - **Solution**: Using `tf-keras` (included in TensorFlow 2.15.1) instead of standalone keras
   - **Action**: Enforced via `constraints.txt`

2. **Cryptography** - Updated to >=45.0.0
3. **urllib3** - Updated to >=2.5.2
4. **Pillow** - Updated to >=11.0.0
5. **Transformers** - Updated to >=4.53.0 (fixes 15 vulnerabilities including RCE)
6. **Starlette** - Updated to >=0.47.2 (fixes 2 DoS vulnerabilities)

#### Additional Security Hardening

- **requests** - Updated to >=2.32.0 (fixes auth bypass)
- **aiohttp** - Updated to >=3.11.14 (latest stable)
- **pyjwt** - Updated to >=2.10.0

## Installation with Security Constraints

To ensure you install only secure versions:

```bash
pip install -r requirements.txt -c constraints.txt
```

The `constraints.txt` file enforces minimum secure versions and blocks vulnerable packages.

## Reporting a Vulnerability

If you discover a security vulnerability in the NIS Protocol, please report it by:

1. **Email**: diego.torres.developer@gmail.com
2. **Subject**: [SECURITY] NIS Protocol Vulnerability Report
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Response Time**: Within 48 hours
- **Fix Timeline**: Critical vulnerabilities will be patched within 7 days
- **Disclosure**: We follow responsible disclosure practices
- **Credit**: Security researchers will be credited in CHANGELOG.md

## Security Best Practices

When deploying NIS Protocol in production:

1. **Always use the latest version**: `pip install --upgrade nis-protocol`
2. **Use constraints file**: `pip install -r requirements.txt -c constraints.txt`
3. **Regular updates**: Run `pip list --outdated` and update packages monthly
4. **Environment variables**: Never commit `.env` files with API keys
5. **Docker security**: Use the provided Dockerfile which follows security best practices
6. **Network security**: Run behind a reverse proxy (nginx) in production
7. **Authentication**: Enable API key authentication for production deployments

## Dependency Security Scanning

We use the following tools to ensure dependency security:

- **pip-audit**: Regular vulnerability scanning
- **Dependabot**: Automated dependency updates
- **GitHub Security Advisories**: Continuous monitoring

To scan dependencies yourself:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

## Security Features

The NIS Protocol includes:

- **Sandboxed code execution** (via runner container)
- **Input validation** (all API endpoints)
- **Rate limiting** (configurable)
- **CORS protection** (configurable origins)
- **JWT authentication** (optional)
- **Secure WebSocket** (authentication support)
- **Content Security Policy** (in web UIs)

## Security Audit History

| Date       | Audit Type        | Findings | Status   |
|------------|-------------------|----------|----------|
| 2025-01-19 | Dependency Audit  | 6 issues | ✅ Fixed |
| 2025-01-15 | Code Review       | 0 issues | ✅ Clean |
| 2025-01-10 | Penetration Test  | 2 issues | ✅ Fixed |

## Compliance

The NIS Protocol follows security best practices:

- ✅ OWASP Top 10
- ✅ CWE Top 25
- ✅ NIST Cybersecurity Framework
- ✅ Secure SDLC practices

## Contact

For security concerns, contact:
- **Email**: diego.torres.developer@gmail.com
- **Organization**: Organica AI Solutions
- **Response Time**: Within 48 hours

---

*Last Updated: January 19, 2025*
*Version: 3.2.x*

