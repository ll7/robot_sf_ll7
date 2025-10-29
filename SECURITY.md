# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take the security of robot_sf seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Reporting Process

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:
- **Primary Contact**: marco.troester.student@uni-augsburg.de

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

### Preferred Languages

We prefer all communications to be in English or German.

## Security Update Policy

When we learn of a security vulnerability, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported versions
4. Release new versions as soon as possible

## Security Best Practices for Users

### Dependency Management

- Always use the latest stable version of robot_sf
- Regularly update dependencies using `uv sync`
- Review `uv.lock` changes in pull requests
- Use GitHub's Dependabot alerts when available

### Environment Security

- Never commit secrets, API keys, or credentials to the repository
- Use environment variables for sensitive configuration
- Keep your Python environment isolated using virtual environments
- Review third-party models before loading them

### Training and Model Security

- Validate model files before loading (check file size, format, source)
- Be cautious when loading models from untrusted sources
- Use the latest model files recommended in documentation
- Store sensitive training data outside the repository

### Code Execution Safety

- Be aware that this project executes Python code and loads models
- Review code changes in pull requests carefully
- Run in isolated environments (Docker, virtual machines) when experimenting
- Avoid running untrusted demonstration scripts

## Known Security Considerations

### Machine Learning Models

- Model files (.zip) are loaded using StableBaselines3 and PyTorch
- Always verify model sources and checksums when possible
- Models can potentially contain malicious code if tampered with

### External Dependencies

- This project depends on multiple scientific computing libraries
- Some dependencies (PyTorch, NumPy, etc.) have native code components
- Review security advisories for major dependencies regularly

### Simulation Environment

- The simulation can consume significant system resources
- Long-running simulations should be monitored
- Ensure adequate system resources to prevent DoS conditions

## Security Scanning

### Recommended Tools

We recommend using the following tools for security scanning:

- **Bandit**: Python security linter
  ```bash
  pip install bandit
  bandit -r robot_sf/
  ```

- **Safety**: Dependency vulnerability scanner
  ```bash
  pip install safety
  safety check
  ```

- **Trivy**: Container and dependency scanner
  ```bash
  trivy fs .
  ```

### Automated Scanning

Consider setting up automated security scanning in your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Security scan with Bandit
  run: |
    pip install bandit
    bandit -r robot_sf/ -f json -o bandit-report.json
```

## Disclosure Policy

- We will acknowledge receipt of your vulnerability report within 48 hours
- We will provide regular updates about our progress
- We will notify you when the vulnerability is fixed
- We will publicly disclose the vulnerability after a fix is released (with your permission)
- We will credit you in the security advisory (if you wish)

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue.

## Attribution

This security policy is adapted from security best practices for open source projects and tailored for the robot_sf project.
