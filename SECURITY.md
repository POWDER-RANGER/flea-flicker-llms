# Security Policy

## Supported Versions

Flea-Flicker LLMs is currently in active development. Security updates are provided for:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Security Context

Flea-Flicker is designed for **local inference workloads**. It is **NOT** intended for:

- Production web services with untrusted input
- Multi-tenant environments
- Processing adversarial prompts without sandboxing

### Threat Model

We assume:

- **Trusted models**: GGUF files are not malicious
- **Local execution**: No network-exposed inference endpoints
- **Single-user**: Hardware is not shared with untrusted processes

### Out of Scope

- Model extraction attacks
- Prompt injection vulnerabilities (responsibility of application layer)
- Side-channel attacks via memory access patterns
- Physical hardware attacks

## Reporting a Vulnerability

**DO NOT** open public issues for security vulnerabilities.

### How to Report

1. **Email**: Send details to [security contact will be added]
2. **Subject**: "[SECURITY] Flea-Flicker: [Brief Description]"
3. **Include**:
   - Vulnerability description
   - Steps to reproduce
   - Impact assessment
   - Proof of concept (if applicable)
   - Hardware/software environment

### Response Timeline

- **Initial response**: Within 48 hours
- **Triage**: Within 7 days
- **Fix timeline**: Depends on severity
  - Critical: 14 days
  - High: 30 days
  - Medium/Low: Next release

### Disclosure Policy

- We follow **coordinated disclosure**
- You will be credited (unless you prefer anonymity)
- We will notify you before public disclosure
- Typical embargo: 90 days from report

## Known Security Considerations

### 1. GGUF File Parsing

**Risk**: Malformed GGUF files could cause buffer overflows

**Mitigation**:
- Validate tensor shapes before allocation
- Bounds-check all file I/O operations
- Use models only from trusted sources

### 2. Memory-Mapped I/O

**Risk**: mmap'd regions could be modified by other processes

**Mitigation**:
- Use read-only mappings
- Validate checksums after loading
- Run with appropriate filesystem permissions

### 3. CUDA Kernel Launches

**Risk**: Malicious CUDA code could escalate privileges

**Mitigation**:
- We do not compile user-provided CUDA
- All kernels are statically compiled
- Use GPU sandboxing if available (container runtimes)

### 4. Resource Exhaustion

**Risk**: Unbounded allocations could cause OOM

**Mitigation**:
- Hard limits on model size
- Graceful degradation on allocation failure
- Monitoring hooks for resource usage

## Security Best Practices

### For Users

```bash
# Run with minimal privileges
sudo -u lowpriv python inference.py

# Use containers for isolation
docker run --gpus all --security-opt=no-new-privileges \\
  flea-flicker:latest

# Validate model checksums
sha256sum -c models/model.gguf.sha256

# Monitor resource usage
python inference.py --max-vram 8GB --max-disk-io 5GB/s
```

### For Developers

- **Fuzz testing**: Run `./fuzz/run_gguf_fuzzer.sh`
- **Static analysis**: `bandit -r src/`
- **Dependency auditing**: `pip-audit`
- **Sanitizers**: Build with `-fsanitize=address,undefined`

## Security Tooling

### Integrated Checks

```bash
# Run security test suite
pytest tests/security/

# Check for vulnerable dependencies
pip-audit

# Scan for secrets in commits
git-secrets --scan
```

### Continuous Monitoring

- Dependabot enabled for dependency updates
- CodeQL scanning on all PRs
- SAST via Semgrep rules

## Security Champions

Maintainers with security focus:

- [@POWDER-RANGER](https://github.com/POWDER-RANGER) - Project lead
- [Additional contacts TBD]

## Acknowledgments

We thank the security researchers who help keep Flea-Flicker safe:

- [Your name here - report a vulnerability!]

## Additional Resources

- [OWASP Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [NVIDIA GPU Security](https://developer.nvidia.com/security)
- [Hugging Face Model Security](https://huggingface.co/docs/hub/security)

---

**Last Updated**: January 2026

**Security Contact**: [Will be added - use GitHub Issues for now with "security" label]

For general questions, use GitHub Discussions. For vulnerabilities, follow the reporting process above.
