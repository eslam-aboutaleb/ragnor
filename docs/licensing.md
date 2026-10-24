# Ragnor Licensing & Distribution Policy

## Current License

Ragnor is currently distributed under the **PolyForm Noncommercial License 1.0.0** (`PolyForm-Noncommercial-1.0.0`), as declared in:

- [`LICENSE`](../LICENSE)
- [`pyproject.toml`](../pyproject.toml) (`project.license = "PolyForm-Noncommercial-1.0.0"`)
- [`README.md`](../README.md)

An automated release gate (`tests/test_release_metadata_contract.py`) verifies that `pyproject.toml`, `LICENSE`, `README.md`, and this document remain strictly synchronized so no contradictory license metadata can be published to PyPI or GitHub Releases.

## Scope of Permitted Use

Under the active `PolyForm-Noncommercial-1.0.0` license:

1. **Permitted**: Personal research, academic use, non-commercial evaluation, educational use, and contributions back to the project.
2. **Restricted without separate commercial authorization**: Production commercial deployment, offering Ragnor as a hosted/managed SaaS or DBaaS product, or embedding Ragnor in paid commercial offerings.

## Governance for Future License Transitions (#19)

Any future transition of the repository license (such as the evaluated move to **Elastic License 2.0 (ELv2)** tracked in [#19](https://github.com/eslam-aboutaleb/ragnor/issues/19)) requires explicit completion of the following governance checklist before modifying `LICENSE` or `pyproject.toml`:

1. **Copyright & Contributor Audit**: Confirm all commits and contributions in the repository are covered by copyright ownership or explicit contributor sign-off.
2. **Third-Party Dependency Compatibility**: Verify all runtime and optional dependencies in `pyproject.toml` are permissive (MIT, Apache-2.0, BSD-3-Clause, PSF-2.0) and compatible with target license terms.
3. **Maintainer Legal Sign-Off**: Explicit maintainer approval on the relicensing PR updating `LICENSE`, `pyproject.toml`, `README.md`, and `tests/test_release_metadata_contract.py` atomically.
