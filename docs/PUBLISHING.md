# Publishing the Python package

The distribution name is `colosseum-microcontroller`; the import name is `colosseum`. The wheel includes the 18 checkpoints and exposes the `colosseum` command.

Before the first PyPI release, register a **pending trusted publisher** on PyPI for project `colosseum-microcontroller` with these GitHub Actions fields:

- Owner: `Aqshalikhsan`
- Repository: `Colosseum-Microcontroller`
- Workflow: `publish-pypi.yml`
- Environment: `pypi`

The workflow in `.github/workflows/publish-pypi.yml` builds a wheel and source distribution and publishes on a GitHub release tagged `v0.1.2`. PyPI uses GitHub's short-lived OIDC identity; no PyPI token is stored in this repository.

To test locally before a release:

```bash
python -m pip install build
python -m build
python -m pip install --no-deps dist/colosseum_microcontroller-0.1.2-py3-none-any.whl
```

Each new release must update `project.version` in `pyproject.toml` and use a matching `v<version>` GitHub release tag.
