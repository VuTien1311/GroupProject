# Contributing

See `docs/TEAM_ROLES.md` for the proposed three-person split.

Before a pull request:

1. Run `python -m unittest discover -s tests -v`.
2. Run `python scripts/check_repository.py`.
3. Keep model/config changes separate from presentation changes.
4. Do not include weights, BDD100K data, private images or credentials.
5. Do not reuse paper metric values without rerunning the stated experiment.

If changing inference/fusion, update the reference-provenance documentation and
the AST-equivalence test intentionally. Do not silently remove a failing test.
For GitHub uploads, review both staged paths and image/data provenance first.
