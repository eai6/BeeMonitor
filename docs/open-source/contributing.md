# Contributing

Contributions are welcome, including new study setups, hardware variants, models, bug fixes and better docs.

## Ways to help

- **Report a problem or ask a question**: [open an issue](https://github.com/eai6/BeeMonitor/issues). For a
  unit, include the camera, the Pi model and what the device page shows under Health.
- **Share a setup**: if you used BeeMonitor for a new kind of study, open an issue describing the camera
  setup, the reference and how you read the tables, so others can reuse it.
- **Share data and models**: [publish](../platform/sharing.md#publish) a labelled project or a trained model
  so others can build on it.
- **Improve the docs**: every page has an edit button (the pencil, top right) that opens the Markdown source
  on GitHub.
- **Code and hardware**: fork the repository, make your change on a branch and open a pull request. Keep a
  change to one thing, and describe how you tested it.

## Docs locally

```bash
pip install -r docs/requirements.txt
mkdocs serve        # http://127.0.0.1:8000
```

The site rebuilds and publishes when `docs/` changes on `main`.

## Licence of contributions

By contributing you agree that your contribution is licensed under the project's
[AGPLv3](../about/license.md).
