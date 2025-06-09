# AGENTS

This repository contains a modular ML pipeline under `ml/pipeline`.

* When adding or modifying Python files inside `ml/pipeline` or `ml`, always run:
  `python -m py_compile ml/pipeline/*.py`
  to ensure there are no syntax errors. Include this in your PR testing steps.
* Keep the pipeline modules self-contained and import them in `ml/pipeline/__init__.py` if
  they are intended to be part of the training workflow.
* Use relative imports within the `ml/pipeline` package.
