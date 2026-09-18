#!/usr/bin/env bash
set -euo pipefail
# Prepare the development environment.
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m pytest
