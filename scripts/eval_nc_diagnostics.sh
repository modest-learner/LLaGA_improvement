#!/bin/bash
set -euo pipefail

# Independent entry point so running Table 1/5 evaluation scripts stay intact.
python eval/eval_nc_diagnostics.py "$@"
