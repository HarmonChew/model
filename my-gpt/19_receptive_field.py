from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# Stage 19 intentionally reuses the complete paired-training implementation
# from Stage 18. This isolated module instance changes only experiment identity,
# output paths, and the local-window geometry policy.
_CORE_PATH = Path(__file__).with_name("18_attention_span.py")
_CORE_MODULE_NAME = "stage_18_attention_span_for_stage_19"
_CORE_SPEC = importlib.util.spec_from_file_location(
    _CORE_MODULE_NAME,
    _CORE_PATH,
)
assert _CORE_SPEC is not None and _CORE_SPEC.loader is not None
_CORE = importlib.util.module_from_spec(_CORE_SPEC)
sys.modules[_CORE_MODULE_NAME] = _CORE
_CORE_SPEC.loader.exec_module(_CORE)

_CORE.EXPERIMENT_STAGE = 19
_CORE.EXPERIMENT_TITLE = "bounded theoretical receptive field"
_CORE.COMPARISON_VARIABLE = "maximum_attention_span"
_CORE.PROTOCOL_GEOMETRY = "half_effective_receptive_field"
_CORE.PROTOCOL_NAME = "effective-receptive-field"
_CORE.INITIAL_EQUIVALENCE_LABEL = "shared-prefix"
_CORE.SUMMARY_SPAN_LABEL = "per-layer span"
_CORE.PRINT_RECEPTIVE_FIELD_DIAGNOSTIC = True
_CORE.DEFAULT_LOCAL_CONTEXT = 17
_CORE.DEFAULT_LOCAL_CHECKPOINT_PATH = (
    _CORE.CHECKPOINT_DIRECTORY / "stage_19_local_17_best_checkpoint.pt"
)
_CORE.DEFAULT_FULL_CHECKPOINT_PATH = (
    _CORE.CHECKPOINT_DIRECTORY / "stage_19_full_128_best_checkpoint.pt"
)

# Re-export the tested experiment API. Function globals continue to resolve in
# the isolated core module above, where the Stage 19 settings are installed.
for _name in dir(_CORE):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_CORE, _name)

validate_stage_19_checkpoint = _CORE.validate_experiment_checkpoint


def main() -> None:
    _CORE.main()


if __name__ == "__main__":
    main()
