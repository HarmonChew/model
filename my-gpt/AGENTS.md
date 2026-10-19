# Project agent instructions

These instructions apply to all work under `my-gpt/`.

## Stage reports

- A stage report is evidence: it contains the stage script's complete stdout in
  its original order and precision. Do not summarize, omit, reformat, or add
  commentary inside the stdout block.
- Generate reports with `experiments/publish_report.py`; do not copy output by
  hand. Keep runner metadata outside the stdout block.
- Put agent-authored conclusions, comparisons, caveats, and recommendations in
  `docs/interpretations/stage_NN.md`.
- Keep raw logs, checkpoints, JSON measurements, figures, reports, and
  interpretations in the locations defined by `docs/ARTIFACTS.md`.
- Keep `README.md` as an index. Link to evidence and interpretation instead of
  duplicating either one there.
- Historical stages use the legacy layout until they are deliberately migrated.
  When touching one, prefer migrating it to this layout rather than extending
  the legacy cumulative documents.

## Agent handoffs

- Lead with the outcome and keep the final response concise.
- Include only material changes, verification, blockers, and paths to relevant
  artifacts unless the user requests more detail.
- Do not paste training transcripts or long metric tables into chat; link to the
  stage report.

## Subagent dispatch hazards

- **Before** decoding any binary or image file (a `read` of `.png`, `.jpg`,
  `.pt`, or video, or any inline image), confirm no `task` subagent is in
  flight: the parent's context is replayed into child prompts, so an image in
  the parent's history makes every child request fail with a 400 on models that
  accept zero images — and when a child does die that way, inspect the artifact
  tree before redoing its work, because its file writes usually survive (possibly
  truncated mid-file).

