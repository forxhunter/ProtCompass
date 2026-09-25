# Post-freeze log of the pre-registered determinacy test

Everything below happened after prereg/freeze.json was written (2026-09-23T21:32:37Z, sha256 7881049...). None of it
changes a label, a floor, a readout, a selection rule or a decision rule.

- 16:33 CDT. Scoring started on the local GPU: all 24 floors (one process) and the 22 encoders listed in
  prereg/sources_local.txt (one process), with scripts/prereg_run.py as frozen.
- 16:35 CDT. The other 10 encoders (prereg/sources_remote.txt) are scored on the remote GPU (RTX PRO 6000). They read
  a bundle written by scripts/prereg_export_subset.py: the 4,500 domains of the released seed-0 draw in their original
  order, with the embeddings and the pre-registered labels sliced to the same rows (the script asserts that the
  bundle reproduces the released rows). The remote copies of run_probing_suite.py, atom_matched_floors.py,
  prereg_floors.py and prereg_run.py (prereg/remote_scripts/) differ from the frozen files in one line each, the
  ROOT path; `diff` shows nothing else.
- Cohort B: GCPNet (pretrained and random) were extracted on cohort B with extract/prereg_extract_cohortB.py, which
  only points the domain-file path at confirmation2/structures, and merged with scripts/confirmation_merge.py
  (which rewrites confirmation2/merged/labels.h5 deterministically; prereg/labels_cohortB_merged.h5 was rebuilt and
  its sha256 is unchanged).
- 17:03 CDT. The first two remote launches ended without an error message (logs/score_a.log, score_b.log); ESM-2 150M
  was scored remotely in the foreground. The remote half was relaunched one encoder per process with exit status
  logged (prereg/remote_scripts/run_remote2.sh). A local hedge (logs/prereg_hedge_local.sh) scores the same encoders
  in reverse order once the local half ends, skipping any the remote has finished; if an encoder is scored on both
  machines, the local result is used. This rule was set before any result was read.
- 17:10 CDT. On the remote GPU, processes are occasionally ended without a message about a minute after start (the GPU
  is shared with other users' jobs); run_remote2.sh retries each encoder up to three times, and ESM-2 650M completed on
  its second attempt. A retry recomputes the same fits from the start; nothing is carried over from a failed attempt.
- 17:30 CDT. The local hedge was stopped before it scored any encoder (ESM-2 3B was waiting on GPU memory); the remote had 7 of 10 done.
- 17:45 CDT. Cohort B scoring started locally: every floor and the eight encoders of cohort_B_panel (logs/prereg_score_cohortB.sh).
- 17:43 CDT. All 56 released-split sources scored (22 encoders and 24 floors locally, 10 encoders remotely; no encoder scored on both machines). scripts/prereg_decide.py run as frozen.
- 17:43 CDT. prereg_decide.py stopped at its alignment assertion: remote prediction files carried the bundle's protein index (0..4499). scripts/prereg_relabel_remote.py mapped them to the released index (bundle protein k is released protein idx[k]) and asserted that every label's protein and label arrays then equal a locally scored source; the unmapped files are in prereg/results/remote_raw/. prereg_decide.py was then run unchanged.
- 17:45 CDT. Released-split decision (prereg/decision.json): H1 confirmed (5/5 units, 11/11 cells), H2 not confirmed (23/50 units, 0.46), H3 confirmed (Delta_G +0.136, family bootstrap [+0.081, +0.188]). Frozen interpretation rows: 'H3 holds, an 80% sign rule fails' and 'undetermined side fails'.
- 17:48 CDT. Cohort B: esm2_650m, prottrans_t5_xl_u50 and ankh_base scored locally; the local process was stopped while waiting on GPU memory (a 17 GB AlphaFold job shares the card), and the other five encoders are scored on the remote GPU from a copy of confirmation2/merged and prereg/labels_cohortB_merged.h5 (prereg/remote_scripts_b/, ROOT line only changed).
- 18:26 CDT. All cohort B sources scored; prereg_decide.py --cohort B run as frozen.
- 18:27 CDT. Cohort B decision (prereg/decision_cohortB.json): H4 confirmed, Delta_G^B +0.253, direction accuracy 20/22 = 0.909 (7 determined cells); on the six frozen cohort encoders alone 14/16 = 0.875 with one determined cell.
