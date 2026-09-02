# CNN10 corruption-space audit and remediation log

## Material Passport

- Project: NoiseGap
- Audit date: 2026-09-02
- Status: raw results verified; remediation in progress
- Verification code revision: `e3df1b9`
- Statistical replication unit: independent training seed
- Current seed count: 3
- New training performed by this audit: no
- Primary metric: UAR; supporting metrics: Macro-F1 and Accuracy

This is the corrected living audit for the TIMIT Sentence-Type CNN10
feature-space and waveform-space experiments and the SpeechCommands control.
It supersedes the uncommitted report from the separate
`claude/cnn10-corruption-audit-c98419` worktree where statements conflict.

`G→R` means training with Gaussian corruption and testing with recorded-noise
corruption. `R→G` means the reverse. Every directional difference below is
`G→R − R→G`.

## Evidence and reproducible outputs

The command `noisegap-diagnose` reconstructs the confusion matrix from the test
manifest and `test_results.csv`, recalculates Accuracy, UAR, and Macro-F1, and
requires exact agreement with the provenance-bound `test_holistic.yaml`.
It also reports class prediction shares and marks a cell as collapsed when one
class receives at least 90% of its predictions.

The command `noisegap-contrast` pairs the two transfer directions within each
training seed before calculating the mean difference, sample SD, 95% t interval,
and uncorrected paired t-test. It never treats SNR cells as independent
replicates.

Server output root:

```text
/data/agent-host/projects/NoiseGap-corrected/generated/audit-2026-09-02
```

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `timit-feature-legacy-cells.csv` | 432 | `fde18588f3717700b73def0c6519f4d9ab811c5987d3c8e02f463158825b6a04` |
| `timit-waveform-v3-cells.csv` | 432 | `e06c4b9262e2c5e028359b57e4b82fa14384c42c461ebadedb10042ba8e6bca1` |
| `speechcommands-waveform-cnn10-cells.csv` | 72 | `3a88ec163dabbf65f77aa536079afe6633374ed3d37fd6eef90a009a625d1ee4` |
| `timit-feature-legacy-contrasts.csv` | 56 | `4c9772e192523a7173fcd964117f0048528c8df1f6fdb07aa1a174a2bfc23e14` |
| `timit-waveform-v3-contrasts.csv` | 56 | `5bb968510821f727ec612ec0668cc22ff07b4bba5b54684cd872e5371c1374dd` |
| `speechcommands-waveform-cnn10-contrasts.csv` | 36 | `f37bb714672e568ae82562886d491e29242a2014bee6fd0472ce06ab2a2df4b6` |

The TIMIT runs contain a provenance-bound test split and hashed holistic
metrics. They predate raw-prediction hashes, so the diagnostic output records
`test_results_provenance_verified=False` while independently requiring its
recomputed metrics to equal the hashed holistic metrics exactly.

The SpeechCommands run predates both raw-prediction and dataset-split hashes.
Its exact test CSV was supplied explicitly and its output remains marked
`test_split_provenance_verified=False`; it cannot be promoted to the same
provenance status as TIMIT.

## Verified current results

### Three-seed overall directional contrasts

Values are percentage points and 95% t intervals across the three paired seed
differences.

| Pipeline | Metric | G→R | R→G | Difference | 95% CI |
|---|---|---:|---:|---:|---:|
| feature, article legacy | Accuracy | 56.97 | 54.74 | +2.23 | [−1.02, +5.47] |
| feature, article legacy | UAR | 57.51 | 54.56 | +2.95 | [−2.89, +8.78] |
| feature, article legacy | Macro-F1 | 54.50 | 49.58 | +4.92 | [+1.16, +8.68] |
| waveform v3 | Accuracy | 61.08 | 65.16 | −4.08 | [−11.18, +3.01] |
| waveform v3 | UAR | 59.12 | 62.80 | −3.68 | [−8.38, +1.03] |
| waveform v3 | Macro-F1 | 57.26 | 60.27 | −3.01 | [−12.08, +6.05] |

These are descriptive three-seed observations. The p-values in the CSV are
explicitly uncorrected; no claim of robust statistical significance is made.

### Low-SNR collapse does not explain the complete result

The TIMIT test distribution is SA=20%, SI=30%, SX=50%. A constant SX predictor
therefore obtains 50% Accuracy, while a constant SI predictor obtains 30%.
That 50% value is the majority-class baseline, not chance. Uniform random
Accuracy and the constant-class UAR baseline are both 33.3%.

At test −5 dB:

| Pipeline | Metric | G→R | R→G | Difference |
|---|---|---:|---:|---:|
| feature, article legacy | Accuracy | 49.31 | 38.80 | +10.52 |
| feature, article legacy | UAR | 44.35 | 33.16 | +11.19 |
| feature, article legacy | Macro-F1 | 41.13 | 21.29 | +19.84 |
| waveform v3 | Accuracy | 53.60 | 53.19 | +0.41 |
| waveform v3 | UAR | 45.68 | 44.28 | +1.41 |
| waveform v3 | Macro-F1 | 43.89 | 37.65 | +6.24 |

For feature-space test −5 dB, 2/18 G→R cells and 13/18 R→G cells collapse.
For waveform test −5 dB, the corresponding counts are 6/18 and 10/18.
Collapse therefore strongly confounds Accuracy and is itself a transfer failure
mode, but the direction difference persists in class-balanced metrics. The
statement that low-SNR Accuracy measures only collapse and not transfer is too
strong.

## Confirmed methodological defects and claim boundaries

### 1. Legacy recorded feature noise has a time/mel axis defect

The article-compatible `LegacyArticleRecordedLogMelNoise` generates
`[channel, mel, time]`, fits its final axis to 64 using the speech mel count,
then interpolates it as though it were `[channel, time, mel]`. This severely
distorts the temporal structure. It remains waveform-derived noise, but it is
not a faithful representation of the recorded AudioSet event structure.

Required action: retain this run only as a historical reproduction. Remove the
mechanistic explanation based on sparse AudioSet events and run a corrected
feature-space experiment with the same 32 kHz PANN frontend, padding, data,
noise manifests, development policy, and evaluation realizations as the
waveform experiment.

### 2. The existing comparison does not isolate injection point

Legacy feature versus waveform v3 changes the injection position, recorded
domain representation, frontend, padding representation, development-noise
policy, and random-noise realization. The observed sign change proves that the
architecture-only interpretation is unstable to the corruption pipeline. It
does not prove that injection position alone caused it.

Current defensible statement: directional transfer has an SNR-dependent
crossover, and its observed location changes across complete corruption
pipelines.

### 3. Effective representation-space SNR differs by spectrum

The existing 40-utterance diagnostic reproduces a difference between nominal
waveform SNR and the power change after the PANN frontend. At nominal −5 dB,
Gaussian is approximately −1.0 dB and several AudioSet categories are
approximately −3.8 to −4.7 dB in that diagnostic.

This metric includes frontend nonlinearity and STFT cross-terms and becomes
unstable at high SNR. It is useful as a representation-space diagnostic, not as
ground-truth waveform SNR. The hypothesis that it causes the directional result
requires a calibrated sensitivity experiment.

Required action: retain nominal waveform SNR as the physical protocol, publish
the representation-space diagnostic beside it, and add a matched-effective-SNR
sensitivity analysis rather than silently replacing one definition with the
other.

### 4. SpeechCommands is not a valid recorded-noise control

Its training recorded domain includes `white_noise.wav` with spectral flatness
1.000 and `pink_noise.wav`; its development and test phases use the same noise
manifest. The run also predates dataset-split provenance. The result can be kept
as an engineering smoke result but not as evidence about transfer between
Gaussian and real environmental noise.

Required action: use source/file-disjoint train, development, and test splits
from a larger recorded-noise corpus such as MUSAN, DEMAND, or a curated
AudioSet split. Splitting only the four remaining SpeechCommands files would be
too small for a persuasive control.

### 5. Three seeds are insufficient for the headline sign

The waveform overall paired differences by seed are −4.96, −0.89, and −6.40
percentage points in Accuracy. The 95% interval includes zero. Some legacy
feature p-values are below 0.05 before multiple-comparison correction, but none
should be described as robust significance with only three seeds and many
related comparisons.

Required action: use at least five and preferably eight training seeds for the
predeclared headline contrasts. Per-utterance bootstrap may describe conditional
test-set uncertainty but cannot replace independent training seeds.

### 6. Recorded waveform crop selection can amplify near-silence

The current implementation rejects only exactly zero-power crops. Almost silent
crops can therefore be scaled dramatically to the target SNR. A Pink-noise
diagnostic at nominal 10 dB produced a median effective SNR near 11 dB but
extreme values above 40 dB and 170 dB for very low-power crops.

Required action: implement and validate an active-crop or source-relative RMS
threshold, record crop RMS and applied scale, then use the same policy in every
waveform condition.

### 7. Training augmentation streams can collide across adjacent seeds

autrainer seeds worker `w` as `run_seed + w`. Thus seed 0 worker 1 and seed 1
worker 0 can share an augmentation seed even though model initialization and
shuffle remain different.

Required action: derive the augmentation base seed from the run seed with a
large reserved worker range and record both seeds in provenance. Evaluation
realizations remain fixed across training seeds.

### 8. Epoch-budget evidence is only a warning

Waveform best epoch equals the 15-epoch boundary in 7/36 trainings; feature
legacy does so in 5/36, not 5/37. This suggests that the budget may bind for
some conditions but does not prove that every model is undertrained.

Required action: extend representative boundary runs with checkpointing or
early stopping before changing the full matrix budget.

## Provenance corrections

- The experimental commits are pushed to `origin/feat/speechcommands-waveform`;
  they are absent from `main`, not local-only.
- `timit-waveform-cnn10-v2` still exists on the server. Its 432 metric rows and
  checkpoint hashes equal v3, although aggregate CSV bytes differ because of
  provenance/path fields.
- The historical source revision `f46af323` belongs to the repository documented
  in `NOTICE.md` (`yiyi-cs/ASL-ConNo`).
- The individual raw files behind the article's two-run CNN10 matrix were not
  found in the current local or server artifacts, so the quoted per-run gaps
  cannot presently be independently rechecked.

## Remediation gates

1. **Completed:** provenance-aware class/collapse diagnostics and seed-paired
   contrasts for all existing cells.
2. **Next:** matched 32 kHz feature-space frontend, padding mask, independent
   augmentation seeds, and recorded-crop policy, all covered by unit tests.
3. **Then:** source-disjoint real-noise manifests for SpeechCommands or replace
   that control with a suitable recorded corpus.
4. **Before training:** local transform parity tests and server CPU/GPU smoke
   runs with clean Git provenance.
5. **Headline experiment:** corrected feature-space and waveform CNN10 under one
   contract with 5–8 seeds and predeclared contrasts.
6. **Paper update:** UAR first, Macro-F1 and Accuracy secondary, raw seed points,
   collapse diagnostics, 33.3% chance and 50% majority baselines, and cautious
   crossover wording.
