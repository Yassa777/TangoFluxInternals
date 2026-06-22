# Paper Skeleton

Working title: **"Generative Audio Models Encode a Low-Dimensional Timbre Geometry that
Mirrors Physical Output Statistics"**

(Alt: "Probing Generative Audio Models Against Physical Ground Truth".)

Status: skeleton assembled from committed results. Two gaps flagged inline as [GAP-1]
(patching causal confirmation) and [GAP-2] (bootstrap CIs).

---

## Abstract (draft)

Interpretability of generative models is hampered by not knowing the *true* generative
factors. Text-to-audio diffusion models are a rare exception: their output's generative
factors (brightness, loudness, harmonicity, ...) are exactly and cheaply measurable from the
waveform. We use this to probe the representational geometry of two text-to-audio models
(TangoFlux, an MMDiT; AudioLDM2, a U-Net latent diffusion) against physical ground truth,
with controls that separate real structure from chance (grouped cross-validation,
out-of-sample monotonicity, split-half direction stability against a random-vector baseline).
We find that the linearly-accessible representation is **low-dimensional** -- a spectral axis
plus a loudness axis -- and that the model's internal factor geometry **quantitatively mirrors
the physical correlation structure of its output** (Pearson r = 0.88-0.98 on TangoFlux,
r = 0.95 on AudioLDM2). The encoding is linear (not merely ordinal) and non-linear probes add
nothing. The structure is sharply restricted to *stationary* timbre factors; *time-varying*
factors (attack, onset rate, decay) are recovered only weakly and only as temporal pooling is
refined, indicating they are resolution- and measurement-limited rather than absent. Finally,
these axes are linearly *readable* but not cleanly linearly *steerable*: additive steering
moves the target factor only in a narrow range before the audio degenerates, and clean control
requires replacing the activation trajectory (patching) [GAP-1]. Our methodology and the
physical-ground-truth testbed are a reusable tool for interpreting any generator with
measurable outputs.

---

## Contributions

1. A **physical-ground-truth probing methodology** for generative audio with a controls suite
   the field usually omits (grouped CV, held-out monotonicity, split-half stability vs random
   baseline, linear-vs-non-linear).
2. The **geometry-mirrors-physics** finding, with **cross-architecture replication**
   (TangoFlux r=0.88-0.98; AudioLDM2 r=0.95).
3. A characterization of the linearly-accessible representation as **low-dimensional and
   stationary**, with a clean stationary-vs-temporal boundary and a resolution/noise diagnosis
   of the temporal factors.
4. A **read-vs-write** result: factors are linearly readable but not cleanly linearly
   steerable; clean causal control needs trajectory patching [GAP-1].

---

## Section outline

1. **Introduction** -- the unknown-features problem in interpretability; audio gives exact
   ground truth; preview of the mirroring result + cross-model replication.
2. **Related work** -- linear probes, SAEs, activation steering/patching, diffusion
   interpretability; the gap = no physical ground truth to falsify "is this a real feature?".
   (Seed: `docs/research_positioning.md`.)
3. **Method**
   - Realized-corpus binning (why designed text sweeps fail; measure realized factors over a
     diverse 250-clip corpus).
   - Feature capture: per-site audio-token pooling over the DiT/U-Net, conditional CFG row.
   - Controls: grouped-CV R^2, out-of-sample monotonicity, split-half direction stability,
     random-vector baseline, PCA, non-linear probe.
4. **The factor geometry mirrors physics**
   - Per-factor encodability/stability (18-factor panel).
   - Direction-cosine matrix vs realized-correlation matrix; r=0.88 (+ CIs [GAP-2]).
   - Low-dimensionality (spectral axis + loudness axis); brightness is linear not ordinal;
     non-linearity adds nothing; dual-stream localization (TangoFlux).
5. **Cross-model replication** -- AudioLDM2, N=80, 9/10 stable, r=0.95; method ports across
   architectures.
6. **The stationary-temporal boundary** -- stable factors are all stationary; temporal factors
   resolution-limited (K-sweep K=1->8) and noise-limited (Spearman > R2; readout method does
   not matter -- PCA optimal).
7. **Read vs write: geometry is readable, not cleanly steerable** -- additive steering
   magnitude effect + saturation/degeneration; patching as the clean operator [GAP-1].
8. **Limitations & future work** -- temporal recovery (multi-seed denoising), time-localized
   control (attention bleed), more models, statistical power.
9. **Conclusion.**

---

## Figures & tables (mapped to committed artifacts)

- **F1 Method schematic** -- DiT/U-Net block -> [tokens x d] -> token pooling -> probe;
  realized factor measured from waveform. (new diagram)
- **F2 Per-factor panel (TangoFlux)** -- stability / linR2 / nonlinR2 bars, colored by
  family. Source: `results/diverse-corpus-geometry-v2/geometry-map-rows.csv`.
- **F3 Mirroring (TangoFlux)** -- direction-cosine heatmap vs realized-|corr| heatmap +
  scatter, r=0.88. Source: `results/diverse-corpus-geometry-v2/geometry-summary.json`.
- **F4 Cross-model** -- AudioLDM2 mirroring scatter, r=0.95, beside F3. Source:
  `results/second-model-audioldm2-real-geometry-v1/`.
- **F5 Stationary vs temporal** -- stability by family; temporal K-sweep curve (K=1/4/8).
  Sources: tb4/tb8 dirs + `docs/representation_geometry.md`.
- **F6 Read vs write** -- steering scale sweep (centroid vs loudness, clean vs distortion
  regime); patching result [GAP-1]. Source: `results/brightness-axis-steer-fixed-v1/`.
- **T1 Stable-factor table** (both models). **T2 Cross-model summary** (r, #stable, N, arch).

---

## Key numbers (committed)

- TangoFlux 18-factor: 12-13 stable; mirroring r=0.88 (66 pairs), r=0.98 (6-factor panel).
- Brightness linear R^2 0.59 (dual block 3); non-linear gain <= 0 across factors.
- AudioLDM2 N=80: 9/10 stable; brightness R^2 0.57 / rho 0.83; loudness R^2 0.77 / rho 0.91;
  mirroring r=0.95 (36 pairs).
- Temporal: density R^2 0.23 / rho 0.49, F0 R^2 0.34; attack/decay/tail/AM-rate weak; PCA is
  the best readout (no-PCA/PLS worse), Spearman > R^2 => metric noise.
- Steering: centroid +89 Hz at scale 10 with loudness flat (-0.7 dB); sign-flips/degenerates
  past scale ~20.

---

## Remaining work before a full-conference submission

- **[GAP-1] Patching causal confirmation** -- bin clips by realized brightness, patch
  high->low (and reverse) at a stable site, measure centroid moves while loudness does not.
  Turns the causal section from a negative into a result. (1 GPU run.)
- **[GAP-2] Bootstrap CIs** -- resample factor-pairs/clips to put CIs on the mirroring r and
  on per-factor stability/R^2. (local, no GPU.)
- Optional: third model; multi-seed temporal denoising; tighter related-work.
