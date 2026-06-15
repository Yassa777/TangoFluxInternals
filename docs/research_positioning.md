# Research Positioning

## Current Framing

This project is a mechanistic interpretability study of TangoFlux, focused on how a text-to-audio
DiT represents and realizes perceptual acoustic concepts.

The strongest current framing is not "activation steering for audio exists." That claim is already
covered by related work on music and audio diffusion models. The more defensible research question is:

> When and where does a text-to-audio model turn a semantic prompt instruction into a measurable
> acoustic property in the generated waveform?

## What We Have So Far

- A Modal-backed TangoFlux environment for batch generation and activation interventions.
- Contrastive prompt-pair sets for brightness, clear/muffled, sharp/soft, dry/reverberant, and percussive/sustained.
- Metric-grounded audio analysis for spectral, onset, transient, decay, tail, and direct-to-late proxies.
- A 960-row brightness layer-only activation patching sweep over the DiT stack.
- A top-site brightness movement test showing patched audio usually moves toward the source centroid.

## Strongest Empirical Signals

Brightness:

- Bright mean spectral centroid exceeds dark by about 543 Hz on average.
- Brightness patching produces a mean signed centroid movement of about 498 Hz.
- Top-site movement test moved toward source in 155/160 rows.

Percussive/sustained:

- Onset strength max is in the expected direction for 20/20 pairs.
- Decay and tail metrics move strongly in the expected direction.
- This is currently the best target for linear probes and follow-up causal tracing.

## Novelty Target

The likely novelty is a semantic-to-acoustic causal tracing protocol:

1. Test whether prompt labels are linearly decodable from activations.
2. Test whether realized acoustic metrics are linearly predictable from activations.
3. Patch activations and measure causal movement in waveform metrics.
4. Use probe or difference directions for steering.
5. Compare target metric movement against off-target movement.

The key contribution would be the comparison between:

- prompt-label decodability
- audio-metric predictability
- causal patch effect
- steering specificity

If these maps separate across layer, stream, or denoising step, the paper has a sharper claim:

> TangoFlux separates semantic concept representation from acoustic realization, with a measurable
> handoff from text-conditioned representations to audio-token causal control sites.

## Immediate Next Experiment

Run grouped linear probes for `percussive-sustained-v1`.

Minimum protocol:

- Capture pooled activations at all 24 DiT block sites.
- Split by `pair_id`, not by individual row.
- Train prompt-label probes: `percussive=1`, `sustained=0`.
- Train metric regressors or binary metric probes from onset/decay/tail scores.
- Compare probe maps with causal patching maps.
- Exclude or regenerate the clipped pair 19 negative sample before using audio-metric regression.

## Paper Risk

The broad space is crowded. Related work already covers activation patching and steering in audio or music generation. This project needs to avoid claiming novelty at that level.

The strongest defensible contribution is a controlled, metric-grounded causal evaluation for short text-to-audio acoustic concepts, especially outside music-only generation.
