export const architecture = {
  modelName: "declare-lab/TangoFlux",
  hiddenSize: 1024,
  dualBlocks: 6,
  singleBlocks: 18,
  audioTokens: 645,
  cfgBatch: 2,
  tensorShapes: [
    ["Text encoder tokens", "[B, T_text, 1024]"],
    ["Dual audio stream", "[B=2, T_audio=645, 1024]"],
    ["Single merged stream", "[B=2, T_text + 645, 1024]"],
    ["Probe feature", "[B=2, 1024] after audio-token pooling"],
    ["Probe bundle", "[N=40, sites=24, cfg=2, D=1024]"]
  ],
  runtime: [
    ["GPU", "L40S"],
    ["Torch", "2.4.0"],
    ["Transformers", "4.44.0"],
    ["Volumes", "hf-cache, outputs, activations"]
  ]
};

export const siteNotes = {
  "transformer.transformer_blocks.3": {
    title: "Dual block 3",
    stack: "dual",
    output: "output_index=1 audio stream",
    rawShape: "[B=2, T_audio=645, D=1024]",
    probeShape: "[B=2, D=1024] after mean over audio tokens",
    components: "text/audio stream attention, cross-stream mixing, MLP, residual path",
    note: "Best dry/reverb direct-to-late probe site, and a strong percussive label / patch site."
  },
  "transformer.transformer_blocks.4": {
    title: "Dual block 4",
    stack: "dual",
    output: "output_index=1 audio stream",
    rawShape: "[B=2, T_audio=645, D=1024]",
    probeShape: "[B=2, D=1024] after mean over audio tokens",
    components: "text/audio stream attention, cross-stream mixing, MLP, residual path",
    note: "Best percussive high/low-energy probe site and one of the cleaner onset patch sites."
  },
  "transformer.single_transformer_blocks.6": {
    title: "Single block 6",
    stack: "single",
    output: "merged [text || audio], audio suffix pooled",
    rawShape: "[B=2, T_text + 645, D=1024]",
    probeShape: "[B=2, D=1024] from trailing 645 audio tokens",
    components: "merged-stream attention, MLP, residual path, audio suffix hook",
    note: "Best dry/reverb label probe site and strongest dry-direction steering site."
  },
  "transformer.single_transformer_blocks.8": {
    title: "Single block 8",
    stack: "single",
    output: "merged [text || audio], audio suffix pooled",
    rawShape: "[B=2, T_text + 645, D=1024]",
    probeShape: "[B=2, D=1024] from trailing 645 audio tokens",
    components: "merged-stream attention, MLP, residual path, audio suffix hook",
    note: "Best percussive onset-strength probe site."
  },
  "transformer.single_transformer_blocks.14": {
    title: "Single block 14",
    stack: "single",
    output: "merged [text || audio], audio suffix pooled",
    rawShape: "[B=2, T_text + 645, D=1024]",
    probeShape: "[B=2, D=1024] from trailing 645 audio tokens",
    components: "merged-stream attention, MLP, residual path, audio suffix hook",
    note: "Best percussive spectral-centroid probe site and a top brightness movement site."
  },
  "transformer.single_transformer_blocks.17": {
    title: "Single block 17",
    stack: "single",
    output: "merged [text || audio], audio suffix pooled",
    rawShape: "[B=2, T_text + 645, D=1024]",
    probeShape: "[B=2, D=1024] from trailing 645 audio tokens",
    components: "merged-stream attention, MLP, residual path, audio suffix hook",
    note: "Best percussive decay-time probe site and a strong brightness movement site."
  }
};

export const modes = {
  model: {
    title: "TangoFlux DiT Flow",
    eyebrow: "Model",
    copy:
      "Text conditioning and audio latents pass through six dual-stream DiT blocks, merge into eighteen single-stream blocks, then decode to waveform audio. Probe features pool audio tokens at all 24 hook sites.",
    rows: [
      ["Dual stream", "6 transformer_blocks"],
      ["Single stream", "18 single_transformer_blocks"],
      ["Dual audio tensor", "[2, 645, 1024]"],
      ["Single raw tensor", "[2, T_text + 645, 1024]"],
      ["Probe feature", "audio-token pooled [2, 1024]"]
    ],
    highlights: {}
  },
  brightness: {
    title: "Brightness Patching",
    eyebrow: "Causal Patch",
    copy:
      "Brightness was the first patching proof: replacing activations moved spectral centroid toward the source in most rows.",
    rows: [
      ["Baseline centroid diff", "+543.0 Hz"],
      ["Layer sweep", "960/960 valid rows"],
      ["Mean signed effect", "+498.1 Hz"],
      ["Top-site movement", "155/160 toward source"]
    ],
    highlights: {
      "transformer.single_transformer_blocks.13": 0.74,
      "transformer.single_transformer_blocks.14": 0.9,
      "transformer.single_transformer_blocks.17": 1,
      "transformer.single_transformer_blocks.15": 0.62
    }
  },
  percussive: {
    title: "Percussive/Sustained",
    eyebrow: "Probe -> Patch -> Steer",
    copy:
      "The cleanest concept: onset separates 20/20 pairs, onset becomes predictable in the single stream, and full-trajectory patching moves onset strongly.",
    rows: [
      ["Onset metric screen", "20/20 expected"],
      ["Best label probe", "0.95 at dual block 3"],
      ["Best onset R2", "+0.348 at single block 8"],
      ["Patch onset movement", "0.86-0.99 toward source"],
      ["Steering caveat", "moves spectral/energy more than onset"]
    ],
    highlights: {
      "transformer.transformer_blocks.3": 0.95,
      "transformer.transformer_blocks.4": 0.8,
      "transformer.single_transformer_blocks.8": 1,
      "transformer.single_transformer_blocks.14": 0.82,
      "transformer.single_transformer_blocks.17": 0.7
    }
  },
  dry: {
    title: "Dry/Reverberant",
    eyebrow: "Negative Control",
    copy:
      "The dry/reverb label is readable and the dry direction is steerable, but realized reverb metrics are not stable grouped linear targets in the first 20-pair set.",
    rows: [
      ["Direct-to-late screen", "16/20 expected"],
      ["Best label probe", "0.95 at single block 6"],
      ["Best direct-to-late R2", "-0.647 at dual block 3"],
      ["Best steering site", "single block 6, mean 1.07"],
      ["Specificity", "off-target movement remains high"]
    ],
    highlights: {
      "transformer.single_transformer_blocks.3": 0.7,
      "transformer.single_transformer_blocks.6": 1,
      "transformer.single_transformer_blocks.7": 0.62,
      "transformer.single_transformer_blocks.9": 0.72,
      "transformer.transformer_blocks.3": 0.78
    }
  }
};

export const workflowNodes = [
  ["Prompt pairs", "contrastive JSONL"],
  ["Modal run", "L40S generation"],
  ["Metrics", "waveform evidence"],
  ["Activations", "24 hook sites"],
  ["Probes", "GroupKFold by pair"],
  ["Patch", "trajectory replacement"],
  ["Steer", "mean direction"],
  ["Results", "CSV/JSON/Markdown"]
];
