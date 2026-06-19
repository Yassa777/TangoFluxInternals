# TangoFlux Architecture Visualization

Standalone Three.js view of the TangoFlux internals lab architecture.

```bash
npm install
npm run dev
```

The app visualizes:

- six dual-stream `transformer.transformer_blocks.*` hook sites
- eighteen merged `transformer.single_transformer_blocks.*` hook sites
- audio-token suffix pooling for single-stream features
- tensor shapes for text/audio/merged streams and pooled probe features
- a label visibility toggle for switching between clean geometry and annotated views
- a details-panel toggle for inspecting the full scene without the right-side card
- the surrounding prompt -> Modal -> metrics -> activations -> probe/patch/steer workflow
- current brightness, percussive/sustained, and dry/reverberant evidence sites
