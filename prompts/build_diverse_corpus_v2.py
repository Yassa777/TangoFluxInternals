"""Rebuild the diverse corpus with natural-caption phrasing.

Motivation
----------
TangoFlux was trained on WavCaps + AudioCaps natural-language captions and its
own hard-eval prompts are fluent scene descriptions (e.g. "A pile of coins
spills onto a wooden table with a metallic clatter ..."). The v1 corpus used
comma-stacked adjective templates ("kick drum thump, very bright and sparkling,
with a soft gradual onset, long and ringing") that are out-of-distribution and
sometimes self-contradictory, which hurt prompt adherence.

This script parses each v1 prompt back into its four factor slots
(brightness / onset / decay / density) and re-renders them as a grammatical
sentence, preserving the exact factor combination and seed of every record so
the experimental design is unchanged. It also moves to the paper's recommended
inference settings: duration=10s (native length; fixes the 3.5s truncation that
silenced late-onset clips), 50 steps, guidance 4.5.
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path(__file__).with_name("diverse_corpus_v1.jsonl")
DST = Path(__file__).with_name("diverse_corpus_v2.jsonl")

DURATION = 10.0
STEPS = 50
GUIDANCE = 4.5

# --- factor vocab -> axis -------------------------------------------------
BRIGHTNESS = {
    "dark and muffled": "with a dark, muffled tone",
    "mellow": "with a mellow tone",
    "neutral": "",
    "bright": "with a bright tone",
    "crisp and airy": "with a crisp, airy tone",
    "very bright and sparkling": "with a bright, sparkling tone",
}
ONSET = {
    "with a soft gradual onset": "fading in gently",
    "with a smooth entry": "easing in smoothly",
    "with a sudden hit": "starting abruptly",
    "with a sharp percussive attack": "with a sharp attack",
}
DECAY = {
    "quickly damped": "quickly damped",
    "very short and staccato": "short and staccato",
    "long and ringing": "ringing out for a long time",
    "sustained for several seconds": "sustained for several seconds",
    "moderately sustained": "moderately sustained",
}
DENSITY = {
    "a single isolated event": "as a single isolated event",
    "a sparse pattern": "in a sparse pattern",
    "a steady rhythm": "in a steady rhythm",
    "a rapid dense burst": "in a rapid, dense burst",
}

# --- per-source natural event phrase (subject + verb) ---------------------
SOURCE_EVENT = {
    "dog_bark": "A dog barks",
    "kick_drum_thump": "A kick drum is struck",
    "church_bell": "A church bell rings",
    "buzzing_insect": "An insect buzzes",
    "piano_note": "A single piano note is played",
    "bird_chirp": "A bird chirps",
    "blowing_wind": "Wind blows",
    "car_engine_rev": "A car engine revs",
    "choir_voices": "A choir sings",
    "crackling_fire": "A fire crackles",
    "cymbal_crash": "A cymbal crashes",
    "door_knock": "A knock sounds on a door",
    "double_bass_note": "A double bass note is played",
    "electric_organ_chord": "An electric organ chord is played",
    "flute_tone": "A flute plays a tone",
    "glass_shattering": "Glass shatters",
    "gong_strike": "A gong is struck",
    "hand_clap": "Hands clap",
    "harp_glissando": "A harp glissando is played",
    "hi-hat_tick": "A hi-hat ticks",
    "metal_pipe_clang": "A metal pipe clangs",
    "music_box_melody": "A music box plays a melody",
    "ocean_waves": "Ocean waves roll onto the shore",
    "plucked_guitar_string": "A guitar string is plucked",
    "rain_on_a_roof": "Rain falls on a roof",
    "rushing_water": "Water rushes",
    "sawtooth_synth_lead": "A sawtooth synth lead plays",
    "snare_drum_hit": "A snare drum is hit",
    "steam_hiss": "Steam hisses",
    "struck_bell": "A bell is struck",
    "synth_pad": "A synth pad plays",
    "tambourine_shake": "A tambourine shakes",
    "thunder_clap": "Thunder claps",
    "ticking_clock": "A clock ticks",
    "train_horn": "A train horn sounds",
    "triangle_ding": "A triangle dings",
    "trumpet_blast": "A trumpet blasts",
    "vinyl_crackle": "A vinyl record crackles",
    "violin_note": "A violin note is played",
    "wood_block_tap": "A wood block is tapped",
}


def parse_clauses(prompt: str, source: str) -> dict[str, str]:
    """Map the comma clauses of a v1 prompt onto the four factor axes."""
    parts = [p.strip() for p in prompt.split(",")]
    slots = {"brightness": None, "onset": None, "decay": None, "density": None}
    for clause in parts[1:]:
        if clause in BRIGHTNESS:
            slots["brightness"] = clause
        elif clause in ONSET:
            slots["onset"] = clause
        elif clause in DECAY:
            slots["decay"] = clause
        elif clause in DENSITY:
            slots["density"] = clause
        else:
            raise ValueError(f"Unrecognised clause {clause!r} in {prompt!r} (source {source})")
    return slots


def render(source: str, slots: dict[str, str]) -> str:
    event = SOURCE_EVENT[source]
    bright = BRIGHTNESS.get(slots["brightness"] or "", "")
    onset = ONSET.get(slots["onset"] or "", "")
    decay = DECAY.get(slots["decay"] or "", "")
    density = DENSITY.get(slots["density"] or "", "")

    # Tone rides on the event verb; onset then decay describe the envelope as
    # separate comma clauses so the sentence reads like a natural caption.
    head = f"{event} {bright}" if bright else event

    clauses = [head]
    if onset:
        clauses.append(onset)
    if decay:
        clauses.append(f"then {decay}" if onset else decay)
    if density:
        clauses.append(density)

    return ", ".join(clauses).rstrip(".") + "."


def main() -> None:
    out = []
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        source = r["metadata"]["source"]
        slots = parse_clauses(r["prompt"], source)
        new = dict(r)
        new["prompt"] = render(source, slots)
        new["duration"] = DURATION
        new["steps"] = STEPS
        new["guidance_scale"] = GUIDANCE
        new["metadata"] = {**r["metadata"], "v1_prompt": r["prompt"], "slots": slots}
        out.append(new)

    DST.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {len(out)} records -> {DST}")
    print("\nsamples:")
    for r in out[:8]:
        print(f"  v1: {r['metadata']['v1_prompt']}")
        print(f"  v2: {r['prompt']}\n")


if __name__ == "__main__":
    main()
