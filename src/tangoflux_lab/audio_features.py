from __future__ import annotations

import math
from typing import Any


def finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def db_ratio(numerator: float, denominator: float, *, eps: float = 1e-12) -> float | None:
    numerator = max(float(numerator), eps)
    denominator = max(float(denominator), eps)
    return finite_or_none(10.0 * math.log10(numerator / denominator))


def dbfs(value: float, *, eps: float = 1e-12) -> float | None:
    value = max(float(value), eps)
    return finite_or_none(20.0 * math.log10(value))


def waveform_quality_metrics(waveform: Any, sample_rate: int) -> dict[str, float | int | None]:
    import torch

    mono = waveform.mean(dim=0).float()
    if mono.numel() == 0:
        return {
            "sample_rate": int(sample_rate),
            "duration_seconds": 0.0,
            "rms_dbfs": None,
            "peak_dbfs": None,
            "clip_fraction": None,
            "silence_fraction": None,
        }

    abs_mono = mono.abs()
    rms = torch.sqrt(torch.mean(mono.square())).item()
    peak = abs_mono.max().item()
    silence_fraction = (abs_mono < 1e-4).float().mean().item()
    clip_fraction = (abs_mono >= 0.999).float().mean().item()
    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": finite_or_none(mono.numel() / float(sample_rate)),
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "clip_fraction": finite_or_none(clip_fraction),
        "silence_fraction": finite_or_none(silence_fraction),
    }


def spectral_metrics(
    waveform: Any,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    low_band: tuple[float, float] = (20.0, 2000.0),
    high_band: tuple[float, float] = (4000.0, 11025.0),
) -> dict[str, float | None]:
    import torch

    mono = waveform.mean(dim=0).float()
    if mono.numel() == 0:
        return _empty_spectral_metrics()

    n_fft = min(n_fft, max(256, int(2 ** max(8, (mono.numel() // 8).bit_length() - 1))))
    hop_length = min(hop_length, max(128, n_fft // 4))
    window = torch.hann_window(n_fft, device=mono.device)
    magnitude = torch.stft(
        mono,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    ).abs()
    power = magnitude.square()
    freqs = torch.linspace(0, sample_rate / 2, magnitude.shape[0], device=magnitude.device)

    denominator = magnitude.sum(dim=0).clamp_min(1e-12)
    centroid = (freqs[:, None] * magnitude).sum(dim=0) / denominator

    total_energy = float(power.sum().item())
    low_energy = _band_energy(power, freqs, low_band)
    high_energy = _band_energy(power, freqs, high_band)

    return {
        "spectral_centroid_mean_hz": finite_or_none(float(centroid.mean().item())),
        "spectral_centroid_median_hz": finite_or_none(float(centroid.median().item())),
        "spectral_centroid_std_hz": finite_or_none(float(centroid.std(unbiased=False).item())),
        "rolloff_85_mean_hz": _rolloff_mean(magnitude, freqs, 0.85),
        "rolloff_85_median_hz": _rolloff_median(magnitude, freqs, 0.85),
        "rolloff_95_mean_hz": _rolloff_mean(magnitude, freqs, 0.95),
        "rolloff_95_median_hz": _rolloff_median(magnitude, freqs, 0.95),
        "low_band_energy": finite_or_none(low_energy),
        "high_band_energy": finite_or_none(high_energy),
        "total_energy": finite_or_none(total_energy),
        "high_to_low_ratio": finite_or_none(high_energy / max(low_energy, 1e-12)),
        "high_to_low_db": db_ratio(high_energy, low_energy),
        "high_to_total_ratio": finite_or_none(high_energy / max(total_energy, 1e-12)),
    }


def onset_transient_metrics(
    waveform: Any,
    sample_rate: int,
    *,
    onset_hop_length: int = 256,
    onset_pre_ms: float = 20.0,
    onset_post_ms: float = 120.0,
    attack_window_ms: float = 300.0,
) -> dict[str, float | int | None]:
    import librosa
    import numpy as np

    mono = waveform.mean(dim=0).float().cpu().numpy()
    if mono.size == 0:
        return _empty_onset_metrics()

    envelope = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=onset_hop_length)
    if envelope.size == 0:
        return _empty_onset_metrics()

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=onset_hop_length,
        units="frames",
        backtrack=False,
    )
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=onset_hop_length)
    onset_samples = onset_samples[onset_samples < mono.size]
    onset_count = int(onset_samples.size)

    attack_times = _attack_times_ms(
        mono,
        sample_rate,
        onset_samples,
        hop_length=128,
        frame_length=512,
        attack_window_ms=attack_window_ms,
    )
    transient_energy = _transient_energy_fraction(
        mono,
        sample_rate,
        onset_samples,
        pre_ms=onset_pre_ms,
        post_ms=onset_post_ms,
    )
    duration_seconds = mono.size / float(sample_rate)
    inter_onset_ms = None
    if onset_samples.size >= 2:
        inter_onset_ms = np.diff(onset_samples) / float(sample_rate) * 1000.0

    return {
        "onset_strength_mean": finite_or_none(float(np.mean(envelope))),
        "onset_strength_median": finite_or_none(float(np.median(envelope))),
        "onset_strength_max": finite_or_none(float(np.max(envelope))),
        "onset_strength_p95": finite_or_none(float(np.percentile(envelope, 95))),
        "onset_count": onset_count,
        "onset_rate_per_second": finite_or_none(onset_count / max(duration_seconds, 1e-12)),
        "mean_inter_onset_interval_ms": finite_or_none(float(np.mean(inter_onset_ms)))
        if inter_onset_ms is not None
        else None,
        "median_inter_onset_interval_ms": finite_or_none(float(np.median(inter_onset_ms)))
        if inter_onset_ms is not None
        else None,
        "attack_time_ms_first": finite_or_none(attack_times[0]) if attack_times else None,
        "attack_time_ms_median": finite_or_none(float(np.median(attack_times)))
        if attack_times
        else None,
        "attack_time_ms_min": finite_or_none(float(np.min(attack_times))) if attack_times else None,
        "attack_time_ms_valid_count": len(attack_times),
        "transient_energy_total": finite_or_none(transient_energy["transient_energy_total"]),
        "non_transient_energy_total": finite_or_none(transient_energy["non_transient_energy_total"]),
        "transient_energy_fraction": finite_or_none(transient_energy["transient_energy_fraction"]),
        "transient_energy_per_onset_mean": finite_or_none(
            transient_energy["transient_energy_per_onset_mean"]
        ),
    }


def all_core_metrics(waveform: Any, sample_rate: int) -> dict[str, float | int | None]:
    return {
        **waveform_quality_metrics(waveform, sample_rate),
        **spectral_metrics(waveform, sample_rate),
        **onset_transient_metrics(waveform, sample_rate),
        **decay_reverb_metrics(waveform, sample_rate),
        **extended_timbre_metrics(waveform, sample_rate),
    }


_EXTENDED_TIMBRE_KEYS = (
    "spectral_flatness_mean",
    "zero_crossing_rate_mean",
    "spectral_bandwidth_mean_hz",
    "spectral_contrast_mean_db",
    "f0_median_hz",
    "voiced_fraction",
    "hnr_db",
    "am_rate_hz",
    "am_depth",
    "tempo_bpm",
    "pulse_clarity",
    "crest_factor_db",
)


def extended_timbre_metrics(waveform: Any, sample_rate: int) -> dict[str, float | None]:
    """Independent-axis timbre metrics: noisiness, pitch, harmonicity, modulation, rhythm.

    Each block is guarded so one failure does not void the rest. Tuned for short
    (~3-4 s) clips; values that cannot be estimated are returned as None.
    """
    import librosa
    import numpy as np

    out: dict[str, float | None] = {key: None for key in _EXTENDED_TIMBRE_KEYS}
    mono = waveform.mean(dim=0).float().cpu().numpy()
    if mono.size == 0 or float(np.sum(np.square(mono))) <= 1e-12:
        return out

    n_fft = min(2048, max(256, 2 ** int(np.log2(max(mono.size // 4, 256)))))
    hop = max(128, n_fft // 4)

    # --- noisiness / tonality ---
    try:
        flat = librosa.feature.spectral_flatness(y=mono, n_fft=n_fft, hop_length=hop)[0]
        out["spectral_flatness_mean"] = finite_or_none(float(np.mean(flat)))
    except Exception:  # noqa: BLE001
        pass
    try:
        zcr = librosa.feature.zero_crossing_rate(mono, frame_length=n_fft, hop_length=hop)[0]
        out["zero_crossing_rate_mean"] = finite_or_none(float(np.mean(zcr)))
    except Exception:  # noqa: BLE001
        pass
    try:
        bw = librosa.feature.spectral_bandwidth(y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop)[0]
        out["spectral_bandwidth_mean_hz"] = finite_or_none(float(np.mean(bw)))
    except Exception:  # noqa: BLE001
        pass
    try:
        contrast = librosa.feature.spectral_contrast(y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop)
        out["spectral_contrast_mean_db"] = finite_or_none(float(np.mean(contrast)))
    except Exception:  # noqa: BLE001
        pass

    # --- pitch / harmonicity ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            mono, fmin=50.0, fmax=2000.0, sr=sample_rate, frame_length=max(n_fft, 1024)
        )
        voiced = f0[np.isfinite(f0)]
        out["voiced_fraction"] = finite_or_none(float(np.mean(voiced_flag)))
        out["f0_median_hz"] = finite_or_none(float(np.median(voiced))) if voiced.size else None
    except Exception:  # noqa: BLE001
        pass
    try:
        harmonic = librosa.effects.harmonic(mono)
        residual = mono - harmonic
        h_energy = float(np.sum(np.square(harmonic)))
        n_energy = float(np.sum(np.square(residual)))
        out["hnr_db"] = db_ratio(h_energy, n_energy)
    except Exception:  # noqa: BLE001
        pass

    # --- amplitude modulation (tremolo / roughness / pulsing) ---
    try:
        env = librosa.feature.rms(y=mono, frame_length=n_fft, hop_length=hop)[0]
        if env.size >= 8 and float(np.mean(env)) > 1e-8:
            env_rate = sample_rate / hop
            centered = env - float(np.mean(env))
            spec = np.abs(np.fft.rfft(centered))
            freqs = np.fft.rfftfreq(centered.size, d=1.0 / env_rate)
            band = (freqs >= 1.0) & (freqs <= 30.0)
            if band.any() and spec[band].size:
                peak = int(np.argmax(spec[band]))
                out["am_rate_hz"] = finite_or_none(float(freqs[band][peak]))
                out["am_depth"] = finite_or_none(float(spec[band][peak] / (np.mean(env) * env.size)))
    except Exception:  # noqa: BLE001
        pass

    # --- rhythm ---
    try:
        onset_env = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop)
        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sample_rate, hop_length=hop)
        out["tempo_bpm"] = finite_or_none(float(np.atleast_1d(tempo)[0]))
        if onset_env.size >= 8:
            ac = librosa.autocorrelate(onset_env)
            if ac.size > 1 and ac[0] > 1e-8:
                out["pulse_clarity"] = finite_or_none(float(np.max(ac[1:]) / ac[0]))
    except Exception:  # noqa: BLE001
        pass

    # --- crest factor ---
    try:
        rms = float(np.sqrt(np.mean(np.square(mono))))
        peak = float(np.max(np.abs(mono)))
        out["crest_factor_db"] = db_ratio(peak, rms)
    except Exception:  # noqa: BLE001
        pass

    return out


def decay_reverb_metrics(
    waveform: Any,
    sample_rate: int,
    *,
    direct_ms: float = 50.0,
    late_start_ms: float = 80.0,
    late_end_ms: float = 600.0,
) -> dict[str, float | None]:
    import librosa
    import numpy as np

    mono = waveform.mean(dim=0).float().cpu().numpy()
    if mono.size == 0:
        return _empty_decay_reverb_metrics()

    total_energy = float(np.sum(np.square(mono)))
    if total_energy <= 1e-12:
        return _empty_decay_reverb_metrics()

    event_sample = _strongest_event_sample(mono, sample_rate)
    post_energy = _window_energy(mono, event_sample, mono.size)
    direct_end = min(mono.size, event_sample + _ms_to_samples(direct_ms, sample_rate))
    late_start = min(mono.size, event_sample + _ms_to_samples(late_start_ms, sample_rate))
    late_end = min(mono.size, event_sample + _ms_to_samples(late_end_ms, sample_rate))

    direct_energy = _window_energy(mono, event_sample, direct_end)
    late_energy = _window_energy(mono, late_start, late_end)
    direct_to_late_ratio = direct_energy / max(late_energy, 1e-12)

    tail_250 = _tail_fraction(mono, sample_rate, event_sample, 250.0, post_energy)
    tail_500 = _tail_fraction(mono, sample_rate, event_sample, 500.0, post_energy)
    tail_1000 = _tail_fraction(mono, sample_rate, event_sample, 1000.0, post_energy)
    late_300 = _tail_fraction(mono, sample_rate, event_sample, 300.0, post_energy)
    late_700 = _tail_fraction(mono, sample_rate, event_sample, 700.0, post_energy)

    decay = _decay_curve_metrics(mono, sample_rate, event_sample)
    reverb_proxy_score = None
    if late_300 is not None and late_700 is not None:
        inverse_direct = 1.0 / (1.0 + direct_to_late_ratio)
        reverb_proxy_score = (late_300 + late_700 + inverse_direct) / 3.0

    return {
        "event_sample": int(event_sample),
        "event_time_seconds": finite_or_none(event_sample / float(sample_rate)),
        "post_onset_energy_100ms": finite_or_none(
            _window_energy(mono, event_sample, event_sample + _ms_to_samples(100.0, sample_rate))
        ),
        "post_onset_energy_250ms": finite_or_none(
            _window_energy(mono, event_sample, event_sample + _ms_to_samples(250.0, sample_rate))
        ),
        "post_onset_energy_500ms": finite_or_none(
            _window_energy(mono, event_sample, event_sample + _ms_to_samples(500.0, sample_rate))
        ),
        "post_onset_energy_1000ms": finite_or_none(
            _window_energy(mono, event_sample, event_sample + _ms_to_samples(1000.0, sample_rate))
        ),
        "tail_energy_fraction_250ms": finite_or_none(tail_250),
        "tail_energy_fraction_500ms": finite_or_none(tail_500),
        "tail_energy_fraction_1000ms": finite_or_none(tail_1000),
        "late_energy_fraction_300ms": finite_or_none(late_300),
        "late_energy_fraction_700ms": finite_or_none(late_700),
        "direct_energy": finite_or_none(direct_energy),
        "late_energy": finite_or_none(late_energy),
        "direct_to_late_ratio": finite_or_none(direct_to_late_ratio),
        "direct_to_late_db": db_ratio(direct_energy, late_energy),
        "decay_time_to_minus_20db_ms": finite_or_none(decay["decay_time_to_minus_20db_ms"]),
        "decay_slope_db_per_second": finite_or_none(decay["decay_slope_db_per_second"]),
        "reverb_proxy_score": finite_or_none(reverb_proxy_score),
    }


def _empty_spectral_metrics() -> dict[str, None]:
    return {
        "spectral_centroid_mean_hz": None,
        "spectral_centroid_median_hz": None,
        "spectral_centroid_std_hz": None,
        "rolloff_85_mean_hz": None,
        "rolloff_85_median_hz": None,
        "rolloff_95_mean_hz": None,
        "rolloff_95_median_hz": None,
        "low_band_energy": None,
        "high_band_energy": None,
        "total_energy": None,
        "high_to_low_ratio": None,
        "high_to_low_db": None,
        "high_to_total_ratio": None,
    }


def _empty_onset_metrics() -> dict[str, None | int]:
    return {
        "onset_strength_mean": None,
        "onset_strength_median": None,
        "onset_strength_max": None,
        "onset_strength_p95": None,
        "onset_count": 0,
        "onset_rate_per_second": None,
        "mean_inter_onset_interval_ms": None,
        "median_inter_onset_interval_ms": None,
        "attack_time_ms_first": None,
        "attack_time_ms_median": None,
        "attack_time_ms_min": None,
        "attack_time_ms_valid_count": 0,
        "transient_energy_total": None,
        "non_transient_energy_total": None,
        "transient_energy_fraction": None,
        "transient_energy_per_onset_mean": None,
    }


def _empty_decay_reverb_metrics() -> dict[str, None]:
    return {
        "event_sample": None,
        "event_time_seconds": None,
        "post_onset_energy_100ms": None,
        "post_onset_energy_250ms": None,
        "post_onset_energy_500ms": None,
        "post_onset_energy_1000ms": None,
        "tail_energy_fraction_250ms": None,
        "tail_energy_fraction_500ms": None,
        "tail_energy_fraction_1000ms": None,
        "late_energy_fraction_300ms": None,
        "late_energy_fraction_700ms": None,
        "direct_energy": None,
        "late_energy": None,
        "direct_to_late_ratio": None,
        "direct_to_late_db": None,
        "decay_time_to_minus_20db_ms": None,
        "decay_slope_db_per_second": None,
        "reverb_proxy_score": None,
    }


def _band_energy(power: Any, freqs: Any, band: tuple[float, float]) -> float:
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not bool(mask.any().item()):
        return 0.0
    return float(power[mask].sum().item())


def _rolloff_values(magnitude: Any, freqs: Any, fraction: float) -> Any:
    import torch

    cumulative = torch.cumsum(magnitude, dim=0)
    thresholds = cumulative[-1, :].clamp_min(1e-12) * fraction
    indices = torch.argmax((cumulative >= thresholds[None, :]).to(torch.int64), dim=0)
    return freqs[indices]


def _rolloff_mean(magnitude: Any, freqs: Any, fraction: float) -> float | None:
    values = _rolloff_values(magnitude, freqs, fraction)
    return finite_or_none(float(values.mean().item()))


def _rolloff_median(magnitude: Any, freqs: Any, fraction: float) -> float | None:
    values = _rolloff_values(magnitude, freqs, fraction)
    return finite_or_none(float(values.median().item()))


def _attack_times_ms(
    mono: Any,
    sample_rate: int,
    onset_samples: Any,
    *,
    hop_length: int,
    frame_length: int,
    attack_window_ms: float,
) -> list[float]:
    import librosa
    import numpy as np

    if onset_samples.size == 0:
        return []

    rms = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=hop_length)[0]
    frame_samples = librosa.frames_to_samples(np.arange(rms.size), hop_length=hop_length)
    attack_times: list[float] = []
    window_samples = int(round(attack_window_ms / 1000.0 * sample_rate))

    for onset_sample in onset_samples:
        start = int(onset_sample)
        end = min(start + window_samples, int(frame_samples[-1]) if frame_samples.size else start)
        if end <= start:
            continue

        mask = (frame_samples >= start) & (frame_samples <= end)
        if not mask.any():
            continue
        local_frames = frame_samples[mask]
        local_env = rms[mask]
        peak = float(np.max(local_env))
        if peak <= 1e-8:
            continue

        ten = peak * 0.10
        ninety = peak * 0.90
        above_ten = np.flatnonzero(local_env >= ten)
        above_ninety = np.flatnonzero(local_env >= ninety)
        if above_ten.size == 0 or above_ninety.size == 0:
            continue
        start_idx = int(above_ten[0])
        end_candidates = above_ninety[above_ninety >= start_idx]
        if end_candidates.size == 0:
            continue
        end_idx = int(end_candidates[0])
        attack_times.append((local_frames[end_idx] - local_frames[start_idx]) / sample_rate * 1000.0)

    return attack_times


def _transient_energy_fraction(
    mono: Any,
    sample_rate: int,
    onset_samples: Any,
    *,
    pre_ms: float,
    post_ms: float,
) -> dict[str, float | None]:
    import numpy as np

    total_energy = float(np.sum(np.square(mono)))
    if total_energy <= 1e-12 or onset_samples.size == 0:
        return {
            "transient_energy_total": None,
            "non_transient_energy_total": finite_or_none(total_energy),
            "transient_energy_fraction": None,
            "transient_energy_per_onset_mean": None,
        }

    mask = np.zeros(mono.size, dtype=bool)
    pre = int(round(pre_ms / 1000.0 * sample_rate))
    post = int(round(post_ms / 1000.0 * sample_rate))
    per_onset: list[float] = []
    for onset_sample in onset_samples:
        start = max(0, int(onset_sample) - pre)
        end = min(mono.size, int(onset_sample) + post)
        if end <= start:
            continue
        mask[start:end] = True
        per_onset.append(float(np.sum(np.square(mono[start:end]))))

    transient_energy = float(np.sum(np.square(mono[mask])))
    non_transient_energy = max(total_energy - transient_energy, 0.0)
    return {
        "transient_energy_total": finite_or_none(transient_energy),
        "non_transient_energy_total": finite_or_none(non_transient_energy),
        "transient_energy_fraction": finite_or_none(transient_energy / max(total_energy, 1e-12)),
        "transient_energy_per_onset_mean": finite_or_none(float(np.mean(per_onset)))
        if per_onset
        else None,
    }


def _strongest_event_sample(mono: Any, sample_rate: int) -> int:
    import librosa
    import numpy as np

    hop_length = 256
    envelope = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop_length)
    if envelope.size:
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=envelope,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=False,
        )
        if onset_frames.size:
            strongest_frame = int(onset_frames[np.argmax(envelope[onset_frames])])
        else:
            strongest_frame = int(np.argmax(envelope))
        return int(min(librosa.frames_to_samples(strongest_frame, hop_length=hop_length), mono.size - 1))

    return int(np.argmax(np.abs(mono)))


def _ms_to_samples(milliseconds: float, sample_rate: int) -> int:
    return int(round(milliseconds / 1000.0 * sample_rate))


def _window_energy(mono: Any, start: int, end: int) -> float:
    import numpy as np

    start = max(0, min(int(start), mono.size))
    end = max(start, min(int(end), mono.size))
    return float(np.sum(np.square(mono[start:end])))


def _tail_fraction(
    mono: Any,
    sample_rate: int,
    event_sample: int,
    start_ms: float,
    post_energy: float,
) -> float | None:
    if post_energy <= 1e-12:
        return None
    start = event_sample + _ms_to_samples(start_ms, sample_rate)
    return _window_energy(mono, start, mono.size) / max(post_energy, 1e-12)


def _decay_curve_metrics(mono: Any, sample_rate: int, event_sample: int) -> dict[str, float | None]:
    import librosa
    import numpy as np

    hop_length = 256
    frame_length = 1024
    rms = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size < 2:
        return {"decay_time_to_minus_20db_ms": None, "decay_slope_db_per_second": None}

    frame_samples = librosa.frames_to_samples(np.arange(rms.size), hop_length=hop_length)
    start_frame = int(np.searchsorted(frame_samples, event_sample, side="left"))
    start_frame = min(start_frame, rms.size - 1)
    local = rms[start_frame:]
    if local.size < 2:
        return {"decay_time_to_minus_20db_ms": None, "decay_slope_db_per_second": None}

    peak_offset = int(np.argmax(local))
    peak_frame = start_frame + peak_offset
    peak_value = float(rms[peak_frame])
    if peak_value <= 1e-10:
        return {"decay_time_to_minus_20db_ms": None, "decay_slope_db_per_second": None}

    tail = rms[peak_frame:]
    tail_times = (frame_samples[peak_frame:] - frame_samples[peak_frame]) / float(sample_rate)
    tail_db = 20.0 * np.log10(np.maximum(tail, 1e-12) / peak_value)
    below = np.flatnonzero(tail_db <= -20.0)
    decay_time_ms = float(tail_times[int(below[0])] * 1000.0) if below.size else float(tail_times[-1] * 1000.0)

    slope = None
    if tail_times.size >= 3 and float(tail_times[-1]) > 0:
        slope = float(np.polyfit(tail_times, tail_db, 1)[0])

    return {
        "decay_time_to_minus_20db_ms": decay_time_ms,
        "decay_slope_db_per_second": slope,
    }
