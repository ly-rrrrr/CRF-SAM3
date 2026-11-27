"""Sa2VA-style joint mixing, converted from per-video samples to per-expression samples.

Sa2VA (projects/sa2va/configs/sa2va_in30_8b.py, sa2va_data_03_refvos.py): one sample = 1 video with
5 expressions drawn with replacement, len = videos x repeats; repeats ReVOS 10, MeViS 4, Ref-YTVOS 4.
Our sample = 1 expression. Keep each dataset's share of expression-visits per epoch equal to Sa2VA's:
    r_d = round(c * 5 * V_d * R_d / E_d),   c = 1 / min_d(5 * V_d * R_d / E_d)
so the smallest r is 1. DAVIS is not trained by Sa2VA; we give it R = 4 (its MeViS / Ref-YTVOS value).
"""
from __future__ import annotations

from typing import Dict, Tuple

SA2VA_R = {"revos": 10, "mevis": 4, "refytvos": 4, "davis17": 4}
SA2VA_K = 5   # expressions per Sa2VA sample


def repeats(counts: Dict[str, Tuple[int, int]], R: Dict[str, int] = SA2VA_R) -> Dict[str, dict]:
    """counts: dataset -> (videos V, expressions E) actually in the training data."""
    w = {d: SA2VA_K * v * R[d] / e for d, (v, e) in counts.items() if e > 0}
    if not w:
        return {}
    c = 1.0 / min(w.values())
    return {d: {"videos": counts[d][0], "expressions": counts[d][1], "sa2va_R": R[d],
                "weight": round(w[d], 4), "r": max(1, int(round(c * w[d]))),
                "samples": max(1, int(round(c * w[d]))) * counts[d][1]} for d in w}
