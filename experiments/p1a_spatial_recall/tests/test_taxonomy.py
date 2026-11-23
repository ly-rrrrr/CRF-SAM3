import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1a.failure_taxonomy import classify

H, W, T = 60, 100, 20
def m(x0, x1, y0=10, y1=50):
    a = np.zeros((H, W), bool); a[y0:y1, x0:x1] = True; return a
G = m(60, 90)
gt = {t: G for t in range(T)}
cases = {
 "wrong_instance": {0: {t: m(5, 30) for t in range(T)}},
 "temporal_gap": {0: {t: G for t in range(6)}},
 "id_drift": {0: {t: (G if t < 8 else m(5, 30)) for t in range(T)}},
 "merged": {0: {t: m(5, 95) for t in range(T)}},
 "partial": {0: {t: m(60, 70) for t in range(T)}},
 "fragmented": {0: {t: G for t in range(7)}, 1: {t: G for t in range(7, 14)}, 2: {t: G for t in range(14, T)}},
 "ok": {0: {t: G for t in range(T)}},
}
for want, c in cases.items():
    got = classify(c, gt, 0.5)["type"]
    print(f"{want:15s} -> {got}")
    assert got == want, (want, got)
print("OK")
