#!/usr/bin/env python3
"""Server smoke test (one video, ~1-2 min): verify the SAM3 calls used by run_p1a.

  python -m p1a.check_sam3_api --dms-repo .../SAM3-DMS --sam3-checkpoint .../sam3.pt \\
      --video .../revos/MOSE/train/08746283 --text cat
Exit code 0 only if text recall, low-threshold detection, and box / point tracker seeds all work
(each seed must return a tube covering frames other than the seed frame).
"""
import argparse
import sys
import time
import traceback

from .sam3_ops import STATE, SAM3Session, build_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms-repo", required=True)
    ap.add_argument("--sam3-checkpoint", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    p = build_predictor(a.dms_repo, a.sam3_checkpoint, a.gpu)
    ok = True
    with SAM3Session(p, a.video, True) as s:
        s.warm_text = a.text
        t = time.time(); tubes = s.text_recall(a.text)
        print(f"text_recall: {len(tubes)} tubes, frames={[len(v) for v in tubes.values()]} ({time.time()-t:.1f}s)")
        for th in (0.7, 0.5, 0.3, 0.2, 0.1):
            dets = s.detect_on_frame(a.frame, a.text, th)
            print(f"detect_on_frame thr={th}: {len(dets)} dets, scores={[round(d['score'], 2) for d in dets][:10]}")
        dets = s.detect_on_frame(a.frame, a.text, 0.1)
        if not dets:
            print("FAIL: no detection at thr 0.1, cannot test tracker seeds; try another --text / --frame")
            sys.exit(1)
        d = dets[-1]
        cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
        for name, kw in (("box", dict(box_xyxy=d["box"])), ("point", dict(points=[[cx, cy]], labels=[1])),
                         ("box+point", dict(box_xyxy=d["box"], points=[[cx, cy]], labels=[1]))):
            try:
                t = time.time(); tube = s.track_seed(a.frame, **kw)
                good = len(tube) > 1
                print(f"track_seed({name}): {len(tube)} frames ({time.time()-t:.1f}s) "
                      f"[{'text-warmed' if STATE['needs_warm'] else 'tracker-only'}] {'OK' if good else 'FAIL'}")
                ok &= good
            except Exception:
                ok = False
                print(f"track_seed({name}): EXCEPTION")
                traceback.print_exc()
        try:
            t = time.time(); tubes = s.exemplar_recall(a.frame, d["box"], a.text)
            print(f"exemplar_recall(text+box): {len(tubes)} tubes ({time.time()-t:.1f}s)  (optional, not used by revos_v2)")
        except Exception as e:
            print(f"exemplar_recall: {type(e).__name__}: {e}  (optional, not used by revos_v2)")
    print("ALL OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
