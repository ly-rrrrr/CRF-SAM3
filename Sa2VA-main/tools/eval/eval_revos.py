###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os.path as osp
import time
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import multiprocessing as mp
import pandas as pd
from pycocotools import mask as cocomask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from third_parts.revos.utils.metircs import db_eval_iou, db_eval_boundary, get_r2vos_accuracy, get_r2vos_robustness

def eval_queue(q, rank, out_dict):
    while True:
        job = q.get()
        if job is None:
            break
        vid_name, exp = job
        vid = exp_dict[vid_name]
        exp_name = f'{vid_name}_{exp}'

        pred = mask_pred_dict[vid_name][exp]

        vid_len, h, w = len(vid['frames']), vid['height'], vid['width']
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        foreground_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            # all instances in the same frame
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)
            mask_fore_rle = mask_dict_foreground[vid_name]["masks_rle"][frame_idx]
            mask_fore = cocomask.decode(mask_fore_rle)
            mask_fore = mask_fore.sum(axis=2).astype(np.uint8) if mask_fore.ndim == 3 else mask_fore.astype(np.uint8)
            foreground_masks[frame_idx] = mask_fore

            pred_mask = cocomask.decode(pred['prediction_masks'][frame_idx])
            pred_masks[frame_idx] += pred_mask
        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        a = get_r2vos_accuracy(gt_masks, pred_masks).mean()
        r = get_r2vos_robustness(gt_masks, pred_masks, foreground_masks).mean()

        out_dict[exp_name] = [j, f, a, r]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str, )
    parser.add_argument("--exp_path", type=str, default="data/video_datas/revos/meta_expressions_valid_.json")
    parser.add_argument("--mask_path", type=str, default="data/video_datas/revos/mask_dict.json")
    parser.add_argument("--foreground_mask_path", type=str, default="data/video_datas/revos/mask_dict_foreground.json")
    parser.add_argument("--save_json_name", type=str, default="revos_valid.json")
    parser.add_argument("--save_csv_name", type=str, default="revos_valid.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    queue                = mp.Queue()
    exp_dict             = json.load(open(args.exp_path))['videos']
    mask_dict            = json.load(open(args.mask_path))
    mask_dict_foreground = json.load(open(args.foreground_mask_path))
    #mask_dict_foreground=mask_dict
    mask_pred = json.load(open(args.pred_path))

    manager = mp.Manager()
    output_dict = manager.dict()
    mask_pred_dict = mask_pred

    jobs = []
    missing = []
    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            prediction = mask_pred.get(vid_name, {}).get(exp)
            expected_frames = len(vid['frames'])
            actual_frames = len(prediction.get('prediction_masks', [])) if isinstance(prediction, dict) else 0
            if actual_frames != expected_frames:
                missing.append((vid_name, exp, expected_frames, actual_frames))
            else:
                jobs.append([vid_name, exp])

    if missing:
        first = missing[0]
        raise RuntimeError(
            f"Incomplete predictions: {len(missing)} expressions missing or incomplete; "
            f"first={first[0]}/{first[1]} expected_frames={first[2]} actual_frames={first[3]}"
        )

    worker_count = max(1, min(args.workers, len(jobs)))
    for job in jobs:
        queue.put(job)
    for _ in range(worker_count):
        queue.put(None)

    start_time = time.time()
    if worker_count > 1:
        processes = []
        for rank in range(worker_count):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        failed = [(p.pid, p.exitcode) for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Evaluation workers failed: {failed}")
    else:
        eval_queue(queue, 0, output_dict)
    if len(output_dict) != len(jobs):
        raise RuntimeError(f"Evaluation incomplete: expected={len(jobs)} completed={len(output_dict)}")


    # save average results
    output_json_path = osp.join(osp.dirname(args.pred_path), args.save_json_name)
    output_csv_path  = osp.join(osp.dirname(args.pred_path), args.save_csv_name)

    data_list = []
    for videxp, (j, f, a, r) in output_dict.items():
        vid_name, exp = videxp.rsplit('_', 1)
        data = {}

        data['video_name'] = vid_name
        data['exp_id']     = exp
        data['exp']        = exp_dict[vid_name]['expressions'][exp]['exp']
        data['videxp']     = videxp
        data['J']          = round(100 * j, 2)
        data['F']          = round(100 * f, 2)
        data['JF']         = round(100 * (j + f) / 2, 2)
        data['A']          = round(100 * a, 2)
        data['R']          = round(100 * r, 2)
        data['type_id']    = exp_dict[vid_name]['expressions'][exp]['type_id']

        data_list.append(data)

    is_referring = lambda x: x['type_id'] == 0
    is_reason    = lambda x: x['type_id'] == 1
    is_nan       = lambda x: x['type_id'] == 2

    j_referring  = np.array([d['J'] for d in data_list if is_referring(d)]).mean()
    f_referring  = np.array([d['F'] for d in data_list if is_referring(d)]).mean()
    a_referring  = np.array([d['A'] for d in data_list if is_referring(d)]).mean()
    r_referring  = np.array([d['R'] for d in data_list if is_referring(d)]).mean()
    jf_referring = (j_referring + f_referring) / 2

    j_reason  = np.array([d['J'] for d in data_list if is_reason(d)]).mean()
    f_reason  = np.array([d['F'] for d in data_list if is_reason(d)]).mean()
    a_reason  = np.array([d['A'] for d in data_list if is_reason(d)]).mean()
    r_reason  = np.array([d['R'] for d in data_list if is_reason(d)]).mean()
    jf_reason = (j_reason + f_reason) / 2

    j_referring_reason  = (j_referring + j_reason) / 2
    f_referring_reason  = (f_referring + f_reason) / 2
    a_referring_reason  = (a_referring + a_reason) / 2
    r_referring_reason  = (r_referring + r_reason) / 2
    jf_referring_reason = (jf_referring + jf_reason) / 2

    results = {
        "referring": {
            "J" : j_referring,
            "F" : f_referring,
            "A" : a_referring,
            "R" : r_referring,
            "JF": jf_referring
        },
        "reason": {
            "J" : j_reason,
            "F" : f_reason,
            "A" : a_reason,
            "R" : r_reason,
            "JF": jf_reason
        },
        "overall": {
            "J" : j_referring_reason,
            "F" : f_referring_reason,
            "A" : a_referring_reason,
            "R" : r_referring_reason,
            "JF": jf_referring_reason
        }
    }

    print(results)
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json_path}")

    data4csv = {}
    for data in data_list:
        for k, v in data.items():
            data4csv[k] = data4csv.get(k, []) + [v]

    df = pd.DataFrame(data4csv)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))