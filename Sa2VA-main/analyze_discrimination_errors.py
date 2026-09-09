"""
深度分析 CRF-SAM3 判别器的错误模式

基于 crf_sam3_valid_u_outputs_n2/scores.json 的结果，
分析判别器在哪些类型的表达式上失败，为TTRL训练提供insight。
"""

import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_scores(scores_path):
    """加载评分数据"""
    with open(scores_path, 'r') as f:
        return json.load(f)


def analyze_overall_performance(data):
    """分析整体性能"""
    print("="*80)
    print("整体性能分析")
    print("="*80)

    total_samples = len(data)
    gt_match = sum(1 for d in data if d['gt_match'])

    # 统计候选对象
    total_candidates = 0
    gt_candidates = 0
    selected_candidates = 0

    # 判别性能统计
    correct_yes = 0  # GT对象被正确判为Yes/Partial (prob_yes > 0.5)
    false_yes = 0     # 非GT对象被错误判为Yes (prob_yes > 0.5)
    correct_no = 0    # 非GT对象被正确判为No (prob_yes < 0.5)
    false_no = 0      # GT对象被错误判为No (prob_yes < 0.5)

    for sample in data:
        total_candidates += sample['num_candidates']
        gt_candidates += sample['num_gt_identified']
        selected_candidates += sample['num_selected']

        for cand in sample['candidates']:
            is_gt = cand['is_gt']
            prob_yes = cand['prob_yes']

            if is_gt:
                if prob_yes > 0.5:
                    correct_yes += 1
                else:
                    false_no += 1
            else:
                if prob_yes > 0.5:
                    false_yes += 1
                else:
                    correct_no += 1

    print(f"\n【数据集规模】")
    print(f"  视频-表达式对: {total_samples}")
    print(f"  GT Match率: {gt_match}/{total_samples} = {gt_match/total_samples*100:.2f}%")
    print(f"  总候选对象数: {total_candidates}")
    print(f"  总GT对象数: {gt_candidates}")
    print(f"  被选中的对象数: {selected_candidates}")

    print(f"\n【判别器性能】")
    print(f"  真阳性 (GT → prob_yes>0.5): {correct_yes}/{gt_candidates} = {correct_yes/gt_candidates*100:.2f}%")
    print(f"  假阴性 (GT → prob_yes<0.5): {false_no}/{gt_candidates} = {false_no/gt_candidates*100:.2f}%")
    print(f"  真阴性 (非GT → prob_yes<0.5): {correct_no}/{total_candidates-gt_candidates} = {correct_no/(total_candidates-gt_candidates)*100:.2f}%")
    print(f"  假阳性 (非GT → prob_yes>0.5): {false_yes}/{total_candidates-gt_candidates} = {false_yes/(total_candidates-gt_candidates)*100:.2f}%")
    print(f"  整体准确率: {(correct_yes+correct_no)/total_candidates*100:.2f}%")

    return {
        'total_samples': total_samples,
        'gt_match_rate': gt_match/total_samples,
        'correct_yes': correct_yes,
        'false_no': false_no,
        'correct_no': correct_no,
        'false_yes': false_yes,
        'accuracy': (correct_yes+correct_no)/total_candidates
    }


def analyze_false_negatives(data):
    """深度分析假阴性：GT对象被错误拒绝"""
    print("\n" + "="*80)
    print("假阴性分析 (False Negatives: GT对象被错误判为No)")
    print("="*80)

    false_negatives = []

    for sample in data:
        for cand in sample['candidates']:
            if cand['is_gt'] and cand['prob_yes'] < 0.5:
                false_negatives.append({
                    'video_id': sample['video_id'],
                    'exp_id': sample['exp_id'],
                    'obj_id': cand['obj_id'],
                    'expression': sample['expression'],
                    'category': sample['category'],
                    'prob_yes': cand['prob_yes'],
                    'prob_partial': cand['prob_partial'],
                    'prob_no': cand['prob_no'],
                    'score': cand['score'],
                    'iou': cand['iou'],
                    'prediction': cand['prediction']
                })

    print(f"\n【假阴性统计】")
    print(f"  总数: {len(false_negatives)}")
    print(f"  平均 prob_yes: {np.mean([fn['prob_yes'] for fn in false_negatives]):.4f}")
    print(f"  平均 prob_no: {np.mean([fn['prob_no'] for fn in false_negatives]):.4f}")
    print(f"  平均 IoU: {np.mean([fn['iou'] for fn in false_negatives]):.4f}")

    # 按表达式类型分析
    print(f"\n【表达式特征分析】")

    # 检测表达式中的关键词
    motion_keywords = ['moving', 'walking', 'running', 'jumping', 'eating', 'playing',
                       'turning', 'facing', 'directing', 'enjoying', 'sitting']
    location_keywords = ['left', 'right', 'top', 'bottom', 'corner', 'front', 'back',
                        'distance', 'center', 'side']
    color_keywords = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'orange',
                     'brown', 'gray', 'dark', 'light', 'bright']

    motion_fn = [fn for fn in false_negatives if any(kw in fn['expression'].lower() for kw in motion_keywords)]
    location_fn = [fn for fn in false_negatives if any(kw in fn['expression'].lower() for kw in location_keywords)]
    color_fn = [fn for fn in false_negatives if any(kw in fn['expression'].lower() for kw in color_keywords)]

    # 组合条件（多个条件同时出现）
    combo_fn = []
    for fn in false_negatives:
        expr_lower = fn['expression'].lower()
        num_conditions = sum([
            any(kw in expr_lower for kw in motion_keywords),
            any(kw in expr_lower for kw in location_keywords),
            any(kw in expr_lower for kw in color_keywords)
        ])
        if num_conditions >= 2:
            combo_fn.append(fn)

    print(f"  包含动作描述: {len(motion_fn)} ({len(motion_fn)/len(false_negatives)*100:.1f}%)")
    print(f"    平均 prob_yes: {np.mean([fn['prob_yes'] for fn in motion_fn]) if motion_fn else 0:.4f}")
    print(f"  包含位置描述: {len(location_fn)} ({len(location_fn)/len(false_negatives)*100:.1f}%)")
    print(f"    平均 prob_yes: {np.mean([fn['prob_yes'] for fn in location_fn]) if location_fn else 0:.4f}")
    print(f"  包含颜色描述: {len(color_fn)} ({len(color_fn)/len(false_negatives)*100:.1f}%)")
    print(f"    平均 prob_yes: {np.mean([fn['prob_yes'] for fn in color_fn]) if color_fn else 0:.4f}")
    print(f"  组合条件 (≥2类): {len(combo_fn)} ({len(combo_fn)/len(false_negatives)*100:.1f}%)")
    print(f"    平均 prob_yes: {np.mean([fn['prob_yes'] for fn in combo_fn]) if combo_fn else 0:.4f}")

    # 最严重的假阴性案例
    print(f"\n【最严重的假阴性案例】（高IoU但被强烈拒绝）")
    severe_fn = sorted(false_negatives, key=lambda x: x['iou'] - x['prob_yes'], reverse=True)[:10]
    for i, fn in enumerate(severe_fn, 1):
        print(f"\n  {i}. {fn['video_id']}/{fn['exp_id']}/obj_{fn['obj_id']}")
        print(f"     表达式: \"{fn['expression']}\"")
        print(f"     prob_yes={fn['prob_yes']:.4f}, prob_no={fn['prob_no']:.4f}, IoU={fn['iou']:.4f}")
        print(f"     差距: IoU - prob_yes = {fn['iou'] - fn['prob_yes']:.4f}")

    return false_negatives


def analyze_false_positives(data):
    """深度分析假阳性：非GT对象被错误接受"""
    print("\n" + "="*80)
    print("假阳性分析 (False Positives: 非GT对象被错误判为Yes)")
    print("="*80)

    false_positives = []

    for sample in data:
        for cand in sample['candidates']:
            if not cand['is_gt'] and cand['prob_yes'] > 0.5:
                false_positives.append({
                    'video_id': sample['video_id'],
                    'exp_id': sample['exp_id'],
                    'obj_id': cand['obj_id'],
                    'expression': sample['expression'],
                    'category': sample['category'],
                    'prob_yes': cand['prob_yes'],
                    'prob_partial': cand['prob_partial'],
                    'prob_no': cand['prob_no'],
                    'score': cand['score'],
                    'iou': cand['iou'],
                    'prediction': cand['prediction']
                })

    print(f"\n【假阳性统计】")
    print(f"  总数: {len(false_positives)}")
    print(f"  平均 prob_yes: {np.mean([fp['prob_yes'] for fp in false_positives]):.4f}")
    print(f"  平均 prob_no: {np.mean([fp['prob_no'] for fp in false_positives]):.4f}")
    print(f"  平均 IoU: {np.mean([fp['iou'] for fp in false_positives]):.4f}")

    # 最严重的假阳性案例
    print(f"\n【最严重的假阳性案例】（高prob_yes但IoU=0）")
    severe_fp = sorted(false_positives, key=lambda x: x['prob_yes'], reverse=True)[:10]
    for i, fp in enumerate(severe_fp, 1):
        print(f"\n  {i}. {fp['video_id']}/{fp['exp_id']}/obj_{fp['obj_id']}")
        print(f"     表达式: \"{fp['expression']}\"")
        print(f"     prob_yes={fp['prob_yes']:.4f}, prob_no={fp['prob_no']:.4f}, IoU={fp['iou']:.4f}")

    return false_positives


def analyze_hard_samples(data):
    """分析困难样本：所有候选都被判为No"""
    print("\n" + "="*80)
    print("困难样本分析 (Hard Samples: 所有候选都被判为No)")
    print("="*80)

    hard_samples = []

    for sample in data:
        # 检查是否所有候选的prob_yes都很低
        max_prob_yes = max([cand['prob_yes'] for cand in sample['candidates']])

        if max_prob_yes < 0.2:  # 所有候选的prob_yes都小于0.2
            hard_samples.append({
                'video_id': sample['video_id'],
                'exp_id': sample['exp_id'],
                'expression': sample['expression'],
                'category': sample['category'],
                'num_candidates': sample['num_candidates'],
                'max_prob_yes': max_prob_yes,
                'avg_prob_yes': np.mean([cand['prob_yes'] for cand in sample['candidates']]),
                'candidates': sample['candidates'],
                'num_gt_identified': sample['num_gt_identified'],
                'gt_match': sample['gt_match']
            })

    print(f"\n【困难样本统计】")
    print(f"  总数: {len(hard_samples)}")
    print(f"  占比: {len(hard_samples)/len(data)*100:.2f}%")
    print(f"  平均候选数: {np.mean([hs['num_candidates'] for hs in hard_samples]):.2f}")
    print(f"  平均最高prob_yes: {np.mean([hs['max_prob_yes'] for hs in hard_samples]):.4f}")

    # 分析困难样本的特征
    print(f"\n【困难样本特征】")
    motion_keywords = ['moving', 'walking', 'running', 'jumping', 'eating', 'playing']
    location_keywords = ['left', 'right', 'top', 'bottom', 'corner', 'front', 'back']

    motion_hard = [hs for hs in hard_samples if any(kw in hs['expression'].lower() for kw in motion_keywords)]
    location_hard = [hs for hs in hard_samples if any(kw in hs['expression'].lower() for kw in location_keywords)]

    # 组合条件
    combo_hard = []
    for hs in hard_samples:
        expr_lower = hs['expression'].lower()
        num_conditions = sum([
            any(kw in expr_lower for kw in motion_keywords),
            any(kw in expr_lower for kw in location_keywords)
        ])
        if num_conditions >= 2:
            combo_hard.append(hs)

    print(f"  包含动作描述: {len(motion_hard)} ({len(motion_hard)/len(hard_samples)*100:.1f}%)")
    print(f"  包含位置描述: {len(location_hard)} ({len(location_hard)/len(hard_samples)*100:.1f}%)")
    print(f"  组合条件 (≥2类): {len(combo_hard)} ({len(combo_hard)/len(hard_samples)*100:.1f}%)")

    # 典型困难样本
    print(f"\n【典型困难样本】")
    for i, hs in enumerate(hard_samples[:10], 1):
        print(f"\n  {i}. {hs['video_id']}/{hs['exp_id']}")
        print(f"     表达式: \"{hs['expression']}\"")
        print(f"     类别: {hs['category']}, 候选数: {hs['num_candidates']}")
        print(f"     最高prob_yes: {hs['max_prob_yes']:.4f}")
        print(f"     GT识别数: {hs['num_gt_identified']}, GT Match: {hs['gt_match']}")

    return hard_samples


def analyze_for_ttrl(data):
    """为TTRL训练生成分析报告"""
    print("\n" + "="*80)
    print("TTRL 训练价值分析")
    print("="*80)

    # 统计需要TTRL改进的样本
    total_candidates = sum(len(s['candidates']) for s in data)

    # 需要提升prob_yes的GT对象
    needs_increase = []
    # 需要降低prob_yes的非GT对象
    needs_decrease = []

    for sample in data:
        for cand in sample['candidates']:
            if cand['is_gt'] and cand['prob_yes'] < 0.5:
                needs_increase.append({
                    'video_id': sample['video_id'],
                    'exp_id': sample['exp_id'],
                    'obj_id': cand['obj_id'],
                    'expression': sample['expression'],
                    'prob_yes': cand['prob_yes'],
                    'iou': cand['iou'],
                    'gap': 1.0 - cand['prob_yes']  # 需要提升的幅度
                })
            elif not cand['is_gt'] and cand['prob_yes'] > 0.5:
                needs_decrease.append({
                    'video_id': sample['video_id'],
                    'exp_id': sample['exp_id'],
                    'obj_id': cand['obj_id'],
                    'expression': sample['expression'],
                    'prob_yes': cand['prob_yes'],
                    'iou': cand['iou'],
                    'gap': cand['prob_yes'] - 0.0  # 需要降低的幅度
                })

    print(f"\n【TTRL 改进目标】")
    print(f"  需要提升prob_yes的GT对象: {len(needs_increase)} ({len(needs_increase)/total_candidates*100:.2f}%)")
    print(f"    平均当前prob_yes: {np.mean([ni['prob_yes'] for ni in needs_increase]):.4f}")
    print(f"    平均需要提升: {np.mean([ni['gap'] for ni in needs_increase]):.4f}")
    print(f"    平均IoU: {np.mean([ni['iou'] for ni in needs_increase]):.4f}")

    print(f"\n  需要降低prob_yes的非GT对象: {len(needs_decrease)} ({len(needs_decrease)/total_candidates*100:.2f}%)")
    print(f"    平均当前prob_yes: {np.mean([nd['prob_yes'] for nd in needs_decrease]):.4f}")
    print(f"    平均需要降低: {np.mean([nd['gap'] for nd in needs_decrease]):.4f}")
    print(f"    平均IoU: {np.mean([nd['iou'] for nd in needs_decrease]):.4f}")

    print(f"\n【TTRL 预期收益】")
    print(f"  如果TTRL成功修正所有错误判断:")
    current_acc = (total_candidates - len(needs_increase) - len(needs_decrease)) / total_candidates
    potential_acc = 1.0
    improvement = (potential_acc - current_acc) / current_acc * 100
    print(f"    当前准确率: {current_acc*100:.2f}%")
    print(f"    理论上限: {potential_acc*100:.2f}%")
    print(f"    潜在提升: +{improvement:.1f}%")

    print(f"\n  基于TTRL论文在AIME任务上的表现（+211%）:")
    print(f"    保守估计（50%改善率）: {current_acc*100:.2f}% → {(current_acc + (potential_acc - current_acc) * 0.5)*100:.2f}%")
    print(f"    乐观估计（75%改善率）: {current_acc*100:.2f}% → {(current_acc + (potential_acc - current_acc) * 0.75)*100:.2f}%")

    return {
        'needs_increase': needs_increase,
        'needs_decrease': needs_decrease,
        'current_acc': current_acc,
        'potential_acc': potential_acc
    }


def generate_summary_report(data, output_path="discrimination_analysis_report.txt"):
    """生成完整的分析报告"""
    import sys
    from io import StringIO

    # 捕获所有print输出
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    # 运行所有分析
    overall_stats = analyze_overall_performance(data)
    false_negatives = analyze_false_negatives(data)
    false_positives = analyze_false_positives(data)
    hard_samples = analyze_hard_samples(data)
    ttrl_analysis = analyze_for_ttrl(data)

    # 恢复stdout
    sys.stdout = old_stdout

    # 保存报告
    report_content = buffer.getvalue()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    # 同时打印到控制台
    print(report_content)
    print(f"\n报告已保存到: {output_path}")

    return {
        'overall_stats': overall_stats,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'hard_samples': hard_samples,
        'ttrl_analysis': ttrl_analysis
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分析CRF-SAM3判别器的错误模式")
    parser.add_argument(
        "--scores_path",
        type=str,
        default="crf_sam3_valid_u_outputs_n2/scores.json",
        help="Path to scores.json"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="discrimination_analysis_report.txt",
        help="Output report path"
    )

    args = parser.parse_args()

    # 加载数据
    print(f"加载数据: {args.scores_path}")
    data = load_scores(args.scores_path)

    # 生成完整报告
    analysis_results = generate_summary_report(data, args.output_report)

    print("\n分析完成！")
