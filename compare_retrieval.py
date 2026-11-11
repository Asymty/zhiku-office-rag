import argparse
import sys
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import HTTPException

import app
from evaluate_rag import run


def compare(args):
    settings = [
        ("层级分块 + BM25", "hierarchical", False, False, False),
        ("父子分块 + BM25", "parent_child", False, False, False),
    ]
    if args.full:
        settings += [
            ("父子分块 + 混合检索", "parent_child", True, False, False),
            ("父子分块 + 混合检索 + 重排", "parent_child", True, True, False),
            ("父子分块 + 混合检索 + 重排 + Critic", "parent_child", True, True, True),
        ]
    lines = [
        f"按最终送入生成模型的证据评分，K={args.top_k}。耗时包含各组启用的检索与证据整理步骤。",
        "",
        "| 配置 | 完成 / 总题数 | 调用失败 | 证据 Recall@K | NDCG@K | MRR@K | P95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    failures = []
    for name, strategy, hybrid, rerank, adaptive in settings:
        options = argparse.Namespace(**vars(args), answers=False, from_case=None, context_only=True,
                                     chunk_strategy=strategy, hybrid=hybrid, rerank=rerank, adaptive=adaptive)
        try:
            report = run(options)
        except (ValueError, OSError, HTTPException) as error:
            failures.append(f"- {name} 未完成。{getattr(error, 'detail', error)}")
            break
        summary = report["summary"]
        complete = summary["positive_cases"] + summary["negative_cases"]
        total = len(report["case_ids"])
        errors = summary["retrieval_error_count"]
        metrics = [summary[key] for key in
                   ("evidence_recall_at_k", "ndcg_at_k", "mrr_at_k", "p95_retrieval_ms")]
        formatted = [f"{value:.3f}" if value is not None else "—" for value in metrics]
        lines.append("| " + " | ".join([name, f"{complete}/{total}", str(errors), *formatted]) + " |")
        if complete < total:
            failures.append(f"- {name} 有未完成题，指标仅基于已完成题。")
        if report["retry_from"]:
            failures.append(f"- 评测因 429 暂停，首个受限用例为 `{report['retry_from']}`。")
            break
    if failures:
        lines.extend(["", *failures])
    return "\n".join(lines) + "\n", bool(failures)


def main():
    parser = argparse.ArgumentParser(description="用同一批样本对照分块与检索配置。默认只运行本地 BM25。")
    parser.add_argument("--dataset", type=Path, default=app.ROOT / "evaluation" / "baseline.json")
    parser.add_argument("--files", type=Path, default=app.ROOT / "evaluation" / "fixtures")
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=4)
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--full", action="store_true", help="追加混合检索、重排和 Critic，可能消耗 API 额度。")
    parser.add_argument("--output", type=Path, help="将对照表保存为 Markdown。")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("输出文件已存在，请换一个文件名。")
    with redirect_stdout(sys.stderr):
        report, failed = compare(args)
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(report)
        except OSError as error:
            parser.exit(2, f"无法写入评估报告：{error}\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(report, end="")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
