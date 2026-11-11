import argparse
import json
import math
import re
import sys
import time
import unicodedata
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from fastapi import HTTPException
from pydantic import BaseModel, Field

import app


class Evidence(BaseModel):
    file: str
    location: str | None = None
    contains: list[str] = Field(min_length=1)


class Case(BaseModel):
    id: str
    question: str
    evidence: list[Evidence]
    requires_ocr: bool = False
    answer_contains: list[list[str]] = Field(default_factory=list)
    reference: str | None = None


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def contains_all(text, terms):
    normalized_text = normalized(text)
    for term in terms:
        value = normalized(term)
        if not value:
            return False
        pattern = re.escape(value)
        if value[0].isdigit():
            pattern = r"(?<!\d)(?<!\d[.,])" + pattern
        if value[-1].isdigit():
            pattern += r"(?!\d|[.,]\d)"
        if not re.search(pattern, normalized_text):
            return False
    return True


def matches(source, label):
    return (source["name"] == label.file
            and (label.location is None or source["location"] == label.location)
            and contains_all(source["text"], label.contains))


def score_retrieval(sources, labels, top_k):
    if not labels:
        return {"empty_retrieval": not sources}
    relevance = [any(matches(source, label) for label in labels) for source in sources]
    matched_labels = sum(any(matches(source, label) for source in sources) for label in labels)
    rank = next((rank for rank, relevant in enumerate(relevance, 1) if relevant), None)
    covered = set()
    dcg = 0.0
    for position, source in enumerate(sources, 1):
        label = next((index for index, item in enumerate(labels) if index not in covered and matches(source, item)), None)
        if label is not None:
            covered.add(label)
            dcg += 1 / math.log2(position + 1)
    ideal_dcg = sum(1 / math.log2(position + 1) for position in range(1, min(top_k, len(labels)) + 1))
    return {"first_hit_rank": rank, "hit": rank is not None,
            "precision_at_k": sum(relevance) / top_k, "evidence_recall_at_k": matched_labels / len(labels),
            "reciprocal_rank": 1 / rank if rank else 0.0, "ndcg_at_k": dcg / ideal_dcg}


def evaluate_case(case, top_k, answers, diagnose=False, export_ragas=False, *,
                  rerank=False, adaptive=False, context_only=False):
    request = app.RetrievalQuery(text=case.question, top_k=top_k, rerank=rerank, adaptive=adaptive)
    prepared = adaptive or context_only
    started = time.perf_counter()
    try:
        if prepared:
            settings, query, context, sources = app.prepare_answer(request)
        else:
            sources = app.retrieve(request)
    except HTTPException as error:
        result = {"id": case.id, "question": case.question, "retrieval_error_status": error.status_code}
        if answers:
            result.update(answer_pass=None, answer_error_status=error.status_code)
        return result
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    result = {
        "id": case.id, "question": case.question, "retrieval_ms": elapsed_ms,
        "retrieved": [{key: source[key] for key in ("name", "location", "chunk")} for source in sources],
        **score_retrieval(sources, case.evidence, top_k),
    }
    if answers:
        if prepared:
            packed = sources
        else:
            query = case.question
            context, packed = app.pack_context(query, sources, request.context_tokens)
        trace = {"generation_evidence": context} if diagnose else None
        if trace is not None:
            result["diagnostics"] = trace
        try:
            if not prepared:
                settings = app.prompt_settings()
            answer = app.generate_answer(settings, query, context, packed, diagnostics=trace)
        except HTTPException as error:
            result.update(answer_pass=None, answer_error_status=error.status_code)
            retry = (error.headers or {}).get("Retry-After")
            if error.status_code == 429 and retry and retry.isdecimal():
                result["retry_after_seconds"] = int(retry)
            return result
        result["verification"] = answer["verification"]
        result["answer"] = answer["answer"]
        if export_ragas and case.reference:
            result["ragas_input"] = {
                "id": case.id, "user_input": case.question, "reference": case.reference,
                "retrieved_contexts": [item["text"] for item in context],
                "response": answer["answer"], "verification": answer["verification"],
            }
        result["cited"] = [{key: source[key] for key in ("name", "location", "chunk")} for source in answer["sources"]]
        if not case.evidence:
            result["answer_pass"] = answer["verification"] == "no_evidence" and not answer["sources"]
        else:
            cited_all = all(any(matches(source, label) for source in answer["sources"]) for label in case.evidence)
            terms_match = any(group and contains_all(answer["answer"], group) for group in case.answer_contains)
            result.update(answer_pass=cited_all and terms_match and answer["verification"] == "supported",
                          expected_evidence_cited=cited_all, answer_terms_match=terms_match)
    return result


def run(args):
    cases = [Case.model_validate(item) for item in json.loads(args.dataset.read_text(encoding="utf-8-sig"))]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("评估集必须包含用例，且用例 ID 不可重复。")
    skipped = [case.id for case in cases if args.skip_ocr and case.requires_ocr]
    cases = [case for case in cases if case.id not in skipped]
    case_ids = [case.id for case in cases]
    if not any(case.evidence for case in cases):
        raise ValueError("评估集至少需要一条有标注证据的题目。")
    if args.answers and any(case.evidence and not case.answer_contains for case in cases):
        raise ValueError("回答测试需要给所有正例标注 answer_contains。")
    export_ragas = bool(getattr(args, "export_ragas", None))
    chunk_strategy = getattr(args, "chunk_strategy", "hierarchical")
    rerank = getattr(args, "rerank", False)
    adaptive = getattr(args, "adaptive", False)
    if export_ragas and any(case.evidence and not (case.reference or "").strip() for case in cases):
        raise ValueError("导出 RAGAS 输入需要给正例标注 reference。")
    if args.hybrid and app.EMBED_MODEL == "none":
        raise ValueError("混合评估需要先配置嵌入模型。")
    files = sorted({label.file for case in cases for label in case.evidence})
    if args.from_case:
        start = next((index for index, case in enumerate(cases) if case.id == args.from_case), None)
        if start is None:
            raise ValueError("续跑题目不存在，或已被当前选项跳过。")
        cases = cases[start:]
    only_case = getattr(args, "case", None)
    if only_case:
        cases = [case for case in cases if case.id == only_case]
        if not cases:
            raise ValueError("指定题目不存在，或已被当前选项跳过。")
    completed = set(getattr(args, "completed_case", []))
    if not completed.issubset(case_ids):
        raise ValueError("已完成题目不在当前评估集内。")
    cases = [case for case in cases if case.id not in completed]
    data_root = args.files.resolve()
    previous = app.DATABASE, app.EMBED_MODEL, app.INDEX_MODEL, app.MODEL_REQUEST_INTERVAL
    if (args.answers or adaptive or args.hybrid) and app.MODELSCOPE_TOKEN:
        app.MODEL_REQUEST_INTERVAL = max(app.MODEL_REQUEST_INTERVAL, 3.0)
    if not args.hybrid:
        app.EMBED_MODEL = app.INDEX_MODEL = "none"
    retry_from = None
    try:
        with TemporaryDirectory() as directory:
            app.DATABASE = Path(directory) / "eval.sqlite3"
            for name in files:
                path = (data_root / name).resolve()
                if not path.is_relative_to(data_root) or path.name != name:
                    raise ValueError(f"评估文件必须位于样本目录下：{name}")
                if path.stat().st_size > app.MAX_UPLOAD:
                    raise ValueError(f"评估文件超过 20 MB：{name}")
                try:
                    app.index_file(name, path.read_bytes(), app.DocumentOptions(strategy=chunk_strategy), None, None)
                except HTTPException as error:
                    raise ValueError(f"{name} 导入失败：{error.detail}") from error
            results = []
            for index, case in enumerate(cases):
                result = evaluate_case(case, args.top_k, args.answers, getattr(args, "diagnose", False), export_ragas,
                                       rerank=rerank, adaptive=adaptive,
                                       context_only=getattr(args, "context_only", False))
                results.append(result)
                if 429 in (result.get("answer_error_status"), result.get("retrieval_error_status")):
                    retry_from = case.id
                    skipped.extend(item.id for item in cases[index + 1:])
                    break
    finally:
        app.DATABASE, app.EMBED_MODEL, app.INDEX_MODEL, app.MODEL_REQUEST_INTERVAL = previous
    positives = [item for item in results if "hit" in item]
    negatives = [item for item in results if "empty_retrieval" in item]
    durations = sorted(item["retrieval_ms"] for item in results if "retrieval_ms" in item)
    summary = {
        "positive_cases": len(positives), "negative_cases": len(negatives),
        "retrieval_error_count": len(results) - len(durations),
        "hit_rate_at_k": mean(item["hit"] for item in positives) if positives else None,
        "precision_at_k": mean(item["precision_at_k"] for item in positives) if positives else None,
        "evidence_recall_at_k": mean(item["evidence_recall_at_k"] for item in positives) if positives else None,
        "mrr_at_k": mean(item["reciprocal_rank"] for item in positives) if positives else None,
        "ndcg_at_k": mean(item["ndcg_at_k"] for item in positives) if positives else None,
        "negative_empty_count": sum(item["empty_retrieval"] for item in negatives),
        "mean_retrieval_ms": round(mean(durations), 2) if durations else None,
        "p95_retrieval_ms": durations[math.ceil(.95 * len(durations)) - 1] if durations else None,
    }
    if args.answers:
        summary["answer_pass_count"] = sum(item["answer_pass"] is True for item in results)
        summary["answer_scored_count"] = sum(item["answer_pass"] is not None for item in results)
        summary["answer_error_count"] = sum(item["answer_pass"] is None for item in results)
    return {"retrieval": "hybrid" if args.hybrid else "bm25", "top_k": args.top_k, "chunk_strategy": chunk_strategy,
            "answer_checks": args.answers, "rerank": rerank, "adaptive": adaptive,
            "case_ids": case_ids, "skipped": skipped, "retry_from": retry_from,
            "summary": summary, "cases": results}


def main():
    parser = argparse.ArgumentParser(description="在临时知识库运行离线评估。默认只使用本地 BM25。")
    parser.add_argument("--dataset", type=Path, default=app.ROOT / "evaluation" / "baseline.json")
    parser.add_argument("--files", type=Path, default=app.ROOT / "evaluation" / "fixtures")
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=4)
    parser.add_argument("--chunk-strategy", choices=("hierarchical", "parent_child"), default="hierarchical")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--hybrid", action="store_true", help="使用项目配置的嵌入服务，可能消耗 API 额度。")
    parser.add_argument("--rerank", action="store_true", help="使用已配置的重排服务。")
    parser.add_argument("--adaptive", action="store_true", help="使用 Critic 和多轮检索，可能消耗 API 额度。")
    parser.add_argument("--answers", action="store_true", help="调用项目配置的回答和校验模型，可能消耗 API 额度。")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--from-case", help="从指定用例 ID 开始评估，知识库仍使用完整样本。")
    selection.add_argument("--case", help="只评估指定用例，知识库仍使用完整样本。")
    parser.add_argument("--diagnose", action="store_true", help="输出单题的模型初稿与校验证据，仅供本地诊断。")
    parser.add_argument("--completed-case", action="append", default=[], help="跳过已评分的题目。可多次传入。")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-ragas", type=Path, help="导出带最终答案及实际生成上下文的 RAGAS JSON 输入。")
    args = parser.parse_args()
    if args.diagnose and not (args.answers and args.case):
        parser.error("--diagnose 需要同时指定 --answers 和 --case。")
    if args.export_ragas and not args.answers:
        parser.error("--export-ragas 需要 --answers。")
    if any(path and path.exists() for path in (args.output, args.export_ragas)):
        parser.error("输出文件已存在，请换一个文件名。")
    if args.output and args.export_ragas and args.output.resolve() == args.export_ragas.resolve():
        parser.error("评估报告和 RAGAS 输入必须使用不同文件名。")
    try:
        with redirect_stdout(sys.stderr):
            report = run(args)
        if args.export_ragas:
            rows = [case.pop("ragas_input") for case in report["cases"] if "ragas_input" in case]
            if not rows:
                raise ValueError("没有完成的带参考答案用例可以导出。")
            with args.export_ragas.open("x", encoding="utf-8") as file:
                json.dump(rows, file, ensure_ascii=False, indent=2)
                file.write("\n")
            report["ragas_exported"] = len(rows)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(rendered + "\n")
        sys.stdout.reconfigure(encoding="utf-8")
        print(rendered)
    except (OSError, ValueError, HTTPException) as error:
        parser.exit(2, f"评估失败：{getattr(error, 'detail', error)}\n")


if __name__ == "__main__":
    main()
