import argparse
import asyncio
import json
import math
import os
import sys
import time
from contextlib import AsyncExitStack, redirect_stdout
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODELSCOPE_URL = "https://api-inference.modelscope.cn/v1"
METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevance")


class Sample(BaseModel):
    id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    retrieved_contexts: list[str]
    response: str = Field(min_length=1)
    verification: str | None = None


class RequestState:
    def __init__(self):
        self.limited = False
        self.status = None
        self.last_request = None
        self.lock = asyncio.Lock()

    async def before_request(self, request):
        async with self.lock:
            if self.limited:
                raise RuntimeError("Evaluation stopped after HTTP 429")
            if request.url.host != "api-inference.modelscope.cn":
                return
            if self.last_request is not None:
                await asyncio.sleep(max(0, self.last_request + 3 - time.monotonic()))
            if self.limited:
                raise RuntimeError("Evaluation stopped after HTTP 429")
            self.last_request = time.monotonic()

    async def after_response(self, response):
        if response.status_code >= 400:
            self.status = response.status_code
        if response.status_code == 429:
            self.limited = True


async def score(samples, metrics, state):
    results = []
    for sample in samples:
        scores = dict.fromkeys(METRICS)
        errors = {}
        results.append({"id": sample.id, "scores": scores, "errors": errors})
        fields = sample.model_dump()
        for key, metric in metrics.items():
            if state.limited:
                errors[key] = "rate_limit"
                continue
            if metric is None:
                errors[key] = "embedding_not_configured"
                continue
            if key in {"faithfulness", "answer_relevance"} and sample.verification in {
                "no_evidence", "unsupported", "invalid_citation"
            }:
                errors[key] = "answer_withheld"
                continue
            if key != "answer_relevance" and not sample.retrieved_contexts:
                errors[key] = "no_context"
                continue
            arguments = {"user_input": fields["user_input"]}
            if key != "answer_relevance":
                arguments["retrieved_contexts"] = fields["retrieved_contexts"]
            if key in {"context_precision", "context_recall"}:
                arguments["reference"] = fields["reference"]
            else:
                arguments["response"] = fields["response"]
            state.status = None
            try:
                value = float((await metric.ascore(**arguments)).value)
                if math.isfinite(value):
                    scores[key] = value
                else:
                    errors[key] = "non_finite_score"
            except Exception as error:
                errors[key] = (f"http_{state.status}" if state.status else type(error).__name__)
    summary = {}
    for key in METRICS:
        values = [row["scores"][key] for row in results if row["scores"][key] is not None]
        summary[key] = {"mean": mean(values) if values else None, "scored": len(values)}
    return {"summary": summary, "cases": results, "rate_limited": state.limited}


async def run(samples):
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    try:
        import httpx
        import instructor
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms.base import InstructorLLM
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except ImportError as error:
        raise ValueError("请先安装 requirements-evaluation.txt。") from error

    token = os.getenv("MODELSCOPE_API_KEY", "").strip()
    ollama_url = os.getenv("INKATLAS_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + "/v1"
    judge_url = os.getenv("INKATLAS_RAGAS_JUDGE_URL", MODELSCOPE_URL if token else ollama_url).rstrip("/")
    judge_model = os.getenv("INKATLAS_RAGAS_JUDGE_MODEL", os.getenv(
        "INKATLAS_CHAT_MODEL", "Qwen/Qwen3.5-35B-A3B" if token else "qwen2.5:3b"
    ))
    judge_key = os.getenv("INKATLAS_RAGAS_JUDGE_KEY", token if judge_url == MODELSCOPE_URL else "local")
    if judge_url == MODELSCOPE_URL and (not judge_key or ":" in judge_model):
        raise ValueError("魔搭评估需配置令牌，并使用无提供方后缀的模型 ID。")
    embed_model = os.getenv("INKATLAS_RAGAS_EMBED_MODEL", os.getenv("INKATLAS_EMBED_MODEL", ""))
    embed_url = (os.getenv("INKATLAS_RAGAS_EMBED_URL") or os.getenv("INKATLAS_EMBED_API_URL") or ollama_url).rstrip("/")
    embed_key = os.getenv("INKATLAS_RAGAS_EMBED_KEY", os.getenv(
        "INKATLAS_EMBED_API_KEY", token if embed_url == MODELSCOPE_URL else "local"
    ))
    if embed_model not in {"", "none"} and embed_url == MODELSCOPE_URL and (not embed_key or ":" in embed_model):
        raise ValueError("魔搭嵌入需配置令牌，并使用无提供方后缀的模型 ID。")
    state = RequestState()
    hooks = {"request": [state.before_request], "response": [state.after_response]}

    async with AsyncExitStack() as clients:
        async def connect(url, key):
            return await clients.enter_async_context(AsyncOpenAI(
                base_url=url, api_key=key, max_retries=0, timeout=60,
                http_client=httpx.AsyncClient(timeout=60, event_hooks=hooks),
            ))

        judge = await connect(judge_url, judge_key)
        kwargs = {"max_retries": 1}
        if judge_url == MODELSCOPE_URL and judge_model.startswith("Qwen/Qwen3.5-"):
            kwargs["extra_body"] = {"enable_thinking": False}
        llm = InstructorLLM(client=instructor.from_openai(judge, mode=instructor.Mode.MD_JSON),
                            model=judge_model, provider="openai", **kwargs)
        relevance = None
        if embed_model not in {"", "none"}:
            embeddings = embedding_factory("openai", model=embed_model, client=await connect(embed_url, embed_key))
            relevance = AnswerRelevancy(llm=llm, embeddings=embeddings)
        report = await score(samples, {
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevance": relevance,
        }, state)
    return report | {"judge_model": judge_model, "embedding_model": embed_model or None}


def main():
    parser = argparse.ArgumentParser(description="对导出的问答样本批量运行 RAGAS。")
    parser.add_argument("input", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="仅检查输入格式，不调用评估模型。")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("输出文件已存在，请换一个文件名。")
    try:
        raw = json.loads(args.input.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("RAGAS 输入必须是非空 JSON 数组。")
        samples = [Sample.model_validate(item) for item in raw]
        if len({sample.id for sample in samples}) != len(samples):
            raise ValueError("RAGAS 样本 ID 不可重复。")
        report = {"valid": len(samples)}
        if not args.validate_only:
            load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")
            with redirect_stdout(sys.stderr):
                report = asyncio.run(run(samples))
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(rendered + "\n")
        sys.stdout.reconfigure(encoding="utf-8")
        print(rendered)
    except (OSError, ValueError) as error:
        parser.exit(2, f"RAGAS 评估失败：{error}\n")


if __name__ == "__main__":
    main()
