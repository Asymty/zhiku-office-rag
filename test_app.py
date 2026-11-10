import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app


class DocumentLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "test.sqlite3"
        self.enterContext(patch.dict(os.environ, {"INKATLAS_ACCOUNTS_FILE": ""}))
        self.enterContext(patch.object(app, "DATABASE", self.path))
        self.enterContext(patch.multiple(app, MODELSCOPE_TOKEN="", EMBED_API="", EMBED_MODEL="bge-m3",
                                         INDEX_MODEL="bge-m3", CHAT_MODEL="qwen2.5:3b"))
        self.model = self.enterContext(patch.object(app, "ollama", side_effect=self.fake_ollama))
        self.client = TestClient(app.app)
        self.addCleanup(self.client.close)

    @staticmethod
    def fake_ollama(route, payload):
        if route == "/api/embed":
            return {"embeddings": [[1, 0] for _ in payload["input"]]}
        if "format" in payload:
            properties = payload["format"]["properties"]
            content = {"supported": True} if "supported" in properties else {"queries": ["独立检索问题"]}
            return {"message": {"content": json.dumps(content)}, "done_reason": "stop"}
        return {"message": {"content": "以文档为准。[S1]"}, "done_reason": "stop"}

    def test_update_and_offline_change_retrieval_and_links(self):
        original = {"name": "政策.txt", "text": "旧政策"}
        created = self.client.post("/documents", json=original)
        self.assertEqual(created.status_code, 200)
        document = created.json()
        path = "/documents/" + document["id"]
        self.assertTrue(self.client.post("/documents", json=original).json()["already_indexed"])
        source = self.client.post("/questions", json={"text": "政策"}).json()["sources"][0]
        self.assertEqual(self.client.get(source["url"]).text, "旧政策")

        replacement = {"name": original["name"], "text": "新政策"}
        with patch.object(app, "embeddings", side_effect=app.HTTPException(502)):
            self.assertEqual(self.client.put(path, json=replacement).status_code, 502)
        self.assertEqual(self.client.get(source["url"]).text, "旧政策")
        self.assertEqual(self.client.put(path, json=replacement).json()["version"], 2)
        self.assertEqual(self.client.get(source["url"]).status_code, 404)

        self.model.reset_mock()
        sources = self.client.post("/retrieve", json={"text": "政策", "top_k": 1}).json()["sources"]
        self.model.assert_not_called()
        self.assertEqual([item["text"] for item in sources], ["新政策"])
        self.assertEqual(self.client.get(sources[0]["url"]).text, "新政策")
        self.assertEqual(self.client.delete(path).status_code, 204)
        self.assertFalse(self.client.get("/documents").json()[0]["active"])
        self.assertEqual(self.client.get(sources[0]["url"]).status_code, 404)
        self.assertEqual(self.client.post("/retrieve", json={"text": "政策"}).json()["sources"], [])

    def test_existing_index_is_migrated_and_stays_available(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE chunks (
                    document TEXT, position INTEGER, model TEXT, name TEXT, text TEXT, vector TEXT,
                    PRIMARY KEY (document, position, model)
                )
            """)
            connection.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                ("legacy", 0, app.EMBED_MODEL, "旧文档.txt", "原有片段", "[1, 0]"),
            )
        connection.close()
        self.assertEqual(self.client.get("/documents").json()[0]["version"], 1)
        self.assertEqual(self.client.get("/documents/legacy/versions/1/chunks/1").text, "原有片段")

    def test_hybrid_and_metadata_filter_before_rerank(self):
        documents = [
            {"name": "目标.txt", "text": "报销 单据 必须盖章", "department": "财务"},
            {"name": "干扰.txt", "text": "年度预算", "department": "财务"},
            {"name": "权限.txt", "text": "报销", "department": "财务", "access_tags": ["manager"]},
            {"name": "另一个库.txt", "text": "报销", "department": "财务", "namespace": "other"},
        ]
        for document in documents:
            self.assertEqual(self.client.post("/documents", json=document).status_code, 200)
        query = {"text": "报销", "department": "财务", "top_k": 2}
        sources = self.client.post("/retrieve", json=query).json()["sources"]
        self.assertEqual([item["name"] for item in sources], ["目标.txt", "干扰.txt"])
        with patch.dict(os.environ, {"INKATLAS_RERANK_URL": "http://example.test"}):
            with patch.object(app, "model_json", return_value=[{"index": 1, "score": .9}, {"index": 0, "score": .2}]) as model:
                sources = self.client.post("/retrieve", json=query | {"rerank": True}).json()["sources"]
                self.assertEqual(sources[0]["name"], "干扰.txt")
                self.assertEqual(set(model.call_args.args[1]["texts"]), {"报销 单据 必须盖章", "年度预算"})
        query.update(access_tags=["manager"], top_k=4)
        self.assertIn("权限.txt", [item["name"] for item in self.client.post("/retrieve", json=query).json()["sources"]])
        self.assertEqual(self.client.post("/retrieve", json=query | {
            "uploaded_after": "2099-01-01T00:00:00Z",
        }).json()["sources"], [])

    def test_stream_blocks_unsupported_answer_and_plans_query(self):
        self.client.post("/documents", json={"name": "证据.txt", "text": "库存为 5 件。"})
        original = self.fake_ollama

        def unsupported(route, payload):
            if "supported" in payload.get("format", {}).get("properties", {}):
                return {"message": {"content": '{"supported": false}'}, "done_reason": "stop"}
            return original(route, payload)

        self.model.side_effect = unsupported
        response = self.client.post("/questions/stream", json={
            "text": "数量？", "strategy": "decompose", "history": ["正在问库存"],
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: status", response.text)
        self.assertIn("资料不足。", response.text)
        self.assertIn('"verification": "unsupported"', response.text)
        self.assertNotIn("以文档为准", response.text)
        self.assertTrue(any(call.args[1].get("input") == ["数量？", "独立检索问题"]
                            for call in self.model.call_args_list))

    def test_upload_formats_and_versioned_original(self):
        import pymupdf
        from docx import Document
        from openpyxl import Workbook
        from pptx import Presentation

        workbook = Workbook()
        workbook.active.append(["合同金额", 350])
        docx = Document()
        docx.add_heading("年度合同")
        docx.add_table(rows=1, cols=2).cell(0, 0).text = "合同金额"
        pptx = Presentation()
        pptx.slides.add_slide(pptx.slide_layouts[0]).shapes.title.text = "Contract"
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((50, 50), "Contract")
            pdf.new_page()
            pdf_bytes = pdf.tobytes()
        samples = {
            "合同.pdf": (pdf_bytes, "Contract", "第 1 页"),
            "合同.txt": ("合同金额".encode("gb18030"), "合同金额", ""),
            "合同.md": (b"# Contract\n\nAmount 350", "Contract", ""),
            "合同.html": ("<main><table><tr><td>合同金额</td></tr></table></main>".encode(), "合同金额", "HTML 元素 1"),
        }
        for filename, obj, term, location in (
            ("合同.xlsx", workbook, "合同金额", "Sheet · 第 1 行"),
            ("合同.docx", docx, "合同金额", "元素 2"),
            ("合同.pptx", pptx, "Contract", "第 1 张幻灯片"),
        ):
            buffer = BytesIO()
            obj.save(buffer)
            samples[filename] = buffer.getvalue(), term, location
        for i, (filename, (data, term, location)) in enumerate(samples.items()):
            with self.subTest(filename=filename):
                namespace = f"sample-{i}"
                response = self.client.post("/files", files={"file": (filename, data)}, data={
                    "options": json.dumps({"namespace": namespace, "strategy": "hierarchical"}),
                })
                self.assertEqual(response.status_code, 200, response.text)
                result = self.client.post("/retrieve", json={"namespace": namespace, "text": term}).json()
                source = next(item for item in result["sources"] if term in item["text"])
                self.assertEqual(source["location"], location)
                self.assertEqual(self.client.get(source["original_url"]).content, data)
                if filename.endswith(".pdf"):
                    self.assertTrue(source["original_url"].endswith("#page=1"))
                    self.assertEqual(len(result["sources"]), 1)

    def test_semantic_chunking_and_context_budget(self):
        response = self.client.post("/documents", json={
            "name": "说明.txt", "text": "报销需要单据。打印需要纸张。" * 70,
            "strategy": "semantic", "chunk_size": 140, "overlap": 30,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(response.json()["chunks"], 1)
        _, _, context, _ = app.prepare_answer(app.RetrievalQuery(text="报销", context_tokens=512))
        self.assertLessEqual(len(json.dumps(context, ensure_ascii=False).encode()), 512)

    def test_hnsw_and_ivf_find_the_closest_vector(self):
        import faiss

        from retrieval import ann_index, dense_ranks

        faiss.omp_set_num_threads(1)
        rows = [{"rowid": i + 1, "vector": json.dumps([1, i / 20])} for i in range(80)]
        for mode in ("hnsw", "ivf"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"INKATLAS_VECTOR_INDEX": mode}):
                ranking = dense_ranks(rows, [[1, 0]], 4)
                self.assertEqual(ranking[0][0], 1)
                serialized = tuple(row["vector"] for row in rows)
                self.assertEqual(ann_index(mode, serialized).ntotal, 80)
                self.assertIs(ann_index(mode, serialized), ann_index(mode, serialized))

    def test_remote_embedding_provider_orders_and_caches_results(self):
        with patch.multiple(app, EMBED_API="https://example.test/v1", INDEX_MODEL="remote-test"):
            with patch.object(app, "model_json", return_value={"data": [
                {"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]},
            ]}) as upstream:
                self.assertEqual(app.embeddings(["a", "b", "a"]), [[1, 0], [0, 1], [1, 0]])
                self.assertEqual(app.embeddings(["b", "a"]), [[0, 1], [1, 0]])
                upstream.assert_called_once()
                self.assertEqual(upstream.call_args.args[0], "https://example.test/v1/embeddings")

    def test_static_web_ingestion_only_accepts_configured_origins(self):
        page = BytesIO(b"<main><h1>Expense policy</h1><p>Limit is 500.</p></main>")
        page.headers = Message()
        page.headers["Content-Type"] = "text/html; charset=utf-8"
        with patch.dict(os.environ, {"INKATLAS_WEB_ORIGINS": "https://docs.example.test"}):
            with patch.object(app, "build_opener") as opener:
                rejected = self.client.post("/web-pages", json={"url": "https://wrong.example.test/private"})
                self.assertEqual(rejected.status_code, 422)
                opener.assert_not_called()
                opener.return_value.open.return_value = page
                url = "https://docs.example.test/expense"
                imported = self.client.post("/web-pages", json={"url": url})
                self.assertEqual(imported.status_code, 200, imported.text)
                response = self.client.post("/retrieve", json={"text": "limit"})
                self.assertEqual(response.json()["sources"][0]["source"], url)

    def test_modelscope_answers_and_verifies_without_ollama(self):
        with patch.multiple(app, MODELSCOPE_TOKEN="test-token", EMBED_MODEL="none", INDEX_MODEL="none",
                            CHAT_MODEL="Qwen/Qwen3.5-35B-A3B"):
            document = self.client.post("/documents", json={"name": "政策.txt", "text": "报销上限为 500 元。"})
            self.assertEqual(document.status_code, 200, document.text)
            with patch.object(app, "model_json", side_effect=[
                {"choices": [{"finish_reason": "stop", "message": {"content": "报销上限为 500 元。[S1]"}}]},
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"supported":true}'}}]},
            ]) as upstream:
                response = self.client.post("/questions", json={"text": "报销上限是多少？"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["verification"], "supported")
            self.assertEqual(response.json()["sources"][0]["document_id"], document.json()["id"])
            self.assertEqual(upstream.call_count, 2)
            for request in upstream.call_args_list:
                url, payload, token = request.args
                self.assertEqual(url, app.MODELSCOPE_URL + "/chat/completions")
                self.assertEqual(payload["model"], app.CHAT_MODEL)
                self.assertEqual(token, "test-token")
            self.assertIn("JSON Schema", upstream.call_args.args[1]["messages"][0]["content"])
            self.model.assert_not_called()

    def test_account_scope_applies_to_retrieval_and_document_routes(self):
        from access import COOKIE

        legacy = self.client.post("/documents", json={"name": "legacy.txt", "text": "legacy evidence"}).json()
        config = self.path.with_suffix(".json")
        config.write_text(json.dumps([
            {"name": name, "token_sha256": hashlib.sha256(name.encode()).hexdigest(), "tenant": tenant,
             "role": role, "namespaces": scopes, "access_tags": tags}
            for name, tenant, role, scopes, tags in [
                ("owner", "default", "editor", ["default", "finance"], ["confidential"]),
                ("reader", "default", "reader", ["default", "finance"], []),
                ("outsider", "other", "editor", ["finance"], ["confidential"]),
            ]
        ]), encoding="utf-8")
        auth = lambda name: {"Authorization": f"Bearer {name}"}
        with patch.dict(os.environ, {"INKATLAS_ACCOUNTS_FILE": str(config)}):
            for path in ("/documents", "/session"):
                self.assertEqual(self.client.get(path).status_code, 401)
            options = json.dumps({"namespace": "finance", "access_tags": ["confidential"]})
            files = {"file": ("report.txt", b"secret amount 420")}
            first = self.client.post("/files", files=files, data={"options": options}, headers=auth("owner"))
            second = self.client.post("/files", files=files, data={"options": options}, headers=auth("outsider"))
            self.assertEqual((first.status_code, second.status_code), (200, 200))
            identity = first.json()["id"]
            other_id = second.json()["id"]
            self.assertNotEqual(identity, other_id)
            base = f"/documents/{identity}"
            original = base + "/versions/1/original"
            query = {"text": "secret", "namespace": "finance"}
            for account in ("reader", "outsider"):
                visible = self.client.get("/documents", headers=auth(account)).json()
                self.assertNotIn(identity, [row["id"] for row in visible])
                hits = self.client.post("/retrieve", json=query, headers=auth(account)).json()["sources"]
                self.assertNotIn(identity, [item["document_id"] for item in hits])
                self.assertEqual(self.client.get(original, headers=auth(account)).status_code, 404)
                self.assertEqual(self.client.get(base + "/versions/1/chunks/1", headers=auth(account)).status_code, 404)
            self.assertEqual(self.client.get(original).status_code, 401)
            self.assertEqual(self.client.post("/retrieve", json=query | {"access_tags": ["confidential"]},
                                              headers=auth("reader")).status_code, 403)
            self.assertEqual(self.client.post("/questions/stream", json={"text": "legacy"},
                                              headers=auth("outsider")).status_code, 403)
            self.assertEqual(self.client.delete(base, headers=auth("outsider")).status_code, 404)
            self.assertEqual(self.client.put(f"/files/{identity}", files=files, data={"options": options},
                                             headers=auth("outsider")).status_code, 404)
            self.assertEqual(self.client.delete(f"/documents/{legacy['id']}", headers=auth("reader")).status_code, 403)
            self.assertEqual(self.client.post("/files", files=files, data={"options": options},
                                              headers=auth("reader")).status_code, 403)
            self.assertEqual(self.client.get(f"/documents/{legacy['id']}/versions/1/chunks/1",
                                             headers=auth("reader")).status_code, 200)
            login = self.client.post("/session", headers=auth("owner"))
            self.assertIn("HttpOnly", login.headers["Set-Cookie"])
            self.assertIn(COOKIE, self.client.cookies)
            self.assertEqual(self.client.get(original).content, files["file"][1])
            self.assertEqual(self.client.delete(base, headers={"Origin": "https://other.example"}).status_code, 403)
            self.assertEqual(self.client.delete("/session").status_code, 204)
            self.assertEqual(self.client.get(original).status_code, 401)

    def test_admin_provisions_member_and_logout_revokes_session(self):
        from access import MEMBER_COOKIE, Member, store_member
        from contextlib import closing

        password = "Ab1234"
        member = {"name": "staff", "password": "Cd5678", "namespace": "default"}
        def challenge():
            with patch("access.new_captcha_code", return_value="AB234"):
                result = self.client.get("/session/captcha")
            self.assertEqual(result.status_code, 200)
            self.assertTrue(result.json()["image"].startswith("data:image/png;base64,"))
            return {"captcha_id": result.json()["id"], "captcha": "ab234"}

        self.assertEqual(self.client.post("/members", json=member).status_code, 403)
        with closing(app.database()) as connection, connection:
            store_member(connection, Member(name="boss", role="admin"), password)
        self.assertEqual(self.client.get("/documents").status_code, 401)
        admin = {"name": "boss", "password": password}
        self.assertEqual(self.client.post("/session", json=admin).status_code, 422)
        once = challenge()
        self.assertEqual(self.client.post("/session", json=admin | once | {"captcha": "XY999"}).status_code, 400)
        self.assertEqual(self.client.post("/session", json=admin | once).status_code, 400)
        once = challenge()
        self.assertEqual(self.client.post("/session", json=admin | once | {"password": "incorrect"}).status_code, 401)
        self.assertEqual(self.client.post("/session", json=admin | once).status_code, 400)
        expired = challenge()
        with closing(app.database()) as connection, connection:
            connection.execute("UPDATE login_challenges SET expires_at = 0 WHERE id = ?", (expired["captcha_id"],))
        self.assertEqual(self.client.post("/session", json=admin | expired).status_code, 400)
        self.assertEqual(self.client.post("/session", json=admin | challenge()).status_code, 200)
        self.assertTrue(self.client.get("/session").json()["manage_members"])
        old_session = self.client.cookies.get(MEMBER_COOKIE)
        self.assertEqual(self.client.post("/members", json=member | {"namespace": "outside"}).status_code, 403)
        self.assertEqual(self.client.post("/members", json=member | {"role": "admin"}).status_code, 422)
        self.assertEqual(self.client.post("/members", json=member | {"password": "short"}).status_code, 422)
        self.assertEqual(self.client.post("/members", json=member).status_code, 201)
        self.assertEqual(self.client.post("/members", json=member).status_code, 409)
        self.assertEqual(self.client.delete("/session").status_code, 204)
        self.client.cookies.set(MEMBER_COOKIE, old_session)
        self.assertEqual(self.client.get("/session").status_code, 401)
        self.client.cookies.clear()
        credentials = {key: member[key] for key in ("name", "password")}
        self.assertEqual(self.client.post("/session", json=credentials | challenge()).status_code, 200)
        self.assertEqual(self.client.get("/session").json()["role"], "reader")
        self.assertEqual(self.client.post("/members", json=member).status_code, 403)
        self.assertEqual(self.client.get("/documents").status_code, 200)

    def test_audit_records_activity_and_restricts_tenant_access(self):
        config = self.path.with_suffix(".json")
        config.write_text(json.dumps([
            {"name": name, "tenant": tenant, "role": role,
             "token_sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name, tenant, role in [
                ("admin", "default", "admin"), ("writer", "default", "editor"),
                ("reader", "default", "reader"), ("other-admin", "other", "admin"),
            ]
        ]), encoding="utf-8")
        def auth(name):
            return {"Authorization": f"Bearer {name}"}

        with patch.dict(os.environ, {"INKATLAS_ACCOUNTS_FILE": str(config)}):
            doc = {"name": "sample.txt", "text": "private-fixture-text"}
            created = self.client.post("/documents", json=doc, headers=auth("writer")).json()
            identity = created["id"]
            changed = self.client.put(f"/documents/{identity}", json=doc | {"text": "private-fixture-updated"},
                                      headers=auth("writer"))
            self.assertEqual(changed.status_code, 200)
            response = self.client.post("/questions/stream", json={"text": "private-fixture-query"},
                                        headers=auth("reader"))
            self.assertIn("event: done", response.text)
            with patch.object(app, "generate_answer", side_effect=app.HTTPException(502, "provider error")):
                self.assertEqual(self.client.post("/questions", json={"text": "failure-fixture"},
                                                  headers=auth("reader")).status_code, 502)
            self.assertEqual(self.client.delete(f"/documents/{identity}", headers=auth("writer")).status_code, 204)
            self.assertEqual(self.client.post("/documents", json=doc, headers=auth("other-admin")).status_code, 200)
            self.assertEqual(self.client.get("/audit", headers=auth("reader")).status_code, 403)
            result = self.client.get("/audit", headers=auth("admin"))
            self.assertEqual(result.status_code, 200)
            rows = result.json()
            self.assertEqual([row["action"] for row in rows], ["offline", "question", "question", "update", "ingest"])
            self.assertEqual(rows[0]["version"], 2)
            self.assertEqual(rows[1]["status"], 502)
            self.assertEqual(rows[2]["verification"], "supported")
            self.assertEqual(rows[2]["sources"][0]["document_id"], identity)
            self.assertNotIn("private-fixture", result.text)
            filtered = self.client.get("/audit", params={"actor": "writer", "action": "update"},
                                       headers=auth("admin")).json()
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["document_name"], "sample.txt")
            future = self.client.get("/audit", params={"after": "2099-01-01T00:00:00Z"}, headers=auth("admin"))
            self.assertEqual(future.json(), [])
            other = self.client.get("/audit", headers=auth("other-admin")).json()
            self.assertEqual([row["actor"] for row in other], ["other-admin"])

    def test_answer_evaluation_requires_correct_answer_and_citation(self):
        from evaluate_rag import Case, evaluate_case

        case = Case.model_validate({"id": "answer-check", "question": "What is the limit?",
                                    "evidence": [{"file": "policy.txt", "contains": ["680"]}],
                                    "answer_contains": [["680"]]})
        correct = {"id": "S1", "name": "policy.txt", "location": "", "chunk": 1, "text": "Limit is 680."}
        wrong = correct | {"id": "S2", "name": "other.txt"}
        with patch.object(app, "retrieve", return_value=[correct, wrong]):
            for answer, citations, expected in [
                ("680 [S1]", [correct], True), ("680 [S2]", [wrong], False), ("6800 [S1]", [correct], False),
            ]:
                with self.subTest(answer=answer), patch.object(app, "generate_answer", return_value={
                    "answer": answer, "sources": citations, "verification": "supported"
                }):
                    self.assertEqual(evaluate_case(case, 4, True)["answer_pass"], expected)

    def test_baseline_route_requires_admin_and_runs_isolated_evaluator(self):
        from subprocess import CompletedProcess

        config = self.path.with_suffix(".json")
        config.write_text(json.dumps([
            {"name": role, "role": role, "token_sha256": hashlib.sha256(role.encode()).hexdigest()}
            for role in ("admin", "reader")
        ]), encoding="utf-8")
        path = "/evaluation/baseline"
        with patch.dict(os.environ, {"INKATLAS_ACCOUNTS_FILE": str(config)}):
            self.assertEqual(self.client.post(path, json={}).status_code, 401)
            self.assertEqual(self.client.post(path, json={}, headers={"Authorization": "Bearer reader"}).status_code, 403)
            admin = {"Authorization": "Bearer admin"}
            with patch.object(app, "MODELSCOPE_TOKEN", ""):
                self.assertEqual(self.client.post(path, json={}, headers=admin).status_code, 503)
            report = {"summary": {"answer_pass_count": 8}}
            with patch.object(app, "MODELSCOPE_TOKEN", "fixture"), patch.object(
                app.subprocess, "run", return_value=CompletedProcess([], 0, json.dumps(report), "")
            ) as runner:
                response = self.client.post(path, json={}, headers=admin)
                self.assertEqual(response.json(), report)
                self.assertIn("--answers", runner.call_args.args[0])
                self.assertIn("--skip-ocr", runner.call_args.args[0])

    def test_evaluation_429_is_unscored_and_can_resume(self):
        from argparse import Namespace
        from evaluate_rag import run

        args = Namespace(dataset=app.ROOT / "evaluation" / "baseline.json",
                         files=app.ROOT / "rag-test-samples", top_k=4, skip_ocr=True,
                         answers=True, hybrid=False, from_case="ppt-drill")
        with patch.object(app, "generate_answer", side_effect=app.HTTPException(429, "limited")) as model:
            report = run(args)
        self.assertEqual(report["retry_from"], "ppt-drill")
        self.assertEqual(report["summary"]["answer_scored_count"], 0)
        self.assertEqual(report["summary"]["answer_error_count"], 1)
        self.assertEqual([row["id"] for row in report["cases"]], ["ppt-drill"])
        self.assertIsNone(report["cases"][0]["answer_pass"])
        model.assert_called_once()

    def test_feedback_ownership_and_monitoring_breakdown(self):
        config = self.path.with_suffix(".json")
        config.write_text(json.dumps([
            {"name": name, "role": role, "tenant": tenant,
             "token_sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name, role, tenant in [
                ("admin", "admin", "default"), ("reader", "reader", "default"),
                ("other", "admin", "other"),
            ]
        ]), encoding="utf-8")
        def auth(name):
            return {"Authorization": f"Bearer {name}"}

        with patch.dict(os.environ, {"INKATLAS_ACCOUNTS_FILE": str(config)}):
            self.client.post("/documents", json={"name": "evidence.txt", "text": "sample"}, headers=auth("admin"))
            stream = self.client.post("/questions/stream", json={"text": "sample"}, headers=auth("reader"))
            payload = next(json.loads(line[6:]) for event in stream.text.split("\n\n")
                           if event.startswith("event: done") for line in event.splitlines()[1:])
            path = f"/questions/{payload['feedback_id']}/feedback"
            negative = {"helpful": False, "reason": "wrong_source"}
            for user in ("admin", "other"):
                self.assertEqual(self.client.put(path, json=negative, headers=auth(user)).status_code, 404)
            self.assertEqual(self.client.put(path, json={"helpful": True}, headers=auth("reader")).status_code, 204)
            self.assertEqual(self.client.put(path, json=negative, headers=auth("reader")).status_code, 204)
            for response in [
                app.HTTPException(429),
                app.HTTPException(502),
                {"answer": "资料不足。", "sources": [], "verification": "unsupported"},
                {"answer": "资料不足。", "sources": [], "verification": "no_evidence"},
            ]:
                options = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
                with patch.object(app, "generate_answer", **options):
                    self.client.post("/questions", json={"text": "sample"}, headers=auth("reader"))
            self.assertEqual(self.client.get("/monitoring", headers=auth("reader")).status_code, 403)
            report = self.client.get("/monitoring?days=30", headers=auth("admin")).json()
            self.assertEqual(report["metrics"]["total"], 5)
            for key in ("rate_limited", "service_errors", "unsupported", "no_evidence", "supported"):
                self.assertEqual(report["metrics"][key], 1)
            self.assertIsNotNone(report["metrics"]["p95_ms"])
            self.assertEqual(report["feedback"], [{"helpful": 0, "reason": "wrong_source", "count": 1}])
            self.assertEqual(len(self.client.get("/audit?feedback=unhelpful", headers=auth("admin")).json()), 1)
            other = self.client.get("/monitoring", headers=auth("other")).json()
            self.assertEqual(other["metrics"]["total"], 0)
            self.assertIsNone(other["metrics"]["p95_ms"])

    def test_agent_tool_uses_server_side_scope_and_retrieval(self):
        config = self.path.with_suffix(".json")
        config.write_text(json.dumps([
            {"name": "agent", "role": "reader", "namespaces": ["default"],
             "token_sha256": hashlib.sha256(b"agent").hexdigest()},
            {"name": "writer", "role": "editor", "namespaces": ["default", "restricted"],
             "token_sha256": hashlib.sha256(b"writer").hexdigest()},
        ]), encoding="utf-8")
        path = "/tools/knowledge-base"
        agent = {"Authorization": "Bearer agent"}
        writer = {"Authorization": "Bearer writer"}
        with patch.dict(os.environ, {"INKATLAS_ACCOUNTS_FILE": str(config)}):
            public = {"name": "policy.txt", "text": "Expense limit is 680."}
            self.assertEqual(self.client.post("/documents", json=public, headers=writer).status_code, 200)
            self.assertEqual(self.client.post("/documents", json=public | {
                "name": "restricted.txt", "namespace": "restricted",
            }, headers=writer).status_code, 200)
            self.assertEqual(self.client.get(path).status_code, 401)
            schema = self.client.get(path, headers=agent).json()["function"]
            self.assertEqual(schema["name"], "knowledge_base_search")
            self.assertEqual(set(schema["parameters"]["properties"]), {"query", "knowledge_base_id", "top_k", "rerank"})
            self.model.reset_mock()
            result = self.client.post(path, headers=agent, json={"query": "Expense limit", "top_k": 1})
            source = result.json()["sources"][0]
            self.assertEqual(source["name"], "policy.txt")
            self.assertIn("680", self.client.get(source["url"], headers=agent).text)
            self.assertTrue(all(call.args[0] == "/api/embed" for call in self.model.call_args_list))
            self.assertEqual(self.client.post(path, headers=agent, json={
                "query": "Expense limit", "knowledge_base_id": "restricted",
            }).status_code, 403)
            self.assertEqual(self.client.post(path, headers=agent, json={
                "query": "Expense limit", "access_tags": ["private"],
            }).status_code, 422)

    def test_operational_alerts_use_tenant_and_recent_audited_events(self):
        config = self.path.with_suffix(".json")
        config.write_text(json.dumps([
            {"name": name, "role": role, "tenant": tenant,
             "token_sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name, role, tenant in [
                ("admin", "admin", "default"), ("other", "admin", "other"),
                ("reader", "reader", "default"),
            ]
        ]), encoding="utf-8")
        admin = {"Authorization": "Bearer admin"}
        with patch.dict(os.environ, {"INKATLAS_ACCOUNTS_FILE": str(config)}):
            invalid = self.client.post("/files", files={"file": ("broken.pdf", b"not a PDF")}, headers=admin)
            self.assertEqual(invalid.status_code, 422)
            for code in (429, 502):
                with patch.object(app, "generate_answer", side_effect=app.HTTPException(code)):
                    for _ in range(3):
                        self.assertEqual(self.client.post("/questions", json={"text": "test"},
                                                          headers=admin).status_code, code)
            for verdict, count in [("unsupported", 3), ("supported", 7), ("no_evidence", 3)]:
                with patch.object(app, "generate_answer", return_value={
                    "answer": "Fixture", "sources": [], "verification": verdict,
                }):
                    for _ in range(count):
                        self.assertEqual(self.client.post("/questions", json={"text": "test"},
                                                          headers=admin).status_code, 200)
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            with patch.object(app, "generate_answer", side_effect=error):
                with self.assertRaises(sqlite3.OperationalError):
                    self.client.post("/questions", json={"text": "test"}, headers=admin)
            report = self.client.get("/alerts", headers=admin).json()
            alerts = {item["code"]: item for item in report["alerts"]}
            self.assertEqual(set(alerts), {"document_parse", "database_busy", "model_error",
                                           "rate_limit", "verification_spike"})
            self.assertEqual(alerts["verification_spike"]["sample_count"], 10)
            self.assertEqual(alerts["verification_spike"]["rate"], 0.3)
            self.assertEqual(self.client.get("/alerts", headers={"Authorization": "Bearer other"}).json()["alerts"], [])
            self.assertEqual(self.client.get("/alerts", headers={"Authorization": "Bearer reader"}).status_code, 403)
            with sqlite3.connect(self.path) as connection:
                connection.execute("UPDATE audit_events SET created_at = 0")
            connection.close()
            self.assertEqual(self.client.get("/alerts", headers=admin).json()["alerts"], [])

    def test_queued_ingestion_survives_startup_and_preserves_active_version_on_failure(self):
        import time
        from threading import Event

        file = {"file": ("policy.txt", b"Expense limit 680.")}
        submitted = self.client.post("/jobs/files", files=file)
        self.assertEqual(submitted.status_code, 202)
        job = submitted.json()
        self.assertEqual(job["status"], "queued")
        self.assertEqual(self.client.get("/documents").json(), [])

        def wait_for(job_id, expected):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                row = self.client.get(f"/jobs/files/{job_id}").json()
                if row["status"] == expected:
                    return row
                time.sleep(.05)
            self.fail(f"Job did not reach {expected}: {row}")

        with self.client:
            completed = wait_for(job["id"], "completed")
            document = completed["result"]["id"]
            original = f"/documents/{document}/versions/1/original"
            self.assertEqual(self.client.get(original).content, file["file"][1])
            entered, release = Event(), Event()
            def fail_embedding(texts):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("Embedding test timed out")
                raise app.HTTPException(502, "Fixture embedding failed")
            with patch.object(app, "embeddings", side_effect=fail_embedding):
                replacement = self.client.put(f"/jobs/files/{document}", files={
                    "file": ("policy.txt", b"Expense limit 990."),
                }).json()
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(self.client.get(original).content, file["file"][1])
                    self.assertEqual(self.client.get(f"/jobs/files/{replacement['id']}").json()["stage"], "indexing")
                finally:
                    release.set()
                failed = wait_for(replacement["id"], "failed")
                self.assertEqual(failed["error"], "Fixture embedding failed")
            self.assertEqual(self.client.get(original).content, file["file"][1])
            updated = self.client.put(f"/jobs/files/{document}", files={
                "file": ("policy.txt", b"Expense limit 990."),
            }).json()
            self.assertEqual(wait_for(updated["id"], "completed")["result"]["version"], 2)
            self.assertEqual(self.client.get(original).status_code, 404)
            self.assertEqual(self.client.get(f"/documents/{document}/versions/2/original").content, b"Expense limit 990.")

    def test_ingestion_jobs_are_visible_only_to_the_submitter(self):
        config = self.path.with_suffix(".json")
        config.write_text(json.dumps([
            {"name": name, "role": "editor", "token_sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in ("first", "second")
        ]), encoding="utf-8")
        with patch.dict(os.environ, {"INKATLAS_ACCOUNTS_FILE": str(config)}):
            first = {"Authorization": "Bearer first"}
            second = {"Authorization": "Bearer second"}
            created = self.client.post("/jobs/files", files={"file": ("test.txt", b"Test document.")}, headers=first)
            self.assertEqual(created.status_code, 202)
            path = "/jobs/files/" + created.json()["id"]
            self.assertEqual(self.client.get(path, headers=first).status_code, 200)
            self.assertEqual(self.client.get(path, headers=second).status_code, 404)
            self.assertEqual(self.client.get("/jobs/files", headers=second).json(), [])

    def test_ragas_export_contains_the_final_answer_and_actual_context(self):
        from evaluate_rag import Case, evaluate_case

        case = Case.model_validate({"id": "sample", "question": "What is the limit?",
                                    "reference": "The limit is 680.", "evidence": [],
                                    "answer_contains": [["680"]]})
        source = {"id": "S1", "name": "policy.txt", "location": "", "chunk": 1, "text": "Full source"}
        answer = {"answer": "The limit is 680. [S1]", "sources": [source], "verification": "supported"}
        with patch.object(app, "retrieve", return_value=[source]), patch.object(
            app, "pack_context", return_value=([{"id": "S1", "text": "Trimmed context"}], [source])
        ), patch.object(app, "generate_answer", return_value=answer):
            baseline = evaluate_case(case, 4, True)
            exported = evaluate_case(case, 4, True, export_ragas=True)
        self.assertNotIn("ragas_input", baseline)
        self.assertEqual(exported["ragas_input"]["retrieved_contexts"], ["Trimmed context"])
        self.assertEqual(exported["ragas_input"]["response"], answer["answer"])
        self.assertEqual(exported["ragas_input"]["reference"], case.reference)

    def test_ragas_score_preserves_missing_metric_and_withheld_answers(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from evaluate_ragas import RequestState, Sample, score

        source = Sample(id="sample", user_input="What is the limit?", reference="The limit is 680.",
                        retrieved_contexts=["The limit is 680."], response="680")
        refused = source.model_copy(update={"id": "refused", "response": "Insufficient", "verification": "unsupported"})
        metric = SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=1.0)))
        report = asyncio.run(score([source, refused], {
            "context_precision": metric, "context_recall": metric,
            "faithfulness": metric, "answer_relevance": None,
        }, RequestState()))
        self.assertEqual(report["summary"]["context_precision"]["scored"], 2)
        self.assertEqual(report["summary"]["faithfulness"]["scored"], 1)
        self.assertEqual(report["summary"]["answer_relevance"], {"mean": None, "scored": 0})
        self.assertEqual(report["cases"][1]["errors"]["faithfulness"], "answer_withheld")

    def test_sensitive_policy_filters_old_sources_and_blocks_before_model_calls(self):
        from contextlib import closing

        old = self.client.post("/files", files={"file": ("legacy.txt", b"Expense secretcode 680")}).json()
        self.client.post("/documents", json={"name": "public.txt", "text": "Expense limit 390"})
        policy = self.path.with_suffix(".txt")
        policy.write_text("# ignored\nsecretcode\n", encoding="utf-8")
        with patch.dict(os.environ, {"INKATLAS_SENSITIVE_WORDS_FILE": str(policy)}):
            self.model.reset_mock()
            blocked_document = self.client.post("/documents", json={
                "name": "incoming.txt", "text": "Contains ＳＥＣＲＥＴ CODE.",
            })
            blocked_query = self.client.post("/questions", json={
                "text": "What is ＳＥＣＲＥＴ CODE?", "strategy": "rewrite",
            })
            blocked_history = self.client.post("/tools/knowledge-base", json={"query": "secretcode"})
            for response in (blocked_document, blocked_query, blocked_history):
                self.assertEqual(response.status_code, 403)
                self.assertNotIn("secretcode", response.text)
            self.model.assert_not_called()
            result = self.client.post("/retrieve", json={"text": "Expense limit"}).json()
            self.assertEqual([row["name"] for row in result["sources"]], ["public.txt"])
            original = f"/documents/{old['id']}/versions/1"
            self.assertEqual(self.client.get(original + "/original").status_code, 403)
            self.assertEqual(self.client.get(original + "/chunks/1").status_code, 403)
            with closing(app.database()) as connection:
                events = connection.execute("""
                    SELECT action, status, sources FROM audit_events WHERE failure_kind = 'sensitive_content'
                    ORDER BY id
                """).fetchall()
            self.assertEqual(events, [("ingest", 403, "[]"), ("question", 403, "[]"), ("retrieve", 403, "[]")])

    def test_sensitive_model_draft_is_not_sent_to_verifier(self):
        policy = self.path.with_suffix(".txt")
        policy.write_text("hiddenword\n", encoding="utf-8")
        self.client.post("/documents", json={"name": "source.txt", "text": "Allowed evidence"})
        original = self.fake_ollama
        def draft(route, payload):
            if route == "/api/chat":
                return {"message": {"content": "hiddenword [S1]"}, "done_reason": "stop"}
            return original(route, payload)

        self.model.reset_mock(side_effect=True)
        self.model.side_effect = draft
        with patch.dict(os.environ, {"INKATLAS_SENSITIVE_WORDS_FILE": str(policy)}):
            response = self.client.post("/questions/stream", json={"text": "Allowed evidence"})
        self.assertIn("event: error", response.text)
        self.assertNotIn("hiddenword", response.text)
        chat_calls = [call for call in self.model.call_args_list if call.args[0] == "/api/chat"]
        self.assertEqual(len(chat_calls), 1)
        self.assertNotIn("format", chat_calls[0].args[1])
