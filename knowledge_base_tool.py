import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class KnowledgeBaseError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class KnowledgeBaseTool:
    """Call the read-only search tool on a trusted Zhiku service."""

    def __init__(self, base_url="http://127.0.0.1:8010", token=None, *, timeout=100):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                or parsed.query or parsed.fragment):
            raise ValueError("base_url 必须是知库服务地址。")
        if token and parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("远程服务需使用 HTTPS 发送访问令牌。")
        self.endpoint = base_url.rstrip("/") + "/tools/knowledge-base"
        self.token = token
        self.timeout = timeout
        self.opener = build_opener(NoRedirect())

    def _request(self, payload=None):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.endpoint, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise ValueError
            return result
        except HTTPError as error:
            raise KnowledgeBaseError(f"知库工具返回 HTTP {error.code}。", status=error.code) from error
        except (URLError, OSError, ValueError) as error:
            raise KnowledgeBaseError("无法读取知库工具响应。") from error

    def schema(self):
        """Return the function declaration for Chat Completions tool calling."""
        return self._request()

    def __call__(self, query, knowledge_base_id="default", top_k=4, rerank=False):
        return self._request({
            "query": query, "knowledge_base_id": knowledge_base_id, "top_k": top_k, "rerank": rerank,
        })
