# 知库问答｜企业知识库 RAG

FastAPI 提供文件入库、混合检索和问答。SQLite 保存原文件、文档版本、文本片段、向量、BM25 索引和向量缓存。默认由本地 Ollama 运行 BGE-M3 和 Qwen，可接 OpenAI 兼容的嵌入接口及 Hugging Face TEI 重排服务。

## 魔搭免费 API 启动

先在 [魔搭访问令牌](https://modelscope.cn/my/myaccesstoken) 获取令牌。把 `.env.example` 复制为项目根目录的 `.env`（已有 `.env` 可直接编辑），在 `MODELSCOPE_API_KEY=` 后填入令牌，然后在 PowerShell 运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010
```

服务和离线评估会自动读取项目根目录的 `.env`。启动窗口已设置的同名环境变量优先，修改 `.env` 后需重启服务。`.env` 已加入 Git 忽略规则。打开 [知识库](http://127.0.0.1:8010)，选「层级分块」和「直接检索」试问。默认使用 `Qwen/Qwen3.5-35B-A3B` 回答和校验，问题和命中的证据片段会发送给魔搭。使用没有提供方后缀的模型 ID，额度以 [魔搭账户用量](https://modelscope.cn/magicube/usage) 为准。

只配置令牌时使用本地 BM25，无需 Ollama。之前按其他嵌入模型入库的文件需重新导入一次，才能在当前模式下检索。若需要向量混合检索，在 Ollama 下载 `bge-m3`，启动本项目之前设置 `$env:INKATLAS_EMBED_MODEL = 'bge-m3'`，回答和校验仍由魔搭处理。也可通过下文的嵌入 API 配置接入账号已有的向量服务。

## Ollama 本地启动

安装 Python 3.11+ 和 Ollama，启动 Ollama 后在 PowerShell 运行：

```powershell
ollama pull bge-m3
ollama pull qwen2.5:3b
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010
```

打开 [知识库](http://127.0.0.1:8010) 上传文件并问答，或打开 [API 文档](http://127.0.0.1:8010/docs) 调用接口。项目目录下的 `atlas.sqlite3` 保存原件和索引。刷新页面或重启服务后，文档库会重新加载已保存的文件。可在文档卡片打开原件、上传新版或下线。已有 SQLite 索引会自动迁移，迁移前可备份数据库文件。

## 文档入库

| 格式 | 提取与定位 |
| --- | --- |
| PDF | 提取文本、可识别表格和页码，引用可打开原 PDF 对应页。扫描页用 Tesseract OCR，清洗会识别重复的首尾行、页码和低透明度的大号文字水印。 |
| Word `.docx` | 提取标题、段落、表格及文档内位置。 |
| Excel `.xlsx` | 提取单元格缓存值、工作表和行号。公式依赖工作簿中保存的计算结果。 |
| PowerPoint `.pptx` | 提取幻灯片标题、文本、表格和幻灯片编号。 |
| HTML `.html/.htm` | 提取标题、列表、段落和表格，去掉导航、页脚和脚本。支持上传网页快照和从配置清单抓取静态 HTTPS 网页。 |
| Markdown、TXT | 支持 UTF-8、带 BOM 的 UTF-16 和 GB18030。Markdown 标题用于层级分块。 |

扫描件需把 [Tesseract 官方语言文件](https://github.com/tesseract-ocr/tessdata_fast) 中的 `chi_sim.traineddata` 和 `eng.traineddata` 放在同一目录，并将 `TESSDATA_PREFIX` 指向该目录。默认语言为 `chi_sim+eng`，可用 `INKATLAS_OCR_LANG` 覆盖。复杂版面、扫描表格、浮动水印和低清扫描件的提取质量取决于文件。原文件保留原始版式。Office 使用现代格式，二进制旧格式 `.doc/.xls/.ppt` 需先转换。

文件最多 20 MB。入库选项包括 `source`、`department`、`namespace`、`access_tags`、`strategy`、`chunk_size` 和 `overlap`。固定分块使用滑动窗口。语义分块比较相邻句子的向量相似度。层级分块保留标题路径，并以段落、表格和页面等结构为边界。正文清洗后会过滤空白和无文字的块。

## 启用账号与知识库权限

### 管理员创建成员

首次启用时，先停止服务。在项目目录执行下面第一条命令，按提示创建首个管理员，再启动服务。原有文档仍在默认租户。

```powershell
.\.venv\Scripts\python.exe .\create_admin.py
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010
```

进入 [知库问答](http://127.0.0.1:8010/)，用管理员用户名、密码和图形验证码登录。点右上角「添加成员」，设置用户名、至少 6 个字符的初始密码、角色和知识库。密码框的小眼睛可切换显示和隐藏。该入口只允许管理员为本租户创建只读或编辑成员。

首个管理员只需创建一次。账户保存在 SQLite 中，密码使用带随机盐的 scrypt 摘要，浏览器会话有效期 8 小时，退出时服务端会撤销会话。图形验证码 3 分钟内有效，每次登录尝试后需换一张。登录尝试过多会暂时限流。

### 旧版令牌账号

设置 `INKATLAS_ACCOUNTS_FILE` 后，网页需要登录，API 可用 `Authorization: Bearer <访问令牌>`。账号文件只保存令牌的 SHA-256 摘要。以下命令生成一对供本机测试的独立令牌，把配置写入 `accounts.local.json`。在项目目录运行一次，保存打印的令牌，再用同一个 PowerShell 窗口重启服务。

```powershell
@'
import hashlib, json, secrets
from pathlib import Path

accounts = []
for name, tenant, role in [
    ("local-admin", "default", "admin"),
    ("other-reader", "other", "reader"),
]:
    token = secrets.token_urlsafe(32)
    accounts.append({"name": name, "tenant": tenant, "role": role, "namespaces": ["default"],
                     "access_tags": [], "token_sha256": hashlib.sha256(token.encode()).hexdigest()})
    print(name, token)
with Path("accounts.local.json").open("x", encoding="utf-8") as file:
    json.dump(accounts, file, indent=2)
'@ | .\.venv\Scripts\python.exe -
$env:INKATLAS_ACCOUNTS_FILE = (Resolve-Path .\accounts.local.json).Path
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010
```

旧文档归入 `default` 租户，保留原有知识库和标签。用 `local-admin` 登录可看到已有的默认知识库文档，另一个租户的账号只能看到自己的数据。每个账号可在配置中指定 `namespaces` 和 `access_tags`。账号需持有文档的全部标签才能检索或打开原件。角色 `reader` 可问答和读文档，`editor` 也可入库、更新和下线，`admin` 另外可以修改已有文档的权限标签。所有角色都受本租户、知识库和标签约束。网页退出会清除浏览器登录状态，撤销令牌需从账号文件移除对应记录。

数据库中没有成员且未设置账号文件时是免登录的本地单人模式，仅供绑定 `127.0.0.1` 使用。当前多租户使用共享 SQLite 的逻辑隔离。魔搭和本地模型配置独立于账号配置。

### 审计日志

管理员登录后，点击右上角「审计日志」。可按账号、操作和时间筛选本租户最近 50 条匹配记录。导入、更新、下线和问答记录包含账号、时间、知识库、结果和耗时，文档操作记录关联文档 ID 和版本，问答记录包含事实校验状态和引用片段 ID。问题、回答、片段正文和凭据不进入审计表。历史引用可能随文档替换或下线而失效。

API 为 `GET /audit`，可传 `actor`、`action`、`after`、`before` 和 `limit`（1–100），时间使用带时区的 ISO 8601。审计在操作处理后写入 SQLite，写入故障会记到服务日志。未进入业务处理的请求（例如认证失败和请求参数校验失败）不在这批审计事件中。

### 问答质量监控与反馈

回答结束后可点「有用」或「没用」，后者可附选一个预设原因。同一账号可以修改自己对该回答的反馈，统计只计算最新一票。日志保存反馈原因和关联的问答记录，不保存问题、回答或证据正文。管理员可在「审计日志」筛选「没用」记录，查看仍然可用的引用。

管理员的「质量监控」按最近 7 天或 30 天显示租户内实际问答数、429、服务错误、资料不足、校验未通过、引用无效和有用／没用票数。平均耗时和 P95 仅统计已完成请求，从检索开始到回答校验结束。离线评估使用独立数据库，不计入这些数据。

`GET /monitoring?days=7` 返回监控统计。JSON 问答响应和 SSE 的 `done` 事件携带 `feedback_id`。通过 `PUT /questions/{feedback_id}/feedback` 提交 `helpful` 和可选的 `reason`，只能评价本人问答。未建立审计记录时不会提供有效的反馈 ID。

### 运行告警

管理员打开网页后，每分钟读取一次本租户的运行告警，点击「质量监控」可看详情。规则使用最近 15 分钟的审计事件：

| 告警 | 触发条件 |
| --- | --- |
| 文档解析失败 | 至少 1 次解析失败，包括 OCR 缺少语言文件 |
| 数据库忙或锁超时 | 至少 1 次 SQLite 忙或锁错误 |
| 模型调用异常 | 问答返回 502 或 504 累计至少 3 次 |
| 模型限流 | 问答返回 429 累计至少 3 次 |
| 事实校验未通过比例偏高 | 至少 10 次实际校验，至少 3 次未通过且比例达到 25% |

「资料不足」和引用无效不进入事实校验比例的分母。告警是滚动窗口里的网页提示，由 `GET /alerts` 提供。数据库完全不可用时，网页会显示告警读取失败，未能写入的事件仍可从服务日志排查。

### 敏感词规则

在项目目录新建 UTF-8 的 `sensitive-words.local.txt`，每行一个词组，例如 `示例内部代号`。空行和以 `#` 开头的行会跳过。然后在 `.env` 设置 `INKATLAS_SENSITIVE_WORDS_FILE=sensitive-words.local.txt` 并重启服务。启用后编辑词表即可生效。

规则在入库建立向量前检查提取正文，在检索前检查问题、历史和改写结果，在排序及生成前排除命中规则的历史片段，在事实校验和发布前检查模型初稿。直接访问片段或原文件时也会检查已提取的内容。拦截返回 HTTP 403。导入、更新、检索和问答的审计记录包含失败类型，日志省略命中的词和正文。规则使用规范化后的字面匹配，覆盖项目已提取的文字，图片、附件和解析器未提取的内容不在扫描范围内。

## 检索和生成

- **索引**。SQLite FTS5 用 BM25 计算关键词排名，中文经 jieba 分词。向量侧可选择精确余弦、Faiss HNSW 或 IVF。两路候选经倒数排名融合，可再交给 TEI 的交叉编码模型重排。
- **元数据过滤**。候选进入向量检索和重排前，按知识库、部门、类型、入库时间、权限标签及最短文本长度过滤。启用账号时租户和授权标签来自服务端配置，请求只能进一步缩小范围。
- **查询增强**。请求可选择 `none`、`rewrite`、`expand`、`decompose` 或 `hyde`，可传入最近对话历史。HyDE 的假想答案只参与向量检索。生成输入使用真实检索证据。
- **上下文与事实校验**。按预算抽取相关句段，使用 UTF-8 序列化大小作为保守的 token 预算估算。生成后检查引用编号，再调用模型检查事实和引用证据的一致性。未通过时返回资料不足。模型判定质量需要用业务样本评估。
- **流式接口**。SSE 先发送检索和生成阶段事件，事实校验后分段发送回答，最后发送引用。上游模型的原始 token 在校验完成前不对业务侧发布。

## 接口

| 路径 | 请求和用途 |
| --- | --- |
| `POST /files` · `PUT /files/{id}` | Multipart 表单字段 `file` 和可选 JSON 字符串 `options`。返回文档 ID、版本和片段数。 |
| `POST /jobs/files` · `PUT /jobs/files/{id}` | 异步导入或更新，表单同上。返回 HTTP 202 和任务 ID。 |
| `GET /jobs/files` · `GET /jobs/files/{job_id}` | 查询本人入库任务、阶段和失败原因。 |
| `POST /web-pages` · `PUT /web-pages/{id}` | JSON 字段 `url`、可选 `name` 和入库选项。需配置来源清单。 |
| `POST /documents` · `PUT /documents/{id}` | 兼容已有客户端，直接提交 JSON 文本 `name`、`text` 和入库选项。 |
| `GET /documents` · `DELETE /documents/{id}` | 列出文档和元数据。下线会删除原文件、片段和索引记录，保留文档清单。 |
| `POST /retrieve` | 接收 `text` 和可选 `top_k`、`namespace`、`department`、`document_type`、`access_tags`、`uploaded_after`、`uploaded_before`、`min_chars`、`strategy`、`history` 和 `rerank`。时间使用带时区的 ISO 8601。 |
| `GET /tools/knowledge-base` · `POST /tools/knowledge-base` | 获取 Agent 函数定义，或执行只读知识库检索。 |
| `POST /questions` · `POST /questions/stream` | 接受同一组检索参数和 `context_tokens`。前者返回 JSON，后者返回 `status`、`delta`、`done` 或 `error` 事件。 |

新版生成全部向量后，系统在一个事务中替换旧原件、片段和关键词索引。替换失败时旧版仍可访问。更新或下线后，旧版引用返回 404。只保留有效版本的正文、原件和索引。

网页默认使用异步入库。「入库任务」显示排队、解析／OCR、分块与索引、保存及最终结果。文件仍需先完成上传，索引在后台建立，单文件限制为 20 MB。排队任务保存在 SQLite 中，重启后继续处理。运行中的任务若进程异常退出，约两分钟后标记中断，可重新提交。完成或失败后会清掉队列中的临时原件，成功的原件保留在文档库。文档更新以提交时的版本为准，排队期间已有其他更新时会提示冲突。

### Agent 接入

`KnowledgeBaseTool` 可作为 Chat Completions 风格的工具使用。给 Agent 单独配置一个 `reader` 类型的服务账号，按「旧版令牌账号」的格式保存令牌摘要。将账号的 `tenant`、`namespaces` 和 `access_tags` 限定为 Agent 需要的范围。在 Agent 进程设置 `INKATLAS_AGENT_TOKEN` 保存原始令牌。该令牌由调用端附加到请求，工具入参中只有问题、知识库 ID、返回数量和是否重排。

```python
import os
from knowledge_base_tool import KnowledgeBaseTool

tool = KnowledgeBaseTool(token=os.environ["INKATLAS_AGENT_TOKEN"])
tools = [tool.schema()]  # 传给 Chat Completions 的 tools 参数
result = tool(query="跨市住宿报销每晚上限是多少？",
              knowledge_base_id="default", top_k=4, rerank=False)
print(result["sources"])
```

当模型返回 `knowledge_base_search` 调用时，用同一个 `tool` 执行解码后的 JSON 参数，再将结果作为工具消息交回模型。结果包含片段正文、文件名、位置、版本和相对引用链接。需要重排时先配置 TEI。本工具只做检索，Agent 可自行决定怎样依据片段组织回答。启用 BM25 时，工具检索不调用生成模型。HTTP 接口同样可供其他语言调用。

## 配置

| 环境变量 | 默认值和作用 |
| --- | --- |
| `INKATLAS_OLLAMA_URL` | `http://127.0.0.1:11434`，Ollama 地址。 |
| `MODELSCOPE_API_KEY` | 可选魔搭令牌。设置后，查询增强、回答和事实校验走魔搭。 |
| `INKATLAS_EMBED_MODEL` · `INKATLAS_CHAT_MODEL` | 本地默认 `bge-m3`、`qwen2.5:3b`。魔搭模式默认 `none`、`Qwen/Qwen3.5-35B-A3B`。如配置嵌入 API，请显式填写其模型名。切换嵌入配置后，重新提交当前文档生成对应索引。 |
| `INKATLAS_EMBED_API_URL` · `INKATLAS_EMBED_API_KEY` | 可选 OpenAI 兼容嵌入服务前缀，例如 `http://127.0.0.1:8080/v1`。启用后嵌入文本发送给该服务。 |
| `INKATLAS_VECTOR_INDEX` | `flat`、`hnsw` 或 `ivf`。默认 `flat`。Faiss 使用过滤后的 SQLite 向量构建并缓存索引，进程重启后重建。IVF 在候选不足 80 条时用精确检索。 |
| `INKATLAS_RERANK_URL` · `INKATLAS_RERANK_API_KEY` | 可选 [TEI](https://github.com/huggingface/text-embeddings-inference#using-re-rankers-models) 地址，例如 `http://127.0.0.1:8080`，可部署 BGE Reranker。设置后请求传 `rerank: true`。 |
| `INKATLAS_RERANK_MIN_SCORE` | 默认 0，可按验证集配置 0–1 的最低重排分数。 |
| `INKATLAS_VERIFY_MODEL` | 默认与回答模型相同，可换成独立校验模型。 |
| `INKATLAS_PROMPT_FILE` | 可选 JSON 路径，字段为 `system`、`rules`、`citation`（`brackets` 或 `parentheses`）、`insufficient`。 |
| `INKATLAS_WEB_ORIGINS` | 默认空。静态网页抓取允许的来源以逗号分隔，例如 `https://docs.example.com,https://kb.example.org`。重定向会被拒绝。 |

## 回归测试

### 离线检索基准

从项目目录运行。默认把样本文档导入临时 SQLite 数据库，只用本地 BM25 检索，不调用模型，也不修改当前知识库。

```powershell
.\.venv\Scripts\python.exe .\evaluate_rag.py --top-k 4 --skip-ocr
```

已配置 `TESSDATA_PREFIX` 后去掉 `--skip-ocr`，可测完整的 8 个证据问答和 1 个无匹配问题。评估集见 `evaluation/baseline.json`，每条正例按文件、可选位置、证据关键词标注。脚本输出 JSON，包含 Hit@K、Precision@K、证据 Recall@K、MRR@K、无匹配题的空检索结果和逐题命中位置。Precision@K 以 K 为分母，证据 Recall@K 以题目中标注的证据数为分母。

如需导出，可加 `--output 新报告.json`。加 `--answers` 会调用已配置的回答和校验模型，对照答案关键词和引用证据检查结果，可能消耗魔搭额度。加 `--hybrid` 使用已配置的嵌入服务评估混合检索。关键词核对适合回归样本，语义质量和 RAGAS 指标需独立评估。

也可以登录管理员账号，在右上角打开「问答评估」并点击「运行魔搭评估」。网页使用启动服务时配置的魔搭令牌和固定样本，默认跳过扫描 PDF，最多调用 14 次模型，勾选扫描 PDF 后最多 16 次。结果显示逐题回答、引用和是否满足标注规则，评估数据库独立于文档库。收到限流或额度错误时停止后续调用。

模型调用失败的题单独标为未评分。魔搭评估请求间隔至少 3 秒，遇到 429 时暂停，若上游提供等待秒数会显示等待时间。恢复后点「从受限题继续」可跳过所有已评分题。同一标签页刷新后可恢复新版报告。命令行也支持 `--from-case` 和可重复传入的 `--completed-case`，检索仍使用完整样本。

单题诊断示例为 `evaluate_rag.py --answers --case html-visitor --diagnose`。诊断报告额外显示模型初稿、传入证据和校验结果，适合区分检索、生成与校验问题。该命令会调用模型，诊断报告只输出到本地，普通问答与审计日志仍不包含被拦截的初稿。

### RAGAS 批量评分

在项目目录运行，建议先用一题验证评估模型配置。

```powershell
.\.venv\Scripts\python.exe .\evaluate_rag.py --answers --case html-visitor --export-ragas .\ragas-input.json
python -m venv .venv-ragas
.\.venv-ragas\Scripts\python.exe -m pip install -r requirements-evaluation.txt
.\.venv-ragas\Scripts\python.exe .\evaluate_ragas.py .\ragas-input.json --output .\ragas-report.json
```

去掉 `--case html-visitor` 可批量导出。也可导入自己的 JSON 数组，每行提供 `id`、`user_input`、`reference`、`retrieved_contexts`（字符串数组）和 `response`。加 `--validate-only` 仅检查格式。导出包含实际送给生成模型的上下文、最终回答和参考答案，不使用预期答案代替模型回答。未完成的 API 请求和无参考答案的拒答题留在原基准报告。正例被系统拦截时，仍可计算两项上下文分数。

评分会输出 Context Precision、Context Recall、Faithfulness 和 Answer Relevance 的逐题分数、均值及有效样本数。默认使用已配置的魔搭模型评判，也可在 `.env` 用 `INKATLAS_RAGAS_JUDGE_URL`、`INKATLAS_RAGAS_JUDGE_MODEL` 和 `INKATLAS_RAGAS_JUDGE_KEY` 单独配置 OpenAI 兼容评审服务。未配置魔搭时默认走 Ollama 的兼容接口。Answer Relevance 另需嵌入模型，可设置 `INKATLAS_RAGAS_EMBED_MODEL=bge-m3` 使用已运行的 Ollama，或另外设置 `INKATLAS_RAGAS_EMBED_URL` 和 `INKATLAS_RAGAS_EMBED_KEY`。缺少嵌入配置时，该项为 `null` 并标注原因。魔搭 429 会停止后续评估请求，已有分数会保留。

评分命令会把样本发送给配置的评估服务，可能消耗 API 额度。导出文件含回答和上下文，请与项目审计日志分开保存。

### 接口回归测试

```powershell
.\.venv\Scripts\python.exe -m pip install httpx
.\.venv\Scripts\python.exe -m unittest -v
```

测试使用临时 SQLite 数据库，生成文档样本并模拟模型接口。真实模型的检索相关性、重排阈值和事实校验质量需另外评估。

## 许可证

本项目采用 [GNU Affero General Public License v3.0](LICENSE)（AGPL-3.0-only）。第三方依赖遵循各自的许可证，PDF 解析使用的 [PyMuPDF 提供 AGPL 和商业许可](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)。
