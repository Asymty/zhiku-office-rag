import json
import math
import os
import re
from functools import lru_cache

import jieba
from fastapi import HTTPException


def lexical_text(text):
    return " ".join(token for token in jieba.cut_for_search(text.lower())
                    if re.fullmatch(r"[^\W_]+", token))


def cosine(query, candidate):
    if len(query) != len(candidate):
        raise HTTPException(409, "向量维度不一致，请更换模型标识后重新索引。")
    norm = math.hypot(*query) * math.hypot(*candidate)
    return sum(a * b for a, b in zip(query, candidate)) / norm if norm else 0


def dense_ranks(rows, vectors, limit):
    mode = os.getenv("INKATLAS_VECTOR_INDEX", "flat").lower()
    if mode not in {"flat", "hnsw", "ivf"}:
        raise HTTPException(503, "向量索引配置须为 flat、hnsw 或 ivf。")
    if mode == "flat" or (mode == "ivf" and len(rows) < 80):
        candidates = [json.loads(row["vector"]) for row in rows]
        return [
            [row["rowid"] for _, row in sorted(
                zip((cosine(query, candidate) for candidate in candidates), rows),
                key=lambda item: -item[0],
            )[:limit]]
            for query in vectors
        ]
    import faiss
    import numpy as np

    queries = np.ascontiguousarray(vectors, dtype="float32")
    index = ann_index(mode, tuple(row["vector"] for row in rows))
    if queries.shape[1] != index.d:
        raise HTTPException(409, "向量维度不一致，请更换模型标识后重新索引。")
    faiss.normalize_L2(queries)
    _, positions = index.search(queries, min(limit, len(rows)))
    return [[rows[int(position)]["rowid"] for position in result if position >= 0] for result in positions]


@lru_cache(maxsize=4)
def ann_index(mode, serialized):
    import faiss
    import numpy as np

    candidates = [json.loads(vector) for vector in serialized]
    if len({len(candidate) for candidate in candidates}) != 1:
        raise HTTPException(409, "向量维度不一致，请更换模型标识后重新索引。")
    matrix = np.ascontiguousarray(candidates, dtype="float32")
    faiss.normalize_L2(matrix)
    dimensions = matrix.shape[1]
    # shortcut: keep a few filtered snapshots in memory until corpus size requires a maintained ANN index.
    if mode == "hnsw":
        index = faiss.IndexHNSWFlat(dimensions, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 80
    else:
        clusters = min(128, len(candidates) // 40)
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimensions), dimensions, clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)
        index.nprobe = min(clusters, 8)
    index.add(matrix)
    return index


def fuse(rankings, limit):
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
    return sorted(scores, key=scores.get, reverse=True)[:limit]
