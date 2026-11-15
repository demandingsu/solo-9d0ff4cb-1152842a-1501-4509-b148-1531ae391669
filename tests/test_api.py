"""API 层测试：三个文档示例的端到端行为 + 输入校验。"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)


def test_example1_derivable_gets_complementary_mask():
    resp = client.post("/mask", json={"table": [[2, 3, 9], [8, 8, 8], [8, 8, 8]]})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "status": "publishable",
        "row_totals": [14, 24, 24],
        "column_totals": [18, 19, 25],
        "grand_total": 62,
        "cells": [[None, None, 9], [None, None, 8], [8, 8, 8]],
        "primary_masked_cells": [[0, 0], [0, 1]],
        "complementary_masked_cells": [[1, 0], [1, 1]],
        "masked_cells": [[0, 0], [0, 1], [1, 0], [1, 1]],
    }


def test_example2_minimal_lexicographic_complement():
    resp = client.post("/mask", json={"table": [[2, 7, 6], [5, 8, 6], [7, 5, 6]]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "publishable"
    assert body["row_totals"] == [15, 19, 18]
    assert body["column_totals"] == [14, 20, 18]
    assert body["cells"] == [[None, None, 6], [None, None, 6], [7, 5, 6]]
    assert body["primary_masked_cells"] == [[0, 0]]
    assert body["complementary_masked_cells"] == [[0, 1], [1, 0], [1, 1]]


def test_example3_not_publishable_leaks_nothing():
    resp = client.post("/mask", json={"table": [[1, 0, 0], [9, 9, 9], [9, 9, 9]]})
    assert resp.status_code == 200
    body = resp.json()
    # 只返回“不可发布”，不返回合计/格值等任何信息
    assert body == {"status": "not_publishable"}


@pytest.mark.parametrize(
    "table",
    [
        [[10, 0, 0], [0, 0, 0], [0, 0, 0]],          # 超上限
        [[-1, 0, 0], [0, 0, 0], [0, 0, 0]],          # 负数
        [[1.5, 0, 0], [0, 0, 0], [0, 0, 0]],         # 非整数
        [[2.0, 0, 0], [0, 0, 0], [0, 0, 0]],         # 浮点形式的整数也不接受
        [["3", 0, 0], [0, 0, 0], [0, 0, 0]],         # 字符串
        [[True, 0, 0], [0, 0, 0], [0, 0, 0]],        # 布尔
        [[1, 0, 0], [0, 0, 0]],                      # 只有 2 行
        [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],  # 4 行
        [[1, 0], [0, 0], [0, 0]],                    # 列数不对
        [[1, 0, None], [0, 0, 0], [0, 0, 0]],        # null
    ],
)
def test_invalid_tables_rejected(table):
    resp = client.post("/mask", json={"table": table})
    assert resp.status_code == 422


def test_openapi_docs_expose_examples():
    spec = client.get("/openapi.json").json()
    examples = spec["paths"]["/mask"]["post"]["requestBody"]["content"][
        "application/json"
    ]["examples"]
    assert set(examples) == {"derivable_if_naive", "needs_complementary", "no_solution"}
