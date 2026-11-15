"""疾控中心 3×3 新发病例表脱敏 API。

提交每格 0–9 人的 3×3 整数表（3 区县 × 3 年龄组），返回精确的行列
合计与遮蔽后的格值；1–4 人的格必遮蔽，必要时以最少补充遮蔽保证每个
被遮蔽的小人数格至少有两种可能取值；无解时只返回“不可发布”。
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from annotated_types import Ge, Le
from fastapi import Body, FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, StrictInt, conlist

from .suppress import N, solve

DESCRIPTION = """
## 用途

疾控中心统计员向外部研究人员发布 **3 区县 × 3 年龄组** 的新发病例表
（每格 0–9 人）前，调用本接口脱敏，避免小人数从合计中被倒推。

## 规则

1. 提交 3×3 整数表，每格 0–9；
2. 返回**精确的**行合计、列合计，以及遮蔽后的格值（`null` 表示被遮蔽）；
3. **1–4 人的格必须遮蔽**（首要遮蔽 `primary_masked_cells`）；
4. 对每个首要遮蔽格：在所有与**公开格值、行列合计、取值范围（0–9）**
   相容的原表中，该格**至少有两种可能人数**；
5. 为满足第 4 条而额外遮蔽的格（补充遮蔽 `complementary_masked_cells`）
   **数量最少**；并列时按坐标列表的**字典序**选取；
6. 无解时只返回 `{"status": "not_publishable"}`，不返回合计等任何信息。

坐标为 `[行, 列]`，从 0 开始（如 `[0, 2]` 表示第 1 行第 3 列）。

## 示例一：可反推（只遮蔽 1–4 会被唯一倒推）

请求：

```json
{"table": [[2, 3, 9], [8, 8, 8], [8, 8, 8]]}
```

若只遮蔽 (0,0)=2 与 (0,1)=3，攻击者用列合计即可唯一倒推：
x(0,0) = 18 − 8 − 8 = **2**，x(0,1) = 19 − 8 − 8 = **3**。
只补 1 格不够：(0,0) 被第 1 列的 8 + 8 钉死、(0,1) 被第 2 列的
8 + 8 钉死，需各解一处；字典序最小的 2 格方案为 [(1,0), (1,1)]。响应：

```json
{
  "status": "publishable",
  "row_totals": [14, 24, 24],
  "column_totals": [18, 19, 25],
  "grand_total": 62,
  "cells": [[null, null, 9], [null, null, 8], [8, 8, 8]],
  "primary_masked_cells": [[0, 0], [0, 1]],
  "complementary_masked_cells": [[1, 0], [1, 1]],
  "masked_cells": [[0, 0], [0, 1], [1, 0], [1, 1]]
}
```

此时与公开信息相容的原表中 x(0,0) ∈ {1, 2, 3}、x(0,1) ∈ {2, 3, 4}，
均无法被唯一确定。

## 示例二：需补遮蔽（最少 3 格，按字典序选取）

请求：

```json
{"table": [[2, 7, 6], [5, 8, 6], [7, 5, 6]]}
```

(0,0)=2 须遮蔽；但只遮蔽它时，行合计 15 − 7 − 6 = **2**（列合计
14 − 5 − 7 同样）即被倒推。可验证 1 格或 2 格补充遮蔽都不够（同行、
同列的公开格会重新把它钉死），最少需 3 格；在所有 3 格方案中按坐标
字典序取到 [(0,1), (1,0), (1,1)]。响应：

```json
{
  "status": "publishable",
  "row_totals": [15, 19, 18],
  "column_totals": [14, 20, 18],
  "grand_total": 52,
  "cells": [[null, null, 6], [null, null, 6], [7, 5, 6]],
  "primary_masked_cells": [[0, 0]],
  "complementary_masked_cells": [[0, 1], [1, 0], [1, 1]],
  "masked_cells": [[0, 0], [0, 1], [1, 0], [1, 1]]
}
```

此时 x(0,0) ∈ {0, 1, 2, 3}。

## 示例三：无解（只返回不可发布）

请求：

```json
{"table": [[1, 0, 0], [9, 9, 9], [9, 9, 9]]}
```

行合计 (1, 27, 27)、列合计 (19, 18, 18) 会强制 (0,0)=1：第 1 行合计
为 1 故 x(0,0) ≤ 1；第 1 列合计为 19 而另两格至多 9 + 9，故
x(0,0) ≥ 1。即使把其余 8 格全部遮蔽，该格也只有一种可能人数，无解。
响应（不返回合计等任何信息）：

```json
{"status": "not_publishable"}
```
"""

REQUEST_EXAMPLES = {
    "derivable_if_naive": {
        "summary": "可反推：仅遮蔽 1–4 会被列合计唯一倒推",
        "description": "接口将额外遮蔽 (1,0) 与 (1,1)，使 (0,0)、(0,1) 各有 ≥2 种可能取值。",
        "value": {"table": [[2, 3, 9], [8, 8, 8], [8, 8, 8]]},
    },
    "needs_complementary": {
        "summary": "需补遮蔽：最少 3 格补充遮蔽，按字典序选取",
        "description": "1–2 格补充遮蔽不足；字典序最小的可行方案为 [(0,1),(1,0),(1,1)]。",
        "value": {"table": [[2, 7, 6], [5, 8, 6], [7, 5, 6]]},
    },
    "no_solution": {
        "summary": "无解：行列合计强制 (0,0)=1，只返回 not_publishable",
        "description": "任何遮蔽方案都无法让 (0,0) 出现第二种可能，接口拒绝发布且不泄露原表。",
        "value": {"table": [[1, 0, 0], [9, 9, 9], [9, 9, 9]]},
    },
}

CellValue = Annotated[StrictInt, Ge(0), Le(9)]
Row3 = conlist(CellValue, min_length=N, max_length=N)
Table3x3 = conlist(Row3, min_length=N, max_length=N)
CoordPair = conlist(int, min_length=2, max_length=2)


class MaskRequest(BaseModel):
    """待脱敏的 3×3 整数表，每格 0–9。"""

    table: Table3x3


class PublishableResponse(BaseModel):
    """可发布：精确行列合计 + 遮蔽后的格值（null 表示被遮蔽）。"""

    status: Literal["publishable"] = "publishable"
    row_totals: list[int]
    column_totals: list[int]
    grand_total: int
    cells: list[list[int | None]]
    primary_masked_cells: list[CoordPair]
    complementary_masked_cells: list[CoordPair]
    masked_cells: list[CoordPair]


class NotPublishableResponse(BaseModel):
    """不可发布：不返回合计、格值等任何信息，以免泄露原表。"""

    status: Literal["not_publishable"] = "not_publishable"


app = FastAPI(
    title="病例表脱敏 API（3×3 小人数遮蔽）",
    description=DESCRIPTION,
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/mask",
    response_model=Union[PublishableResponse, NotPublishableResponse],
    summary="脱敏 3×3 病例表",
)
def mask_table(
    payload: Annotated[MaskRequest, Body(openapi_examples=REQUEST_EXAMPLES)],
) -> Union[PublishableResponse, NotPublishableResponse]:
    table = [list(row) for row in payload.table]
    solution = solve(table)
    if solution is None:
        # 只返回“不可发布”，不返回合计/格值，避免泄露原表
        return NotPublishableResponse()
    masked = set(solution.masked)
    cells = [
        [None if (i, j) in masked else table[i][j] for j in range(N)]
        for i in range(N)
    ]
    return PublishableResponse(
        row_totals=solution.row_totals,
        column_totals=solution.column_totals,
        grand_total=sum(solution.row_totals),
        cells=cells,
        primary_masked_cells=[list(c) for c in solution.primary],
        complementary_masked_cells=[list(c) for c in solution.complementary],
        masked_cells=[list(c) for c in solution.masked],
    )
