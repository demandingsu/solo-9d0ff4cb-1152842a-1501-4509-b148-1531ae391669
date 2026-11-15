"""3×3 病例表的最小遮蔽求解器（首要遮蔽 + 补充遮蔽）。

规则（与 API 文档一致）：
- 每格取值范围为 0..9；
- 1..4 人的格必须遮蔽（首要遮蔽，primary suppression）；
- 对每个首要遮蔽格：在所有与“公开格值 + 行列合计 + 取值范围”相容的
  原表中，该格至少要有两种可能人数；
- 为满足上一条而额外遮蔽的格（补充遮蔽，complementary suppression）
  数量取最少；并列时按其坐标列表的字典序选取；
- 无解时返回 None（调用方只回复“不可发布”，不泄露原表）。

实现说明：3×3 表在行列合计固定时有 4 个自由度，取左上角 2×2 为自由
变量，其余 5 格由合计唯一推出。枚举量最大 10^4，配合 0..9 界剪枝，
对 2^9 个候选补充遮蔽集做检查在本场景下开销可忽略。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Optional

N = 3
LOW, HIGH = 0, 9               # 每格取值范围
PRIMARY_LOW, PRIMARY_HIGH = 1, 4  # 必须遮蔽的小人数区间

Coord = tuple[int, int]
Table = list[list[int]]

ALL_CELLS: tuple[Coord, ...] = tuple((i, j) for i in range(N) for j in range(N))


@dataclass(frozen=True)
class Solution:
    row_totals: list[int]
    column_totals: list[int]
    primary: list[Coord]        # 首要遮蔽格（值为 1..4）
    complementary: list[Coord]  # 补充遮蔽格（最少 + 字典序最小）

    @property
    def masked(self) -> list[Coord]:
        """全部遮蔽格，按 (行, 列) 字典序。"""
        return sorted(set(self.primary) | set(self.complementary))


def primary_cells(table: Table) -> list[Coord]:
    return [c for c in ALL_CELLS if PRIMARY_LOW <= table[c[0]][c[1]] <= PRIMARY_HIGH]


def consistent_tables(
    masked: set[Coord], table: Table, row_totals: list[int], col_totals: list[int]
) -> Iterator[tuple[int, ...]]:
    """枚举所有与公开信息相容的 3×3 表（按行展开为 9 元组）。

    相容 = 未遮蔽格等于原表值 + 行列合计一致 + 每格 0..9。
    遮蔽格在 0..9 内自由取值（攻击者无法区分首要/补充遮蔽）。
    """
    r0, r1, r2 = row_totals
    c0, c1, _c2 = col_totals
    t = table
    m = masked

    def domain(cell: Coord):
        return range(LOW, HIGH + 1) if cell in m else (t[cell[0]][cell[1]],)

    for x00 in domain((0, 0)):
        for x01 in domain((0, 1)):
            x02 = r0 - x00 - x01
            if not LOW <= x02 <= HIGH:
                continue
            if (0, 2) not in m and x02 != t[0][2]:
                continue
            for x10 in domain((1, 0)):
                x20 = c0 - x00 - x10
                if not LOW <= x20 <= HIGH:
                    continue
                if (2, 0) not in m and x20 != t[2][0]:
                    continue
                for x11 in domain((1, 1)):
                    x12 = r1 - x10 - x11
                    if not LOW <= x12 <= HIGH:
                        continue
                    if (1, 2) not in m and x12 != t[1][2]:
                        continue
                    x21 = c1 - x01 - x11
                    if not LOW <= x21 <= HIGH:
                        continue
                    if (2, 1) not in m and x21 != t[2][1]:
                        continue
                    x22 = r2 - x20 - x21  # 与 c2 - x02 - x12 恒等（合计一致）
                    if not LOW <= x22 <= HIGH:
                        continue
                    if (2, 2) not in m and x22 != t[2][2]:
                        continue
                    yield (x00, x01, x02, x10, x11, x12, x20, x21, x22)


def _every_primary_has_two_options(
    masked: set[Coord], primaries: list[Coord], table: Table,
    row_totals: list[int], col_totals: list[int],
) -> bool:
    """检查每个首要遮蔽格在相容表集合中是否都至少有两种可能取值。"""
    if not primaries:
        return True
    index = {cell: cell[0] * N + cell[1] for cell in primaries}
    seen: dict[Coord, set[int]] = {cell: set() for cell in primaries}
    for flat in consistent_tables(masked, table, row_totals, col_totals):
        for cell, i in index.items():
            seen[cell].add(flat[i])
        if all(len(s) >= 2 for s in seen.values()):
            return True
    return False


def solve(table: Table) -> Optional[Solution]:
    """求最小遮蔽方案；无解返回 None。

    按补充遮蔽格数 k 递增、同 k 按坐标列表字典序枚举，第一个可行解即
    “格数最少、并列时字典序最小”的方案。
    """
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[i][j] for i in range(N)) for j in range(N)]
    primaries = primary_cells(table)
    base = set(primaries)
    others = [c for c in ALL_CELLS if c not in base]
    for k in range(len(others) + 1):
        for extra in combinations(others, k):
            if _every_primary_has_two_options(
                base | set(extra), primaries, table, row_totals, col_totals
            ):
                return Solution(row_totals, col_totals, primaries, list(extra))
    return None
