"""核心算法测试：三个文档示例 + 边界情形 + 独立暴力枚举复核。

可直接运行：python3 tests/test_suppress.py
也可用 pytest 运行。
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.suppress import ALL_CELLS, _every_primary_has_two_options, primary_cells, solve

# 文档中的三个示例
EX1_DERIVABLE = [[2, 3, 9], [8, 8, 8], [8, 8, 8]]      # 可反推 -> 补 (1,1)
EX2_COMPLEMENTARY = [[2, 7, 6], [5, 8, 6], [7, 5, 6]]  # 需补遮蔽 -> 3 格
EX3_NO_SOLUTION = [[1, 0, 0], [9, 9, 9], [9, 9, 9]]    # 无解


def brute_force_options(table, masked, target):
    """独立实现的暴力枚举：target 格在所有相容表中的取值集合。

    与 app.suppress.consistent_tables 不同代码路径，用于交叉复核。
    """
    rows = [sum(r) for r in table]
    cols = [sum(table[i][j] for i in range(3)) for j in range(3)]
    values = set()
    for x00, x01, x10, x11 in itertools.product(range(10), repeat=4):
        grid = [
            [x00, x01, rows[0] - x00 - x01],
            [x10, x11, rows[1] - x10 - x11],
            [cols[0] - x00 - x10, cols[1] - x01 - x11, None],
        ]
        grid[2][2] = rows[2] - grid[2][0] - grid[2][1]
        if any(not (0 <= grid[i][j] <= 9) for i in range(3) for j in range(3)):
            continue
        if any(
            (i, j) not in masked and grid[i][j] != table[i][j]
            for i in range(3)
            for j in range(3)
        ):
            continue
        values.add(grid[target[0]][target[1]])
    return values


def test_example1_naive_masking_is_derivable():
    # 只遮蔽 1–4 的格时，(0,0) 与 (0,1) 都能被唯一倒推
    masked = {(0, 0), (0, 1)}
    assert brute_force_options(EX1_DERIVABLE, masked, (0, 0)) == {2}
    assert brute_force_options(EX1_DERIVABLE, masked, (0, 1)) == {3}


def test_example1_solution():
    sol = solve(EX1_DERIVABLE)
    assert sol is not None
    assert sol.primary == [(0, 0), (0, 1)]
    # (0,0) 被第 1 列 8+8 钉死、(0,1) 被第 2 列 8+8 钉死，需各解一处
    assert sol.complementary == [(1, 0), (1, 1)]
    assert sol.masked == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert sol.row_totals == [14, 24, 24]
    assert sol.column_totals == [18, 19, 25]
    masked = set(sol.masked)
    assert brute_force_options(EX1_DERIVABLE, masked, (0, 0)) == {1, 2, 3}
    assert brute_force_options(EX1_DERIVABLE, masked, (0, 1)) == {2, 3, 4}
    # 独立验证“最少 2 格”：所有 1 格补充遮蔽都不可行
    primaries = sol.primary
    others = [c for c in ALL_CELLS if c not in set(primaries)]
    for extra in itertools.combinations(others, 1):
        assert not _every_primary_has_two_options(
            set(primaries) | set(extra), primaries, EX1_DERIVABLE,
            sol.row_totals, sol.column_totals,
        ), f"size-1 set {extra} unexpectedly feasible"


def test_example2_solution_is_minimal_and_lexicographic():
    sol = solve(EX2_COMPLEMENTARY)
    assert sol is not None
    assert sol.primary == [(0, 0)]
    assert sol.complementary == [(0, 1), (1, 0), (1, 1)]
    masked = set(sol.masked)
    assert brute_force_options(EX2_COMPLEMENTARY, masked, (0, 0)) == {0, 1, 2, 3}
    # 独立验证“最少 3 格”：所有 0/1/2 格补充遮蔽都不可行
    others = [c for c in ALL_CELLS if c != (0, 0)]
    rows, cols = sol.row_totals, sol.column_totals
    for k in (0, 1, 2):
        for extra in itertools.combinations(others, k):
            assert not _every_primary_has_two_options(
                {(0, 0)} | set(extra), [(0, 0)], EX2_COMPLEMENTARY, rows, cols
            ), f"size-{k} set {extra} unexpectedly feasible"


def test_example3_no_solution():
    assert solve(EX3_NO_SOLUTION) is None
    # 即使 9 格全部遮蔽，(0,0) 仍被行列合计强制为 1
    assert brute_force_options(EX3_NO_SOLUTION, set(ALL_CELLS), (0, 0)) == {1}


def test_no_primary_cells_publishes_everything():
    table = [[0, 5, 6], [7, 8, 9], [5, 6, 7]]
    sol = solve(table)
    assert sol is not None
    assert sol.primary == [] and sol.complementary == [] and sol.masked == []


def test_all_cells_primary_may_still_publish():
    table = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    sol = solve(table)
    assert sol is not None
    assert sol.complementary == []
    assert sol.masked == sorted(ALL_CELLS)  # 只发布合计，全表遮蔽
    assert len(brute_force_options(table, set(ALL_CELLS), (0, 0))) >= 2


def test_primary_only_sufficient_case():
    # 2×2 的首要遮蔽块互相“解锁”，无需补充遮蔽
    table = [[2, 3, 0], [3, 2, 0], [0, 0, 9]]
    sol = solve(table)
    assert sol is not None
    assert sol.primary == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert sol.complementary == []
    masked = set(sol.masked)
    assert brute_force_options(table, masked, (0, 0)) == {0, 1, 2, 3, 4, 5}


def test_primary_cells_detection():
    assert primary_cells([[0, 1, 4], [5, 9, 3], [2, 8, 0]]) == [
        (0, 1), (0, 2), (1, 2), (2, 0),
    ]


def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
