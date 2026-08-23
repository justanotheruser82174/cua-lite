"""A webvoyager step whose calls were all rejected still consumes -- and scores -- the budget.

Run:
    uv run pytest tests/gym/envs/webharbor/webvoyager/test_webvoyager_step_budget.py -q

Regression guard. The rejected-calls early return used to sit BEFORE
``self._step_count += 1``, so a model emitting only invalid calls never
consumed budget and could never truncate -- bounded only by the agent loop's
10k-step cap. The sibling ``online_mind2web`` increments before the same
branch; this pins webvoyager to that contract.

This file lives with WebVoyager, which owns the rejected-call budget branch.
The final test also pins the sibling Online Mind2Web host/container count
contract so the two singleton-browser envs stay aligned.
"""

from __future__ import annotations

import ast
import pathlib


def _step_body() -> ast.AsyncFunctionDef:
    src = pathlib.Path("lite/gym/envs/webharbor/webvoyager/main.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "step":
            return node
    raise AssertionError("webvoyager step() not found")


def _first_lineno(node: ast.AST, predicate) -> int | None:
    return min(
        (n.lineno for n in ast.walk(node) if predicate(n)),
        default=None,
    )


def test_budget_is_consumed_before_the_rejected_calls_early_return() -> None:
    step = _step_body()

    increment = _first_lineno(
        step,
        lambda n: isinstance(n, ast.AugAssign)
        and isinstance(n.target, ast.Attribute)
        and n.target.attr == "_step_count",
    )
    early_return = _first_lineno(
        step,
        lambda n: isinstance(n, ast.If)
        and isinstance(n.test, ast.UnaryOp)
        and isinstance(n.test.op, ast.Not)
        and isinstance(n.test.operand, ast.Name)
        and n.test.operand.id == "actions_to_send",
    )

    assert increment is not None, "step() no longer increments _step_count"
    assert early_return is not None, "the rejected-calls branch moved or was renamed"
    assert increment < early_return, (
        "_step_count must be incremented BEFORE the all-calls-rejected early "
        "return, or a model emitting only invalid calls never truncates"
    )


def test_the_rejected_calls_branch_reports_truncation() -> None:
    src = pathlib.Path("lite/gym/envs/webharbor/webvoyager/main.py").read_text()
    assert "truncated = self._step_count >= self.max_steps" in src, (
        "the rejected-calls early return must compute truncated, not hardcode False"
    )


def test_the_rejected_calls_branch_evaluates_when_it_truncates() -> None:
    """Consuming the last turn is not enough -- that turn must also be SCORED.

    The branch used to return ``truncated=True, reward=None``, which ends the
    episode with no score and discards the whole trajectory. Behavioural cover
    is in ``test_webharbor_webvoyager.py``; this pins the shape so the
    ``_evaluate`` call cannot be dropped again without a failure here.
    """
    step = _step_body()
    early_return = next(
        n for n in ast.walk(step)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.UnaryOp)
        and isinstance(n.test.op, ast.Not)
        and isinstance(n.test.operand, ast.Name)
        and n.test.operand.id == "actions_to_send"
    )
    evaluates = any(
        isinstance(n, ast.Attribute) and n.attr == "_evaluate"
        for n in ast.walk(early_return)
    )
    assert evaluates, (
        "the all-calls-rejected branch must call self._evaluate() when it "
        "truncates, or the final trajectory is thrown away unscored"
    )


def test_the_host_ships_the_count_it_owns_and_the_container_keeps_none() -> None:
    """One owner: the host counts turns, the ``/step`` body carries the count.

    The container never sees a turn we reject client-side, so a counter of its own
    could only lag -- and both consumers of the count (truncation and the
    model-facing ``[Turn n/max]`` banner) would then read different numbers.
    """
    for main_path, server_path in (
        (
            "lite/gym/envs/webharbor/webvoyager/main.py",
            "lite/gym/envs/webharbor/webvoyager/docker/server.py",
        ),
        (
            "lite/gym/envs/online_mind2web/main.py",
            "lite/gym/envs/online_mind2web/docker/server.py",
        ),
    ):
        main_src = pathlib.Path(main_path).read_text()
        server_src = pathlib.Path(server_path).read_text()
        assert '"step_count": self._step_count,' in main_src, (
            f"{main_path} must ship the host count in the /step body"
        )
        assert 'inst.step_count = int(body["step_count"])' in server_src, (
            f"{server_path} must adopt the shipped count, not keep its own"
        )
        assert "inst.step_count += 1" not in server_src, (
            f"{server_path} keeps a private counter that can only drift"
        )
        assert (
            'bool(resp.get("truncated")) or self._step_count >= self.max_steps'
            not in main_src
        ), (
            f"{main_path}: the truncation backstop is unnecessary once the "
            "container derives truncation from the shipped count"
        )


def test_sibling_online_mind2web_holds_the_same_contract() -> None:
    src = pathlib.Path("lite/gym/envs/online_mind2web/main.py").read_text()
    tree = ast.parse(src)
    step = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "step"
    )
    increment = _first_lineno(
        step,
        lambda n: isinstance(n, ast.AugAssign)
        and isinstance(n.target, ast.Attribute)
        and n.target.attr == "_step_count",
    )
    early_return = _first_lineno(
        step,
        lambda n: isinstance(n, ast.If)
        and isinstance(n.test, ast.UnaryOp)
        and isinstance(n.test.op, ast.Not)
        and isinstance(n.test.operand, ast.Name)
        and n.test.operand.id == "actions_to_send",
    )
    assert increment is not None and early_return is not None
    assert increment < early_return
