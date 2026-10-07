"""Every hard-coded solver, as the stub server serves it (shared helpers inlined), must be a
single compilable code block and define main()."""

import ast
import re
from pathlib import Path

import pytest

from capx.serving.static_solver_server import load_solver_source

ROOT = Path(__file__).resolve().parents[1] / "heuristics"
SOLVERS = sorted(ROOT.glob("*/solver_s3.py"))


@pytest.mark.parametrize("solver_path", SOLVERS, ids=[p.parent.name for p in SOLVERS])
def test_served_solver_compiles(solver_path: Path) -> None:
    source = load_solver_source(solver_path, ROOT)
    assert not re.search(r"^from heuristics\.common import", source, re.M)
    tree = ast.parse(source, filename=str(solver_path))
    compile(tree, str(solver_path), "exec")
    top_level_defs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "main" in top_level_defs
