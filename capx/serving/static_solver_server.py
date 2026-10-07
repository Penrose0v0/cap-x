"""Stub HTTP server that returns a pre-written solver as if it were an LLM answer.

Wire this into the sweep pipeline exactly like ``claude_code_server.py``: the
sweep points its ``server_url`` here, and every ``POST /chat/completions``
returns the contents of ``heuristics/<task>/solver_s3.py`` wrapped in a
```python fence -- capx's ``_extract_code()`` then pulls that out and hands it
to ``_exec_user_code`` as if a model had written it.

Task dispatch is by ``request.model``: convention is ``heuristic/<task_short>``
(e.g. ``heuristic/cube_restack``), so callers just set ``MODEL`` per worker.
No auth, no state -- one file per task on disk is the whole "policy".
"""

from __future__ import annotations

import itertools
import re
import time
from pathlib import Path
from typing import Literal

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


_id_counter = itertools.count()


class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentItem] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "heuristic/cube_restack"
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]


def _resolve_solver_path(model: str, model_prefix: str, solver_root: Path) -> Path:
    if not model.startswith(model_prefix):
        raise HTTPException(
            status_code=400,
            detail=f"model must start with '{model_prefix}', got '{model}'",
        )
    task = model[len(model_prefix):]
    path = solver_root / task / "solver_s3.py"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"no solver at {path} for task '{task}'",
        )
    return path


_COMMON_IMPORT = re.compile(r"^from heuristics\.common import \*.*$", re.M)


def load_solver_source(solver_path: Path, solver_root: Path) -> str:
    """The solver file with ``from heuristics.common import *`` replaced by the helpers'
    source, so the sandbox receives one self-contained block (the sandbox has no
    ``heuristics`` package on its path, and a served policy should read as a single
    submission). Any ``from __future__`` line in common.py is dropped: it would no longer
    be at the top of the file."""
    source = solver_path.read_text(encoding="utf-8")
    if not _COMMON_IMPORT.search(source):
        return source
    common = (solver_root / "common.py").read_text(encoding="utf-8")
    common = "\n".join(line for line in common.splitlines() if not line.startswith("from __future__"))
    return _COMMON_IMPORT.sub(lambda _: "# --- heuristics/common.py (inlined) ---\n" + common + "\n# --- end common ---", source, count=1)


def create_app(solver_root: Path, model_prefix: str) -> FastAPI:
    app = FastAPI(title="Static Solver Stub", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming not supported")

        solver_path = _resolve_solver_path(request.model, model_prefix, solver_root)
        # Re-read on every request so the file is hot-swappable without restart.
        solver_src = load_solver_source(solver_path, solver_root)
        content = f"```python\n{solver_src}\n```\n"

        return ChatCompletionResponse(
            id=f"static-solver-{next(_id_counter)}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "solver_root": str(solver_root)}

    return app


def main(
    solver_root: str = "heuristics",
    host: str = "127.0.0.1",
    port: int = 8118,
    model_prefix: str = "heuristic/",
):
    """Start the static-solver stub.

    Requests are matched by ``model = "<model_prefix><task>"`` (default prefix
    ``heuristic/``) and served from ``<solver_root>/<task>/solver_s3.py``. The
    file is re-read on every request so you can edit a solver mid-sweep.
    """
    root = Path(solver_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"solver_root does not exist: {root}")
    app = create_app(solver_root=root, model_prefix=model_prefix)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
