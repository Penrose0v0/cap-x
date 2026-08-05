"""Proxy server that fronts the `claude` CLI's non-interactive `-p` mode with an
OpenAI-chat-completions-shaped HTTP endpoint, so capx's existing query_model()/
server_url plumbing (capx/llm/client.py) can drive it exactly like a local vLLM
server -- point a LaunchArgs.server_url at this, pick any model name that isn't
in GPT_MODELS/OPENROUTER_MODELS/CLAUDE_MODELS/OSS_MODELS, and it hits the same
generic {model, temperature, max_tokens, messages} payload path vLLM does.

Auth is via CLAUDE_CODE_OAUTH_TOKEN (a long-lived token tied to your Claude
subscription, generated with `claude setup-token`), not ANTHROPIC_API_KEY --
usage is billed against your Pro/Max plan, not pay-per-token API credits.
`claude -p` treats OAuth and a raw API key as mutually exclusive auth paths
(and `--bare` mode refuses to read OAuth at all), so this only works with the
OAuth token.

`claude -p` is not a stateless chat-completions endpoint: it's a full agentic
CLI that starts a fresh session per invocation. This proxy disables all tools
(`--tools ""`) so it can only return text, skips CLAUDE.md/hooks/skills
(`--safe-mode`) and session persistence, and flattens the incoming messages
list into one text block (with a `--system-prompt` for any system message)
since there's no real multi-turn chat state to replay between calls.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
    model: str = "claude-code/sonnet"
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


def _load_oauth_token(key_file: str) -> str:
    """Load a CLAUDE_CODE_OAUTH_TOKEN from a file (one token, first non-comment line)."""
    path = Path(key_file)
    if not path.exists():
        raise FileNotFoundError(f"OAuth token file not found: {key_file}")
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise ValueError(f"No token found in {key_file}")


def _message_text(content: str | list[ContentItem] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append(item.text)
        elif item.type == "image_url":
            logger.warning(
                "claude_code_server: dropping image_url content -- "
                "claude -p has no inline image input"
            )
    return "\n".join(parts)


def _build_prompt(messages: list[Message]) -> tuple[str | None, str]:
    """Split messages into (system_prompt, flattened user/assistant transcript)."""
    system_parts = []
    turn_parts = []
    for m in messages:
        text = _message_text(m.content)
        if m.role == "system":
            system_parts.append(text)
        else:
            label = "User" if m.role == "user" else "Assistant"
            turn_parts.append(f"[{label}]\n{text}")
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    user_prompt = "\n\n".join(turn_parts)
    return system_prompt, user_prompt


async def _run_claude_p(
    prompt: str,
    system_prompt: str | None,
    model: str,
    oauth_token: str,
    timeout: float,
) -> str:
    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--model", model,
        "--tools", "",
        "--safe-mode",
        "--no-session-persistence",
    ]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": oauth_token},
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"claude -p timed out after {timeout}s")

    if proc.returncode != 0:
        raise RuntimeError(
            f"claude -p exited {proc.returncode}: {stderr.decode('utf-8', 'replace')}"
        )

    body = json.loads(stdout.decode("utf-8"))
    if body.get("is_error"):
        raise RuntimeError(f"claude -p error: {body.get('result')}")
    return body["result"]


def create_app(oauth_token: str, timeout: float, model_prefix: str) -> FastAPI:
    app = FastAPI(title="Claude Code CLI Proxy", version="1.0.0")

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
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported by the claude-code-cli proxy.",
            )

        model = request.model
        if model.startswith(model_prefix):
            model = model[len(model_prefix):]

        system_prompt, user_prompt = _build_prompt(request.messages)

        try:
            content = await _run_claude_p(
                user_prompt, system_prompt, model, oauth_token, timeout
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return ChatCompletionResponse(
            id=f"claude-code-{next(_id_counter)}",
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
        return {"status": "ok"}

    return app


def main(
    key_file: str = ".claudeoauthtoken",
    oauth_token: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8112,
    timeout: float = 900.0,
    model_prefix: str = "claude-code/",
):
    """Start the Claude Code CLI proxy server.

    Reads a CLAUDE_CODE_OAUTH_TOKEN from --oauth-token or from the key file (one
    token per line, `claude setup-token` to generate one). Request `model` is
    expected as "<model_prefix><alias>" (default prefix "claude-code/"), e.g.
    "claude-code/opus" or "claude-code/sonnet" -- the prefix is stripped and the
    rest is passed straight through to `claude -p --model`.
    """
    if oauth_token is None:
        oauth_token = _load_oauth_token(key_file)
        logger.info(f"Loaded OAuth token from {key_file}")

    app = create_app(oauth_token=oauth_token, timeout=timeout, model_prefix=model_prefix)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
