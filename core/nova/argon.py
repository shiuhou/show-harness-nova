"""Stateless Argon provider with explicit API selection and no automatic retry."""

from __future__ import annotations

import base64
import hashlib
import math
import os
import queue
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal, NoReturn, Sequence
from urllib.parse import urlsplit

import jsonschema
from google import genai
from google.genai import types

from core.nova.contracts import RGB, png_bytes, strict_json
from core.vlm.vlm_client import VLMResponse

ARGON_MODEL = "gemini-4-argon"
SYSTEM_INSTRUCTION = (
    "You are the visual reasoning model for a supervised Nova research harness. "
    "Use only the requested response contract. No device control tools are available. "
    "If a safe action cannot be determined, use the permitted end token."
)


def local_settings(path: Path) -> dict[str, str]:
    """Read data only, never evaluate shell commands or print credential values."""
    allowed = {"GEMINI_API_KEY", "GOOGLE_GEMINI_BASE_URL", "GEMINI_MODEL"}
    result = {name: os.environ[name] for name in allowed if name in os.environ}
    if path.exists():
        if path.stat().st_mode & 0o077:
            raise ValueError("local credential file must have mode 0600")
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            name, sep, value = line.partition("=")
            if not sep or name not in allowed:
                raise ValueError("invalid local provider setting name")
            parts = shlex.split(value, comments=True)
            if len(parts) > 1:
                raise ValueError("provider setting must contain one literal value")
            result[name] = parts[0] if parts else ""
    if result.get("GEMINI_MODEL", ARGON_MODEL) != ARGON_MODEL:
        raise ValueError("Nova baseline requires gemini-4-argon; model substitution is disabled")
    if not result.get("GEMINI_API_KEY", "").strip():
        raise ValueError("fill GEMINI_API_KEY in the ignored local credential file")
    base = result.get("GOOGLE_GEMINI_BASE_URL", "")
    if base:
        url = urlsplit(base)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("provider base URL must be an explicit HTTPS URL without credentials")
    return result


def sdk_client(settings: dict[str, str], timeout_s: float) -> genai.Client:
    options = types.HttpOptions(
        timeout=math.ceil(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)
    )
    if settings.get("GOOGLE_GEMINI_BASE_URL"):
        options.base_url = settings["GOOGLE_GEMINI_BASE_URL"]
    client = genai.Client(api_key=settings["GEMINI_API_KEY"], http_options=options)
    # In pinned SDK 2.30.1, the base client rewrites attempts=0 to 1, then the
    # Interactions bridge treats it as a RETRY count. Disable retries through
    # the generated resource's configuration as well. Wire tests cover both
    # HTTP and transport failures; remove this workaround when the SDK fixes it.
    client.interactions.sdk_configuration.retry_config = None
    return client


class ArgonClient:
    def __init__(
        self,
        sdk: Any,
        *,
        timeout_s: float,
        max_tokens: int = 512,
        request_sink: Callable[[dict[str, Any]], None] | None = None,
        response_sink: Callable[[dict[str, Any]], None] | None = None,
        api: Literal["interactions", "generate-content"] = "interactions",
    ) -> None:
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 30:
            raise ValueError("Argon timeout must be in (0,30] seconds")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("bounded output token budget required")
        if api not in ("interactions", "generate-content"):
            raise ValueError("select one supported API explicitly; no endpoint fallback")
        self.api = api
        self.sdk, self.timeout_s, self.max_tokens = sdk, timeout_s, max_tokens
        self.request_sink = request_sink
        self.response_sink = response_sink
        self._lock = threading.Lock()

    def _reject(self, reason: str) -> NoReturn:
        if self.response_sink:
            self.response_sink({"api": self.api, "outcome": "rejected", "reason": reason})
        raise RuntimeError(reason)

    def _complete(
        self,
        prompt: str,
        agentview_image: RGB,
        wrist_image: RGB | None,
        schema: dict[str, Any] | None,
        max_tokens: int | None,
        temperature: float | None,
        agentview_label: str,
        wrist_label: str,
    ) -> VLMResponse:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("one pending Argon request only")
        inputs: list[dict[str, Any]] = []
        parts: list[types.Part] = []
        manifest = []
        try:
            for label, rgb in ((agentview_label, agentview_image), (wrist_label, wrist_image)):
                if rgb is None:
                    continue
                png = png_bytes(rgb)
                parts.extend(
                    [
                        types.Part.from_text(text=label),
                        types.Part.from_bytes(data=png, mime_type="image/png"),
                    ]
                )
                inputs.extend(
                    [
                        {"type": "text", "text": label},
                        {
                            "type": "image",
                            "data": base64.b64encode(png).decode(),
                            "mime_type": "image/png",
                        },
                    ]
                )
                manifest.append(
                    {
                        "label": label,
                        "sha256": hashlib.sha256(png).hexdigest(),
                        "mime_type": "image/png",
                        "shape": list(rgb.shape),
                    }
                )
            inputs.append({"type": "text", "text": prompt})
            parts.append(types.Part.from_text(text=prompt))
            budget = self.max_tokens if max_tokens is None else max_tokens
            if type(budget) is not int or not 1 <= budget <= self.max_tokens:
                raise ValueError("request exceeds provider token budget")
            generation: dict[str, Any] = {"thinking_level": "medium", "max_output_tokens": budget}
            if temperature is not None:
                if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                    raise ValueError("invalid temperature")
                generation["temperature"] = temperature
            request: dict[str, Any] = {
                "model": ARGON_MODEL,
                "input": inputs,
                "system_instruction": SYSTEM_INSTRUCTION,
                "generation_config": generation,
                "store": False,
                "stream": False,
                "background": False,
                "timeout": self.timeout_s,
            }
            if schema is not None:
                jsonschema.Draft202012Validator.check_schema(schema)
                request["response_format"] = {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                }
            public_request = {key: value for key, value in request.items() if key != "input"}
            if self.api == "generate-content":
                # This gateway supports generateContent but not Interactions. The
                # caller selects the API before sending; never switch on failure.
                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
                    max_output_tokens=budget,
                    temperature=temperature,
                    candidate_count=1,
                    response_mime_type="application/json" if schema is not None else "text/plain",
                    response_json_schema=schema,
                )
                request = {
                    "model": ARGON_MODEL,
                    "contents": types.Content(role="user", parts=parts),
                    "config": config,
                }
                public_request = {
                    "model": ARGON_MODEL,
                    "config": config.model_dump(mode="json", exclude_none=True),
                }
            public_request["api"] = self.api
            public_request.update(prompt=prompt, images=manifest)
            if self.request_sink:
                self.request_sink(public_request)
        except BaseException:
            self._lock.release()
            raise
        results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def call() -> None:
            try:
                response = (
                    self.sdk.interactions.create(**request)
                    if self.api == "interactions"
                    else self.sdk.models.generate_content(**request)
                )
                results.put((True, response))
            except BaseException as exc:
                # Provider errors can contain URLs/headers. Preserve error type only.
                results.put((False, type(exc).__name__))
            finally:
                self._lock.release()

        started = time.monotonic()
        threading.Thread(target=call, name="argon-one-request", daemon=True).start()
        try:
            ok, interaction = results.get(timeout=self.timeout_s)
        except queue.Empty:
            if self.response_sink:
                self.response_sink({"api": self.api, "outcome": "deadline_exceeded"})
            raise TimeoutError("Argon request deadline exceeded; late response discarded") from None
        if not ok:
            if self.response_sink:
                self.response_sink(
                    {"api": self.api, "outcome": "transport_failed", "error_type": interaction}
                )
            raise RuntimeError(f"Argon request failed ({interaction}); no retry") from None
        # Preserve bounded final output even when validation below rejects it.
        # Never dump SDK objects (headers/URLs), tool arguments or thought parts.
        if self.api == "interactions":
            audit_text = interaction.output_text
            audit_status = str(interaction.status)
            audit_model = str(interaction.model)
        else:
            audit_candidates = interaction.candidates or []
            audit_text = "".join(
                p.text or ""
                for c in audit_candidates[:1]
                for p in (c.content.parts if c.content and c.content.parts else [])
                if not p.thought
            )
            audit_status = ",".join(str(c.finish_reason) for c in audit_candidates[:2])
            if interaction.prompt_feedback and interaction.prompt_feedback.block_reason:
                audit_status += ":" + str(interaction.prompt_feedback.block_reason)
            audit_model = interaction.model_version or "NOT_REPORTED"
        if self.response_sink:
            raw_bytes = audit_text.encode() if isinstance(audit_text, str) else b""
            self.response_sink(
                {
                    "api": self.api,
                    "status": audit_status[:256],
                    "model_version": audit_model[:256],
                    "raw_text": raw_bytes[:65536].decode("utf-8", errors="replace"),
                    "truncated_for_evidence": len(raw_bytes) > 65536,
                    "latency_s": time.monotonic() - started,
                }
            )
        version = None
        if self.api == "interactions":
            if interaction.status != "completed" or interaction.model != ARGON_MODEL:
                self._reject("Argon returned incomplete/refused output or a different model")
            raw = interaction.output_text
            usage = getattr(interaction, "usage", None)
        else:
            candidates = interaction.candidates or []
            if (
                len(candidates) != 1
                or candidates[0].finish_reason != types.FinishReason.STOP
                or interaction.prompt_feedback
                and interaction.prompt_feedback.block_reason
            ):
                self._reject("Argon returned incomplete/refused output")
            version = interaction.model_version
            if version and version != ARGON_MODEL and not version.startswith(ARGON_MODEL + "-"):
                self._reject("Argon returned a different model version")
            content = candidates[0].content
            output_parts = content.parts if content and content.parts else []
            if any(
                not p.thought and (p.text is None or p.function_call is not None)
                for p in output_parts
            ):
                self._reject("Argon returned non-text output")
            raw = "".join(p.text or "" for p in output_parts if not p.thought)
            usage = interaction.usage_metadata
        if not isinstance(raw, str) or not raw.strip() or len(raw.encode()) > 65536:
            self._reject("Argon response is empty or exceeds bounded response size")
        payload: dict[str, Any] = {
            "model": ARGON_MODEL,
            "model_version": version or "NOT_REPORTED",
            "api": self.api,
            "status": "completed",
            "latency_s": time.monotonic() - started,
            "request": public_request,
        }
        payload["usage"] = (
            usage.model_dump(mode="json")
            if usage is not None and hasattr(usage, "model_dump")
            else usage
        )
        token = ""
        if schema is not None:
            try:
                obj = strict_json(raw)
                jsonschema.Draft202012Validator(schema).validate(obj)
            except (ValueError, jsonschema.ValidationError):
                self._reject("Argon response violates JSON contract; no fallback")
            payload["json"] = obj
            if isinstance(obj, dict) and isinstance(obj.get("decision"), str):
                token = obj["decision"]
        return VLMResponse(token, raw, payload)

    def complete_json(
        self,
        prompt: str,
        agentview_image: RGB,
        wrist_image: RGB | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        chat_template_kwargs: Any = None,
        debug: bool = False,
        agentview_label: str = "Image A: front RGB",
        wrist_label: str = "Image B: wrist RGB",
        image_detail: Any = None,
    ) -> VLMResponse:
        if schema is None:
            raise ValueError("Argon JSON requests require an explicit schema")
        return self._complete(
            prompt,
            agentview_image,
            wrist_image,
            schema,
            max_tokens,
            temperature,
            agentview_label,
            wrist_label,
        )

    def complete_text(
        self,
        prompt: str,
        agentview_image: RGB,
        wrist_image: RGB | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        chat_template_kwargs: Any = None,
        debug: bool = False,
        agentview_label: str = "Image A: front RGB",
        wrist_label: str = "Image B: wrist RGB",
        image_detail: Any = None,
        strip_reasoning: bool = True,
    ) -> VLMResponse:
        return self._complete(
            prompt,
            agentview_image,
            wrist_image,
            None,
            max_tokens,
            temperature,
            agentview_label,
            wrist_label,
        )

    def complete_token(
        self,
        prompt: str,
        allowed_tokens: Sequence[str],
        agentview_image: RGB,
        wrist_image: RGB | None = None,
        chat_template_kwargs: Any = None,
        debug: bool = False,
        agentview_label: str = "Image A: front RGB",
        wrist_label: str = "Image B: wrist RGB",
        image_detail: Any = None,
    ) -> VLMResponse:
        schema = {
            "type": "object",
            "properties": {"decision": {"type": "string", "enum": list(allowed_tokens)}},
            "required": ["decision"],
            "additionalProperties": False,
        }
        return self._complete(
            prompt, agentview_image, wrist_image, schema, None, 0.0, agentview_label, wrist_label
        )
