"""
new_arch model API - one common ModelClient interface, one clean-room
implementation per backend (Ollama / OpenAI / HF Inference / HF Transformers
local), ALL real generation routed through LangChain
(ChatOllama / ChatOpenAI / ChatHuggingFace). Direct HTTP calls exist only for
lightweight availability checks (listing locally-pulled Ollama models,
pinging a server) - never for generation.

This is an independent, from-scratch implementation guided by (not
importing) the old project's src/llm/*.py, per the repository restriction
that new_arch must not depend at runtime on the old application code.

No API key is ever hardcoded. A missing key fails immediately with a clear,
actionable message - never guessed, never silently skipped in explicit-backend
mode.

New in new_arch (approved addition, did not exist in the old project): a
small bounded retry with backoff for transient failures (network hiccups,
malformed structured-output replies), since batch/eval runs are long and a
single dropped connection previously meant losing the whole run.
"""
import time
import logging

import requests

import config
from utils import parse_json_object

logger = logging.getLogger("new_arch.model_api")


class ModelAPIError(RuntimeError):
    """Raised when a backend cannot serve a request (not running, model not
    pulled, no API key, etc). Carries a clear, user-facing message. Never
    retried automatically - these are configuration problems, not transient
    network blips."""


def _normalize_messages(messages):
    """Accepts a plain string prompt, or a list of {"role","content"} dicts
    / (role, content) tuples. Returns a list of (role, content) tuples for
    LangChain's `.invoke()`. The caller's text is passed through verbatim -
    it is treated as data, never re-interpreted as instructions here (see
    safety.py for where the "treat as data" framing is applied at the prompt
    layer)."""
    if isinstance(messages, str):
        return [("human", messages)]
    normalized = []
    for m in messages:
        if isinstance(m, dict):
            normalized.append((m["role"], m["content"]))
        else:
            normalized.append(tuple(m))
    return normalized


class ModelClient:
    """Common interface every backend implements. Matches the shape:
    generate(messages, response_schema=None, temperature=None, max_tokens=None, **kwargs)."""
    backend_name = "base"

    def _check_or_raise(self):
        """Raise ModelAPIError with an actionable message if this backend
        cannot currently serve requests. Default: no-op (subclasses override)."""

    def _build_chat_model(self, temperature, max_tokens):
        raise NotImplementedError

    def is_available(self) -> bool:
        return True

    def generate(self, messages, response_schema=None, temperature=None, max_tokens=None, **kwargs):
        """messages: str or list of role/content pairs.
        response_schema: optional list of required top-level JSON keys - if
        given, the raw output is parsed as JSON; on parse failure or missing
        keys, one corrective follow-up turn is added and the call retried
        (bounded by config.MAX_RETRIES). Returns a str if response_schema is
        None, otherwise a dict (possibly {} after exhausting retries - same
        graceful-failure contract as utils.parse_json_object; callers must
        check for emptiness rather than assume success)."""
        temperature = config.DEFAULT_TEMPERATURE if temperature is None else temperature
        max_tokens = config.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        lc_messages = _normalize_messages(messages)

        self._check_or_raise()  # configuration problems fail fast, never retried

        last_exc = None
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                llm = self._build_chat_model(temperature, max_tokens)
                raw = llm.invoke(lc_messages).content.strip()
            except Exception as exc:
                last_exc = exc
                if attempt < config.MAX_RETRIES:
                    logger.warning("%s: transient failure (attempt %d/%d): %s",
                                   self.backend_name, attempt + 1, config.MAX_RETRIES + 1, exc)
                    time.sleep(config.RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                if response_schema is not None:
                    return {}
                raise ModelAPIError(
                    f"{self.backend_name} request failed after {attempt + 1} attempt(s): {exc}"
                ) from exc
            else:
                if response_schema is None:
                    return raw
                parsed = parse_json_object(raw)
                if parsed and all(k in parsed for k in response_schema):
                    return parsed
                if attempt < config.MAX_RETRIES:
                    lc_messages = lc_messages + [(
                        "human",
                        f"Your previous reply was not valid JSON with all required fields "
                        f"{list(response_schema)}. Reply again with ONLY a single valid JSON "
                        "object containing exactly those fields, no commentary.",
                    )]
                    continue
                return {}
        return {} if response_schema is not None else ""


# ---------------------------------------------------------------------------
# Ollama (default backend)
# ---------------------------------------------------------------------------
def list_local_models(host: str = None) -> list:
    """Lightweight availability-check-only listing of locally-pulled Ollama
    model tags (direct HTTP GET, no generation) - used to populate the
    Streamlit model dropdown."""
    host = host or config.OLLAMA_HOST
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


class OllamaClient(ModelClient):
    backend_name = "ollama"

    def __init__(self, model: str = None):
        self.model = model or config.DEFAULT_MODEL
        self.host = config.OLLAMA_HOST

    def _server_running(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def is_available(self) -> bool:
        if not self._server_running():
            return False
        installed = list_local_models(self.host)
        base_tag = self.model.split(":")[0]
        return any(self.model == m or base_tag == m.split(":")[0] for m in installed)

    def _check_or_raise(self):
        if not self._server_running():
            raise ModelAPIError(
                f"Ollama server not reachable at {self.host}. Install it from "
                "https://ollama.com and run `ollama serve`."
            )
        installed = list_local_models(self.host)
        base_tag = self.model.split(":")[0]
        if not any(self.model == m or base_tag == m.split(":")[0] for m in installed):
            raise ModelAPIError(f"Model '{self.model}' is not pulled locally. Run: ollama pull {self.model}")

    def _build_chat_model(self, temperature, max_tokens):
        from langchain_ollama import ChatOllama
        return ChatOllama(model=self.model, base_url=self.host, temperature=temperature, num_predict=max_tokens)


# ---------------------------------------------------------------------------
# OpenAI (optional fallback, never default)
# ---------------------------------------------------------------------------
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIClient(ModelClient):
    backend_name = "openai"

    def __init__(self, model: str = None):
        if not config.OPENAI_API_KEY:
            raise ModelAPIError(
                "OPENAI_API_KEY is not set. Set it as an environment variable before using "
                "the openai backend (optional fallback only). This project never inserts or "
                "guesses API keys for you."
            )
        self.model = model or DEFAULT_OPENAI_MODEL

    def is_available(self) -> bool:
        try:
            llm = self._build_chat_model(0.0, 4)
            llm.invoke([("human", "ping")])
            return True
        except Exception:
            return False

    def _build_chat_model(self, temperature, max_tokens):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.model, api_key=config.OPENAI_API_KEY, temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# HF Inference (remote, hosted)
# ---------------------------------------------------------------------------
def _split_model_and_provider(model: str):
    if ":" in model:
        repo_id, provider = model.rsplit(":", 1)
        return repo_id, provider
    return model, None


class HFInferenceClient(ModelClient):
    backend_name = "hf-inference"

    def __init__(self, model: str):
        if not config.HF_TOKEN:
            raise ModelAPIError(
                "HF_TOKEN is not set. Set it as an environment variable before using the "
                "hf-inference backend."
            )
        self.model_name, self.provider = _split_model_and_provider(model)

    def is_available(self) -> bool:
        try:
            llm = self._build_chat_model(0.0, 4)
            llm.invoke([("human", "ping")])
            return True
        except Exception:
            return False

    def _build_chat_model(self, temperature, max_tokens):
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint = HuggingFaceEndpoint(
            repo_id=self.model_name,
            huggingfacehub_api_token=config.HF_TOKEN,
            temperature=max(temperature, 1e-4),
            max_new_tokens=max_tokens,
        )
        return ChatHuggingFace(llm=endpoint)


class _BoundHFChatModel:
    """Thin per-call wrapper so ModelClient.generate()'s generic
    `llm.invoke(lc_messages)` (no extra kwargs - shared across all 4
    backends) still ends up passing the CALLER's requested temperature/
    max_tokens through to the underlying transformers pipeline for this
    specific call. Needed because langchain_huggingface's HuggingFacePipeline
    only reads generation kwargs from `pipeline_kwargs=` passed at
    `.invoke()` time (see HuggingFacePipeline._generate: `kwargs.get(
    "pipeline_kwargs", {})`) - the `pipeline_kwargs` stored on the langchain
    wrapper *object* itself is never actually read anywhere except
    `_identifying_params` (cosmetic/logging only). Wraps the SAME
    already-loaded ChatHuggingFace/pipeline - no reload, just closes over
    the per-call kwargs."""

    def __init__(self, chat_model, pipeline_kwargs: dict):
        self._chat_model = chat_model
        self._pipeline_kwargs = pipeline_kwargs

    def invoke(self, messages):
        return self._chat_model.invoke(messages, pipeline_kwargs=self._pipeline_kwargs)


# ---------------------------------------------------------------------------
# HF Transformers (local, in-process)
# ---------------------------------------------------------------------------
class HFTransformersClient(ModelClient):
    backend_name = "hf-transformers"

    def __init__(self, model: str):
        self.model_name = model
        self._chat_model = None
        self._load_error = None
        self._load()

    def _load(self):
        try:
            import torch
            from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

            quant_config = None
            if config.USE_4BIT_QUANTIZATION and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    quant_config = None

            model_kwargs = {"device_map": "auto"}
            if quant_config is not None:
                model_kwargs["quantization_config"] = quant_config
            elif torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16

            pipeline_llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_name, task="text-generation",
                model_kwargs=model_kwargs,
                # return_full_text=False is critical, not cosmetic: transformers'
                # text-generation pipeline defaults to returning prompt+completion
                # concatenated, and langchain_huggingface only strips the prompt
                # back off if skip_prompt=True is explicitly passed at call time
                # (it isn't, anywhere in this codebase). Without this, every
                # agent's raw output is [the entire prompt, including its own
                # embedded JSON-schema example] + [the model's real answer] as
                # one string - parse_json_object() then finds the wrong (or no)
                # JSON object and every single call "fails to parse", regardless
                # of how good the model actually is. Found via live testing on
                # Colab: 100% parse failure across every agent, every call, with
                # Qwen2.5-32B-Instruct - a model far too strong for that to be a
                # model-quality problem.
                pipeline_kwargs={"max_new_tokens": config.DEFAULT_MAX_TOKENS, "return_full_text": False},
            )
            self._chat_model = ChatHuggingFace(llm=pipeline_llm)
        except Exception as exc:
            self._load_error = (
                f"Failed to load local model '{self.model_name}': {exc}. Common causes: "
                "missing/incompatible bitsandbytes build, insufficient VRAM, missing "
                "CUDA-enabled torch build, or no internet for the first-time weight "
                "download. Try a smaller model or a different backend."
            )

    def is_available(self) -> bool:
        return self._load_error is None and self._chat_model is not None

    def _check_or_raise(self):
        if self._load_error is not None:
            raise ModelAPIError(self._load_error)

    def _build_chat_model(self, temperature, max_tokens):
        # Previously returned self._chat_model unconditionally, silently ignoring
        # both arguments - every agent (Judge included, which requests up to 900
        # tokens for its larger JSON payload) was always capped at the load-time
        # default of config.DEFAULT_MAX_TOKENS=512 regardless of what it asked
        # for, risking truncated/invalid JSON on top of the return_full_text bug
        # above. _BoundHFChatModel threads the real per-call values through
        # without reloading the (multi-GB) model.
        pipeline_kwargs = {
            "max_new_tokens": max_tokens, "return_full_text": False,
            "do_sample": bool(temperature and temperature > 0),
            # Soft nudge against the degenerate repetition-loop failure mode
            # (the same sentence repeated verbatim several times in a row)
            # observed on low-resource-language output, e.g. Tamil, where
            # the model's next-token distribution is less peaked and more
            # prone to getting stuck. NOTE: no_repeat_ngram_size was tried
            # here too and reverted - it's a HARD ban on any repeated
            # 3-token sequence, which every JSON schema call violates by
            # necessity (repeated key names/delimiters across list items,
            # e.g. every candidate_personas entry repeating '"role_type":
            # "'), so it broke JSON parsing for every single agent at once,
            # including case_analysis_agent which had never failed before.
            # repetition_penalty is a proportional discouragement, not a
            # hard ban, so it doesn't have this failure mode.
            "repetition_penalty": 1.15,
        }
        if temperature and temperature > 0:
            pipeline_kwargs["temperature"] = temperature
        return _BoundHFChatModel(self._chat_model, pipeline_kwargs)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def _build_ollama(model):
    return OllamaClient(model)


def _build_hf_inference(model):
    return HFInferenceClient(model)


def _build_hf_transformers(model):
    return HFTransformersClient(model)


def _build_openai(model):
    return OpenAIClient(model if model and "/" not in model else None)


_BUILDERS = {
    "ollama": _build_ollama,
    "hf-inference": _build_hf_inference,
    "hf-transformers": _build_hf_transformers,
    "openai": _build_openai,
}


def get_client(backend: str = None, model: str = None) -> ModelClient:
    """backend=None -> config.DEFAULT_BACKEND ("ollama"). model=None ->
    config.DEFAULT_MODEL. backend="auto" tries ollama -> hf-inference ->
    hf-transformers -> openai (only if OPENAI_API_KEY set) in order, raising
    with the full attempts log if nothing works. An explicit backend choice
    never silently falls back - if it's unavailable, this raises immediately."""
    backend = backend or config.DEFAULT_BACKEND
    model = model or config.DEFAULT_MODEL

    if backend not in config.SUPPORTED_BACKENDS:
        raise ModelAPIError(f"Unknown backend '{backend}'. Supported: {config.SUPPORTED_BACKENDS}")

    if backend != "auto":
        client = _BUILDERS[backend](model)
        if not client.is_available():
            # Surface the real reason (e.g. HFTransformersClient._load_error - a CUDA
            # OOM, missing bitsandbytes, gated-model auth, etc.) instead of a generic
            # message - found via live testing that the generic message alone gave no
            # way to tell "genuinely broken" apart from "GPU already holding another
            # copy of the model in a different process".
            try:
                client._check_or_raise()
            except ModelAPIError:
                raise
            raise ModelAPIError(f"Backend '{backend}' with model '{model}' is not available (no fallback attempted).")
        return client

    attempts = []
    for name in ("ollama", "hf-inference", "hf-transformers"):
        try:
            client = _BUILDERS[name](model)
            if client.is_available():
                logger.info("auto backend selection: using '%s'", name)
                return client
            attempts.append(f"{name}: built but not available")
        except Exception as exc:
            attempts.append(f"{name}: {exc}")
    if config.OPENAI_API_KEY:
        try:
            client = _build_openai(model)
            if client.is_available():
                logger.info("auto backend selection: using 'openai'")
                return client
            attempts.append("openai: built but not available")
        except Exception as exc:
            attempts.append(f"openai: {exc}")
    else:
        attempts.append("openai: OPENAI_API_KEY not set, skipped")

    raise ModelAPIError("No backend available.\n" + "\n".join(attempts))
