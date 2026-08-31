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

    def generate_batch(self, prompts: list, response_schema=None, temperature=None, max_tokens=None) -> list:
        """Batched sibling of generate(): given N independent prompts (no
        data dependency between them), returns N results in the same order.
        Built on LangChain's own .batch() - the framework-provided way to do
        this safely (concurrent I/O for remote backends; whatever batching
        the underlying model/pipeline supports for local ones) - so this is
        never worse than calling generate() N times in a loop even in the
        worst case, and may be substantially faster. Added to parallelize
        the per-claim query-formulation calls in evidence_verifier.py, which
        were previously N sequential single-item generate() calls per round
        for no real reason - the claims are independent of each other.

        Any single item that fails to come back as valid JSON falls back to
        a full single-item generate() call (with its own corrective-retry
        turn) rather than re-implementing that retry loop for the batch
        path - keeps this addition small and reuses already-proven logic."""
        if not prompts:
            return []
        temperature = config.DEFAULT_TEMPERATURE if temperature is None else temperature
        max_tokens = config.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        self._check_or_raise()

        lc_batches = [_normalize_messages(p) for p in prompts]
        # _build_chat_model() is INSIDE this try/except, not just llm.batch() -
        # found via a real test failure (test_fast_track_explicit_hate_smoke)
        # that a test double not implementing _build_chat_model() crashed
        # generate_batch() with an uncaught NotImplementedError instead of
        # falling back, defeating the whole point of this being a safe,
        # never-worse-than-the-loop fallback path. Any failure in EITHER step
        # now degrades to sequential generate() the same way.
        try:
            llm = self._build_chat_model(temperature, max_tokens)
            raw_results = llm.batch(lc_batches)
        except Exception as exc:
            logger.warning(
                "%s: batch call failed (%s) - falling back to sequential generate() for all %d prompts",
                self.backend_name, exc, len(prompts),
            )
            return [self.generate(p, response_schema=response_schema, temperature=temperature, max_tokens=max_tokens)
                    for p in prompts]

        results = []
        for prompt, raw_msg in zip(prompts, raw_results):
            raw = (getattr(raw_msg, "content", "") or "").strip()
            if response_schema is None:
                results.append(raw)
                continue
            parsed = parse_json_object(raw)
            if parsed and all(k in parsed for k in response_schema):
                results.append(parsed)
            else:
                results.append(self.generate(prompt, response_schema=response_schema,
                                              temperature=temperature, max_tokens=max_tokens))
        return results


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
        # base_url stays None (real OpenAI API) unless OPENAI_BASE_URL is
        # explicitly set - see config.py. Lets this same client point at a
        # local OpenAI-compatible server (e.g. vLLM) for a latency
        # validation test without touching any actual generation run.
        return ChatOpenAI(model=self.model, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL,
                           temperature=temperature, max_tokens=max_tokens)


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


class _SimpleMessage:
    """Minimal stand-in for a LangChain AIMessage - just enough for
    ModelClient.generate_batch()'s `getattr(raw_msg, "content", "")` read.
    Used by _BoundHFChatModel.batch()'s direct-pipeline path, which doesn't
    go through ChatHuggingFace and so has no real AIMessage to return."""
    def __init__(self, content: str):
        self.content = content


# LangChain's internal role names ("human"/"ai") -> the role names HF chat
# templates expect ("user"/"assistant"). Every current caller sends a plain
# string prompt (see prompts.py - single-turn, role "human" only), but this
# stays a mapping rather than a hardcoded "user" so a future multi-turn
# caller isn't silently mislabeled.
_HF_CHAT_ROLE_MAP = {"human": "user", "user": "user", "ai": "assistant",
                      "assistant": "assistant", "system": "system"}


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

    def batch(self, list_of_messages):
        """True batched inference - genuinely different from calling invoke()
        N times. ChatHuggingFace has no .batch() of its own (that's why
        ModelClient.generate_batch()'s default `llm.batch(...)` call fails
        for this backend and falls back to sequential), so this bypasses it
        and calls the underlying HuggingFacePipeline's LLM-level .generate()
        directly with a LIST of prompt strings + a batch_size, which is what
        actually gets transformers' pipeline to run them as one batched
        forward pass instead of N separate ones. Formats each prompt with
        the SAME chat template ChatHuggingFace uses internally, so a single
        item's output should match what invoke() would have produced for it.

        Batched causal-LM generation requires left-padding and a pad token
        (see _fix_tokenizer_for_batching, applied once at model load) -
        without that, this doesn't crash, it silently generates malformed
        output. Do not remove that call."""
        llm = self._chat_model.llm
        tokenizer = self._chat_model.tokenizer
        prompts = []
        for messages in list_of_messages:
            chat = [{"role": _HF_CHAT_ROLE_MAP.get(role, "user"), "content": content} for role, content in messages]
            prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

        batch_kwargs = dict(self._pipeline_kwargs)
        batch_kwargs["batch_size"] = min(len(prompts), 8)
        result = llm.generate(prompts, pipeline_kwargs=batch_kwargs)
        return [_SimpleMessage(gens[0].text if gens else "") for gens in result.generations]


# ---------------------------------------------------------------------------
# HF Transformers (local, in-process)
# ---------------------------------------------------------------------------
def _is_mistral_model(model_name: str) -> bool:
    return "mistral" in (model_name or "").lower()


def _hf_transformers_tokenizer_kwargs(model_name: str, hf_token: str = None) -> dict:
    """Extra kwargs for AutoTokenizer.from_pretrained() ONLY - split into
    its own pure, directly-testable function (no model load needed).
    hf_token defaults to config.HF_TOKEN (an explicit override param
    exists only for tests).

    fix_mistral_regex=True: Mistral-family tokenizers (confirmed live:
    gghfez/Mistral-Small-3.2-24B-Instruct-hf) ship a known-buggy split
    regex that transformers warns about at load time ("incorrect regex
    pattern... set fix_mistral_regex=True to fix this"). This is a
    TOKENIZER-ONLY constructor argument - passing it to
    AutoModelForCausalLM.from_pretrained() crashes with
    "MistralForCausalLM.__init__() got an unexpected keyword argument
    'fix_mistral_regex'" (confirmed via a real Colab run - the original
    version of this fix shared one kwargs dict between tokenizer and model
    loading via HuggingFacePipeline.from_model_id, which doesn't support
    giving them different kwargs). Scoped to Mistral models only
    (case-insensitive name match).

    token: reused from config.HF_TOKEN (config.py's own
    os.environ.get("HF_TOKEN", "") - the project's established never-
    hardcoded pattern, see HFInferenceClient above), only included when
    actually set, so a public model loads exactly as before - a missing
    `token` kwarg is treated as unauthenticated/public access, not an
    error. Never logged or printed - it only ever exists as a dict value
    passed straight into from_pretrained()."""
    hf_token = hf_token if hf_token is not None else config.HF_TOKEN
    kwargs = {}
    if _is_mistral_model(model_name):
        kwargs["fix_mistral_regex"] = True
    if hf_token:
        kwargs["token"] = hf_token
    return kwargs


def _hf_transformers_model_auth_kwargs(hf_token: str = None) -> dict:
    """Extra kwargs for AutoModelForCausalLM.from_pretrained() ONLY - just
    the auth token, reused from config.HF_TOKEN by default. Deliberately
    does NOT include fix_mistral_regex (see
    _hf_transformers_tokenizer_kwargs's docstring) - that is a tokenizer-
    only constructor argument the model class does not accept."""
    hf_token = hf_token if hf_token is not None else config.HF_TOKEN
    return {"token": hf_token} if hf_token else {}


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
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
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
            # Auth only - deliberately NOT fix_mistral_regex, which
            # AutoModelForCausalLM.from_pretrained() does not accept (see
            # _hf_transformers_tokenizer_kwargs's docstring for the real
            # crash this caused when both loads shared one kwargs dict).
            model_kwargs.update(_hf_transformers_model_auth_kwargs())

            tokenizer_kwargs = _hf_transformers_tokenizer_kwargs(self.model_name)

            # Built manually (tokenizer and model loaded separately, each
            # with its own kwargs, then wired into a plain transformers
            # pipeline) rather than via HuggingFacePipeline.from_model_id -
            # that convenience method loads the tokenizer and the model
            # with the exact same shared kwargs dict (confirmed by reading
            # its source: both calls are `X.from_pretrained(model_id,
            # **_model_kwargs)`), so it has no way to give the tokenizer a
            # kwarg the model class doesn't accept, which is exactly what
            # fix_mistral_regex is.
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

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
            pipeline_kwargs = {"max_new_tokens": config.DEFAULT_MAX_TOKENS, "return_full_text": False}
            # model_kwargs is re-passed here (not just used for the
            # from_pretrained() call above) to mirror HuggingFacePipeline.
            # from_model_id's own internal call shape exactly - transformers'
            # pipeline() only actually consults model_kwargs when it has to
            # load the model itself from a checkpoint string, which isn't
            # the case here since `model=model` is already a live object, so
            # this is inert but kept for parity with the upstream library.
            raw_pipeline = hf_pipeline(
                task="text-generation", model=model, tokenizer=tokenizer,
                device=None, batch_size=4, model_kwargs=model_kwargs, **pipeline_kwargs,
            )
            pipeline_llm = HuggingFacePipeline(
                pipeline=raw_pipeline, model_id=self.model_name,
                model_kwargs=model_kwargs, pipeline_kwargs=pipeline_kwargs, batch_size=4,
            )
            self._chat_model = ChatHuggingFace(llm=pipeline_llm)
            self._fix_tokenizer_for_batching()
        except Exception as exc:
            self._load_error = (
                f"Failed to load local model '{self.model_name}': {exc}. Common causes: "
                "missing/incompatible bitsandbytes build, insufficient VRAM, missing "
                "CUDA-enabled torch build, or no internet for the first-time weight "
                "download. Try a smaller model or a different backend."
            )

    def _fix_tokenizer_for_batching(self):
        """Batched generation on a decoder-only (causal LM) model requires
        LEFT padding and a real pad token - right-padding (the tokenizer
        default for most models) puts padding BEFORE the model's own
        continuation for shorter sequences in the batch, which the model
        then generates from as if it were real context, silently producing
        garbled/truncated output for anything but the longest prompt in the
        batch. This is called once at load time (not per-batch-call) since
        it mutates the tokenizer's own config, and _BoundHFChatModel.batch()
        and the single-item invoke() path share the exact same tokenizer/
        pipeline instance - fixing it once here is sufficient for both, and
        padding side is a no-op for a batch of size 1 anyway.
        Defensive best-effort: if a future langchain_huggingface version
        changes where the tokenizer lives, this silently no-ops rather than
        blocking model load - batch() will then surface any real problem
        itself (loudly, via its own exception -> ModelClient.generate_batch's
        fallback), rather than this failing model load for an unrelated reason."""
        try:
            tokenizer = self._chat_model.llm.pipeline.tokenizer
        except AttributeError:
            return
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

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
