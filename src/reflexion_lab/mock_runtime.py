from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import urllib.error
import urllib.request

from pydantic import ValidationError

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))


@dataclass
class RuntimeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_CURRENT_USAGE = RuntimeUsage()


def runtime_mode() -> str:
    return os.getenv("REFLEXION_RUNTIME", "ollama").strip().lower()


def reset_runtime_usage() -> None:
    global _CURRENT_USAGE
    _CURRENT_USAGE = RuntimeUsage()


def consume_runtime_usage() -> RuntimeUsage:
    usage = RuntimeUsage(
        prompt_tokens=_CURRENT_USAGE.prompt_tokens,
        completion_tokens=_CURRENT_USAGE.completion_tokens,
        latency_ms=_CURRENT_USAGE.latency_ms,
        calls=_CURRENT_USAGE.calls,
    )
    reset_runtime_usage()
    return usage


def _record_ollama_usage(payload: dict) -> None:
    _CURRENT_USAGE.prompt_tokens += int(payload.get("prompt_eval_count") or 0)
    _CURRENT_USAGE.completion_tokens += int(payload.get("eval_count") or 0)
    total_duration = int(payload.get("total_duration") or 0)
    if total_duration:
        _CURRENT_USAGE.latency_ms += max(1, round(total_duration / 1_000_000))
    _CURRENT_USAGE.calls += 1


def _call_ollama(system_prompt: str, user_prompt: str, *, json_output: bool = False, num_predict: int = OLLAMA_NUM_PREDICT) -> str:
    payload: dict = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": num_predict,
        },
    }
    if json_output:
        payload["format"] = "json"

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Start Ollama locally or set REFLEXION_RUNTIME=mock. "
            f"URL={OLLAMA_URL!r}, model={OLLAMA_MODEL!r}"
        ) from exc

    _record_ollama_usage(result)
    return str(result.get("message", {}).get("content", ""))


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.replace("```", "").strip()


def _extract_json(text: str) -> dict:
    clean = _strip_thinking(text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(clean):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(clean[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"LLM did not return a JSON object: {clean[:300]}")


def _context_block(example: QAExample) -> str:
    return "\n\n".join(
        f"[{index}] {chunk.title}\n{chunk.text}"
        for index, chunk in enumerate(example.context, start=1)
    )


def _answer_type_hint(question: str) -> str:
    first_word = question.strip().split(maxsplit=1)[0].lower().strip("?,.")
    if first_word in {"who", "what", "which", "when", "where", "how"}:
        return "This is not a yes/no question. Return the requested answer span, never yes or no."
    if first_word in {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could", "would", "will"}:
        return "This appears to be a yes/no question. Return only yes or no if the context supports it."
    return "Return the shortest answer span supported by the context."


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])


def _guard_judge(example: QAExample, answer: str, judge: JudgeResult) -> JudgeResult:
    gold = normalize_answer(example.gold_answer)
    predicted = normalize_answer(answer)
    yes_no = {"yes", "no"}

    if gold == predicted:
        return JudgeResult(score=1, reason=judge.reason or "Final answer matches the gold answer after normalization.", missing_evidence=[], spurious_claims=[])
    if predicted in yes_no and gold not in yes_no:
        return JudgeResult(
            score=0,
            reason="The prediction is yes/no, but the gold answer is a specific entity, date, place, number, or phrase.",
            missing_evidence=["Need to extract the requested answer span from the context."],
            spurious_claims=[answer],
        )
    if gold in yes_no and predicted not in yes_no:
        return JudgeResult(
            score=0,
            reason="The gold answer is yes/no, but the prediction is not a yes/no answer.",
            missing_evidence=["Need to answer the comparison or verification question directly."],
            spurious_claims=[answer],
        )
    if judge.score == 1 and gold not in predicted:
        return JudgeResult(
            score=0,
            reason="The evaluator marked the answer correct, but the normalized prediction does not contain the normalized gold answer.",
            missing_evidence=[f"Expected answer: {example.gold_answer}"],
            spurious_claims=[answer],
        )
    return judge


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if runtime_mode() == "mock":
        return _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)

    reflections = "\n".join(f"- {item}" for item in reflection_memory) or "None"
    user_prompt = f"""
Question:
{example.question}

Answer type guidance:
{_answer_type_hint(example.question)}

Context:
{_context_block(example)}

Attempt number: {attempt_id}
Agent type: {agent_type}
Reflection memory:
{reflections}

Return only JSON: {{"answer": "short answer span"}}.
"""
    text = _call_ollama(ACTOR_SYSTEM, user_prompt, json_output=True, num_predict=80)
    try:
        payload = _extract_json(text)
        answer = str(payload.get("answer", "")).strip()
        if answer:
            return answer
    except ValueError:
        pass
    clean = _strip_thinking(text).strip().strip('"').strip("'")
    answer_match = re.search(r"^(?:final answer|answer)\s*:\s*(.+)$", clean, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return clean.splitlines()[-1].strip() if clean else ""


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if runtime_mode() == "mock":
        return _mock_evaluator(example, answer)

    user_prompt = f"""
Question:
{example.question}

Gold answer:
{example.gold_answer}

Predicted answer:
{answer}

Context:
{_context_block(example)}
"""
    try:
        payload = _extract_json(_call_ollama(EVALUATOR_SYSTEM, user_prompt, json_output=True, num_predict=160))
        return _guard_judge(example, answer, JudgeResult.model_validate(payload))
    except (ValueError, json.JSONDecodeError, ValidationError):
        return _mock_evaluator(example, answer)


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if runtime_mode() == "mock":
        return _mock_reflector(example, attempt_id, judge)

    user_prompt = f"""
Question:
{example.question}

Context:
{_context_block(example)}

Failed attempt id:
{attempt_id}

Evaluator reason:
{judge.reason}

Missing evidence:
{json.dumps(judge.missing_evidence, ensure_ascii=False)}

Spurious claims:
{json.dumps(judge.spurious_claims, ensure_ascii=False)}
"""
    try:
        payload = _extract_json(_call_ollama(REFLECTOR_SYSTEM, user_prompt, json_output=True, num_predict=180))
        payload["attempt_id"] = attempt_id
        return ReflectionEntry.model_validate(payload)
    except (ValueError, json.JSONDecodeError, ValidationError):
        return _mock_reflector(example, attempt_id, judge)
