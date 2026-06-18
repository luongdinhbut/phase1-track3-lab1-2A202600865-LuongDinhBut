from __future__ import annotations
from dataclasses import dataclass
import re
import time
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, consume_runtime_usage, evaluator, reflector, reset_runtime_usage
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def _count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))

def _estimate_attempt_tokens(
    example: QAExample,
    answer: str,
    judge: JudgeResult,
    reflection_memory: list[str],
    reflection: ReflectionEntry | None,
) -> int:
    context_text = "\n".join(f"{chunk.title}: {chunk.text}" for chunk in example.context)
    parts = [
        example.question,
        context_text,
        "\n".join(reflection_memory),
        answer,
        judge.reason,
        "\n".join(judge.missing_evidence),
        "\n".join(judge.spurious_claims),
    ]
    if reflection is not None:
        parts.extend([reflection.failure_reason, reflection.lesson, reflection.next_strategy])
    return sum(_count_tokens(part) for part in parts if part)

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            reset_runtime_usage()
            start_time = time.perf_counter()
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = evaluator(example, answer)
            final_answer = answer
            final_score = judge.score
            reflection = None
            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflections.append(reflection)
                reflection_memory.append(
                    f"Attempt {attempt_id}: {reflection.lesson} Next strategy: {reflection.next_strategy}"
                )
            wall_latency_ms = max(1, round((time.perf_counter() - start_time) * 1000))
            runtime_usage = consume_runtime_usage()
            latency_ms = runtime_usage.latency_ms or wall_latency_ms
            token_estimate = runtime_usage.total_tokens or _estimate_attempt_tokens(example, answer, judge, reflection_memory, reflection)
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, reflection=reflection, token_estimate=token_estimate, latency_ms=latency_ms)
            traces.append(trace)
            if judge.score == 1:
                break
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
