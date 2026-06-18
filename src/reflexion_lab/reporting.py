from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord
from .utils import normalize_answer

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def _latest_reason(record: RunRecord) -> str:
    return record.traces[-1].reason if record.traces else ""

def _analysis_failure_mode(record: RunRecord) -> str:
    if record.is_correct:
        return "none"

    answers = [normalize_answer(trace.answer) for trace in record.traces]
    predicted = normalize_answer(record.predicted_answer)
    gold = normalize_answer(record.gold_answer)
    reason = normalize_answer(_latest_reason(record))

    if len(set(answers)) == 1 and len(answers) > 1:
        return "looping"
    if record.agent_type == "reflexion" and record.reflections and not record.is_correct:
        return "reflection_overfit"
    if predicted in gold or gold in predicted:
        return "incomplete_multi_hop"
    if any(term in reason for term in ["partial", "first hop", "omitted", "does not contain", "specific city within", "category"]):
        return "incomplete_multi_hop"
    if any(term in reason for term in ["wrong second hop", "different person", "different date", "wrong entity", "contradicts", "selected the wrong"]):
        return "entity_drift"
    return record.failure_mode if record.failure_mode != "none" else "wrong_final_answer"

def failure_breakdown(records: list[RunRecord]) -> dict:
    descriptions = {
        "none": "The final answer matched the gold answer.",
        "wrong_final_answer": "The model produced an unsupported or non-equivalent final answer.",
        "incomplete_multi_hop": "The model stopped at a partial answer, an over/under-specific span, or missed one required hop.",
        "entity_drift": "The model followed the wrong entity, date, person, location, or relation in the second hop.",
        "looping": "The model repeated the same wrong answer across attempts.",
        "reflection_overfit": "The Reflexion agent used reflection but still converged on a wrong final answer.",
    }
    grouped: dict[str, dict] = {}
    for mode, description in descriptions.items():
        rows = [record for record in records if _analysis_failure_mode(record) == mode]
        if not rows:
            continue
        by_agent = Counter(record.agent_type for record in rows)
        grouped[mode] = {
            "count": len(rows),
            "by_agent": dict(by_agent),
            "description": description,
            "examples": [
                {
                    "qid": record.qid,
                    "agent_type": record.agent_type,
                    "gold_answer": record.gold_answer,
                    "predicted_answer": record.predicted_answer,
                    "reason": _latest_reason(record),
                }
                for record in rows[:3]
            ],
        }
    return grouped

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    summary = summarize(records)
    failure_modes = failure_breakdown(records)
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": _analysis_failure_mode(r), "reflection_count": len(r.reflections)} for r in records]
    discussion = (
        "The Ollama run shows Reflexion improving exact match over the single-shot ReAct baseline, "
        "but the gain costs extra attempts, tokens, and latency. The main remaining failures fall into "
        "incomplete multi-hop answers, entity drift, generic wrong final answers, and Reflexion attempts "
        "that overfit to an unhelpful reflection. This is expected for a very small local model: it often "
        "finds a relevant first-hop entity but loses the requested answer type or relation on the second hop. "
        "The structured evaluator and guard rules make these mistakes visible in the report instead of "
        "silently marking broad paraphrases or yes/no answers as correct."
    )
    return ReportPayload(meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})}, summary=summary, failure_modes=failure_modes, examples=examples, extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding"], discussion=discussion)

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
