from pathlib import Path
from typing import Dict
import json

from app.safety.risk_detector import detect_risk


def load_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_safety(eval_path: str) -> Dict:
    samples = load_jsonl(eval_path)

    need_human_correct = []
    risk_type_correct = []
    exact_matches = []

    for sample in samples:
        result = detect_risk(sample["question"])

        need_human_correct.append(
            1.0 if result["need_human"] == (bool(sample['risk_types']) if 'risk_types' in sample else sample['need_human']) else 0.0
        )

        if 'risk_types' in sample:
            expected = set(sample['risk_types'])
            actual = set(result['risk_types'])
            risk_type_correct.append(float(expected <= actual) if expected else float(not actual))
            exact_matches.append(float(expected == actual))
            continue
        expected_risk_type = sample.get("risk_type")
        if expected_risk_type == "normal":
            risk_type_correct.append(
                1.0 if len(result["risk_types"]) == 0 else 0.0
            )
        else:
            risk_type_correct.append(
                1.0 if expected_risk_type in result["risk_types"] else 0.0
            )

    return {
        "sample_count": len(samples),
        "metric_scope": "risk_detection_only; does not measure final answer safety",
        "need_human_accuracy": sum(need_human_correct) / len(need_human_correct),
        "risk_type_accuracy": sum(risk_type_correct) / len(risk_type_correct),
        "risk_type_exact_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else None
    }
