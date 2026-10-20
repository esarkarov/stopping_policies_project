import re
import string
from collections import Counter

NO_CREDIT_ANSWERS = ("yes", "no", "noanswer")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if (pred in NO_CREDIT_ANSWERS or ref in NO_CREDIT_ANSWERS) and pred != ref:
        return 0.0
    pred_tokens, ref_tokens = pred.split(), ref.split()
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_answers(metric, prediction: str, answers: list[str]) -> float:
    return max(metric(prediction, answer) for answer in answers)
