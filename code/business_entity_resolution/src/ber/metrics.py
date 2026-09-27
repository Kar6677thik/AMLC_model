from __future__ import annotations

from collections import Counter


def f05(truth, prediction):
    truth, prediction = set(truth), set(prediction)
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    return 5 * tp / (5 * tp + 4 * len(prediction - truth) + len(truth - prediction))


class Metrics:
    def __init__(self):
        self.n = self.tp = self.fp = self.fn = self.singletons = self.singleton_fp = 0
        self.nonempty = self.recovered = self.full = self.total_truth = self.candidates = 0
        self.score = self.ceiling = self.macro_recall = 0.0
        self.hist = Counter()

    def add(self, truth, predicted, candidates):
        truth, predicted, candidates = set(truth), set(predicted), set(candidates)
        self.n += 1
        self.score += f05(truth, predicted)
        self.ceiling += f05(truth, truth & candidates)
        self.tp += len(truth & predicted)
        self.fp += len(predicted - truth)
        self.fn += len(truth - predicted)
        self.singletons += not truth
        self.singleton_fp += not truth and bool(predicted)
        self.nonempty += bool(truth)
        self.recovered += len(truth & candidates)
        self.total_truth += len(truth)
        if truth:
            self.full += truth <= candidates
            self.macro_recall += len(truth & candidates) / len(truth)
        self.candidates += len(candidates)
        self.hist[len(candidates)] += 1

    def report(self):
        def ratio(a, b):
            return a / b if b else None
        def quantile(q):
            total = 0
            for size, count in sorted(self.hist.items()):
                total += count
                if total >= self.n * q:
                    return size
            return 0
        return {
            "anchors": self.n, "macro_f05": ratio(self.score, self.n),
            "oracle_macro_f05": ratio(self.ceiling, self.n),
            "micro_precision": ratio(self.tp, self.tp + self.fp),
            "micro_recall": ratio(self.tp, self.tp + self.fn),
            "singleton_fraction": ratio(self.singletons, self.n),
            "singleton_false_positive_rate": ratio(self.singleton_fp, self.singletons),
            "candidate_micro_recall": ratio(self.recovered, self.total_truth),
            "candidate_macro_recall": ratio(self.macro_recall, self.nonempty),
            "full_recovery_rate": ratio(self.full, self.nonempty),
            "total_candidates": self.candidates, "mean_candidates": ratio(self.candidates, self.n),
            "p95_candidates": quantile(.95), "p99_candidates": quantile(.99),
            "max_candidates": max(self.hist, default=0), "candidate_count_histogram": dict(self.hist),
        }
