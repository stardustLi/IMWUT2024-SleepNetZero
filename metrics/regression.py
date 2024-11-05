from collections import OrderedDict
import pandas as pd
import torch
import torchmetrics as tm
import typing as t
from wuji.Evaluator.EventDetector import EventDetectorEvaluator
from wuji.Evaluator.NightlySleepMetrics import NightlySleepMetricsEvaluator
from wuji.algo.Oxygen import detect_oxygen_desaturation
import multiprocessing as mp
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import numpy as np

class RegressionMetrics(tm.MetricCollection):
    SCALAR_KEYS = {"mae", "pearson"}

    def __init__(self) -> None:
        common = {
            "task": "regression",
        }
        super().__init__(
            {
                "mae": tm.MeanAbsoluteError(),
                "pearson": tm.PearsonCorrCoef(),
            }
        )

    def loggable_items(self) -> OrderedDict[str, tm.Metric]:
        self._compute_groups_create_state_ref(False)
        od: OrderedDict[str, tm.Metric] = OrderedDict()
        for k, v in self._modules.items():
            if k in self.SCALAR_KEYS:
                od[self._set_name(k)] = v
        return od

    def yield_event_based_matrix(self, pred, target) -> dict[str, float]:
        def evaluate_events(e_in_t, e_in_p):
            evaluator = EventDetectorEvaluator(e_in_t, e_in_p, t_collar=1.)
            return evaluator.recall(), evaluator.precision(), evaluator.f1_score(), len(e_in_t), len(e_in_p)
        pred = 80 + pred
        target = 80 + target
        
        pred[torch.isnan(target)] = -100
        target[torch.isnan(target)] = -100
        events_in_target = [detect_oxygen_desaturation(t) for t in target.cpu().to(dtype=torch.float32).numpy()]
        events_in_pred = [detect_oxygen_desaturation(p, spo2_des_min_thre=1.4) for p in pred.cpu().to(dtype=torch.float32).numpy()]
        results = Parallel(n_jobs=32)(delayed(evaluate_events)(e_in_t, e_in_p) for e_in_t, e_in_p in zip(events_in_target, events_in_pred))
        recall_list, precision_list, f1_list, n_events_in_target, n_events_in_pred = zip(*results)
        p = pearsonr(n_events_in_target, n_events_in_pred)[0]
        return {
            f"{self.prefix}pearsonOfAHI": p,
            f"{self.prefix}recall": sum(recall_list) / len(recall_list),
            f"{self.prefix}precision": sum(precision_list) / len(precision_list),
            f"{self.prefix}f1": sum(f1_list) / len(f1_list),
        }
