"""Integrated-Gradients explanations of world-model forecasts.

For a context window X (T x D) the explained quantity is the model's
probability of infiltration within the next K windows (deterministic
prior-mean rollout, so the attribution is exact and repeatable).

Integrated Gradients (Sundararajan et al., 2017) attributes that score to every
input cell relative to a baseline X0:

    IG_i = (x_i - x0_i) * \\int_0^1 dF(X0 + a (X - X0)) / dx_i da

It is the Aumann-Shapley value of the model - the continuous analogue of the
Shapley values that SHAP approximates - and satisfies completeness:
sum_i IG_i = F(X) - F(X0). Our baseline X0 is the average *benign* network
state, so a positive attribution reads "this feature, deviating from normal
traffic, pushed the forecast towards compromise".

Outputs are signed per-feature totals (summed over time) and per-time-step
totals (which recent windows drove the forecast).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from src.models.world_model import CyberWorldModel


class ForecastExplainer:
    def __init__(self, model: CyberWorldModel, horizon: int, baseline_state: np.ndarray, steps: int = 32) -> None:
        self.model = model.eval()
        self.horizon = horizon
        self.baseline = torch.as_tensor(np.asarray(baseline_state, dtype=np.float32))
        self.steps = steps

    def explain(self, context: np.ndarray, feature_names: Sequence[str], site: int = -1) -> Dict[str, object]:
        x = torch.as_tensor(np.asarray(context, dtype=np.float32))            # (T, D)
        x0 = self.baseline.expand_as(x)
        alphas = torch.linspace(1.0 / self.steps, 1.0, self.steps).view(-1, 1, 1)
        path = (x0 + alphas * (x - x0)).requires_grad_(True)                  # (steps, T, D)
        s = torch.full((self.steps,), site, dtype=torch.long) if getattr(self.model, "n_sites", 0) else None
        s1 = s[:1] if s is not None else None
        score = self.model.forecast_score(path, self.horizon, site=s)
        grads, = torch.autograd.grad(score.sum(), path)
        ig = ((x - x0) * grads.mean(0)).detach().numpy()                      # (T, D)
        with torch.no_grad():
            f_x = float(self.model.forecast_score(x[None], self.horizon, site=s1)[0])
            f_0 = float(self.model.forecast_score(x0[None], self.horizon, site=s1)[0])
        per_feature = ig.sum(0)
        order = np.argsort(-np.abs(per_feature))
        return {
            "score": f_x,
            "baseline_score": f_0,
            "completeness_gap": float(per_feature.sum() - (f_x - f_0)),
            "features": [str(feature_names[i]) for i in order],
            "attribution": [float(per_feature[i]) for i in order],
            "time_attribution": ig.sum(1).tolist(),
        }


def _selftest() -> None:
    torch.manual_seed(0)
    T, D, K = 6, 5, 3
    model = CyberWorldModel(D, latent_dim=8, hidden_dim=16, dropout=0.0).eval()
    ex = ForecastExplainer(model, K, np.zeros(D), steps=64)
    res = ex.explain(np.random.default_rng(0).normal(size=(T, D)), [f"f{i}" for i in range(D)])
    assert len(res["features"]) == D and len(res["time_attribution"]) == T
    # Completeness: attributions sum to F(x) - F(baseline) (up to Riemann error).
    assert abs(res["completeness_gap"]) < 0.02, res["completeness_gap"]
    print("explainer selftest: OK")


if __name__ == "__main__":
    _selftest()
