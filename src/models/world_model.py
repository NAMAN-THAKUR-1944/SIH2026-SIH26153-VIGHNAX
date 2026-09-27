"""Stochastic latent world model of network state dynamics.

Architecture (an RSSM-style latent dynamics model, cf. Ha & Schmidhuber 2018,
Hafner et al. "Dreamer" 2019-2023, adapted to network telemetry):

    posterior  q(z_t | S_t)            StateEncoder  - Gaussian latent of the observed state
    memory     h_t = GRU(h_{t-1}, z_{t-1})           - deterministic temporal context
    prior      p(z_t | h_t)            TransitionPrior - Gaussian over the *next* latent state
    decoder    p(S_t | z_t)            StateDecoder  - reconstructs the network state
    heads      p(attack | h_t, z_t), p(stage | h_t, z_t)   - infiltration + MITRE stage

Together prior and decoder define the learned transition distribution
P(S_{t+1} | S_<=t) = \\int p(S_{t+1} | z) p(z | h_{t+1}) dz: the model does not
classify states, it predicts how the network state will *evolve*. Forward
simulation samples z from the prior K times (Monte-Carlo) and reads the heads
on those imagined states, giving an infiltration probability per future window
with an uncertainty band, and a predicted attack-stage distribution.

The heads are trained on both observed (posterior) states and on imagined
(prior) rollouts against the true future labels ("latent overshooting"), so
forecasts are supervised by the ground-truth attack timeline, as the PS asks.

``surprise_t = KL(q(z_t | S_t) || p(z_t | h_t))`` measures how far the observed
state deviates from what the model expected - an unsupervised novelty signal
for attack patterns never seen in training.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(inp: int, hidden: int, out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out),
    )


class _Gaussian(nn.Module):
    """MLP producing (mean, std) of a diagonal Gaussian."""

    def __init__(self, inp: int, hidden: int, latent: int, dropout: float) -> None:
        super().__init__()
        self.net = _mlp(inp, hidden, 2 * latent, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, raw = self.net(x).chunk(2, dim=-1)
        return mean, F.softplus(raw) + 0.05


def _kl(m_q, s_q, m_p, s_p) -> torch.Tensor:
    """KL(N(m_q, s_q) || N(m_p, s_p)) summed over the latent dimension."""
    return (torch.log(s_p / s_q) + (s_q ** 2 + (m_q - m_p) ** 2) / (2 * s_p ** 2) - 0.5).sum(-1)


class CyberWorldModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128,
                 n_stages: int = 7, dropout: float = 0.1, obs_skip: bool = False, n_sites: int = 0,
                 site_dim: int = 8) -> None:
        super().__init__()
        self.input_dim, self.latent_dim, self.hidden_dim, self.n_stages = input_dim, latent_dim, hidden_dim, n_stages
        # Site conditioning: a learned embedding of *which network* is being watched (index n_sites =
        # unknown network). Lets one model keep a per-network notion of "normal" without negative
        # transfer between very different networks; trained with random site dropout so the
        # unknown-network mode works too.
        self.n_sites = n_sites
        self.site_dim = site_dim if n_sites else 0
        if n_sites:
            self.site_emb = nn.Embedding(n_sites + 1, site_dim)
        # obs_skip: heads also see the latest *observed* state S_t (a skip connection past the
        # latent bottleneck), so per-window evidence is not lost while dynamics carry the future.
        self.obs_skip = obs_skip
        head_in = latent_dim + hidden_dim + (input_dim if obs_skip else 0) + self.site_dim
        self.encoder = _Gaussian(input_dim + self.site_dim, hidden_dim, latent_dim, dropout)      # q(z_t | S_t, site)
        self.prior = _Gaussian(hidden_dim + self.site_dim, hidden_dim, latent_dim, dropout)       # p(z_t | h_t, site)
        self.rnn = nn.GRUCell(latent_dim, hidden_dim)                             # h_t = f(h_{t-1}, z_{t-1})
        self.decoder = _mlp(latent_dim, hidden_dim, input_dim, dropout)           # p(S_t | z_t)
        self.attack_head = _mlp(head_in, hidden_dim // 2, 1, dropout)
        self.stage_head = _mlp(head_in, hidden_dim // 2, n_stages, dropout)

    # ------------------------------------------------------------------ core
    def site_vec(self, site: Optional[torch.Tensor], batch: int, device) -> Optional[torch.Tensor]:
        """(B, site_dim) embedding; None / -1 entries mean 'unknown network'."""
        if not self.n_sites:
            return None
        if site is None:
            site = torch.full((batch,), self.n_sites, dtype=torch.long, device=device)
        site = torch.as_tensor(site, device=device).long().reshape(-1)
        if site.numel() == 1 and batch > 1:
            site = site.expand(batch)
        site = torch.where(site < 0, torch.full_like(site, self.n_sites), site)
        return self.site_emb(site)

    def _with(self, t: torch.Tensor, e: Optional[torch.Tensor]) -> torch.Tensor:
        if e is None:
            return t
        if t.dim() == 3:
            e = e[:, None, :].expand(-1, t.shape[1], -1)
        return torch.cat([t, e], dim=-1)

    def heads(self, h: torch.Tensor, z: torch.Tensor, x: Optional[torch.Tensor] = None,
              e: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = [h, z] + ([x] if self.obs_skip else [])
        feat = self._with(torch.cat(parts, dim=-1), e)
        return self.attack_head(feat).squeeze(-1), self.stage_head(feat)

    def observe(self, x: torch.Tensor, sample: bool = True, site: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Filter a sequence x (B, T, D) through the model.

        Returns per-step posterior/prior parameters, latents, memory states
        (h_t is the memory *before* seeing S_t), and head logits.
        """
        B, T, _ = x.shape
        e = self.site_vec(site, B, x.device)
        m_q, s_q = self.encoder(self._with(x, e))
        z = m_q + s_q * torch.randn_like(s_q) if sample else m_q
        h = x.new_zeros(B, self.hidden_dim)
        hs, m_ps, s_ps = [], [], []
        for t in range(T):
            if t > 0:
                h = self.rnn(z[:, t - 1], h)
            m_p, s_p = self.prior(self._with(h, e))
            hs.append(h); m_ps.append(m_p); s_ps.append(s_p)
        h_seq = torch.stack(hs, 1)
        m_p, s_p = torch.stack(m_ps, 1), torch.stack(s_ps, 1)
        att, stg = self.heads(h_seq, z, x, e)
        return {"m_q": m_q, "s_q": s_q, "z": z, "h": h_seq, "m_p": m_p, "s_p": s_p, "e": e,
                "attack_logit": att, "stage_logit": stg, "recon": self.decoder(z),
                "surprise": _kl(m_q, s_q, m_p, s_p)}

    def imagine(self, h: torch.Tensor, z: torch.Tensor, k_steps: int, sample: bool = True,
                x_ctx: Optional[torch.Tensor] = None, e: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Roll the latent dynamics K steps forward from state (h_t, z_t) without observations.

        x_ctx is the last observed state (used by the heads when obs_skip is on)."""
        hs, zs, att, stg = [], [], [], []
        for _ in range(k_steps):
            h = self.rnn(z, h)
            m_p, s_p = self.prior(self._with(h, e))
            z = m_p + s_p * torch.randn_like(s_p) if sample else m_p
            a, s = self.heads(h, z, x_ctx, e)
            hs.append(h); zs.append(z); att.append(a); stg.append(s)
        z_seq = torch.stack(zs, 1)
        return {"h": torch.stack(hs, 1), "z": z_seq, "attack_logit": torch.stack(att, 1),
                "stage_logit": torch.stack(stg, 1), "recon": self.decoder(z_seq)}

    # --------------------------------------------------------------- training
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, stage: torch.Tensor, horizon: int,
                     w: Dict[str, float], pos_weight: Optional[torch.Tensor] = None,
                     stage_weight: Optional[torch.Tensor] = None, min_context: int = 4,
                     site: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """x (B,T,D) scaled states, y (B,T) 0/1 attack, stage (B,T) stage index, site (B,) network id."""
        B, T, D = x.shape
        out = self.observe(x, sample=True, site=site)
        recon = F.mse_loss(out["recon"], x)
        kl = torch.clamp(out["surprise"][:, 1:], min=w.get("kl_free_nats", 0.0)).mean()
        now_bce = F.binary_cross_entropy_with_logits(out["attack_logit"], y.float(), pos_weight=pos_weight)
        now_ce = F.cross_entropy(out["stage_logit"].reshape(-1, self.n_stages), stage.reshape(-1),
                                 weight=stage_weight)

        # Latent overshooting: imagine K steps from every start t in [min_context-1, T-K-1]
        starts = list(range(min_context - 1, T - horizon))
        if starts:
            idx = torch.tensor(starts, device=x.device)
            h0 = out["h"][:, idx].reshape(-1, self.hidden_dim)
            z0 = out["z"][:, idx].reshape(-1, self.latent_dim)
            x0 = x[:, idx].reshape(-1, D)
            e0 = None if out["e"] is None else out["e"][:, None, :].expand(-1, len(starts), -1).reshape(-1, self.site_dim)
            im = self.imagine(h0, z0, horizon, sample=True, x_ctx=x0, e=e0)
            fut = torch.stack([torch.arange(s + 1, s + 1 + horizon) for s in starts]).to(x.device)  # (S,K)
            y_f = y[:, fut].reshape(-1, horizon).float()
            st_f = stage[:, fut].reshape(-1, horizon)
            x_f = x[:, fut].reshape(-1, horizon, D)
            f_bce = F.binary_cross_entropy_with_logits(im["attack_logit"], y_f, pos_weight=pos_weight)
            f_ce = F.cross_entropy(im["stage_logit"].reshape(-1, self.n_stages), st_f.reshape(-1),
                                   weight=stage_weight)
            f_rec = F.mse_loss(im["recon"], x_f)
        else:
            f_bce = f_ce = f_rec = x.new_zeros(())

        total = (w["recon_weight"] * (recon + 0.5 * f_rec) + w["kl_weight"] * kl
                 + w["cls_weight"] * now_bce + w["stage_weight"] * now_ce
                 + w["forecast_weight"] * (f_bce + 0.5 * f_ce))
        return total, {"loss": total.item(), "recon": recon.item(), "kl": kl.item(),
                       "bce_now": now_bce.item(), "ce_now": now_ce.item(),
                       "bce_fut": f_bce.item(), "ce_fut": f_ce.item()}

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def forecast(self, x: torch.Tensor, k_steps: int, n_samples: int = 32,
                 site: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Monte-Carlo K-step forward simulation from the end of each context.

        x: (B, T, D) context windows. Returns, per batch item:
          nowcast (B,), stage_now (B, S), surprise (B,),
          p_step (B, K) mean infiltration probability per future window,
          p_lo / p_hi (B, K) 10th / 90th percentile across rollouts,
          p_any (B,) probability of infiltration within the next K windows (+ p_any_lo / p_any_hi),
          stage_future (B, K, S) predicted stage distribution per future window.
        """
        B = x.shape[0]
        out = self.observe(x, sample=False, site=site)
        h_T, z_T = out["h"][:, -1], out["z"][:, -1]
        h0 = h_T.repeat_interleave(n_samples, 0)
        z0 = z_T.repeat_interleave(n_samples, 0)
        e0 = None if out["e"] is None else out["e"].repeat_interleave(n_samples, 0)
        im = self.imagine(h0, z0, k_steps, sample=True, x_ctx=x[:, -1].repeat_interleave(n_samples, 0), e=e0)
        p = torch.sigmoid(im["attack_logit"]).view(B, n_samples, k_steps)
        stage_p = torch.softmax(im["stage_logit"], -1).view(B, n_samples, k_steps, -1)
        p_any = 1 - torch.prod(1 - p, dim=-1)
        return {
            "nowcast": torch.sigmoid(out["attack_logit"][:, -1]),
            "stage_now": torch.softmax(out["stage_logit"][:, -1], -1),
            "surprise": out["surprise"][:, -1],
            "p_step": p.mean(1), "p_lo": p.quantile(0.1, dim=1), "p_hi": p.quantile(0.9, dim=1),
            "p_any": p_any.mean(1), "p_any_lo": p_any.quantile(0.1, dim=1), "p_any_hi": p_any.quantile(0.9, dim=1),
            "stage_future": stage_p.mean(1),
            "state_future": im["recon"].view(B, n_samples, k_steps, -1).mean(1),
        }

    def forecast_score(self, x: torch.Tensor, k_steps: int, site: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Deterministic (prior-mean) differentiable P(infiltration within K) - used for attribution."""
        out = self.observe(x, sample=False, site=site)
        im = self.imagine(out["h"][:, -1], out["z"][:, -1], k_steps, sample=False, x_ctx=x[:, -1], e=out["e"])
        p = torch.sigmoid(im["attack_logit"])
        return 1 - torch.prod(1 - p, dim=-1)


def _selftest() -> None:
    torch.manual_seed(0)
    B, T, D, K = 4, 12, 20, 3
    model = CyberWorldModel(D, latent_dim=8, hidden_dim=32)
    x = torch.randn(B, T, D)
    y = torch.randint(0, 2, (B, T))
    s = torch.randint(0, 7, (B, T))
    w = dict(recon_weight=1, kl_weight=1, cls_weight=1, stage_weight=1, forecast_weight=1, kl_free_nats=0.1)
    loss, parts = model.compute_loss(x, y, s, K, w)
    loss.backward()
    assert torch.isfinite(loss) and "bce_fut" in parts
    model.eval()
    fc = model.forecast(x, K, n_samples=8)
    assert fc["p_step"].shape == (B, K) and fc["stage_future"].shape == (B, K, 7)
    assert torch.all((fc["p_any"] >= 0) & (fc["p_any"] <= 1))
    assert torch.all(fc["p_lo"] <= fc["p_hi"] + 1e-6)
    sc = model.forecast_score(x.requires_grad_(True), K)
    sc.sum().backward()
    assert x.grad is not None
    sm = CyberWorldModel(D, latent_dim=8, hidden_dim=32, n_sites=3).eval()
    a = sm.forecast_score(torch.randn(B, T, D), K, site=torch.tensor([0, 1, 2, -1]))
    b = sm.forecast_score(torch.randn(B, T, D), K)                       # unknown network
    sm.train(); sm.compute_loss(x, y, s, K, w, site=torch.tensor([0, 1, 2, 3]))[0].backward()
    assert a.shape == (B,) and b.shape == (B,)
    print("world_model selftest: OK")


if __name__ == "__main__":
    _selftest()
