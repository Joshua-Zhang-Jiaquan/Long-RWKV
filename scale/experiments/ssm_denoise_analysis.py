"""Spectral-submanifold analysis of the BiRWKV masked-diffusion denoiser.

Empirical companion to DAN/theory: measures the premises and predictions of the
diffusion-conditioned SSM program on a REAL trained checkpoint (no-latent
b3-sft lineage), using the soft relaxation of the denoising map:

    G(X)_t = softmax(logits(canvas(X)))_t   for t in the masked set,

where the canvas at masked position t is the simplex point X_t in Delta^V
(realised as the soft embedding sum_v X_t[v] * emb(v)); visible positions are
fixed one-hot tokens (the prompt).  G is exactly the deterministic core the
sampler approximates: `iterative_denoise` commits argmax(G(X)) positions by
confidence; G itself is smooth, so it is the object the discrete-time SSM
theory applies to.

Experiments
-----------
E1  anchors        fixed-point iteration of G from gamma-corrupted starts:
                   residual decay (=> contraction rho_perp), plateau epsilon*,
                   argmax accuracy of the anchor vs the clean tokens.
E2  spectrum       projected Jacobian J~ of G at anchors in the top-K logit
                   observable (the "registered observable" of the theory doc),
                   materialised column-by-column with VJPs; full eigenvalue
                   spectrum, spectral gap, slow/fast split.
E2b interventions  matched-norm perturbations along slow vs fast eigvecs;
                   theory predicts slow directions persist, fast ones decay.
E3  tracking       non-autonomous soft-reverse family
                   Phi_k(X) = (g_{k+1}/g_k) X + (1 - g_{k+1}/g_k) G(X),
                   g_k linear 1 -> 0; distance-to-anchor along the trajectory,
                   and confidence-gated vs residual-gated commit precision.
E4  reduction      PCA + quadratic reduced map on E3 trajectories vs a linear
                   baseline (data-driven reduction in observable space).

Requires CUDA (fla kernels).  Runs inside a qz job only.

Usage
-----
python -m experiments.ssm_denoise_analysis \
    --ckpt_dir .../b3-sft-2p9b/step_00008000_probe_copy \
    --model_dir .../RWKV7-Goose-World3-2.9B-HF \
    --corpus_dir .../owt_rwkv_tokens/train \
    --outdir .../ssm_theory_exp/b3_s8000 [--probe]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

MASK_TOKEN_ID = 65535
PAD_TOKEN_ID = 0


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def load_model(ckpt_dir: str, model_dir: str, device: str) -> torch.nn.Module:
    from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion

    model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(model_dir, dtype=torch.bfloat16)
    state = torch.load(str(Path(ckpt_dir) / "model.pt"), map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class _EmbedStub(torch.nn.Module):
    """Swap model.embeddings for a canvas of precomputed soft embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.h: torch.Tensor | None = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.h


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_docs(corpus_dir: str, n_docs: int, window: int) -> list[torch.Tensor]:
    """First `n_docs` docs with a clean contiguous `window`-token run."""
    out: list[torch.Tensor] = []
    for fp in sorted(Path(corpus_dir).glob("*.npz")):
        data = np.load(fp, allow_pickle=True)
        ids = torch.from_numpy(data["input_ids"].astype("int64"))
        am = torch.from_numpy(data["attention_mask"].astype(bool))
        ok = am & ids.ne(PAD_TOKEN_ID) & ids.ne(MASK_TOKEN_ID)
        # longest contiguous run of clean tokens
        run = 0
        best = (0, 0)
        for i in range(ids.numel()):
            run = run + 1 if bool(ok[i]) else 0
            if run > best[1] - best[0]:
                best = (i + 1 - run, i + 1)
        s, e = best
        if e - s >= window:
            out.append(ids[s : s + window].clone())
        if len(out) >= n_docs:
            break
    return out


# ----------------------------------------------------------------------
# The soft map G
# ----------------------------------------------------------------------
class SoftMap:
    """G(X) = softmax(logits(canvas(X))) on masked positions, canvas = soft emb."""

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.device = device
        self.emb_w = model.embeddings.weight.detach()  # [V, d] bf16
        self.stub = _EmbedStub()
        self._orig_emb = model.embeddings

    def canvas_embed(
        self,
        X: torch.Tensor,
        vis_ids: torch.Tensor,
        mask_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Full [1, T, d] embedding: one-hot at visible, soft X at masked.

        The canvas is built on the MODEL's device, not on X's.  Deriving the
        device from the caller's X is a live defect: when a document is still on
        the host (or X was built from a CPU tensor) the buffer lands on CPU while
        ``vis`` comes from ``emb_w`` on the accelerator, and the copy-in fails
        with PyTorch's ambiguous "found at least two devices, cuda:0 and cpu".
        That is the failure that stopped the E1-E4 run at E3 on 2026-09-03, so
        every tensor here is pinned to one device and X is moved onto it.
        """
        device = self.emb_w.device
        X = X.to(device)
        mask_idx = mask_idx.to(device)
        vis_ids = vis_ids.to(device)
        T = vis_ids.numel() + mask_idx.numel()
        h = torch.zeros(1, T, self.emb_w.shape[1], device=device, dtype=self.emb_w.dtype)
        vis = self.emb_w[vis_ids]  # [n_vis, d]
        h_mask = X.to(self.emb_w.dtype) @ self.emb_w  # [n_mask, d]
        vis_pos = torch.ones(T, dtype=torch.bool, device=device)
        vis_pos[mask_idx] = False
        h[0, vis_pos] = vis
        h[0, mask_idx] = h_mask
        return h

    def _forward_logits(self, h: torch.Tensor) -> torch.Tensor:
        self.stub.h = h
        self.model.embeddings = self.stub
        try:
            ids = torch.zeros(h.shape[0], h.shape[1], dtype=torch.long, device=h.device)
            logits = self.model(ids, False)
        finally:
            self.model.embeddings = self._orig_emb
        return logits

    def probs(self, X: torch.Tensor, vis_ids: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        """G(X): [n_mask, V] fp32 simplex point, on the model's device.

        The returned simplex point is pinned to the model device so that callers
        combining it with their own state cannot mix devices: the E3 crash was a
        downstream symptom of exactly that, and pinning here means a caller that
        keeps its state on the host gets a clear failure at its own line rather
        than an ambiguous one inside a copy.
        """
        h = self.canvas_embed(X, vis_ids, mask_idx)
        logits = self._forward_logits(h)
        index = mask_idx.to(logits.device)
        probs = torch.softmax(logits[0, index].float(), dim=-1)
        return probs.to(self.emb_w.device)


def make_start(
    clean_ids: torch.Tensor,
    mask_idx: torch.Tensor,
    gamma: float,
    device: str,
) -> torch.Tensor:
    """Gamma-corrupted start: (1-gamma)*onehot(clean) + gamma*onehot(MASK)."""
    n = mask_idx.numel()
    V = 65536
    X = torch.zeros(n, V, device=device)
    X[torch.arange(n, device=device), clean_ids[mask_idx]] = 1.0 - gamma
    X[:, MASK_TOKEN_ID] = gamma
    return X


def anchor_accuracy(X: torch.Tensor, clean_ids: torch.Tensor, mask_idx: torch.Tensor) -> float:
    pred = X.argmax(dim=-1)
    return float((pred == clean_ids[mask_idx]).float().mean())


# ----------------------------------------------------------------------
# E1: anchors
# ----------------------------------------------------------------------
def run_anchors(
    smap: SoftMap,
    docs: list[torch.Tensor],
    mask_idx: torch.Tensor,
    gammas: list[float],
    n_iters: int,
    n_seeds: int,
    device: str,
) -> tuple[dict, dict]:
    """Fixed-point iteration of G from gamma-corrupted starts.

    Returns (results dict keyed by gamma, anchors keyed (gamma, doc, seed)).
    """
    res: dict[str, list] = {}
    anchors: dict[tuple, torch.Tensor] = {}
    for gamma in gammas:
        curves = []
        for seed in range(n_seeds):
            for doc_i, clean_ids in enumerate(docs):
                # the masked set is the SAME fixed mask_idx for every doc/seed,
                # so anchors are directly comparable with E2/E3 which use it too
                m_idx = mask_idx
                X = make_start(clean_ids.to(device), m_idx, gamma, device)
                curve = []
                with torch.no_grad():
                    for _ in range(n_iters):
                        Xn = smap.probs(X, _vis_ids(clean_ids.to(device), m_idx), m_idx)
                        curve.append(
                            float((Xn - X).abs().sum().item() / m_idx.numel())
                        )
                        X = Xn
                curves.append(
                    {
                        "doc": doc_i,
                        "seed": seed,
                        "residual": curve,
                        "anchor_acc": anchor_accuracy(X, clean_ids.to(device), m_idx),
                        "anchor_conf": float(X.max(dim=-1).values.mean()),
                        # observable fidelity (\bar eps_K in the theory): mean
                        # mass inside the per-position top-K entries of the anchor
                        "anchor_topk_mass": float(
                            X.topk(16, dim=-1).values.sum(dim=-1).mean()
                        ),
                    }
                )
                anchors[(gamma, doc_i, seed)] = X.cpu()
        res[f"{gamma:.2f}"] = curves
    return res, anchors


def _vis_ids(clean_ids: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
    T = clean_ids.numel()
    vis = torch.ones(T, dtype=torch.bool, device=clean_ids.device)
    vis[mask_idx] = False
    return clean_ids[vis]


# ----------------------------------------------------------------------
# E2: projected Jacobian spectrum
# ----------------------------------------------------------------------
def projected_jacobian(
    smap: SoftMap,
    X_star: torch.Tensor,
    clean_ids: torch.Tensor,
    mask_idx: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialise J~ = d P_out G / d P_in X at X_star in top-K logit coords.

    P_in: X restricted to the top-K entries of each row of X_star.
    P_out: probs restricted to the same top-K entries.
    Returns (J~ [n*K, n*K] fp32 cpu, topk_idx [n, K], P_out(G(X_star)) [n, K]).
    """
    device = X_star.device
    n = mask_idx.numel()
    topv, topk_idx = X_star.topk(K, dim=-1)  # [n, K]
    vis_ids = _vis_ids(clean_ids, mask_idx)

    X = X_star.clone().detach().requires_grad_(True)

    def out() -> torch.Tensor:
        probs = smap.probs(X, vis_ids, mask_idx)  # [n, V]
        return probs.gather(1, topk_idx)  # [n, K]

    o = out()
    flat_out = o.reshape(-1)  # [n*K]
    rows = []
    for i in range(flat_out.numel()):
        g = torch.autograd.grad(flat_out[i], X, retain_graph=True)[0]  # [n, V]
        rows.append(g.gather(1, topk_idx).reshape(-1))  # [n*K]
    J = torch.stack(rows, dim=0)  # row i = d out_i / d (X at topk entries)
    return J.detach().cpu(), topk_idx.cpu(), o.detach().cpu()


def analyze_spectrum(J: torch.Tensor) -> dict:
    evals = torch.linalg.eigvals(J)
    lam_sorted, order = evals.abs().sort(descending=True)
    return {
        "eig_abs_sorted": lam_sorted.tolist(),
        "eig_top20_real": evals[order[:20]].real.tolist(),
        "eig_top20_imag": evals[order[:20]].imag.tolist(),
        "sigma_max": float(torch.linalg.matrix_norm(J, ord=2)),
        "abs_lam_max": float(lam_sorted[0]),
        "gap_ratio_q": {
            str(q): float(lam_sorted[q] / lam_sorted[0]) for q in (1, 2, 4, 8, 16, 32)
        },
        "n_eigs": int(evals.numel()),
    }


# ----------------------------------------------------------------------
# E2b: tangent / normal interventions
# ----------------------------------------------------------------------
def persistence(
    smap: SoftMap,
    X_star: torch.Tensor,
    J: torch.Tensor,
    topk_idx: torch.Tensor,
    clean_ids: torch.Tensor,
    mask_idx: torch.Tensor,
    epsilons: tuple[float, ...],
    horizon: int,
) -> list[dict]:
    """Perturb the anchor along slow / median / fast eigvecs; track decay."""
    device = X_star.device
    evals, evecs = torch.linalg.eig(J.to(device))
    lam_abs = evals.abs()
    order = lam_abs.argsort(descending=True)
    n = mask_idx.numel()
    vis_ids = _vis_ids(clean_ids, mask_idx)

    ranks = {
        "slow": int(order[0]),
        "median": int(order[len(order) // 2]),
        "fast": int(order[-1]),
    }
    out = []
    for name, r in ranks.items():
        v = evecs[:, r].real
        v = v / v.norm()
        lam = float(lam_abs[r])
        for eps in epsilons:
            delta_y = (eps * v).reshape(n, -1)  # [n, K]
            Xp = X_star.clone()
            delta_X = torch.zeros_like(Xp)
            delta_X.scatter_(1, topk_idx.to(device), delta_y)
            Xp = (Xp + delta_X).clamp_min(0)
            Xp = Xp / Xp.sum(dim=-1, keepdim=True)
            with torch.no_grad():
                y_ref = smap.probs(X_star, vis_ids, mask_idx).gather(1, topk_idx.to(device))
                y0 = smap.probs(Xp, vis_ids, mask_idx).gather(1, topk_idx.to(device))
                d0 = float((y0 - y_ref).norm())
                ratios = []
                Xc = Xp
                for _ in range(horizon):
                    Xc = smap.probs(Xc, vis_ids, mask_idx)
                    yc = Xc.gather(1, topk_idx.to(device))
                    ratios.append(float((yc - y_ref).norm()) / max(d0, 1e-12))
            out.append(
                {
                    "direction": name,
                    "eig_abs": lam,
                    "eps": eps,
                    "d0": d0,
                    "decay_ratios": ratios,
                }
            )
    return out


# ----------------------------------------------------------------------
# E3: schedule tracking + commit timing
# ----------------------------------------------------------------------
def reverse_trajectory(
    smap: SoftMap,
    clean_ids: torch.Tensor,
    mask_idx: torch.Tensor,
    K_steps: int,
    anchor: torch.Tensor,
) -> dict:
    """Phi_k(X) = a X + (1-a) G(X), a = g_{k+1}/g_k, g_k = 1 - k/K."""
    # Pinned to the MODEL's device, not to the caller's document: the trajectory
    # is mixed with G(X) every step, and deriving the device from clean_ids left
    # X on the host whenever the document had not been moved -- which is the
    # 2026-09-03 failure this function is being repaired for.
    device = smap.emb_w.device
    clean_ids = clean_ids.to(device)
    mask_idx = mask_idx.to(device)
    anchor = anchor.to(device)
    vis_ids = _vis_ids(clean_ids, mask_idx)
    X = torch.zeros(mask_idx.numel(), 65536, device=device)
    X[:, MASK_TOKEN_ID] = 1.0
    gammas = [1.0 - k / K_steps for k in range(K_steps + 1)]
    dist_anchor, conf_mean, resids = [], [], []
    traj = []
    with torch.no_grad():
        for k in range(K_steps):
            gk = smap.probs(X, vis_ids, mask_idx)
            a = gammas[k + 1] / gammas[k] if gammas[k] > 0 else 0.0
            X = a * X + (1.0 - a) * gk
            dist_anchor.append(float((X - anchor).abs().sum() / mask_idx.numel()))
            conf_mean.append(float(X.max(dim=-1).values.mean()))
            resids.append(float((gk - X).abs().sum() / mask_idx.numel()))
            traj.append(X.detach().to("cpu"))
    final_acc = anchor_accuracy(traj[-1].to(device), clean_ids, mask_idx)
    # commit-gate comparison on the FINAL iterate's per-position history
    hist = torch.stack(traj)  # [K, n, V] on CPU
    final_pred = hist[-1].argmax(dim=-1).to(device)  # [n]
    correct = final_pred == clean_ids[mask_idx]
    gates = {}
    for conf_th in (0.5, 0.9, 0.99):
        committed, right = 0, 0
        for t in range(hist.shape[1]):
            c = hist[:, t, :].max(dim=-1).values
            if bool((c >= conf_th).any()):
                committed += 1
                if bool(correct[t]):
                    right += 1
        gates[f"conf{conf_th}"] = {
            "committed_frac": committed / hist.shape[1],
            "precision": right / max(committed, 1),
        }
    for dist_th in (0.2, 0.1, 0.05):
        committed, right = 0, 0
        anchor_dev = anchor.to(device)
        for t in range(hist.shape[1]):
            # ``hist[:, t, :]`` is [K, V] -- every step AT POSITION t -- so the
            # reference is that position's anchor row, not the whole [n_mask, V]
            # anchor. Subtracting the full anchor raised only when n_mask != K
            # and silently returned nonsense when they happened to be equal,
            # which is the worse of the two failure modes. The confidence gate
            # above reduces the same axis the same way.
            d = (hist[:, t, :].to(device) - anchor_dev[t]).abs().sum(dim=-1)
            if bool((d <= dist_th).any()):
                committed += 1
                if bool(correct[t]):
                    right += 1
        gates[f"dist{dist_th}"] = {
            "committed_frac": committed / hist.shape[1],
            "precision": right / max(committed, 1),
        }
    return {
        "dist_anchor_l1_per_pos": dist_anchor,
        "conf_mean": conf_mean,
        "schedule_residual": resids,
        "final_acc": final_acc,
        "commit_gates": gates,
        "traj": traj,  # kept in memory for E4; stripped before JSON
    }


# ----------------------------------------------------------------------
# E4: PCA + quadratic reduced dynamics
# ----------------------------------------------------------------------
def reduced_dynamics(
    trajs: list[dict],
    topk_fn,
    q_list: tuple[int, ...],
    n_test: int,
) -> dict:
    """Fit eta_{k+1} = R(eta_k) per doc trajectory in PCA coords of the
    top-K observable; quadratic vs linear; held-out doc error."""
    # observable matrix: rows = (doc, k), cols = flattened probs at top-K
    Y = []
    for d_i, tr in enumerate(trajs):
        topk_idx = topk_fn(d_i)  # [n, K]
        for Xk in tr["traj"]:
            y = Xk.gather(1, topk_idx).reshape(-1)
            Y.append(y)
    Y = torch.stack(Y)  # [M*K, n*K]
    Yc = Y - Y.mean(dim=0)
    U, S, Vt = torch.linalg.svd(Yc, full_matrices=False)
    var_expl = (S**2 / (S**2).sum()).tolist()
    out = {"pca_var_explained": var_expl[:32]}
    n_steps = len(trajs[0]["traj"])
    for q in q_list:
        Vq = Vt[:q]  # [q, D]
        # coordinates per doc: [K, q]
        etas = []
        for d_i, tr in enumerate(trajs):
            topk_idx = topk_fn(d_i)
            ys = torch.stack(
                [Xk.gather(1, topk_idx).reshape(-1) for Xk in tr["traj"]]
            )
            etas.append((ys - Y.mean(dim=0)) @ Vq.T)
        train = etas[:-n_test] if n_test > 0 else etas[:-1]
        test = etas[-n_test:] if n_test > 0 else etas[-1:]

        def fit(deg: int) -> tuple[torch.Tensor, callable]:
            rows, targ = [], []
            for E in train:
                for k in range(E.shape[0] - 1):
                    eta = E[k]
                    feats = [torch.ones(1)]
                    feats.append(eta)
                    if deg >= 2:
                        iu = torch.triu_indices(q, q)
                        feats.append((eta[iu[0]] * eta[iu[1]]))
                    rows.append(torch.cat(feats))
                    targ.append(E[k + 1])
            A = torch.stack(rows)
            B = torch.stack(targ)
            coef = torch.linalg.lstsq(A, B).solution
            def pred(eta: torch.Tensor) -> torch.Tensor:
                feats = [torch.ones(1)]
                feats.append(eta)
                if deg >= 2:
                    iu = torch.triu_indices(q, q)
                    feats.append((eta[iu[0]] * eta[iu[1]]))
                return torch.cat(feats) @ coef
            return coef, pred

        errs = {}
        for deg, name in ((1, "linear"), (2, "quadratic")):
            _, pred = fit(deg)
            e1, e8 = [], []
            for E in test:
                eta = E[0].clone()
                std = E.std(dim=0).clamp_min(1e-8)
                for k in range(E.shape[0] - 1):
                    eta_hat = pred(eta)
                    e1.append(float(((eta_hat - E[k + 1]) / std).norm() / math.sqrt(q)))
                    eta = eta_hat
                # 8-step rollout from the true start
                eta = E[0].clone()
                for k in range(min(8, E.shape[0] - 1)):
                    eta = pred(eta)
                e8.append(float(((eta - E[min(8, E.shape[0] - 1)]) / std).norm() / math.sqrt(q)))
            errs[name] = {"nmte1": float(np.mean(e1)), "nmte8": float(np.mean(e8))}
        out[f"q{q}"] = errs
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = load_model(args.ckpt_dir, args.model_dir, device)
    smap = SoftMap(model, device)
    print(f"[ssm] model loaded in {time.time()-t0:.0f}s", flush=True)

    if args.probe:
        docs = load_docs(args.corpus_dir, n_docs=1, window=64)
        assert docs, "no usable docs"
        clean = docs[0].to(device)
        mask_idx = torch.arange(8, 56, device=device)
        X = make_start(clean, mask_idx, 0.5, device)
        with torch.no_grad():
            P = smap.probs(X, _vis_ids(clean, mask_idx), mask_idx)
        print(f"[ssm] PROBE OK probs shape={tuple(P.shape)} conf={P.max(-1).values.mean():.3f}", flush=True)
        return 0

    # --- configuration ---
    T = 128
    n_mask = 64
    K = 16
    n_docs = 8
    gammas = [0.15, 0.30, 0.50, 0.70, 0.90, 1.00]
    n_iters = 20
    n_seeds = 2
    K_steps = 16

    docs = load_docs(args.corpus_dir, n_docs=n_docs, window=T)
    assert len(docs) >= n_docs, f"only {len(docs)} docs"
    mask_idx = torch.arange((T - n_mask) // 2, (T - n_mask) // 2 + n_mask, device=device)

    # E1: anchors
    t1 = time.time()
    anchors_res, anchors = run_anchors(smap, docs, mask_idx, gammas, n_iters, n_seeds, device)
    print(f"[ssm] E1 anchors done in {time.time()-t1:.0f}s", flush=True)
    # effective contraction from the tail of the residual curves
    rho_est = {}
    for g, curves in anchors_res.items():
        rr = []
        for c in curves:
            tail = c["residual"][-6:]
            if min(tail) > 1e-6:
                rr.append((tail[-1] / tail[0]) ** (1 / (len(tail) - 1)))
        rho_est[g] = float(np.mean(rr)) if rr else None
    print(f"[ssm] rho_est={rho_est}", flush=True)

    def _dump(name: str, payload: dict) -> None:
        with open(outdir / f"ssm_{name}.json", "w") as f:
            json.dump(payload, f, indent=1)

    _dump("e1", {"E1_anchors": anchors_res, "E1_rho_est": rho_est})

    # E2: spectrum at the gamma=0.50 anchors of docs 0 and 1 (seed 0)
    spec = {}
    persist = []
    for doc_i in (0, 1):
        X_star = anchors[(0.50, doc_i, 0)].to(device)
        clean = docs[doc_i].to(device)
        t2 = time.time()
        J, topk_idx, _ = projected_jacobian(smap, X_star, clean, mask_idx, K)
        print(f"[ssm] E2 jacobian doc{doc_i} done in {time.time()-t2:.0f}s", flush=True)
        spec[f"doc{doc_i}"] = analyze_spectrum(J)
        persist.extend(
            persistence(smap, X_star, J, topk_idx.to(device), clean, mask_idx, (0.05, 0.2), 4)
        )
    print(f"[ssm] E2+E2b done", flush=True)
    _dump("e2", {"E2_spectrum": spec, "E2b_persistence": persist})

    # E3: tracking + commit timing (use the gamma=1.0 anchor as the reference)
    trajs = []
    for doc_i in range(n_docs):
        clean = docs[doc_i].to(device)
        anchor = anchors[(1.00, doc_i, 0)].to(device)
        tr = reverse_trajectory(smap, clean, mask_idx, K_steps, anchor)
        trajs.append(tr)
    print(f"[ssm] E3 done: final_acc={[t['final_acc'] for t in trajs]}", flush=True)
    _dump("e3", {"E3_tracking": [{k: v for k, v in tr.items() if k != "traj"} for tr in trajs]})

    # E4: reduced dynamics in per-doc top-K observables
    topk_cache = {}
    def topk_fn(d_i: int) -> torch.Tensor:
        if d_i not in topk_cache:
            topk_cache[d_i] = anchors[(1.00, d_i, 0)].topk(K, dim=-1).indices  # cpu
        return topk_cache[d_i]
    red = reduced_dynamics(trajs, topk_fn, (2, 4, 8), n_test=2)

    # ---- assemble JSON (strip heavy trajectory tensors) ----
    result = {
        # Stamped so a partial run is detectable from the artifact alone. The
        # 2026-09-03 crash left no ssm_analysis.json at all, which is why its
        # absence (rather than its contents) is the signal; a reader should not
        # have to infer completion from which keys happen to be present.
        "status": "COMPLETE",
        "stages_completed": ["E1", "E2", "E2b", "E3", "E4"],
        "meta": {
            "ckpt_dir": args.ckpt_dir,
            "model_dir": args.model_dir,
            "corpus_dir": args.corpus_dir,
            "seed": args.seed,
            "T": T,
            "n_mask": n_mask,
            "K": K,
            "n_docs": n_docs,
            "gammas": gammas,
            "n_iters": n_iters,
            "K_steps": K_steps,
            "elapsed_s": time.time() - t0,
        },
        "E1_anchors": anchors_res,
        "E1_rho_est": rho_est,
        "E2_spectrum": spec,
        "E2b_persistence": persist,
        "E3_tracking": [
            {k: v for k, v in tr.items() if k != "traj"} for tr in trajs
        ],
        "E4_reduced": red,
    }
    with open(outdir / "ssm_analysis.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"[ssm] wrote {outdir/'ssm_analysis.json'}", flush=True)

    # ---- figures ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = outdir / "figs"
        figs.mkdir(exist_ok=True)
        # anchor convergence
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for g in gammas:
            curves = anchors_res[f"{g:.2f}"]
            lens = min(len(c["residual"]) for c in curves)
            ys = np.mean([c["residual"][:lens] for c in curves], axis=0)
            ax.plot(range(1, lens + 1), ys, label=f"γ={g:.2f}")
        ax.set_xlabel("iteration j")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\|G(X_j)-X_j\|_1/n$")
        ax.set_title("E1: anchor convergence of the soft map G")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / "e1_anchor_convergence.png", dpi=160)
        # spectrum
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for key, s in spec.items():
            lam = np.array(s["eig_abs_sorted"])
            ax.plot(range(1, len(lam) + 1), lam, label=key)
        ax.set_xlabel("eigenvalue rank")
        ax.set_ylabel(r"$|\lambda|$")
        ax.set_title("E2: projected Jacobian spectrum at anchors")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "e2_spectrum.png", dpi=160)
        # persistence
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for p in persist:
            if p["eps"] != 0.2:
                continue
            ax.plot(range(1, len(p["decay_ratios"]) + 1), p["decay_ratios"],
                    marker="o", ms=3, label=f"{p['direction']} (|λ|={p['eig_abs']:.2f})")
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.set_xlabel("iterations after perturbation")
        ax.set_ylabel(r"$\|\Delta y_m\|/\|\Delta y_0\|$")
        ax.set_title("E2b: matched-norm perturbation persistence")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / "e2b_persistence.png", dpi=160)
        # tracking
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for i, tr in enumerate(trajs):
            ax.plot(range(len(tr["dist_anchor_l1_per_pos"])), tr["dist_anchor_l1_per_pos"],
                    alpha=0.6, label=f"doc{i}")
        ax.set_xlabel("reverse step k")
        ax.set_ylabel(r"$\|X_k-X^*\|_1/n$")
        ax.set_title("E3: distance to anchor along the soft-reverse schedule")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figs / "e3_tracking.png", dpi=160)
        print(f"[ssm] figures in {figs}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[ssm] figure generation skipped: {exc}", flush=True)

    print("[ssm] ALL DONE", flush=True)
    return 0


def _guarded_main() -> int:
    """Run :func:`main`, and on any failure leave a machine-readable receipt.

    The 2026-09-03 run is the reason this exists: the script raised at E3, and
    the shell wrapper around it still printed ``===SSM DONE exit=0===``.  Any
    reader of the wrapper's output -- including the schedule that decided
    whether to retry -- saw a clean run.  Making the wrapper honest is not
    something this file can do; making the failure survive in the artifact
    directory is, so a crashed analysis can never be mistaken for a completed
    one no matter what the caller reports.

    Returns 0 only when every stage actually ran.
    """
    argv = sys.argv[1:]
    outdir: Path | None = None
    if "--outdir" in argv:
        try:
            outdir = Path(argv[argv.index("--outdir") + 1])
        except IndexError:
            outdir = None
    try:
        return main()
    except Exception:  # noqa: BLE001 - the receipt is the point
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "ssm_FAILED.json").write_text(
                json.dumps({
                    "schema": "nonlatent_ssm_analysis_failure_v1",
                    "status": "FAILED",
                    "argv": argv,
                    "traceback": detail,
                }, indent=1),
                encoding="utf-8",
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(_guarded_main())
