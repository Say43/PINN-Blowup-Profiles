#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gCLM self-similar blow-up profiles found by a PINN in self-similar coordinates.

    omega_t + a*u*omega_x = u_x*omega,   u_x = H[omega]   (Hilbert transform)
    omega = tau^-1 Omega(xi),  xi = x/tau^c,  u = tau^(c-1) U(xi),  tau = T - t
    R = Omega + c*xi*Omega' + a*U*Omega' - H[Omega]*Omega = 0,  U' = H[Omega], U(0)=0

Step 0: numerical building blocks (sympy check, Hilbert transform on R via
        xi = L*tan(theta/2) + FFT, gate < 1e-8).
Step 1: reference c_ref(a) from pseudospectral time integration.
Step 2: PINN scan over c (fixed), then learnable c from every local minimum.
Step 3: evaluation, plots, SUMMARY.md.

Deviations from the original task text (parity of Omega, normalisation,
far-field representation) are explained in NOTES.md, written to the output dir.

Usage:  python gclm_blowup.py [--smoke] [--out DIR] [--device auto|cpu|cuda]
        python gclm_blowup.py --replot DIR   (redraw loss_vs_c.png from results.csv)
Only torch, numpy, scipy, sympy, pandas, matplotlib. No network access.
"""
import argparse
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
import scipy.integrate as si
import sympy as sp
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T_START = time.time()
ERRORS = []
L_MAP = 1.0          # xi = L*tan(theta/2)
W_DR = 0.1           # weight of mean((dR/dxi)^2)
W_NORM = 1.0         # weight of the normalisation penalty (Omega'(0) + 1)^2
LR = 1e-3
C_TOL = 0.02         # |c - c_ref| criterion
PROF_TOL = 0.05      # relative L2 profile criterion on |xi| <= 10
RES_SMALL = 1e-6     # "small residual": mean(R^2) below this
XI_PLOT = np.linspace(-10.0, 10.0, 801)
COLORS = {0.0: "#2a78d6", 0.25: "#eb6834", 0.5: "#1baf7a"}


def log(msg):
    print(f"[{(time.time() - T_START) / 60:6.2f} min] {msg}", flush=True)


def record_error(where, exc):
    ERRORS.append(f"{where}: {type(exc).__name__}: {exc}")
    log(f"FEHLER in {where}: {exc}")
    traceback.print_exc()


def make_config(smoke):
    if smoke:
        return dict(A_LIST=[0.0], N_PINN=256, ADAM=100, LBFGS=10, NC=3,
                    SIM_N=(512, 8192), SIM_N_ADV=2048, SIM_WALL=10.0, SIM_TMAX=3.0, SIM_MSTOP=1e4,
                    DEADLINE=110.0, STEP2_BUDGET=80.0, RESERVE=8.0, BENCH_STEPS=5)
    return dict(A_LIST=[0.0, 0.25, 0.5], N_PINN=2048, ADAM=3000, LBFGS=300, NC=20,
                SIM_N=(4096, 65536), SIM_N_ADV=16384, SIM_WALL=140.0, SIM_TMAX=40.0, SIM_MSTOP=1e4,
                DEADLINE=95 * 60.0, STEP2_BUDGET=60 * 60.0, RESERVE=180.0, BENCH_STEPS=30)


# ----------------------------------------------------------------------------
# Step 0: symbolic and numerical building blocks
# ----------------------------------------------------------------------------
def step0_sympy():
    out = {}
    t, x, T, c, a, X = sp.symbols("t x T c a xi", real=True)
    tau = T - t
    Om, U = sp.Function("Omega"), sp.Function("U")
    omega = Om(x / tau**c) / tau
    u = tau**(c - 1) * U(x / tau**c)
    ux = sp.diff(u, x)
    pde = sp.diff(omega, t) + a * u * sp.diff(omega, x) - ux * omega
    got = sp.simplify((pde * tau**2).subs(x, X * tau**c).doit())
    claimed = (Om(X) + c * X * Om(X).diff(X) + a * U(X) * Om(X).diff(X)
               - U(X).diff(X) * Om(X))
    out["profile_eq_residual"] = str(sp.simplify(got - claimed))
    # u_x = tau^-1 U'(xi); H is scale invariant, so H[omega] = tau^-1 H[Omega](xi)
    out["closure_residual"] = str(sp.simplify((ux * tau).subs(x, X * tau**c).doit()
                                              - U(X).diff(X)))
    # Even part of R for an even Omega: Omega + c xi Omega' = 0
    sol = sp.dsolve(sp.Eq(Om(X) + c * X * Om(X).diff(X), 0))
    out["even_part_solution"] = str(sol)
    # Exact CLM profile (a = 0, c = 1), normalised to Omega'(0) = -1
    Oe = -4 * X / (4 + X**2)
    He = 8 / (4 + X**2)
    Ue = 4 * sp.atan(X / 2)
    out["clm_residual"] = str(sp.simplify(Oe + X * sp.diff(Oe, X) - He * Oe))
    out["clm_U_prime_minus_H"] = str(sp.simplify(sp.diff(Ue, X) - He))
    out["clm_Omega_prime_0"] = str(sp.diff(Oe, X).subs(X, 0))
    out["ok"] = (out["profile_eq_residual"] == "0" and out["closure_residual"] == "0"
                 and out["clm_residual"] == "0")
    return out


def hilbert_np(g):
    """Hilbert transform on R of f(xi(theta)) sampled on the theta grid (index 0 = +-pi)."""
    N = g.size
    G = np.fft.rfft(g)
    m = np.full(G.shape, -1j)
    m[0] = 0.0
    m[-1] = 0.0
    h = np.fft.irfft(m * G, n=N)
    return h - h[0]      # constant fixed by H[f](xi -> inf) = 0


def step0_numeric(N=2048):
    th = -np.pi + 2 * np.pi * np.arange(N) / N
    tests = [
        ("H[1/(1+x^2)] = x/(1+x^2), L=1 (Gate)", 1.0,
         lambda x: 1 / (1 + x**2), lambda x: x / (1 + x**2)),
        ("H[x/(1+x^2)] = -1/(1+x^2), L=1", 1.0,
         lambda x: x / (1 + x**2), lambda x: -1 / (1 + x**2)),
        ("H[1/(1+x^2)^2] = x(x^2+3)/(2(1+x^2)^2), L=1", 1.0,
         lambda x: 1 / (1 + x**2)**2, lambda x: x * (x**2 + 3) / (2 * (1 + x**2)**2)),
        ("H[1/(1+x^2)] = x/(1+x^2), L=1.7", 1.7,
         lambda x: 1 / (1 + x**2), lambda x: x / (1 + x**2)),
    ]
    res = []
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for name, L, f, Hf in tests:
            xi = L * np.tan(th / 2)
            g = f(xi)
            g[0] = 0.0
            h = hilbert_np(g)
            res.append((name, float(np.max(np.abs(h[1:] - Hf(xi[1:]))))))
    gate = res[0][1] < 1e-8
    all_ok = all(e < 1e-8 for _, e in res)

    # Exact singular far-field term: H[Im E_s] = -Re E_s, E_s = (L/(L - i xi))^s
    sing = []
    L = 1.0
    for s in (0.5, 1.5):
        f = lambda y, s=s: float(np.imag((L / (L - 1j * y))**s))
        for x0 in (0.3, 1.0, 4.0):
            B = 20.0
            pv = si.quad(f, -B, B, weight="cauchy", wvar=x0, limit=400)[0]
            tail = (si.quad(lambda y: f(y) / (y - x0), B, np.inf, limit=400)[0]
                    + si.quad(lambda y: f(y) / (y - x0), -np.inf, -B, limit=400)[0])
            Hq = -(pv + tail) / np.pi
            He = -float(np.real((L / (L - 1j * x0))**s))
            sing.append((s, x0, abs(Hq - He)))
    # Closed form of U_sing = int_0^xi (-Re E_s) (A = 1) against quadrature
    using = []
    for s in (0.5, 1.0, 1.5):
        for x0 in (0.5, 2.0, 10.0):
            q = si.quad(lambda y: -float(np.real((L / (L - 1j * y))**s)), 0, x0, limit=200)[0]
            thv = 2 * np.arctan(x0 / L)
            cf = -L * np.cos(thv / 2)**(s - 1) * (thv / 2) * np.sinc((1 - s) * thv / (2 * np.pi))
            using.append((s, x0, abs(q - cf)))
    sing_ok = max(e for *_, e in sing) < 1e-6 and max(e for *_, e in using) < 1e-8
    return dict(tests=res, gate=gate, all_ok=all_ok, sing=sing, using=using, sing_ok=sing_ok)


# ----------------------------------------------------------------------------
# PINN machinery (torch)
# ----------------------------------------------------------------------------
class Geo:
    """Uniform theta grid; interior points j = 1..N-1 (j = 0 is xi = +-inf)."""

    def __init__(self, N, device, dtype):
        th = -np.pi + 2 * np.pi * np.arange(N) / N
        t = lambda v: torch.tensor(v, device=device, dtype=dtype)
        self.N = N
        self.th = t(th[1:])
        self.m = t(2 * np.cos(th[1:] / 2)**2 / L_MAP)      # d theta / d xi
        self.sin = t(np.sin(th[1:]))                        # xi * d theta/d xi
        self.dth = 2 * np.pi / N
        self.ic = N // 2 - 1                                # interior index of theta = 0
        self.xi = L_MAP * np.tan(th[1:] / 2)


def hilbert_t(g_full):
    N = g_full.shape[0]
    G = torch.fft.rfft(g_full)
    m = torch.full(G.shape, -1j, dtype=G.dtype, device=G.device)
    m[0] = 0
    m[-1] = 0
    h = torch.fft.irfft(G * m, n=N)
    return h - h[0]


def cumint_odd(h, geo):
    """U(theta) = int_0^theta h dtheta' (h even) by the trapezoidal rule, U odd."""
    ic = geo.ic
    right = h[ic:]
    trap = 0.5 * (right[1:] + right[:-1]) * geo.dth
    Ur = torch.cat([torch.zeros(1, dtype=h.dtype, device=h.device), torch.cumsum(trap, 0)])
    Ul = -Ur[1:ic + 1].flip(0)
    return torch.cat([Ul, Ur])


def residual(Om, Om_th, Psi, U, geo, a, c):
    R = Om + c * geo.sin * Om_th + a * U * geo.m * Om_th - Psi * Om
    dR = geo.m[1:-1] * (R[2:] - R[:-2]) / (2 * geo.dth)
    return R, dR


class Profile(torch.nn.Module):
    """Omega(theta) = A*Im E_s + sin(theta/2) cos(theta/2)^(s+1) * N(cos k theta), odd in xi."""

    def __init__(self, c0, learn_c):
        super().__init__()
        s0 = 1.0 / c0
        layers, width = [torch.nn.Linear(4, 64), torch.nn.Tanh()], 64
        for _ in range(3):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        last = torch.nn.Linear(width, 1)
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.fill_(-L_MAP / 2)
        self.net = torch.nn.Sequential(*layers, last)
        self.s_raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(s0 - 1e-3))))
        self.A = torch.nn.Parameter(torch.tensor(-L_MAP / (2 * s0)))
        self.c = torch.nn.Parameter(torch.tensor(float(c0)), requires_grad=learn_c)

    def s(self):
        return F.softplus(self.s_raw) + 1e-3

    def c_eff(self):
        return self.c.clamp(0.05, 10.0)

    def parts(self, th):
        s = self.s()
        tiny = 1e-300 if th.dtype == torch.float64 else 1e-30
        lc = torch.log(torch.clamp(torch.cos(th / 2), min=tiny))
        Es = torch.exp(s * lc)
        Om_sing = self.A * Es * torch.sin(s * th / 2)
        Psi_sing = -self.A * Es * torch.cos(s * th / 2)
        U_sing = (-self.A * L_MAP * torch.exp((s - 1) * lc) * (th / 2)
                  * torch.sinc((1 - s) * th / (2 * math.pi)))
        feats = torch.stack([torch.cos(k * th) for k in range(1, 5)], dim=-1)
        Om_reg = torch.sin(th / 2) * torch.exp((s + 1) * lc) * self.net(feats).squeeze(-1)
        return Om_sing, Om_reg, Psi_sing, U_sing

    def fields(self, geo, create_graph=True):
        th = geo.th.detach().clone().requires_grad_(True)
        Om_sing, Om_reg, Psi_sing, U_sing = self.parts(th)
        Om = Om_sing + Om_reg
        Om_th, = torch.autograd.grad(Om.sum(), th, create_graph=create_graph)
        zero = torch.zeros(1, dtype=Om.dtype, device=Om.device)
        Psi_reg = hilbert_t(torch.cat([zero, Om_reg]))[1:]
        U_reg = cumint_odd(Psi_reg / geo.m, geo)
        return Om, Om_th, Psi_sing + Psi_reg, U_sing + U_reg

    @torch.no_grad()
    def omega_xi(self, xi):
        p = next(self.parameters())
        th = torch.tensor(2 * np.arctan(np.asarray(xi) / L_MAP), dtype=p.dtype, device=p.device)
        Om_sing, Om_reg, _, _ = self.parts(th)
        return (Om_sing + Om_reg).double().cpu().numpy()


def loss_fn(model, geo, a, create_graph=True):
    c = model.c_eff()
    Om, Om_th, Psi, U = model.fields(geo, create_graph)
    R, dR = residual(Om, Om_th, Psi, U, geo, a, c)
    # Residuals relative to the profile size: an absolute mean(R^2) can be driven
    # to zero by shrinking any profile (width w: mean R^2 ~ w^3), see NOTES.md.
    scale = (Om**2).mean()
    res_abs = (R**2).mean()
    res = res_abs / scale
    dres = (dR**2).mean() / scale
    om_xi0 = (2.0 / L_MAP) * Om_th[geo.ic]
    norm = (om_xi0 + 1.0)**2
    loss = res + W_DR * dres + W_NORM * norm
    return loss, dict(res=res, dres=dres, norm=norm, om_xi0=om_xi0, res_abs=res_abs), Om_th


def count_zeros_dOmega(Om_th, geo):
    v = (Om_th * geo.m).detach().double().cpu().numpy()[geo.ic + 1:]
    xi = geo.xi[geo.ic + 1:]
    v = v[xi <= 1e3]
    if v.size == 0:
        return 0
    v = v[np.abs(v) > 1e-8 * np.max(np.abs(v))]
    return int(np.sum(np.sign(v[1:]) != np.sign(v[:-1])))


def train_run(a, c0, learn_c, dtype, device, cfg, init_state=None, hard_stop=None,
              adam_steps=None):
    t0 = time.time()
    torch.manual_seed(0)
    geo = Geo(cfg["N_PINN"], device, dtype)
    model = Profile(c0, learn_c).to(device=device, dtype=dtype)
    if init_state is not None:
        model.load_state_dict(init_state)
        with torch.no_grad():
            model.c.fill_(float(c0))
    model.c.requires_grad_(learn_c)
    params = [p for p in model.parameters() if p.requires_grad]
    steps = cfg["ADAM"] if adam_steps is None else adam_steps
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1**(1.0 / max(1, steps)))
    status = "ok"
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = loss_fn(model, geo, a)
        if not torch.isfinite(loss):
            status = "nan_adam"
            break
        loss.backward()
        opt.step()
        sched.step()
        if hard_stop is not None and it % 50 == 0 and time.time() > hard_stop:
            status = "timeout_adam"
            break
    if status == "ok" and cfg["LBFGS"] > 0 and adam_steps is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_loss = float(loss_fn(model, geo, a, create_graph=False)[0])
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg["LBFGS"], history_size=50,
                               tolerance_grad=1e-14, tolerance_change=1e-16,
                               line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad(set_to_none=True)
            l, _, _ = loss_fn(model, geo, a)
            if torch.isfinite(l):
                l.backward()
            return l
        try:
            lb.step(closure)
        except Exception as e:  # noqa: BLE001
            status = f"lbfgs_error:{type(e).__name__}"
        new_loss = float(loss_fn(model, geo, a, create_graph=False)[0])
        if not np.isfinite(new_loss) or new_loss > best_loss:
            model.load_state_dict(best_state)
            status = "ok_lbfgs_rejected" if status == "ok" else status
    loss, parts, Om_th = loss_fn(model, geo, a, create_graph=False)
    row = dict(a=a, c_init=c0, c=float(model.c_eff()), final_loss=float(loss),
               loss_res=float(parts["res"]), loss_dres=float(parts["dres"]),
               loss_norm=float(parts["norm"]), omega_xi0=float(parts["om_xi0"]),
               loss_res_abs=float(parts["res_abs"]),
               s=float(model.s()), one_over_c=1.0 / float(model.c_eff()), A=float(model.A),
               n_zeros_dOmega=count_zeros_dOmega(Om_th, geo) if torch.isfinite(loss) else -1,
               status=status, runtime_s=time.time() - t0)
    prof = model.omega_xi(XI_PLOT)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return row, state, prof


def selftest_pipeline(device):
    """CLM (a=0, c=1) exact profile through both torch paths; U against 4*atan(xi/2)."""
    out = {}
    dtype = torch.float64
    geo = Geo(2048, device, dtype)
    th = geo.th
    # (i) regular path only: Omega = -4 xi/(4+xi^2) = -4 sin/(5+3cos) for L=1
    Om = -4 * torch.sin(th) / (5 + 3 * torch.cos(th))
    Om_th = -4 * (5 * torch.cos(th) + 3) / (5 + 3 * torch.cos(th))**2
    zero = torch.zeros(1, dtype=dtype, device=device)
    Psi = hilbert_t(torch.cat([zero, Om]))[1:]
    U = cumint_odd(Psi / geo.m, geo)
    R, _ = residual(Om, Om_th, Psi, U, geo, 0.0, 1.0)
    xi = torch.tensor(geo.xi, dtype=dtype, device=device)
    core = torch.abs(xi) <= 50
    out["reg_path_max_R"] = float(R.abs().max())
    out["reg_path_U_err_core"] = float((U - 4 * torch.atan(xi / 2))[core].abs().max())
    # (ii) singular path: A=-2, s=1, net == 0  ->  Omega = -2 xi/(1+xi^2)
    model = Profile(1.0, False).to(device=device, dtype=dtype)
    with torch.no_grad():
        model.s_raw.fill_(math.log(math.expm1(1.0 - 1e-3)))
        model.A.fill_(-2.0)
        model.net[-1].weight.zero_()
        model.net[-1].bias.zero_()
    Om2, Om2_th, Psi2, U2 = model.fields(geo, create_graph=False)
    R2, _ = residual(Om2, Om2_th, Psi2, U2, geo, 0.0, 1.0)
    out["sing_path_max_R"] = float(R2.abs().max())
    out["sing_path_U_err_core"] = float((U2 - 2 * torch.atan(xi))[core].abs().max())
    out["ok"] = (out["reg_path_max_R"] < 1e-9 and out["sing_path_max_R"] < 1e-9
                 and out["reg_path_U_err_core"] < 1e-4 and out["sing_path_U_err_core"] < 1e-9)
    return out


# ----------------------------------------------------------------------------
# Step 1: pseudospectral time integration (numpy, FP64)
# ----------------------------------------------------------------------------
def simulate(a, N, cfg, keep_snaps, tol=1e-8):
    x = -np.pi + 2 * np.pi * np.arange(N) / N
    dx = 2 * np.pi / N
    k = np.arange(N // 2 + 1, dtype=float)
    mH = np.where(k > 0, -1j, 0).astype(complex)
    mU = np.zeros_like(k, dtype=complex)
    mU[1:] = -1.0 / k[1:]
    mD = 1j * k
    for m_ in (mH, mU, mD):
        m_[-1] = 0
    filt = np.exp(-36.0 * (k / (N // 2))**36)
    i0, x0 = N // 4, -np.pi / 2
    khi = int(N / 3)
    irf = lambda v: np.fft.irfft(v, n=N)

    def rhs(wh):
        w = irf(wh)
        return np.fft.rfft(-a * irf(mU * wh) * irf(mD * wh) + irf(mH * wh) * w)

    wh = np.fft.rfft(-np.cos(x))
    t, rows, snaps, next_snap, tw, reason = 0.0, [], [], 1.0, time.time(), ""
    while True:
        w = irf(wh)
        aw = np.abs(w)
        i = int(np.argmax(aw))
        M = aw[i]
        wx0 = irf(mD * wh)[i0]
        ratio = np.max(np.abs(wh[khi:])) / np.max(np.abs(wh))
        fm, f0, fp = aw[i - 1], aw[i], aw[(i + 1) % N]
        den = fm - 2 * f0 + fp
        d = 0.5 * (fm - fp) / den if den < 0 else 0.0
        ell = abs((x[i] + d * dx - x0 + np.pi) % (2 * np.pi) - np.pi)
        rows.append((t, M, ell, wx0, ratio))
        if keep_snaps and M >= next_snap:
            snaps.append((t, w.copy(), wx0, M))
            next_snap = M * 1.2
        if not np.isfinite(M):
            reason = "NaN"
        elif M >= cfg["SIM_MSTOP"]:
            reason = "max|omega| >= 1e4"
        elif ratio > tol:
            reason = f"Aufloesung (Spektrum bei k>N/3 > {tol:g})"
        elif t > cfg["SIM_TMAX"]:
            reason = "t_max erreicht (kein Blow-up erkannt)"
        elif time.time() - tw > cfg["SIM_WALL"]:
            reason = "Zeitbox"
        if reason:
            break
        dt = 0.02 / M
        if a != 0:
            # advection CFL for explicit RK4 (the 0.02/max|omega| rule alone is unstable for a > 0)
            umax = np.max(np.abs(irf(mU * wh)))
            dt = min(dt, 1.0 / (abs(a) * max(umax, 1e-12) * (N // 2)))
        k1 = rhs(wh)
        k2 = rhs(wh + 0.5 * dt * k1)
        k3 = rhs(wh + 0.5 * dt * k2)
        k4 = rhs(wh + dt * k3)
        wh = (wh + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)) * filt
        wh[-1] = 0
        t += dt
    arr = np.array(rows)
    return dict(a=a, N=N, rows=arr, snaps=snaps, reason=reason, x=x, x0=x0,
                steps=len(rows), wall=time.time() - tw)


def fit_sim(sim):
    t, M, ell, wx0 = sim["rows"][:, 0], sim["rows"][:, 1], sim["rows"][:, 2], sim["rows"][:, 3]
    Mend = M[-1]
    out = dict(M_end=float(Mend), t_end=float(t[-1]), ok=False, note="")
    if not np.isfinite(Mend) or Mend < 20:
        out["note"] = "kein Blow-up (max|omega| < 20)"
        return out
    w = M >= Mend / 20
    if w.sum() < 20:
        out["note"] = f"zu wenige Punkte im Fit-Fenster ({int(w.sum())})"
        return out
    p = np.polyfit(t[w], 1 / M[w], 1)
    if p[0] >= 0:
        out["note"] = "1/max|omega| faellt nicht"
        return out
    T = -p[1] / p[0]
    tau = T - t[w]
    v = (tau > 0) & (ell[w] > 0)
    lt = np.log(tau[v])
    out.update(T=float(T), c=float(np.polyfit(lt, np.log(ell[w][v]), 1)[0]),
               c_alt=float(-np.polyfit(lt, np.log(np.abs(wx0[w][v])), 1)[0] - 1),
               gamma=float(np.polyfit(lt, np.log(M[w][v]), 1)[0]),
               n_fit=int(v.sum()), tau_min=float(tau[v].min()), tau_max=float(tau[v].max()),
               ell_end=float(ell[-1]), ok=True)
    return out


def sim_profile(sim, fit, target):
    """Rescaled profile tau*omega(x0 + lam*xi), lam fixed by Omega'(0) = -1."""
    snaps = [s for s in sim["snaps"] if s[3] >= fit["M_end"] / 20]
    if not snaps:
        return None
    ts, w, wx0, M = min(snaps, key=lambda s: abs(math.log(s[3] / target)))
    tau = fit["T"] - ts
    lam = -1.0 / (tau * wx0)
    P = tau * np.interp(sim["x0"] + lam * XI_PLOT, sim["x"], w, period=2 * np.pi)
    return dict(P=P, M=float(M), tau=float(tau))


def rel_l2(p, q):
    return float(np.linalg.norm(p - q) / max(np.linalg.norm(q), 1e-300))


# ----------------------------------------------------------------------------
# Step 2 helpers
# ----------------------------------------------------------------------------
def pick_device(req, cfg):
    info = {}
    if req != "auto":
        return req, info
    if not torch.cuda.is_available():
        info["note"] = "CUDA nicht verfuegbar -> CPU"
        return "cpu", info
    try:
        torch.ones(8, device="cuda", dtype=torch.float64).sum().item()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["n_gpu"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        info["note"] = f"CUDA-Test fehlgeschlagen ({e}) -> CPU"
        return "cpu", info
    times = {}
    for dev in ("cuda", "cpu"):
        try:
            train_run(0.0, 1.0, False, torch.float64, dev, cfg, adam_steps=3)
            t0 = time.time()
            train_run(0.0, 1.0, False, torch.float64, dev, cfg, adam_steps=cfg["BENCH_STEPS"])
            times[dev] = (time.time() - t0) / cfg["BENCH_STEPS"]
        except Exception as e:  # noqa: BLE001
            times[dev] = float("inf")
            info[f"bench_error_{dev}"] = str(e)
    info["s_per_adam_step"] = times
    return min(times, key=times.get), info


def local_minima(cs, losses):
    cs, losses = np.asarray(cs, float), np.asarray(losses, float)
    ok = np.isfinite(losses)
    cs, losses = cs[ok], losses[ok]
    o = np.argsort(cs)
    cs, losses = cs[o], losses[o]
    n = len(cs)
    mins = []
    for i in range(n):
        if (i == 0 or losses[i] < losses[i - 1]) and (i == n - 1 or losses[i] < losses[i + 1]):
            mins.append((float(cs[i]), float(losses[i]), i in (0, n - 1)))
    return sorted(mins, key=lambda m: m[1])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--replot", metavar="DIR", help="only redraw loss_vs_c.png in DIR")
    args, _ = ap.parse_known_args()
    if args.replot:
        replot(args.replot)
        return
    cfg = make_config(args.smoke)
    out = args.out or ("/kaggle/working/results" if os.path.isdir("/kaggle/working")
                       else ("results_smoke" if args.smoke else "results"))
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    hard_stop = T_START + cfg["DEADLINE"]
    timing = {}
    log(f"Konfiguration: {'SMOKE' if args.smoke else 'FULL'}, Ausgabe: {out}")

    # ---------------- Step 0 ----------------
    t0 = time.time()
    s0 = {}
    for name, fn in (("sympy", step0_sympy), ("numeric", step0_numeric),
                     ("pipeline", lambda: selftest_pipeline("cpu"))):
        try:
            s0[name] = fn()
            log(f"Schritt 0 {name}: ok={s0[name].get('ok', s0[name].get('all_ok'))}")
        except Exception as e:  # noqa: BLE001
            record_error(f"Schritt 0 {name}", e)
    timing["step0_s"] = time.time() - t0
    gate = bool(s0.get("numeric", {}).get("gate", False))
    if not gate:
        ERRORS.append("Schritt 0: Hilbert-Gate (Fehler < 1e-8) VERFEHLT")

    # ---------------- Step 1 ----------------
    t1 = time.time()
    sims, fits, simprof, cref = {}, {}, {}, {}
    for a in cfg["A_LIST"]:
        nf = cfg["SIM_N"][1] if a == 0 else cfg["SIM_N_ADV"]   # a > 0: CFL-limited dt
        for N in (cfg["SIM_N"][0], nf):
            try:
                sim = simulate(a, N, cfg, keep_snaps=(N == nf))
                fit = fit_sim(sim)
                sims[(a, N)], fits[(a, N)] = sim, fit
                log(f"Schritt 1 a={a} N={N}: {sim['steps']} Schritte, {sim['wall']:.1f}s, "
                    f"Stopp: {sim['reason']}, max|w|={fit['M_end']:.3g}, "
                    f"T={fit.get('T', float('nan')):.5f}, c={fit.get('c', float('nan')):.4f}, "
                    f"c_alt={fit.get('c_alt', float('nan')):.4f} {fit['note']}")
            except Exception as e:  # noqa: BLE001
                record_error(f"Schritt 1 a={a} N={N}", e)
        fine, spec = fits.get((a, nf), {}), fits.get((a, cfg["SIM_N"][0]), {})
        src = fine if fine.get("ok") else spec
        cref[a] = src.get("c", float("nan")) if src.get("ok") else float("nan")
        if fine.get("ok"):
            p4 = sim_profile(sims[(a, nf)], fine, fine["M_end"] / 4)
            p16 = sim_profile(sims[(a, nf)], fine, fine["M_end"] / 16)
            if p4 is not None:
                simprof[a] = p4
                if p16 is not None:
                    simprof[a]["selfsim_relL2"] = rel_l2(p16["P"], p4["P"])
    sim_rows = []
    for (a, N), f in fits.items():
        sim_rows.append(dict(a=a, N=N, reason=sims[(a, N)]["reason"], steps=sims[(a, N)]["steps"],
                             **{k: v for k, v in f.items()}))
    pd.DataFrame(sim_rows).to_csv(os.path.join(out, "sim_summary.csv"), index=False)
    np.savez(os.path.join(out, "sim_profiles.npz"), xi=XI_PLOT,
             **{f"a{a}": p["P"] for a, p in simprof.items()})
    timing["step1_s"] = time.time() - t1

    # ---------------- Step 2 ----------------
    t2 = time.time()
    runs, states, profiles = [], {}, {}
    thin_info = {}
    dev, dev_info = "cpu", {}
    try:
        dev, dev_info = pick_device(args.device, cfg)
        log(f"Geraet: {dev} {dev_info}")
    except Exception as e:  # noqa: BLE001
        record_error("Geraetewahl", e)
    step2_end = min(time.time() + cfg["STEP2_BUDGET"], hard_stop - cfg["RESERVE"])
    est = {"t": None}

    def do_run(phase, a, c0, dtype, learn=False, init=None):
        if est["t"] is not None and time.time() + est["t"] > step2_end:
            log(f"uebersprungen (Zeit): {phase} a={a} c={c0:.4f}")
            return None
        rid = len(runs)
        try:
            row, state, prof = train_run(a, c0, learn, dtype, dev, cfg, init_state=init,
                                         hard_stop=hard_stop - cfg["RESERVE"])
        except Exception as e:  # noqa: BLE001
            record_error(f"Lauf {phase} a={a} c={c0:.4f}", e)
            row, state, prof = dict(a=a, c_init=c0, c=float("nan"), final_loss=float("nan"),
                                    status=f"error:{type(e).__name__}", runtime_s=0.0), None, None
        row = dict(run_id=rid, phase=phase, dtype=str(dtype).replace("torch.", ""),
                   device=dev, **row)
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path),
                                   index=False)
        runs.append(row)
        if state is not None:
            states[rid], profiles[rid] = state, prof
        n = len(runs)
        est["t"] = row["runtime_s"] if est["t"] is None else \
            max(est["t"] * (n - 1) / n + row["runtime_s"] / n, row["runtime_s"])
        log(f"{phase:9s} a={a:<4} c0={c0:7.4f} -> c={row['c']:.4f} loss={row['final_loss']:.3e} "
            f"({row.get('status')}, {row['runtime_s']:.1f}s)")
        return row

    grid = list(np.geomspace(0.2, 3.0, cfg["NC"]))
    alist = cfg["A_LIST"]
    try:
        # first full run: a0 at c_ref (always kept) -> timing -> thin the grid
        a0 = alist[0]
        first_c = cref[a0] if np.isfinite(cref.get(a0, np.nan)) else grid[0]
        do_run("scan", a0, first_c, torch.float64)
        t_run = max(runs[0]["runtime_s"], 1e-3)
        n_learn_est = 3 * len(alist)
        budget = step2_end - time.time()
        n_planned = len(alist) * (cfg["NC"] + 1) + (cfg["NC"] + 1) + n_learn_est - 1
        thin_info = dict(t_first_run_s=t_run, budget_s=budget, n_planned=n_planned,
                         est_total_s=n_planned * t_run, nc_before=cfg["NC"])
        if n_planned * t_run > budget:
            nc = int((budget / t_run - n_learn_est) / (len(alist) + 1)) - 1
            nc = max(3, min(cfg["NC"], nc))
            idx = np.unique(np.round(np.linspace(0, cfg["NC"] - 1, nc)).astype(int))
            grid = [grid[i] for i in idx]
        thin_info["nc_after"] = len(grid)
        log(f"c-Gitter: {len(grid)} Werte (+ c_ref); Plan {thin_info}")
        for a in alist:
            cs = ([] if a == a0 else [cref[a]] if np.isfinite(cref.get(a, np.nan)) else []) + grid
            if a == a0 and not np.isfinite(cref.get(a0, np.nan)):
                cs = grid[1:]
            for c0 in cs:
                do_run("scan", a, c0, torch.float64)
        # learnable c, started at every local minimum of the FP64 scan
        for a in alist:
            sc = [r for r in runs if r["phase"] == "scan" and r["a"] == a]
            mins = local_minima([r["c_init"] for r in sc], [r["final_loss"] for r in sc])
            log(f"a={a}: lokale Minima (c, loss, Rand): {mins}")
            for cm, _, _ in mins:
                src = min(sc, key=lambda r: abs(r["c_init"] - cm))
                do_run("learn_c", a, cm, torch.float64, learn=True,
                       init=states.get(src["run_id"]))
        # FP32 control at a = 0 with the same c list
        if 0.0 in alist:
            for c0 in sorted(r["c_init"] for r in runs if r["phase"] == "scan" and r["a"] == 0.0):
                do_run("scan_fp32", 0.0, c0, torch.float32)
    except Exception as e:  # noqa: BLE001
        record_error("Schritt 2", e)
    timing["step2_s"] = time.time() - t2
    np.savez(os.path.join(out, "pinn_profiles.npz"), xi=XI_PLOT,
             **{f"run{k}": v for k, v in profiles.items()})

    # ---------------- Step 3 ----------------
    t3 = time.time()
    try:
        evaluate(out, cfg, runs, profiles, cref, fits, simprof, s0, gate, dev, dev_info,
                 thin_info, timing, args.smoke)
    except Exception as e:  # noqa: BLE001
        record_error("Schritt 3", e)
        with open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8") as fh:
            fh.write("# SUMMARY (Notfall)\n\nAuswertung fehlgeschlagen:\n\n"
                     + "\n".join(f"- {e}" for e in ERRORS))
    log(f"Fertig in {(time.time() - T_START) / 60:.2f} min (Schritt 3: {time.time() - t3:.1f}s)")


# ----------------------------------------------------------------------------
# Step 3: evaluation, plots, SUMMARY.md, NOTES.md
# ----------------------------------------------------------------------------
def plot_loss_vs_c(out, df, cref, alist):
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in alist:
        col = COLORS.get(a, "#555555")
        for phase, ls, mk, lab in (("scan", "-", "o", f"a={a} (FP64)"),
                                   ("scan_fp32", "--", "s", f"a={a} (FP32-Kontrolle)")):
            d = df[(df.phase == phase) & (df.a == a)].sort_values("c_init") if len(df) else df
            if len(d):
                ax.plot(d.c_init, d.final_loss, ls=ls, marker=mk, ms=4, lw=1.5, color=col,
                        mfc=col if phase == "scan" else "none", label=lab)
        if np.isfinite(cref.get(a, np.nan)):
            ax.axvline(cref[a], color=col, ls=":", lw=1.2)
            ax.text(cref[a], 0.97, f" c_ref={cref[a]:.3f}", color="#333333", fontsize=8,
                    rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())
        d = df[(df.phase == "learn_c") & (df.a == a)] if len(df) else df
        if len(d):
            ax.plot(d.c, d.final_loss, ls="none", marker="*", ms=11, color=col,
                    mec="white", mew=0.8, label=f"a={a} lernbares c")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Skalierungsexponent c")
    ax.set_ylabel("finaler Loss")
    ax.set_title("PINN-Loss gegen c (gepunktete Senkrechte: c_ref aus Zeitintegration)")
    ax.grid(True, which="major", color="#dddddd", lw=0.6)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "loss_vs_c.png"), dpi=150)
    plt.close(fig)


def replot(out):
    """Rebuild loss_vs_c.png from results.csv + sim_summary.csv (no recomputation)."""
    df = pd.read_csv(os.path.join(out, "results.csv"))
    sim = pd.read_csv(os.path.join(out, "sim_summary.csv"))
    cref = {}
    for a, g in sim.groupby("a"):
        g = g[g.ok == True].sort_values("N")  # noqa: E712
        if len(g):
            cref[float(a)] = float(g.iloc[-1]["c"])
    plot_loss_vs_c(out, df, cref, sorted(df.a.unique().tolist()))


def evaluate(out, cfg, runs, profiles, cref, fits, simprof, s0, gate, dev, dev_info,
             thin_info, timing, smoke):
    df = pd.DataFrame(runs)
    alist = cfg["A_LIST"]
    exact0 = -4 * XI_PLOT / (4 + XI_PLOT**2)
    for r in runs:
        rid = r["run_id"]
        if rid in profiles:
            r["relL2_sim"] = (rel_l2(profiles[rid], simprof[r["a"]]["P"])
                              if r["a"] in simprof else float("nan"))
            r["relL2_exact"] = rel_l2(profiles[rid], exact0) if r["a"] == 0.0 else float("nan")
    df = pd.DataFrame(runs)
    df.to_csv(os.path.join(out, "results_evaluated.csv"), index=False)

    # ---- plots ----
    plot_loss_vs_c(out, df, cref, alist)

    cand = {}
    for a in alist:
        d = [r for r in runs if r["phase"] == "learn_c" and r["a"] == a and r["run_id"] in profiles
             and np.isfinite(r.get("c", np.nan))]
        cand[a] = d
    fig, axes = plt.subplots(1, len(alist), figsize=(5.2 * len(alist), 4.2), squeeze=False)
    for ax, a in zip(axes[0], alist):
        col = COLORS.get(a, "#555555")
        if a in simprof:
            ax.plot(XI_PLOT[::20], simprof[a]["P"][::20], "o", ms=4, color="#555555",
                    mfc="none", label="Zeitintegral (reskaliert)")
        if a == 0.0:
            ax.plot(XI_PLOT, exact0, ":", color="#111111", lw=1.2, label="exakt CLM, c=1")
        ref = cref.get(a, np.nan)
        dd = sorted(cand[a], key=lambda r: abs(r["c"] - ref) if np.isfinite(ref) else r["final_loss"])
        for j, r in enumerate(dd[:4]):
            ax.plot(XI_PLOT, profiles[r["run_id"]], ls="-" if j == 0 else "--",
                    lw=2 if j == 0 else 1.2, color=col, alpha=1 if j == 0 else 0.6,
                    label=f"PINN c={r['c']:.3f}, R²={r['loss_res']:.1e}")
        ax.set_title(f"a = {a}")
        ax.set_xlabel("ξ")
        ax.set_ylabel("Ω(ξ)  (Ω'(0) = -1)")
        ax.grid(True, color="#dddddd", lw=0.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "profiles.png"), dpi=150)
    plt.close(fig)

    # ---- SUMMARY.md ----
    yn = lambda b: "Ja" if b else "Nein"
    f6 = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.4g}"
    L = [f"# SUMMARY — gCLM Blow-up-Profile per PINN{' (SMOKE-TEST)' if smoke else ''}", ""]
    L += [f"Laufzeit gesamt {(time.time() - T_START) / 60:.1f} min; Gerät: {dev} {dev_info}.",
          f"Schritt 0 {timing.get('step0_s', 0):.0f}s, Schritt 1 {timing.get('step1_s', 0):.0f}s, "
          f"Schritt 2 {timing.get('step2_s', 0) / 60:.1f} min. Ausdünnung: {thin_info}", ""]
    L += ["## Schritt 0", "",
          f"- Hilbert-Gate H[1/(1+x²)] < 1e-8: **{yn(gate)}**"]
    num = s0.get("numeric", {})
    for name, e in num.get("tests", []):
        L.append(f"  - {name}: max. Fehler {e:.2e}")
    if num:
        L.append(f"- Exakter Fernfeldterm H[Im E_s] = -Re E_s (Quadratur, s=0.5/1.5): "
                 f"max. Fehler {max(e for *_, e in num['sing']):.2e}; "
                 f"U_sing geschlossen vs. Quadratur {max(e for *_, e in num['using']):.2e}")
    sy = s0.get("sympy", {})
    if sy:
        L.append(f"- sympy: Profilgleichung bestätigt (Differenz = {sy['profile_eq_residual']}), "
                 f"Schließung U' = H[Ω] (Differenz = {sy['closure_residual']}), "
                 f"exaktes CLM-Profil Ω = -4ξ/(4+ξ²) erfüllt R = {sy['clm_residual']}.")
    pl = s0.get("pipeline", {})
    if pl:
        L.append(f"- Selbsttest Residuum-Pipeline mit exaktem CLM-Profil: max|R| regulär "
                 f"{pl['reg_path_max_R']:.1e}, singulär {pl['sing_path_max_R']:.1e}; "
                 f"U-Fehler {pl['reg_path_U_err_core']:.1e} / {pl['sing_path_U_err_core']:.1e} "
                 f"→ {yn(pl['ok'])}")
    L += ["- Abweichungen von der Aufgabenstellung (Parität, Normierung, Fernfeld): siehe NOTES.md.", ""]
    L += ["## Schritt 1 — Referenz durch Zeitintegration", "",
          "| a | N | Stopp | max|ω| | T | c (Kernbreite) | c_alt (ω_x(x0)) | γ (max|ω|~τ^γ) |",
          "|---|---|---|---|---|---|---|---|"]
    for (a, N), f in sorted(fits.items()):
        L.append(f"| {a} | {N} | {f.get('note') or '-'} | {f6(f.get('M_end'))} | {f6(f.get('T'))} "
                 f"| {f6(f.get('c'))} | {f6(f.get('c_alt'))} | {f6(f.get('gamma'))} |")
    L.append("")
    for a in alist:
        sp_ = simprof.get(a)
        if sp_:
            L.append(f"- a={a}: Selbstähnlichkeit des reskalierten Profils (max|ω|={sp_['M']:.0f} vs. "
                     f"Viertel davon): relL2 = {sp_.get('selfsim_relL2', float('nan')):.3f}")
    L.append("")
    L += ["## Schritt 2/3 — Ergebnisse je a", ""]
    for a in alist:
        ref = cref.get(a, np.nan)
        dd = cand[a]
        best = min(dd, key=lambda r: abs(r["c"] - ref)) if dd and np.isfinite(ref) else None
        c_ok = best is not None and abs(best["c"] - ref) < C_TOL
        p_ok = best is not None and np.isfinite(best.get("relL2_sim", np.nan)) \
            and best["relL2_sim"] < PROF_TOL
        fixed = [r for r in runs if r["phase"] == "scan" and r["a"] == a
                 and np.isfinite(ref) and abs(r["c_init"] - ref) < 1e-12]
        others = []
        for r in sorted(dd, key=lambda r: r["final_loss"]):
            if best is not None and r is best:
                continue
            if np.isfinite(ref) and abs(r["c"] - ref) < C_TOL:
                continue
            if any(abs(r["c"] - o["c"]) < C_TOL for o in others):
                continue
            others.append(r)
        small = [r for r in others if r.get("loss_res", np.inf) < RES_SMALL]
        c_txt = "n/a" if best is None else "c={:.4f}, Δ={:+.4f}".format(best["c"], best["c"] - ref)
        p_txt = "n/a" if best is None else f6(best.get("relL2_sim"))
        L += [f"### a = {a}", "",
              "| Kriterium | Wert | Ja/Nein |", "|---|---|---|",
              f"| Referenz c_ref aus Zeitintegration | {f6(ref)} | {yn(np.isfinite(ref))} |",
              f"| Lernbares c konvergiert nahe c_ref (|c−c_ref| < {C_TOL}) | {c_txt} | {yn(c_ok)} |",
              f"| Profil = reskaliertes Zeitintegral (relL2 < {PROF_TOL}, |ξ|≤10) | {p_txt} | {yn(p_ok)} |",
              f"| **Stabiles Profil bestätigt** (beide Kriterien) | | **{yn(c_ok and p_ok)}** |",
              f"| Weitere Minima mit kleinem Residuum (mean R²/mean Ω² < {RES_SMALL:g}) | {len(small)} | "
              f"{yn(len(small) > 0)} |"]
        if a == 0.0 and best is not None:
            L.append(f"| Vergleich mit exaktem CLM-Profil (c=1) | relL2 = {f6(best.get('relL2_exact'))}, "
                     f"|c−1| = {abs(best['c'] - 1):.4f} | {yn(abs(best['c'] - 1) < C_TOL and best.get('relL2_exact', 1) < PROF_TOL)} |")
        if fixed:
            fr = fixed[0]
            L.append(f"| (Info) Lauf mit festem c = c_ref | loss={fr['final_loss']:.2e}, "
                     f"relL2 zum Zeitintegral {f6(fr.get('relL2_sim'))} | – |")
        L += ["", "Läufe mit lernbarem c (Start an lokalen Minima des Scans):", "",
              "| c_start | c_final | final loss | mean R²/mean Ω² | mean R² (abs) | mean (dR/dξ)²/mean Ω² | Norm.-Strafe | s (Fernfeld) | 1/c | Nullstellen Ω' (ξ>0) | Status |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in dd:
            L.append(f"| {r['c_init']:.4f} | {r['c']:.4f} | {r['final_loss']:.2e} | {r['loss_res']:.2e} "
                     f"| {r['loss_res_abs']:.2e} | {r['loss_dres']:.2e} | {r['loss_norm']:.1e} | {r['s']:.3f} | {r['one_over_c']:.3f} "
                     f"| {r['n_zeros_dOmega']} | {r['status']} |")
        if not dd:
            L.append("| – | keine Läufe | | | | | | | | | |")
        if others:
            L += ["", "Weitere Minima (Kandidaten für instabile Profile, **nicht bewiesen**): " +
                  "; ".join(f"c={r['c']:.4f} (mean R²/mean Ω²={r['loss_res']:.1e}, {r['n_zeros_dOmega']} "
                            f"Nullst. von Ω')" for r in others)]
        if a == 0.0:
            L += ["", "Hinweis a=0: Nach der Herleitung in NOTES.md (komplexe Riccati-Gleichung) hat "
                  "CLM nur **ein** glattes, abklingendes selbstähnliches Profil (c=1). Jedes weitere "
                  "Minimum bei a=0 ist daher ein Artefakt der Methode, kein instabiles Profil."]
        sc = df[(df.phase == "scan") & (df.a == a)].sort_values("c_init") if len(df) else df
        if len(sc):
            L += ["", "Scan (festes c): " + ", ".join(f"{c:.3f}→{l:.1e}" for c, l in
                                                      zip(sc.c_init, sc.final_loss))]
        L.append("")
    # FP32 control
    L += ["## Kontrolle FP32 (a = 0)", ""]
    if len(df) and (df.phase == "scan_fp32").any():
        m64 = local_minima(df[(df.phase == "scan") & (df.a == 0.0)].c_init,
                           df[(df.phase == "scan") & (df.a == 0.0)].final_loss)
        m32 = local_minima(df[df.phase == "scan_fp32"].c_init, df[df.phase == "scan_fp32"].final_loss)
        L += [f"- Lokale Minima FP64: {len(m64)} bei c = {[round(m[0], 4) for m in m64]}",
              f"- Lokale Minima FP32: {len(m32)} bei c = {[round(m[0], 4) for m in m32]}",
              f"- Minima in beiden (gleiches c): "
              f"{len(set(round(m[0], 6) for m in m64) & set(round(m[0], 6) for m in m32))}"]
    else:
        L.append("- Nicht gelaufen (Zeit oder Fehler).")
    L += ["", "## Fehler und Auslassungen", ""]
    skipped = [r for r in runs if str(r.get("status", "")).startswith(("error", "nan", "timeout"))]
    L += [f"- {e}" for e in ERRORS] or ["- keine Laufzeitfehler"]
    L += [f"- Lauf {r['run_id']} ({r['phase']}, a={r['a']}, c={r['c_init']:.4f}): {r['status']}"
          for r in skipped]
    with open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    write_notes(out, s0)


NOTES_TEXT = r"""# NOTES — Mathematik, Abweichungen, Annahmen

## 1. Profilgleichung (Schritt 0, sympy)
Einsetzen von ω = τ⁻¹Ω(ξ), ξ = x/τ^c, u = τ^(c−1)U(ξ), τ = T−t in
ω_t + a·u·ω_x − u_x·ω liefert nach Multiplikation mit τ² exakt
R = Ω + cξΩ' + aUΩ' − U'Ω, und u_x = H[ω] wird wegen der Skaleninvarianz der
Hilbert-Transformation zu U' = H[Ω]. **Die Gleichung ist korrekt.**

## 2. Korrektur: Ω ist ungerade, nicht gerade
Für gerades Ω ist H[Ω] ungerade, U gerade (U(0)=0), Ω' ungerade. Dann ist
Ω + cξΩ' gerade und aUΩ' − H[Ω]Ω ungerade; R = 0 verlangt beide Teile = 0.
Der gerade Teil Ω + cξΩ' = 0 hat nur Ω = K·|ξ|^(−1/c) (sympy-dsolve), singulär
bei ξ = 0. **Es gibt kein glattes gerades Profil außer Ω ≡ 0.** Für ungerades Ω
sind alle Terme ungerade — konsistent.
Physikalisch: Für ω0 = −cos x ist der Blow-up-Punkt x0 = −π/2 (dort ω0 = 0 und
H[ω0] = −sin x maximal = 1; CLM: T = 2/1 = 2). Um x0 gilt ω0 = −sin(x − x0):
ungerade. Die Symmetrie bleibt unter gCLM erhalten (u = −|∂|⁻¹ω ist dann
ebenfalls ungerade um x0, also u(x0) = 0).

## 3. Korrektur: Normierung
Ω(1) = Ω(0)/2 degeneriert für ungerades Ω (Ω(0) = 0 ⇒ Ω(1) = 0). Ersetzt durch
**Ω'(0) = −1** als Strafterm (Gewicht 1). Auch das bricht die Skalensymmetrie
Ω(ξ) → Ω(λξ) (die Amplitude ist durch die Gleichung fixiert). Vorzeichen: ω0
fällt bei x0. Einschränkung: Profile mit Ω'(0) = 0 (höhere Nullstelle) sind
damit ausgeschlossen.

## 4. Referenzlösung a = 0 (analytisch, eigene Herleitung)
F = Ω + iH[Ω] ist Randwert einer in der oberen Halbebene analytischen Funktion.
Wegen ΩH[Ω] = Im(F²)/2 und H[Re F²] = Im F² ist R = Re G mit
G = F + cξF' + (i/2)F², G analytisch und abklingend ⇒ G ≡ 0. Lösung:
F = K/(ξ^(1/c) − iK/2). Glattheit bei ξ = 0 verlangt 1/c = n ∈ ℕ; keine Pole in
der oberen Halbebene nur für n = 1. Also: **CLM hat genau ein glattes
selbstähnliches Profil, c = 1, Ω = −2ξ/(1+ξ²)** (normiert: −4ξ/(4+ξ²)).
Das dient als Test der Residuum-Pipeline und als Kontrolle: weitere PINN-Minima
bei a = 0 sind Artefakte. Nicht aus der Literatur geprüft.

## 5. Hilbert-Transformation auf ℝ
ξ = L·tan(θ/2) ist die Cayley-Abbildung; analytische Funktionen der oberen
Halbebene gehen in analytische Funktionen der Kreisscheibe über. Daher gilt
H_ℝ[f](ξ(θ)) = H_T[g](θ) + C mit der periodischen Hilbert-Transformation H_T
(Multiplikator −i·sign(k), FFT) und g(θ) = f(ξ(θ)). Die Konstante C folgt aus
H_ℝ[f](±∞) = 0: C = −H_T[g](π). Ohne diese Konstante ist z. B. H[x/(1+x²)]
falsch (Test 2 in Schritt 0).

## 6. Abweichung: Fernfeld-Darstellung
Das Fernfeld eines Profils ist Ω ~ sign(ξ)|ξ|^(−1/c), in θ also |π−θ|^(1/c).
Für nicht ganzzahliges s = 1/c ist das bei θ = π nicht glatt; die FFT-Hilbert-
Transformation konvergiert dann nur wie N^(−s). Für c = 3 (s = 1/3) wäre die
Konstante C bei N = 2048 um etwa 8 % falsch — das würde die Kurve Loss(c) für
große c systematisch verfälschen. Deshalb ist |cos(θ/2)|^s mit lernbarem s
nicht als Input-Feature, sondern als exakter Fernfeldterm eingebaut:
  Ω(θ) = A·Im E_s(θ) + sin(θ/2)·cos(θ/2)^(s+1)·N(cos θ, …, cos 4θ),
  E_s = (L/(L−iξ))^s = cos(θ/2)^s·e^(isθ/2)   (analytisch in der oberen Halbebene)
  ⇒ H[Im E_s] = −Re E_s exakt, U_s = −A·L·cos(θ/2)^(s−1)·sin((1−s)θ/2)/(1−s).
N ist das MLP (4×64, tanh) auf cos(kθ), k=1..4 (gerade), die Vorfaktoren machen
Ω exakt ungerade. Nur der Restterm läuft durch die FFT; seine Singularität ist
|π−θ|^(s+1), Fehler ~N^(−1−s). s wird gelernt und mit 1/c verglichen
(Konsistenzcheck, SUMMARY). A, s lernbar; Initialisierung für alle c gleich
aufgebaut (s0 = 1/c, Ω'(0) ≈ −1), keine Kenntnis von c_ref.

## 7. Weitere numerische Details
- ξΩ' = sin θ·Ω_θ; Ω' = m·Ω_θ mit m = 2cos²(θ/2)/L; Ω_θ per Autograd.
- U: geschlossene Form (singulärer Teil) + Trapezregel in θ (Restteil), exakt ungerade.
- dR/dξ = m·(R_{j+1} − R_{j−1})/(2Δθ) (zentrale Differenzen; vermeidet
  spektrale Ableitung einer nicht glatten Funktion).
- Kollokation: alle θ-Gitterpunkte außer θ = ±π (ξ = ∞).
- **Abweichung Loss:** Loss = mean(R²)/mean(Ω²) + 0.1·mean((dR/dξ)²)/mean(Ω²)
  + (Ω'(0)+1)². Grund (im Smoke-Test beobachtet): Mit absolutem mean(R²) hat
  jedes Nicht-Lösungs-Profil eine entartete Richtung — ein schmales Profil der
  Breite w mit Ω'(0) = −1 hat Amplitude ~w, R ~ w auf einem Anteil ~w der
  Punkte, also mean(R²) ~ w³ → 0 für jedes c. Der Loss fiel monoton zu kleinem
  c, lernbares c lief an die Klemme 0.05. Die relative Form ist skaleninvariant
  in diesem Sinn. Das absolute mean(R²) wird zusätzlich geloggt (loss_res_abs).
  "Kleines Residuum" bezieht sich auf das relative mean(R²)/mean(Ω²).
- Zeitintegration: rfft, H = −i·sign(k), u = −|k|⁻¹ω̂ (u_x = Hω), RK4 mit
  dt = 0.02/max|ω|, Filter exp(−36(k/k_max)^36), Stopp bei max|ω| = 1e4 oder
  wenn |ω̂| oberhalb k = N/3 relativ 1e-8 übersteigt. Neben N = 4096 (Vorgabe)
  läuft N = 65536 (a = 0) bzw. N = 16384 (a > 0), weil N = 4096 nur bis
  max|ω| ≈ 40 auflöst; c_ref stammt aus dem feinen Lauf (Fallback N = 4096).
  Für a > 0 zusätzlich Advektions-CFL dt ≤ 1/(a·max|u|·N/2): Die Vorgabe
  dt = 0.02/max|ω| allein war in Lauf v1 instabil (Explosion nach 3–6 Schritten). Fit-Fenster: max|ω| ≥ max|ω|_end/20.
  T aus 1/max|ω| linear in t; c aus log(Kernbreite) vs. log(T−t), Kernbreite =
  Abstand des Maximums von |ω| zu x0. Kontrolle c_alt aus ω_x(x0) ~ τ^(−1−c).
- Profilvergleich: τ·ω(x0 + λξ) mit λ = −1/(τ·ω_x(x0)) (dieselbe Normierung
  Ω'(0) = −1), relL2 auf |ξ| ≤ 10, Schwelle 0.05 (eigene Wahl).
- Hardware: Kaggle 2×T4 (Nutzerwunsch). Es wird eine GPU genutzt; zu Beginn
  misst das Skript Adam-Schritte auf CPU und GPU in FP64 und nimmt das
  schnellere Gerät (T4 hat schwache FP64-Leistung).
"""


def write_notes(out, s0):
    L = [NOTES_TEXT, "## Automatische Prüfausgaben (Schritt 0)", ""]
    for k, v in s0.get("sympy", {}).items():
        L.append(f"- sympy {k}: {v}")
    for k, v in s0.get("pipeline", {}).items():
        L.append(f"- pipeline {k}: {v}")
    num = s0.get("numeric", {})
    for name, e in num.get("tests", []):
        L.append(f"- Hilbert {name}: {e:.3e}")
    for s, x0, e in num.get("sing", []):
        L.append(f"- H[Im E_s] Quadraturtest s={s}, x={x0}: {e:.2e}")
    with open(os.path.join(out, "NOTES.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
