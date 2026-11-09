"""FINAL_PLAN P14: stress tests of the claims that hold. Modes, in order: "build" | "overlap" | "claim2" | "claim5" | "claim3" | "merge"."""
import json
import os
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "merge"
sys.argv = [sys.argv[0], "merge"]
HERE = Path(__file__).parent
src = (HERE / "nvc_method_check.py").read_text(encoding="utf-8")
src = src[:src.index('if ARGS != ["merge"]:')]
M13 = {"__name__": "nvc_m14", "__file__": str(HERE / "nvc_method_check.py")}
exec(src, M13)
src_d = (HERE / "external_claim2.py").read_text(encoding="utf-8")
src_d = src_d[:src_d.index("D = {k: load(k) for k in CACHES}")]
E2 = {"__name__": "nvc_e2", "__file__": str(HERE / "external_claim2.py")}
exec(src_d, E2)

import numpy as np
import scipy.signal as sps
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV

BP, LC, NS2 = M13["BP"], M13["LC"], M13["NS2"]
windows, GENS, CH, SEED = M13["windows"], M13["GENS"], NS2["CH"], BP["SEED"]
subject_level_test, verdict_text, interpolation_matrix = E2["subject_level_test"], E2["verdict_text"], E2["interpolation_matrix"]
zs_person, demean, ALPHAS = E2["zs_person"], E2["demean"], E2["ALPHAS"]
CACHE = HERE.parent / "neurovascnet_merge_v1_cache" / "p14_windows"
SMOKE = int(os.environ.get("SMOKE", "0"))
FAMILY = 12
ALPHA = 0.05 / FAMILY
COHORTS = ("hbci", "mental_workload", "finemi", "mendeley")


def hook2(eeg, e_fs, hbo, n_fs, off):
    eeg, hbo = np.asarray(eeg, float), np.asarray(hbo, float)
    w = windows(eeg, e_fs, hbo, n_fs, off)
    if w is None:
        return None
    f, p = sps.welch(eeg, fs=e_fs, nperseg=int(2 * e_fs), noverlap=int(e_fs), axis=-1)
    lp = np.log(p[:, (f >= 1) & (f < 45)] + 1e-30)
    return w, lp.reshape(lp.shape[0], -1, 2).mean(-1).astype(np.float32)


LC["run_pair"] = hook2
BP["run_pair_bp"] = hook2


def npz(name):
    z = np.load(CACHE / f"{name}.npz")
    return {k: z[k] for k in z.files}


def corr_matrix(A, B):
    A, B = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    A, B = A / np.linalg.norm(A, axis=1, keepdims=True), B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def fingerprints(d, idx, part=None):
    out = []
    for s in np.unique(d["psd_sid"]):
        k = np.flatnonzero(d["psd_sid"] == s)
        out.append(d["psd"][k if part is None else k[part::2]][:, idx].mean(0).ravel())
    return np.array(out)


def transfer(D, src_, dst, drop):
    a, b = D[src_], D[dst]
    ma, mb = ~np.isin(a["sid"], list(drop[src_])), ~np.isin(b["sid"], list(drop[dst]))
    W = interpolation_matrix(b["ch"], a["ch"])
    assert W is not None
    Fb = np.einsum("ij,njt->nit", W, b["F"][mb])
    sb, tb = b["sid"][mb], b["tid"][mb]
    mdl = RidgeCV(alphas=ALPHAS).fit(zs_person(a["F"][ma], a["sid"][ma]), demean(a["Y"][ma], a["sid"][ma].astype(np.int64) * 1000 + a["tid"][ma]))
    g = sb.astype(np.int64) * 1000 + tb
    st = subject_level_test(demean(mdl.predict(zs_person(Fb, sb)), g), demean(b["Y"][mb], g), sb)
    st["verdict"] = verdict_text(st)
    st["pass"] = bool(st["verdict"] == "detectable")
    return st


def trial_corr(P, T):
    P, T = P - P.mean(1, keepdims=True), T - T.mean(1, keepdims=True)
    return (P * T).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-12)


def zs(F, ref):
    return (F - ref.mean(0)) / (ref.std(0) + 1e-8)


def hybrid(F, Y, ca, te, pop, gain):
    base = Y[ca].mean(0)
    Fca, Fte = zs(F[ca], F[ca]), zs(F[te], F[ca])
    pers = RidgeCV(alphas=ALPHAS).fit(Fca, Y[ca] - base).predict(Fte)
    h = trial_corr(base + 0.5 * (pers + pop.predict(Fte)), Y[te]).mean()
    return h - trial_corr(np.broadcast_to(base, Y[te].shape), Y[te]).mean() if gain else h


def reliability(d, gain):
    F, Y, sid, rec = d["F"].reshape(len(d["F"]), -1).astype(np.float64), d["Y"].astype(np.float64), d["sid"], d["rec"]
    Fz, R = np.zeros_like(F), np.zeros_like(Y)
    for s in np.unique(sid):
        m = sid == s
        Fz[m], R[m] = zs(F[m], F[m]), Y[m] - Y[m].mean(0)
    rng = np.random.default_rng(SEED)
    per = {}
    for s in np.unique(sid):
        m = np.flatnonzero(sid == s)
        runs = np.unique(rec[m])
        if len(runs) < 3:
            continue
        pop = RidgeCV(alphas=ALPHAS).fit(Fz[sid != s], R[sid != s])
        cal = list(rng.choice(runs, 2, replace=False))
        c0, c1 = (m[rec[m] == r] for r in cal)
        score = 0.5 * (hybrid(F, Y, c0, c1, pop, gain) + hybrid(F, Y, c1, c0, pop, gain))
        ca = np.concatenate([c0, c1])
        test = float(np.mean([hybrid(F, Y, ca, m[rec[m] == r], pop, gain) for r in runs if r not in cal]))
        per[int(s)] = {"cal_score": float(score), "test": test}
    sc, te = np.array([v["cal_score"] for v in per.values()]), np.array([v["test"] for v in per.values()])
    rho, p = spearmanr(sc, te)
    boot = [spearmanr(sc[i], te[i])[0] for i in (rng.integers(0, len(sc), len(sc)) for _ in range(5000))]
    ci = [float(x) for x in np.nanpercentile(boot, [2.5, 97.5])]
    return {"n": len(per), "rho": float(rho), "p": float(p), "ci": ci, "pass": bool(rho > 0.2 and ci[0] > 0), "mean_test": float(te.mean()), "people": per}


if MODE == "build":
    CACHE.mkdir(parents=True, exist_ok=True)
    for cohort in os.environ.get("P14_COHORTS", ",".join(COHORTS)).split(","):
        X, Y, S, R, P, PS, PR = [], [], [], [], [], [], []
        for i, (pid, rows) in enumerate(GENS[cohort]()):
            if SMOKE and i >= SMOKE:
                break
            rows = [q for q in rows if q is not None]
            print(cohort, pid, len(rows), flush=True)
            if len(rows) < 2:
                continue
            for r, (w, psd) in enumerate(rows):
                X.append(w[0]), Y.append(w[2]), S.append(np.full(len(w[2]), i + 1)), R.append(np.full(len(w[2]), r)), P.append(psd), PS.append(i + 1), PR.append(r)
        ch = list(CH[cohort]) if cohort in CH else []
        nch = X[0].shape[1] // 30
        assert not ch or nch == len(ch), (cohort, nch, len(ch))
        F = np.concatenate(X)
        np.savez(CACHE / f"{cohort}{'_smoke' if SMOKE else ''}.npz", F=F.reshape(len(F), nch, 30), Y=np.concatenate(Y), sid=np.concatenate(S), rec=np.concatenate(R), ch=np.array(ch, dtype=str), psd=np.stack(P), psd_sid=np.array(PS), psd_run=np.array(PR))
        print(cohort, "windows", len(F), flush=True)
    print("DONE")
    sys.exit(0)

if MODE == "overlap":
    h, m = npz("hbci"), npz("mental_workload")
    hc, mc = [str(c) for c in h["ch"]], [str(c) for c in m["ch"]]
    common = [c for c in hc if c in mc]
    ih, im = [hc.index(c) for c in common], [mc.index(c) for c in common]
    control, within = {}, []
    for name, d, idx in (("hbci", h, ih), ("mental_workload", m, im)):
        A, B = fingerprints(d, idx, 0), fingerprints(d, idx, 1)
        S = corr_matrix(A - A.mean(0), B - B.mean(0))
        control[name] = {"identification": float(np.mean(S.argmax(1) == np.arange(len(S)))), "median_same_person": float(np.median(np.diag(S))), "median_other": float(np.median(S[~np.eye(len(S), dtype=bool)]))}
        Fp = fingerprints(d, idx)
        Fp = Fp - Fp.mean(0)
        C_ = corr_matrix(Fp, Fp)
        within.append(C_[np.triu_indices(len(C_), 1)])
    T = float(np.percentile(np.concatenate(within), 99))
    Fh, Fm = fingerprints(h, ih), fingerprints(m, im)
    C = corr_matrix(Fh - Fh.mean(0), Fm - Fm.mean(0))
    sh, sm = np.unique(h["psd_sid"]), np.unique(m["psd_sid"])
    cand = [(int(sh[i]), int(sm[j]), float(C[i, j])) for i in range(len(sh)) for j in range(len(sm)) if C[i, j] > T and C[i].argmax() == j and C[:, j].argmax() == i]
    Q, top = C.copy(), []
    for _ in range(5):
        i, j = np.unravel_index(np.argmax(Q), Q.shape)
        top.append((int(sh[i]), int(sm[j]), float(C[i, j])))
        Q[i, :], Q[:, j] = -np.inf, -np.inf
    drop = {"hbci": {a for a, _, _ in cand + top}, "mental_workload": {b for _, b, _ in cand + top}}
    D = {k: E2["load"](k) for k in ("hbci", "mental_workload")}
    none = {"hbci": set(), "mental_workload": set()}
    saved = json.loads((HERE / "results" / "external_claim2_results.json").read_text())
    res = {"common_channels": common, "control": control, "control_ok": bool(min(v["identification"] for v in control.values()) >= 0.8), "threshold_T": T, "max_cross_similarity": float(C.max()),
           "candidates": cand, "top5": top, "excluded": {k: sorted(v) for k, v in drop.items()}, "reproduce_hbci_to_mw": transfer(D, "hbci", "mental_workload", none)["r_subject"], "saved_hbci_to_mw": saved["hbci_to_mental_workload"]["r_subject"],
           "hbci_to_mw_excluded": transfer(D, "hbci", "mental_workload", drop), "mw_to_hbci_excluded": transfer(D, "mental_workload", "hbci", drop)}
    (HERE / "nvc_held_overlap.json").write_text(json.dumps(res, indent=1, default=float))
    print({k: res[k] for k in ("control", "control_ok", "threshold_T", "max_cross_similarity", "candidates", "top5", "reproduce_hbci_to_mw", "saved_hbci_to_mw")})
    print(res["hbci_to_mw_excluded"])
    print("DONE")
    sys.exit(0)

if MODE == "claim2":
    D = {}
    for k in ("hbci", "mental_workload"):
        d = npz(k)
        n = len(d["sid"])
        D[k] = {"F": d["F"].astype(np.float64), "Y": d["Y"].astype(np.float64).reshape(n, -1, 30).mean(1), "sid": d["sid"], "tid": np.zeros(n, int), "ch": [str(c) for c in d["ch"]]}
    none = {k: set() for k in D}
    res = {"hbci_to_mental_workload": transfer(D, "hbci", "mental_workload", none), "mental_workload_to_hbci": transfer(D, "mental_workload", "hbci", none)}
    (HERE / "nvc_held_claim2.json").write_text(json.dumps(res, indent=1, default=float))
    print(res)
    print("DONE")
    sys.exit(0)

if MODE == "claim5":
    res = {}
    for c in COHORTS:
        d = npz(c)
        res[c] = {f: reliability(d, f == "gain") for f in ("raw", "gain")}
        print(c, {f: (round(v["rho"], 3), [round(x, 3) for x in v["ci"]], round(v["p"], 4), v["pass"], v["n"]) for f, v in res[c].items()}, flush=True)
    (HERE / "nvc_held_claim5.json").write_text(json.dumps(res, indent=1))
    print("DONE")
    sys.exit(0)

if MODE == "claim3":
    allp = {}
    for f in sorted(HERE.glob("nvc_mendeley_shard_*_of_*.json")):
        allp.update(json.loads(f.read_text()))
    d = np.array([v["delay_s"] for v in allp.values()])
    w = float(wilcoxon(d).pvalue)
    res = {"n": len(d), "median_delay_s": float(np.median(d)), "iqr": [float(x) for x in np.percentile(d, [25, 75])], "wilcoxon_p": w, "physiological": bool(2 <= np.median(d) <= 10 and w < 0.05), "note": "post hoc, not counted"}
    (HERE / "nvc_held_claim3.json").write_text(json.dumps(res, indent=1))
    print(res)
    print("DONE")
    sys.exit(0)

O, C2, C5, C3 = (json.loads((HERE / f"nvc_held_{k}.json").read_text()) for k in ("overlap", "claim2", "claim5", "claim3"))
C1 = {c: json.loads((HERE / f"claim1_fresh_{c}_uncued_results.json").read_text()) for c in ("mental_workload", "finemi")}
rows = [["P14.1 Claim 2, shared-people check", "hBCI -> MW", O["hbci_to_mw_excluded"]["r_subject"], O["hbci_to_mw_excluded"]["p_signflip"], O["hbci_to_mw_excluded"]["pass"]],
        ["P14.2 Claim 2, no cues", "hBCI -> MW", C2["hbci_to_mental_workload"]["r_subject"], C2["hbci_to_mental_workload"]["p_signflip"], C2["hbci_to_mental_workload"]["pass"]]]
rows += [["P14.3 Claim 5, no cues" if c != "mendeley" else "P14.4 Claim 5, third lab", f"{c} {f}", C5[c][f]["rho"], C5[c][f]["p"], C5[c][f]["pass"]] for c in COHORTS for f in ("raw", "gain")]
rows += [["P14.5 Claim 1, no cues", c, C1[c]["fnirs"]["joint"]["r_subject"], max(C1[c]["fnirs"]["joint"]["p_signflip"], C1[c]["fmri"]["joint"]["p_signflip"]), C1[c]["claim1_shared_space_holds"]] for c in C1]
assert len(rows) == FAMILY
res = {"alpha": ALPHA, "tests": [{"test": a, "what": b, "value": v, "p": p, "pass": bool(k), "survives_alpha": bool(k and p < ALPHA)} for a, b, v, p, k in rows],
       "control_ok": O["control_ok"], "candidates": O["candidates"], "excluded": O["excluded"], "reproduction": [O["reproduce_hbci_to_mw"], O["saved_hbci_to_mw"]], "control": O["control"], "threshold_T": O["threshold_T"], "max_cross_similarity": O["max_cross_similarity"],
       "claim5_no_cues_holds": bool(C5["hbci"]["raw"]["pass"] and (C5["mental_workload"]["raw"]["pass"] or C5["finemi"]["raw"]["pass"])), "claim3_post_hoc": C3}
(HERE / "nvc_held_results.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
print("DONE")
