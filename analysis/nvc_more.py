"""FINAL_PLAN P15: improving the claims that hold. Modes: "claim2" | "claim5" | "claim3 COHORT K N" | "merge"."""
import json
import os
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "merge"
ARGV = sys.argv[2:]
sys.argv = [sys.argv[0], "merge"]
HERE = Path(__file__).parent
SEED = 1337
FAMILY = 15
ALPHA = 0.05 / FAMILY
NBOOT, NFLIP, NSURR = 5000, 10000, int(os.environ.get("NSURR", "20"))
SMOKE = int(os.environ.get("SMOKE", "0"))
COHORTS = ("hbci", "mental_workload", "finemi", "mendeley")
R = HERE / "results"

import numpy as np
from scipy.stats import rankdata, spearmanr, t as t_dist


def bootstrap_mean(d, rng):
    i = rng.integers(0, len(d), (NBOOT, len(d)))
    m = d[i].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi), float(min(1.0, 2 * min(np.mean(m <= 0), np.mean(m >= 0)) + 1 / NBOOT))


def signflip_p(d, rng):
    s = rng.choice([-1.0, 1.0], (NFLIP, len(d)))
    return float((np.sum(np.abs((s * d).mean(1)) >= abs(d.mean())) + 1) / (NFLIP + 1))


if MODE == "claim2":
    src_d = (HERE / "external_claim2.py").read_text(encoding="utf-8")
    src_d = src_d[:src_d.index("D = {k: load(k) for k in CACHES}")]
    E2 = {"__name__": "nvc_e2", "__file__": str(HERE / "external_claim2.py")}
    exec(src_d, E2)
    from sklearn.linear_model import RidgeCV

    zs_person, demean, ALPHAS, trial_corr = E2["zs_person"], E2["demean"], E2["ALPHAS"], E2["trial_corr"]
    D = {k: E2["load"](k) for k in ("hbci", "mental_workload")}
    pairs = json.loads((R / "nvc_held_overlap.json").read_text())["candidates"]
    PARTNER = {"mental_workload": {int(m): int(h) for h, m, _ in pairs}, "hbci": {int(h): int(m) for h, m, _ in pairs}}
    saved = json.loads((R / "external_claim2_results.json").read_text())
    res = {}
    for src, dst in (("hbci", "mental_workload"), ("mental_workload", "hbci")):
        a, b = D[src], D[dst]
        W = E2["interpolation_matrix"](b["ch"], a["ch"])
        assert W is not None
        Xa = zs_person(a["F"], a["sid"])
        Ya = demean(a["Y"], a["sid"].astype(np.int64) * 1000 + a["tid"])
        Fb = np.einsum("ij,njt->nit", W, b["F"])
        Xb = zs_person(Fb, b["sid"])
        gb = b["sid"].astype(np.int64) * 1000 + b["tid"]
        Yb = demean(b["Y"], gb)
        fit = lambda X, Y: RidgeCV(alphas=ALPHAS).fit(X, Y)
        st = E2["subject_level_test"](demean(fit(Xa, Ya).predict(Xb), gb), Yb, b["sid"])
        key = f"{src}_to_{dst}"
        reproduced, stored = float(st["r_subject"]), float(saved[key]["r_subject"])
        assert abs(reproduced - stored) < 1e-6, (key, reproduced, stored)
        people = np.unique(b["sid"])[:SMOKE or None]
        rows = []
        for p in people:
            te, tr = b["sid"] == p, b["sid"] != p
            keep = a["sid"] != PARTNER[dst].get(int(p), -1)
            score = lambda m: float(trial_corr(demean(m.predict(Xb[te]), gb[te]), Yb[te]).mean())
            rows.append([score(fit(Xa[keep], Ya[keep])), score(fit(Xb[tr], Yb[tr])), score(fit(np.concatenate([Xa[keep], Xb[tr]]), np.concatenate([Ya[keep], Yb[tr]])))])
            print(key, int(p), np.round(rows[-1], 3), flush=True)
        S = np.array(rows)
        rng = np.random.default_rng(SEED)
        out = {"target": dst, "n": len(S), "reproduced_p32": reproduced, "saved_p32": stored, "a_only": float(S[:, 0].mean()), "b_only": float(S[:, 1].mean()), "pool": float(S[:, 2].mean()), "per_person": S.tolist()}
        for name, d in (("pool_minus_a", S[:, 2] - S[:, 0]), ("pool_minus_b", S[:, 2] - S[:, 1])):
            lo, hi, _ = bootstrap_mean(d, rng)
            out[name] = {"mean": float(d.mean()), "ci": [lo, hi], "p": signflip_p(d, rng), "pass": bool(lo > 0), "positive": int(np.sum(d > 0))}
        out["improved"] = bool(out["pool_minus_a"]["pass"] and out["pool_minus_b"]["pass"])
        res[key] = out
        print({k: v for k, v in out.items() if k != "per_person"}, flush=True)
    (HERE / "nvc_more_claim2.json").write_text(json.dumps(res, indent=1))
    print("DONE")
    sys.exit(0)

if MODE == "claim5":
    C5 = json.loads((R / "nvc_held_claim5.json").read_text())
    Z = HERE.parent / "neurovascnet_merge_v1_cache" / "p14_windows"
    people = {c: {int(k): v for k, v in C5[c]["raw"]["people"].items()} for c in COHORTS}
    xy = {c: (np.array([v["cal_score"] for v in people[c].values()]), np.array([v["test"] for v in people[c].values()])) for c in COHORTS}
    transport = {}
    for held in COHORTS:
        train = [c for c in COHORTS[:3] if c != held]
        X, Y = np.concatenate([xy[c][0] for c in train]), np.concatenate([xy[c][1] for c in train])
        b, a = np.polyfit(X, Y, 1)
        x, y = xy[held]
        err_m, err_c = np.abs(y - (a + b * x)), np.abs(y - Y.mean())
        half = np.percentile(np.abs(Y - (a + b * X)), 80)
        lo, hi, p = bootstrap_mean(err_c - err_m, np.random.default_rng(SEED))
        cov = float(np.mean(err_m <= half))
        transport[held] = {"n": len(x), "train": train, "slope": float(b), "intercept": float(a), "mae_mapping": float(err_m.mean()), "mae_constant": float(err_c.mean()), "gain": float((err_c - err_m).mean()), "ci": [lo, hi], "p": p, "pass": bool(lo > 0), "half_width_80": float(half), "coverage_80": cov, "coverage_in_range": bool(0.65 <= cov <= 0.95)}
        print(held, transport[held], flush=True)
    partial = {}
    for c in COHORTS[:3]:
        z = np.load(Z / f"{c}.npz")
        Yall, sid, rec, psd, psid, prun = z["Y"].astype(np.float64), z["sid"], z["rec"], z["psd"], z["psd_sid"], z["psd_run"]
        rng = np.random.default_rng(SEED)
        cov = {}
        for s in np.unique(sid):
            m = np.flatnonzero(sid == s)
            runs = np.unique(rec[m])
            if len(runs) < 3:
                continue
            cal = list(rng.choice(runs, 2, replace=False))
            yc = Yall[m[np.isin(rec[m], cal)]]
            cov[int(s)] = [float(np.log((yc - yc.mean(0)).std())), float(psd[(psid == s) & np.isin(prun, cal)].mean())]
        ids = list(people[c])
        assert set(ids) == set(cov), c
        x, y = np.array([people[c][i]["cal_score"] for i in ids]), np.array([people[c][i]["test"] for i in ids])
        C = np.array([cov[i] for i in ids])

        def prho(idx):
            G = np.column_stack([np.ones(len(idx)), rankdata(C[idx, 0]), rankdata(C[idx, 1])])
            rx, ry = rankdata(x[idx]), rankdata(y[idx])
            ex, ey = rx - G @ np.linalg.lstsq(G, rx, rcond=None)[0], ry - G @ np.linalg.lstsq(G, ry, rcond=None)[0]
            return float(ex @ ey / (np.linalg.norm(ex) * np.linalg.norm(ey) + 1e-12))

        n = len(ids)
        rho = prho(np.arange(n))
        tv = rho * np.sqrt((n - 4) / (1 - rho ** 2))
        rng2 = np.random.default_rng(SEED)
        ci = [float(v) for v in np.nanpercentile([prho(i) for i in rng2.integers(0, n, (NBOOT, n))], [2.5, 97.5])]
        partial[c] = {"n": n, "partial_rho": rho, "ci": ci, "p": float(2 * t_dist.sf(abs(tv), n - 4)), "pass": bool(rho > 0.2 and ci[0] > 0), "plain_rho": float(spearmanr(x, y)[0]), "cov_vs_test": [float(spearmanr(C[:, k], y)[0]) for k in (0, 1)], "cov_vs_score": [float(spearmanr(C[:, k], x)[0]) for k in (0, 1)]}
        print(c, partial[c], flush=True)
    (HERE / "nvc_more_claim5.json").write_text(json.dumps({"transport": transport, "partial": partial}, indent=1))
    print("DONE")
    sys.exit(0)


def load_bp():
    """nvc_bandpower helpers, with the generators patched to return HbO and HbR stacked along channels."""
    t = (HERE / "nvc_bandpower.py").read_text(encoding="utf-8")
    t = t[:t.index('if MODE in ("shard", "all"):')]
    t = t.replace('ns = {"__name__": "nvc"', 'for a_, b_ in PATCH:\n    assert a_ in src, a_\n    src = src.replace(a_, b_)\nns = {"__name__": "nvc"', 1)
    a = 'hbo = hb.get_data(picks=[i for i, c in enumerate(hb.ch_names) if c.endswith("hbo")])'
    assert a in t
    t = t.replace(a, 'hbo = np.concatenate([hb.get_data(picks=[i for i, c in enumerate(hb.ch_names) if c.endswith(k_)]) for k_ in ("hbo", "hbr")], 0)')
    PATCH = [("hbo, _ = beer_lambert(lo, hi)", "hbo = np.concatenate([np.asarray(q_, float) for q_ in beer_lambert(lo, hi)], 0)"),
             ("hbo = np.asarray(cn.oxy.x, float).T", "hbo = np.concatenate([np.asarray(cn.oxy.x, float).T, np.asarray(cn.deoxy.x, float).T], 0)")]
    g = {"__name__": "nvc_m15", "__file__": str(HERE / "nvc_bandpower.py"), "PATCH": PATCH}
    exec(t, g)
    return g


if MODE == "claim3":
    COHORT, K, N = ARGV[0], int(ARGV[1]), int(ARGV[2])
    BP = load_bp()
    LAGS, curve, MINSHIFT = BP["LAGS"], BP["curve"], BP["MINSHIFT"]
    if COHORT == "mendeley":
        sys.argv = [sys.argv[0], "none"]
        s_ = (HERE / "nvc_mendeley.py").read_text(encoding="utf-8")
        s_ = s_[:s_.index('if MODE == "coupling":')]
        M12 = {"__name__": "nvc_m15b", "__file__": str(HERE / "nvc_mendeley.py")}
        exec(s_, M12)

        def gen():
            for sid in M12["INCL"]:
                x, names, hbo, hbr, n_fs, q = M12["load_person"](M12["ALL_S"][sid])
                prs = (BP["run_pair_bp"](e, 200.0, np.concatenate([h, r], 0), n_fs, 0.0) for e, h, r in M12["segments"](x, hbo, hbr, n_fs, q["offset"]))
                yield sid, [p for p in prs if p is not None]
    else:
        gen = BP["GENS"][COHORT]
    out = HERE / f"nvc_more_hbr_{COHORT}_{K}_of_{N}{'_smoke' if SMOKE else ''}.json"
    done = json.loads(out.read_text()) if out.exists() else {}
    ci = COHORTS.index(COHORT)
    for pi, (pid, pairs) in enumerate(gen()):
        if SMOKE and pi >= SMOKE:
            break
        if pi % N != K or pid in done or len(pairs) < 2:
            continue
        nc = pairs[0][1].shape[1] // 2
        O = [(E, H[:, :nc]) for E, H in pairs]
        H_ = [(E, H[:, nc:]) for E, H in pairs]
        cO, cR = curve(O, "local"), curve(H_, "local")
        rng = np.random.default_rng([SEED, ci, pi])
        null = [curve([(E, np.roll(H, rng.integers(MINSHIFT, max(MINSHIFT + 1, len(H) - MINSHIFT)), axis=0)) for E, H in H_], "local").round(5).tolist() for _ in range(NSURR)]
        done[pid] = {"curve_hbo": cO.round(5).tolist(), "curve_hbr": cR.round(5).tolist(), "null_hbr": null, "n_runs": len(pairs)}
        print(pid, "peaks", int(LAGS[np.argmax(cO)]), int(LAGS[np.argmax(cR)]), round(float(cR.max()), 3), flush=True)
        out.write_text(json.dumps(done))
    print("DONE")
    sys.exit(0)

# merge
LAGS = np.arange(-10, 21)
WINDOW = (LAGS >= 0) & (LAGS <= 15)


def smooth(c):
    return np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3, mode="same"), -1, c)


def peak(c):
    return float(LAGS[WINDOW][np.argmax(smooth(c)[..., WINDOW], axis=-1)]) if c.ndim == 1 else LAGS[WINDOW][np.argmax(smooth(c)[..., WINDOW], axis=-1)].astype(float)


C2 = json.loads((HERE / "nvc_more_claim2.json").read_text())
C5 = json.loads((HERE / "nvc_more_claim5.json").read_text())
saved_bp = json.loads((R / "nvc_bandpower_results.json").read_text())
saved_men = {}
for f in sorted(HERE.glob("nvc_mendeley_shard_*_of_*.json")):
    saved_men.update(json.loads(f.read_text()))
hbr, delays = {}, {}
for c in COHORTS:
    people = {}
    for f in sorted(HERE.glob(f"nvc_more_hbr_{c}_*_of_*.json")):
        if "smoke" not in f.name:
            people.update(json.loads(f.read_text()))
    ref = saved_men if c == "mendeley" else saved_bp[c]["people"]
    diff = max(float(np.max(np.abs(np.array(v["curve_hbo"]) - np.array(ref[p]["curve_local"])))) for p, v in people.items())
    A, B = np.array([v["curve_hbo"] for v in people.values()]), np.array([v["curve_hbr"] for v in people.values()])
    NU = np.array([v["null_hbr"] for v in people.values()]).mean(0)
    gate = bool(smooth(B.mean(0))[WINDOW].max() > smooth(NU)[:, WINDOW].max())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(A), (NBOOT, len(A)))
    d = peak(B[idx].mean(1)) - peak(A[idx].mean(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)) + 1 / NBOOT))
    obs = peak(B.mean(0)) - peak(A.mean(0))
    hbr[c] = {"n": len(A), "hbo_peak_s": peak(A.mean(0)), "hbr_peak_s": peak(B.mean(0)), "difference_s": obs, "ci": [float(lo), float(hi)], "p": p, "gate": gate, "hbo_reproduction_max_abs_diff": diff, "pass": bool(lo > 0 and gate), "peak_r_hbo": float(smooth(A.mean(0))[WINDOW].max()), "peak_r_hbr": float(smooth(B.mean(0))[WINDOW].max())}
    dl = np.array([v["delay_s"] for v in ref.values()], float)
    dr = np.random.default_rng(SEED)
    delays[c] = {"n": len(dl), "median_s": float(np.median(dl)), "ci": [float(v) for v in np.percentile(np.median(dl[dr.integers(0, len(dl), (NBOOT, len(dl)))], axis=1), [2.5, 97.5])]}
rows = []
for k, v in C2.items():
    rows += [[f"P15.1 target {v['target']}, pool - {w}", v[f"pool_minus_{w[0]}"]["mean"], v[f"pool_minus_{w[0]}"]["p"], v[f"pool_minus_{w[0]}"]["pass"]] for w in ("a only", "b only")]
rows += [[f"P15.2 HbR - HbO peak, {c}", hbr[c]["difference_s"], hbr[c]["p"], hbr[c]["pass"]] for c in COHORTS]
rows += [[f"P15.3 mapping gain, {c}", C5["transport"][c]["gain"], C5["transport"][c]["p"], C5["transport"][c]["pass"]] for c in COHORTS]
rows += [[f"P15.4 partial rho, {c}", C5["partial"][c]["partial_rho"], C5["partial"][c]["p"], C5["partial"][c]["pass"]] for c in COHORTS[:3]]
assert len(rows) == FAMILY
res = {"alpha": ALPHA, "tests": [{"test": a, "value": v, "p": p, "pass": bool(k), "survives_alpha": bool(k and p < ALPHA)} for a, v, p, k in rows], "claim2": C2, "hbr": hbr, "hbr_signature_supported": bool(hbr["hbci"]["pass"] and any(hbr[c]["pass"] for c in COHORTS[1:])), "claim5": C5, "delays_descriptive": delays}
(HERE / "nvc_more_results.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: res[k] for k in ("alpha", "tests", "hbr", "hbr_signature_supported", "delays_descriptive")}, indent=1, default=float))
print("DONE")
