"""Generate a synthetic maintenance log (CMMS work orders, operator rounds, incident reports).

Everything is invented: assets, tags, dates and text. Each asset follows a hidden degradation
process, so the true answer ("did this asset break down within 30 days?") is known exactly.

Hidden process per asset:
  * Breakdowns arrive at random (mean ~2 years apart).
  * 80% of breakdowns are preceded by a degradation period of 2-11 weeks in which symptoms get
    written into the logs more and more often. 20% are sudden, with no warning in the text.
  * Some degradation periods are caught: a corrective work order fixes the problem and no
    breakdown follows. In a plant that is a success, and it is what a "false alarm" often is.
  * Healthy assets still get the odd symptom-like remark ("vib slightly high, rechecked normal").

Usage:  python -m src.data.generate_synthetic [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "raw"
START, YEARS = date(2020, 1, 1), 6
DAYS = 365 * YEARS + 1

ASSET_TYPES = {
    "pump": dict(n=40, prefix="P", modes=["bearing", "seal", "cavitation"]),
    "compressor": dict(n=12, prefix="K", modes=["bearing", "lube", "surge"]),
    "motor": dict(n=24, prefix="M", modes=["winding", "bearing"]),
    "control_valve": dict(n=28, prefix="FV", modes=["sticking", "packing"]),
    "transmitter": dict(n=16, prefix="PT", modes=["drift", "impulse"]),
}

SYMPTOMS = {  # failure mode -> early-warning remarks, mild to severe
    "bearing": ["bearing temperature slightly up at {t} C", "vibration increasing on DE bearing",
                "noise from bearing housing", "bearing temperature rising, now {t} C",
                "lube oil looks discoloured", "high vibration on NDE bearing, trending up"],
    "seal": ["mechanical seal weeping", "small drips from mech seal", "seal pot level dropping",
             "seal flush line running hot", "seal leak increasing, drip tray filling"],
    "cavitation": ["crackling noise at suction", "discharge pressure fluctuating",
                   "flow lower than normal at same opening", "pump running rough, suction strainer dp high"],
    "lube": ["lube oil filter dp high", "lube oil pressure low alarm cleared", "oil cooler outlet temperature up",
             "lube oil sample shows metal particles"],
    "surge": ["surge margin low on anti-surge display", "recycle valve opening more than usual",
              "discharge temperature high", "vibration trending up after load change"],
    "winding": ["motor winding temperature high", "current imbalance between phases",
                "smell of burning near motor terminal box", "insulation resistance low on last test"],
    "sticking": ["valve sticking, stroke slow", "positioner hunting around setpoint",
                 "travel deviation alarm on valve", "valve not following output below 20%"],
    "packing": ["gland packing leak on valve stem", "stem leak increasing", "packing retightened, still weeping"],
    "drift": ["reading drifting compared with local gauge", "transmitter reading noisy", "intermittent bad PV",
              "value frozen for a few minutes"],
    "impulse": ["impulse line weeping at fitting", "impulse line partly plugged, slow response",
                "manifold valve leaking"],
}
BREAKDOWN = {
    "bearing": "Breakdown: bearing failed, {a} stopped. Bearing replaced.",
    "seal": "Breakdown: mechanical seal failed with heavy leak, {a} isolated. Seal replaced.",
    "cavitation": "Breakdown: impeller damaged by cavitation, {a} out of service. Impeller replaced.",
    "lube": "Breakdown: lube oil system failure tripped {a}. Pump and filters overhauled.",
    "surge": "Breakdown: {a} tripped on surge, damage to seals. Overhaul done.",
    "winding": "Breakdown: motor winding burnt, {a} tripped. Motor rewound.",
    "sticking": "Breakdown: valve stuck, {a} could not control. Actuator overhauled.",
    "packing": "Breakdown: heavy stem leak, {a} isolated for repacking.",
    "drift": "Breakdown: transmitter failed, {a} reading bad. Transmitter replaced.",
    "impulse": "Breakdown: impulse line blocked, {a} reading lost. Line cleared.",
    "sudden": "Breakdown: {a} tripped suddenly, cause found during repair. Repaired and returned to service.",
}
FIXED = {  # corrective work order that removes a degradation before it becomes a breakdown
    "bearing": "Corrective: bearing replaced on {a} after high vibration, vibration now normal.",
    "seal": "Corrective: mechanical seal replaced on {a}, no leak after restart.",
    "cavitation": "Corrective: suction strainer cleaned on {a}, flow and pressure back to normal.",
    "lube": "Corrective: lube oil changed and filters replaced on {a}.",
    "surge": "Corrective: anti-surge controller retuned on {a}.",
    "winding": "Corrective: motor terminal box cleaned and reconnected on {a}, currents balanced.",
    "sticking": "Corrective: positioner recalibrated and valve stroked on {a}.",
    "packing": "Corrective: valve {a} repacked, stem leak stopped.",
    "drift": "Corrective: transmitter {a} recalibrated against standard.",
    "impulse": "Corrective: impulse lines of {a} flushed and fittings tightened.",
}
COMMON = ["Routine round, all normal.", "Checked, no abnormality.", "PM completed, no findings.",
          "Area housekeeping done.", "Painting and insulation repair done."]
ROTATING = ["Greased bearings as per PM.", "Oil topped up.", "Running normal, standby checked.",
            "Guard refitted after inspection.", "Strainer cleaned as per PM."]
ROUTINE = {
    "pump": COMMON + ROTATING, "compressor": COMMON + ROTATING, "motor": COMMON + ROTATING[:4],
    "control_valve": COMMON + ["Valve stroked full range OK.", "Positioner checked, signal OK."],
    "transmitter": COMMON + ["Calibration within tolerance.", "Zero check done, OK."],
}
FALSE_ALARMS = [
    "vibration slightly high, rechecked normal", "small drip at seal, stopped after flushing",
    "temperature reading high, local gauge normal", "noise heard, found loose guard, tightened",
    "valve slow once, normal on second stroke",
]
INCIDENTS = [
    "Incident report: minor leak near {a}, area barricaded.", "Incident report: housekeeping issue at {a} corrected.",
    "Incident report: {a} trip caused short process upset.", "Incident report: alarm flood after {a} disturbance.",
]

ABBREV = {"temperature": ["temp"], "vibration": ["vib", "vibn"], "bearing": ["brg"], "mechanical": ["mech"],
          "pressure": ["press"], "increasing": ["incr"], "replaced": ["repl"], "normal": ["nrml"],
          "high": ["hi"], "low": ["lo"], "leak": ["lkg"], "valve": ["vlv"]}


def noisy(text, rng):
    for w, shorts in ABBREV.items():
        pat = re.compile(rf"\b{w}\b")
        if pat.search(text) and rng.random() < 0.4:
            text = pat.sub(rng.choice(shorts), text)
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 5 and w.isalpha() and rng.random() < 0.03:
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    text = " ".join(words)
    return text.upper() if rng.random() < 0.15 else text


def assets(rng):
    rows = []
    for typ, spec in ASSET_TYPES.items():
        for k in range(spec["n"]):
            unit = rng.choice([10, 20, 30, 40, 50, 60])
            tag = f"{spec['prefix']}-{unit}{k + 1:02d}" + ("A" if typ in ("pump", "motor") and rng.random() < 0.5 else "")
            rows.append(dict(asset_id=tag, asset_type=typ, unit=f"U{unit}", criticality=rng.choice(["A", "B", "B", "C"]),
                             install_year=rng.randint(1995, 2018)))
    return pd.DataFrame(rows)


def generate(seed=42):
    rng = random.Random(seed)
    reg = assets(rng)
    logs, failures, episodes = [], [], []

    def log(d, a, src, text, wo_type=""):
        logs.append(dict(date=(START + timedelta(days=int(d))).isoformat(), asset_id=a, source=src,
                         wo_type=wo_type, text=noisy(text, rng)))

    for r in reg.itertuples():
        modes = ASSET_TYPES[r.asset_type]["modes"]
        busy = set()  # days already inside a degradation episode
        # Breakdowns and caught (fixed) degradations, as two Poisson processes.
        t = rng.expovariate(1 / 700)
        while t < DAYS:
            mode = rng.choice(modes)
            sudden = rng.random() < 0.2
            lead = 0 if sudden else rng.randint(14, 77)
            fail_day = int(t)
            failures.append(dict(asset_id=r.asset_id, date=(START + timedelta(days=fail_day)).isoformat(),
                                 failure_mode="sudden" if sudden else mode, warning_days=lead))
            episodes.append(dict(asset_id=r.asset_id, start=fail_day - lead, end=fail_day,
                                 mode=mode, outcome="sudden_breakdown" if sudden else "breakdown"))
            log(fail_day, r.asset_id, "corrective_wo", BREAKDOWN["sudden" if sudden else mode].format(a=r.asset_id),
                "breakdown")
            busy.update(range(fail_day - lead, fail_day + 1))
            t += rng.expovariate(1 / 700) + 14
        t = rng.expovariate(1 / 1100)
        while t < DAYS:
            mode, lead = rng.choice(modes), rng.randint(14, 60)
            fix_day = int(t)
            if not busy.intersection(range(fix_day - lead, fix_day + 1)):
                episodes.append(dict(asset_id=r.asset_id, start=fix_day - lead, end=fix_day, mode=mode, outcome="fixed"))
                log(fix_day, r.asset_id, "corrective_wo", FIXED[mode].format(a=r.asset_id), "corrective")
            t += rng.expovariate(1 / 1100) + 14

        # Text written during degradation episodes: symptom rate ramps from ~0.02 to ~0.25 per day.
        for e in [e for e in episodes if e["asset_id"] == r.asset_id and e["end"] > e["start"]]:
            span = e["end"] - e["start"]
            for d in range(max(e["start"], 0), min(e["end"], DAYS)):
                frac = (d - e["start"]) / span
                if rng.random() < 0.02 + 0.23 * frac ** 1.5:
                    s = SYMPTOMS[e["mode"]][min(int(frac * len(SYMPTOMS[e["mode"]])), len(SYMPTOMS[e["mode"]]) - 1)]
                    if rng.random() < 0.3:
                        s = rng.choice(SYMPTOMS[e["mode"]])
                    log(d, r.asset_id, rng.choice(["operator_round", "operator_round", "pm_task"]),
                        s.format(t=rng.randint(70, 95)) + ".")

        # Background text for every asset.
        for d in range(DAYS):
            if rng.random() < 0.06:
                log(d, r.asset_id, rng.choice(["operator_round", "pm_task"]), rng.choice(ROUTINE[r.asset_type]))
            if rng.random() < 0.004:
                log(d, r.asset_id, "operator_round", rng.choice(FALSE_ALARMS) + ".")
            if rng.random() < 0.0008:
                log(d, r.asset_id, "incident_report", rng.choice(INCIDENTS).format(a=r.asset_id))

    logs = pd.DataFrame(logs).sort_values(["date", "asset_id"]).reset_index(drop=True)
    logs.insert(0, "log_id", [f"L{i + 1:06d}" for i in range(len(logs))])
    ep = pd.DataFrame(episodes)
    ep["start"] = ep.start.map(lambda d: (START + timedelta(days=int(d))).isoformat())
    ep["end"] = ep.end.map(lambda d: (START + timedelta(days=int(d))).isoformat())
    return reg, logs, pd.DataFrame(failures).sort_values("date"), ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    reg, logs, failures, ep = generate(ap.parse_args().seed)
    (OUT / "ground_truth").mkdir(parents=True, exist_ok=True)
    reg.to_csv(OUT / "asset_register.csv", index=False)
    logs.to_csv(OUT / "maintenance_logs.csv", index=False)
    ep.to_csv(OUT / "ground_truth" / "degradation_episodes.csv", index=False)
    failures.to_csv(OUT / "ground_truth" / "failure_modes.csv", index=False)
    (OUT.parent / "sample").mkdir(exist_ok=True)
    logs.head(200).to_csv(OUT.parent / "sample" / "maintenance_logs_sample.csv", index=False)
    print(f"{len(reg)} assets, {len(logs)} log entries, {len(failures)} breakdowns, "
          f"{(ep.outcome == 'fixed').sum()} degradations fixed before breakdown")


if __name__ == "__main__":
    main()
