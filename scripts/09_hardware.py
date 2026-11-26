"""Run the trained VQC head on IBM quantum hardware, and on simulators of it.

One trained model (fixed in advance, never chosen by test score) is
evaluated on the SAME images under four conditions:

  exact       ideal simulation, exact <Z_i>                     (Qiskit Statevector)
  shots       ideal simulation, <Z_i> estimated from 1024 shots (Aer, no noise)
  calibrated  noisy simulation built from an IBM device's measured
              error rates (gate errors, readout errors, T1/T2)   (Aer, from backend)
  hardware    the real IBM processor, 1024 shots                (Qiskit Runtime)

Images: 100 stratified test images (scored) + 100 stratified validation
images (used only to recalibrate the decision threshold under each condition).

Steps (run in order):
  python scripts/09_hardware.py prepare                      # pick images, build circuit, zero-noise check
  python scripts/09_hardware.py simulate --fake FakeTorino   # exact, shots, calibrated (offline device copy)
  python scripts/09_hardware.py submit   --backend ibm_xxx   # send ONE job to the real device (needs saved IBM account)
  python scripts/09_hardware.py fetch                        # download the job result when it has finished
  python scripts/09_hardware.py analyze                      # comparison table

Outputs go to results/hardware/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import ReducedData  # noqa: E402
from src.features import load_features  # noqa: E402
from src.heads import build_head  # noqa: E402
from src.metrics import evaluate, youden_threshold  # noqa: E402
from src.utils import load_config  # noqa: E402

# ---- fixed in advance (pre-registered choice, not tuned on test results) ----
MODEL = dict(head="vqc", d=4, n=1000, fold=0, seed=0)
N_TEST, N_VAL, SHOTS, SEED = 100, 100, 1024, 2026


# ----------------------------------------------------------------------------- helpers
def load_model(cfg):
    m = MODEL
    path = Path(cfg["paths"]["results"]) / "weights" / f"{m['head']}_d{m['d']}_n{m['n']}_f{m['fold']}_s{m['seed']}.pt"
    model = build_head(m["head"], m["d"], cfg)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def stratified_pick(y, k, rng):
    """k indices with the same class ratio as y."""
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        take = int(round(k * len(pool) / len(y)))
        idx += list(rng.choice(pool, size=take, replace=False))
    return np.sort(np.array(idx[:k]))


def build_circuit(model):
    """The SAME circuit as VQCHead, in Qiskit. Inputs x are free parameters;
    trained angles are fixed numbers. Qiskit's RY/RZ match PennyLane's."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    d, w = model.d, model.weights.detach().numpy()
    x = ParameterVector("x", d)
    qc = QuantumCircuit(d)
    for i in range(d):
        qc.ry(x[i], i)
    for l in range(model.layers):
        for i in range(d):
            qc.ry(float(w[l, i, 0]), i)
            qc.rz(float(w[l, i, 1]), i)
            qc.ry(float(w[l, i, 2]), i)
        if model.entangle and d > 1:
            for i in range(d):
                qc.cx(i, (i + 1) % d)
    return qc, x


def exact_expvals(qc, X):
    """Exact <Z_i>, shape (images, d). Z_i built by qubit INDEX, so Qiskit's
    reversed bit order cannot silently swap qubits."""
    from qiskit.quantum_info import SparsePauliOp, Statevector
    d = qc.num_qubits
    ops = [SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=d) for i in range(d)]
    out = np.empty((len(X), d))
    for k, xk in enumerate(X):
        sv = Statevector(qc.assign_parameters(xk))
        out[k] = [sv.expectation_value(o).real for o in ops]
    return out


def z_from_bitarray(bits, d):
    """<Z_i> per image from measured bitstrings. In a Qiskit bitstring the
    RIGHTMOST character is qubit 0, so qubit i is character -1-i."""
    out = np.empty((bits.shape[0], d))
    for k in range(bits.shape[0]):
        counts = bits.get_counts(loc=k)
        total = sum(counts.values())
        for i in range(d):
            ones = sum(c for b, c in counts.items() if b[-1 - i] == "1")
            out[k, i] = 1 - 2 * ones / total
    return out


def run_sampler(mode, qc, X):
    """Transpile for `mode` (a device, a simulator of one, or an ideal simulator),
    run all images as ONE job, return (<Z_i> array, transpiled circuit, job)."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    meas = qc.copy()
    meas.measure_all()
    isa = generate_preset_pass_manager(optimization_level=3, backend=mode, seed_transpiler=SEED).run(meas)
    sampler = SamplerV2(mode=mode)
    sampler.options.default_shots = SHOTS
    job = sampler.run([(isa, X)])
    return job, isa


def resources(isa):
    ops = dict(isa.count_ops())
    two_q = sum(1 for ins in isa.data if ins.operation.num_qubits == 2)
    return {"depth": isa.depth(), "two_qubit_gates": two_q, "ops": {k: int(v) for k, v in ops.items()},
            "physical_qubits": [int(q) for q in isa.layout.final_index_layout()[: MODEL["d"]]]}


def device_errors(backend, qubits):
    """Reported error rates on the qubits actually used (for the report)."""
    t = backend.target
    ro = [t["measure"][(q,)].error for q in qubits if t["measure"].get((q,)) and t["measure"][(q,)].error is not None]
    twoq = []
    for name in ("cz", "ecr", "cx"):
        if name in t.operation_names:
            for (a, b), p in t[name].items():
                if a in qubits and b in qubits and p is not None and p.error is not None:
                    twoq.append(p.error)
    return {"mean_readout_error": float(np.mean(ro)) if ro else None,
            "mean_two_qubit_error": float(np.mean(twoq)) if twoq else None}


# ----------------------------------------------------------------------------- steps
def prepare(cfg, out):
    data = ReducedData(load_features(cfg["paths"]["features"]), [MODEL["d"]], cfg["clip_percentiles"])
    X, y = data.X[MODEL["d"]], data.y
    rng = np.random.default_rng(SEED)
    ti, vi = stratified_pick(y["test"], N_TEST, rng), stratified_pick(y["val"], N_VAL, rng)
    model = load_model(cfg)
    qc, _ = build_circuit(model)

    # Zero-noise check: Qiskit circuit must reproduce the trained PennyLane model exactly
    Xs = np.vstack([X["test"][ti], X["val"][vi]])
    with torch.no_grad():
        ref = model(torch.as_tensor(Xs, dtype=torch.float32)).numpy()
    z = exact_expvals(qc, Xs)
    got = model.scale.item() * z.mean(axis=1) + model.bias.item()
    err = float(np.abs(got - ref).max())
    assert err < 1e-5, f"Qiskit circuit does not match the trained model (max diff {err:.2e})"

    sel = {"model": MODEL, "test_idx": ti.tolist(), "val_idx": vi.tolist(),
           "scale": model.scale.item(), "bias": model.bias.item(), "zero_noise_max_diff": err,
           "test_pos": int(y["test"][ti].sum()), "val_pos": int(y["val"][vi].sum())}
    json.dump(sel, open(out / "selection.json", "w"), indent=1)
    np.save(out / "X.npy", Xs)
    np.save(out / "y.npy", np.concatenate([y["test"][ti], y["val"][vi]]))
    print(qc.draw(fold=120))
    print(f"zero-noise check PASSED (max diff {err:.1e}); {N_TEST} test ({sel['test_pos']} pneumonia), "
          f"{N_VAL} val ({sel['val_pos']} pneumonia) images saved")


def simulate(cfg, out, fake, backend_name):
    from qiskit_aer import AerSimulator
    model = load_model(cfg)
    qc, _ = build_circuit(model)
    X = np.load(out / "X.npy")
    np.save(out / "z_exact.npy", exact_expvals(qc, X))

    job, _ = run_sampler(AerSimulator(seed_simulator=SEED), qc, X)
    np.save(out / "z_shots.npy", z_from_bitarray(job.result()[0].data.meas, qc.num_qubits))

    if backend_name:
        from qiskit_ibm_runtime import QiskitRuntimeService
        device = QiskitRuntimeService().backend(backend_name)
    else:
        from qiskit_ibm_runtime import fake_provider
        device = getattr(fake_provider, fake)()
    noisy = AerSimulator.from_backend(device, seed_simulator=SEED)
    job, isa = run_sampler(noisy, qc, X)
    np.save(out / "z_calibrated.npy", z_from_bitarray(job.result()[0].data.meas, qc.num_qubits))
    info = {"device": device.name, **resources(isa)}
    info.update(device_errors(device, info["physical_qubits"]))
    json.dump(info, open(out / "calibrated_info.json", "w"), indent=1)
    print("simulations saved;", json.dumps(info))


def submit(cfg, out, backend_name):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    device = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    model = load_model(cfg)
    qc, _ = build_circuit(model)
    X = np.load(out / "X.npy")
    job, isa = run_sampler(device, qc, X)
    info = {"device": device.name, "job_id": job.job_id(), **resources(isa)}
    info.update(device_errors(device, info["physical_qubits"]))
    json.dump(info, open(out / "hardware_job.json", "w"), indent=1)
    print(f"submitted job {job.job_id()} to {device.name}:", json.dumps(info))


def fetch(out):
    from qiskit_ibm_runtime import QiskitRuntimeService
    info = json.load(open(out / "hardware_job.json"))
    job = QiskitRuntimeService().job(info["job_id"])
    print("job status:", job.status())
    res = job.result()                                   # waits if still queued or running
    np.save(out / "z_hardware.npy", z_from_bitarray(res[0].data.meas, MODEL["d"]))
    print("hardware result saved")


def analyze(cfg, out):
    sel = json.load(open(out / "selection.json"))
    y = np.load(out / "y.npy")
    yt, yv = y[:N_TEST], y[N_TEST:]
    runs = pd.read_csv(Path(cfg["paths"]["results"]) / "vqc.csv")
    m = MODEL
    thr0 = float(runs.query("head == @m['head'] and d == @m['d'] and n == @m['n'] "
                            "and fold == @m['fold'] and seed == @m['seed']")["threshold"].iloc[0])
    rows = []
    for cond in ["exact", "shots", "calibrated", "hardware"]:
        f = out / f"z_{cond}.npy"
        if not f.exists():
            continue
        s = sel["scale"] * np.load(f).mean(axis=1) + sel["bias"]
        st, sv = s[:N_TEST], s[N_TEST:]
        fixed = evaluate(yt, st, thr0)
        recal = evaluate(yt, st, youden_threshold(yv, sv))
        rows.append({"condition": cond, "test_auroc": fixed["auroc"],
                     "bal_acc_noiseless_threshold": fixed["balanced_accuracy"],
                     "bal_acc_recalibrated_threshold": recal["balanced_accuracy"],
                     "score_spread_sd": float(np.std(st))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "hardware_comparison.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "simulate", "submit", "fetch", "analyze"])
    ap.add_argument("--fake", default="FakeTorino", help="offline copy of an IBM device for the calibrated simulation")
    ap.add_argument("--backend", default=None, help="real IBM device name, e.g. ibm_torino")
    a = ap.parse_args()
    cfg = load_config()
    out = Path(cfg["paths"]["results"]) / "hardware"
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "prepare":
        prepare(cfg, out)
    elif a.step == "simulate":
        simulate(cfg, out, a.fake, a.backend)
    elif a.step == "submit":
        submit(cfg, out, a.backend)
    elif a.step == "fetch":
        fetch(out)
    else:
        analyze(cfg, out)
