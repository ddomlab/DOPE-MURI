"""One ExactGP fit per task, with atomic results and portable CPU checkpoints."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import os
import random
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import gpytorch

from .config import TARGET, read_json, write_json, object_hash, file_hash, versions
from .data import load_bundle, verify_bundle
from .features import feature_frame, fit_transform_fold
from .splits import load_split, validate_split


class ExactYieldGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, settings):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        lo, hi = settings["lengthscale_bounds"]
        base = gpytorch.kernels.RBFKernel(
            ard_num_dims=train_x.shape[-1] if settings["ard"] else None,
            lengthscale_constraint=gpytorch.constraints.Interval(lo, hi))
        self.covar_module = gpytorch.kernels.ScaleKernel(base) if settings["learn_outputscale"] else base
        self.base_kernel.lengthscale = settings["initial_lengthscale"]
        if settings["learn_outputscale"]:
            self.covar_module.outputscale = 1.0

    @property
    def base_kernel(self):
        return self.covar_module.base_kernel if isinstance(self.covar_module, gpytorch.kernels.ScaleKernel) else self.covar_module

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def build_model(x, y, settings):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-12)).to(device=x.device, dtype=x.dtype)
    likelihood.noise = settings["noise_variance"]
    likelihood.raw_noise.requires_grad_(settings["learn_noise"])
    model = ExactYieldGP(x, y, likelihood, settings).to(device=x.device, dtype=x.dtype)
    # Initialize constrained parameters after dtype conversion: otherwise inverse
    # transforms performed in float32 perturb a nominal float64 baseline.
    model.base_kernel.lengthscale = settings["initial_lengthscale"]
    if settings["learn_outputscale"]:
        model.covar_module.outputscale = 1.0
    return model, likelihood


def solver_context(settings):
    stack = ExitStack()
    exact = settings["solver"] == "cholesky"
    stack.enter_context(gpytorch.settings.max_cholesky_size(10**9 if exact else 0))
    stack.enter_context(gpytorch.settings.fast_computations(
        covar_root_decomposition=not exact, log_prob=not exact, solves=not exact))
    stack.enter_context(gpytorch.settings.cholesky_jitter(double_value=settings["cholesky_jitter"]))
    stack.enter_context(gpytorch.settings.cg_tolerance(settings["cg_tolerance"]))
    stack.enter_context(gpytorch.settings.eval_cg_tolerance(settings["cg_tolerance"]))
    stack.enter_context(gpytorch.settings.max_cg_iterations(settings["max_cg_iterations"]))
    stack.enter_context(gpytorch.settings.fast_pred_var(False))
    return stack


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def fit_exact(x, y, settings, seed=42, device="cpu"):
    """Target-standardized marginal-likelihood optimization; never reads test targets."""
    torch.set_num_threads(settings["threads"])
    dtype = torch.float64
    train_x = torch.as_tensor(x, dtype=dtype, device=device)
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    if std == 0.0:
        std = 1.0
    train_y = torch.as_tensor((y - mean) / std, dtype=dtype, device=device)
    best_loss, best_state, best_restart = float("inf"), None, None
    history, restart_summaries, captured = [], [], []
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_log, solver_context(settings):
        warnings.simplefilter("always")
        for restart in range(settings["restarts"]):
            restart_seed = seed + restart
            random.seed(restart_seed)
            np.random.seed(restart_seed)
            torch.manual_seed(restart_seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(restart_seed)
            model, likelihood = build_model(train_x, train_y, settings)
            if restart:
                rng = np.random.default_rng(restart_seed)
                lo, hi = settings["lengthscale_bounds"]
                initial = np.clip(settings["initial_lengthscale"] * np.exp(rng.normal(0, 1)), lo * 1.01, hi / 1.01)
                model.base_kernel.lengthscale = float(initial)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings["learning_rate"])
            objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            local_best, significant_best, stale = float("inf"), float("inf"), 0
            stop_reason = "max_steps"
            for step in range(settings["max_steps"]):
                optimizer.zero_grad(set_to_none=True)
                loss = -objective(model(train_x), train_y)
                value = float(loss.detach().cpu())
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite GP objective")
                history.append({"restart": restart, "step": step + 1, "negative_mll_per_point": value,
                                "lengthscale_mean": float(model.base_kernel.lengthscale.detach().mean().cpu()),
                                "noise_variance_standardized": float(likelihood.noise.detach().cpu().item())})
                local_best = min(local_best, value)
                if value < best_loss:
                    best_loss, best_state, best_restart = value, cpu_state(model), restart
                threshold = settings["relative_tolerance"] * max(1, abs(significant_best))
                if not np.isfinite(significant_best) or significant_best - value > threshold:
                    significant_best, stale = value, 0
                else:
                    stale += 1
                if step + 1 >= settings["min_steps"] and stale >= settings["patience"]:
                    stop_reason = "training_objective_plateau"
                    break
                loss.backward()
                for param in model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise FloatingPointError("Nonfinite GP gradient")
                optimizer.step()
            # Evaluate the last updated state as well; never select an unscored state.
            with torch.no_grad():
                final_loss = float((-objective(model(train_x), train_y)).cpu())
            if not np.isfinite(final_loss):
                raise FloatingPointError("Nonfinite final GP objective")
            if final_loss < best_loss:
                best_loss, best_state, best_restart = final_loss, cpu_state(model), restart
            restart_summaries.append({"restart": restart, "steps": step + 1, "stop_reason": stop_reason,
                                      "best_negative_mll_per_point": min(local_best, final_loss)})
        captured = sorted(set(str(w.message) for w in warning_log))
    model, likelihood = build_model(train_x, train_y, settings)
    model.load_state_dict(best_state)
    model.eval()
    likelihood.eval()
    summary = {"fit_seconds": time.perf_counter() - t0, "target_mean": mean, "target_std": std,
               "best_negative_mll_per_point": best_loss, "selected_restart": best_restart,
               "restart_summaries": restart_summaries, "warnings": captured,
               "lengthscales": model.base_kernel.lengthscale.detach().cpu().reshape(-1).tolist(),
               "noise_variance_standardized": float(likelihood.noise.detach().cpu().item()),
               "noise_variance_target_units": float(likelihood.noise.detach().cpu().item()) * std**2,
               "outputscale": float(model.covar_module.outputscale.detach().cpu()) if settings["learn_outputscale"] else 1.0}
    return model, likelihood, summary, pd.DataFrame(history)


def predict_exact(model, likelihood, x, summary, settings):
    means, latent, predictive = [], [], []
    device = next(model.parameters()).device
    with torch.no_grad(), solver_context(settings):
        for start in range(0, len(x), settings["prediction_batch_size"]):
            xx = torch.as_tensor(x[start:start + settings["prediction_batch_size"]], dtype=torch.float64, device=device)
            f = model(xx)
            obs = likelihood(f)
            means.append(f.mean.cpu().numpy() * summary["target_std"] + summary["target_mean"])
            latent.append(f.variance.clamp_min(0).sqrt().cpu().numpy() * summary["target_std"])
            predictive.append(obs.variance.clamp_min(0).sqrt().cpu().numpy() * summary["target_std"])
    if not means:
        raise ValueError("Prediction requires at least one row")
    return np.concatenate(means), np.concatenate(latent), np.concatenate(predictive)


def code_fingerprint():
    return object_hash({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def run_task(bundle, runs, task_id, device=None, max_steps=None, retry=False):
    from .results import regression_metrics
    bundle, runs = Path(bundle).resolve(), Path(runs).resolve()
    manifest, cfg, reactions, ligands, reference = load_bundle(bundle)
    settings = copy.deepcopy(cfg["gp"])
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        settings["max_steps"] = int(max_steps)
        settings["min_steps"] = min(settings["min_steps"], settings["max_steps"])
    requested_device = device or settings["device"]
    if requested_device not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid device")
    actual_device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if actual_device == "auto":
        actual_device = "cpu"
    if actual_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check GPU allocation and the installed PyTorch wheel.")
    settings["device"] = requested_device
    environment_versions = {k: v for k, v in versions().items() if k != "platform"}
    identity = {"bundle_id": manifest["bundle_id"], "code_hash": code_fingerprint(),
                "settings": settings, "software_versions": environment_versions, "schema_version": 1}
    run_id = object_hash(identity)
    # Immutable run specification with race-safe create and atomic rename.
    runs.mkdir(parents=True, exist_ok=True)
    run_meta = runs / "run.json"
    run_lock = runs / ".initializing"
    for _ in range(100):
        if run_meta.exists():
            if read_json(run_meta)["run_id"] != run_id:
                raise ValueError("Run directory belongs to different code, input data or settings. Use a new runs folder.")
            break
        try:
            fd = os.open(run_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(0.1)
            continue
        else:
            os.close(fd)
            try:
                if run_meta.exists():
                    if read_json(run_meta)["run_id"] != run_id:
                        raise ValueError("Concurrent initialization selected a different run")
                else:
                    shutil_copy_inputs(bundle, runs)
                    write_json(run_meta, {**identity, "run_id": run_id, "n_tasks": manifest["n_tasks"]})
            finally:
                run_lock.unlink(missing_ok=True)
            break
    else:
        raise RuntimeError("Run initialization is busy or interrupted; inspect .initializing before retrying")
    tasks = pd.read_csv(bundle / "tasks.csv", keep_default_na=False)
    selected = tasks.loc[tasks.task_id == int(task_id)]
    if len(selected) != 1:
        raise ValueError(f"Unknown task_id={task_id}")
    task = selected.iloc[0].to_dict()
    out = runs / "tasks" / f"task_{int(task_id):04d}"
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".running"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', '')}\n".encode())
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"Task locked: {lock}. Confirm the old job has ended before removing a stale lock.") from exc
    try:
        status_path = out / "status.json"
        if status_path.exists():
            old = read_json(status_path)
            if old.get("run_id") != run_id:
                raise ValueError("Task belongs to a different run")
            if old["state"] == "completed":
                for name, digest in old["output_hashes"].items():
                    if file_hash(out / name) != digest:
                        raise ValueError(f"Completed output changed: {out / name}")
                return old
            if not retry:
                raise RuntimeError("Task already attempted; pass --retry after reviewing its failure/status")
        write_json(status_path, {"state": "running", "run_id": run_id, "task": task})
        split, train, test = load_split(bundle, task["split_id"])
        validate_split(train, test, len(reactions), reactions.ligand.to_numpy(), split["method"], split["reference_group"])
        frame = feature_frame(reactions, ligands, task["model"], cfg, reference)
        xtr, xte, pre, names, unknown = fit_transform_fold(frame, train, test, task["model"], cfg, reference)
        y = reactions[TARGET].to_numpy(dtype=float)
        model, likelihood, summary, history = fit_exact(xtr, y[train], settings, int(task["fit_seed"]), actual_device)
        mean, latent, predictive = predict_exact(model, likelihood, xte, summary, settings)
        pred = reactions.iloc[test][["row_id", "Reaction_No", "ligand"]].copy().reset_index(drop=True)
        pred["test_index"] = test
        for key in ("task_id", "model", "split_id", "method", "reference_group", "repeat", "fold"):
            pred[key] = task[key]
        pred["y_true"], pred["y_pred"], pred["latent_std"], pred["predictive_std"] = y[test], mean, latent, predictive
        pred.to_csv(out / "predictions.csv", index=False)
        history.to_csv(out / "training_history.csv", index=False)
        write_json(out / "metrics.json", regression_metrics(y[test], mean, predictive))
        write_json(out / "fit.json", {**summary, "task": task, "n_train": len(train), "n_test": len(test),
                                      "n_features": len(names), "feature_names": names, "unknown_categories": unknown,
                                      "device": actual_device, "gpu_name": torch.cuda.get_device_name() if actual_device == "cuda" else None,
                                      "versions": versions(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                      "settings": settings, "run_id": run_id})
        joblib.dump(pre, out / "preprocessor.joblib")
        outputs = ["predictions.csv", "training_history.csv", "metrics.json", "fit.json", "preprocessor.joblib"]
        if settings["save_checkpoints"]:
            checkpoint = {"model_state": cpu_state(model), "train_x": model.train_inputs[0].detach().cpu(),
                          "train_y_standardized": model.train_targets.detach().cpu(),
                          "settings": settings, "target_mean": summary["target_mean"],
                          "target_std": summary["target_std"], "feature_names": names,
                          "train_row_ids": reactions.iloc[train].row_id.tolist(), "bundle_id": manifest["bundle_id"]}
            torch.save(checkpoint, out / "model.pt")
            outputs.append("model.pt")
        status = {"state": "completed", "run_id": run_id, "task": task,
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "output_hashes": {name: file_hash(out / name) for name in outputs}}
        write_json(status_path, status)
        return status
    except BaseException as exc:
        # Preserve an already completed status if later integrity checking failed.
        path = out / "status.json"
        current = read_json(path) if path.exists() else {}
        if current.get("state") != "completed":
            write_json(path, {"state": "failed", "run_id": run_id, "task": task,
                              "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def shutil_copy_inputs(bundle: Path, runs: Path):
    """Small provenance copies let downloaded results identify their source experiment."""
    import shutil
    for name in ("manifest.json", "config.json", "tasks.csv", "sources.json", "model_table.csv", "evaluation_table.csv"):
        shutil.copy2(bundle / name, runs / ("input_" + name))
    # Written by newer preparations; bundles built before it stay usable.
    for name in ("model_features.csv",):
        if (bundle / name).is_file():
            shutil.copy2(bundle / name, runs / ("input_" + name))


def load_checkpoint(task_directory, device="cpu"):
    """Load only checkpoints and joblib preprocessors you created and trust."""
    task_directory = Path(task_directory)
    cp = torch.load(task_directory / "model.pt", map_location=device, weights_only=True)
    x, y = cp["train_x"].to(device), cp["train_y_standardized"].to(device)
    model, likelihood = build_model(x, y, cp["settings"])
    model.load_state_dict(cp["model_state"])
    model.eval()
    likelihood.eval()
    return model, likelihood, cp, joblib.load(task_directory / "preprocessor.joblib")


def predict_from_checkpoint(task_directory, raw_feature_frame, device="cpu"):
    model, likelihood, cp, pre = load_checkpoint(task_directory, device)
    x = np.asarray(pre.transform(raw_feature_frame), dtype=float)
    return predict_exact(model, likelihood, x, cp, cp["settings"])


def check_environment(device="cuda") -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device accessible in this allocation")
    x = torch.eye(8, dtype=torch.float64, device=device)
    torch.linalg.cholesky(x)
    return {**versions(), "device": device, "torch_cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "float64_cholesky": "passed"}
