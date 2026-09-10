"""End-to-end small synthetic benchmark: completeness, aggregation and corruption guards."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from hazel_gp.config import default_config
from hazel_gp.data import PreparedData, export_prepared, verify_bundle
from hazel_gp.features import ReferencePCA, ligand_table
from hazel_gp.splits import make_splits
from hazel_gp.train import run_task, predict_from_checkpoint
from hazel_gp.results import collect_results


def test_complete_benchmark_collection_and_integrity(tmp_path):
    cfg = default_config()
    cfg["features"]["models"] = ["selected_2"]
    cfg["data"].update(expected_rows=None, expected_ligands=None,
                        common_categorical=["substrate_pair"])
    cfg["evaluation"]["iid_repeats"] = 2
    cfg["gp"].update(max_steps=3, min_steps=1, patience=2, threads=1)
    rng = np.random.default_rng(5)
    cols = ["vbur_vbur_boltz", "vbur_vbur_min", "vbur_vbur_delta", "dipolemoment_boltz",
            "fmo_e_homo_boltz", "fmo_e_lumo_boltz", "a", "b"]
    desc = pd.DataFrame(rng.normal(size=(24, 8)), columns=cols)
    desc.insert(0, "id", np.arange(24))
    ref = ReferencePCA.fit(desc)
    reactions = pd.DataFrame({"row_id": [f"synthetic:{i}" for i in range(40)],
                              "Reaction_No": np.arange(40), "ligand": np.repeat(["A", "B"], 20),
                              "kraken_id": np.repeat([0, 1], 20),
                              "substrate_pair": np.tile(["pair1", "pair2"], 20),
                              "Product_Yield_PCT_Area_UV": 10 + 20 * np.repeat([0, 1], 20) + np.sin(np.arange(40))})
    entries, arrays, tasks = make_splits(reactions, cfg)
    prepared = PreparedData(cfg, reactions, desc, pd.DataFrame(), ligand_table(desc, ref, cfg), ref,
                            pd.DataFrame(columns=["row_id", "exclusion_reason"]),
                            pd.DataFrame({"ligand": ["A", "B"], "id": [0, 1]}),
                            {"synthetic_test": True}, {}, entries, arrays, tasks)
    bundle = export_prepared(prepared, tmp_path / "inputs")
    runs = tmp_path / "runs"
    run_task(bundle, runs, 0, device="cpu")
    with pytest.raises(RuntimeError, match="incomplete"):
        collect_results(bundle, runs)
    partial = collect_results(bundle, runs, allow_partial=True)
    assert not partial["summary"].run_complete.any()
    for task_id in tasks.task_id:
        run_task(bundle, runs, int(task_id), device="cpu")
    tables = collect_results(bundle, runs)
    assert tables["summary"].run_complete.all() and tables["summary"].complete_method.all()
    assert len(tables["metrics_by_split"]) == 12
    lolo = tables["predictions"].query("method == 'lolo'")
    direct_r2 = 1 - np.square(lolo.y_true - lolo.y_pred).sum() / np.square(lolo.y_true - lolo.y_true.mean()).sum()
    summary = tables["summary"].query("method == 'lolo'").set_index("aggregation")
    assert summary.loc["pooled_predictions", "r2"] == pytest.approx(direct_r2)
    assert abs(summary.loc["pooled_predictions", "r2"] - summary.loc["mean_split_metrics", "r2"]) > 0.1
    assert not tables["summary"].query("method == 'iid_matched'").aggregation.eq("pooled_predictions").any()
    # Completed jobs skip without overwriting their saved checkpoint.
    checkpoint = runs / "tasks/task_0000/model.pt"
    before = checkpoint.stat().st_mtime_ns
    run_task(bundle, runs, 0, device="cpu")
    assert checkpoint.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="different"):
        run_task(bundle, runs, 0, device="cpu", max_steps=2)
    # Data/result corruption must stop collection and training reuse.
    input_file = bundle / "reactions.csv"
    original = input_file.read_bytes()
    input_file.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(bundle)
    input_file.write_bytes(original)
    pred_file = runs / "tasks/task_0000/predictions.csv"
    original = pred_file.read_bytes()
    pred_file.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Corrupt"):
        collect_results(bundle, runs)
    pred_file.write_bytes(original)
