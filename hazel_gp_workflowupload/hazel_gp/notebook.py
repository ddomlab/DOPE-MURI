"""Small optional notebook controls; model code does not depend on widgets."""
from pathlib import Path
from IPython.display import display, clear_output


def export_controls(prepared, root):
    import ipywidgets as w
    from .data import export_prepared, create_upload_archive
    root = Path(root)
    directory = w.Text(value=str(root / prepared.config["paths"]["prepared"]), description="Prepared:",
                       layout=w.Layout(width="95%"), style={"description_width": "90px"})
    archive = w.Text(value=str(root / "exports/hazel_gp_upload_v2.zip"), description="Upload ZIP:",
                     layout=w.Layout(width="95%"), style={"description_width": "90px"})
    button = w.Button(description="Export reviewed inputs", button_style="success", layout=w.Layout(width="220px"))
    output = w.Output()

    def clicked(_):
        with output:
            clear_output(wait=True)
            try:
                # Precheck both destinations before the first write.
                if Path(directory.value).exists() or Path(archive.value).exists():
                    raise FileExistsError("Choose new prepared and ZIP destinations to preserve the previous version.")
                dest = export_prepared(prepared, directory.value)
                zip_path = create_upload_archive(root, dest, archive.value)
                print(f"Prepared: {dest}\nUpload archive: {zip_path}\nTasks: {len(prepared.tasks)}")
            except Exception as exc:
                print(f"Export stopped: {exc}")
    button.on_click(clicked)
    return w.VBox([directory, archive, button, output])


def review_controls(tables):
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from .results import plot_parity
    pred = tables["predictions"]
    model = w.Dropdown(options=pred.model.unique().tolist(), description="Model:")
    method = w.Dropdown(options=pred.method.unique().tolist(), description="Method:")
    group = w.Dropdown(options=[("All", "")], description="Reference:")
    out = w.Output()

    def update_groups(change=None):
        part = pred[pred.method == method.value]
        values = [x for x in part.reference_group.unique().tolist() if x]
        group.options = [("All", "")] + [(x, x) for x in values]

    def redraw(change=None):
        with out:
            clear_output(wait=True)
            try:
                fig = plot_parity(pred, model.value, method.value, group.value or None)
                display(fig)
                plt.close(fig)
            except ValueError as exc:
                print(str(exc))
                return
            part = tables["metrics_by_split"]
            part = part[(part.model == model.value) & (part.method == method.value)]
            if group.value:
                part = part[part.reference_group == group.value]
            display(part)
    method.observe(update_groups, names="value")
    for control in (model, method, group):
        control.observe(redraw, names="value")
    update_groups()
    redraw()
    return w.VBox([w.HBox([model, method, group]), out])
