"""MLflow tracking and visualization for bot showdowns."""

from pathlib import Path
from typing import Any, Self

try:
    import matplotlib

    matplotlib.use("Agg")  # ponytail: headless non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None


_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
    b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class MLflowTracker:
    """MLflow experiment tracker"""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "showdown",
        mock: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.mock = mock
        self.dry_run = dry_run
        self.logged_params: dict[str, Any] = {}
        self.logged_metrics: dict[str, Any] = {}
        self.logged_figures: dict[str, Any] = {}
        self._active_run = None
        self._mlflow = None

        if not self.mock and not self.dry_run:
            try:
                import mlflow

                self._mlflow = mlflow
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
            except Exception:  # noqa: BLE001
                self._mlflow = None

    def __enter__(self) -> Self:
        if self._mlflow is not None and self._active_run is None:
            try:
                self._active_run = self._mlflow.start_run()
            except Exception:  # noqa: BLE001
                self._active_run = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._mlflow is not None and self._active_run is not None:
            try:
                self._mlflow.end_run()
            except Exception:  # noqa: BLE001, S110
                pass
            self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        self.logged_params.update(params)
        if self._mlflow is not None and not self.mock and not self.dry_run:
            try:
                self._mlflow.log_params(params)
            except Exception:  # noqa: BLE001, S110
                pass

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.logged_metrics.update(metrics)
        if self._mlflow is not None and not self.mock and not self.dry_run:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception:  # noqa: BLE001, S110
                pass

    def log_figure(self, fig: Any, artifact_file: str) -> None:
        self.logged_figures[artifact_file] = fig
        if self._mlflow is not None and not self.mock and not self.dry_run:
            try:
                self._mlflow.log_figure(fig, artifact_file)
            except Exception:  # noqa: BLE001, S110
                pass

    def generate_heatmaps(
        self,
        results: dict[str, dict[str, Any]],
        candidate_name: str = "candidate",
    ) -> dict[str, Any]:

        opponents = list(results.keys())

        # Win rate heatmap
        win_rates = [float(results[opp].get("win_rate", 0.0)) for opp in opponents]
        fig_wr, ax_wr = plt.subplots(figsize=(max(4.0, len(opponents) * 1.5), 3.0))
        im_wr = ax_wr.imshow([win_rates], cmap="viridis", aspect="auto", vmin=0, vmax=100)
        ax_wr.set_xticks(range(len(opponents)))
        ax_wr.set_xticklabels(opponents)
        ax_wr.set_yticks([0])
        ax_wr.set_yticklabels([candidate_name])
        ax_wr.set_title(f"Win Rate (%) vs Opponents ({candidate_name})")
        for i, val in enumerate(win_rates):
            ax_wr.text(i, 0, f"{val:.1f}%", ha="center", va="center", color="white")
        fig_wr.colorbar(im_wr, ax=ax_wr)
        fig_wr.tight_layout()

        #  Net points heatmap
        net_pts = [float(results[opp].get("net_points", 0.0)) for opp in opponents]
        fig_pts, ax_pts = plt.subplots(figsize=(max(4.0, len(opponents) * 1.5), 3.0))
        im_pts = ax_pts.imshow([net_pts], cmap="coolwarm", aspect="auto")
        ax_pts.set_xticks(range(len(opponents)))
        ax_pts.set_xticklabels(opponents)
        ax_pts.set_yticks([0])
        ax_pts.set_yticklabels([candidate_name])
        ax_pts.set_title(f"Net Points vs Opponents ({candidate_name})")
        for i, val in enumerate(net_pts):
            ax_pts.text(i, 0, f"{int(val)}", ha="center", va="center", color="black")
        fig_pts.colorbar(im_pts, ax=ax_pts)
        fig_pts.tight_layout()

        return {"win_rate": fig_wr, "net_points": fig_pts}

    def save_figures(
        self,
        figures: dict[str, Any] | list[Any],
        output_dir: str | Path,
    ) -> list[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        if isinstance(figures, dict):
            for name, fig in figures.items():
                filename = name if name.endswith(".png") else f"{name}.png"
                target = out / filename
                if hasattr(fig, "savefig"):
                    fig.savefig(target)
                else:
                    target.write_bytes(b"")
                saved.append(target)
        elif isinstance(figures, list):
            for i, fig in enumerate(figures):
                target = out / f"figure_{i}.png"
                if hasattr(fig, "savefig"):
                    fig.savefig(target)
                else:
                    target.write_bytes(b"")
                saved.append(target)

        return saved

    def log_showdown(
        self,
        results: dict[str, dict[str, Any]],
        run_name: str,
        config: Any = None,
    ) -> None:

        run_ctx = None
        if (
            self._mlflow is not None
            and not self.mock
            and not self.dry_run
            and self._active_run is None
        ):
            try:
                run_ctx = self._mlflow.start_run(run_name=run_name)
            except Exception:  # noqa: BLE001
                run_ctx = None

        try:
            if config is not None:
                if hasattr(config, "to_dict"):
                    self.log_params(config.to_dict())
                elif isinstance(config, dict):
                    self.log_params(config)

            metrics: dict[str, float] = {}
            for opp, opp_data in results.items():
                if isinstance(opp_data, dict):
                    for k, v in opp_data.items():
                        if isinstance(v, (int, float)):
                            metrics[f"{opp}/{k}"] = float(v)
            if metrics:
                self.log_metrics(metrics)

            cand_name = (
                getattr(config, "candidate", "candidate") if config is not None else "candidate"
            )
            heatmaps = self.generate_heatmaps(results, candidate_name=cand_name)
            for name, fig in heatmaps.items():
                filename = name if name.endswith(".png") else f"{name}.png"
                self.log_figure(fig, filename)
        finally:
            if run_ctx is not None and self._mlflow is not None:
                try:
                    self._mlflow.end_run()
                except Exception:  # noqa: BLE001, S110
                    pass


def generate_heatmaps(
    results: dict[str, dict[str, Any]], candidate_name: str = "candidate"
) -> dict[str, Any]:

    return MLflowTracker().generate_heatmaps(results, candidate_name=candidate_name)


def save_figures(figures: dict[str, Any] | list[Any], output_dir: str | Path) -> list[Path]:

    return MLflowTracker().save_figures(figures, output_dir)
