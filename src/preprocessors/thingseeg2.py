import gc
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

StrPath = Union[str, os.PathLike[str]]

console = Console()

CHAN_ORDER = [
    "Fp1",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
]


def create_progress() -> Progress:
    """Create a progress bar that avoids Unicode-only separators."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("-"),
        TaskProgressColumn(),
        TextColumn("-"),
        TimeElapsedColumn(),
        TextColumn("-"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


def epoching(
    subject_id: int,
    num_sessions: int,
    dsfreq: float,
    tmin: float,
    tmax: float,
    data_dir: StrPath,
    data_part: Literal["training", "test"],
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], np.ndarray]:
    """Epoch THINGS-EEG2 raw EEG sessions and sort trials by image condition."""
    import mne
    from sklearn.utils import shuffle

    epoched_data = []
    img_conditions = []

    subject_label = format(subject_id, "02d")
    console.print(
        Panel.fit(
            f"[bold cyan]Epoching {data_part.upper()} Data[/bold cyan]\n"
            f"Subject: [yellow]{subject_label}[/yellow] | "
            f"Sessions: [yellow]{num_sessions}[/yellow] | "
            f"Downsample: [yellow]{dsfreq}Hz[/yellow]",
            border_style="cyan",
        )
    )

    ch_names: list[str] = []
    times = np.array([], dtype=np.float64)

    with create_progress() as progress:
        session_task = progress.add_task("[cyan]Processing sessions", total=num_sessions)

        for session_idx in range(num_sessions):
            progress.update(
                session_task,
                description=f"[cyan]Session {session_idx + 1}/{num_sessions}",
            )

            session_label = format(session_idx + 1, "02d")
            eeg_path = Path(data_dir) / (
                f"sub-{subject_label}/ses-{session_label}/raw_eeg_{data_part}.npy"
            )
            eeg_file = np.load(eeg_path, allow_pickle=True).item()
            raw_eeg_data = eeg_file["raw_eeg_data"].astype(np.float32, copy=False)

            info = mne.create_info(eeg_file["ch_names"], eeg_file["sfreq"], eeg_file["ch_types"])
            raw = mne.io.RawArray(raw_eeg_data, info)
            del raw_eeg_data, eeg_file

            events = mne.find_events(raw, stim_channel="stim")
            raw.pick(picks=CHAN_ORDER)
            events = np.delete(events, np.where(events[:, 2] == 99999)[0], axis=0)

            epochs = mne.Epochs(
                raw,
                events,
                tmin=tmin,
                tmax=tmax,
                baseline=(None, 0),
                preload=True,
            )
            del raw

            sfreq = float(epochs.info["sfreq"])
            if dsfreq > sfreq:
                raise ValueError(
                    f"Downsampling frequency {dsfreq}Hz cannot exceed source {sfreq}Hz."
                )
            if dsfreq < sfreq:
                epochs.resample(dsfreq)

            ch_names = list(epochs.info["ch_names"])
            times = epochs.times
            data = epochs.get_data(copy=False).astype(np.float32, copy=False) * 1e6
            events = epochs.events[:, 2]
            img_cond = np.unique(events)
            del epochs

            max_rep = 20 if data_part == "test" else 2
            sorted_data = np.empty(
                (len(img_cond), max_rep, data.shape[1], data.shape[2]), dtype=data.dtype
            )
            for cond_idx, cond in enumerate(img_cond):
                trial_idx = np.where(events == cond)[0]
                trial_idx = shuffle(trial_idx, random_state=seed, n_samples=max_rep)
                sorted_data[cond_idx] = data[trial_idx]
            del data

            time_indices = _target_time_indices(times=times, dsfreq=dsfreq)
            sorted_data = sorted_data[:, :, :, time_indices]
            epoched_data.append(sorted_data)
            img_conditions.append(img_cond)
            times = times[time_indices]

            progress.advance(session_task)

    console.print(
        f"  [green]OK[/green] {data_part.capitalize()} epoching complete: "
        f"[dim]{len(epoched_data)} sessions, shape {epoched_data[0].shape}[/dim]"
    )
    return epoched_data, img_conditions, ch_names, times


def _target_time_indices(times: np.ndarray, dsfreq: float) -> np.ndarray:
    """Return indices for the one-second post-stimulus window [0, 1)."""
    n_timepoints = int(dsfreq)
    tolerance = 0.5 / float(dsfreq)
    candidate_indices = np.flatnonzero((times >= -tolerance) & (times < 1.0 - tolerance))
    if candidate_indices.size < n_timepoints:
        raise ValueError(
            "Epoch does not contain enough post-stimulus samples. "
            f"Expected {n_timepoints}, found {candidate_indices.size}."
        )
    return candidate_indices[:n_timepoints]


def mvnn(
    num_sessions: int,
    mvnn_dim: Literal["time", "epoch"],
    epoched_test: list[np.ndarray],
    epoched_train: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Apply multivariate noise normalization independently per session."""
    import scipy
    from sklearn.discriminant_analysis import _cov

    whitened_test = []
    whitened_train = []

    console.print(
        Panel.fit(
            f"[bold magenta]MVNN Whitening[/bold magenta]\n"
            f"Dimension: [yellow]{mvnn_dim}[/yellow] | "
            f"Sessions: [yellow]{num_sessions}[/yellow]\n"
            "[dim]Using training data covariance only[/dim]",
            border_style="magenta",
        )
    )

    with create_progress() as progress:
        session_task = progress.add_task("[magenta]Sessions", total=num_sessions)

        for session_idx in range(num_sessions):
            progress.update(
                session_task,
                description=f"[magenta]Session {session_idx + 1}/{num_sessions}",
            )
            session_data = [epoched_test[session_idx], epoched_train[session_idx]]
            sigma_part = np.empty(
                (len(session_data), session_data[0].shape[2], session_data[0].shape[2])
            )

            for part_idx, part_name in enumerate(["test", "train"]):
                sigma_cond = np.empty(
                    (
                        session_data[part_idx].shape[0],
                        session_data[0].shape[2],
                        session_data[0].shape[2],
                    )
                )
                cond_task = progress.add_task(
                    f"[dim]  {part_name} covariance",
                    total=session_data[part_idx].shape[0],
                )

                for cond_idx in range(session_data[part_idx].shape[0]):
                    cond_data = session_data[part_idx][cond_idx]
                    if mvnn_dim == "time":
                        sigma_cond[cond_idx] = np.mean(
                            [
                                _cov(cond_data[:, :, t], shrinkage="auto")
                                for t in range(cond_data.shape[2])
                            ],
                            axis=0,
                        )
                    elif mvnn_dim == "epoch":
                        sigma_cond[cond_idx] = np.mean(
                            [
                                _cov(np.transpose(cond_data[e]), shrinkage="auto")
                                for e in range(cond_data.shape[0])
                            ],
                            axis=0,
                        )
                    else:
                        raise ValueError(f"Unsupported mvnn_dim: {mvnn_dim}")
                    progress.advance(cond_task)

                sigma_part[part_idx] = sigma_cond.mean(axis=0)
                progress.remove_task(cond_task)

            sigma_inv = scipy.linalg.fractional_matrix_power(sigma_part[1], -0.5)
            whitened_test.append(_apply_whitening(session_data[0], sigma_inv))
            whitened_train.append(_apply_whitening(session_data[1], sigma_inv))
            epoched_test[session_idx] = None
            epoched_train[session_idx] = None
            del session_data
            gc.collect()
            progress.advance(session_task)

    console.print("  [green]OK[/green] MVNN whitening complete")
    return whitened_test, whitened_train


def _apply_whitening(data: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    """Apply a channel whitening matrix while preserving the input EEG shape."""
    whitened = (
        np.reshape(data, (-1, data.shape[2], data.shape[3])).swapaxes(1, 2) @ sigma_inv
    ).swapaxes(1, 2)
    return np.reshape(whitened, data.shape).astype(np.float32, copy=False)


def zscore(
    epoched_test: list[np.ndarray],
    epoched_train: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Apply per-session channel z-score using training data statistics."""
    normalized_test = []
    normalized_train = []

    console.print(
        Panel.fit(
            "[bold magenta]Z-score Normalization[/bold magenta]\n"
            f"Sessions: [yellow]{len(epoched_train)}[/yellow]\n"
            "[dim]Using training data mean and std only[/dim]",
            border_style="magenta",
        )
    )

    for session_idx, train_data in enumerate(epoched_train):
        mean = train_data.mean(axis=(0, 1, 3), keepdims=True)
        std = train_data.std(axis=(0, 1, 3), keepdims=True)
        std = np.where(std == 0, 1.0, std)

        normalized_test.append(
            ((epoched_test[session_idx] - mean) / std).astype(np.float32, copy=False)
        )
        normalized_train.append(((train_data - mean) / std).astype(np.float32, copy=False))

    console.print("  [green]OK[/green] Z-score normalization complete")
    return normalized_test, normalized_train


def save_prepr(
    subject_id: int,
    num_sessions: int,
    save_dir: StrPath,
    img_data_dir: StrPath,
    whitened_test: list[np.ndarray] | None,
    whitened_train: list[np.ndarray] | None,
    img_conditions_train: list[np.ndarray] | None,
    ch_names: list[str],
    times: np.ndarray,
) -> dict[str, str]:
    """Merge THINGS-EEG2 sessions and save preprocessed PyTorch dictionaries."""
    subject_label = format(subject_id, "02d")
    console.print(
        Panel.fit(
            f"[bold green]Saving Preprocessed Data[/bold green]\n"
            f"Subject: [yellow]{subject_label}[/yellow] | "
            f"Sessions: [yellow]{num_sessions}[/yellow]\n"
            f"Output: [dim]{save_dir}[/dim]",
            border_style="green",
        )
    )

    subject_save_dir = Path(save_dir) / f"sub-{subject_label}"
    subject_save_dir.mkdir(parents=True, exist_ok=True)

    if whitened_test is None and (whitened_train is None or img_conditions_train is None):
        raise ValueError("At least one of test or training data must be provided.")

    outputs = {}
    if whitened_test is not None:
        outputs["test"] = _save_test_partition(
            subject_save_dir=subject_save_dir,
            img_data_dir=Path(img_data_dir),
            num_sessions=num_sessions,
            whitened_test=whitened_test,
            ch_names=ch_names,
            times=times,
        )

    if whitened_train is not None and img_conditions_train is not None:
        outputs["training"] = _save_training_partition(
            subject_save_dir=subject_save_dir,
            img_data_dir=Path(img_data_dir),
            num_sessions=num_sessions,
            whitened_train=whitened_train,
            img_conditions_train=img_conditions_train,
            ch_names=ch_names,
            times=times,
        )

    table = Table(title="Output Files", box=box.ROUNDED, border_style="green")
    table.add_column("Partition", style="cyan")
    table.add_column("Path", style="dim")
    for partition, path in outputs.items():
        table.add_row(partition, path)
    console.print(table)
    return outputs


def _save_test_partition(
    subject_save_dir: Path,
    img_data_dir: Path,
    num_sessions: int,
    whitened_test: list[np.ndarray],
    ch_names: list[str],
    times: np.ndarray,
) -> str:
    """Merge test sessions into one memmapped tensor and save its metadata."""
    n_test_images = whitened_test[0].shape[0]
    n_rep_per_session = whitened_test[0].shape[1]
    n_total_reps = n_rep_per_session * num_sessions
    session_list = np.empty((n_test_images, n_total_reps), dtype=np.int16)

    tmp_path = subject_save_dir / "_tmp_test_eeg.npy"
    merged_test = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_test_images, n_total_reps, whitened_test[0].shape[2], whitened_test[0].shape[3]),
    )

    try:
        for session_idx in range(num_sessions):
            start_idx = session_idx * n_rep_per_session
            end_idx = (session_idx + 1) * n_rep_per_session
            merged_test[:, start_idx:end_idx] = whitened_test[session_idx].astype(
                np.float32, copy=False
            )
            session_list[:, start_idx:end_idx] = session_idx
            whitened_test[session_idx] = None

        gc.collect()

        imgs, labels, texts = _load_image_metadata(img_data_dir / "test_images")
        _validate_metadata_lengths(
            partition="test",
            expected_rows=n_test_images,
            imgs=imgs,
            labels=labels,
            texts=texts,
        )
        output_path = subject_save_dir / "test.pt"
        merged_shape = merged_test.shape
        torch.save(
            {
                "eeg": merged_test,
                "label": np.tile(np.array(labels)[:, np.newaxis], (1, n_total_reps)),
                "img": np.tile(np.array(imgs)[:, np.newaxis], (1, n_total_reps)),
                "text": np.tile(np.array(texts)[:, np.newaxis], (1, n_total_reps)),
                "session": session_list,
                "ch_names": ch_names,
                "times": times,
            },
            output_path,
            pickle_protocol=5,
        )
        console.print(f"  [green]OK[/green] Saved test data: [dim]{merged_shape}[/dim]")
    finally:
        del merged_test
        tmp_path.unlink(missing_ok=True)
    return str(output_path)


def _save_training_partition(
    subject_save_dir: Path,
    img_data_dir: Path,
    num_sessions: int,
    whitened_train: list[np.ndarray],
    img_conditions_train: list[np.ndarray],
    ch_names: list[str],
    times: np.ndarray,
) -> str:
    """Merge training sessions by image condition and save their metadata."""
    n_rep_per_train = whitened_train[0].shape[1]
    unique_conditions = np.unique(np.concatenate(img_conditions_train, axis=0))
    condition_to_idx = {int(cond): i for i, cond in enumerate(unique_conditions.tolist())}
    condition_counts = {
        int(cond): int(np.count_nonzero(np.concatenate(img_conditions_train, axis=0) == cond))
        for cond in unique_conditions
    }
    observed_condition_counts = set(condition_counts.values())
    if len(observed_condition_counts) != 1:
        raise RuntimeError(
            "Training conditions are not repeated uniformly across sessions. "
            f"Observed session counts: {sorted(observed_condition_counts)}."
        )
    n_total_reps = n_rep_per_train * observed_condition_counts.pop()
    session_list = np.empty((len(unique_conditions), n_total_reps), dtype=np.int16)
    rep_offsets = np.zeros(len(unique_conditions), dtype=np.int16)

    tmp_path = subject_save_dir / "_tmp_train_eeg.npy"
    merged_train = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(unique_conditions),
            n_total_reps,
            whitened_train[0].shape[2],
            whitened_train[0].shape[3],
        ),
    )

    try:
        for session_idx in range(num_sessions):
            for row_idx, cond in enumerate(img_conditions_train[session_idx]):
                cond_idx = condition_to_idx[int(cond)]
                start = int(rep_offsets[cond_idx])
                end = start + n_rep_per_train
                merged_train[cond_idx, start:end] = whitened_train[session_idx][row_idx].astype(
                    np.float32, copy=False
                )
                session_list[cond_idx, start:end] = session_idx
                rep_offsets[cond_idx] = end

            whitened_train[session_idx] = None
            img_conditions_train[session_idx] = None

        gc.collect()

        if not np.all(rep_offsets == n_total_reps):
            raise RuntimeError(
                "Unexpected training repetition count per condition. "
                f"Expected {n_total_reps}, got min={rep_offsets.min()}, max={rep_offsets.max()}."
            )

        imgs, labels, texts = _load_image_metadata(img_data_dir / "training_images")
        _validate_metadata_lengths(
            partition="training",
            expected_rows=len(unique_conditions),
            imgs=imgs,
            labels=labels,
            texts=texts,
        )
        output_path = subject_save_dir / "training.pt"
        merged_shape = merged_train.shape
        torch.save(
            {
                "eeg": merged_train,
                "label": np.tile(np.array(labels)[:, np.newaxis], (1, n_total_reps)),
                "img": np.tile(np.array(imgs)[:, np.newaxis], (1, n_total_reps)),
                "text": np.tile(np.array(texts)[:, np.newaxis], (1, n_total_reps)),
                "session": session_list,
                "ch_names": ch_names,
                "times": times,
            },
            output_path,
            pickle_protocol=5,
        )
        console.print(f"  [green]OK[/green] Saved train data: [dim]{merged_shape}[/dim]")
    finally:
        del merged_train
        tmp_path.unlink(missing_ok=True)
    return str(output_path)


def _validate_metadata_lengths(
    partition: str,
    expected_rows: int,
    imgs: Sequence[str],
    labels: Sequence[int],
    texts: Sequence[str],
) -> None:
    """Ensure image metadata aligns one-to-one with EEG condition rows."""
    metadata_lengths = {
        "images": len(imgs),
        "labels": len(labels),
        "texts": len(texts),
    }
    if len(set(metadata_lengths.values())) != 1 or len(imgs) != expected_rows:
        raise ValueError(
            f"{partition} metadata does not match EEG rows. "
            f"Expected {expected_rows}, got {metadata_lengths}."
        )


def _load_image_metadata(image_dir: Path) -> tuple[list[str], list[int], list[str]]:
    """Load image paths, integer labels, and text labels from class folders."""
    imgs = []
    labels = []
    texts = []
    class_dirs = sorted(path for path in image_dir.iterdir() if path.is_dir())

    with create_progress() as progress:
        task = progress.add_task(
            f"[green]Loading metadata from {image_dir.name}", total=len(class_dirs)
        )
        for class_idx, class_dir in enumerate(class_dirs):
            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            imgs.extend(str(path) for path in image_paths)
            labels.extend([class_idx] * len(image_paths))
            texts.extend(" ".join(path.stem.split("_")[:-1]) for path in image_paths)
            progress.advance(task)

    return imgs, labels, texts


@dataclass
class Thingseeg2Preprocessor:
    """High-level preprocessing runner for the THINGS-EEG2 EEG dataset."""

    subject_id: int
    num_sessions: int
    sfreq: int
    tmin: float
    tmax: float
    mvnn_dim: Literal["time", "epoch"] | None
    raw_data_dir: StrPath
    img_data_dir: StrPath
    save_dir: StrPath
    seed: int

    @property
    def subject_label(self) -> str:
        """Return the zero-padded subject id used by THINGS-EEG2 paths."""
        return format(self.subject_id, "02d")

    def run(self) -> dict[str, str | int]:
        """Run epoching, optional MVNN, and saving for one subject."""
        console.print()
        console.print(
            Panel.fit(
                "[bold white]THINGS-EEG2 Preprocessing Pipeline[/bold white]\n\n"
                "[dim]Convert raw EEG data to model-ready PyTorch files[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        self._print_config()

        identity_mode = self.sfreq >= 1000 or self.mvnn_dim is None
        outputs: dict[str, str] = {}

        if identity_mode:
            outputs.update(self._run_partitioned_without_mvnn())
        else:
            outputs.update(self._run_with_mvnn(self.mvnn_dim))

        console.print()
        console.print(
            Panel.fit(
                "[bold green]Preprocessing Complete[/bold green]\n\n"
                f"[dim]Subject {self.subject_label} data saved to {self.save_dir}[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )
        return {"subject_id": self.subject_id, "output_dir": str(self.save_dir), **outputs}

    def _run_partitioned_without_mvnn(self) -> dict[str, str]:
        """Run z-score normalization when MVNN whitening is disabled."""
        epoched_test, _, ch_names, times = epoching(
            subject_id=self.subject_id,
            num_sessions=self.num_sessions,
            dsfreq=self.sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
            data_dir=self.raw_data_dir,
            data_part="test",
            seed=self.seed,
        )
        epoched_train, img_conditions_train, _, _ = epoching(
            subject_id=self.subject_id,
            num_sessions=self.num_sessions,
            dsfreq=self.sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
            data_dir=self.raw_data_dir,
            data_part="training",
            seed=self.seed,
        )

        normalized_test, normalized_train = zscore(
            epoched_test=epoched_test,
            epoched_train=epoched_train,
        )
        del epoched_test, epoched_train
        gc.collect()

        outputs = save_prepr(
            subject_id=self.subject_id,
            num_sessions=self.num_sessions,
            save_dir=self.save_dir,
            img_data_dir=self.img_data_dir,
            whitened_test=normalized_test,
            whitened_train=normalized_train,
            img_conditions_train=img_conditions_train,
            ch_names=ch_names,
            times=times,
        )
        del normalized_test, normalized_train, img_conditions_train
        gc.collect()
        return outputs

    def _run_with_mvnn(self, mvnn_dim: Literal["time", "epoch"]) -> dict[str, str]:
        """Run full preprocessing with MVNN whitening before saving."""
        epoched_test, _, ch_names, times = epoching(
            subject_id=self.subject_id,
            num_sessions=self.num_sessions,
            dsfreq=self.sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
            data_dir=self.raw_data_dir,
            data_part="test",
            seed=self.seed,
        )
        epoched_train, img_conditions_train, _, _ = epoching(
            subject_id=self.subject_id,
            num_sessions=self.num_sessions,
            dsfreq=self.sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
            data_dir=self.raw_data_dir,
            data_part="training",
            seed=self.seed,
        )
        whitened_test, whitened_train = mvnn(
            num_sessions=self.num_sessions,
            mvnn_dim=mvnn_dim,
            epoched_test=epoched_test,
            epoched_train=epoched_train,
        )
        del epoched_test, epoched_train
        gc.collect()

        return save_prepr(
            subject_id=self.subject_id,
            num_sessions=self.num_sessions,
            save_dir=self.save_dir,
            img_data_dir=self.img_data_dir,
            whitened_test=whitened_test,
            whitened_train=whitened_train,
            img_conditions_train=img_conditions_train,
            ch_names=ch_names,
            times=times,
        )

    def _print_config(self) -> None:
        """Print the resolved THINGS-EEG2 preprocessing configuration."""
        table = Table(title="Configuration", box=box.ROUNDED, border_style="blue")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow")
        table.add_row("Subject ID", self.subject_label)
        table.add_row("Sessions", str(self.num_sessions))
        table.add_row("Downsample Freq", f"{self.sfreq} Hz")
        table.add_row("Epoch Window", f"[{self.tmin}, {self.tmax}] sec")
        table.add_row("MVNN Dimension", str(self.mvnn_dim))
        table.add_row("Random Seed", str(self.seed))
        console.print(table)
        console.print()
