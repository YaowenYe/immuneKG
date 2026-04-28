"""
immuneKG utility functions

Provides config loading, logging, timer, and GPU detection.
"""

import os
import json
import time
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_dir / f'{name}_{timestamp}.log', encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger


class Timer:
    """
    Stage timer with cumulative tracking and JSON export.

    Usage:
        timer = Timer()
        timer.start("data_loading")
        timer.stop("data_loading")
        timer.save("results/timing.json")
    """

    def __init__(self):
        self.records = {}
        self._start_times = {}

    def start(self, stage_name: str):
        self._start_times[stage_name] = time.time()

    def stop(self, stage_name: str) -> float:
        if stage_name not in self._start_times:
            raise ValueError(f"Stage '{stage_name}' was not started")
        duration = time.time() - self._start_times.pop(stage_name)
        self.records[stage_name] = duration
        return duration

    def format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.2f}min"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}min"

    def save(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        output = {
            stage: {"seconds": round(s, 2), "formatted": self.format_duration(s)}
            for stage, s in self.records.items()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        lines = [f"{'Stage':<30} {'Duration':>15}"]
        lines.append("-" * 47)
        for stage, seconds in self.records.items():
            lines.append(f"{stage:<30} {self.format_duration(seconds):>15}")
        if self.records:
            total = sum(self.records.values())
            lines.append("-" * 47)
            lines.append(f"{'Total':<30} {self.format_duration(total):>15}")
        return "\n".join(lines)


def setup_device(config: dict) -> torch.device:
    """Set up compute device (GPU/CPU) based on config."""
    gpu_config = config.get('gpu', {})
    device_id = gpu_config.get('device_id', '0')
    auto_select = gpu_config.get('auto_select', False)

    if not torch.cuda.is_available():
        print(f"{Fore.YELLOW}[WARNING] CUDA not available, using CPU{Style.RESET_ALL}")
        return torch.device('cpu')

    if not auto_select:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"{Fore.GREEN}[GPU] {gpu_name} | {gpu_mem:.2f} GB | cuda:{device_id} | CUDA {torch.version.cuda}{Style.RESET_ALL}")
    return device


def print_banner(title: str, width: int = 68):
    border = "=" * width
    padding = (width - len(title) - 4) // 2
    print(f"\n{Fore.CYAN}+{border}+")
    print(f"|{' ' * padding}  {title}  {' ' * (width - padding - len(title) - 4)}|")
    print(f"+{border}+{Style.RESET_ALL}")


def print_stage(stage_num: int, total_stages: int, description: str):
    print(f"\n{Fore.YELLOW}--- Stage [{stage_num}/{total_stages}] {description} ---{Style.RESET_ALL}")


def print_success(message: str):
    print(f"{Fore.GREEN}[OK] {message}{Style.RESET_ALL}")


def print_warning(message: str):
    print(f"{Fore.YELLOW}[WARN] {message}{Style.RESET_ALL}")


def print_error(message: str):
    print(f"{Fore.RED}[ERROR] {message}{Style.RESET_ALL}")


def print_info(message: str):
    print(f"{Fore.WHITE}  -> {message}{Style.RESET_ALL}")


def print_stat(label: str, value, indent: int = 2):
    prefix = " " * indent + "|-"
    print(f"{prefix} {label}: {Fore.CYAN}{value}{Style.RESET_ALL}")


def print_dict_stats(data: dict, title: str = "Statistics"):
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 10
    print(f"\n  {title}:")
    for i, (k, v) in enumerate(data.items()):
        connector = "|-" if i < len(data) - 1 else "+-"
        print(f"  {connector} {str(k):<{max_key_len}} : {Fore.CYAN}{v}{Style.RESET_ALL}")


def checkpoint_exists(filepath: str) -> bool:
    return Path(filepath).exists()


def save_checkpoint(data, filepath: str, description: str = ""):
    import pickle
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    if description:
        print_info(f"Checkpoint saved: {description} -> {filepath.name}")


def load_checkpoint(filepath: str, description: str = ""):
    import pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    if description:
        print_success(f"Checkpoint loaded: {description} <- {Path(filepath).name}")
    return data
