"""Collect software, hardware, and source-control metadata for IPV runs."""

import datetime as dt
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from . import __version__ as FRAMEWORK_VERSION


DEPENDENCY_DISTRIBUTIONS = (
    'numpy', 'torch', 'torchvision', 'pandas', 'matplotlib', 'opencv-python',
    'opencv-python-headless', 'scikit-image', 'scipy', 'numba', 'openpyxl',
)


def utc_now_iso():
    """Return a timezone-aware UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def collect_runtime_metadata(device, use_amp):
    """Return serialisable software, compute-device, and source-control details."""
    device = torch.device(device)
    cuda_available = bool(torch.cuda.is_available())
    selected_index = None
    selected_device = None

    if device.type == 'cuda' and cuda_available:
        selected_index = int(torch.cuda.current_device() if device.index is None else device.index)
        properties = torch.cuda.get_device_properties(selected_index)
        selected_device = {
            'index': selected_index,
            'name': properties.name,
            'compute_capability': [int(properties.major), int(properties.minor)],
            'total_memory_bytes': int(properties.total_memory),
            'multi_processor_count': int(properties.multi_processor_count),
        }

    return {
        'captured_at': utc_now_iso(),
        'framework': {'name': 'landmark-identification', 'version': FRAMEWORK_VERSION},
        'python': {'version': platform.python_version(), 'implementation': platform.python_implementation(), 'executable': sys.executable},
        'platform': {'system': platform.system(), 'release': platform.release(), 'version': platform.version(),
                     'machine': platform.machine(), 'processor': platform.processor()},
        'dependencies': {name: get_distribution_version(name) for name in DEPENDENCY_DISTRIBUTIONS},
        'pytorch': {
            'version': torch.__version__,
            'deterministic_algorithms_enabled': bool(torch.are_deterministic_algorithms_enabled()),
            'deterministic_algorithms_warn_only': bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        },
        'compute': {'resolved_device': str(device), 'device_type': device.type, 'device_index': selected_index,
                    'amp_enabled': bool(use_amp and device.type == 'cuda')},
        'cuda': {'available': cuda_available, 'pytorch_cuda_build_version': torch.version.cuda,
                 'device_count': int(torch.cuda.device_count()) if cuda_available else 0,
                 'selected_device': selected_device,
                 'nvidia_driver_version': get_nvidia_driver_version(selected_index) if selected_index is not None else None},
        'cudnn': {'available': bool(torch.backends.cudnn.is_available()), 'version': torch.backends.cudnn.version(),
                  'enabled': bool(torch.backends.cudnn.enabled), 'deterministic': bool(torch.backends.cudnn.deterministic),
                  'benchmark': bool(torch.backends.cudnn.benchmark), 'allow_tf32': bool(torch.backends.cudnn.allow_tf32)},
        'source_control': collect_git_metadata(Path(__file__).resolve()),
    }


def get_distribution_version(distribution):
    """Return an installed distribution version, or ``None`` when unavailable."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_nvidia_driver_version(device_index):
    """Return the selected device's NVIDIA driver version."""
    result = run_command(['nvidia-smi', f'--id={int(device_index)}', '--query-gpu=driver_version', '--format=csv,noheader'])
    return result.splitlines()[0].strip() if result else None


def collect_git_metadata(start_path):
    """Return the current Git revision and dirty state."""
    repository_root = find_repository_root(start_path)

    if repository_root is None:
        return {'repository_root': None, 'commit': None, 'branch': None, 'describe': None, 'is_dirty': None}

    prefix = ['git', '-c', f'safe.directory={repository_root.as_posix()}', '-C', str(repository_root)]
    status = run_command([*prefix, 'status', '--porcelain'])
    return {
        'repository_root': str(repository_root),
        'commit': run_command([*prefix, 'rev-parse', 'HEAD']),
        'branch': run_command([*prefix, 'rev-parse', '--abbrev-ref', 'HEAD']),
        'describe': run_command([*prefix, 'describe', '--always', '--dirty', '--tags']),
        'is_dirty': None if status is None else bool(status),
    }


def find_repository_root(start_path):
    """Find the nearest parent containing a Git worktree marker."""
    candidate = Path(start_path)
    candidate = candidate if candidate.is_dir() else candidate.parent

    for directory in (candidate, *candidate.parents):
        if (directory / '.git').exists():
            return directory

    return None


def run_command(arguments):
    """Run a short metadata command and return stripped output on success."""
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=5)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None

    return completed.stdout.strip()
