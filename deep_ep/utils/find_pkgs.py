import functools
import glob
import os
import sys
import sysconfig
from importlib.metadata import distributions
from typing import Optional


def find_pkg_root(name: str, lib_name: Optional[str] = None, optional: bool = False):
    """
    Find the root directory of an installed NVIDIA package by inspecting Python package metadata.
    Checks environment variables `EP_{NAME}_ROOT_DIR` and `{NAME}_DIR` first.

    Arguments:
        name: the package name (e.g., `'nccl'`, `'nvshmem'`).
        lib_name: the library filename to search for within the package files.
        optional: if ``False``, raises an assertion error when the package is not found.

    Returns:
        root: the package root directory, or `None` if not found and optional.
    """
    upper = name.upper()
    for env_name in (f'EP_{upper}_ROOT_DIR', f'{upper}_DIR'):
        if env_name in os.environ:
            return os.environ[env_name]

    path_priority = {p: i for i, p in enumerate(sys.path)}
    best, best_priority = None, len(sys.path)

    for dist in distributions():
        dist_name = (dist.metadata['Name'] or '').lower()
        if f'nvidia-{name}' not in dist_name and f'nvidia_{name}' not in dist_name:
            continue

        dist_site = str(dist._path.parent)
        priority = path_priority.get(dist_site, len(sys.path))
        if priority > best_priority:
            continue

        if lib_name is not None:
            for f in (dist.files or []):
                if lib_name in str(f):
                    lib_dir = os.path.dirname(str(f.locate()))
                    root = os.path.dirname(lib_dir) if os.path.basename(lib_dir) == 'lib' else lib_dir
                    best, best_priority = root, priority
                    break
        else:
            pkg_dir = os.path.join(dist_site, 'nvidia', name)
            if os.path.isdir(pkg_dir):
                best, best_priority = pkg_dir, priority

    # Raise error if not optional
    if not optional:
        assert best is not None, f'Cannot find package: {name}'
    return best


@functools.lru_cache()
def find_nccl_root(optional: bool = False):
    """
    Find the NCCL installation root directory, cached.

    Arguments:
        optional: if `False`, raises an assertion error when NCCL is not found.

    Returns:
        root: the NCCL root directory.
    """
    return find_pkg_root('nccl', lib_name='libnccl.so', optional=optional)


@functools.lru_cache()
def find_nvshmem_root(optional: bool = False):
    """
    Find the NVSHMEM installation root directory, cached.

    Arguments:
        optional: if `False`, raises an assertion error when NVSHMEM is not found.

    Returns:
        root: the NVSHMEM root directory.
    """
    return find_pkg_root('nvshmem', optional=optional)


def _find_existing_dir(candidates, required_file):
    for directory in candidates:
        if directory and os.path.isfile(os.path.join(directory, required_file)):
            return os.path.realpath(directory)
    return None


def _include_candidates(root, system_candidates):
    candidates = []
    if root is not None:
        candidates.extend((os.path.join(root, 'include'), root))
    candidates.extend(system_candidates)
    return candidates


def _lib_candidates(root, system_candidates):
    candidates = []
    if root is not None:
        candidates.extend((os.path.join(root, 'lib'), os.path.join(root, 'lib64')))
        multiarch = sysconfig.get_config_var('MULTIARCH')
        if multiarch:
            candidates.append(os.path.join(root, 'lib', multiarch))
    candidates.extend(system_candidates)
    return candidates


@functools.lru_cache()
def find_nccl_include_dir(optional: bool = False):
    root = find_nccl_root(optional=True)
    include_dir = _find_existing_dir(
        _include_candidates(root, ('/usr/include', '/usr/local/include')),
        'nccl.h',
    )
    if not optional:
        assert include_dir is not None, 'Cannot find NCCL headers (nccl.h)'
    return include_dir


@functools.lru_cache()
def find_nccl_lib_dir(optional: bool = False):
    root = find_nccl_root(optional=True)
    multiarch = sysconfig.get_config_var('MULTIARCH')
    system_candidates = [
        '/usr/local/lib',
        '/usr/local/lib64',
        '/usr/lib',
        '/usr/lib64',
    ]
    if multiarch:
        system_candidates.insert(0, os.path.join('/usr/lib', multiarch))
        system_candidates.insert(1, os.path.join('/usr/local/lib', multiarch))
    lib_dir = _find_existing_dir(_lib_candidates(root, system_candidates), 'libnccl.so')
    if not optional:
        assert lib_dir is not None, 'Cannot find NCCL library (libnccl.so)'
    return lib_dir


@functools.lru_cache()
def find_nvshmem_include_dir(optional: bool = False):
    root = find_nvshmem_root(optional=True)
    system_candidates = [
        '/usr/local/cuda/include',
        '/usr/include',
        *sorted(glob.glob('/usr/include/nvshmem_*'), reverse=True),
    ]
    include_dir = _find_existing_dir(_include_candidates(root, system_candidates), 'nvshmem.h')
    if not optional:
        assert include_dir is not None, 'Cannot find NVSHMEM headers (nvshmem.h)'
    return include_dir


@functools.lru_cache()
def find_nvshmem_lib_dir(optional: bool = False):
    root = find_nvshmem_root(optional=True)
    multiarch = sysconfig.get_config_var('MULTIARCH')
    system_candidates = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        '/usr/local/lib',
        '/usr/local/lib64',
        '/usr/lib',
        '/usr/lib64',
    ]
    if multiarch:
        system_candidates = [
            *sorted(glob.glob(os.path.join('/usr/lib', multiarch, 'nvshmem', '*')), reverse=True),
            os.path.join('/usr/lib', multiarch),
            *system_candidates,
        ]
    lib_dir = _find_existing_dir(_lib_candidates(root, system_candidates), 'libnvshmem_device.a')
    if not optional:
        assert lib_dir is not None, 'Cannot find NVSHMEM device library (libnvshmem_device.a)'
    return lib_dir
