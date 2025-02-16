#! coding: utf-8

import hashlib
import os
import torch
import tarfile
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):
    assert name in DATA_HUB, f'{name} is not existed in {DATA_HUB}'
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'Downloading {fname} from {url} ...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)

    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files could be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    for name in DATA_HUB:
        download(name)


def all_cpu_devices() -> list[torch.device]:
    return [torch.device('cpu')]


def all_gpu_devices() -> list[torch.device]:
    device_list = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_list.append(torch.device(f'cuda:{i}'))
    return device_list


def all_mps_devices() -> list[torch.device]:
    device_list = []
    # As of now, the MPS backend supports only a single device. Therefore, there
    # should be only one MPS device.
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_list.append(torch.device('mps'))

    return device_list


def devices(device_name: str = None) -> list[torch.device]:
    # First priority, we should find all cuda devices,
    # Second priority, we should find all mps devices,
    # Finally, we fall back to cpu.

    device_func_map = {'cpu': all_cpu_devices, 'cuda': all_gpu_devices, 'mps': all_mps_devices}

    if device_name:
        # "device name must be in ['cpu', 'cuda', 'mps']."
        assert device_name in device_func_map.keys()
        return device_func_map[device_name]()

    # Find all cuda devices
    cuda_devices = device_func_map['cuda']()
    if cuda_devices:
        return cuda_devices

    # Find all mps devices
    mps_devices = device_func_map['mps']()
    if mps_devices:
        return mps_devices

    return device_func_map['cpu']()
