import importlib.util

spec = importlib.util.spec_from_file_location(
      "find_pkgs", "deep_ep/utils/find_pkgs.py"
  )
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print("NCCL include:  ", module.find_nccl_include_dir())
print("NCCL library:  ", module.find_nccl_lib_dir())
print("NVSHMEM include:", module.find_nvshmem_include_dir())
print("NVSHMEM library:", module.find_nvshmem_lib_dir())


import importlib.util
import deep_ep
import torch

print("DeepEP:", deep_ep.__version__)
print("DeepEP source:", deep_ep.__file__)
print("Extension:", importlib.util.find_spec("deep_ep._C").origin)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

import importlib.util; 
print(importlib.util.find_spec("deep_ep._C").origin)