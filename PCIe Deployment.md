# DeepEP v2 — 纯 PCIe 部署指南

本指南介绍如何在**无 NVLink、无 RDMA 网卡**的机器上部署 DeepEP v2，仅使用 GPU 之间的 PCIe 互联。所有数据传输和 barrier 同步均通过 NCCL LSA（Local Shared Access）域在 PCIe 上完成，网络流量为零。

## 前置条件

### 硬件要求

- NVIDIA GPU（SM 9.0+ ，如 H100/H200/H800/B200 等）
- GPU 通过 PCIe 连接（不需要 NVLink）
- 不需要 RDMA 网卡

**拓扑要求**：同一个 EP 组内的 GPU 必须位于同一个 NCCL LSA 域内。在典型的 PCIe 机器上，同一 NUMA 节点内的 GPU 构成一个 LSA 域。检查拓扑：

```bash
nvidia-smi topo -m
```

- GPU 之间为 `PXB` / `NODE` / `PHB` 连接 → 同一 LSA 域，EP 可以正常工作
- `SYS` 连接（跨 NUMA）→ 可能不在同一 LSA 域内，跨这些 GPU 的 EP 可能失败

典型拓扑示例（8 卡，两个 NUMA 各 4 卡）：
```
        NUMA 0                          NUMA 1
GPU0  GPU1  GPU2  GPU3          GPU4  GPU5  GPU6  GPU7
 └─PXB──┘    └─PXB──┘           └─PXB──┘    └─PXB──┘
    └──NODE──┘                      └──NODE──┘
    └── LSA 域 0 ──┘                └── LSA 域 1 ──┘
              └──────── SYS（跨 NUMA）────────┘
```

在这种拓扑下，**EP=2 和 EP=4 可以正常运行**（同一 NUMA / LSA 域内），**EP=8 不可用**（跨 NUMA 的 SYS 连接不在同一 LSA 域内）。

### 软件要求

| 组件            | 已测试版本                    |
|----------------|------------------------------|
| 操作系统        | Ubuntu 22.04                 |
| NVIDIA 驱动     | ≥ 580.x（Open Kernel）       |
| CUDA Toolkit   | 13.0                         |
| Python         | 3.12                         |
| PyTorch        | 2.12.x（CUDA 13.0）          |
| NCCL           | 2.30.x（支持 LSA）            |

> **注意**：NCCL 必须支持 LSA（Local Shared Access）。NCCL ≥ 2.30 包含此功能。

## 环境搭建

### 1. 准备 Python 和编译工具

确保系统已安装 Python 3.12、`venv`、Python 开发头文件和基本编译工具。以下命令适用于软件源中已提供 Python 3.12 的 Debian/Ubuntu 环境：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.12 python3.12-venv python3.12-dev \
  build-essential ninja-build git
```

如果当前发行版的软件源没有 Python 3.12，请先通过该发行版支持的软件源或系统镜像安装 Python 3.12，然后确认：

```bash
python3.12 --version
```

### 2. 创建 Python venv

```bash
python3.12 -m venv "$HOME/venvs/deepep"
source "$HOME/venvs/deepep/bin/activate"

python -m pip install --upgrade pip setuptools wheel ninja
```

后续所有 `python` 和 `pip` 命令都应在这个 venv 激活后执行。重新登录或打开新终端后，使用以下命令重新激活：

```bash
source "$HOME/venvs/deepep/bin/activate"
```

### 3. 安装 PyTorch

安装与 CUDA 版本匹配的 PyTorch：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
```

验证安装：
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### 4. 安装 NCCL

需要 NCCL ≥ 2.30（支持 LSA）。PyTorch pip 包自带的 NCCL 版本可能低于 2.30，需要检查并升级。

检查当前 NCCL 版本：
```bash
python -c "
import nvidia.nccl, os
nccl_h = os.path.join(nvidia.nccl.__path__[0], 'include', 'nccl.h')
with open(nccl_h) as f:
    for line in f:
        if 'NCCL_VERSION_CODE' in line and '#define' in line:
            ver = int(line.split()[-1])
            print(f'NCCL version: {ver // 10000}.{(ver % 10000) // 100}.{ver % 100}')
            break
"
```

如果版本 < 2.30，需要从源码编译 NCCL 2.30+：
```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout v2.30.7-1  # 或更高版本
make -j$(nproc) src.build
# 编译产物在 build/ 目录下
```

然后**替换** PyTorch pip 包中的 NCCL 库和头文件（编译和运行时必须使用同一版本，否则 DeepEP 会报版本不匹配错误）：
```bash
NCCL_PKG=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")

# 备份原文件
cp $NCCL_PKG/lib/libnccl.so.2 $NCCL_PKG/lib/libnccl.so.2.bak

# 替换库文件
cp /path/to/nccl/build/lib/libnccl.so.2.* $NCCL_PKG/lib/libnccl.so.2

# 替换头文件
cp /path/to/nccl/build/include/nccl.h $NCCL_PKG/include/
cp /path/to/nccl/build/include/nccl_device.h $NCCL_PKG/include/
cp -r /path/to/nccl/build/include/nccl_device/* $NCCL_PKG/include/nccl_device/
```

或者，设置环境变量指向自编译的 NCCL（仅影响编译路径，运行时仍需替换 pip 包中的 .so）：
```bash
export EP_NCCL_ROOT_DIR=/path/to/nccl/build
```

### 5. NVSHMEM

NVSHMEM 头文件和库在编译时需要（用于 legacy 内核），纯 PCIe 模式运行时**不会使用** NVSHMEM。

PyTorch ≥ 2.12 的 pip 包可能自带 `nvidia-nvshmem-cu13`，DeepEP 的构建系统会自动检测 Python 包；也支持自动检测通过 apt 安装在系统多架构目录中的 NVSHMEM。

验证 NVSHMEM 已随 pip 安装：
```bash
python -c "import nvidia.nvshmem; print('NVSHMEM path:', nvidia.nvshmem.__path__[0])"
```

如果 PyTorch 版本较低未自带 NVSHMEM，需手动安装 NVSHMEM ≥ 2.11 并设置：
```bash
export EP_NVSHMEM_ROOT_DIR=/path/to/nvshmem
```

### 6. 安装其他依赖

测试脚本需要 numpy：
```bash
python -m pip install numpy
```

## 编译安装

```bash
git clone https://github.com/MengYu10151/DeepEP.git
cd DeepEP
git checkout pcie-no-atomic

python -m pip install --no-build-isolation -e .
```

> **注意**：必须加 `--no-build-isolation`，否则 pip 会在隔离环境中编译，找不到 PyTorch 导致报错 `No module named 'torch'`。

编译需要几分钟。JIT 编译的内核会在首次运行时按需编译并缓存到 `~/.deep_ep/cache/`。

> **更新代码后**：如果更新了 DeepEP 代码（如 `git pull`），建议清除 JIT 缓存以避免使用过期的编译产物：
> ```bash
> rm -rf ~/.deep_ep/cache/*
> ```

## 配置

纯 PCIe 模式需要设置两个环境变量：

| 变量                  | 值          | 说明                                              |
|-----------------------|-------------|---------------------------------------------------|
| `EP_DISABLE_GIN`      | `1`         | 禁用 RDMA GIN 后端（不需要网卡）                    |
| `NCCL_LSA_TEAM_SIZE`  | `<EP规模>`  | 设置 NCCL LSA 域大小，与 EP 规模匹配                |

可选调试变量：

| 变量               | 值    | 说明                              |
|--------------------|-------|-----------------------------------|
| `EP_BUFFER_DEBUG`  | `1`   | 打印初始化调试信息                  |

## 运行测试

### 正确性验证（EP=2）

```bash
EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=2 \
  python tests/elastic/test_ep.py \
    --num-processes 2 --hidden 4096 --num-topk 6 --num-experts 256 \
    --num-tokens 128 --num-sms 8 \
    --allow-hybrid-mode 0
```

测试使用 `torch.equal`（逐比特精确匹配）校验 dispatch 和 combine 结果，不是近似比较。如果正确性校验失败，程序会报错退出；正常运行结束（exit code 0）即表示正确性全部通过。

预期输出包含类似以下行（性能数据）：
```
   * EP:   0/2 | dispatch: 0 GB/s (SO), 56 GB/s (SU), ...
   @ EP:   0/2 | combine: 0 GB/s (SO), 61 GB/s (SU), ...
```

### 正确性验证（EP=4）

```bash
EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=4 \
  python tests/elastic/test_ep.py \
    --num-processes 4 --hidden 4096 --num-topk 6 --num-experts 256 \
    --num-tokens 128 --num-sms 8 \
    --allow-hybrid-mode 0
```

### 性能基准测试

```bash
for NP in 2 4; do
  for TOKENS in 16 32 64 128 512 1024 2048 4096; do
    echo ">>> EP=$NP TOKENS=$TOKENS"
    EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=$NP \
      python tests/elastic/test_ep.py \
        --num-processes $NP --hidden 4096 --num-topk 6 --num-experts 256 \
        --num-tokens $TOKENS --num-sms 8 \
        --allow-hybrid-mode 0 --test-first-only \
      | grep "EP:.*0/$NP"
  done
done
```

### 验证零网卡流量

确认所有数据走 PCIe、无 RDMA 流量：

```bash
# 测试前
NIC=<你的网卡名>  # 例如 enp115s0f0np0 或 eth0
RX_BEFORE=$(cat /sys/class/net/$NIC/statistics/rx_bytes)
TX_BEFORE=$(cat /sys/class/net/$NIC/statistics/tx_bytes)

# 运行测试
EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=2 \
  python tests/elastic/test_ep.py \
    --num-processes 2 --hidden 4096 --num-topk 6 --num-experts 256 \
    --num-tokens 128 --num-sms 8 \
    --allow-hybrid-mode 0

# 测试后
RX_AFTER=$(cat /sys/class/net/$NIC/statistics/rx_bytes)
TX_AFTER=$(cat /sys/class/net/$NIC/statistics/tx_bytes)
echo "网卡流量: rx=$((RX_AFTER - RX_BEFORE))B tx=$((TX_AFTER - TX_BEFORE))B"
```

流量应接近零（仅有后台 ARP/LLDP 的几百字节噪声）。

## 性能数据

测试环境：8× NVIDIA GPU（SM 12.0 / Blackwell，PCIe，无 NVLink），8× ConnectX-8 400Gbps 网卡（测试中**未使用**，EP_DISABLE_GIN=1），hidden=4096，topk=6，experts=256，num_sms=8。

### Dispatch 带宽（GB/s）

| Tokens | EP=2 | EP=4 |
|--------|------|------|
| 16     | 14   | 15   |
| 32     | 25   | 26   |
| 64     | 40   | 32   |
| 128    | 57   | 35   |
| 512    | 73   | 40   |
| 1024   | 77   | 41   |
| 2048   | 79   | 42   |
| 4096   | 80   | 42   |

### Combine 带宽（GB/s）

| Tokens | EP=2 | EP=4 |
|--------|------|------|
| 16     | 21   | 22   |
| 32     | 36   | 31   |
| 64     | 48   | 35   |
| 128    | 60   | 39   |
| 512    | 71   | 41   |
| 1024   | 74   | 41   |
| 2048   | 75   | 41   |
| 4096   | 76   | 41   |

## 技术细节

### 工作原理

标准 DeepEP v2 使用 NVLink 进行节点内通信，RDMA（GIN）进行节点间通信。GPU 之间的 barrier 同步依赖 PCIe 原子操作（`ptx::red_add_rel_sys`），但许多 PCIe 拓扑**不支持**原子操作。

本分支消除了这两项依赖：

1. **无需 RDMA**：设置 `EP_DISABLE_GIN=1` 跳过 GIN（RDMA）资源分配。使用 NCCL LSA 域作为 scale-up 域，所有数据通过 TMA 对称指针在 PCIe BAR 内存上传输——零网络流量。

2. **无需 PCIe 原子操作**：barrier 机制从共享计数器原子归约改为逐 rank 标志位写入：
   - 每个 rank 通过 LSA 对称指针向所有 peer 的内存写入标志位（`st_release_sys`，posted PCIe write）
   - 每个 rank 轮询自身本地标志位（`ld_acquire_sys`），等待所有 peer 到达
   - `st_release_sys` 提供 release 语义，保证在 barrier 标志位之前的数据写入对对端可见——替代了原子操作提供的内存序保证
   - 每个标志位只有单一写入者，不需要原子性

### 限制

- **EP 规模 ≤ LSA 域大小**：最大 EP 规模受 NCCL LSA 域内 GPU 数量限制。在 GPU 跨 NUMA 节点（`SYS` 连接）的机器上，EP 可能仅限于单个 NUMA 域内的 GPU。
- **不支持多机（scale-out）**：纯 PCIe 模式仅支持单机。没有 RDMA 就没有节点间通信路径。

## 故障排查

### "CPU side received count: 0 0 0..."
内核未传输数据。确保同时设置了 `EP_DISABLE_GIN=1` 和 `NCCL_LSA_TEAM_SIZE=<EP规模>`。

### "DeepEP NVLink barrier timeout"
barrier 等待 peer rank 超时。检查 EP 组内所有 GPU 是否在同一 NCCL LSA 域内。减少 `--num-processes` 以仅使用同一 NUMA 节点上的 GPU。

### "NCCL GIN is unavailable"
忘记设置 `EP_DISABLE_GIN=1`。

### 编译失败 "No module named 'torch'"
确保使用 `--no-build-isolation` 选项编译：
```bash
source "$HOME/venvs/deepep/bin/activate"
python -m pip install --no-build-isolation -e .
```

### JIT 编译失败 "Arguments mismatch for instruction 'mov'"
JIT 缓存中有过期的编译产物。清除缓存后重试：
```bash
rm -rf ~/.deep_ep/cache/*
```
