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
- `SYS` 连接（跨 NUMA）→ 默认不在同一 LSA 域内；设置 `NCCL_LSA_TEAM_SIZE=<EP规模>` 可以让 LSA 域覆盖跨 NUMA 的 GPU，但此时应使用 copy engine 模式或 host 内存模式（见[跨 NUMA 的 EP=8](#跨-numa-的-ep8)）

典型拓扑示例（8 卡，两个 NUMA 各 4 卡）：
```
        NUMA 0                          NUMA 1
GPU0  GPU1  GPU2  GPU3          GPU4  GPU5  GPU6  GPU7
 └─PXB──┘    └─PXB──┘           └─PXB──┘    └─PXB──┘
    └──NODE──┘                      └──NODE──┘
    └── LSA 域 0 ──┘                └── LSA 域 1 ──┘
              └──────── SYS（跨 NUMA）────────┘
```

在这种拓扑下，**EP=2 和 EP=4 可以正常运行**（同一 NUMA / LSA 域内）；**EP=8 需要使用 copy engine 模式或 host 内存模式**（默认的 SM 直写跨 NUMA 时带宽会塌缩到约 1 GB/s）。

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

### 1. 安装 Conda（如未安装）

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
```

### 2. 创建 Conda 环境

```bash
conda create -n deepep python=3.12 -y
conda activate deepep
```

### 3. 安装 PyTorch

安装与 CUDA 版本匹配的 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
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

PyTorch ≥ 2.12 的 pip 包自带 `nvidia-nvshmem-cu13`，DeepEP 的构建系统会自动检测到，**无需手动安装或设置环境变量**。

验证 NVSHMEM 已随 pip 安装：
```bash
python -c "import nvidia.nvshmem; print('NVSHMEM path:', nvidia.nvshmem.__path__[0])"
```

如果 PyTorch 版本较低未自带 NVSHMEM，需手动安装 NVSHMEM ≥ 2.11 并设置：
```bash
export NVSHMEM_ROOT=/path/to/nvshmem
```

### 6. 安装其他依赖

测试脚本需要 numpy：
```bash
pip install numpy
```

## 编译安装

```bash
git clone https://github.com/MengYu10151/DeepEP.git
cd DeepEP
git checkout pcie-no-atomic

pip install --no-build-isolation -e .
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

## 跨 NUMA 的 EP=8

在两路 CPU、每个 NUMA 4 卡的机器上，默认的 SM 直写模式跨 socket 时会塌缩：单卡同时向 ≥3 个对面 socket 的 GPU 写入，带宽从 20+ GB/s 掉到约 1 GB/s（CPU 对 PCIe↔PCIe 转发的处理能力有限）。EP=8 请使用以下两种模式之一，二者都让跨 CPU 的数据经 host 内存中转，只有同一 CPU root port（同一 PCIe switch）下的 GPU 之间直接 P2P：

| | copy engine 模式（`EP_PCIE_CE=1`） | host 内存模式（`EP_PCIE_SHM=1`） |
|---|---|---|
| 数据搬运 | copy engine（host 发起 `cudaMemcpyAsync`） | SM（kernel 内完成） |
| CPU sync | 需要 | 不需要 |
| CUDA graph 捕获 | 不支持（自动退回 SM 直写） | 支持 |
| cached dispatch | 退回 SM 直写 | 退回 SM 直写 |
| 额外显存 | 每张卡约 2 GB（对端 CUDA context）+ landing 缓冲区 + 暂存区 | 无 |
| 额外 host 锁页内存 | 无（驱动内部中转） | 每个 rank 约 1.02 × buffer 大小 |
| dispatch / combine（150 MB/rank，8 SMs） | 23.6 / 23.7 GB/s | 21.1 / 24.4 GB/s |

两个变量同时设置时，host 内存模式优先。

```bash
# copy engine 模式
EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=8 EP_PCIE_CE=1 \
  python tests/elastic/test_ep.py \
    --num-processes 8 --hidden 4096 --num-topk 6 --num-experts 256 \
    --num-tokens 4096 --num-sms 8 \
    --allow-hybrid-mode 0

# host 内存模式
EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=8 EP_PCIE_SHM=1 \
  python tests/elastic/test_ep.py \
    --num-processes 8 --hidden 4096 --num-topk 6 --num-experts 256 \
    --num-tokens 4096 --num-sms 8 \
    --allow-hybrid-mode 0
```

| 变量 | 值 | 说明 |
|------|----|------|
| `NCCL_LSA_TEAM_SIZE` | `8` | 让 NCCL LSA 域覆盖跨 NUMA 的全部 8 卡 |
| `EP_PCIE_CE` | `1` | copy engine 模式 |
| `EP_PCIE_SHM` | `1` | host 内存模式 |
| `EP_PCIE_SHM_PULL_WARPS` | 可选，默认 `4` | host 内存模式下每个 SM 的接收（pull）warp 数 |
| `EP_PCIE_CE_DIRECT_GROUP` | 可选 | 手动指定直连组大小，两种模式通用（默认按 sysfs 自动识别同一 CPU root port 下的 GPU） |

> **不要设置 `NCCL_P2P_LEVEL=SYS`**：两种模式都不需要它。NCCL 在进程内首次读取该变量后即缓存，之后同一进程中的所有 communicator（包括 PyTorch 的 `all_to_all` 等集合通信）都会走跨 socket 直连 P2P，带宽塌缩到约 0.9 GB/s。不设置时，DeepEP 与 PyTorch 集合通信可在同一进程内同时达到约 23 GB/s。

> **直连组不要按 NUMA 划分**：实测 `EP_PCIE_CE_DIRECT_GROUP=4`（同一 NUMA 内全部直连）会让 EP=8 的 dispatch 从约 23 GB/s 降到约 17 GB/s。同一张 GPU 同时往 host 内存写和往对端 GPU 写时，P2P 写会被 host 写挤占（实测从 51.7 GB/s 降到 11.2 GB/s），跨 root port 的对端因此成为瓶颈。

### copy engine 模式工作方式

kernel 只把 token 写入本地显存的暂存区（按目标 rank 连续排列），host 再为每个目标 rank 发起一次 `cudaMemcpyAsync`，写入对端的 landing 缓冲区（`cudaMalloc` + CUDA IPC），最后通过 stream memory op 发送/等待到达信号。同一 CPU root port 下的 GPU 直接 P2P；其余 GPU 的 IPC 句柄在对端设备的 context 中打开，驱动会把这些跨 context 拷贝拆成经 host 内存中转的多 copy engine 流水线，从而避开跨 socket P2P 的塌缩。

### host 内存模式工作方式

- 每个 rank 在其 GPU 所在的 NUMA 节点上分配一块锁页 host 内存（`cuMemCreate`，`CU_MEM_LOCATION_TYPE_HOST_NUMA`），布局与该 rank 的 GPU 接收缓冲区相同（`[源 rank][slot]`），后面附带每个 slot 的到达标志。初始化时各 rank 通过私有临时目录（`0700`）中的 Unix socket 以 `SCM_RIGHTS` 交换这块内存的 FD，双方用 `SO_PEERCRED` 校验对端必须是本任务中的 rank 进程，然后全部映射给本 GPU。
- 发送端：发往远端 rank 的 token 用 TMA 写入接收端的 host 内存，写完成后发布该 slot 的标志（值为本次调用的 epoch）；同一 switch 下的对端照常直接 P2P 写入。
- 接收端：同一个 kernel 中的 pull warp 轮询标志（一次合并的 `ld.volatile` 读回 32 个 slot 的标志），到达后把连续的 slot 一次性拷回本地显存，使写 host 与读 host 流水重叠。pull 按"slot 区间优先、源 rank 其次"的顺序进行，与各 rank 按 slot 顺序到达的数据一致。
- combine 的发送端也按"(slot, 源 rank)"的顺序交错分配 token，使每个接收端按 slot 顺序收到数据、并发写入分散到所有接收端，且在各 rank token 数不均衡时各 warp 负载仍然均衡。
- epoch 由 GPU 在 workspace 中维护，因此不需要 CPU sync，也可以被 CUDA graph 捕获（已验证 graph replay 的结果与 eager 执行逐位一致）。

### 实测

8× SM 12.0 PCIe、2× EPYC 9575F，BF16，hidden=4096，topk=6，experts=256，8 SMs，端到端含 epilogue；PyTorch 基线为默认 NCCL 环境变量、每个对端的分块 4 KB 对齐，与 nccl-tests `alltoall_perf` 结果一致（单位 GB/s）：

| 每 rank 数据量 | CE dispatch | CE combine | SHM dispatch | SHM combine | PyTorch NCCL all_to_all |
|---------------|-------------|------------|--------------|-------------|-------------------------|
| 75 MB（2048 tokens） | 22.8 | 22.9 | 21.2 | 23.9 | 22.2 |
| 150 MB（4096 tokens） | 23.6 | 23.7 | 21.1 | 24.4 | 23.7 |
| 300 MB（8192 tokens） | 23.2 | 23.7 | 21.2 | 24.6 | 24.5 |

host 内存模式下 dispatch 用 12 个 SM 可达约 23.4 GB/s。

### 注意事项

- 两种模式都要求单机（`num_scaleout_ranks == 1`）。expand 且不允许多次归约的 combine 会自动退回 SM 直写路径；rank layout（`allow_multiple_reduction` 且 `num_ranks <= num_topk`）的 combine 同样走 `[src rank][slot]` 暂存布局，由 epilogue 按 `dst_buffer_slot_idx` 定位。
- copy engine 模式需要 CPU sync（`do_cpu_sync=True`，默认）；无 CPU sync、CUDA graph 捕获时退回 SM 直写路径。
- host 内存模式初始化时需要可写的临时目录（`tempfile.mkdtemp`），运行后自动删除。
- 两种模式只对 EP > 4 生效：EP≤4 时即使设置了 `EP_PCIE_CE` / `EP_PCIE_SHM` 也会被忽略（rank 0 打印提示），始终走 SM 直写路径（更快，如 EP4 top-k 8 combine：SM 7.4 ms，CE 10.1 ms，SHM 10.0 ms）。
- 与 PyTorch NCCL `all_to_all` 对比时，每个对端的分块需按 4 KB 对齐，否则 NCCL 会慢 2 倍以上（例如 150.3 MB/rank 不对齐时仅约 8.6 GB/s）。

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

- **EP 规模 ≤ LSA 域大小**：最大 EP 规模受 NCCL LSA 域内 GPU 数量限制。在 GPU 跨 NUMA 节点（`SYS` 连接）的机器上，需设置 `NCCL_LSA_TEAM_SIZE=<EP规模>` 并使用 copy engine 模式（`EP_PCIE_CE=1`）或 host 内存模式（`EP_PCIE_SHM=1`）。
- **不支持多机（scale-out）**：纯 PCIe 模式仅支持单机。没有 RDMA 就没有节点间通信路径。

## 故障排查

### "DeepEP PCIe host-memory receive timeout" / "PCIe host-memory count timeout"
host 内存模式下等待对端数据超时。通常是某个 rank 未进入同一次 dispatch/combine（各 rank 调用次数不一致），或某个 rank 已崩溃；检查所有 rank 的日志。

### "PCIe host-memory FD exchange failed"
host 内存模式初始化时 FD 交换失败。确认所有 rank 在同一台机器、同一用户下运行，且临时目录（`TMPDIR`，默认 `/tmp`）对所有 rank 可见（同一容器或共享的 `/tmp`）。

### "CPU side received count: 0 0 0..."
内核未传输数据。确保同时设置了 `EP_DISABLE_GIN=1` 和 `NCCL_LSA_TEAM_SIZE=<EP规模>`。

### "DeepEP NVLink barrier timeout"
barrier 等待 peer rank 超时。检查 EP 组内所有 GPU 是否在同一 NCCL LSA 域内。减少 `--num-processes` 以仅使用同一 NUMA 节点上的 GPU。

### "NCCL GIN is unavailable"
忘记设置 `EP_DISABLE_GIN=1`。

### 编译失败 "No module named 'torch'"
确保使用 `--no-build-isolation` 选项编译：
```bash
conda activate deepep
pip install --no-build-isolation -e .
```

### JIT 编译失败 "Arguments mismatch for instruction 'mov'"
JIT 缓存中有过期的编译产物。清除缓存后重试：
```bash
rm -rf ~/.deep_ep/cache/*
```
