# DeepEP 纯 PCIe 通信 kernel 测试记录

测试日期：2026-07-30  
测试工作区：`/workdir/tmp/DeepEP`  
Git HEAD：`67bf8f3c3e499e22029141ff152e9e6474f9952e`

> 注意：测试时工作区已有未提交修改，因此这个 commit ID 只能标识基线，不能完整代表被测源码。本次新增的可复跑脚本是
> `tests/elastic/bench_cuda_p2p.py` 和 `tests/elastic/bench_pcie_ep.py`；没有覆盖已有的
> `PCIe Deployment.md`。

## 1. 目标和结论

目标是只测本机 GPU 间的纯通信性能，不使用 NVLink、RDMA，也不与计算重叠。测试包含：

1. CUDA 单向 P2P copy，作为 PCIe 链路基线；
2. DeepEP elastic dispatch、cached dispatch 和 combine 主通信 kernel；
3. 单远端 peer 的最好情况；
4. 一个 rank 同时访问多个远端 peer 的 all-to-all 压力情况；
5. BF16、FP8、token 数量和 `num_sms` 扫描。

本机上的主要结论如下。

- CUDA 单向 P2P copy 在 256 MiB 时平均为 **28.475 GB/s**。
- DeepEP 单远端 peer、BF16、`num_sms=8` 在消息足够大时，单 rank useful payload
  约为 **14.15 GB/s**，4 rank 汇总约为 **56.6–57.4 GB/s**。
- 单 peer 路径只用 8 个 SM 就达到平台；继续增加 SM 没有收益。
- FP8 dispatch 需要更大的 token 数才填满链路。4096 tokens/rank 时为
  **57.065 GB/s aggregate useful payload**，单 rank 的源 token 吞吐约为 BF16
  2048 tokens/rank 时的 1.95 倍。
- 当每个 token 同时发往 3 个远端 rank 时，性能明显下降；该模式在本次扫描中偏好
  `num_sms=112`，而不是 8。2048 BF16 tokens/rank 的 cached dispatch 最好结果为
  **3.485 GB/s aggregate useful payload**。
- 因此，“纯通信 kernel 的最好情况”和“真实的多 peer all-to-all”必须分别报告。
  `remote_fanout=1` 是链路/实现上限，不应被当成通用 MoE all-to-all 的结果。

## 2. 测试环境

| 项目 | 值 |
|---|---|
| GPU | 4 × NVIDIA H100 PCIe 80 GB |
| 每卡 SM | 114 |
| PCIe | Gen4 ×16 |
| GPU 拓扑 | 所有 GPU pair 均显示 `NODE`，无 `NV#` |
| NUMA | 4 张卡均位于 NUMA node 1 |
| CUDA peer access | 所有不同 GPU pair 均为 `True` |
| Driver | 595.58.03 |
| PyTorch | `2.12.0a0+5aff3928d8.nv26.05` |
| PyTorch CUDA | 13.2 |
| NCCL | 2.30.4 |

拓扑检查命令：

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,pcie.link.gen.current,pcie.link.width.current \
  --format=csv,noheader
nvidia-smi topo -m
```

测试使用以下环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EP_DISABLE_GIN=1
export NCCL_LSA_TEAM_SIZE=4
export EP_BUFFER_DEBUG=0
```

其中 `EP_DISABLE_GIN=1` 是最关键的约束：它禁止 GIN/RDMA 路径。初始化后还检查
`buffer.get_logical_domain_size() == (1, 4)`，即 1 个 scale-out rank、4 个本机
LSA/PCIe rank。测试不使用 NVLink 或 RDMA。

## 3. 参数选择

| 参数 | 本次取值 | 原因 |
|---|---:|---|
| `allow_hybrid_mode` | `False` | 单机纯 PCIe，不走分层 RDMA + NVLink |
| `allow_multiple_reduction` | `True` | 允许 combine 分阶段归并，避免为了单次 reduction 增加传输布局/数据量 |
| `prefer_overlap_with_compute` | `False` | 没有计算 kernel，不为重叠而主动减少通信资源 |
| `do_expand` | `False` | 只测普通通信布局，不加入 one-token-per-expert-slot 展开 |
| `async_with_compute_stream` | `False` | 不测试与计算流异步重叠 |
| `do_handle_copy` | `False` | 性能段不复制 handle，避免测入额外工作 |
| `do_cpu_sync` | `False` | 性能段只分析 GPU kernel；正确性 smoke 使用 `True` |
| `expert_alignment` | `1` | 不加入 padding 带来的额外 token |
| `num_qps` 参数 | `0` | 使用 API 默认值；当前 LSA/PCIe 路径不使用 RDMA QP |
| `num_sms` | 扫描后决定 | 单 peer 选 8；3 peers 选 112 |
| `flush_l2` | `True` | 每轮测量前清 L2，减少缓存命中带来的虚高 |

日志里的 `theoretical_num_qps_unused` 只是 API 根据 `num_sms` 算出的理论值。在
`EP_DISABLE_GIN=1` 且逻辑域为 `(1, 4)` 的测试里没有 RDMA 请求，因此不能把它理解为
实际创建或使用了这些 RDMA QP。

## 4. 路由和指标定义

`bench_pcie_ep.py` 生成确定性、均衡、完全不落到本 rank 的路由：

- `--remote-fanout 1`：每个 token 的所有 top-k expert 都在同一个远端 rank；
  4 rank 构成循环流量，适合观察最好情况。
- `--remote-fanout 2`：每个 token 访问两个远端 rank。
- `--remote-fanout 3`：每个 token 访问全部三个远端 rank，代表本机全 peer 压力。

`num_topk=6`，所以 fanout 1、2、3 都能均匀分配，而且一个 token 不会重复选择同一个
expert。

计时使用项目现有的 `bench_kineto`，分别提取：

- `dispatch_impl` / `combine_impl`：主通信 kernel；
- `dispatch_copy_epilogue_impl` / `combine_reduce_epilogue_impl`：本地整理或 reduction
  epilogue。

表格中的主时间取各 rank 平均 kernel 时间的最大值。带宽定义为：

```text
aggregate useful GB/s = 所有 rank 收到的有效 payload 字节之和
                        / 最慢 rank 的主 kernel 时间 / 1e9

per-rank useful GB/s  = aggregate useful GB/s / rank 数
```

“useful payload”包括输入 tensor、top-k index 和可选权重，不包括内部对齐、同步状态、
控制元数据和 PCIe 协议开销。因此它适合比较 DeepEP 参数，不等于 PCIe 线速，也不能与
CUDA copy 的单向原始带宽直接按百分比换算。

## 5. 正确性测试

先用仓库原有的 `test_ep.py` 做 exact correctness。命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EP_DISABLE_GIN=1 \
NCCL_LSA_TEAM_SIZE=4 \
EP_BUFFER_DEBUG=1 \
python tests/elastic/test_ep.py \
  --num-processes 4 \
  --hidden 4096 \
  --num-topk 6 \
  --num-experts 256 \
  --num-tokens 64 \
  --num-sms 8 \
  --allow-hybrid-mode 0 \
  --prefer-overlap-with-compute 0 \
  --allow-multiple-reduction 1 \
  --test-first-only \
  --skip-perf-test
```

结果：退出码 0；输出确认 `Ranks: 1 x 4` 和
`Pure PCIe mode: GIN disabled, using LSA domain (lsaSize=4)`。

随后对新基准做 remote-only 路由检查：

```bash
EP_DISABLE_GIN=1 \
NCCL_LSA_TEAM_SIZE=4 \
python tests/elastic/bench_pcie_ep.py \
  --num-processes 4 \
  --tokens 64 \
  --num-max-tokens 64 \
  --hidden 4096 \
  --num-topk 6 \
  --num-experts 256 \
  --remote-fanout 3 \
  --num-sms 8 \
  --dtype bf16 \
  --profile all \
  --num-tests 2 \
  --check
```

检查内容包括：

- 每个 rank 收到的 token 数与 fanout 完全一致；
- 收到的 top-k lane 数量正确；
- 收到的所有 token 均来自远端 rank；
- dispatch 后执行 combine，输出 shape 和权重正确；
- BF16 与 FP8、fanout 1/2/3 的 smoke 均通过。

## 6. CUDA P2P 原始基线

复跑命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tests/elastic/bench_cuda_p2p.py \
  --sizes-mib 16,64,256 \
  --warmups 5 \
  --num-tests 20 \
  --check
```

测试会遍历全部 12 个有向 GPU pair。每个 pair 只做单向 `dst.copy_(src)`：

| 大小 | 最小 GB/s | 平均 GB/s | 最大 GB/s |
|---:|---:|---:|---:|
| 16 MiB | 26.713 | 27.618 | 27.985 |
| 64 MiB | 28.279 | 28.336 | 28.415 |
| 256 MiB | 28.466 | 28.475 | 28.496 |

所有 pair 差异很小，未发现单张卡或单条 GPU pair 明显异常。

## 7. DeepEP 测试结果

公共参数：

```text
hidden=7168, num_topk=6, num_experts=256, with_topk_weights=True
allow_hybrid_mode=False, allow_multiple_reduction=True
prefer_overlap_with_compute=False, do_expand=False
```

除非特别说明，每个 Kineto active window 包含 5 次 kernel 调用，并在每次调用前清 L2。

### 7.1 单 peer：`num_sms` 扫描

条件：4 ranks、BF16、2048 tokens/rank、`remote_fanout=1`、只测 dispatch。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EP_DISABLE_GIN=1 \
NCCL_LSA_TEAM_SIZE=4 \
python tests/elastic/bench_pcie_ep.py \
  --num-processes 4 \
  --tokens 2048 \
  --num-max-tokens 2048 \
  --hidden 7168 \
  --num-topk 6 \
  --num-experts 256 \
  --remote-fanout 1 \
  --num-sms 8,16,32,48,64,80,96,112,114 \
  --dtype bf16 \
  --profile dispatch \
  --num-tests 5
```

| `num_sms` | dispatch 主时间 µs | aggregate useful GB/s |
|---:|---:|---:|
| **8** | **2056** | **57.408** |
| 16 | 2110 | 55.939 |
| 32 | 2122 | 55.622 |
| 48 | 2084 | 56.636 |
| 64 | 2089 | 56.501 |
| 80 | 2143 | 55.077 |
| 96 | 2143 | 55.077 |
| 112 | 2112 | 55.886 |
| 114 | 2148 | 54.949 |

后续独立复测中，8 SM 的 dispatch 为 56.609 GB/s，说明本表最好值约有 1.4% 的运行间
波动，但“8 SM 已经进入平台”的结论不变。

### 7.2 单 peer：消息大小和三类 kernel

条件：4 ranks、BF16、`remote_fanout=1`、`num_sms=8`。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EP_DISABLE_GIN=1 \
NCCL_LSA_TEAM_SIZE=4 \
python tests/elastic/bench_pcie_ep.py \
  --num-processes 4 \
  --tokens 512,2048,4096 \
  --num-max-tokens 4096 \
  --hidden 7168 \
  --num-topk 6 \
  --num-experts 256 \
  --remote-fanout 1 \
  --num-sms 8 \
  --dtype bf16 \
  --profile all \
  --num-tests 5 \
  --check
```

| tokens/rank | dispatch µs / GB/s | cached dispatch µs / GB/s | combine µs / GB/s |
|---:|---:|---:|---:|
| 512 | 938.7 / 31.433 | 954.3 / 30.922 | 937.5 / 31.371 |
| 2048 | 2085 / 56.609 | 2071 / **56.992** | 2088 / 56.340 |
| 4096 | 4169 / 56.623 | 4167 / 56.650 | 4137 / **56.871** |

2048 tokens/rank 已经足以让 BF16 进入带宽平台；继续增加 token 主要是线性增加时间。

### 7.3 单 peer：FP8 dispatch

条件与上一节相同，只把 dispatch 输入改为 FP8。

| tokens/rank | dispatch 主时间 µs | aggregate useful GB/s | per-rank useful GB/s |
|---:|---:|---:|---:|
| 512 | 508.6 | 30.054 | 7.514 |
| 2048 | 1461 | 41.852 | 10.463 |
| 4096 | 2143 | **57.065** | **14.266** |

FP8 每个 token 的通信字节更少，所以 2048 tokens/rank 尚未完全填满链路；4096 时才达到
与 BF16 大消息相近的 useful-bandwidth 平台。以源 token 数计算：

- BF16，2048 tokens：约 0.982 M tokens/s/rank；
- FP8，4096 tokens：约 1.911 M tokens/s/rank。

### 7.4 单 peer 与多 peer fanout

条件：4 ranks、BF16、2048 tokens/rank、`num_sms=112`。为了只比较 fanout，三组使用
相同 SM 数。

| remote fanout | dispatch 主时间 µs | aggregate useful GB/s | per-rank useful GB/s |
|---:|---:|---:|---:|
| 1 | 2104 | **56.098** | **14.025** |
| 2 | 55837 | 4.228 | 1.057 |
| 3 | 104933 | 3.374 | 0.844 |

fanout 增加后，虽然 useful payload 的计算已包含额外远端副本，主 kernel 时间仍然大幅
增长。这个结果说明当前纯 PCIe direct 路径对多 peer 流量非常敏感。

### 7.5 全部三个远端 peer：`num_sms` 扫描

条件：4 ranks、BF16、4096 tokens/rank、`remote_fanout=3`、只测 dispatch。

| `num_sms` | dispatch 主时间 µs | aggregate useful GB/s |
|---:|---:|---:|
| auto（64） | 241026 | 2.938 |
| 8 | 247349 | 2.863 |
| 16 | 248124 | 2.854 |
| 32 | 241134 | 2.937 |
| 48 | 236417 | 2.996 |
| 64 | 239714 | 2.954 |
| 80 | 231871 | 3.054 |
| 96 | 230972 | 3.066 |
| **112** | **227598** | **3.112** |
| 114 | 228127 | 3.104 |

这里 112 SM 最好，96–114 SM 的差异约为 1.5%。它与单 peer 的最优值不同，说明 SM
应按实际路由 fanout 调优。

### 7.6 全部三个远端 peer：消息大小

条件：4 ranks、BF16、`remote_fanout=3`、`num_sms=112`。

| tokens/rank | dispatch µs / GB/s | cached dispatch µs / GB/s | combine µs / GB/s |
|---:|---:|---:|---:|
| 128 | 8802 / 2.514 | 8708 / 2.541 | 11234 / 1.963 |
| 512 | 28039 / 3.157 | 28971 / 3.056 | 33943 / 2.599 |
| 2048 | 104933 / 3.374 | 101606 / **3.485** | 125884 / 2.803 |
| 4096 | 224824 / 3.150 | 226580 / 3.126 | 250852 / **2.814** |
| 8192 | 475267 / 2.980 | 473835 / 2.989 | 503970 / 2.801 |

dispatch/cached dispatch 的峰值在 2048 tokens/rank 左右；combine 从 2048 开始基本进入
约 2.8 GB/s 的平台。

FP8、同样 fanout 3 和 112 SM 的 dispatch 结果：

| tokens/rank | dispatch 主时间 µs | aggregate useful GB/s |
|---:|---:|---:|
| 512 | 16555 | 2.770 |
| 2048 | 52829 | **3.472** |
| 4096 | 109193 | 3.360 |

### 7.7 两张卡的隔离测试

两张卡时每个 rank 只有一个远端 peer。BF16、`num_sms=112`：

| tokens/rank | dispatch 主时间 µs | aggregate useful GB/s | per-rank useful GB/s |
|---:|---:|---:|---:|
| 512 | 969.9 | 15.211 | 7.606 |
| 2048 | 2105 | 28.036 | 14.018 |
| 4096 | 4150 | **28.441** | **14.221** |

这里的 per-rank 平台与 4-rank、fanout 1 基本一致。4-rank 循环流量能同时形成四条有向
数据流，所以 aggregate useful bandwidth 约为两卡测试的两倍。

## 8. 推荐配置

如果目的确实是“只看纯通信 kernel 能跑到的最好情况”，建议先使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EP_DISABLE_GIN=1 \
NCCL_LSA_TEAM_SIZE=4 \
python tests/elastic/bench_pcie_ep.py \
  --num-processes 4 \
  --tokens 2048,4096 \
  --num-max-tokens 4096 \
  --hidden 7168 \
  --num-topk 6 \
  --num-experts 256 \
  --remote-fanout 1 \
  --num-sms 8 \
  --dtype both \
  --profile all \
  --num-tests 10 \
  --check
```

参数建议：

- BF16 至少用 2048 tokens/rank，FP8 至少用 4096 tokens/rank；
- 单 peer 从 8 SM 开始，不必为了“通信越多 SM 越快”而占满整卡；
- 保持 `do_expand=False`、`prefer_overlap_with_compute=False`；
- QP 不需要扫描；纯 PCIe LSA 路径没有 RDMA QP 可调；
- 首次运行加 `--check`，正式重复测量可去掉它；
- 最终报告同时给出 main kernel 时间和 useful GB/s，不把 epilogue 混进通信时间。

如果实际模型会把每个 token 发到全部三个远端 rank，则应改用：

```text
remote_fanout=3, num_sms=112, tokens≈2048/rank
```

但必须将这组结果标成“多 peer all-to-all”，不能与单 peer 最好情况混在一起。

## 9. 复验注意事项

1. 每次测试前用 `nvidia-smi topo -m` 确认没有出现 NVLink 拓扑变化。
2. 必须保留 `EP_DISABLE_GIN=1`；否则结果可能混入 GIN/RDMA。
3. 保证其他进程不占用 GPU，并固定相同的 `CUDA_VISIBLE_DEVICES` 顺序。
4. 如果换 GPU 型号、PCIe switch、NUMA 布局或 DeepEP 源码，重新扫描 `num_sms`。
5. 至少重复完整命令 3 次，报告中位数和范围；本记录中的短 Kineto 扫描适合找参数，
   不应替代长时间稳定性测试。
6. 多 peer 结果远低于单 peer，后续若要定位原因，应用 Nsight Systems/Compute 分析
   peer 轮询、同步等待、访存事务和 PCIe 并发；本记录只陈述可复现现象，不据此断定
   唯一瓶颈。
