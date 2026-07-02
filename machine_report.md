# Machine Report — `spark-166a`
**Generated:** 2026-07-02

---

## Overview

| Component | Summary |
|---|---|
| Hostname | `spark-166a` |
| OS | Ubuntu Linux (kernel 6.17.0-1026-nvidia) |
| Architecture | aarch64 (ARM 64-bit) |
| Platform | Dell Spark (NVIDIA GB10 SoC) |
| Uptime | 8 h 24 min at time of report |
| Active users | 57 sessions |

---

## CPU

| Property | Value |
|---|---|
| Total logical cores | 20 (no hyperthreading — 1 thread/core) |
| Layout | big.LITTLE: 10× Cortex-X925 + 10× Cortex-A725 |
| Performance cores | 10× ARM Cortex-X925 — up to **3.9 GHz** |
| Efficiency cores | 10× ARM Cortex-A725 — up to **2.8 GHz** |
| Socket(s) | 1 |
| NUMA nodes | 1 (all 20 cores in a single NUMA domain) |
| L1d / L1i cache | 1.3 MiB each (20 instances) |
| L2 cache | 25 MiB total (20 instances) |
| L3 cache | 24 MiB (2 instances) |
| SIMD / ML extensions | SVE, SVE2, BF16, I8MM, AES, SHA3, SM4 |
| Frequency boost | Disabled |

The Cortex-X925 P-cores and Cortex-A725 E-cores reflect NVIDIA's GB10 DeskEdge SoC design. The SVE2/BF16/I8MM extensions give native acceleration for ML inference workloads without GPU involvement.

---

## Memory

| Property | Value |
|---|---|
| Total RAM | **121 GiB** |
| Used (at report time) | 17 GiB |
| Free | 89 GiB |
| Available | 104 GiB |
| Swap | None configured |

---

## GPU — NVIDIA GB10

| Property | Value |
|---|---|
| Model | NVIDIA GB10 |
| Driver version | 580.173.02 |
| CUDA version | 13.0 |
| Persistence mode | On |
| Temperature (at report time) | 58 °C |
| Power draw | 24 W |
| GPU utilisation | **61%** |
| GPU memory | Not individually queryable (unified memory architecture) |

### Active GPU processes (at report time)

| Process | Type | GPU Memory |
|---|---|---|
| `/usr/lib/xorg/Xorg` | Graphics | 126 MiB |
| `/usr/bin/gnome-shell` | Graphics | 37 MiB |
| `/isaac-sim/kit/kit` | Compute + Graphics | **5,845 MiB** |

Isaac Sim is the dominant GPU consumer.

---

## Storage

| Property | Value |
|---|---|
| Drive | NVMe SSD (`/dev/nvme0n1`) |
| Total capacity | **3.7 TB** |
| Used | 57 GB (2%) |
| Available | 3.5 TB |
| Interface | NVMe (PCIe, non-rotating) |

---

## Network

### Active interfaces

| Interface | Type | State | Notes |
|---|---|---|---|
| `wlP9s9` | WiFi | UP | Primary internet uplink; 12.4 GB TX / 5.0 GB RX total |
| `enp1s0f0np0` | Ethernet (RoCE) | UP | Mellanox ConnectX-7 port 0 |
| `enP2p1s0f0np0` | Ethernet (RoCE) | UP | Mellanox ConnectX-7 port 2 |
| `tailscale0` | VPN (mesh) | UP | 10.0 GB TX / 487 MB RX total |
| `docker0` | Docker bridge | DOWN (no container) | — |
| `enp1s0f1np1`, `enP2p1s0f1np1`, `enP7s7` | Ethernet (RoCE) | DOWN | Ports unplugged |

### RDMA / RoCE NICs — Mellanox ConnectX-7

This machine has **two dual-port Mellanox ConnectX-7** adapters (4 ports total). ConnectX-7 supports up to **400 Gb/s** per port and is the standard NIC for high-performance distributed compute clusters.

| NIC | RDMA State |
|---|---|
| `rocep1s0f0` | **PORT_ACTIVE** — ready for RDMA |
| `rocep1s0f1` | PORT_DOWN — cable unplugged |
| `roceP2p1s0f0` | **PORT_ACTIVE** — ready for RDMA |
| `roceP2p1s0f1` | PORT_DOWN — cable unplugged |

> Two ports are live but carrying only low-volume local traffic at present. The hardware is configured for high-bandwidth inter-node compute; it has not been paired with a workload manager (no Slurm/MPI installed).

---

## Connectivity — Tailscale Mesh VPN

This machine is a member of a shared Tailscale network (`fd7a:115c:a1e0::/48`) with the following peers:

| Peer hostname | OS | Tailscale IP | Status | Data sent | Data received |
|---|---|---|---|---|---|
| `DESKTOP-3V6MO1N` | Windows | 100.101.212.58 | **Active** (direct via WiFi) | 4.82 GB | 0.42 GB |
| `UK-J4NG4XNGJH` | macOS | 100.81.239.26 | **Active** (relay: `mad`) | 4.10 GB | 0.10 GB |
| `ip-10-195-180-234` | Linux (AWS) | 100.88.5.83 | **Active** (direct) | ~0 | ~0 |
| `ES-CNDFERNI` | Linux | 100.90.253.125 | Inactive | — | — |

The Windows peer (`DESKTOP-3V6MO1N`) has the heaviest traffic volume (4.82 GB sent) and establishes a **direct** link over the shared WiFi, suggesting it is a second Dell Spark on the same local network.

---

## Distributed / Shared Compute Assessment

| Technology | Present? | Notes |
|---|---|---|
| Slurm / PBS | No | No cluster scheduler installed |
| MPI (`mpirun`) | No | Not installed |
| Kubernetes | No | No `kubectl` |
| Ray | No | Not installed |
| PyTorch Distributed | No | PyTorch not installed |
| NVLink (multi-GPU) | No | Single GPU, no NVLink bridge |
| RDMA / RoCE | **Yes** | 2× ConnectX-7, 2 ports active — hardware ready |
| Tailscale VPN mesh | **Yes** | 4 peers; active cross-machine traffic with 3 |
| Docker | **Yes** | Running (version 29.2.1), currently no active containers |
| Shared NFS/GPFS/Ceph | No | No network filesystem mounts |

### Conclusion

`spark-166a` is **hardware-capable of shared compute** — the Mellanox ConnectX-7 RoCE NICs are specifically designed for low-latency, high-bandwidth inter-node RDMA workloads. It also participates in a Tailscale mesh connecting multiple machines under the same team. However, no distributed workload framework (Slurm, Ray, MPI, etc.) is currently installed or running, so compute is not being pooled in an automated way at this time.

---

*Report generated by system introspection on `spark-166a` — 2026-07-02.*
