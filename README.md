# CUDA GEMM 优化学习项目

> **10天深度学习之旅**（2026年2月16-27日）：从朴素实现到超越cuBLAS性能，掌握CUDA矩阵乘法优化技术

**核心成果**：在 RTX 3070 Laptop GPU 上达到 9461 GFLOPS（cuBLAS基准的113%）

![性能演进](benchmark_results_hq.png)
![实验对比](gemm_performance_clean_hq.png)

## 🎯 项目概述

这是一个**学习导向的项目**，基于 [siboehm的CUDA GEMM教程](https://github.com/siboehm/SGEMM_CUDA)。我花费10天时间深入理解每一项优化技术，手写关键代码，进行全面的性能测试，并记录完整的学习过程。

**并非从零实现** - 我在优秀开源教程的基础上进行了深度学习和扩展：
- ✅ 详细的中文学习笔记（10天，40+页）
- ✅ 全面的性能基准测试
- ✅ 每项优化技术的深入分析
- ✅ 手写关键kernel以确保理解
- ✅ 性能可视化与对比分析

## 📊 性能测试结果

**测试环境**：
- GPU: NVIDIA GeForce RTX 3070 Laptop
- CUDA: 13.0
- 矩阵规模: 4096×4096

| Kernel | 优化技术 | GFLOPS | 相对cuBLAS |
|--------|---------|--------|-----------|
| K1 | 朴素实现 | 137 | 1.6% |
| K2 | 内存合并访问 | 975 | 11.7% |
| K3 | 共享内存 | 1311 | 15.7% |
| K4 | 1D分块 | 3624 | 43.4% |
| K5 | 2D分块 | 6965 | 83.4% |
| K6 | 向量化 | 8118 | 97.3% |
| **K10** | **Warp级分块** | **9461** | **113%** ⭐ |
| K0 | cuBLAS（基准） | 8347 | 100% |

**总体加速比**：相比朴素实现提升 68.9 倍

## 📚 学习文档

完整记录学习过程的详细笔记：

- **[学习历程总结（LEARNING_JOURNEY.md）](LEARNING_JOURNEY.md)** - 完整学习总结
- **[每日学习笔记（docs/）](docs/)** - 逐日详细记录
  - [Day 1-6: 基础优化](docs/) - 内存优化、分块技术
  - [Day 7: 深度剖析](docs/day7.md) - 2D分块、向量化
  - [Day 8: Bank Conflicts](docs/day8.md) - 共享内存优化
  - [Day 9-10: 高级技术](docs/day9.md) - Warptiling，超越cuBLAS

## 🚀 核心收获

1. **内存是瓶颈** - 内存合并访问带来7倍加速
2. **寄存器复用威力巨大** - 1D分块带来2.7倍提升
3. **Warp级优化有效** - 超越cuBLAS 13%
4. **理论≠实践** - Bank Conflicts优化反而更慢
5. **验证至关重要** - 始终与可信基准对比测试

## 🛠️ 编译运行

```bash
mkdir build && cd build
cmake ..
make

# 运行特定kernel
./sgemm 10

# 运行完整基准测试
../benchmark_kernels.sh
```

## 🙏 致谢

本项目基于：
- **原始教程**: [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)

## 📄 许可

原始代码来自 [siboehm](https://github.com/siboehm/SGEMM_CUDA)。学习笔记和分析由Mingjiu完成（2026年）。

---

**说明**：这是一个学习项目。核心实现来自siboehm的优秀教程，我的贡献在于深入理解、全面测试、详细文档和深度分析。
