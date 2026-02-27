# Day 7 学习笔记（2026-02-23）

## 📅 学习概况

- **日期**：2026年2月23日（周一）
- **学习时间**：约5-6小时
- **主要内容**：Kernel 5、Kernel 6、完整性能测试、Kernel 7预览
- **状态**：核心kernel学习完成，性能达到cuBLAS 86.7%

---

## 📚 学习内容

### 1. Kernel 5：2D Blocktiling

#### 核心思想

**从"一条线"到"一个面"**

```
Kernel 4 (1D): 每线程计算 8×1 = 8个元素
Kernel 5 (2D): 每线程计算 8×8 = 64个元素

关键改进：双向数据复用
```

#### 主要变化

**1. 模板参数新增TN**
```cpp
template <const int BM, const int BN, const int BK, const int TM, const int TN>
//                                                            ↑ 新增
```

**2. 线程索引计算变化**
```cpp
// Kernel 4
threadCol = threadIdx.x % BN;
threadRow = threadIdx.x / BN;

// Kernel 5
threadCol = threadIdx.x % (BN / TN);  // 除以TN
threadRow = threadIdx.x / (BN / TN);
```

**3. 寄存器数组扩展**
```cpp
// Kernel 4
float threadResults[TM];      // 8个
float tmpB;                   // 1个B值

// Kernel 5
float threadResults[TM * TN]; // 64个
float regM[TM];               // 8个A值
float regN[TN];               // 8个B值
```

**4. 外积计算**
```cpp
// 双重循环计算外积
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
  for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
    threadResults[resIdxM * TN + resIdxN] +=
        regM[resIdxM] * regN[resIdxN];
  }
}

// regM ⊗ regN = TM×TN 矩阵
```

#### 双向数据复用

```
regM[i] 被复用 TN 次（用于计算TN个列）
regN[j] 被复用 TM 次（用于计算TM个行）

算术强度提升：~0.89 → ~4.0（理论4.5倍）
```

#### 性能数据

```
4096³矩阵：
- Kernel 4: 3544 GFLOPS
- Kernel 5: 6521 GFLOPS
- 提升：1.84倍
- vs cuBLAS: 74.3%
```

#### 三层循环结构理解

**最外层（bkIdx）**：K方向的tile（遍历K/BK次）
```
作用：把K维度分块处理
例如：K=16, BK=4 → 遍历4次
```

**中层（loadOffset）**：线程协作加载
```
作用：所有线程合作，把一个tile加载到Shared Memory
原因：线程数不够一次加载完，需要分批
```

**内层（dotIdx）**：用tile计算
```
作用：用当前Shared Memory中的tile计算部分结果
次数：BK次（遍历tile内的K维度）
```

#### 两套索引系统

**系统1：加载A/B时**
```cpp
innerRowA = threadIdx.x / BK;
innerColA = threadIdx.x % BK;

用途：决定线程在4×4加载网格中的位置
目的：协作加载tile到Shared Memory
```

**系统2：计算C时**
```cpp
threadRow = threadIdx.x / (BN/TN);
threadCol = threadIdx.x % (BN/TN);

用途：决定线程计算C的哪个2×2块
目的：外积计算
```

**关键认知**：
- 线程加载的数据 ≠ 线程使用的数据
- 这是协作加载的精髓

#### 核心收获

```
1. 2D扩展：从8×1到8×8，计算量增加8倍
2. 双向复用：A和B都缓存在寄存器，都被复用
3. 外积计算：regM ⊗ regN 生成结果矩阵
4. 算术强度大幅提升：接近4倍
5. 性能飞跃：1.84倍提升，达到cuBLAS 74%
```

---

### 2. Kernel 6：Vectorization（向量化）

#### 核心思想

**用float4一次处理4个float，减少内存事务**

```
标量：4次读取，4条指令（LDS.32）
向量：1次读取，1条指令（LDS.128）

目标：减少内存访问次数
```

#### 主要变化

**1. 加载As：float4 + transpose**
```cpp
// 新的索引（除以4）
innerRowA = threadIdx.x / (BK / 4);
innerColA = threadIdx.x % (BK / 4);

// 向量化读取
float4 tmp = reinterpret_cast<float4 *>(&A[...])[0];

// transpose写入Shared Memory
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
```

**为什么transpose？**
- 让后续读取regM时是连续的
- 可以用LDS.128向量化指令
- 代价：加载时需要分散写入
- 收益：读取更快（读的次数多）

**2. 加载Bs：float4（不transpose）**
```cpp
reinterpret_cast<float4 *>(&Bs[...])[0] =
    reinterpret_cast<float4 *>(&B[...])[0];

// 一次复制4个float
```

**3. 写回C：float4**
```cpp
// 向量化读取C
float4 tmp = reinterpret_cast<float4 *>(&C[...])[0];

// 在寄存器中计算
tmp.x = alpha * threadResults[...] + beta * tmp.x;
tmp.y = alpha * threadResults[...] + beta * tmp.y;
tmp.z = alpha * threadResults[...] + beta * tmp.z;
tmp.w = alpha * threadResults[...] + beta * tmp.w;

// 向量化写回
reinterpret_cast<float4 *>(&C[...])[0] = tmp;
```

#### float4的本质

```cpp
struct float4 {
  float x, y, z, w;
};

作用：
- 告诉编译器这4个float是一组
- 生成向量化内存指令（LDS.128, STG.128）
- 一次内存事务处理4个float
```

#### 向量化 vs 循环展开

**相似点**：
- 都是"一次做多件事"
- 都减少了某种开销

**关键区别**：
```
循环展开：
  - 减少控制流开销（分支、跳转）
  - 增加指令级并行（ILP）
  - 优化层次：指令调度

向量化：
  - 减少内存事务（带宽）
  - 一条指令处理多个数据（DLP）
  - 优化层次：数据通路

float4同时带来：
  1. 向量化内存访问（主要）
  2. 展开式的寄存器操作（次要）
```

#### 性能数据

```
4096³矩阵：
- Kernel 5: 6521 GFLOPS
- Kernel 6: 7610 GFLOPS（批量测试）
           8411 GFLOPS（单独测试）
- 提升：1.17-1.29倍
- vs cuBLAS: 86.7%（批量）/ 109.4%（单独）
```

**性能波动原因**：
- 批量测试：GPU温度升高，频率降低
- 单独测试：GPU冷却，频率更高
- 差异约10%，正常现象

#### 核心收获

```
1. 向量化：float4减少内存事务数量
2. Transpose As：让访问变连续，可用LDS.128
3. 代价可控：复杂索引 vs 快速访问
4. 性能提升：14.8%（单独测试）
5. 超越cuBLAS：特定场景下可能超过通用库
```

---

### 3. 完整性能测试（Benchmark）

#### 测试结果（4096³矩阵）

```
Kernel 0 (cuBLAS):          8774.1 GFLOPS (100%)   基准
Kernel 1 (Naive):            137.2 GFLOPS (1.6%)   
Kernel 2 (Coalescing):       978.0 GFLOPS (11.1%)  ×7.1
Kernel 3 (Shared Memory):   1331.0 GFLOPS (15.2%)  ×1.4
Kernel 4 (1D Blocktiling):  3544.3 GFLOPS (40.4%)  ×2.7
Kernel 5 (2D Blocktiling):  6520.6 GFLOPS (74.3%)  ×1.8
Kernel 6 (Vectorize):       7609.5 GFLOPS (86.7%)  ×1.2

总提升：55.46倍！
```

#### 累计提升曲线

```
Naive (137 GFLOPS, 1.6%)
  ↓ 内存合并 (最大单步提升！)
Coalescing (978 GFLOPS, 11.1%) - 7.1x
  ↓ Shared Memory
Cache Blocking (1331 GFLOPS, 15.2%) - 1.4x
  ↓ 寄存器复用 (第二大提升！)
1D Blocktiling (3544 GFLOPS, 40.4%) - 2.7x
  ↓ 双向复用
2D Blocktiling (6521 GFLOPS, 74.3%) - 1.8x
  ↓ 向量化
Vectorize (7610 GFLOPS, 86.7%) - 1.2x
```

#### 关键洞察

**1. 内存合并最重要（7.1x）**
```
GPU的第一瓶颈是内存访问
合并访问是最基础也是最重要的优化
```

**2. 寄存器复用威力大（2.7x）**
```
通过寄存器缓存，减少Shared Memory访问
带来巨大性能飞跃
```

**3. 高级优化边际递减**
```
K4→K5: 1.8x
K5→K6: 1.2x

原因：已接近硬件极限
但累计效果仍然巨大（55倍）
```

**4. 达到cuBLAS 86.7%**
```
剩余13.3%可能需要：
- Warptiling
- Double buffering
- 更细致调优
- 汇编级优化

但86.7%已经非常优秀！
```

#### 输出文件

```
my_experiments/
├── performance_summary_20260223_190100.txt
├── performance_table_20260223_190100.txt
├── performance_20260223_190100.csv
└── detailed_20260223_190100/
    ├── kernel_0_output.txt
    ├── kernel_1_output.txt
    ├── kernel_2_output.txt
    ├── kernel_3_output.txt
    ├── kernel_4_output.txt
    ├── kernel_5_output.txt
    └── kernel_6_output.txt
```

---

### 4. Kernel 7预览：Resolve Bank Conflicts

#### 核心思想

**解决Shared Memory的bank conflicts问题**

#### 什么是Bank Conflicts？

```
Shared Memory分成32个banks
规则：
  ✓ 不同线程访问不同bank → 并行，快
  ✗ 多个线程访问同一bank → 冲突，串行，慢
```

#### 主要变化

**只改了Bs的存储方式**

```cpp
// Kernel 6: 直接存储
Bs[innerRowB * BN + innerColB * 4] = B[...];

// Kernel 7: "linearize"存储（重新排列）
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
// 复杂的索引，目的是避免bank conflicts
```

#### 解决方案

```
通过重新排列Bs的存储布局（linearize）
让不同线程访问不同的banks
避免冲突 → 更快的Shared Memory访问
```

#### 预期性能

```
理论提升：5-15%（如果K6有严重bank conflicts）
代价：更复杂的索引计算
```

#### 3句话总结

```
1. 解决Shared Memory的bank conflicts
2. 通过linearize Bs存储，避免多线程访问同一bank
3. As、计算、写回都和K6一样，只改Bs存储方式
```

---

## 📊 性能数据汇总

### 各矩阵大小性能对比

| 矩阵 | Naive | Coalesce | Shared | 1D | 2D | Vector | cuBLAS |
|------|-------|----------|--------|-----|-----|--------|--------|
| 128  | 45    | 251      | 306    | 141 | 107 | 130    | 3      |
| 256  | 87    | 712      | 890    | 1089| 496 | 621    | 2272   |
| 512  | 119   | 788      | 955    | 2351| 2143| 2317   | 4688   |
| 1024 | 142   | 941      | 1195   | 2605| 4444| 5605   | 8169   |
| 2048 | 139   | 1004     | 1296   | 3783| 6821| 7884   | 8934   |
| 4096 | 137   | 978      | 1331   | 3544| 6521| 7610   | 8774   |

### 趋势分析

```
- K1-3: 小矩阵较好，大矩阵受限
- K4-6: 大矩阵性能优异
- cuBLAS: 所有大小都强
```

---

## 💡 核心收获

### 技术层面

**1. 优化层次**
```
内存合并 (7.1x)
  ↓
Shared Memory (1.4x)
  ↓
1D Blocktiling (2.7x)
  ↓
2D Blocktiling (1.8x)
  ↓
Vectorization (1.2x)

总计：55.46倍
```

**2. 关键概念**
```
✓ 2D blocktiling：双向数据复用
✓ 外积计算：regM ⊗ regN
✓ 向量化：float4减少内存事务
✓ Transpose：让访问连续
✓ Bank conflicts：Shared Memory的访问冲突
```

**3. 性能优化方法论**
```
1. 找瓶颈（内存 vs 计算）
2. 分层优化（全局→Shared→寄存器）
3. 数据复用（减少访问次数）
4. 向量化（减少指令数）
5. 避免冲突（bank conflicts）
```

### 学习方法

**1. 三层循环的理解方法**
```
最外层（bkIdx）：数据分块（K方向）
中层（loadOffset）：线程协作加载
内层（dotIdx）：用当前数据计算
```

**2. 索引映射的理解**
```
加载索引 ≠ 计算索引
协作加载是关键
```

**3. 概念联系**
```
向量化 ≈ 特殊的循环展开
但优化层次不同
```

---

## 🤔 项目诚信讨论

### 重要认识

**不能写"从零实现"**
- 这是别人的开源项目
- 源码是他人的
- 诚实是最重要的

### 正确的表述

**学习导向（推荐）**：
```
CUDA GEMM性能优化深度学习

· 深入研究siboehm的GEMM优化教程
· 通过复现和分析6个优化层次，掌握GPU优化技术
· 手写关键代码加深理解，完成性能测试
· 达到cuBLAS 86.7%性能（7610 vs 8774 GFLOPS）
· 撰写详细学习笔记和性能分析报告

参考：github.com/siboehm/SGEMM_CUDA
```

### 如何增加原创价值

**1. 深度分析（最简单）**
```
- 写中文教程
- 做完整性能分析
- 画图表和可视化
```

**2. 改进扩展（中等）**
```
- 支持非方阵
- 优化小矩阵
- 添加自动调参
```

**3. 完全原创（最难）**
```
- 参考思路
- 自己重写
- 2-3周时间
```

### 面试时的诚实回答

```
"不是完全从零写的。我跟着Simon Boehm的教程深入学习，
手写了关键部分确保理解，做了完整的性能测试和分析。
通过这个项目掌握了CUDA优化的核心原理。"
```

---

## 📋 下一步计划

### Day 8（2/24，明天）

**上午（2-3小时）**：
```
□ 快速浏览Kernel 7-10
  - 每个30-45分钟
  - 了解优化思想
  - 不深入实现
```

**下午（2-3小时）**：
```
□ 整理所有kernel笔记
□ 制作性能对比表
□ 开始写README框架
```

### Day 9-10（2/25-26）

```
□ 完善README
□ 写技术总结（800-1000字）
□ 整理学习笔记
□ 画性能图表
```

### Day 11（2/28）

```
□ 最后检查
□ 宣布项目完成
□ 如有时间：美化扩展
```

---

## 🎯 当前状态

```
✅ 核心kernel（1-6）深入理解
✅ 性能达到cuBLAS 86.7%
✅ 完整benchmark测试完成
✅ 实现55.46倍性能提升
✅ 有完整的实验数据

待完成：
□ 剩余kernel快速浏览（2-3个）
□ 项目整理和文档
□ 性能图表
```

---

## 🏆 今天的成就

```
学习时间：5-6小时
完成内容：
  ✓ Kernel 5深入理解（2D Blocktiling）
  ✓ Kernel 6深入理解（Vectorization）
  ✓ Kernel 7预览（Bank Conflicts）
  ✓ 完整benchmark测试（kernel 0-6）
  ✓ 性能数据收集和分析
  ✓ 项目诚信讨论

性能成果：
  ✓ 达到cuBLAS 86.7%
  ✓ 实现55倍提升
  ✓ 可写进简历的项目

这是非常充实的一天！🎉
```

---

## 📝 备注

- Kernel 5的三层循环理解花了较多时间，但理解深刻
- 向量化和循环展开的关系是个有趣的发现
- 性能波动（单独vs批量测试）是正常现象
- 诚信讨论很重要，明确了简历表述方式
- 时间管理良好，进度符合预期

---

## 💭 反思

**做得好的**：
- 深入理解核心概念，不急于求成
- 多次询问直到真正理解
- 关注诚信问题，有职业素养
- 有完整的实验数据支撑

**可以改进的**：
- 可以更早做benchmark测试
- 可以更早考虑项目整理

**总体评价**：
优秀！既有深度又有广度，既理解原理又有实践数据。

---

**Day 7完成！明天继续加油！** 💪
