# Day 9 学习笔记（2026-02-25）

## 📅 学习概况

- **日期**：2026年2月25日（周三）
- **学习时间**：约2-3小时
- **主要内容**：Kernel 7/8（Bank Conflicts），Kernel 9（Warptiling深入）
- **状态**：突破性进展！Kernel 9超越cuBLAS

---

## 📚 学习内容

### 快速回顾：Kernel 7和8

#### 作者的说明

```
"I skipped kernels 7 and 8, which I wrote while figuring out 
how to best eliminate shared memory bank conflicts. They 
eliminate the conflicts but were overall still slower, so I 
won't cover them here."

翻译：
作者跳过了kernel 7和8，这是他在研究如何最好地消除
shared memory bank conflicts时写的。虽然消除了冲突，
但整体上反而更慢，所以不详细讨论。
```

#### 性能验证

```
Kernel 6: 7610 GFLOPS（批量测试基准）
Kernel 7: 7315 GFLOPS（-4.0%）
Kernel 8: 7434 GFLOPS（-2.3%）

确实更慢！验证了作者的说法。
```

#### 核心收获

```
1. Bank Conflicts概念
   - 多个线程同时访问同一个bank
   - 导致串行执行
   - 定义：同一时刻 + 多个线程 + 同一bank

2. 两种解决方案
   - Linearize（K7）：复杂映射，完全消除冲突
   - Padding（K8）：简单padding，冲突分散

3. 重要教训
   - 理论最优 ≠ 实践最优
   - 索引计算开销可能大于冲突代价
   - 需要实测验证
```

---

## 🎯 Kernel 9：Warptiling（重点！）

### 核心概念

```
Kernel 9 = Warp级别的协作 + Autotuning

引入新的层次：
  Block → Warptile → Thread
  
关键：256个线程通过循环迭代所有warptile
```

---

### 参数体系

#### 模板参数

```cpp
BM, BN, BK：Blocktile大小（例如：128×128×8）
TM, TN：每线程计算的元素（例如：8×8）
K9_NUM_THREADS = 256：固定线程数
```

#### 计算得出的参数

```cpp
WM = TM * 16  // Warptile的M维度（例如：128）
WN = TN * 16  // Warptile的N维度（例如：128）

WMITER = BM / WM  // M方向warptile数量（例如：1）
WNITER = BN / WN  // N方向warptile数量（例如：1）
```

**为什么是TM * 16？**
```
关键：让256个线程形成合理的网格

WN / TN = (TN * 16) / TN = 16
threadCol = threadIdx.x % 16  // 16列
threadRow = threadIdx.x / 16  // 16行

256个线程 → 16×16网格
```

---

### 线程组织（关键理解！）

#### 正确理解

```
一个Block有256个线程（固定）

这256个线程不是分成4份！
而是：所有256个线程一起计算每个warptile

通过循环（wmIdx, wnIdx），256个线程依次计算所有warptile
```

#### 执行时间线

```
时刻1 (wmIdx=0, wnIdx=0)：
  所有256个线程一起计算Warptile 0,0

时刻2 (wmIdx=0, wnIdx=1)：
  所有256个线程一起计算Warptile 0,1

时刻3 (wmIdx=1, wnIdx=0)：
  所有256个线程一起计算Warptile 1,0

时刻4 (wmIdx=1, wnIdx=1)：
  所有256个线程一起计算Warptile 1,1

这是时间上的迭代，不是空间上的划分！
```

#### 单个Warptile的工作分配

```
Warptile 0,0 (64×64)

256个线程形成16×16网格
每个线程负责4×4块
总共：16×16 × 4×4 = 64×64 ✓

Thread 0:   负责位置(0-3,   0-3)
Thread 1:   负责位置(0-3,   4-7)
...
Thread 255: 负责位置(60-63, 60-63)
```

---

### Shared Memory的共享机制

#### 关键理解

```
所有warptile共享同一个Shared Memory！

不是：
  ❌ 每个warptile加载一次Shared Memory
  ❌ warptile之间重新加载

而是：
  ✓ 一次加载整个Blocktile（BM×BK的As和BK×BN的Bs）
  ✓ 所有warptile通过不同偏移访问
  ✓ 每个bkIdx迭代才重新加载（覆盖旧数据）
```

#### 访问方式

```cpp
// 读取regM（有wmIdx偏移）
regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
                           └──────┬──────┘
                              warptile偏移

// 读取regN（有wnIdx偏移）
regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
                           └──────┬──────┘
                              warptile偏移

通过偏移访问同一个Shared Memory的不同区域
```

---

### Stride加载详解

#### rowStrideA的计算

```cpp
rowStrideA = (K9_NUM_THREADS * 4) / BK
           = (256 * 4) / 8
           = 128

含义：一次循环迭代能加载多少行
```

#### 为什么需要循环？

```
BM可能大于rowStrideA（autotuning时）

例如：
  BM=128, rowStrideA=128 → 1次循环
  BM=256, rowStrideA=128 → 2次循环

循环让代码适应不同的tile大小
```

#### 加载A的详细流程

**线程索引计算**：
```cpp
innerRowA = threadIdx.x / (BK / 4)  // 256个线程
innerColA = threadIdx.x % (BK / 4)  // 形成128×2网格

BK=8时：
  innerRowA = threadIdx.x / 2  // 0-127
  innerColA = threadIdx.x % 2  // 0-1
```

**256个线程的映射**：
```
          innerColA
            0      1
          ┌────┬────┐
        0 │ T0 │ T1 │
        1 │ T2 │ T3 │
        2 │ T4 │ T5 │
inner   . │ .  │ .  │
Row   127 │T254│T255│
A         └────┴────┘

128行 × 2列 = 256个位置
```

**每个线程加载什么**：
```
Thread 0 (innerRowA=0, innerColA=0):
  从A读取：A[0][0-3]（float4）
  
Thread 1 (innerRowA=0, innerColA=1):
  从A读取：A[0][4-7]（float4）
  
...

256个线程协作加载 128行×8列
```

**Transpose写入As**：
```cpp
As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;

规律：A[row][col] → As[col * BM + row]
行列互换！
```

**为什么Transpose？**
```
后续计算时：
  regM[i] = As[dotIdx * BM + ...]
  
Transpose后访问更连续，更有利于向量化和cache
```

---

### 计算流程（三层循环）

#### 外层：K方向的Blocktile迭代

```cpp
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // 加载整个Blocktile到Shared Memory
  加载As[BM × BK]
  加载Bs[BK × BN]
  __syncthreads();
  
  // 计算所有warptile
  ...
  
  // 移动到下一个K tile
  A += BK;
  B += BK * N;
  __syncthreads();
}
```

#### 中层：Warptile迭代

```cpp
for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
  for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
    // 所有256个线程一起计算当前warptile
    // 通过wmIdx和wnIdx偏移访问Shared Memory
    ...
  }
}
```

#### 内层：K方向元素遍历 + 外积计算

```cpp
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  // 读取regM和regN
  for (uint i = 0; i < TM; ++i) {
    regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
  }
  for (uint i = 0; i < TN; ++i) {
    regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
  }
  
  // 外积计算
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      threadResults[(wmIdx*TM + resIdxM) * (WNITER*TN) + 
                    wnIdx*TN + resIdxN] += 
          regM[resIdxM] * regN[resIdxN];
    }
  }
}
```

---

### threadResults数组的组织

#### 大小

```cpp
float threadResults[WMITER * WNITER * TM * TN];

例如：WMITER=2, WNITER=2, TM=4, TN=4
     = 2 * 2 * 4 * 4 = 64个元素
```

#### 逻辑组织（对Thread 0）

```
逻辑上是 (WMITER*TM) × (WNITER*TN) 矩阵 = 8×8

     wnIdx=0  |  wnIdx=1
     (4列)    |  (4列)
   ┌──────────┼──────────┐
   │[0][1]... │[4][5]... │ ← wmIdx=0 (4行)
w  │[8][9]... │[12]...   │
m  │[16]...   │[20]...   │
I  │[24]...   │[28]...   │
d  ├──────────┼──────────┤
x  │[32]...   │[36]...   │ ← wmIdx=1 (4行)
   │[40]...   │[44]...   │
   │[48]...   │[52]...   │
   │[56]...   │[60]...   │
   └──────────┴──────────┘

每个4×4块对应一个warptile的结果
```

---

### 写回结果

```cpp
for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
  for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
    // C_interim指向当前warptile的起始位置
    float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
    
    // 每个线程写回TM×TN个结果
    // 使用float4向量化
    ...
  }
}
```

---

## 📊 性能测试结果

### Kernel 9的突破性表现

```
4096³矩阵：
  Kernel 6: 7610 GFLOPS
  Kernel 7: 7315 GFLOPS
  Kernel 8: 7434 GFLOPS
  Kernel 9: 9123 GFLOPS ← 🚀 大幅提升！
  
  cuBLAS:   8774 GFLOPS

Kernel 9超越cuBLAS：9123 / 8774 = 104%！
```

### 完整性能表（4096³）

| Kernel | GFLOPS | vs K6 | vs cuBLAS | 说明 |
|--------|--------|-------|-----------|------|
| K0 (cuBLAS) | 8774 | - | 100% | 基准 |
| K1 (Naive) | 137 | 1.8% | 1.6% | 起点 |
| K2 (Coalesce) | 978 | 12.9% | 11.1% | ×7.1 |
| K3 (Shared) | 1331 | 17.5% | 15.2% | ×1.4 |
| K4 (1D Block) | 3544 | 46.6% | 40.4% | ×2.7 |
| K5 (2D Block) | 6521 | 85.7% | 74.3% | ×1.8 |
| K6 (Vectorize) | 7610 | 100% | 86.7% | ×1.2 |
| K7 (Linearize) | 7315 | 96.1% | 83.4% | 实验性 |
| K8 (Padding) | 7434 | 97.7% | 84.7% | 实验性 |
| **K9 (Warptile)** | **9123** | **119.9%** | **104%** | **突破！** |

**总提升**：66.5倍（从Naive到K9）

---

### 各矩阵大小性能

| 矩阵 | K6 | K7 | K8 | K9 | K9提升 |
|------|-----|-----|-----|-----|--------|
| 128³ | 130 | 126 | 127 | 162 | +25% |
| 256³ | 621 | 574 | 574 | 750 | +21% |
| 512³ | 2317 | 1740 | 2431 | **3260** | **+34%** |
| 1024³ | 5605 | 4910 | 4711 | 5948 | +6% |
| 2048³ | 7884 | 6572 | 6636 | 8817 | +12% |
| 4096³ | 7610 | 7315 | 7434 | **9123** | **+20%** |

**观察**：
- 512³矩阵上提升最大（34%）
- 大矩阵稳定提升10-20%
- 4096³达到最高性能

---

## 💡 为什么Kernel 9这么快？

### 1. Warptiling带来的优势

```
数据局部性：
  - Warp内32个线程访问相邻内存
  - Cache命中率更高
  - 减少Shared Memory访问冲突

计算组织：
  - 更好的寄存器复用
  - 减少冗余的Shared Memory访问
  - 编译器优化空间更大
```

---

### 2. 固定线程数

```
K9_NUM_THREADS = 256（固定）

优势：
  - 编译器可以做更激进的优化
  - 寄存器分配确定
  - 指令调度更好
  - Occupancy可预测

对比：
  之前的kernel线程数可能变化
  编译器需要更保守
```

---

### 3. Stride循环的灵活性

```
通过循环加载适应不同tile大小：
  - 参数可调（autotuning）
  - 不同GPU可以用不同配置
  - 这组参数可能经过优化
```

---

### 4. 延续了之前的优化

```
✓ 向量化（float4）- 来自K6
✓ Transpose As - 来自K6
✓ Shared Memory - 来自K3
✓ 内存合并 - 来自K2

Warptiling是在坚实基础上的进一步优化
```

---

## 🎓 核心学习收获

### 技术层面

**1. Warptiling核心概念**
```
- 引入Warp级别的协作
- 256个线程通过循环迭代warptile
- 通过偏移访问同一个Shared Memory
```

**2. 线程组织的深入理解**
```
- 所有线程一起计算每个warptile（时间上）
- 不是warptile间分配线程（空间上）
- SIMD执行的必然要求
```

**3. Shared Memory共享机制**
```
- 一次加载，多次使用（在一个bkIdx内）
- warptile通过偏移访问不同区域
- 每个bkIdx重新加载
```

**4. Stride加载的精髓**
```
- 和Kernel 5的stride本质相同
- 循环适应不同参数
- 支持autotuning
```

---

### 方法论

**1. 性能优化不是线性的**
```
K7/K8虽然解决了bank conflicts
但整体性能反而下降
→ 理论最优 ≠ 实践最优
```

**2. 实测验证的重要性**
```
必须运行测试才能知道效果
理论分析有局限
```

**3. 层次化的思维**
```
GPU编程的多层次：
  Block → Warp → Thread → Register
  
每一层都有优化空间
```

**4. 学习真实的工程过程**
```
包括成功的尝试（K9）
也包括失败的尝试（K7/K8）
这是真实的研发
```

---

## 🤔 关键问题的解答

### Q1: rowStride是Kernel 5的stride吗？

```
A: 是的！本质完全相同

区别只是：
  - Kernel 5: 每线程加载1个float
  - Kernel 9: 每线程加载1个float4（×4）

所以公式有×4，但概念一样
```

---

### Q2: 计算流程是什么？

```
A: 不是分批加载warptile

正确流程：
  1. 加载整个Blocktile到Shared Memory（一次性）
  2. 所有warptile共享这个Shared Memory
  3. 通过循环和偏移访问不同区域
  4. 下一个bkIdx才重新加载（覆盖）

关键：warptile是逻辑划分，不是物理加载
```

---

### Q3: 所有线程是一起计算warptile的吗？

```
A: 是的！完全正确

所有256个线程：
  - 先一起计算Warptile 0,0
  - 再一起计算Warptile 0,1
  - 然后Warptile 1,0
  - 最后Warptile 1,1

这是GPU SIMD执行的要求
所有线程必须同步
```

---

## 📋 项目状态

### 已完成

```
✅ 核心Kernel 1-6深入理解
✅ Kernel 7/8概念理解（虽然性能不佳）
✅ Kernel 9深入学习（Warptiling）
✅ 完整性能测试
✅ 超越cuBLAS（104%）

达到：9123 GFLOPS
提升：66.5倍（从Naive）
```

---

### 待完成

```
□ 快速浏览Kernel 10+（如果有）
□ 整理所有kernel笔记
□ 制作性能图表
□ 撰写README
□ 写技术总结
```

---

## 📝 今日总结

### 学习内容

```
- Kernel 7（Linearize Bank Conflicts）
- Kernel 8（Padding Bank Conflicts）
- Kernel 9（Warptiling）深入学习

学习时间：6-7小时
理解深度：深入
学习质量：非常高
```

---

### 关键成果

```
✓ 理解了Warptiling的核心概念
✓ 掌握了线程组织的正确理解
✓ 深入理解Shared Memory共享机制
✓ 完整理解加载流程（包括Transpose）
✓ 验证了性能突破（超越cuBLAS）
```

---

### 重要认知

```
1. Bank Conflicts不是主要瓶颈
   - K7/K8虽然解决但反而慢
   - 其他因素更重要

2. Warptiling是重大优化
   - 22.7%的性能提升
   - 超越cuBLAS
   - 证明手写kernel的价值

3. 理解 > 实现
   - 深入理解比写代码更重要
   - 通过多次提问和讨论
   - 建立了正确的心智模型

4. 实测数据最重要
   - 理论分析有局限
   - 必须验证才知道
```

---

### 学习方法

```
✓ 主动提问，澄清疑惑
✓ 通过图示加深理解
✓ 验证性能数据
✓ 建立正确的概念模型
✓ 不满足于表面理解

特别好的地方：
  - 发现了线程组织的误解并纠正
  - 深入追问Shared Memory共享机制
  - 要求画图详细讲解
```

---

## 💭 反思

**做得好的**：
- 深入理解核心概念，不急于求成
- 发现理解误区并及时纠正
- 通过实测验证性能
- 完整的学习记录

**明天计划**：
- 快速浏览剩余kernel（如果有）
- 开始项目整理
- 性能图表制作
- README撰写

---

## 🏆 Day 9成就

```
学习时间：2-3小时
完成内容：3个kernel深入理解
性能突破：超越cuBLAS 4%
理解深度：★★★★★
学习质量：★★★★★

这是收获最大的一天！🎉
```

---

**Day 9完成！明天继续！** 💪

---

## 附录：关键公式

### Warptile大小

```cpp
WM = TM * 16
WN = TN * 16
```

### 迭代次数

```cpp
WMITER = BM / WM
WNITER = BN / WN
```

### Stride

```cpp
rowStrideA = (NUM_THREADS * 4) / BK
rowStrideB = NUM_THREADS / (BN / 4)
```

### 索引映射

```cpp
// 线程在warptile中的位置
threadCol = threadIdx.x % (WN / TN)
threadRow = threadIdx.x / (WN / TN)

// 加载时的位置
innerRowA = threadIdx.x / (BK / 4)
innerColA = threadIdx.x % (BK / 4)

// Transpose
A[row][col] → As[col * BM + row]

// 读取regM
regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i]

// threadResults索引
index = (wmIdx*TM + resIdxM) * (WNITER*TN) + wnIdx*TN + resIdxN
```
