cat > my_notes/day5.md << 'EOF'
# Day 5: 手写Kernel 3 + 性能对比 (2026-02-19)

## ✅ 完成任务
- [x] 手写Kernel 3代码
- [x] 纠正错误并理解原因
- [x] 收集性能数据
- [x] 对比Kernel 2 vs Kernel 3
- [x] 完全掌握Shared Memory优化

---

## 🔧 手写Kernel 3的错误与修正

### 错误1：Shared Memory缺少类型
```cpp
// ❌ 错误
__shared__ As[BLOCKSIZE * BLOCKSIZE];

// ✅ 正确
__shared__ float As[BLOCKSIZE * BLOCKSIZE];
```

### 错误2：B指针移动方向错误
```cpp
// ❌ 错误
B += BLOCKSIZE * K;

// ✅ 正确
B += BLOCKSIZE * N;  // 向下移动，用N不是K

原因：B[i * N + j]，移动行要乘N
```

### 错误3：点积计算索引错误（核心）
```cpp
// ❌ 错误：没用上循环变量j
for (int j = 0; j < BLOCKSIZE; ++j) {
    tmp += As[threadRow * BLOCKSIZE + threadCol] * 
           Bs[threadRow * BLOCKSIZE + threadCol];
}

// ✅ 正确：A的行 × B的列
for (int j = 0; j < BLOCKSIZE; ++j) {
    tmp += As[threadRow * BLOCKSIZE + j] *      // A的行，列变化
           Bs[j * BLOCKSIZE + threadCol];        // B的列，行变化
}

理解：C[i][j] = A的第i行 · B的第j列
```

### 错误4：缺少第二个同步
```cpp
// 必须有两次同步
for (int i = 0; i < K; i += BLOCKSIZE) {
    // 加载
    __syncthreads();  // 第1次：等加载完
    // 计算
    __syncthreads();  // 第2次：等计算完，防止覆盖
}
```

---

## 📊 性能对比结果

### 实测数据（4096×4096）

| Kernel | 时间(ms) | GFLOPS | vs K2 | vs Naive |
|--------|---------|--------|-------|----------|
| Kernel 1 (Naive) | ~980 | 140 | 0.15x | 1.0x |
| Kernel 2 (Coalescing) | 145.72 | 943.2 | 1.0x | 6.7x |
| **Kernel 3 (Shared Mem)** | **107.27** | **1281.2** | **1.36x** | **9.15x** ✨ |

### 关键发现
- 🚀 vs Kernel 2：**提升36%** (1.36倍)
- 📈 vs Naive：**累计提升9.15倍**
- ⏱️ 时间：145.72ms → 107.27ms（减少26%）
- 🎯 达到cuBLAS的约17%

---

## 💡 核心理解巩固

### Shared Memory的本质
```
作用：片上缓存，比全局内存快20-100倍
使用：显式加载、显式同步
效果：数据加载1次，复用32次
```

### 为什么提升是1.36x？

**Kernel 2已经很优化**：
- 合并访问让带宽利用率高
- 继续优化内存的收益递减

**新瓶颈出现**：
- 从memory-bound转向compute-bound
- 计算延迟开始显现
- Shared Memory用量影响Occupancy

**同步开销**：
- K=4096, BLOCKSIZE=32 → 128次循环
- 每次2个同步 → 256次同步
- 每次都有等待开销

### 点积计算的索引（重点）
```cpp
线程(threadRow=2, threadCol=5)计算C[2][5]：

需要：A的第2行 · B的第5列

for (int j = 0; j < 32; ++j) {
    tmp += As[2*32 + j] * Bs[j*32 + 5];
    //        ↑行固定     ↑列固定
    //        列变化       行变化
}

记忆：横着读A，竖着读B
```

---

## 🔍 关键问题回顾

### Q1: 为什么B移动是 `+= BLOCKSIZE * N` 而不是 `K`？

**A**: B矩阵的内存布局是 `B[row * N + col]`
- 向下移动BLOCKSIZE行 = row增加BLOCKSIZE
- 偏移量 = BLOCKSIZE * N

### Q2: 为什么需要两次`__syncthreads()`？

**A**: 
- 第1次：确保所有线程加载完数据，才能开始计算
- 第2次：确保所有线程计算完，才能加载下一个tile（避免覆盖）

### Q3: 为什么"同时运行"不等于"同时完成"？

**A**:
- 硬件并行度有限（需要调度）
- 内存访问延迟不同（cache命中率）
- Warp执行时间不同（指令流水线）

→ 需要`__syncthreads()`强制对齐

---

## 🎯 完全掌握的技能

- ✅ 能手写完整的Kernel 3
- ✅ 理解Shared Memory的声明和使用
- ✅ 理解Tiling的概念和实现
- ✅ 理解同步的必要性和时机
- ✅ 理解点积计算的索引
- ✅ 能分析性能提升的原因和限制
- ✅ 理解指针预移动的机制
- ✅ 理解宏观（block）vs 微观（thread）

---

## 📈 性能进化路线图
```
Naive (140 GFLOPS, 1.8% of cuBLAS)
    ↓ 内存合并访问 (6.7x)
Coalescing (943 GFLOPS, 12.3% of cuBLAS)
    ↓ Shared Memory + Tiling (1.36x)
Shared Memory (1281 GFLOPS, 16.7% of cuBLAS) ← 我们在这里 ✅
    ↓ 1D Blocktiling
~1800 GFLOPS (~23%)
    ↓ 2D Blocktiling + Vectorization
~4000 GFLOPS (~52%)
    ↓ Warptiling + 高级优化
~6000+ GFLOPS (~78%)
    ↓
cuBLAS (7686 GFLOPS, 100%)
```

---

## ❓ 遗留问题

- [ ] 如何用Nsight Compute详细分析？（WSL2限制，可用Windows）
- [ ] Bank Conflict在这个kernel中是否存在？
- [ ] Occupancy具体降低了多少？
- [ ] 如何进一步优化到2000+ GFLOPS？

---

## 🎯 明天计划 (Day 6)

### 主要任务
- [ ] 学习Kernel 4（1D Blocktiling）
- [ ] 理解为什么每个线程计算多个元素
- [ ] 理解寄存器复用的概念
- [ ] 运行并对比性能
- [ ] 预期：突破1500 GFLOPS

### 学习重点
- 1D Blocktiling的原理
- 寄存器 vs Shared Memory
- 如何增加每线程工作量
- 为什么能提升性能

---

## 📚 今日金句
```
"理解原理比工具重要"
"每次优化都有递减效应"
"同时运行 ≠ 同时完成"
"横着读A，竖着读B"
```

---

## 💭 个人反思

今天最大的收获是通过手写Kernel 3，把之前的理解从"看懂"提升到"能写"。特别是点积计算的索引，通过犯错和纠正，理解得更深了。

虽然没能用上Nsight Compute的详细分析（WSL2限制），但通过性能数据对比和理论分析，已经完全掌握了Shared Memory优化的核心思想。

性能提升1.36倍，累计9.15倍，这个过程让我看到了GPU优化的逐步推进过程。每一步都建立在前一步的基础上。

准备好学习更高级的优化技术了！💪

---

**学习时长**：约2小时  
**完成日期**：2026-02-19  
**下次学习**：2026-02-20 (Day 6 - 1D Blocktiling)
