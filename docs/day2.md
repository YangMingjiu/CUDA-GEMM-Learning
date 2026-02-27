# Day 2: Kernel 2深入分析与内存合并访问 (2026-02-14)

## ✅ 完成任务
- [x] 运行Kernel 0-2并记录性能
- [x] 详细分析Kernel 2源码
- [x] 理解索引计算公式（除法和取模）
- [x] 对比Naive vs Coalescing的内存访问模式
- [x] 理解1D block vs 2D block的配置
- [x] 完整工作流程举例验证

## 📊 性能数据汇总

### RTX 3070 Laptop - 4096×4096矩阵

| Kernel | 名称 | 时间(ms) | GFLOPS | vs cuBLAS | vs Naive |
|--------|------|---------|---------|-----------|----------|
| 0 | cuBLAS (baseline) | 17.88 | 7686.3 | 100% | 54.6x |
| 1 | Naive | 976.99 | 140.7 | 1.8% | 1.0x |
| 2 | **Global Mem Coalescing** | 139.95 | **982.1** | **12.8%** | **6.98x** ✨ |

### 所有矩阵大小的对比

| 矩阵大小 | Naive (GFLOPS) | Coalescing (GFLOPS) | 提升倍数 |
|---------|---------------|---------------------|---------|
| 128³    | 45.3          | 246.1               | 5.43x   |
| 256³    | 85.6          | 651.9               | 7.62x   |
| 512³    | 107.1         | 735.2               | 6.87x   |
| 1024³   | 140.4         | 868.0               | 6.18x   |
| 2048³   | 143.9         | 992.5               | 6.90x   |
| 4096³   | 140.7         | 982.1               | 6.98x   |

### 关键发现
- 🚀 **仅改变线程映射方式，性能提升约7倍！**
- 📈 从占cuBLAS的1.8%提升到12.8%
- ⏱️ 计算时间从977ms降到140ms
- 💡 中大矩阵提升稳定在7倍左右

---

## 💻 Kernel 2源码分析

### 完整代码
```cpp
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  // 关键：使用1D线程块，通过除法和取模映射到2D矩阵
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}
```

### 启动配置
```cpp
// gridDim保持不变
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));

// 关键：blockDim从2D变成1D
dim3 blockDim(32 * 32);  // 1024个线程（1D组织）

sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

---

## 🔍 索引计算详解

### 核心公式
```cpp
cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```

### 设计思想
1. **使用1D线程块**：`blockDim(1024)` 即 `blockDim.x = 1024`
2. **除法决定行**：`threadIdx.x / 32` → 每32个线程换一行
3. **取模决定列**：`threadIdx.x % 32` → 0到31循环

### 映射规律（BLOCKSIZE=32）
```
threadIdx.x    行偏移(÷32)  列偏移(%32)  → 计算的C元素
    0              0            0        → C[blockRow+0][blockCol+0]
    1              0            1        → C[blockRow+0][blockCol+1]
   31              0           31        → C[blockRow+0][blockCol+31]
   32              1            0        → C[blockRow+1][blockCol+0] ← 换行
   33              1            1        → C[blockRow+1][blockCol+1]
   ...
  1023            31           31        → C[blockRow+31][blockCol+31]
```

**关键洞察**：
- 连续32个线程（一个warp）→ C的同一行的连续32列
- 通过除法和取模，把1D线程ID映射到2D矩阵位置

### 具体例子：Block(0,0)
```cpp
blockIdx.x = 0, blockIdx.y = 0
BLOCKSIZE = 32

线程0:   cRow = 0×32 + (0/32)   = 0,  cCol = 0×32 + (0%32)   = 0   → C[0][0]
线程1:   cRow = 0×32 + (1/32)   = 0,  cCol = 0×32 + (1%32)   = 1   → C[0][1]
线程31:  cRow = 0×32 + (31/32)  = 0,  cCol = 0×32 + (31%32)  = 31  → C[0][31]
线程32:  cRow = 0×32 + (32/32)  = 1,  cCol = 0×32 + (32%32)  = 0   → C[1][0]
线程1023:cRow = 0×32 + (1023/32)= 31, cCol = 0×32 + (1023%32)= 31  → C[31][31]
```

**Block(0,0)负责C[0-31][0-31]这个32×32区域** ✅

---

## 🆚 Naive vs Coalescing 内存访问对比

### 线程组织差异

**Kernel 1 (Naive) - 按列组织**：
```
使用2D线程块：dim3 blockDim(32, 32)

同一warp的32个线程：
  线程(0-31, 0) → C[0-31][0] (同一列的连续32行)
  
可视化：
     ↓ warp方向（竖着）
C = [▓ □ □ □]
    [▓ □ □ □]
    [▓ □ □ □]
    [▓ □ □ □]
```

**Kernel 2 (Coalescing) - 按行组织**：
```
使用1D线程块：dim3 blockDim(1024)

同一warp的32个线程：
  threadIdx.x = 0-31 → C[0][0-31] (同一行的连续32列)
  
可视化：
     → → → → warp方向（横着）
C = [▓ ▓ ▓ ▓]
    [□ □ □ □]
    [□ □ □ □]
    [□ □ □ □]
```

---

### 内存访问模式对比（32线程Warp，某次循环i）

#### Kernel 1 (Naive)
```
同一warp：线程(0-31, 0) 计算 C[0-31][0]

访问A矩阵:
  线程(0,0):  A[0×K + i]  = A[i]
  线程(1,0):  A[1×K + i]  = A[K + i]     ← 跳跃K个元素
  线程(2,0):  A[2×K + i]  = A[2K + i]    ← 再跳K个
  ...
  线程(31,0): A[31×K + i] = A[31K + i]

  内存地址：[i, K+i, 2K+i, ..., 31K+i]
  ❌ 完全不连续！需要32个内存事务

访问B矩阵:
  所有32个线程：B[i×N + 0] = B[i×N]
  
  内存地址：[i×N, i×N, i×N, ..., i×N]
  ⚠️ 广播访问，1个内存事务
```

#### Kernel 2 (Coalescing)
```
同一warp：threadIdx.x = 0-31 计算 C[0][0-31]

访问A矩阵:
  所有32个线程：A[0×K + i] = A[i]
  
  内存地址：[i, i, i, ..., i]
  ⚠️ 广播访问，1个内存事务

访问B矩阵:
  线程0:  B[i×N + 0]  = B[i×N]
  线程1:  B[i×N + 1]  = B[i×N + 1]
  线程2:  B[i×N + 2]  = B[i×N + 2]
  ...
  线程31: B[i×N + 31] = B[i×N + 31]

  内存地址：[i×N, i×N+1, i×N+2, ..., i×N+31]
  ✅ 完全连续！1个合并的内存事务（128字节）
```

---

### 内存事务数总结

| Kernel | 访问A | 访问B | 总事务数/迭代 |
|--------|-------|-------|-------------|
| Naive | 32个（跳跃）❌ | 1个（广播）⚠️ | **33个** |
| Coalescing | 1个（广播）⚠️ | 1个（合并）✅ | **2个** |

**理论加速比**: 33 / 2 = **16.5倍**  
**实际加速比**: **6.98倍**

**差距原因**：其他瓶颈（计算延迟、cache、occupancy等）开始显现

---

## 💡 核心理解

### 一句话总结

> **Kernel 1按列组织线程，访问A时跳跃；  
> Kernel 2按行组织线程，访问B时连续。**

### 从列到行的转换

**整体计算层面**（算法需求）：
- 每个线程需要 A 的一行 + B 的一列
- 通过K次循环逐个元素累加

**单次循环层面**（内存访问）：
- Naive：32个线程访问A的不同行的同一列（跳跃）+ B的同一个值（广播）
- Coalescing：32个线程访问A的同一个值（广播）+ B的同一行的连续元素（合并）

**关键洞察**：
```
虽然每个线程最终需要B的一列（竖着的）
但某次迭代时，32个线程各取自己需要的那一列的一个元素
这32个元素恰好组成B的某一行（横着的）
→ 竖着的需求，通过横着的访问实现！
```

---

## 🎯 1D Block vs 2D Block

### 为什么要改成1D？

**Kernel 1 (2D block)**:
```cpp
dim3 blockDim(32, 32);  // 32×32 = 1024个线程

线程编号：threadIdx.x (0-31), threadIdx.y (0-31)
索引计算：
  x = blockIdx.x * 32 + threadIdx.x
  y = blockIdx.y * 32 + threadIdx.y
```

**Kernel 2 (1D block)**:
```cpp
dim3 blockDim(1024);  // 1024个线程（1D）

线程编号：threadIdx.x (0-1023)
索引计算：
  cRow = blockIdx.x * 32 + (threadIdx.x / 32)
  cCol = blockIdx.y * 32 + (threadIdx.x % 32)
```

### 优势

1. **灵活的索引映射**：通过除法和取模可以自由控制行列对应关系
2. **Warp对齐**：连续32个threadIdx.x自然组成一个warp
3. **内存访问可控**：更容易设计出合并访问模式
4. **为后续优化预留空间**：Tiling等高级优化更方便

### 线程数量不变
```
2D: 32 × 32 = 1024个线程
1D: 1024个线程

总数相同，只是组织方式不同！
```

---

## 🧪 完整工作流程示例

### 设定（简化例子）
```
矩阵：A(4×4), B(4×4), C(4×4)
BLOCKSIZE = 2
Block(0,0)的4个线程
```

### Kernel 2的执行过程

**线程分配**：
```
线程0: C[0][0]
线程1: C[0][1]  ← 同一行
线程2: C[1][0]
线程3: C[1][1]  ← 同一行
```

**循环迭代i=0**：
```
线程0: A[0][0] × B[0][0] = A[0] × B[0]
线程1: A[0][0] × B[0][1] = A[0] × B[1]  ← 访问B[0,1]连续
线程2: A[1][0] × B[0][0] = A[4] × B[0]
线程3: A[1][0] × B[0][1] = A[4] × B[1]  ← 访问B[0,1]连续

内存访问：
  A: [0, 0, 4, 4]  - 部分重复
  B: [0, 1, 0, 1]  - 线程0和1访问B[0,1]连续 ✅
```

**K次循环后**：
```
线程0累加了：B的第0列所有元素（通过K次循环）
线程1累加了：B的第1列所有元素（通过K次循环）

但每次循环访问的是B的一行的连续元素！
```

---

## 🤔 深入思考

### 为什么实际提升(7x)小于理论(16.5x)？

**理论分析**：
- 内存事务从33个降到2个 → 16.5倍提升

**实际测量**：
- 性能只提升了约7倍

**可能原因**：
1. **计算延迟**：内存优化后，计算指令的延迟开始显现
2. **Cache效应**：L1/L2 cache的命中率、带宽限制
3. **Occupancy**：寄存器/共享内存使用影响并发线程数
4. **其他开销**：指令发射、warp调度等

→ 这就是为什么需要更多优化（Shared Memory、Vectorization等）

### 为什么不同矩阵大小提升比例不同？
```
小矩阵(128³): 5.43x  - kernel启动开销占比大
中矩阵(512³): 6.87x  - 接近理想
大矩阵(4096³): 6.98x - cache miss增加，但仍接近理想
```

### CEIL_DIV的作用
```cpp
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

作用：向上取整除法
例子：
  CEIL_DIV(100, 32) = 4  (4×32=128 ≥ 100)
  CEIL_DIV(128, 32) = 4  (4×32=128 = 128)
  CEIL_DIV(129, 32) = 5  (5×32=160 ≥ 129)

意义：确保有足够的block覆盖所有矩阵元素
```

---

## 📈 性能进化路线图
```
Naive (140 GFLOPS, 1.8% of cuBLAS)
    ↓ 
    改变线程映射 → 内存合并访问 (7x提升)
    ↓
Coalescing (982 GFLOPS, 12.8% of cuBLAS) ← 我们在这里
    ↓
    Shared Memory Tiling (预计1.5-2x提升)
    ↓
Shared Mem (~1500-2000 GFLOPS, ~20-25% of cuBLAS) ← 明天目标
    ↓
    更多优化...
    ↓
cuBLAS (7686 GFLOPS, 100%)
```

---

## 🔖 今日收获

### 技术层面
1. ✅ **理解了内存合并访问的威力**
   - 简单的线程映射改变 → 7倍性能提升
   
2. ✅ **掌握了索引计算技巧**
   - 除法和取模映射1D到2D
   - 灵活控制线程到数据的映射

3. ✅ **学会了性能分析方法**
   - 理论分析（内存事务数）
   - 实际测试（GFLOPS）
   - 找出差距（其他瓶颈）

### 思维层面
1. **GPU优化的层次性**
   - Level 1: 正确的线程映射（避免跳跃访问）✅ 完成
   - Level 2: 数据复用（Shared Memory）← 下一步
   - Level 3: 更高级优化（Vectorization, Warptiling等）

2. **性能优化的思路**
   - 找瓶颈 → 分析原因 → 针对性优化 → 验证效果

---

## ❓ 遗留疑问

- [ ] 如何用Nsight Compute验证内存事务数？
- [ ] 为什么2048矩阵比4096略快？（cache相关？）
- [ ] Shared Memory能带来多大提升？
- [ ] 如果交换A和B的位置（C=B×A），性能会怎样？

---

## 🎯 明天计划 (Day 3)

### 主要任务
- [ ] 运行Kernel 3（Shared Memory Blocking）
- [ ] 理解Shared Memory的原理和作用
- [ ] 对比Kernel 2和3的性能差异
- [ ] 理解Tiling的概念
- [ ] 画出Shared Memory的数据流向图

### 预期目标
- 理解为什么需要Shared Memory
- 掌握Tiling的基本思想
- 性能达到1500+ GFLOPS

---

## 📚 参考资料

- Simon的博客：[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- 项目源码：`src/kernels/2_kernel_global_mem_coalesce.cuh`
- CUDA编程指南：Memory Coalescing章节
- 我的实验代码：`my_experiments/verify_indexing.cu`

---

## 📝 代码片段记录

### 验证索引计算
```cpp
// my_experiments/verify_indexing.cu
__global__ void print_indices() {
    int tid = threadIdx.x;
    int cRow = blockIdx.x * 32 + (threadIdx.x / 32);
    int cCol = blockIdx.y * 32 + (threadIdx.x % 32);
    
    if (tid < 3 || (tid >= 31 && tid <= 33)) {
        printf("Block(%d,%d) Thread %4d: C[%2d][%2d]\n",
               blockIdx.x, blockIdx.y, tid, cRow, cCol);
    }
}
```

---

**学习时长**: 约2.5小时  
**完成日期**: 2026-02-14  
**下次学习**: 2026-02-15 (Day 3 - Shared Memory)

---

## 💭 个人反思

今天最大的突破是理解了"虽然需要B的一列，但通过横着访问来实现"这个概念。一开始很困惑，但通过完整的工作流程举例，终于明白了：

- 算法层面：每个线程需要B的一列（通过K次循环累加）
- 内存层面：每次循环时，32个线程访问B的一行的连续元素（合并访问）

**关键**：多个线程协作，把横着的访问拼成竖着的需求！

这种思维方式对后续的Shared Memory Tiling应该会很有帮助。

---