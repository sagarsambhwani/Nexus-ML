# Chapter 04: NumPy, Vectorization, and Memory Layout

---

## 1. Big Picture

Python is an interpreted dynamically-typed language with substantial overhead for scalar loops. Numerical computation and ML engineering rely on **NumPy** to achieve C-speed performance. NumPy does this by executing operations on contiguous C-array memory blocks via SIMD (Single Instruction, Multiple Data) CPU vector hardware instructions.

Understanding striding, memory layout (C-contiguous vs Fortran-contiguous), broadcasting, and zero-copy views is essential for writing high-performance data processing code.

---

## 2. Intuition

Imagine calculating total cost for 1,000,000 grocery store items:
- **Python List Loop**: A cashier picking up each item individually, asking a manager "what type of object is this?", looking up its price in a dictionary, adding it to a total counter, and repeating 1 million times.
- **NumPy Vectorization**: An automated assembly belt passing 1,000,000 raw 64-bit float numbers through an industrial scanner that adds 8 numbers simultaneously per CPU clock cycle.

---

## 3. Visualization

```text
Python List (Pointers to Heap Objects):
  [Ptr1, Ptr2, Ptr3] ──► Heap Object 1 (Header, Type, RefCount, Value)
                     ──► Heap Object 2 (Header, Type, RefCount, Value)
  (High cache miss rate, 28 bytes per integer!)

NumPy ndarray (Contiguous Memory Block):
  ┌──────┬──────┬──────┬──────┬──────┐
  │ 1.0  │ 2.0  │ 3.0  │ 4.0  │ 5.0  │  64-bit floats packed contiguously
  └──────┴──────┴──────┴──────┴──────┘
  (High CPU L1/L2 cache hit rate, SIMD vectorized)
```

---

## 4. Mathematics

A NumPy $N$-dimensional array is defined by:
1. **Data Pointer**: Memory address of the first element.
2. **Shape**: Tuple $(d_0, d_1, \dots, d_{k-1})$ specifying dimensions.
3. **Strides**: Tuple $(s_0, s_1, \dots, s_{k-1})$ specifying bytes to step in memory to advance by 1 element along axis $i$.

The memory offset in bytes for element index $(i_0, i_1, \dots, i_{k-1})$ is:

$$\text{Offset} = \sum_{j=0}^{k-1} i_j \cdot s_j$$

For C-contiguous array of shape $(M, N)$ storing 8-byte float64 values:
$$s_0 = N \cdot 8, \quad s_1 = 8$$

---

## 5. Python (From Scratch vs NumPy)

Comparing pure Python loop performance vs NumPy vectorization:

```python
import time
import numpy as np

N = 5_000_000

# 1. Pure Python Loop
a_py = list(range(N))
b_py = list(range(N))

start = time.perf_counter()
c_py = [a + b for a, b in zip(a_py, b_py)]
py_time = time.perf_counter() - start

# 2. NumPy Vectorized Operation
a_np = np.arange(N, dtype=np.int64)
b_np = np.arange(N, dtype=np.int64)

start = time.perf_counter()
c_np = a_np + b_np
np_time = time.perf_counter() - start

print(f"Python Loop Time:  {py_time:.4f} seconds")
print(f"NumPy Vector Time: {np_time:.4f} seconds")
print(f"Speedup Factor:    {py_time / np_time:.1f}x faster!")
```

---

## 6. Production Memory Management (Views vs Copies)

In production pipelines handling gigabyte datasets, unintentional memory copying triggers out-of-memory (OOM) crashes:

```python
# Create 2D Array
arr = np.ones((10000, 10000), dtype=np.float64)  # 800 MB

# Slice operation creates a VIEW (Zero Memory Allocation)
view_arr = arr[:5000, :5000]
print("Shares Memory:", np.shares_memory(arr, view_arr))

# Advanced Indexing (Fancy Indexing) forces a COPY!
copy_arr = arr[[0, 1, 2], :]
print("Shares Memory with Fancy Indexing:", np.shares_memory(arr, copy_arr))
```

---

## 7. Under the Hood

- NumPy arrays store a C `PyArrayObject` structure containing `data` (char pointer), `nd` (dimensions count), `dimensions` (shape array pointer), and `strides` (stride array pointer).
- Broadcasting aligns axes from right to left, expanding dimensions with shape length $1$ without allocating new memory, adjusting strides to $0$ along expanded dimensions!

---

## 8. Engineering Perspective

- **Memory Efficiency**: Python 64-bit integer takes 28 bytes. NumPy 64-bit int takes exactly 8 bytes (3.5x memory reduction).
- **SIMD Vectorization**: Modern CPUs process 256-bit AVX2 vectors (4 double-precision floats at once) or 512-bit AVX-512 (8 floats at once).

---

## 9. Common Mistakes

1. **Explicit Python Loops over Arrays**: Using `for i in range(len(arr)): arr[i] *= 2` instead of `arr *= 2` completely destroys performance.
2. **Accidental Type Promotion**: Adding `int32` to `float64` silently doubles array memory footprint.

---

## 10. Interview Questions

### Q1: What is broadcasting in NumPy and what are the rules governing it?
**Answer**: Broadcasting allows NumPy to perform arithmetic operations on arrays with different shapes. Two dimensions are compatible when (1) they are equal, or (2) one of them is 1. Dimensions are compared starting from the trailing (rightmost) shape tuple elements.

---

## 11. Exercises

1. **Coding**: Implement a zero-copy 2D matrix transpose function using only stride manipulation (`np.lib.stride_tricks.as_strided`).
2. **Benchmarking**: Benchmark matrix multiplication `A @ B` for C-contiguous vs Fortran-contiguous memory layouts.

---

## 12. Mini Project: High-Speed Image Filters

Write a vectorized image processing script `vectorized_filters.py` that applies Gaussian blur and edge detection kernels to HD images using NumPy 2D convolutions without any `for` loops.

---

## 13. Capstone Integration

Utilized throughout `src/house_prices/pipeline.py` and `src/sentiment_analysis/pipeline.py` for dense matrix feature transformations and tensor batching.
