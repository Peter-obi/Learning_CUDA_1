
---

## 0. Thread indexing / 3D blocks

**3D → 1D thread index:**

```cpp
int linear_id =
    threadIdx.z * (blockDim.x * blockDim.y) +
    threadIdx.y * blockDim.x +
    threadIdx.x;
```

* Threads per **xy-plane**: `blockDim.x * blockDim.y`
* Threads per **row**: `blockDim.x`
* **x moves fastest, then y, then z.**

---

## 1. Row-major indexing (burn this in)

For a matrix with shape **rows × cols**:

```cpp
index(row, col) = row * cols + col;
```

* M: `a × b` → `M[row * b + col]`
* N: `b × c` → `N[row * c + col]`
* P: `a × c` → `P[row * c + col]`

Stride is always **number of columns**.

---

## 2. Tiled GEMM mental model

Shapes:

* `M: a × b`
* `N: b × c`
* `P: a × c`
* shared dimension: `b`

Block mapping:

```cpp
int Row = blockIdx.y * TILE + threadIdx.y;
int Col = blockIdx.x * TILE + threadIdx.x;
```

Phase loop over shared dim `b`:

```cpp
int num_phases = (b + TILE - 1) / TILE;  // ceil(b / TILE)
for (int ph = 0; ph < num_phases; ++ph) {
    int kBase = ph * TILE;
    // load tiles of M and N
    // barrier
    // dot product over k in [0..TILE)
    // barrier
}
```

**Ceil division pattern (integer!):**

```cpp
ceil_div(x, T) = (x + T - 1) / T;
```

Remember: integer division **floors**.

---

## 3. Shared memory + `__syncthreads()` rules

**Pattern: load → use → overwrite**

```cpp
// 1. Write tile
tile[...] = ...;
__syncthreads();

// 2. Read tile (compute)
... = tile[...];
__syncthreads();

// 3. Next phase overwrites tile
```

* **First barrier:**
  prevent **read-before-write** (RAW) → true dependence.

* **Second barrier:**
  prevent overwriting tile while others still read → **WAR/WAW** (“false dependence” in PMPP).

Red flag heuristic:

> Works with blockDim = 1, breaks for >1 → *suspect missing `__syncthreads()` / shared-memory race first*.

---

## 4. Dynamic shared memory layout

**Option A – element counts (simpler for you):**

```cpp
extern __shared__ float sh[];

float* Mds = sh;
float* Nds = sh + Mds_elems;  // Mds_elems = TILE * TILE
```

Launch:

```cpp
int tile_elems   = TILE * TILE;
size_t sh_bytes  = 2 * tile_elems * sizeof(float);

kernel<<<grid, block, sh_bytes>>>(..., tile_elems, tile_elems);
```

**Option B – byte counts (PMPP style):**

```cpp
extern __shared__ unsigned char sh[];

float* Mds = (float*)sh;
float* Nds = (float*)(sh + Mds_bytes);
```

Then pass `Mds_bytes`, `Nds_bytes` as **bytes** from host.

Key: don’t mix **bytes** and **elements** in pointer arithmetic.

---

## 5. Occupancy quick checks

Per SM limits example (like Ch.5 problems):

* Max threads/SM = `T_max`
* Max blocks/SM  = `B_max`
* Max regs/SM    = `R_max`
* Max shmem/SM   = `S_max`

Per block:

* Threads/block = `T_block`
* Regs/block    = `R_block = regs_per_thread * T_block`
* Shmem/block   = `S_block`

Then:

```text
limit_threads = floor(T_max / T_block)
limit_regs    = floor(R_max / R_block)
limit_shmem   = floor(S_max / S_block)

blocks_per_SM = min(B_max, limit_threads, limit_regs, limit_shmem)
occupancy     = (blocks_per_SM * T_block) / T_max
```

---

## 6. Global memory traffic & tiling intuition

For N×N GEMM, T×T tiles:

* **No tiling:** each input element loaded ~N times.
* **With tiling:** each input element loaded ~N/T times.
* **Reduction factor:** `T`.

Rule of thumb:

> Loads per element ≈ `N / TILE_WIDTH`
> Bandwidth reduced by factor `TILE_WIDTH`.

---

## 7. FLOPs / byte (arithmetic intensity)

Per thread:

* FLOPs: count muls + adds in the kernel’s math.
* Global memory: `num_accesses * bytes_per_access` (usually 4 for float).

```cpp
intensity = FLOPs / bytes;
```

Compare to device:

```cpp
machine_balance = peak_FLOPS / peak_bandwidth;   // FLOP/byte

if intensity < machine_balance → memory-bound
if intensity > machine_balance → compute-bound
```

---

## 8. SIMT / SIMD / SPMD mental map

* **Logical instruction stream = SIMT**
  (each thread has its own “program”, own registers, own control flow)

* **Physical datapath = SIMD**
  (warp executes same instruction on multiple lanes at once)

* **Programming abstraction = SPMD**
  (Same Program, Multiple Data → your kernel is one program, run by many threads)

---

If you want, tomorrow we can:

* start from this cheat sheet,
* you write the GEMM from scratch,
* and I’ll only point out indexing/sync issues if they appear.
