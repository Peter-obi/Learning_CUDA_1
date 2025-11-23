#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matrixMulKernel(const float* __restrict__ M,
                                const float* __restrict__ N,
                                float* __restrict__ P,
                                int a, int b, int c)
{
    extern __shared__ float sh[];

    float* Mds = sh;
    float* Nds = sh + TILE_WIDTH * TILE_WIDTH;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row      = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    #pragma unroll
    for (int d = 0; d < COARSE_FACTOR; ++d)
        Pvalue[d] = 0.0f;

    int num_phases = (b + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < num_phases; ++ph) {
        int kbase = ph * TILE_WIDTH;

        // Load M tile
        if (Row < a && (kbase + tx) < b)
            Mds[ty * TILE_WIDTH + tx] = M[Row * b + (kbase + tx)];
        else
            Mds[ty * TILE_WIDTH + tx] = 0.0f;

        // For each coarsened column group
        for (int d = 0; d < COARSE_FACTOR; ++d) {
            int col = colStart + d * TILE_WIDTH;

            // Load N tile for this d
            if ((kbase + ty) < b && col < c)
                Nds[ty * TILE_WIDTH + tx] = N[(kbase + ty) * c + col];
            else
                Nds[ty * TILE_WIDTH + tx] = 0.0f;

            __syncthreads();

            // Compute partial dot products for this d
            #pragma unroll
            for (int k = 0; k < TILE_WIDTH; ++k)
                Pvalue[d] += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];

            __syncthreads();
        }
    }

    // Write results
    for (int d = 0; d < COARSE_FACTOR; ++d) {
        int col = colStart + d * TILE_WIDTH;
        if (Row < a && col < c)
            P[Row * c + col] = Pvalue[d];
    }
}

void matrixMul(const float* h_M, const float* h_N, float* h_P,
               int a, int b, int c)
{
    float *d_M, *d_N, *d_P;

    size_t shmem  = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    size_t size_m = (size_t)a * b * sizeof(float);
    size_t size_n = (size_t)b * c * sizeof(float);
    size_t size_p = (size_t)a * c * sizeof(float);

    cudaMalloc((void**)&d_M, size_m);
    cudaMalloc((void**)&d_N, size_n);
    cudaMalloc((void**)&d_P, size_p);

    cudaMemcpy(d_M, h_M, size_m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size_n, cudaMemcpyHostToDevice);

    dim3 Dimblock(TILE_WIDTH, TILE_WIDTH);
    int cols_per_block = TILE_WIDTH * COARSE_FACTOR;
    dim3 Dimgrid((c + cols_per_block - 1) / cols_per_block,
                 (a + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMulKernel<<<Dimgrid, Dimblock, shmem>>>(d_M, d_N, d_P, a, b, c);

    cudaMemcpy(h_P, d_P, size_p, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}
