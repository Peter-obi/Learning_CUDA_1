#define TILE_WIDTH 16

__global__ void matrixMulkernel(const float* __restrict__ M, 
                                const float* __restrict__ N, 
                                float* __restrict__ P, 
                                int a, int b, int c, 
                                int Mds_sz, int Nds_sz) {

    extern __shared__ float sh[];

    float *Mds = sh;
    float *Nds = sh + Mds_sz;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    int num_phases = (b + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < num_phases; ++ph) {
        int kBase = ph * TILE_WIDTH;

        if (Row < a && (kBase + tx) < b)
            Mds[ty * TILE_WIDTH + tx] = M[Row * b + (kBase + tx)];
        else
            Mds[ty * TILE_WIDTH + tx] = 0.0f;

        if ((kBase + ty) < b && Col < c)
            Nds[ty * TILE_WIDTH + tx] = N[(kBase + ty) * c + Col];
        else
            Nds[ty * TILE_WIDTH + tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];

        __syncthreads();
    }

    if (Row < a && Col < c)
        P[Row * c + Col] = Pvalue;
}

void matrixMul(const float* Md,
               const float* Nd,
               float* Pd,
               int a, int b, int c) {

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((c + TILE_WIDTH - 1) / TILE_WIDTH,
                 (a + TILE_WIDTH - 1) / TILE_WIDTH);

    int tile_elems = TILE_WIDTH * TILE_WIDTH;
    size_t shmem_bytes = 2 * tile_elems * sizeof(float);

    matrixMulkernel<<<gridDim, blockDim, shmem_bytes>>>(
        Md, Nd, Pd,
        a, b, c,
        tile_elems, tile_elems
    );
}
