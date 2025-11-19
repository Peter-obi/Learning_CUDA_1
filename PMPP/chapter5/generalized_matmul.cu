#define TILE_WIDTH 16

__global__ void matrixMulkernel(const float* __restrict__ M,
                                const float* __restrict__ N,
                                float* __restrict__ P,
                                int a, int b, int c) {

    // a = height of M or height of P
    // b = width of M or height of N (shared dim)
    // c = width of N or width of P

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // tile over shared dimension b
    for (int ph = 0; ph < (b + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {

        // M: a x b
        if (Row < a && (ph * TILE_WIDTH + tx) < b)
            Mds[ty][tx] = M[Row * b + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        // N: b x c
        if ((ph * TILE_WIDTH + ty) < b && Col < c)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * c + Col];
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    // P: a x c
    if (Row < a && Col < c)
        P[Row * c + Col] = Pvalue;
}
