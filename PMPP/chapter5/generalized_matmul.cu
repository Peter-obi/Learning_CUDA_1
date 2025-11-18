#define TILE_WIDTH 16

__global__ void matrixMulkernel(float* M, float* N, float* P, unsigned int a, unsigned int b, unsigned int c){

    //a = height of M or height of P
    //b = width of M or height of N
    //c = width of N or width of P

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;

    //Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    //Loop over M and N tiles to compute P element
    float Pvalue = 0;
    for(int ph = 0; ph < (b + TILE_WIDTH - 1) / TILE_WIDTH; ++ph){

        //Collaborative loading of M and N into shared memory
        if ((Row < a) && (ph*TILE_WIDTH+tx) < b)
            Mds[ty][tx] = M[Row*b + ph*TILE_WIDTH + tx];
        else Mds[ty][tx] = 0.0f;
        if((ph*TILE_WIDTH + ty) < b && (Col < c))
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty) * c + Col];
        else  Nds[ty][tx] = 0.0f;
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if(Row < a && Col < c)
        P[Row*c + Col] = Pvalue;

}
