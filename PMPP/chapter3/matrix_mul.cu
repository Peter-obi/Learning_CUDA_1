__global__
void MatrixMulKernel(float* M, float* N, float* P, int Width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < Width && col < Width){
        float Pvalue = 0;
        for(int k = 0; k < Width; ++k){
            Pvalue += M[row*Width + k] * N[k*Width + col];
        }
        P[row*Width + col] = Pvalue;
    }
}

void MatrixMul(float* M_h, float* N_h, float* P_h, int width){

    int size = width * width* sizeof(float);
    float* M_d; float* N_d;float* P_d;

    dim3 dimGrid (ceil(width/16.0), ceil(width/16.0), 1);
    dim3 dimBlock (16, 16, 1);

    cudaMalloc((void**)&M_d, size);
    cudaMalloc((void**)&N_d, size);
    cudaMalloc((void**)&P_d, size);

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

}
