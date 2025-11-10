//The input image is encoded as unsigned chars [0, 255]
//Each pixel is 3 consecutive chars for the 3 channels (RGB)

#define CHANNELS 3

__global__
void colortoGrayscaleConversion(unsigned char * Pout, unsigned char * Pin, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < width && row < height){
        //Get ID offset for the grayscale image
        int grayOffset = row * width + col;
        //RGB image having CHANNEL times more columns than the gray scale image
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset ]; //Red value
        unsigned char g = Pin[rgbOffset + 1]; //Green value
        unsigned char b = Pin[rgbOffset + 2]; //Blue value

        //Perform the rescaling and store it -> multiply by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void grayscale(unsigned char * Pout_h, unsigned char * Pin_h, int m, int n){

    int rgb_size = m * n * CHANNELS * sizeof(unsigned char);
    int gray_size = m * n * sizeof(unsigned char);
    unsigned char * Pout_d; unsigned char * Pin_d;

    dim3 dimGrid (ceil(m/16.0), ceil(n/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    cudaMalloc((void**)&Pin_d, rgb_size);
    cudaMalloc((void**)&Pout_d, gray_size);

    cudaMemcpy(Pin_d, Pin_h, rgb_size, cudaMemcpyHostToDevice);

    colortoGrayscaleConversion<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, m, n);

    cudaMemcpy(Pout_h, Pout_d, gray_size, cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);

}
