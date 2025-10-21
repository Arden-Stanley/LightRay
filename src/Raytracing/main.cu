#include <iostream>

#define SCREEN_WIDTH 1200;
#define SCREEN_HEIGHT 800;

__global__ void render(float *fb, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) {
        return;
    }
    int pixel_index = j * width * 3 + i * 3;
    fb[pixel_index + 0] = float(i) / width;
    fb[pixel_index + 1] = float(j) / height;
    fb[pixel_index + 2] = 0.2;
}

int main() {
    int num_pixels_x = SCREEN_WIDTH;
    int num_pixels_y = SCREEN_HEIGHT;
    int num_pixels = num_pixels_x * num_pixels_y;

    size_t fb_size = 3 * num_pixels * sizeof(float); 
    float *fb;
    cudaMallocManaged((void **)&fb, fb_size);

    int num_threads_x = 8;
    int num_threads_y = 8;
    dim3 blocks(num_pixels_x / num_threads_x + 1, num_pixels_y / num_threads_y + 1);
    dim3 threads(num_threads_x, num_threads_y);
    render<<<blocks, threads>>>(fb, num_pixels_x, num_pixels_y);

    cudaGetLastError();
    cudaDeviceSynchronize();

    std::cout << "P3\n" << num_pixels_x << " " << num_pixels_y << "\n255\n";
    for (int j = num_pixels_y - 1; j >= 0; j--) {
        for (int i = 0; i < num_pixels_x; i++) {
            size_t pixel_index = j * 3 * num_pixels_x + i * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    cudaFree(fb);

    return 0;
}