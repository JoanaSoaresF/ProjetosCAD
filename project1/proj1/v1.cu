/*
 * Based on CSC materials from:
 *
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef PNG
#include "pngwriter.h"
#endif

#define BLOCK_SIZE 16
#define NUM_ITERATIONS 10
#define VERSION "V1 - CUDA with memory transfer every step"

/* Convert 2D index layout to unrolled 1D layout
 *
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 *
 * \returns An index in the unrolled 1D array.
 */
int __host__ __device__ getIndex(const int i, const int j, const int width)
{
    return i * width + j;
}

void __host__ __device__ initTemp(float *T, int h, int w)
{
    // Initializing the data with heat from top side
    // all other points at zero
    for (int i = 0; i < w; i++)
    {
        T[i] = 100.0;
    }
}
/* write_pgm - write a PGM image ascii file
 */
void write_pgm(FILE *f, float *img, int width, int height, int maxcolors)
{
    // header
    fprintf(f, "P2\n%d %d %d\n", width, height, maxcolors);
    // data
    for (int l = 0; l < height; l++)
    {
        for (int c = 0; c < width; c++)
        {
            int p = (l * width + c);
            fprintf(f, "%d ", (int)(img[p]));
        }
        putc('\n', f);
    }
}

/* write heat map image
 */
void writeTemp(float *T, int h, int w, int n)
{
    char filename[64];
#ifdef PNG
    sprintf(filename, "../images/v1/heat_%06d.pgm", n);
    save_png(T, h, w, filename, 'c');
#else
    sprintf(filename, "../images/v1/heat_%06d.pgm", n);
    FILE *f = fopen(filename, "w");
    write_pgm(f, T, w, h, 100);
    fclose(f);
#endif
}

__global__ void evolve_kernel(const float *Tn, float *Tnp1, const int nx, const int ny, const float a, const float h2, const float dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 0 && i < nx - 1)
    {
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (j > 0 && j < ny - 1)
        {
            const int index = getIndex(i, j, ny);
            float tij = Tn[index];
            float tim1j = Tn[getIndex(i - 1, j, ny)];
            float tijm1 = Tn[getIndex(i, j - 1, ny)];
            float tip1j = Tn[getIndex(i + 1, j, ny)];
            float tijp1 = Tn[getIndex(i, j + 1, ny)];

            // Explicit scheme
            Tnp1[index] = tij + a * dt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
        }
    }
}

double timedif(struct timespec *t, struct timespec *t0)
{
    return (t->tv_sec - t0->tv_sec) + 1.0e-9 * (double)(t->tv_nsec - t0->tv_nsec);
}

int main()
{
    const int nx = 200;             // Width of the area
    const int ny = 200;             // Height of the area
    const float a = 0.5;            // Diffusion constant
    const float h = 0.005;          // h=dx=dy  grid spacing
    const int numSteps = 100000;    // Number of time steps to simulate (time=numSteps*dt)
    const int outputEvery = 100000; // How frequently to write output image

    const float h2 = h * h;

    const float dt = h2 / (4.0 * a); // Largest stable time step

    int numElements = nx * ny;
    // Allocate two sets of data for current and next timesteps
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(nx / threadsPerBlock.x + 1, ny / threadsPerBlock.y + 1);

    printf("--------------------------------------------------------------------------------------------\n");
    printf("VERSION: %s \n"
           "GENERAL PROBLEM:\n"
           "\tGrid: %d x %d\n"
           "\tGrid spacing(h): %f\n"
           "\tDiffusion constant: %f\n"
           "\tNumber of steps: %d\n"
           "\tOutput: %d steps\n"
           "CUDA PARAMETERS:\n"
           "\tThreads Per Block: %d x %d\n"
           "\tBlocks: %d x %d \n",
           VERSION, nx, ny, h, a, numSteps, outputEvery, threadsPerBlock.x, threadsPerBlock.y, numBlocks.x, numBlocks.y);

    double totalTime = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        float *h_Tn = (float *)calloc(numElements, sizeof(float));
        float *h_Tnp1 = (float *)calloc(numElements, sizeof(float));

        // Initializing the data for T0
        initTemp(h_Tn, nx, ny);

        // Fill in the data on the next step to ensure that the boundaries are identical.
        memcpy(h_Tnp1, h_Tn, numElements * sizeof(float));

        float *d_Tn;
        float *d_Tnp1;
        cudaMalloc((void **)&d_Tn, numElements * sizeof(float));
        cudaMalloc((void **)&d_Tnp1, numElements * sizeof(float));

        writeTemp(h_Tn, nx, ny, 0);

        // Timing
        // clock_t start = clock();
        struct timespec start, finish;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Main loop

        for (int n = 0; n <= numSteps; n++)
        {
            cudaMemcpy(d_Tn, h_Tn, numElements * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Tnp1, h_Tnp1, numElements * sizeof(float), cudaMemcpyHostToDevice);

            evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_Tn, d_Tnp1, nx, ny, a, h2, dt);

            cudaMemcpy(h_Tn, d_Tn, numElements * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Tnp1, d_Tnp1, numElements * sizeof(float), cudaMemcpyDeviceToHost);

            // Check if any error occurred during execution
            cudaError_t errorCode = cudaGetLastError();
            if (errorCode != cudaSuccess)
            {
                printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
                exit(0);
            }

            // Write the output if needed
            if ((n + 1) % outputEvery == 0)
                writeTemp(h_Tnp1, nx, ny, n + 1);

            // Swapping the pointers for the next timestep
            float *t = h_Tn;
            h_Tn = h_Tnp1;
            h_Tnp1 = t;
        }

        // Timing
        // clock_t finish = clock();
        clock_gettime(CLOCK_MONOTONIC, &finish);
        double time = timedif(&finish, &start);
        totalTime += time;
        printf("Iteration %d took %f seconds\n", i, time);

        // Release the memory
        free(h_Tn);
        free(h_Tnp1);

        cudaFree(d_Tn);
        cudaFree(d_Tnp1);
    }

    printf("\nAverage time: %f\n\n", totalTime / (double)NUM_ITERATIONS);
    printf("--------------------------------------------------------------------------------------------\n");

    return 0;
}
