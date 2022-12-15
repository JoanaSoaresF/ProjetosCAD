#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#ifdef PNG
#include "pngwriter.h"
#endif

#define VERSION "V1"
#define BETWEEN_NEIGHBORS 1
#define TO_OUTPUT 2
#define NUM_ITERATIONS 1

/* Convert 2D index layout to unrolled 1D layout
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 * \returns An index in the unrolled 1D array.
 */
int getIndex(const int i, const int j, const int width)
{
    return i * width + j;
}

void initTemp(float *T, int h, int w)
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
    sprintf(filename, "../images/%s/heat_%06d.pgm", VERSION, n);
    save_png(T, h, w, filename, 'c');
#else
    sprintf(filename, "../images/%s/heat_%06d.pgm", VERSION, n);
    FILE *f = fopen(filename, "w");
    write_pgm(f, T, w, h, 100);
    fclose(f);
#endif
}

double timedif(struct timespec *t, struct timespec *t0)
{
    return (t->tv_sec - t0->tv_sec) + 1.0e-9 * (double)(t->tv_nsec - t0->tv_nsec);
}

int main(int argc, char *argv[])
{
    const int nx = 200;             // Width of the area
    const int ny = 200;             // Height of the area
    const float a = 0.5;            // Diffusion constant
    const float h = 0.005;          // h=dx=dy  grid spacing
    const int numSteps = 100000;    // Number of time steps to simulate (time=numSteps*dt)
    const int outputEvery = 100000; // How frequently to write output image

    const float h2 = h * h;

    const float dt = h2 / (4.0 * a); // Largest stable time step

    // MPI configuration
    int nproc, process_id;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

        // each process will compute N rows
    int N = (int)ceil((double)nx / (double)nproc);
    // int N = 12;
    if (process_id == 0 && argc > 1)
    {
        printf("\n--------------------------------------------------------------\n");
        printf("VERSION: %s \n"
               "GENERAL PROBLEM:\n"
               "\tGrid: %d x %d\n"
               "\tGrid spacing(h): %f\n"
               "\tDiffusion constant: %f\n"
               "\tNumber of steps:%d\n"
               "\tOutput: %d steps\n"
               "\tNumber of processes: %d\n"
               "\tNumber of lines per process: %d\n",
               VERSION, nx, ny, h, a, numSteps, outputEvery, nproc, N);
    }

    // Allocate two sets of data for current and next timesteps
    int numElements = (2 + N) * ny;

    float *Tn = (float *)calloc(numElements, sizeof(float));
    float *Tnp1 = (float *)calloc(numElements, sizeof(float));

    // Initializing the data for T0
    if (process_id == 0)
    {
        initTemp(Tn, N, ny);
    }

    // Fill in the data on the next step to ensure that the boundaries are identical.
    memcpy(Tnp1, Tn, numElements * sizeof(float));
    // writeTemp(Tn, N + 2, ny, 0);

    MPI_Status status;

    clock_t start;
    if (process_id == 0)
    {
        // Timing
        start = clock();
    }

    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        // Going through the entire area for one step
        // (borders stay at the same fixed temperatures)

        for (int i = 1; i <= N && i + N * process_id < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                // compute heat equation
                const int index = getIndex(i, j, ny);
                float tij = Tn[index];
                float tim1j = Tn[getIndex(i - 1, j, ny)];
                float tijm1 = Tn[getIndex(i, j - 1, ny)];
                float tip1j = Tn[getIndex(i + 1, j, ny)];
                float tijp1 = Tn[getIndex(i, j + 1, ny)];

                Tnp1[index] = tij + a * dt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
            }
        }

        // SEND data to neighbor
        if (process_id < nproc - 1) // send bottom line - my last computed line to the next processor
        {                           // last processor does not need to share bottom line
            MPI_Send(&Tnp1[N * ny], ny, MPI_FLOAT, process_id + 1, BETWEEN_NEIGHBORS, MPI_COMM_WORLD);
        }

        if (process_id > 0) // send top line - my fist computed line to the previous processor
        {                   // first processor does not need to send top line
            MPI_Send(&Tnp1[1 * ny], ny, MPI_FLOAT, process_id - 1, BETWEEN_NEIGHBORS, MPI_COMM_WORLD);
        }

        // RECEIVE data from neighbor
        if (process_id > 0) // receive top line from previous processor to fill the border line
        {                   // first process does not receive data from previous neighbor
            MPI_Recv(&Tnp1[0], ny, MPI_FLOAT, process_id - 1, BETWEEN_NEIGHBORS, MPI_COMM_WORLD, &status);
        }

        if (process_id < nproc - 1) // receive bottom line from next processor to fill the border line
        {                           // first process does not receive data from previous neighbor
            MPI_Recv(&Tnp1[(N + 1) * ny], ny, MPI_FLOAT, process_id + 1, BETWEEN_NEIGHBORS, MPI_COMM_WORLD, &status);
        }

        // Write the output if needed
        if ((n + 1) % outputEvery == 0)
        {

            if (process_id == 0)
            {
                // centralize all the results
                float *result = (float *)calloc((N + 1) * nproc * ny, sizeof(float));
                memcpy(&result[0], &Tnp1[0], N * ny * sizeof(float));
                for (int p = 1; p < nproc; p++)
                {
                    int computed_lines = (p == nproc - 1) ? nx - ((nproc - 1) * N) : N;
                    MPI_Recv(&result[p * N * ny], computed_lines * ny, MPI_FLOAT, p, TO_OUTPUT, MPI_COMM_WORLD, &status);
                }

                writeTemp(result, nx, ny, n + 1);
                free(result);
            }
            else
            {
                // send data to node 0
                // last process may compute less than N lines
                int computed_lines = (process_id == nproc - 1) ? nx - ((nproc - 1) * N) : N;
                MPI_Send(&Tnp1[0], computed_lines * ny, MPI_FLOAT, 0, TO_OUTPUT, MPI_COMM_WORLD);
            }
        }

        // Swapping the pointers for the next timestep
        float *t = Tn;
        Tn = Tnp1;
        Tnp1 = t;
    }

    // Timing

    if (process_id == 0)
    {
        clock_t finish = clock();
        printf(" %f\n", (double)(finish - start) / CLOCKS_PER_SEC);
    }

    // Release the memory
    free(Tn);
    free(Tnp1);

    MPI_Finalize();

    return 0;
}
