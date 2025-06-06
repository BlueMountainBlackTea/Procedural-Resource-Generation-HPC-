#include "grid.h"
#include "walker.h"


// Host function to run the DLA simulation
void runDLASimulation(Grid *grid, int maxWalkers, int maxSteps) {
    int width = grid->w;
    int height = grid->h;
    
    // Allocate device memory for grid
    int *d_grid;
    cudaMalloc(&d_grid, width * height * sizeof(int));
    cudaCheckError();
    
    // Copy grid data to device
    cudaMemcpy(d_grid, grid->g, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Allocate memory for walkers
    Walker *d_walkers;
    cudaMalloc(&d_walkers, maxWalkers * sizeof(Walker));
    cudaCheckError();
    
    // Allocate memory for random states
    curandState *d_states;
    cudaMalloc(&d_states, maxWalkers * sizeof(curandState));
    cudaCheckError();
    
    // Initialize random states
    int blockSize = 256;
    int numBlocks = (maxWalkers + blockSize - 1) / blockSize;
    
    unsigned long seed = time(NULL);
    initRandomStates<<<numBlocks, blockSize>>>(d_states, seed);
    cudaCheckError();
    
    // Batch processing for walkers
    int batchSize = 1000; // Process 1000 walkers at a time
    int numBatches = (maxWalkers + batchSize - 1) / batchSize;
    
    for (int batch = 0; batch < numBatches; batch++) {
        int startIdx = batch * batchSize;
        int endIdx = min(startIdx + batchSize, maxWalkers);
        int currentBatchSize = endIdx - startIdx;
        
        // Calculate grid and block dimensions for this batch
        numBlocks = (currentBatchSize + blockSize - 1) / blockSize;
        
        // Launch kernel
        dlaKernel<<<numBlocks, blockSize>>>(d_grid, width, height, 
                                            d_walkers + startIdx, 
                                            currentBatchSize, 
                                            maxSteps, 
                                            d_states + startIdx);
        cudaCheckError();
        
        // Synchronize after each batch
        cudaDeviceSynchronize();
        
        // Progress update
        if (batch % 10 == 0 || batch == numBatches - 1) {
            printf("Processed batch %d/%d (walkers %d-%d)\n", 
                   batch+1, numBatches, startIdx, endIdx-1);
        }
    }
    
    // Copy the grid back to host
    cudaMemcpy(grid->g, d_grid, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Free device memory
    cudaFree(d_grid);
    cudaFree(d_walkers);
    cudaFree(d_states);
}

// Main function
int main() {
    int width = 300;
    int height = 300;
    int maxWalkers = width * height;
    int maxSteps = 1000; // Maximum steps per walker
    
    // Initialize grid
    Grid grid = GridInit(width, height);
    
    // Set initial seeds
    grid.g[(height / 2) * width + (width / 2)] = YELLOW;
    grid.g[(height / 2) * width + (width / 2) + 1] = YELLOW;
    
    // Run CUDA DLA simulation
    clock_t start = clock();
    runDLASimulation(&grid, maxWalkers, maxSteps);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CUDA simulation time: %.3f seconds\n", time_spent);
    
    // Export the final grid to PPM
    printf("Exporting grid:\n");
    Counter stats = export_grid_to_ppm(&grid, width, height, "./cuda_output.ppm");
    printCounter(stats);
    
    // Free host memory
    freeGrid(grid);
    
    return 0;
}