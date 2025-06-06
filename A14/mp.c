#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "grid.h"
#include "walker.h"

#define MAX_WALKERS 10000

// MPI worker function to process a batch of walkers
void process_walkers(Grid *local_grid, int start_walker, int end_walker, int rank, int size,
                    int grid_width, int grid_height) {
    unsigned int seed = (unsigned int)time(NULL) ^ ((rank + 1) * 1999);
    
    for (int i = start_walker; i < end_walker; ++i) {
        int color = (i % 2 == 0) ? RED : BLUE;
        Walker w = spawnWalkerAtEdge(local_grid, color, &seed, rank);

        // Random walk until the walker interacts with the grid
        while (1) {
            WalkerMove(&w, local_grid, 0, &seed);

            // Check if the walker is adjacent to any cluster
            if (isAdjacentToCluster(local_grid, &w)) {
                if (isWalkerClrMatch(&w, YELLOW) || isWalkerClrMatch(&w, PURPLE) || isWalkerClrMatch(&w, w.color)) {
                    // Same color: form a cluster
                    WalkerPlaceOnGrid(&w, local_grid);
                    
                    // Spawn a green walker inside the cluster
                    SpawnGreen(w, local_grid, 1, 0.1, &seed);
                } else {
                    // Detonate at the current walker's position
                    if (i < MAX_WALKERS/2) {
                        detonate(local_grid, w.x, w.y, 1, 1.0, 0.2, &seed);
                    } else {
                        detonate(local_grid, w.x, w.y, 1, 0.4, 0.2, &seed);
                    }
                }
                break;
            }
        }
    }
}

// MPI DLA generation function
double generateDLA_MPI(int grid_width, int grid_height, int numWalkers) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Each process has a copy of the entire grid for simplicity
    Grid local_grid = GridInit(grid_width, grid_height);
    
    // Initialize the grid with the seed points (all processes do this)
    local_grid.g[(grid_height / 2) * grid_width + (grid_width / 2)] = YELLOW;
    local_grid.g[(grid_height / 2) * grid_width + (grid_width / 2) + 1] = YELLOW;
    
    // Start timing
    double start_time = MPI_Wtime();
    
    // Calculate local workload
    int walkers_per_proc = numWalkers / size;
    int remainder = numWalkers % size;
    int start_walker = rank * walkers_per_proc + (rank < remainder ? rank : remainder);
    int end_walker = start_walker + walkers_per_proc + (rank < remainder ? 1 : 0);
    
    if (rank == 0) {
        printf("Process %d handling walkers %d to %d\n", rank, start_walker, end_walker-1);
    }
    
    // Process assigned walkers
    process_walkers(&local_grid, start_walker, end_walker, rank, size, grid_width, grid_height);
    
    // Create a final grid buffer for reduction
    int *final_grid_buffer = NULL;
    if (rank == 0) {
        final_grid_buffer = (int*)calloc(grid_width * grid_height, sizeof(int));
    }
    
    // Use MPI_Reduce to combine all grids
    // For each cell, take the maximum value across all processes
    MPI_Reduce(local_grid.g, final_grid_buffer, grid_width * grid_height, 
               MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Rank 0 handles the final grid and output
    if (rank == 0) {
        // Create a final grid with the reduced data
        Grid final_grid = GridInit(grid_width, grid_height);
        
        // Copy reduced data to the final grid
        for (int i = 0; i < grid_width * grid_height; i++) {
            final_grid.g[i] = final_grid_buffer[i];
        }
        
        // Export the final grid
        char filename[50];
        snprintf(filename, sizeof(filename), "mpi_result_%d_procs.ppm", size);
        export_grid_to_ppm(&final_grid, grid_width, grid_height, filename);
        
        // Print some stats about the grid
        int cell_count = 0;
        for (int i = 0; i < grid_width * grid_height; i++) {
            if (final_grid.g[i] != 0) {
                cell_count++;
            }
        }
        printf("Final grid has %d non-empty cells\n", cell_count);
        
        // Clean up
        freeGrid(final_grid);
        free(final_grid_buffer);
    }
    
    // Clean up local grid
    freeGrid(local_grid);
    
    // End timing
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    // Get maximum time across all processes
    double max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    return max_time;
}

// Run benchmarks with MPI
void runBenchmarks() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Use a smaller grid and walker count for testing
    int width = 300;
    int height = 300;
    int numWalkers = 300*300; // Reduced for testing
    
    if (rank == 0) {
        printf("Running DLA Simulation with MPI using %d processes\n", size);
        printf("Grid size: %dx%d, Walkers: %d\n", width, height, numWalkers);
    }
    
    double time = generateDLA_MPI(width, height, numWalkers);
    
    if (rank == 0) {
        printf("Processes: %d, Time: %.4f seconds\n", size, time);
    }
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Run the benchmarks
    runBenchmarks();
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}