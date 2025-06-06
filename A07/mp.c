
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  
#include <omp.h>

#include "grid.h"
#include "walker.h"


#define MAX_WALKERS 100*100 // Maximum number of walkers


// int main() {
//     int width = 100;
//     int height = 100;

//     // Generate multiple DLA maps
//     Grid g1 = GridInit(width, height);

//     // Initial seeds
//     g1.g[(height / 2) * width + (width / 2)] = YELLOW;
//     g1.g[(height / 2) * width + (width / 2) + 1] = YELLOW;

//     // Get number of available cores for parallelization
//     int num_threads = omp_get_max_threads();
//     printf("Running with %d threads\n", num_threads);

//     // Run the parallel DLA generation
//     generateDLAParallel(&g1, MAX_WALKERS, num_threads);

//     GridDisplayColored(g1);

//     // Free memory
//     freeGrid(g1);

//     return 0;
// }

// Parallelize the DLA generation with benchmarking support
double generateDLAParallel(Grid *g, int maxWalkers, int num_threads) {
    // Initialize OpenMP environment
    omp_set_num_threads(num_threads);
    
    // Start timing
    double start_time = omp_get_wtime();
    
    // Each walker is processed in parallel
    #pragma omp parallel
    {
        // Each thread needs its own random seed
        unsigned int thread_seed = (unsigned int)time(NULL) ^ 
                                  ((omp_get_thread_num() + 1) * 1999);
        
        #pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < maxWalkers; ++i) {
            int color = (i % 2 == 0) ? RED : BLUE;
            Walker w = spawnWalkerAtEdgeParallel(g, color, &thread_seed);

            // Random walk until the walker interacts with the grid
            while (1) {
                // Move the walker and update the grid
                // WalkerMoveParallel(&w, g, 1, &thread_seed);
                WalkerMoveParallel(&w, g, 0, &thread_seed);

                // Check if the walker is adjacent to any cluster
                int cellValue;
                if ((cellValue = isAdjacentToClusterParallel(g, &w))) {
                    if (isWalkerClrMatch(&w, YELLOW) || isWalkerClrMatch(&w, PURPLE) || isWalkerClrMatch(&w, w.color)) {
                        // Same color: form a cluster
                        WalkerPlaceOnGridParallel(&w, g);
                        
                        // Spawn a green walker inside the cluster
                        SpawnGreenParallel(w, g, 1, 0.1, &thread_seed);
                    } else {
                        // Detonate at the current walker's position
                        if (i < maxWalkers/2) {
                            detonateParallel(g, w.x, w.y, 1, 1.0, 0.2, &thread_seed);
                        } else {
                            detonateParallel(g, w.x, w.y, 1, 0.4, 0.2, &thread_seed);
                        }
                    }
                    break;
                }
            }
        }
    }
    
    // End timing
    double end_time = omp_get_wtime();
    return (end_time - start_time);
}

// Benchmark function to run with different thread counts
void runBenchmarks() {
    int width = 300;  // Increased grid size for more meaningful benchmarks
    int height = 300;
    int numWalkers = width*height; // width*height*2;  // Increased walker count for better benchmarking
    
    int threadCounts[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    // int threadCounts[] = {512,1024};
    int numTests = sizeof(threadCounts) / sizeof(threadCounts[0]);
    
    printf("Benchmarking DLA Simulation with %d walkers on %dx%d grid\n", numWalkers, width, height);
    printf("%-10s %-15s %-15s\n", "Threads", "Time (sec)", "Speedup");
    printf("------------------------------------\n");

    // Run parallel versions
    for (int i = 0; i < numTests; i++) {
        Grid g = GridInit(width, height);
        g.g[(height / 2) * width + (width / 2)] = YELLOW;
        g.g[(height / 2) * width + (width / 2) + 1] = YELLOW;
        

        double parallelTime = generateDLAParallel(&g, numWalkers, threadCounts[i]);

        // Generate filename dynamically
        char filename[50];
        snprintf(filename, sizeof(filename), "img_%d.ppm", threadCounts[i]);
        export_grid_to_ppm(&g,g.w, g.h, filename);
        double speedup = 121.f / parallelTime;
        
        printf("%-10d %-15.4f %-15.4f\n", threadCounts[i], parallelTime, speedup);
        
        freeGrid(g);
    }
}

int main() {
    // Run the benchmarks
    runBenchmarks();
    return 0;
}