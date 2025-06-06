
// walker.h
#ifndef WALKER_H
#define WALKER_H

#include <stdlib.h>
#include <curand_kernel.h>

// Array of colors structure
typedef struct {
    int clrs[4];
    int size;
} ArrClrs;

// Walker structure
typedef struct {
    int x;
    int y;
    int color;
    ArrClrs arrClrs;
    int active; // Flag to indicate if the walker is active
} Walker;

// Initialize walker
__host__ __device__ static Walker WalkerInit(int x, int y, int color) {
    Walker w = {x, y, color, {{0}, 0}, 1};
    return w;
}



// dla_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>
#include "grid.h"
#include "walker.h"

// Error checking macro
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernels and device functions

// Add a unique color to the array
__device__ void addUniqueColor(ArrClrs *arr, int color) {
    for (int i = 0; i < arr->size; ++i) {
        if (arr->clrs[i] == color) {
            return; // Color already exists
        }
    }
    if (arr->size < 4) { // Ensure we don't exceed array bounds
        arr->clrs[arr->size++] = color;
    }
}

// Check if the walker's color array contains a specific color
__device__ int isWalkerClrMatch(const Walker *w, int color) {
    for (int i = 0; i < w->arrClrs.size; ++i) {
        if (w->arrClrs.clrs[i] == color) {
            return 1;
        }
    }
    return 0;
}

// Check if a walker is adjacent to a cluster
__device__ int isAdjacentToCluster(const int *grid, int width, int height, Walker *w) {
    // Reset the Walker's color array
    w->arrClrs.size = 0;

    int x = w->x;
    int y = w->y;

    // Check Left
    if (x > 0 && grid[y * width + (x - 1)] != 0) {
        addUniqueColor(&w->arrClrs, grid[y * width + (x - 1)]);
    }
    // Check Right
    if (x < width - 1 && grid[y * width + (x + 1)] != 0) {
        addUniqueColor(&w->arrClrs, grid[y * width + (x + 1)]);
    }
    // Check Up
    if (y > 0 && grid[(y - 1) * width + x] != 0) {
        addUniqueColor(&w->arrClrs, grid[(y - 1) * width + x]);
    }
    // Check Down
    if (y < height - 1 && grid[(y + 1) * width + x] != 0) {
        addUniqueColor(&w->arrClrs, grid[(y + 1) * width + x]);
    }

    // Return the number of unique colors found
    return w->arrClrs.size;
}

// Move a walker randomly
__device__ void walkerMove(Walker *w, int *grid, int width, int height, curandState *state) {
    // Clear the old position of the walker in the grid (not needed if we're using atomic operations later)
    // grid[w->y * width + w->x] = 0;

    // Move the walker randomly
    int direction = curand(state) % 4; // 0=up, 1=down, 2=left, 3=right
    int newX = w->x;
    int newY = w->y;

    switch (direction) {
        case 0: if (w->y > 0) newY--; break; // Up
        case 1: if (w->y < height - 1) newY++; break; // Down
        case 2: if (w->x > 0) newX--; break; // Left
        case 3: if (w->x < width - 1) newX++; break; // Right
    }

    // Update the walker's position
    w->x = newX;
    w->y = newY;

    // No need to set the grid here, we'll do it atomically later if needed
}

// Place a walker on the grid
__device__ void walkerPlaceOnGrid(const Walker *w, int *grid, int width) {
    // Use atomic operation to prevent race conditions
    atomicExch(&grid[w->y * width + w->x], w->color);
}

// Spawn green walkers and create purple cells
__device__ void spawnGreen(Walker w, int *grid, int width, int height, int R, float p, curandState *state) {
    if (curand_uniform(state) >= p)
        return;
        
    int greenFlag = 0;
    int a = -1, b = -1;

    // Find an non-empty spot near the walker
    for (int i = -R; i <= R; i++) {
        for (int j = -R; j <= R; j++) {
            int nx = w.x + i;
            int ny = w.y + j;

            // Check if within bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int cellValue = grid[ny * width + nx];

                if (cellValue != 0) {  // Found a non-empty cell
                    greenFlag = 1;
                    a = nx;
                    b = ny;
                    i = R + 1;  // Force exit from outer loop
                    break;      // Exit inner loop
                }
            }
        }
    }

    if (greenFlag == 0)
        return;  // No valid location found

    Walker greenWalker = WalkerInit(a, b, GREEN);

    // Move the green walker inside the cluster
    int maxSteps = 100;  // Prevent infinite loops
    while (maxSteps--) {
        // Move the green walker
        int direction = curand(state) % 4;
        int nx = greenWalker.x;
        int ny = greenWalker.y;

        switch (direction) {
            case 0: if (ny > 0) ny--; break; // Up
            case 1: if (ny < height - 1) ny++; break; // Down
            case 2: if (nx > 0) nx--; break; // Left
            case 3: if (nx < width - 1) nx++; break; // Right
        }

        greenWalker.x = nx;
        greenWalker.y = ny;

        if (grid[greenWalker.y * width + greenWalker.x] == 0) {
            // Found an empty spot, turn the area around it into PURPLE
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nx = greenWalker.x + dx;
                    int ny = greenWalker.y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        if (grid[ny * width + nx] == 0)
                            atomicExch(&grid[ny * width + nx], PURPLE);
                    }
                }
            }
            break;
        }
    }
}

// Detonate an area and change colors
__device__ void detonate(int *grid, int x, int y, int width, int height, int R, float pDet, float pPurp, curandState *state) {
    int detFlag = 0;

    if (curand_uniform(state) < pDet) {
        // Iterate over cells in radius R
        for (int i = -R; i <= R; i++) {
            for (int j = -R; j <= R; j++) {
                int nx = x + i;
                int ny = y + j;

                // Check if within bounds
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int cellValue = grid[ny * width + nx];

                    // Ensure YELLOW cells are not destroyed and destroy only opposing colors
                    if (cellValue != YELLOW) {
                        atomicExch(&grid[ny * width + nx], 0); // Destroy the cell
                        detFlag = 1;
                    }
                }
            }
        }
    }

    // Now handle the radius R+1 and change colors based on probability pPurp
    for (int i = -R - 1; i <= R + 1; i++) {
        for (int j = -R - 1; j <= R + 1; j++) {
            int nx = x + i;
            int ny = y + j;

            // Check if within bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int cellValue = grid[ny * width + nx];
                
                // If it's RED or BLUE, change to PURPLE with probability pPurp
                if ((cellValue == RED) || (cellValue == BLUE)) {
                    if (curand_uniform(state) < pPurp) {
                        // Change color to purple
                        atomicExch(&grid[ny * width + nx], PURPLE);
                    }
                }
            }
        }
    }

    // Destroy the cell at the center of the explosion
    if (detFlag) {
        atomicExch(&grid[y * width + x], 0);
    }
}

// Initialize random states
__global__ void initRandomStates(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Kernel for spawning and moving walkers
__global__ void dlaKernel(int *grid, int width, int height, Walker *walkers, int numWalkers, int maxSteps, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWalkers) return;

    curandState localState = states[idx];
    Walker walker = walkers[idx];

    // Only process active walkers
    if (!walker.active) return;

    // Choose walker color (alternating)
    int color = (idx % 2 == 0) ? RED : BLUE;

    // Spawn walker at edge
    int edge = curand(&localState) % 4;
    
    switch (edge) {
        case 0: // Top edge
            walker.x = curand(&localState) % width;
            walker.y = 0;
            break;
        case 1: // Bottom edge
            walker.x = curand(&localState) % width;
            walker.y = height - 1;
            break;
        case 2: // Left edge
            walker.x = 0;
            walker.y = curand(&localState) % height;
            break;
        case 3: // Right edge
            walker.x = width - 1;
            walker.y = curand(&localState) % height;
            break;
    }
    
    walker.color = color;

    // Random walk until maxSteps or interaction with grid
    for (int step = 0; step < maxSteps; step++) {
        walkerMove(&walker, grid, width, height, &localState);

        // Check if the walker is adjacent to any cluster
        int adjacent = isAdjacentToCluster(grid, width, height, &walker);
        
        if (adjacent) {
            if (isWalkerClrMatch(&walker, YELLOW) || isWalkerClrMatch(&walker, PURPLE) || isWalkerClrMatch(&walker, walker.color)) {
                // Same color: form a cluster
                walkerPlaceOnGrid(&walker, grid, width);
                
                // Spawn a green walker inside the cluster
                spawnGreen(walker, grid, width, height, 1, 0.1, &localState);
            } else {
                // Detonate at the current walker's position
                if (idx < numWalkers/2) {
                    detonate(grid, walker.x, walker.y, width, height, 1, 1.0, 0.2, &localState);
                } else {
                    detonate(grid, walker.x, walker.y, width, height, 1, 0.4, 0.1, &localState);
                }
            }
            
            // Mark walker as inactive
            walker.active = 0;
            break;
        }
    }

    // Save the walker state back
    walkers[idx] = walker;
    states[idx] = localState;
}


#endif // WALKER_H