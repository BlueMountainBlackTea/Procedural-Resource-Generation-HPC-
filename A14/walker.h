/* walker.h */
#ifndef RANDOM_WALKER_H
#define RANDOM_WALKER_H

#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "grid.h"

#define RED 31
#define GREEN 32
#define YELLOW 33
#define BLUE 34
#define PURPLE 35

typedef struct {
    int clrs[4];
    int size;
} ArrClrs;

typedef struct {
    int x;
    int y;
    int color;
    ArrClrs arrClrs;
    int active;  // Flag to indicate if walker is active (used for MPI)
    int owner;   // Process ID that owns this walker
} Walker;

// Function to initialize a walker
static Walker WalkerInit(int x, int y, int color) {
    Walker w = {
        .x = x,
        .y = y,
        .color = color,
        .arrClrs = {
            .clrs = {0},
            .size = 0
        },
        .active = 1,
        .owner = 0
    };
    return w;
}

// MPI-safe random number generator
unsigned int mpi_safe_rand(unsigned int* seed) {
    if (*seed == 0) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        *seed = (unsigned int)time(NULL) ^ (rank << 16);
    }
    return rand_r(seed);
}

// Move walker function for MPI
static void WalkerMove(Walker *w, Grid *g, int GridSetFlag, unsigned int* seed) {
    MP_ASSERT(w != NULL);
    MP_ASSERT(g != NULL);

    // If GridSetFlag is enabled, clear the old position
    if (GridSetFlag) {
        GridSet(g, w->x, w->y, 0);
    }

    // Compute the new position
    int direction = mpi_safe_rand(seed) % 4; // 0=up, 1=down, 2=left, 3=right
    int newX = w->x;
    int newY = w->y;

    switch (direction) {
        case 0: if (w->y > 0) newY--; break;
        case 1: if (w->y < g->h - 1) newY++; break;
        case 2: if (w->x > 0) newX--; break;
        case 3: if (w->x < g->w - 1) newX++; break;
    }

    // Update the walker's coordinates
    w->x = newX;
    w->y = newY;

    // If GridSetFlag is enabled, update the grid
    if (GridSetFlag) {
        GridSet(g, w->x, w->y, w->color);
    }
}

// Function to spawn green walker
void SpawnGreen(Walker w, Grid *g, const int R, float p, unsigned int* seed) {
    if ((float)mpi_safe_rand(seed) / RAND_MAX >= p)
        return;
            
    int greenFlag = 0;
    int a = -1, b = -1;

    // Find an empty spot near the walker
    for (int i = -R; i <= R; i++) {
        for (int j = -R; j <= R; j++) {
            int nx = w.x + i;
            int ny = w.y + j;

            // Check if within bounds
            if (nx >= 0 && nx < g->w && ny >= 0 && ny < g->h) {
                int cellValue = g->g[ny * g->w + nx];

                if (cellValue != 0) {  // Found a non-empty cell
                    greenFlag = 1;
                    a = nx;
                    b = ny;
                    goto exit_loops;  // Exit both loops immediately
                }
            }
        }
    }

exit_loops:
    if (greenFlag == 0)
        return;  // No valid location found

    Walker greenWalker = WalkerInit(a, b, GREEN); // GREEN (32)

    // Move the green walker inside the cluster
    int maxSteps = 100;  // Prevent infinite loops
    while (maxSteps--) {
        // Calculate movement without updating grid
        int direction = mpi_safe_rand(seed) % 4;
        int newX = greenWalker.x;
        int newY = greenWalker.y;
        
        switch (direction) {
            case 0: if (greenWalker.y > 0) newY--; break;
            case 1: if (greenWalker.y < g->h - 1) newY++; break;
            case 2: if (greenWalker.x > 0) newX--; break;
            case 3: if (greenWalker.x < g->w - 1) newX++; break;
        }
        
        greenWalker.x = newX;
        greenWalker.y = newY;
        
        int cellValue = g->g[greenWalker.y * g->w + greenWalker.x];

        if (cellValue == 0) {
            // Found an empty spot, turn the area around it into PURPLE
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nx = greenWalker.x + dx;
                    int ny = greenWalker.y + dy;
                    if (nx >= 0 && nx < g->w && ny >= 0 && ny < g->h) {
                        if (g->g[ny * g->w + nx] == 0)
                            g->g[ny * g->w + nx] = PURPLE;
                    }
                }
            }
            break;
        }
    }
}

// Detonation function for MPI
void detonate(Grid *g, int x, int y, int R, float pDet, float pPurp, unsigned int* seed) {
    int w = g->w;
    int h = g->h;
    int detFlag = 0;

    if ((float)mpi_safe_rand(seed) / RAND_MAX < pDet) {
        // Iterate over cells in radius R
        for (int i = -R; i <= R; i++) {
            for (int j = -R; j <= R; j++) {
                int nx = x + i;
                int ny = y + j;

                // Check if within bounds
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    int cellValue = g->g[ny * w + nx];

                    // Ensure YELLOW cells are not destroyed and destroy only opposing colors
                    if (cellValue != YELLOW) {
                        g->g[ny * w + nx] = 0; // Destroy the cell
                        detFlag = 1;
                    }
                }
            }
        }
    }

    // Now handle the radius R+1 and change colors based on probability p
    for (int i = -R - 1; i <= R + 1; i++) {
        for (int j = -R - 1; j <= R + 1; j++) {
            int nx = x + i;
            int ny = y + j;

            // Check if within bounds
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                int cellValue = g->g[ny * w + nx];
                
                // If it's an opposing color, change the color to purple with probability p
                if ((cellValue == RED) || (cellValue == BLUE)) {
                    // Generate a random float between 0 and 1, and compare with p
                    if ((float)mpi_safe_rand(seed) / RAND_MAX < pPurp) {
                        // Change color to purple
                        g->g[ny * w + nx] = PURPLE;
                    }
                }
            }
        }
    }

    // Destroy the cell at the center of the explosion
    if (detFlag) {
        g->g[y * w + x] = 0;
    }
}

// Function to place a walker on the grid
static void WalkerPlaceOnGrid(const Walker *w, Grid *g) {
    MP_ASSERT(w != NULL);
    MP_ASSERT(g != NULL);

    g->g[w->y * g->w + w->x] = w->color; // Store the walker's color in the grid
}

// Helper function to add a unique color to arrClrs
static void addUniqueColor(ArrClrs *arr, int color) {
    for (int i = 0; i < arr->size; ++i) {
        if (arr->clrs[i] == color) {
            return; // Color already exists
        }
    }
    if (arr->size < 4) { // Ensure we don't exceed array bounds
        arr->clrs[arr->size++] = color;
    }
}

// Function to check if the walker's color exists in ArrClrs
static int isWalkerClrMatch(const Walker *w, int color) {
    MP_ASSERT(w != NULL);

    for (int i = 0; i < w->arrClrs.size; ++i) {
        if (w->arrClrs.clrs[i] == color) { // Check if the walker's color matches
            return 1; 
        }
    }
    return 0;
}

// Function to check if a position is adjacent to a cluster
static int isAdjacentToCluster(const Grid *g, Walker *w) {
    // Reset arrClrs
    w->arrClrs.size = 0;
    
    int x = w->x;
    int y = w->y;
    int wGrid = g->w;
    int hGrid = g->h;
    
    int leftValue = 0, rightValue = 0, upValue = 0, downValue = 0;
    
    if (x > 0) leftValue = g->g[y * wGrid + (x - 1)];
    if (x < wGrid - 1) rightValue = g->g[y * wGrid + (x + 1)];
    if (y > 0) upValue = g->g[(y - 1) * wGrid + x];
    if (y < hGrid - 1) downValue = g->g[(y + 1) * wGrid + x];
    
    if (leftValue != 0) addUniqueColor(&w->arrClrs, leftValue);
    if (rightValue != 0) addUniqueColor(&w->arrClrs, rightValue);
    if (upValue != 0) addUniqueColor(&w->arrClrs, upValue);
    if (downValue != 0) addUniqueColor(&w->arrClrs, downValue);
    
    return w->arrClrs.size;
}

// Modified function to spawn a walker at the edge
static Walker spawnWalkerAtEdge(const Grid *g, int color, unsigned int* seed, int rank) {
    MP_ASSERT(g != NULL);

    int edge = mpi_safe_rand(seed) % 4; // Randomly choose an edge
    Walker w;

    switch (edge) {
        case 0: // Top edge
            w = WalkerInit(mpi_safe_rand(seed) % g->w, 0, color);
            break;
        case 1: // Bottom edge
            w = WalkerInit(mpi_safe_rand(seed) % g->w, g->h - 1, color);
            break;
        case 2: // Left edge
            w = WalkerInit(0, mpi_safe_rand(seed) % g->h, color);
            break;
        case 3: // Right edge
            w = WalkerInit(g->w - 1, mpi_safe_rand(seed) % g->h, color);
            break;
    }
    
    w.owner = rank;
    return w;
}

#endif // RANDOM_WALKER_H
