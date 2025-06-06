#ifndef RANDOM_WALKER_H
#define RANDOM_WALKER_H

#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For seeding random number generator

#include "grid.h"

#define RED 31
#define GREEN 32
#define YELLOW 33
#define BLUE 34
#define PURPLE 35


typedef struct 
{
    int clrs[4];
    int size;
} ArrClrs;


typedef struct {
    int x;
    int y;
    int color; // Color associated with the walker
    ArrClrs arrClrs;
} Walker;

// Function to initialize a walker
static Walker WalkerInit(int x, int y, int color) {
    Walker w = {
        .x = x,
        .y = y,
        .color = color,
        .arrClrs = {
            .clrs = {0}, // Initialize the array to 0
            .size = 0    // Initialize size to 0
        }
    };
    return w;
}


// Function to move the walker randomly and update the grid
static void WalkerMove(Walker *w, Grid *g, int GridSetFlag) {
    MP_ASSERT(w != NULL);
    MP_ASSERT(g != NULL);

    if (GridSetFlag)
        // Clear the old position of the walker
        GridSet(g, w->x, w->y, 0);

    // Move the walker randomly
    int direction = rand() % 4; // 0=up, 1=down, 2=left, 3=right
    int newX = w->x;
    int newY = w->y;

    switch (direction) {
        case 0: if (w->y > 0) newY--; break; // Up
        case 1: if (w->y < g->h - 1) newY++; break; // Down
        case 2: if (w->x > 0) newX--; break; // Left
        case 3: if (w->x < g->w - 1) newX++; break; // Right
    }

    // Update the walker's position
    w->x = newX;
    w->y = newY;

    if (GridSetFlag)
        // Set the new position of the walker
        GridSet(g, w->x, w->y, w->color);
}

void SpawnGreen(Walker w, Grid *g, const int R, int p)
{
    if ((float)rand() / RAND_MAX >= p)
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
        WalkerMove(&greenWalker, g, 0);

        if (g->g[greenWalker.y * g->w + greenWalker.x] == 0) {
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


// Function to detonate a 3x3 area around a position
void detonate(Grid *g, int x, int y, int R, float pDet, float pPurp) {
    int w = g->w;
    int h = g->h;
    int detFlag = 0;


    if ( (float)rand() / RAND_MAX < pDet)
    {
        // Iterate over cells in radius R
        for (int i = -R; i <= R; i++) {
            for (int j = -R; j <= R; j++) {
                int nx = x + i;
                int ny = y + j;

                // Check if within bounds
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    int cellValue = g->g[ny * w + nx];

                    // Ensure YELLOW cells are not destroyed and destroy only opposing colors
                    if (cellValue != YELLOW)
                    {
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
                if ((cellValue == RED ) ||
                    (cellValue == BLUE )) {

                    // Generate a random float between 0 and 1, and compare with p
                    if ((float)rand() / RAND_MAX < pPurp) {
                        // Change color to purple
                        g->g[ny * w + nx] = PURPLE;
                    }
                    
                }
            }
        }
    }

    // Destroy the cell at the center of the explosion
    if (detFlag)
    {
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
static inline int isAdjacentToCluster(const Grid *g, Walker *w) {
    MP_ASSERT(g != NULL);
    MP_ASSERT(g->g != NULL);
    MP_ASSERT(w != NULL);

    int wGrid = g->w;
    int hGrid = g->h;

    // Reset the Walkers' arrClrs
    w->arrClrs.size = 0;

    int x = w->x;
    int y = w->y;

    // Check Left
    if (x > 0 && g->g[y * wGrid + (x - 1)] != 0) {
        addUniqueColor(&w->arrClrs, g->g[y * wGrid + (x - 1)]);
    }
    // Check Right
    if (x < wGrid - 1 && g->g[y * wGrid + (x + 1)] != 0) {
        addUniqueColor(&w->arrClrs, g->g[y * wGrid + (x + 1)]);
    }
    // Check Up
    if (y > 0 && g->g[(y - 1) * wGrid + x] != 0) {
        addUniqueColor(&w->arrClrs, g->g[(y - 1) * wGrid + x]);
    }
    // Check Down
    if (y < hGrid - 1 && g->g[(y + 1) * wGrid + x] != 0) {
        addUniqueColor(&w->arrClrs, g->g[(y + 1) * wGrid + x]);
    }

    // Return the number of unique colors found
    return w->arrClrs.size;
}


// Function to spawn a walker at the edges
static Walker spawnWalkerAtEdge(const Grid *g, int color) {
    MP_ASSERT(g != NULL);

    int edge = rand() % 4; // Randomly choose an edge: 0=top, 1=bottom, 2=left, 3=right
    Walker w;

    switch (edge) {
        case 0: // Top edge
            w = WalkerInit(rand() % g->w, 0, color);
            break;
        case 1: // Bottom edge
            w = WalkerInit(rand() % g->w, g->h - 1, color);
            break;
        case 2: // Left edge
            w = WalkerInit(0, rand() % g->h, color);
            break;
        case 3: // Right edge
            w = WalkerInit(g->w - 1, rand() % g->h, color);
            break;
    }

    return w;
}

#endif // RANDOM_WALKER_H
