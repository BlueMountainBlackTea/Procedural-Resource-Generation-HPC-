#ifndef GRID_H
#define GRID_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Color definitions
#define RED 31
#define GREEN 32
#define YELLOW 33
#define BLUE 34
#define PURPLE 35

// Grid structure
typedef struct {
    int w;
    int h;
    int *g;
} Grid;

// Counter structure to track color counts
typedef struct {
    int RED;
    int YELLOW;
    int BLUE;
    int PURPLE;
} Counter;

// Initialize grid
static inline Grid GridInit(const int w, const int h) {
    assert(w > 0 && h > 0);
    Grid G = {w, h, (int*)malloc(sizeof(int) * w * h)};
    // Initialize grid to zeros
    memset(G.g, 0, sizeof(int) * w * h);
    return G;
}

// Free grid memory
static inline void freeGrid(Grid X) {
    free(X.g);
    X.g = NULL;
}

// Set grid value
static void GridSet(Grid *g, int x, int y, int value) {
    assert(g != NULL);
    g->g[y * g->w + x] = value;
}

// Display grid with ANSI colors
void GridDisplay(const Grid *g) {
    assert(g != NULL);

    const char *RESET = "\033[0m";
    const char *RED_COLOR = "\033[31m";
    const char *BLUE_COLOR = "\033[34m";
    const char *PURPLE_COLOR = "\033[35m";
    const char *YELLOW_COLOR = "\033[33m";

    for (int y = 0; y < g->h; y++) {
        for (int x = 0; x < g->w; x++) {
            int cell = g->g[y * g->w + x];
            if (cell == 0) {
                printf("  ");
            } else if (cell == RED) {
                printf("%sR %s", RED_COLOR, RESET);
            } else if (cell == BLUE) {
                printf("%sB %s", BLUE_COLOR, RESET);
            } else if (cell == PURPLE) {
                printf("%sP %s", PURPLE_COLOR, RESET);
            } else if (cell == YELLOW) {
                printf("%sY %s", YELLOW_COLOR, RESET);
            } else {
                printf("? ");
            }
        }
        printf("\n");
    }
}

// Print counter statistics
void printCounter(Counter x) {
    printf("__STATS___\n");
    printf("RED: %d\n", x.RED);
    printf("BLUE: %d\n", x.BLUE);
    printf("PURPLE: %d\n", x.PURPLE);
    printf("YELLOW: %d\n", x.YELLOW);
}

// Export grid to PPM file
Counter export_grid_to_ppm(const Grid* g, int width, int height, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open file");
        Counter error = {-1, -1, -1, -1};
        return error;
    }

    // PPM header
    fprintf(file, "P3\n%d %d\n255\n", width, height);

    Counter counter = {0};

    // Map grid values to RGB colors
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int cell = g->g[y * width + x];
            int r = 173, g_val = 216, b = 230; // Default: light blue

            if (cell == RED) {      // Red
                r = 255; g_val = 0; b = 0;
                counter.RED++;
            } else if (cell == BLUE) { // Blue
                r = 0; g_val = 0; b = 255;
                counter.BLUE++;
            } else if (cell == PURPLE) { // Purple
                r = 128; g_val = 0; b = 128;
                counter.PURPLE++;
            } else if (cell == YELLOW) { // Yellow
                r = 255; g_val = 255; b = 0;
                counter.YELLOW++;
            }

            // Write RGB values to file
            fprintf(file, "%d %d %d ", r, g_val, b);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Grid exported to %s\n", filename);
    return counter;
}

#endif // GRID_H