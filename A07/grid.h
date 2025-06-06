
#ifndef GRID_H
#define GRID_H

#include "MP_H.h"

typedef struct{
    int w;
    int h;
    int *g;
} Grid;


typedef struct{
    int RED;
    // int GREEN;
    int YELLOW;
    int BLUE;
    int PURPLE;
} Counter;

// constructor
static inline Grid GridInit(const int w, const int h) {
    MP_ASSERT(w > 0 && h > 0);
    
    Grid G = {w, h, (int*) malloc(sizeof(int) * w * h)};
    if (!G.g) { // Check for allocation failure
        perror("Memory allocation failed");
        exit(1);
    }

    // Fast zeroing of memory
    memset(G.g, 0, sizeof(int) * w * h);

    return G;
}

//destructor 
static inline void freeGrid(Grid X)
{
    MP_FREE(X.g);
    X.g = NULL;
}

// Function to set a value on the grid at the given position
static void GridSet(Grid *g, int x, int y, int value) {
    MP_ASSERT(g != NULL);
    // Set the value at the specified grid position
    g->g[y * g->w + x] = value;
}


// Function to display the grid with color values using ANSI escape codes
void GridDisplay(const Grid *g) {
    MP_ASSERT(g != NULL);

    // ANSI escape codes for text colors
    const char *RESET = "\033[0m";
    const char *RED = "\033[31m";    // Red color
    const char *BLUE = "\033[34m";   // Blue color
    const char *PURPLE = "\033[35m"; // Purple color
    const char *YELLOW = "\033[33m"; // Yellow color

    for (int y = 0; y < g->h; y++) {
        for (int x = 0; x < g->w; x++) {
            int cell = g->g[y * g->w + x];
            if (cell == 0) {
                printf("  "); // Empty space (no color)
            } else if (cell == 31) {
                printf("%sR %s", RED, RESET); // Red cluster
            } else if (cell == 34) {
                printf("%sB %s", BLUE, RESET); // Blue cluster
            } else if (cell == 35) {
                printf("%sP %s", PURPLE, RESET); // Purple cluster
            } else if (cell == 33) {
                printf("%sY %s", YELLOW, RESET); // Yellow cluster
            } else {
                printf("? "); // Unknown state (no color)
            }
        }
        printf("\n");
    }
}

void printCounter(Counter x)
{
    printf("__STATS___\n");
    printf("RED:%d\n", x.RED);
    printf("BLUE:%d\n", x.BLUE);
    printf("PURPLE:%d\n", x.PURPLE);
    printf("YELLOW:%d\n", x.YELLOW);
}

Counter export_grid_to_ppm(const Grid* g, int width, int height, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open file");
        return *(Counter*) -1;
    }

    // PPM header
    fprintf(file, "P3\n%d %d\n255\n", width, height);

    Counter counter = {0};

    // Map grid values to RGB colors
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int cell = g->g[y * width + x];
            int r = 173, g = 216, b = 230; // Default: white

            if (cell == 31) {      // Red
                r = 255; g = 0; b = 0;
                counter.RED++;
            } else if (cell == 34) { // Blue
                r = 0; g = 0; b = 255;
                counter.BLUE++;
            } else if (cell == 35) { // Purple
                r = 128; g = 0; b = 128;
                counter.PURPLE++;
            } else if (cell == 33) { // Yellow
                r = 255; g = 255; b = 0;
                counter.YELLOW++;
            } else if (cell == 0) {  // Empty space (white)
                // r = 255; g = 255; b = 255;
                r = 173; g = 216; b = 230;
            }

            // Write RGB values to file
            fprintf(file, "%d %d %d ", r, g, b);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Grid exported to %s\n", filename);

    return counter;
}


// Custom grid display with colored output
static inline void GridDisplayColored(const Grid g) {
    MP_ASSERT(g.g != NULL);
    for (int i = 0; i < g.h; ++i) {
        for (int j = 0; j < g.w; ++j) {
            int cell = g.g[i * g.w + j];
            if (cell != 0)
                printf("\033[1;%dm#\033[0m", cell); // Use the stored color
            else
                printf(" ");
        }
        printf("\n");
    }
}



#endif // GRID_H