#ifndef GRID_H
#define GRID_H

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define MP_ASSERT assert

typedef struct {
    int w;
    int h;
    int *g;
} Grid;

// Function to initialize a grid
static Grid GridInit(int w, int h) {
    MP_ASSERT(w > 0);
    MP_ASSERT(h > 0);
    
    Grid g = {
        .w = w,
        .h = h,
        .g = (int *)calloc(w * h, sizeof(int))
    };
    
    MP_ASSERT(g.g != NULL);
    return g;
}

// Function to set a value in the grid
static void GridSet(Grid *g, int x, int y, int val) {
    MP_ASSERT(g != NULL);
    MP_ASSERT(x >= 0 && x < g->w);
    MP_ASSERT(y >= 0 && y < g->h);
    
    g->g[y * g->w + x] = val;
}

// Function to free grid memory
static void freeGrid(Grid g) {
    free(g.g);
}

// Function to export the grid to a PPM file
void export_grid_to_ppm(const Grid *g, int width, int height, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P3\n%d %d\n255\n", width, height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int val = g->g[y * width + x];
            
            switch (val) {
                case 0:
                    fprintf(fp, "0 0 0 "); // Black for empty space
                    break;
                case 31: // RED
                    fprintf(fp, "255 0 0 ");
                    break;
                case 32: // GREEN
                    fprintf(fp, "0 255 0 ");
                    break;
                case 33: // YELLOW
                    fprintf(fp, "255 255 0 ");
                    break;
                case 34: // BLUE
                    fprintf(fp, "0 0 255 ");
                    break;
                case 35: // PURPLE
                    // fprintf(fp, "255 0 255 ");
		    fprintf(fp, "0 0 0 ");
                    break;
                default:
                    fprintf(fp, "128 128 128 "); // Grey for unknown colors
            }
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
}

#endif // GRID_H