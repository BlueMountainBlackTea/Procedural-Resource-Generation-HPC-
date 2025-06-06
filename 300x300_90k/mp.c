
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  

#include "grid.h"
#include "walker.h"


#define MAX_WALKERS 1000 // Maximum number of walkers

// Function to perform DLA generation on a grid
void generateDLA(Grid *g, int maxWalkers) {
    srand(time(NULL)); // Seed the random number generator

    for (int i = 0; i < maxWalkers; ++i) {
        // Alternate between spawning red and blue walkers

        if (i % 10000 == 0) {
            printf("At walker %d\n", i);
            // Counter tmp = export_grid_to_ppm(g, g->w, g->h, "./output.ppm");
            // printCounter(tmp);
        //     GridDisplayColored(*g);
        }

        int color = (i % 2 == 0) ? RED : BLUE;
        Walker w = spawnWalkerAtEdge(g, color);

        // Random walk until the walker interacts with the grid
        while (1) {
            // Move the walker and update the grid
            WalkerMove(&w, g, 1);

            // Check if the walker is adjacent to any cluster
            int cellValue;
            if ((cellValue = isAdjacentToCluster(g, &w))) {

                if (isWalkerClrMatch(&w, YELLOW) || isWalkerClrMatch(&w, PURPLE) || isWalkerClrMatch(&w, w.color)) {
                    // Same color: form a cluster
                    WalkerPlaceOnGrid(&w, g);
                    
                    // Spawn a green walker inside the cluster
                    SpawnGreen(w,g,1,0.1);

                    
                } else {
                    // Detonate at the current walker's position
                    if (i < MAX_WALKERS/2)
                    {
                        detonate(g, w.x, w.y, 1, 1.0, 0.2);
                    }
                    else
                    {
                        detonate(g, w.x, w.y, 2, 0.4, 0.1);
                    }
                    
                }
                break;
            }
        }
    }
}


int main() {
    int width = 40;
    int height = 10;

    // Generate multiple DLA maps
    Grid g1 = GridInit(width, height);

    // Points
    g1.g[(height / 2) * width + (width / 2)] = YELLOW; // seed1 for g1
    g1.g[(height / 2) * width + (width / 2) + 1] = YELLOW; // seed1 for g2


    // GridDisplayColored(g1);


    generateDLA(&g1, MAX_WALKERS);


    GridDisplayColored(g1);

    // Export the final fused grid
    // printf("Exporting grid:\n");
    // Counter stats = export_grid_to_ppm(&g1, g1.w, g1.h, "./output.ppm");
    // printCounter(stats);
    // Free memory
    freeGrid(g1);

    return 0;
}
