#include <stdio.h>
#include <stdlib.h>
#include <time.h>



#define WIDTH 100        // Width of the grid
#define HEIGHT 100       // Height of the grid
#define NUM_PARTICLES 1000 // Number of particles to simulate


// Directions for the random walk (up, down, left, right)
int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

// Grid to represent the map (0 = empty, 1 = occupied)
int grid[WIDTH][HEIGHT];

// Function to initialize the grid
void initialize_grid() {
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            grid[i][j] = 0;  // Set all cells to empty
        }
    }
}

// Function to check if a position is within the grid boundaries
int is_valid(int x, int y) {
    return (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT);
}

// Function to simulate a random walk of a particle
void random_walk(int *x, int *y) {
    int direction = rand() % 4; // Randomly choose a direction
    *x += dx[direction];
    *y += dy[direction];
}

// Function to check if the particle is adjacent to an occupied cell
int is_adjacent_to_cluster(int x, int y) {
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (is_valid(nx, ny) && grid[nx][ny] == 1) {
            return 1;  // Found an adjacent occupied cell
        }
    }
    return 0;  // No adjacent occupied cells
}

// Function to simulate diffusion limited aggregation
void simulate_dla() {
    int x, y;

    // Start by placing the initial cluster at the center
    grid[WIDTH / 2][HEIGHT / 2] = 1;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        // Place particle at a random position
        x = rand() % WIDTH;
        y = rand() % HEIGHT;

        // Perform a random walk until the particle sticks
        while (1) {
            random_walk(&x, &y);

            // Make sure the particle stays within the grid
            if (!is_valid(x, y)) {
                x = rand() % WIDTH;
                y = rand() % HEIGHT;
            }

            // If adjacent to an occupied cell, the particle sticks
            if (is_adjacent_to_cluster(x, y)) {
                grid[x][y] = 1;  // The particle sticks to the cluster
                break;
            }
        }
    }
}

// Function to print the grid (map)
void print_grid() {
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            if (grid[i][j] == 1) {
                printf("#");  // Represent occupied cells with '#'
            } else {
                printf(" ");  // Represent empty cells with a space
            }
        }
        printf("\n");
    }
}

int main() {
    srand(time(0));  // Initialize random seed

    // Initialize the grid and run DLA simulation
    initialize_grid();
    simulate_dla();

    // Print the generated map
    print_grid();

    return 0;
}
