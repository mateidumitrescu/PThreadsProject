// Author: APD team, except where source was noted
// Dumitrescu Rares Matei, 331CA

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048


#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

#define min(a, b) (a < b) ? a : b

// helpful struct for storing data which the threads will work with
typedef struct {
    int thread_id;
    int sigma;
    pthread_barrier_t *barrier;
    ppm_image *image;
    ppm_image *scaled_image;
    ppm_image **contour_map;
    unsigned char **grid;
    int NUM_THREADS;
} threads_struct;

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// march
void thread_march(threads_struct *threads_data) {
    int step_x, step_y;
    step_x = STEP;
    step_y = STEP;
    int thread_id = threads_data->thread_id;
    int p = threads_data->scaled_image->x / step_x;
    int q = threads_data->scaled_image->y / step_y;
    int nr_threads = threads_data->NUM_THREADS;

    int start = thread_id * (double) p / nr_threads;
    int end = min((thread_id + 1) * (double) p / nr_threads, p);

    // marching starts here
    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k =
            8 * threads_data->grid[i][j] +
            4 * threads_data->grid[i][j + 1] +
            2 * threads_data->grid[i + 1][j + 1] +
            1 * threads_data->grid[i + 1][j];
            update_image(threads_data->scaled_image,
            threads_data->contour_map[k], i * step_x, j * step_y);
        }
    }
}

// allocating memory for grid
unsigned char **grid_allocation(ppm_image *image, int step_x, int step_y) {
    int p = image->x / step_x;
    int q = image->y / step_y;

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char *));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory for grid\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory for grid\n");
            exit(1);
        }
    }

    return grid;
}

// function for grid using parallelism
void thread_sample_grid(threads_struct *threads_data) {
    int thread_id = threads_data->thread_id;
    int step_x = STEP;
    int step_y = STEP;
    int p = threads_data->scaled_image->x / step_x;
    int q = threads_data->scaled_image->y / step_y;
    int nr_threads = threads_data->NUM_THREADS;
    ppm_image *image = threads_data->scaled_image;
    unsigned char curr_color = 0;
    int sigma = threads_data->sigma;

    int start = thread_id * (double) p / nr_threads;
    int end = min((thread_id + 1) * (double) p / nr_threads, p);

    // sample grid starts here using parallelism
    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = image->data[i * step_x * image->y + j * step_y];

            curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                threads_data->grid[i][j] = 0;
            } else {
                threads_data->grid[i][j] = 1;
            }
        }
    }
    threads_data->grid[p][q] = 0;

    for (int i = start; i < end; i++) {
        ppm_pixel curr_pixel = image->data[i * step_x * image->y + image->x - 1];

        curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            threads_data->grid[i][q] = 0;
        } else {
            threads_data->grid[i][q] = 1;
        }
    }
    for (int j = 0; j < q; j++) {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * step_y];

        curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            threads_data->grid[p][j] = 0;
        } else {
            threads_data->grid[p][j] = 1;
        }
    }
}

// deviding image in the number of threads and rescaling it
void thread_rescale(threads_struct *threads_data) {
    uint8_t sample[3];

    int thread_id = threads_data->thread_id;
    int x = threads_data->scaled_image->x;
    int y = threads_data->scaled_image->y;
    int nr_threads = threads_data->NUM_THREADS;

    int start = thread_id * (double) x / nr_threads;
    int end = min((thread_id + 1) * (double) x / nr_threads, x);

    // bicubic
    for (int i = start; i < end; i++) {
        for (int j = 0; j < y; j++) {
            float u = (float)i / (float)(x - 1);
            float v = (float)j / (float)(y - 1);
            sample_bicubic(threads_data->image, u, v, sample);

            threads_data->scaled_image->data[i * threads_data->scaled_image->y + j].red = sample[0];
            threads_data->scaled_image->data[i * threads_data->scaled_image->y + j].green = sample[1];
            threads_data->scaled_image->data[i * threads_data->scaled_image->y + j].blue = sample[2];
        }
    }

}

// actual function that will split work for threads
// this is the function that is passed as argument for pthread_create
void *start_thread(void *arg) {
    threads_struct *threads_data = (threads_struct *)arg;
    
    // rescaling first if needed
    if (threads_data->image->x > RESCALE_X ||
        threads_data->image->y > RESCALE_Y) {
            thread_rescale(threads_data);
    }

    // sample grid
    thread_sample_grid(threads_data);

    // waiting for all threads to finish grid before march
    pthread_barrier_wait(threads_data->barrier);

    // march
    thread_march(threads_data);


    pthread_exit(NULL);
 
}


// allocate memory if needed for newly scaled image
ppm_image *rescale_allocation() {
    // alloc memory for image
    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    new_image->x = RESCALE_X;
    new_image->y = RESCALE_Y;

    new_image->data = (ppm_pixel*)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    return new_image;
}

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *scaled_image,
                    ppm_image *image, ppm_image **contour_map,
                    unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

     for (int i = 0; i <= scaled_image->x / step_x; i++) {
         free(grid[i]);
    }
    free(grid);

    // freeing newly scaled image only if it was allocated
    if (image->x > RESCALE_X || image->y > RESCALE_Y) {
        free(scaled_image->data);
        free(scaled_image);
    }

    free(image->data);
    free(image);

}


int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    // getting the number of threads
    int NUM_THREADS = atoi(argv[3]);
    pthread_t threads[NUM_THREADS];
    threads_struct work_data[NUM_THREADS];

    ppm_image *image = read_ppm(argv[1]);
    int step_x = STEP;
    int step_y = STEP;

    // 0. Initialize contour map
    ppm_image **contour_map = init_contour_map();
    ppm_image *scaled;

    // allocate memory for newly scaled image if needed
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        scaled = image;
        scaled->x = image->x;
        scaled->y = image->x;
    } else {
        scaled = rescale_allocation();
    }

    unsigned char **grid = NULL;
    grid = grid_allocation(scaled, step_x, step_y);

    // our barrier
    pthread_barrier_t *barrier = malloc(sizeof(pthread_barrier_t));
    if (!barrier) {
        fprintf(stderr, "Malloc failed on barrier\n");
        exit(1);
    }

    // create barrier
    pthread_barrier_init(barrier, NULL, NUM_THREADS);

    // assigning data to each struct in our array work_data
    for (int i = 0; i < NUM_THREADS; i++) {
        work_data[i].contour_map = contour_map;
        work_data[i].image = image;
        work_data[i].scaled_image = scaled;
        work_data[i].grid = grid;
        work_data[i].NUM_THREADS = NUM_THREADS;
        work_data[i].barrier = barrier;
        work_data[i].sigma = SIGMA;
    }

    // starting working with our threads here
    for (int i = 0; i < NUM_THREADS; i++) {
        work_data[i].thread_id = i;
        int r = pthread_create(&threads[i], NULL, *start_thread, &work_data[i]);
        if (r) {
            fprintf(stderr, "Starting thread error.\n");
            exit(1);
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        int r = pthread_join(threads[i], NULL);
        if (r) {
            fprintf(stderr, "Unable to join thread\n");
            exit(1);
        }
    }

    // 4. Write output
    write_ppm(work_data[0].scaled_image, argv[2]);

    free_resources(work_data[0].scaled_image, image,
                   contour_map, work_data[0].grid, step_x);

    pthread_barrier_destroy(barrier);

    free(barrier);

    return 0;
}
