#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MAX_DISTANCE 35.0
#define DISTANCE_SCALE 100
#define MAX_BINS ((int)(MAX_DISTANCE * DISTANCE_SCALE) + 1)

typedef struct {
    float x;
    float y;
    float z;
} Point;

int parse_arguments(int argc, char *argv[], int *num_threads) {
    *num_threads = 1;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-t", 2) == 0) {
            *num_threads = atoi(argv[i] + 2);
            if (*num_threads <= 0) {
                fprintf(stderr, "Invalid number of threads specified.\n");
                return -1;
            }
        }
    }
    return 0;
}

Point *read_points(const char *filename, int *num_points) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open the file 'cells'.\n");
        return NULL;
    }

    int capacity = 400000;
    Point *points = (Point *)malloc(capacity * sizeof(Point));
    if (!points) {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(file);
        return NULL;
    }

    char line[100];
    int count = 0;
    while (fgets(line, sizeof(line), file)) {
        if (count >= capacity) {
            capacity *= 2;
            if (capacity * sizeof(Point) > 5 * 1024 * 1024) { // 5 MiB limit
                fprintf(stderr, "Memory limit exceeded.\n");
                free(points);
                fclose(file);
                return NULL;
            }
            points = (Point *)realloc(points, capacity * sizeof(Point));
            if (!points) {
                fprintf(stderr, "Memory reallocation failed.\n");
                fclose(file);
                return NULL;
            }
        }
        sscanf(line, "%f %f %f", &points[count].x, &points[count].y, &points[count].z);
        count++;
    }
    fclose(file);
    *num_points = count;
    return points;
}

int main(int argc, char *argv[]) {
    int num_threads;
    if (parse_arguments(argc, argv, &num_threads) != 0) {
        return EXIT_FAILURE;
    }
    omp_set_num_threads(num_threads);

    int num_points;
    Point *points = read_points("cells", &num_points);
    if (!points) {
        return EXIT_FAILURE;
    }

    int threads = num_threads;
    long **histograms = malloc(threads * sizeof(long *));
    for (int i = 0; i < threads; i++) {
        histograms[i] = calloc(MAX_BINS, sizeof(long));
        if (!histograms[i]) {
            fprintf(stderr, "Memory allocation failed.\n");
            free(points);
            return EXIT_FAILURE;
        }
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *local_histogram = histograms[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < num_points; i++) {
            for (int j = i + 1; j < num_points; j++) {
                float dx = points[i].x - points[j].x;
                float dy = points[i].y - points[j].y;
                float dz = points[i].z - points[j].z;
                float distance = sqrtf(dx * dx + dy * dy + dz * dz);

                int bin_index = (int)(distance * DISTANCE_SCALE + 0.5f);
                if (bin_index >= 0 && bin_index < MAX_BINS) {
                    local_histogram[bin_index]++;
                }
            }
        }
    }

    long *global_histogram = calloc(MAX_BINS, sizeof(long));
    if (!global_histogram) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(points);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < threads; i++) {
        for (int j = 0; j < MAX_BINS; j++) {
            global_histogram[j] += histograms[i][j];
        }
        free(histograms[i]);
    }
    free(histograms);
    free(points);

    for (int i = 0; i < MAX_BINS; i++) {
        if (global_histogram[i] > 0) {
            printf("%05.2f %ld\n", i / (float)DISTANCE_SCALE, global_histogram[i]);
        }
    }
    free(global_histogram);

    return EXIT_SUCCESS;
}