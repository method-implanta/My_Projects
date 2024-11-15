#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#define MAX_DEGREE 9
#define MAX_ITERATIONS 128
#define CONVERGENCE_THRESHOLD 1e-3
#define DIVERGENCE_THRESHOLD 1e10

typedef uint8_t TYPE_ATTR;
typedef uint8_t TYPE_CONV;

int L = 0;
int num_threads = 1;
int degree = 0;

TYPE_ATTR **attractors;
TYPE_CONV **convergences;
double roots_re[MAX_DEGREE];
double roots_im[MAX_DEGREE];

int *row_flags;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

typedef struct {
    int thread_id;
    int start_row;
    int end_row;
} thread_data_t;

int parse_args(int argc, char *argv[]);
void *compute_thread_func(void *arg);
void *write_thread_func(void *arg);
void initialize_roots();
void cleanup();

int main(int argc, char *argv[]) {
    if (parse_args(argc, argv) != 0) {
        fprintf(stderr, "Usage: %s -t<num_threads> -l<image_size> <degree>\n", argv[0]);
        return 1;
    }

    attractors = malloc(L * sizeof(TYPE_ATTR *));
    convergences = malloc(L * sizeof(TYPE_CONV *));
    if (!attractors || !convergences) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }
    for (int i = 0; i < L; i++) {
        attractors[i] = malloc(L * sizeof(TYPE_ATTR));
        convergences[i] = malloc(L * sizeof(TYPE_CONV));
        if (!attractors[i] || !convergences[i]) {
            fprintf(stderr, "Memory allocation failed.\n");
            cleanup();
            return 1;
        }
    }

    row_flags = calloc(L, sizeof(int));
    if (!row_flags) {
        fprintf(stderr, "Memory allocation failed.\n");
        cleanup();
        return 1;
    }

    initialize_roots();

    pthread_t *compute_threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = malloc(num_threads * sizeof(thread_data_t));
    if (!compute_threads || !thread_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        cleanup();
        return 1;
    }

    int rows_per_thread = L / num_threads;
    int extra_rows = L % num_threads;
    int current_row = 0;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_row = current_row;
        thread_data[i].end_row = current_row + rows_per_thread - 1;
        if (i < extra_rows) {
            thread_data[i].end_row++;
        }
        current_row = thread_data[i].end_row + 1;
        if (pthread_create(&compute_threads[i], NULL, compute_thread_func, &thread_data[i]) != 0) {
            fprintf(stderr, "Error creating compute thread %d.\n", i);
            cleanup();
            return 1;
        }
    }

    pthread_t write_thread;
    if (pthread_create(&write_thread, NULL, write_thread_func, NULL) != 0) {
        fprintf(stderr, "Error creating write thread.\n");
        cleanup();
        return 1;
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(compute_threads[i], NULL);
    }

    pthread_mutex_lock(&mutex);
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);

    pthread_join(write_thread, NULL);

    cleanup();
    free(compute_threads);
    free(thread_data);
    free(row_flags);

    return 0;
}

int parse_args(int argc, char *argv[]) {
    int arg_index = 1;
    while (arg_index < argc - 1) {
        if (strncmp(argv[arg_index], "-t", 2) == 0) {
            num_threads = atoi(argv[arg_index] + 2);
        } else if (strncmp(argv[arg_index], "-l", 2) == 0) {
            L = atoi(argv[arg_index] + 2);
        } else {
            return -1;
        }
        arg_index++;
    }

    if (arg_index != argc - 1) {
        return -1;
    }

    degree = atoi(argv[arg_index]);
    if (degree < 1 || degree > MAX_DEGREE) {
        fprintf(stderr, "Degree must be between 1 and %d.\n", MAX_DEGREE);
        return -1;
    }

    if (L <= 0 || num_threads <= 0) {
        fprintf(stderr, "Image size and number of threads must be positive integers.\n");
        return -1;
    }

    return 0;
}

void initialize_roots() {
    for (int k = 0; k < degree; k++) {
        double angle = 2 * M_PI * k / degree;
        roots_re[k] = cos(angle);
        roots_im[k] = sin(angle);
    }
}

void *compute_thread_func(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int start_row = data->start_row;
    int end_row = data->end_row;

    double x_min = -2.0, x_max = 2.0;
    double y_min = -2.0, y_max = 2.0;
    double delta = (x_max - x_min) / (L - 1);

    for (int y = start_row; y <= end_row; y++) {
        for (int x = 0; x < L; x++) {
            double z_re = x_min + x * delta;
            double z_im = y_max - y * delta;

            int iter = 0;
            TYPE_ATTR attr = 0;

            bool converged = false;

            while (iter < MAX_ITERATIONS) {
                double modulus_sq = z_re * z_re + z_im * z_im;
                if (modulus_sq < 1e-6 || fabs(z_re) > DIVERGENCE_THRESHOLD || fabs(z_im) > DIVERGENCE_THRESHOLD) {
                    break;
                }
                for (int k = 0; k < degree; k++) {
                    double delta_re = z_re - roots_re[k];
                    double delta_im = z_im - roots_im[k];
                    double dist_sq = delta_re * delta_re + delta_im * delta_im;
                    if (dist_sq < CONVERGENCE_THRESHOLD * CONVERGENCE_THRESHOLD) {
                        attr = k + 1;
                        converged = true;
                        break;
                    }
                }
                if (converged) {
                    break;
                }
                // Newton iteration
                switch (degree) {
                    case 1:
                        z_re = 1.0;
                        z_im = 0.0;
                        break;
                    case 2: {
                        // z = (z + 1/z) / 2
                        double denom = z_re * z_re + z_im * z_im;
                        if (denom == 0.0) {
                            break;
                        }
                        double z_re_inv = z_re / denom;
                        double z_im_inv = -z_im / denom;
                        z_re = (z_re + z_re_inv) * 0.5;
                        z_im = (z_im + z_im_inv) * 0.5;
                        break;
                    }
                    case 3: {
                        // z = (2 * z^3 + 1) / (3 * z^2)
                        double z_sq_re = z_re * z_re - z_im * z_im;
                        double z_sq_im = 2.0 * z_re * z_im;
                        double denom_re = 3.0 * z_sq_re;
                        double denom_im = 3.0 * z_sq_im;
                        double denom_mod_sq = denom_re * denom_re + denom_im * denom_im;
                        if (denom_mod_sq == 0.0) {
                            break;
                        }
                        double z_cu_re = z_re * z_sq_re - z_im * z_sq_im;
                        double z_cu_im = z_re * z_sq_im + z_im * z_sq_re;
                        double num_re = 2.0 * z_cu_re + 1.0;
                        double num_im = 2.0 * z_cu_im;
                        double z_re_new = (num_re * denom_re + num_im * denom_im) / denom_mod_sq;
                        double z_im_new = (num_im * denom_re - num_re * denom_im) / denom_mod_sq;
                        z_re = z_re_new;
                        z_im = z_im_new;
                        break;
                    }
                    case 5: {
                        // Optimized iteration for degree 5
                        double denom = z_re * z_re + z_im * z_im;
                        if (denom == 0.0) {
                            break;
                        }
                        double z_re_inv = z_re / denom;
                        double z_im_inv = -z_im / denom;
                        // Compute (1/z)^4
                        double z_re_inv_sq = z_re_inv * z_re_inv - z_im_inv * z_im_inv;
                        double z_im_inv_sq = 2.0 * z_re_inv * z_im_inv;
                        double z_re_inv_4 = z_re_inv_sq * z_re_inv_sq - z_im_inv_sq * z_im_inv_sq;
                        double z_im_inv_4 = 2.0 * z_re_inv_sq * z_im_inv_sq;
                        z_re = 0.2 * z_re_inv_4 + 0.8 * z_re;
                        z_im = 0.2 * z_im_inv_4 + 0.8 * z_im;
                        break;
                    }
                    case 7: {
                        // Optimized iteration for degree 7
                        double denom = z_re * z_re + z_im * z_im;
                        if (denom == 0.0) {
                            break;
                        }
                        double z_re_inv = z_re / denom;
                        double z_im_inv = -z_im / denom;
                        // Compute (1/z)^6
                        double z_re_inv_sq = z_re_inv * z_re_inv - z_im_inv * z_im_inv;
                        double z_im_inv_sq = 2.0 * z_re_inv * z_im_inv;
                        double z_re_inv_4 = z_re_inv_sq * z_re_inv_sq - z_im_inv_sq * z_im_inv_sq;
                        double z_im_inv_4 = 2.0 * z_re_inv_sq * z_im_inv_sq;
                        double z_re_inv_6 = z_re_inv_4 * z_re_inv_sq - z_im_inv_4 * z_im_inv_sq;
                        double z_im_inv_6 = z_re_inv_4 * z_im_inv_sq + z_im_inv_4 * z_re_inv_sq;
                        z_re = (1.0 / 7.0) * z_re_inv_6 + (6.0 / 7.0) * z_re;
                        z_im = (1.0 / 7.0) * z_im_inv_6 + (6.0 / 7.0) * z_im;
                        break;
                    }
                    default: {
                        // General case
                        // Newton iteration: z = z - (z^d - 1) / (d * z^(d - 1))
                        // Compute z^d and z^(d - 1)
                        double z_re_pow = z_re;
                        double z_im_pow = z_im;
                        double z_re_pow_prev = 1.0;
                        double z_im_pow_prev = 0.0;
                        for (int k = 1; k < degree; k++) {
                            double temp_re = z_re_pow * z_re - z_im_pow * z_im;
                            double temp_im = z_re_pow * z_im + z_im_pow * z_re;
                            z_re_pow_prev = z_re_pow;
                            z_im_pow_prev = z_im_pow;
                            z_re_pow = temp_re;
                            z_im_pow = temp_im;
                        }
                        // f = z^d - 1
                        double f_re = z_re_pow * z_re - z_im_pow * z_im - 1.0;
                        double f_im = z_re_pow * z_im + z_im_pow * z_re;
                        // f' = d * z^(d - 1)
                        double f_prime_re = degree * z_re_pow_prev;
                        double f_prime_im = degree * z_im_pow_prev;
                        double denom = f_prime_re * f_prime_re + f_prime_im * f_prime_im;
                        if (denom == 0.0) {
                            break;
                        }
                        // Newton step: z = z - f / f'
                        double delta_re = (f_re * f_prime_re + f_im * f_prime_im) / denom;
                        double delta_im = (f_im * f_prime_re - f_re * f_prime_im) / denom;
                        z_re = z_re - delta_re;
                        z_im = z_im - delta_im;
                        break;
                    }
                }
                iter++;
            }
            if (iter > MAX_ITERATIONS) {
                iter = MAX_ITERATIONS;
            }
            attractors[y][x] = attr;
            convergences[y][x] = iter;
        }
        row_flags[y] = 1;

        if ((y - start_row + 1) % 100 == 0) {
            pthread_mutex_lock(&mutex);
            pthread_cond_signal(&cond);
            pthread_mutex_unlock(&mutex);
        }
    }
    return NULL;
}

void *write_thread_func(void *arg) {
    char attr_filename[64];
    char conv_filename[64];
    snprintf(attr_filename, sizeof(attr_filename), "newton_attractors_x%d.ppm", degree);
    snprintf(conv_filename, sizeof(conv_filename), "newton_convergence_x%d.ppm", degree);

    FILE *attr_file = fopen(attr_filename, "wb");
    FILE *conv_file = fopen(conv_filename, "wb");
    if (!attr_file || !conv_file) {
        fprintf(stderr, "Error opening output files.\n");
        exit(1);
    }

    fprintf(attr_file, "P6\n%d %d\n255\n", L, L);
    fprintf(conv_file, "P6\n%d %d\n255\n", L, L);

    uint8_t colors[MAX_DEGREE + 1][3];
    colors[0][0] = colors[0][1] = colors[0][2] = 0; // Black for default attractor
    for (int k = 1; k <= degree; k++) {
        colors[k][0] = (k * 50) % 256;
        colors[k][1] = (k * 80) % 256;
        colors[k][2] = (k * 110) % 256;
    }

    int y = 0;
    while (y < L) {
        pthread_mutex_lock(&mutex);
        while (y < L && row_flags[y] == 0) {
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);

        while (y < L && row_flags[y] == 1) {
            uint8_t attr_row[L * 3];
            for (int x = 0; x < L; x++) {
                int attr = attractors[y][x];
                if (attr > degree) attr = 0;
                attr_row[x * 3] = colors[attr][0];
                attr_row[x * 3 + 1] = colors[attr][1];
                attr_row[x * 3 + 2] = colors[attr][2];
            }
            fwrite(attr_row, 1, L * 3, attr_file);

            uint8_t conv_row[L * 3];
            for (int x = 0; x < L; x++) {
                int gray = (int)(255.0 * convergences[y][x] / MAX_ITERATIONS);
                if (gray > 255) gray = 255;
                conv_row[x * 3] = conv_row[x * 3 + 1] = conv_row[x * 3 + 2] = (uint8_t)gray;
            }
            fwrite(conv_row, 1, L * 3, conv_file);

            y++;
        }
    }

    fclose(attr_file);
    fclose(conv_file);
    return NULL;
}

void cleanup() {
    if (attractors) {
        for (int i = 0; i < L; i++) {
            free(attractors[i]);
        }
        free(attractors);
    }
    if (convergences) {
        for (int i = 0; i < L; i++) {
            free(convergences[i]);
        }
        free(convergences);
    }
}

