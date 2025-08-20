
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

void apply_rotary_emb_batch_getp(float *x, float *cos, float *sin, int batch_size,
                                 int *positions, int n_heads, int head_dim)
{
    int half = head_dim / 2;

#pragma omp parallel for
    for (int b = 0; b < batch_size; b++)
    {
        int pos = positions[b];
        float *x_b = x + b * n_heads * head_dim;
        float *cos_pos = cos + pos * half;
        float *sin_pos = sin + pos * half;

        for (int h = 0; h < n_heads; h++)
        {
            for (int i = 0; i < half; i++)
            {
                // Indexing: batch b, head h, dim i
                float x1 = x_b[h * head_dim + i];        // first half
                float x2 = x_b[h * head_dim + half + i]; // second half

                float c = cos_pos[i];
                float s = sin_pos[i];

                float o1 = x1 * c - x2 * s;
                float o2 = x2 * c + x1 * s;

                x_b[h * head_dim + i] = o1;
                x_b[h * head_dim + half + i] = o2;
            }
        }
    }
}
