
void compute_concentration_and_inv_freq_getp(float base, int head_dim,
                                             float scaling_factor,
                                             float initial_context_length,
                                             float ntk_beta, float ntk_alpha,
                                             float *concentration_out,
                                             float *inv_freq_out // length head_dim/2
)
{
    int d_half = head_dim / 2;

    // freq[i] = base ** (i / head_dim)
    float *freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++)
    {
        freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
    }

    float concentration;
    if (scaling_factor > 1.0f)
    {
        // YaRN concentration
        concentration = 0.1f * logf(scaling_factor) + 1.0f;

        // NTK by parts
        float low = d_half *
                    logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) /
                    logf(base);
        float high = d_half *
                     logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                     logf(base);

        assert(0 < low && low < high && high < d_half - 1);

        // interpolation = 1 / (scaling_factor * freq)
        // extrapolation = 1 / freq
        for (int i = 0; i < d_half; i++)
        {
            float interpolation = 1.0f / (scaling_factor * freq[i]);
            float extrapolation = 1.0f / freq[i];

            float ramp = ((float)i - low) / (high - low);
            if (ramp < 0)
                ramp = 0;
            if (ramp > 1)
                ramp = 1;

            float mask = 1.0f - ramp;
            inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
        }
    }
    else
    {
        concentration = 1.0f;
        for (int i = 0; i < d_half; i++)
        {
            inv_freq_out[i] = 1.0f / freq[i];
        }
    }

    *concentration_out = concentration;

    free(freq);
}

void compute_cos_sin_getp(int pos, // position index
                          float base, int head_dim, float scaling_factor,
                          float initial_context_length, float ntk_beta,
                          float ntk_alpha,
                          float *cos_out, // shape: head_dim/2
                          float *sin_out  // shape: head_dim/2
)
{
    int d_half = head_dim / 2;

    // Get concentration + inv_freq
    float concentration;
    float *inv_freq = (float *)malloc(d_half * sizeof(float));

    compute_concentration_and_inv_freq_getp(base, head_dim, scaling_factor,
                                            initial_context_length, ntk_beta,
                                            ntk_alpha, &concentration, inv_freq);

    // Compute cos and sin for this position
    for (int j = 0; j < d_half; j++)
    {
        float val = (float)pos * inv_freq[j];
        cos_out[j] = cosf(val) * concentration;
        sin_out[j] = sinf(val) * concentration;
    }

    free(inv_freq);
}
