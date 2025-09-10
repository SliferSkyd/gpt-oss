
#pragma once
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <vector>
#include <mutex>
// HIP error checking macro
#define HIP_CHECK(call)                                                                                 \
    do                                                                                                  \
    {                                                                                                   \
        hipError_t error = call;                                                                        \
        if (error != hipSuccess)                                                                        \
        {                                                                                               \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

std::mutex debug_mutex;
// Variadic macro hỗ trợ format như printf
#define THREAD_DEBUG(fmt, ...)                             \
    do                                                     \
    {                                                      \
        HIP_CHECK(hipDeviceSynchronize());                 \
        std::lock_guard<std::mutex> lock(debug_mutex);     \
        std::printf("[Thread %lu] " fmt,                   \
                    (unsigned long)(omp_get_thread_num()), \
                    ##__VA_ARGS__);                        \
        std::fflush(stdout);                               \
    } while (0)

// Float to bfloat16 conversion functions using library function
void convert_float_array_to_bfloat16(const float *src, __hip_bfloat16 *dst, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        dst[i] = __float2bfloat16(src[i]);
    }
}

std::string debug(float *d_val, int size = 1)
{
    float *h_val = (float *)malloc(size * sizeof(float));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_val, d_val, size * sizeof(float), hipMemcpyDeviceToHost));
    std::stringstream ss;

    for (int i = 0; i < size; i++)
    {
        ss << std::fixed << std::setprecision(6) << h_val[i] << " ";
    }
    free(h_val);
    return ss.str();
}

//------------------------------------------------------------------------------//
// Timer class for measuring execution time                                     //
//------------------------------------------------------------------------------//

// Global storage for profiling data
struct ProfileEntry
{
    std::string name;
    float time_ms;
};

class ProfileManager
{
private:
    static std::vector<ProfileEntry> entries;
    static std::mutex entries_mutex;
    static std::string output_filename;

public:
    static void add_entry(const std::string &name, float time_ms)
    {
        std::lock_guard<std::mutex> lock(entries_mutex);
        entries.push_back({name, time_ms});
    }

    static void set_output_file(const std::string &filename)
    {
        output_filename = filename;
    }

    static void write_profile_info()
    {
        std::lock_guard<std::mutex> lock(entries_mutex);

        if (entries.empty())
            return;

        std::string actual_filename = output_filename;
        if (actual_filename.empty())
        {
            // Try to get SLURM_JOB_ID first
            const char *job_id = std::getenv("SLURM_JOB_ID");
            if (job_id)
            {
                actual_filename = "logs/times_job_" + std::string(job_id) + ".csv";
            }
            else
            {
                // Fall back to timestamp
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch()) %
                          1000;

                std::stringstream ss;
                ss << "logs/times_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
                ss << "_" << std::setfill('0') << std::setw(3) << ms.count() << ".csv";
                actual_filename = ss.str();
            }
        }
        else if (actual_filename.find('/') == std::string::npos)
        {
            // If filename doesn't contain path, add logs/ prefix
            actual_filename = "logs/" + actual_filename;
        }

        std::ofstream log_file(actual_filename, std::ios::app);
        if (log_file.tellp() == 0)
        {
            log_file << "name,time_ms\n";
        }

        for (const auto &entry : entries)
        {
            log_file << entry.name << "," << entry.time_ms << "\n";
        }

        log_file.close();
        entries.clear();
    }

    static void clear()
    {
        std::lock_guard<std::mutex> lock(entries_mutex);
        entries.clear();
    }
};

// Static member definitions
std::vector<ProfileEntry> ProfileManager::entries;
std::mutex ProfileManager::entries_mutex;
std::string ProfileManager::output_filename;

// Wrapper function for easy access
void write_profile_info()
{
    ProfileManager::write_profile_info();
}

struct Timer
{
    // hip events
    hipEvent_t hip_start, hip_stop;
    // CPU timing
    std::chrono::high_resolution_clock::time_point cpu_start;

    std::string name;
    bool use_hip;
    bool write_immediately;

    Timer(const std::string &timer_name, bool is_hip = false, const std::string &filename = "", bool immediate_write = false)
        : name(timer_name), use_hip(is_hip), write_immediately(immediate_write)
    {
        if (!filename.empty())
        {
            ProfileManager::set_output_file(filename);
        }

        if (use_hip)
        {
            hipEventCreate(&hip_start);
            hipEventCreate(&hip_stop);
        }
    }

    ~Timer()
    {
        if (use_hip)
        {
            hipEventDestroy(hip_start);
            hipEventDestroy(hip_stop);
        }
    }

    void start()
    {
        if (use_hip)
        {
            hipEventRecord(hip_start);
        }
        else
        {
            cpu_start = std::chrono::high_resolution_clock::now();
        }
    }

    void stop()
    {
        float ms = 0;

        if (use_hip)
        {
            hipEventRecord(hip_stop);
            hipEventSynchronize(hip_stop);
            hipEventElapsedTime(&ms, hip_start, hip_stop);
        }
        else
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - cpu_start);
            ms = duration.count() / 1000.0f;
        }

        if (write_immediately)
        {
            // Old behavior - write immediately (slower)
            std::string actual_filename;
            const char *job_id = std::getenv("SLURM_JOB_ID");
            if (job_id)
            {
                actual_filename = "logs/times_job_" + std::string(job_id) + ".csv";
            }
            else
            {
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                auto ms_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now.time_since_epoch()) %
                             1000;

                std::stringstream ss;
                ss << "logs/times_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
                ss << "_" << std::setfill('0') << std::setw(3) << ms_ts.count() << ".csv";
                actual_filename = ss.str();
            }

            std::ofstream log_file(actual_filename, std::ios::app);
            if (log_file.tellp() == 0)
            {
                log_file << "name,time_ms\n";
            }
            log_file << name << "," << ms << "\n";
            log_file.flush();
            log_file.close();
        }
        else
        {
            // New behavior - store in memory for batch writing (faster)
            ProfileManager::add_entry(name, ms);
        }
    }
};

// RAII wrapper
struct ScopedTimer
{
    Timer &timer;
    ScopedTimer(Timer &t) : timer(t) { timer.start(); }
    ~ScopedTimer() { timer.stop(); }
};

// Block timer for HIP kernel calls and code blocks
struct BlockTimer
{
    Timer timer;
    BlockTimer(const std::string &name, bool is_hip = true) : timer(name, is_hip)
    {
        timer.start();
    }
    ~BlockTimer()
    {
        timer.stop();
    }
};

#define TIME_SCOPE(timer) ScopedTimer _t(timer)
#define TIMER_BLOCK(name) BlockTimer _block_timer(name, true)

// CPU warmup functions (same as before)
void compute_concentration_and_inv_freq_getp(float base, int head_dim,
                                             float scaling_factor,
                                             float initial_context_length,
                                             float ntk_beta, float ntk_alpha,
                                             float *concentration_out,
                                             float *inv_freq_out)
{
    int d_half = head_dim / 2;
    float *freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++)
    {
        freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
    }

    float concentration;
    if (scaling_factor > 1.0f)
    {
        concentration = 0.1f * logf(scaling_factor) + 1.0f;
        float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
        float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
        assert(0 < low && low < high && high < d_half - 1);

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

void compute_cos_sin_getp(int pos, float base, int head_dim, float scaling_factor,
                          float initial_context_length, float ntk_beta,
                          float ntk_alpha, float *cos_out, float *sin_out)
{
    int d_half = head_dim / 2;
    float concentration;
    float *inv_freq = (float *)malloc(d_half * sizeof(float));

    compute_concentration_and_inv_freq_getp(base, head_dim, scaling_factor,
                                            initial_context_length, ntk_beta,
                                            ntk_alpha, &concentration, inv_freq);

    for (int j = 0; j < d_half; j++)
    {
        float val = (float)pos * inv_freq[j];
        cos_out[j] = cosf(val) * concentration;
        sin_out[j] = sinf(val) * concentration;
    }
    free(inv_freq);
}
