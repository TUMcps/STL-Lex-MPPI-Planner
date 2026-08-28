#pragma once

#include <chrono>
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>

// Forward declaration
class SubCostProfiler;

// Structured return type for profiler data
struct ProfilerStats
{
    std::string function_name;
    std::vector<double> evaluation_times_ms;
    std::vector<double> num_robustness_evals;
};

// Simple profiler for subcost functions
class SubCostProfiler
{
public:
    struct Stats
    {
        std::vector<double> evaluation_times_ms;
        std::vector<double> num_robustness_evals;

        void add_time(double time_ms);
        void add_robustness_eval_count(double count);
    };

    SubCostProfiler() = default;
    ~SubCostProfiler() = default;

    void record_time(const std::string &name, double time_ms);
    void record_robustness_eval_count(const std::string &name, double count);
    std::vector<ProfilerStats> get_stats() const;
    void reset();

private:
    std::unordered_map<std::string, Stats> stats_;
    mutable std::mutex mutex_; // Make profiler thread-safe
};

// Simple timer class that respects the profiling flag
class Timer
{
public:
    Timer(SubCostProfiler *profiler, const std::string &name);
    ~Timer();

private:
    SubCostProfiler *profiler_;
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    int start_predicate_count_;
};

// Macro that accepts profiler pointer
#define PROFILE_FUNCTION(profiler, name) Timer timer(profiler, name)
