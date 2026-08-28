#include "profiler.hpp"
#include "static_config.hpp"
#include "predicate.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>

// =====================================================================
// SubCostProfiler::Stats Implementation
// =====================================================================

void SubCostProfiler::Stats::add_time(double time_ms)
{
    evaluation_times_ms.push_back(time_ms);
}

void SubCostProfiler::Stats::add_robustness_eval_count(double count)
{
    num_robustness_evals.push_back(count);
}

// =====================================================================
// SubCostProfiler Implementation
// =====================================================================

void SubCostProfiler::record_time(const std::string &name, double time_ms)
{
    if constexpr (ENABLE_SUBCOST_PROFILING)
    {
        std::lock_guard<std::mutex> lock(mutex_); // Thread-safe
        stats_[name].add_time(time_ms);
    }
}

void SubCostProfiler::record_robustness_eval_count(const std::string &name, double count)
{
    if constexpr (ENABLE_PREDICATE_COUNTING)
    {
        std::lock_guard<std::mutex> lock(mutex_); // Thread-safe
        stats_[name].add_robustness_eval_count(count);
    }
}

std::vector<ProfilerStats> SubCostProfiler::get_stats() const
{
    std::vector<ProfilerStats> result;

    if constexpr (ENABLE_SUBCOST_PROFILING || ENABLE_PREDICATE_COUNTING)
    {
        std::lock_guard<std::mutex> lock(mutex_); // Thread-safe

        // Simply return all raw profiling data
        for (const auto &[name, stat] : stats_)
        {
            ProfilerStats ps;
            ps.function_name = name;
            ps.evaluation_times_ms = stat.evaluation_times_ms;
            ps.num_robustness_evals = stat.num_robustness_evals;
            result.push_back(ps);
        }
    }

    return result;
}

void SubCostProfiler::reset()
{
    if constexpr (ENABLE_SUBCOST_PROFILING || ENABLE_PREDICATE_COUNTING)
    {
        std::lock_guard<std::mutex> lock(mutex_); // Thread-safe
        stats_.clear();
    }
}

// =====================================================================
// Timer Implementation
// =====================================================================

Timer::Timer(SubCostProfiler *profiler, const std::string &name)
    : profiler_(profiler), name_(name), start_predicate_count_(0)
{
    if (profiler_)
    {
        if constexpr (ENABLE_SUBCOST_PROFILING)
        {
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        if constexpr (ENABLE_PREDICATE_COUNTING)
        {
            start_predicate_count_ = Predicate::get_call_count();
        }
    }
}

Timer::~Timer()
{
    if (profiler_)
    {
        if constexpr (ENABLE_SUBCOST_PROFILING)
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_time_);
            double time_ms = duration.count() / 1000000.0;
            profiler_->record_time(name_, time_ms);
        }

        if constexpr (ENABLE_PREDICATE_COUNTING)
        {
            int end_count = Predicate::get_call_count();
            profiler_->record_robustness_eval_count(name_, static_cast<double>(end_count - start_predicate_count_));
        }
    }
}
