//
// Created by wyz on 2021/7/4.
//

#pragma once

#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Utils/logger.hpp>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

VS_START

namespace TimeUnit
{

struct S
{
    using type = std::ratio<1>;
    static constexpr const char *c_str()
    {
        return "s";
    }
};

struct MS
{
    using type = std::milli;
    static constexpr const char *c_str()
    {
        return "ms";
    }
};

struct US
{
    using type = std::micro;
    static constexpr const char *c_str()
    {
        return "us";
    }
};

struct NS
{
    using type = std::nano;
    static constexpr const char *c_str()
    {
        return "ns";
    }
};
} // namespace TimeUnit

using TimerDuration = typename std::chrono::duration<double, typename std::chrono::system_clock::duration::period>;

template <typename Unit>
struct UnitDuration final
{
  public:
    UnitDuration(TimerDuration d) : d(std::chrono::duration_cast<decltype(this->d)>(d))
    {
    }

    auto count() const
    {
        return d.count();
    }

    std::string fmt() const
    {
        std::ostringstream os;
        os << this->count() << "(" << Unit::c_str() << ")";
        return os.str();
    }

    friend std::ostream &operator<<(std::ostream &os, const UnitDuration &d)
    {
        return os << d.fmt();
    }

  private:
    std::chrono::duration<double, typename Unit::type> d;
};

struct Duration final
{
  public:
    Duration() = default;
    template <typename Rep, typename Period> Duration(const std::chrono::duration<Rep, Period> &_) : d(_)
    {
    }

    auto s() const
    {
        return UnitDuration<TimeUnit::S>(d);
    }

    auto ms() const
    {
        return UnitDuration<TimeUnit::MS>(d);
    }

    auto us() const
    {
        return UnitDuration<TimeUnit::US>(d);
    }

    auto ns() const
    {
        return UnitDuration<TimeUnit::NS>(d);
    }

  private:
    TimerDuration d;
};

struct TimePoint
{
    /*
     * see the std::put_time reference for more detail about the fmt string
     * */
    auto to(const char *fmt) const
    {
        return std::put_time(localtime(&_), fmt);
    }

    auto cnt() const
    {
        return _;
    }

    friend std::ostream &operator<<(std::ostream &os, TimePoint const &_)
    {
        return os << _.to("%c");
    }

    TimePoint() = default;

    template <typename C, typename D>
    TimePoint(std::chrono::time_point<C, D> const &_) : _(std::chrono::system_clock::to_time_t(_))
    {
    }

  private:
    std::time_t _;
};

class Timer
{
  public:
    Timer() = default;

    static auto current()
    {
        return TimePoint(std::chrono::system_clock::now());
    }

    void start()
    {
        end = begin = std::chrono::system_clock::now();
    }
    void stop()
    {
        end = std::chrono::system_clock::now();
        d = end - begin;
    }
    auto duration() const
    {
        return d;
    };

    void print_duration(std::string unit = "ms")
    {
        if (unit == "ms")
            LOG_INFO("Duration: {}",d.ms().fmt());
        else if (unit == "us")
            LOG_INFO("Duration: {}",d.us().fmt());
        else if (unit == "ns")
            LOG_INFO("Duration: {}",d.ns().fmt());
        else if (unit == "s")
            LOG_INFO("Duration: {}",d.s().fmt());
        else
            LOG_INFO("wrong time unit");
    }
    auto duration_str(std::string unit = "ms"){
        if (unit == "ms")
            return d.ms().fmt();
        else if (unit == "us")
            return d.us().fmt();
        else if (unit == "ns")
            return d.ns().fmt();
        else if (unit == "s")
            return d.s().fmt();
        else
            return std::string("wrong time unit");
    }
  private:
    std::chrono::time_point<std::chrono::system_clock> begin, end;
    Duration d;
};

class AutoTimer : public Timer
{
  public:
    AutoTimer(std::string msg="",std::string unit = "ms") : msg(std::move(msg)),unit(unit)
    {
        start();
    }
    ~AutoTimer()
    {
        stop();
        LOG_INFO("AutoTimer ({}) {}",msg,duration_str(unit));
    }

  private:
    std::string msg;
    std::string unit;
};

VS_END