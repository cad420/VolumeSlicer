//
// Created by wyz on 2021/7/4.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <functional>
VS_START
namespace TimeUnit{
    struct S{
        using type=std::ratio<1>;
        static constexpr const char* c_str(){ return "s";}
    };
    struct MS{
        using type=std::milli;
        static constexpr const char* c_str(){ return "ms";}
    };
    struct US{
        using type=std::micro;
        static constexpr const char* c_str(){ return "us";}
    };
    struct NS{
        using type=std::nano;
        static constexpr const char* c_str(){ return "ns";}
    };
}
using TimerDuration=typename std::chrono::duration<double,typename std::chrono::system_clock::duration::period>;
template<typename Unit>
struct UnitDuration final{
public:
    UnitDuration(TimerDuration d):d(std::chrono::duration_cast<decltype(this->d)>(d)){}

    auto count() const{
        return d.count();
    }
    std::string fmt() const{
        std::ostringstream os;
        os<<this->count()<<"("<<Unit::c_str()<<")";
        return os.str();
    }
    friend std::ostream &operator <<(std::ostream& os,const UnitDuration& d){
        return os<<d.fmt();
    }
private:
    std::chrono::duration<double,typename Unit::type> d;
};

struct Duration final{
public:
    Duration()=default;
    template <typename Rep, typename Period>
    Duration(const std::chrono::duration<Rep, Period> &_ ) :
            d( _ )
    {
    }

    auto s() const { return UnitDuration<TimeUnit::S>( d ); }
    auto ms() const { return UnitDuration<TimeUnit::MS>( d ); }
    auto us() const { return UnitDuration<TimeUnit::US>( d ); }
    auto ns() const { return UnitDuration<TimeUnit::NS>( d ); }
private:
    TimerDuration d;
};

struct TimePoint{
    /*
     * see the std::put_time reference for more detail about the fmt string
     * */
    auto to(const char * fmt)const{
        return std::put_time(localtime(&_),fmt);
    }

    auto cnt()const{
        return _;
    }
    friend std::ostream & operator<<(std::ostream &os, TimePoint const & _){
        return os << _.to("%c");
    }
    TimePoint()=default;
    template<typename C,typename D>
    TimePoint(std::chrono::time_point<C,D> const &_):
            _(std::chrono::system_clock::to_time_t(_))
    {
    }
private:
    std::time_t _;
};


class Timer{
public:
    Timer()=default;

    static auto current(){
        return TimePoint(std::chrono::system_clock::now());
    }

    void start(){
        end=begin=std::chrono::system_clock::now();
    }
    void stop(){
        end=std::chrono::system_clock::now();
        d = end - begin;
    }
    auto duration() const {return d;};

    void print_duration(std::string unit="ms"){
        if(unit=="ms")
            std::cout<<"Duration: "<<d.ms()<<std::endl;
        else if(unit=="us")
            std::cout<<"Duration: "<<d.us()<<std::endl;
        else if(unit=="ns")
            std::cout<<"Duration: "<<d.ns()<<std::endl;
        else if(unit=="s")
            std::cout<<"Duration: "<<d.s()<<std::endl;
        else
            std::cout<<"wrong time unit"<<std::endl;
    }
private:
    std::chrono::time_point<std::chrono::system_clock> begin,end;
    Duration d;
};
class AutoTimer: public Timer{
public:
    AutoTimer(std::string unit="ms"):unit(unit){
        start();
    }
    ~AutoTimer(){
        stop();
        std::cout<<"AutoTimer ";
        print_duration(unit);
    }

private:
    std::string unit;
};

VS_END