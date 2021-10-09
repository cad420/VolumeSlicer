//
// Created by wyz on 2021/6/11.
//

#ifndef VOLUMESLICER_UTILS_HPP
#define VOLUMESLICER_UTILS_HPP

#include <list>
#include <condition_variable>
#include <mutex>
#include<future>
#include<functional>
#include<queue>
#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/status.hpp>
#include<VolumeSlicer/define.hpp>
#include<iostream>
#include<VolumeSlicer/cuda_context.hpp>
VS_START

#define PrintCUDAMemInfo(string) \
spdlog::info("{0} CUDA free mem: {1:.2f}, used mem: {2:.2f}",string,(GetCUDAFreeMem()/1024/1024)/1024.0,(GetCUDAUsedMem()/1024/1024)/1024.0)

template<typename T>
class ConcurrentQueue
{
public:

/**
 * Pay attention to default construct function, it not explict initialize maxSize,
 * so if not call setSize the value of maxSize will randomly assign, and will cause
 * endless wait for push_back.
 */
    ConcurrentQueue() {}
    ~ConcurrentQueue()=default;
    ConcurrentQueue(size_t size) : maxSize(size) {}
    ConcurrentQueue(const ConcurrentQueue&) = delete;
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

    void setSize(size_t s) {
        maxSize = s;
    }

    void push_back(const T& value) {
        // Do not use a std::lock_guard here. We will need to explicitly
        // unlock before notify_one as the other waiting thread will
        // automatically try to acquire mutex once it wakes up
        // (which will happen on notify_one)
        std::unique_lock<std::mutex> lock(m_mutex);

        auto wasEmpty = m_List.empty();

        while (full()) {
            m_cond.wait(lock);
        }

        m_List.push_back(value);

        if (wasEmpty && !m_List.empty()) {
            lock.unlock();
            m_cond.notify_one();
        }
    }

    T pop_front() {
        std::unique_lock<std::mutex> lock(m_mutex);

        while (m_List.empty()) {
            m_cond.wait(lock);
        }
        auto wasFull = full();
        T data = std::move(m_List.front());
        m_List.pop_front();

        if (wasFull && !full()) {
            lock.unlock();
            m_cond.notify_one();
        }

        spdlog::info("pop front and remain size:{0}.",m_List.size());
        return data;
    }

    T front() {
        std::unique_lock<std::mutex> lock(m_mutex);

        while (m_List.empty()) {
            m_cond.wait(lock);
        }

        return m_List.front();
    }

    size_t size(){
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_List.size();
    }

    bool empty() {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_List.empty();
    }
    void clear() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_List.clear();
    }

    template<class Ty>
    void clear(const std::list<Ty>& reserve){
        std::unique_lock<std::mutex> lock(m_mutex);

        auto wasFull = full();

        for(auto& it:reserve){
            m_List.erase(std::find(m_List.begin(),m_List.end(),it));
        }

        if (wasFull && !full()) {
            lock.unlock();
            m_cond.notify_one();
        }
    }

    void erase(const T& element){
        std::unique_lock<std::mutex> lock(m_mutex);

        auto wasFull = full();

        for(auto& it=m_List.begin();it!=m_List.end();it++){
            if(*it==element){
                m_List.erase(it);
            }
        }

        if (wasFull && !full()) {
            lock.unlock();
            m_cond.notify_one();
        }
    }

    template<class Ty>
    bool find(const Ty& element){
        std::unique_lock<std::mutex> lock(m_mutex);
        for(auto& it:m_List){
            if(it==element){
                return true;
            }
        }
        return false;
    }

    template<class Ty>
    T get(const Ty& target){
        std::unique_lock<std::mutex> lock(m_mutex);

        auto wasFull = full();

        for(auto& it=m_List.begin();it!=m_List.end();it++){
            if(*it==target){
                T t=*it;
                m_List.erase(it);

                if (wasFull && !full()) {
                    lock.unlock();
                    m_cond.notify_one();
                }

                return t;
            }
        }
        return T();
    }
    auto maxsize(){
        return maxSize;
    }
private:
    bool full() {
        if (m_List.size() == maxSize)
            return true;
        return false;
    }

private:
    std::list<T> m_List;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    size_t maxSize;
};



    template <typename T>
    struct Helper;

    template <typename T>
    struct HelperImpl : Helper<decltype( &T::operator() )>
    {
    };

    template <typename T>
    struct Helper : HelperImpl<typename std::decay<T>::type>
    {
    };

    template <typename Ret, typename Cls, typename... Args>
    struct Helper<Ret ( Cls::* )( Args... )>
    {
        using return_type = Ret;
        using argument_type = std::tuple<Args...>;
    };

    template <typename Ret, typename Cls, typename... Args>
    struct Helper<Ret ( Cls::* )( Args... ) const>
    {
        using return_type = Ret;
        using argument_type = std::tuple<Args...>;
    };

    template <typename R, typename... Args>
    struct Helper<R( Args... )>
    {
        using return_type = R;
        using argument_type = std::tuple<Args...>;
    };

    template <typename R, typename... Args>
    struct Helper<R ( * )( Args... )>
    {
        using return_type = R;
        using argument_type = std::tuple<Args...>;
    };

    template <typename R, typename... Args>
    struct Helper<R ( *const )( Args... )>
    {
        using return_type = R;
        using argument_type = std::tuple<Args...>;
    };

    template <typename R, typename... Args>
    struct Helper<R ( *volatile )( Args... )>
    {
        using return_type = R;
        using argument_type = std::tuple<Args...>;
    };

    template <typename Ret, typename Args>
    struct InferFunctionAux
    {
    };

    template <typename Ret, typename... Args>
    struct InferFunctionAux<Ret, std::tuple<Args...>>
    {
        using type = std::function<Ret( Args... )>;
    };


    template <typename F>
    struct InvokeResultOf
    {
        using type = typename Helper<F>::return_type;
    };

    template <typename F>
    struct ArgumentTypeOf
    {
        using type = typename Helper<F>::argument_type;
    };

    template <typename F>
    struct InferFunction
    {
        using type = typename InferFunctionAux<
                typename InvokeResultOf<F>::type,
                typename ArgumentTypeOf<F>::type>::type;
    };



    struct ThreadPool
    {
        ThreadPool( size_t );
        ~ThreadPool();

        template <typename F, typename... Args>
        auto AppendTask( F &&f, Args &&... args );
        void Wait();

    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex mut;
        std::atomic<size_t> idle;
        std::condition_variable cond;
        std::condition_variable waitCond;
        size_t nthreads;
        bool stop;
    };

// the constructor just launches some amount of workers
    inline ThreadPool::ThreadPool( size_t threads ) :
            idle( threads ),
            nthreads( threads ),
            stop( false )
    {
        for ( size_t i = 0; i < threads; ++i )
            workers.emplace_back(
                    [this] {
                        while ( true ) {
                            std::function<void()> task;
                            {
                                std::unique_lock<std::mutex> lock( this->mut );
                                this->cond.wait(
                                        lock, [this] { return this->stop || !this->tasks.empty(); } );
                                if ( this->stop && this->tasks.empty() ) {
                                    return;
                                }
                                idle--;
                                task = std::move( this->tasks.front() );
                                this->tasks.pop();
                            }
                            task();
                            idle++;
                            {
                                std::lock_guard<std::mutex> lk( this->mut );
                                if ( idle.load() == this->nthreads && this->tasks.empty() ) {
                                    waitCond.notify_all();
                                }
                            }
                        }
                    } );
    }

// add new work item to the pool
    template <class F, class... Args>
    auto ThreadPool::AppendTask( F && f, Args && ... args )
    {
        using return_type = typename InvokeResultOf<F>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind( std::forward<F>( f ), std::forward<Args>( args )... ) );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock( mut );
            // don't allow enqueueing after stopping the pool
            if ( stop ) {
                throw std::runtime_error( "enqueue on stopped ThreadPool" );
            }
            tasks.emplace( [task]() { ( *task )(); } );
        }
        cond.notify_one();
        return res;
    }

    inline void ThreadPool::Wait()
    {
        std::mutex m;
        std::unique_lock<std::mutex> l(m);
        waitCond.wait( l, [this]() { return this->idle.load() == nthreads && tasks.empty(); } );
    }

// the destructor joins all threads
    inline ThreadPool::~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock( mut );
            stop = true;
        }
        cond.notify_all();
        for ( std::thread &worker : workers ) {
            worker.join();
        }
    }

    template <typename T>
    struct atomic_wrapper
    {
        std::atomic<T> _a;

        atomic_wrapper()
                :_a()
        {}

        atomic_wrapper(const std::atomic<T> &a)
                :_a(a.load())
        {}

        atomic_wrapper(const atomic_wrapper &other)
                :_a(other._a.load())
        {}

        atomic_wrapper &operator=(const atomic_wrapper &other)
        {
            _a.store(other._a.load());
            return *this;
        }
    };

#define START_CPU_TIMER \
{auto _start=std::chrono::steady_clock::now();

#define END_CPU_TIMER \
auto _end=std::chrono::steady_clock::now();\
auto _t=std::chrono::duration_cast<std::chrono::milliseconds>(_end-_start);\
spdlog::info("CPU cost time {0} ms.",_t.count());}

#define START_CUDA_DRIVER_TIMER \
CUevent start,stop;\
float elapsed_time;\
cuEventCreate(&start,CU_EVENT_DEFAULT);\
cuEventCreate(&stop,CU_EVENT_DEFAULT);\
cuEventRecord(start,0);

#define STOP_CUDA_DRIVER_TIMER \
cuEventRecord(stop,0);\
cuEventSynchronize(stop);\
cuEventElapsedTime(&elapsed_time,start,stop);\
cuEventDestroy(start);\
cuEventDestroy(stop);          \
spdlog::info("GPU cost time {0} ms.",elapsed_time);



#define START_CUDA_RUNTIME_TIMER \
{cudaEvent_t     start, stop;\
float   elapsedTime;\
(cudaEventCreate(&start)); \
(cudaEventCreate(&stop));\
(cudaEventRecord(start, 0));

#define STOP_CUDA_RUNTIME_TIMER \
(cudaEventRecord(stop, 0));\
(cudaEventSynchronize(stop));\
(cudaEventElapsedTime(&elapsedTime, start, stop)); \
spdlog::info("GPU cost time {0} ms.",elapsedTime);\
(cudaEventDestroy(start));\
(cudaEventDestroy(stop));}


VS_END
#endif //VOLUMESLICER_UTILS_HPP
