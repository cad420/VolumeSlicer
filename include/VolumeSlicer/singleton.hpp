//
// Created by wyz on 2021/7/8.
//

#ifndef CGUTILS_SINGLETON_HPP
#define CGUTILS_SINGLETON_HPP

#include <mutex>
#include <utility>
#include <type_traits>

template<typename T,typename enable=void>
class Singleton;

template<typename T>
class Singleton<T,typename std::enable_if<std::is_default_constructible<T>::value>::type>{
private:
    static T& instance(){
        static T ins;
        return ins;
    }
public:
    Singleton()=delete;
    Singleton(const Singleton&)=delete;
    Singleton(Singleton&&)=delete;
    Singleton& operator=(const Singleton&)=delete;
    Singleton& operator=(Singleton&&)=delete;
    static T& get(){
        static std::once_flag o;
        std::call_once(o,[]{
            instance();
        });
        return instance();
    }
};

template<typename T>
class Singleton<T,typename std::enable_if<!std::is_default_constructible<T>::value>::type>{
private:
    static T* ins;
    static bool inited;
    static T* instance(){
        if(inited)
            return ins;
        return nullptr;
    }
public:
    Singleton()=delete;
    Singleton(const Singleton&)=delete;
    Singleton(Singleton&&)=delete;
    Singleton& operator=(const Singleton&)=delete;
    Singleton& operator=(Singleton&&)=delete;

    template<typename... Args>
    static void init(Args&&... args){
        static std::once_flag a;
        std::call_once(a,[&args...]{
            ins=new T(std::forward<Args>(args)...);
            if(ins)
                inited=true;
        });
    }
    static T* get(){
        static std::once_flag o;
        std::call_once(o,[]{
            instance();
        });
        return instance();
    }
};
template<typename T>
T* Singleton<T,typename std::enable_if<!std::is_default_constructible<T>::value>::type>::ins=nullptr;
template<typename T>
bool Singleton<T,typename std::enable_if<!std::is_default_constructible<T>::value>::type>::inited=false;

#endif //CGUTILS_SINGLETON_HPP
