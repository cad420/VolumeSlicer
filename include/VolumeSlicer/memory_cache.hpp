//
// Created by wyz on 2021/12/14.
//

#pragma once
#include <VolumeSlicer/export.hpp>

VS_START

/**
 * @brief class for generic memory cache abstract
 */
class VS_EXPORT AbstractMemoryCache{
  public:
    AbstractMemoryCache():next(nullptr){}

    virtual void SetCacheStorage(size_t GB) = 0;

    struct Cache{
        void* data;//cpu ptr
        size_t size;
        size_t id;
    };

    /**
     * @brief Query if a cache is valid with the cacheID, notice this function is const
     */
    virtual bool Query(size_t cacheID) const = 0;

    virtual void SetCache(Cache cache) = 0;

    virtual Cache GetCache(size_t cacheID) = 0;

    virtual Cache& GetCacheRef(size_t cacheID) = 0;

    virtual void SetNextLevel(AbstractMemoryCache* next) {this->next = next;};

  protected:
    AbstractMemoryCache* next;
};


VS_END
