//
// Created by wyz on 2021/12/10.
//

#pragma once
#include <VolumeSlicer/memory_cache.hpp>
#include <memory>
VS_START

class ChunkCacheImpl;

/**
 * @brief each chunk cache has same storage size
 */
class ChunkCache: public AbstractMemoryCache{
  public:
    void SetCacheStorage(size_t GB) override;

    bool Query(size_t cacheID) override;

    void SetCache(Cache cache) override;

    Cache GetCache(size_t cacheID) override;

    Cache& GetCacheRef(size_t cacheID) override;

    explicit ChunkCache(size_t chunkSize);

    virtual ~ChunkCache();
  private:
    std::unique_ptr<ChunkCacheImpl> impl;
};

VS_END