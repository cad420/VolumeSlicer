//
// Created by wyz on 2021/12/10.
//
#include "chunk_cache.hpp"
#include <VolumeSlicer/LRU.hpp>
#include <VolumeSlicer/Utils/linear_array.hpp>
#include <VolumeSlicer/Utils/logger.hpp>
VS_START

class ChunkCacheImpl{
  public:
    ChunkCacheImpl(size_t chunkSize):chunkSize(chunkSize){}

    size_t chunkSize;
    std::vector<Linear1DArray<uint8_t>> storage;
    std::unique_ptr<LRUCache<size_t,void*>> lruCache;
};

ChunkCache::ChunkCache(size_t chunkSize)
{
    impl = std::make_unique<ChunkCacheImpl>(chunkSize);

}
void ChunkCache::SetCacheStorage(size_t GB)
{
    size_t bytes = GB << 30;
    int chunkCount = bytes / impl->chunkSize;
    for(int i=0;i<chunkCount;i++){
        impl->storage.emplace_back(impl->chunkSize);
    }
    impl->lruCache = std::make_unique<LRUCache<size_t,void*>>(impl->storage.size());
    LOG_INFO("chunk cache storage size: {}",impl->storage.size());
}
bool ChunkCache::Query(size_t cacheID) const
{
    bool cached = impl->lruCache->get_value_without_move(cacheID);
    if(cached){
        return true;
    }
    else if(next){
        return next->Query(cacheID);
    }
    else{
        return false;
    }
}
void ChunkCache::SetCache(AbstractMemoryCache::Cache cache)
{
    assert(impl->chunkSize);
    assert(cache.size == impl->chunkSize);
    if(impl->lruCache->get_load_factor()<1.f){
        auto p = impl->storage[impl->lruCache->get_size()].RawData();
        memcpy(p,cache.data,cache.size);
        impl->lruCache->emplace_back(cache.id,p);
    }
    else{
        auto p = impl->lruCache->get_back().second;
        memcpy(p,cache.data,cache.size);
        impl->lruCache->emplace_back(cache.id,p);
    }
}
AbstractMemoryCache::Cache ChunkCache::GetCache(size_t cacheID)
{
    auto cache = impl->lruCache->get_value(cacheID);
    return Cache{cache,impl->chunkSize,cacheID};
}
AbstractMemoryCache::Cache &ChunkCache::GetCacheRef(size_t cacheID)
{
    static Cache cache;
    cache.id = cacheID;
    cache.size = impl->chunkSize;
    if(impl->lruCache->get_size() == 0){
        cache.data = impl->storage[0].RawData();
        impl->lruCache->emplace_back(cache.id,cache.data);
        return cache;
    }
    else{
        auto p = impl->lruCache->get_value(cacheID);
        if(p){
            cache.data = p;
            return cache;
        }
        else{
            if(impl->lruCache->get_load_factor()<1.f){
                cache.data= impl->storage[impl->lruCache->get_size()].RawData();
                impl->lruCache->emplace_back(cache.id,cache.data);
                return cache;
            }
            else{
                auto item = impl->lruCache->get_back();
                cache.data = item.second;
                impl->lruCache->emplace_back(cache.id,cache.data);
                return cache;
            }
        }
    }
}

ChunkCache::~ChunkCache()
{

}


VS_END
