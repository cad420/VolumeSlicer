//
// Created by wyz on 2021/12/10.
//
#include "chunk_cache.hpp"
#include <VolumeSlicer/LRU.hpp>
#include <VolumeSlicer/Utils/cuda_helper.hpp>
#include <VolumeSlicer/Utils/linear_array.hpp>
#include <VolumeSlicer/Utils/logger.hpp>
#include <cuda.h>
VS_START

class ChunkCacheImpl{
  public:
    ChunkCacheImpl(size_t chunkSize,bool cuda_pinned):chunkSize(chunkSize),cuda_pinned(cuda_pinned){}
    bool cuda_pinned = false;
    size_t chunkSize;
    std::vector<void*> storage;
    std::unique_ptr<LRUCache<size_t,void*>> lruCache;
};

ChunkCache::ChunkCache(size_t chunkSize,bool cuda_pinned)
{
    impl = std::make_unique<ChunkCacheImpl>(chunkSize,cuda_pinned);

}
ChunkCache::~ChunkCache()
{
    for(int i = 0;i<impl->storage.size();i++){
        if(impl->cuda_pinned){
            CUDA_DRIVER_API_CALL(cuMemFreeHost(impl->storage[i]));
        }
        else{
            ::operator delete(impl->storage[i]);
        }
    }
}

void ChunkCache::SetCacheStorage(size_t GB)
{
    size_t bytes = GB << 30;
    int chunkCount = bytes / impl->chunkSize;
    impl->storage.resize(chunkCount,nullptr);
    for(int i=0;i<chunkCount;i++){
        if(impl->cuda_pinned)
            CUDA_DRIVER_API_CALL(cuMemAllocHost(&impl->storage[i],impl->chunkSize));
        else
            impl->storage[i] = ::operator new(impl->chunkSize);
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
        auto p = impl->storage[impl->lruCache->get_size()];
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
        cache.data = impl->storage[0];
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
                cache.data= impl->storage[impl->lruCache->get_size()];
                impl->lruCache->emplace_back(cache.id,cache.data);
                return cache;
            }
            else{
                const auto& item = impl->lruCache->get_back();
                cache.data = item.second;
                impl->lruCache->pop_back();
                impl->lruCache->emplace_back(cache.id,cache.data);
                return cache;
            }
        }
    }
}



VS_END
