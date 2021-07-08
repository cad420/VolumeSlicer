//
// Created by wyz on 2021/7/8.
//

#ifndef CGUTILS_LRU_HPP
#define CGUTILS_LRU_HPP

#include <unordered_map>
#include <list>
#include <utility>
#if _HAS_CXX17
#include <optional>
#endif

template<typename Key,typename Value,typename Hash=std::hash<Key>>
class LRUCache{
public:
    using ItemType= std::pair<Key,Value>;
    using ItemIterator=typename std::list<ItemType>::iterator;
    explicit LRUCache(size_t cap):capacity(cap){}

    Value* get_value_ptr(const Key& key){
        auto it=pos.find(key);
        if(it==pos.end())
            return nullptr;
        move_to_head(it->second);
        return &(data.begin()->second);
    }

#if _HAS_CXX17
    std::optional<Value> get_value(const Key& key){
        auto it=pos.find(key);
        if(it==pos.end())
            return std::optional<Value>(std::nullopt);
        move_to_head(it->second);
        return std::make_optional<Value>(data.begin()->second);
    }
    std::optional<Value> front_value() const{
        if(data.size()==0)
            return std::optional<Value>(std::nullopt);
        return std::make_optional<Value>(data.begin()->second);
    }
#endif
    /**
     * if key exists then the value of key will update and move this item to head
     */
    void emplace_back(const Key& key,Value&& value){
        auto it=pos.find(key);
        if(it != pos.end()){
            it->second->second=std::move(value);
            move_to_head(it->second);
            return;
        }
        if(data.size()>capacity){
            pos.erase(data.back().first);//erase by key for unordered_map
            data.pop_back();
        }
        data.emplace_front(std::make_pair(key,std::move(value)));
        pos[key]=data.begin();
    }
    float get_load_factor() const{
        return 1.f*data.size()/capacity;
    }
private:
    void move_to_head(ItemIterator& it){
        auto key=it->first;
        data.emplace_front(std::move(*it));
        data.erase(it);
        pos[key]=data.begin();
    }
private:
    std::unordered_map<Key,ItemIterator,Hash> pos;
    std::list<ItemType> data;
    size_t capacity;
};

#endif //CGUTILS_LRU_HPP
