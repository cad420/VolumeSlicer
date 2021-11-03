//
// Created by wyz on 2021/10/28.
//
#include "RPCMethod.hpp"
#include <Utils/logger.hpp>
VS_START
namespace remote
{

void RPCMethod::register_method(const std::string &method_name, const std::vector<std::string> &params,
                                RPCHandler handler)
{
    this->rpc_methods.emplace(method_name,std::move(handler));
    this->method_params.emplace(method_name,params);
}
void RPCMethod::invoke(const std::string& method,const mpack_node_t &node,mpack_writer_t* writer)
{
    auto it = rpc_methods.find(method);
    if(it == rpc_methods.end()){
        throw std::runtime_error("RPC method not found");
    }
    auto handler = it->second;
    std::vector<mpack_node_t> params;
    auto it1 = method_params.find(method);
    if(it1 == method_params.end()){
        throw std::runtime_error("RPC method invalid params");
    }
    auto params_names = it1->second;
    params.reserve(params_names.size());
    for(auto& name:params_names){
        if(!mpack_node_map_contains_cstr(node,name.c_str())){
            throw std::runtime_error("Invalid params error");
        }
        params.emplace_back(mpack_node_map_cstr(node,name.c_str()));
    }
    handler(params,writer);
}

}
VS_END
