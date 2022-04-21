//
// Created by wyz on 2021/10/28.
//
#pragma once
#include <VolumeSlicer/Common/export.hpp>
#include <mpack/mpack.h>
#include <seria/deserialize/mpack.hpp>
#include <seria/serialize/mpack.hpp>
#include <unordered_map>
VS_START
namespace remote{

class RPCMethod{
public:
    RPCMethod() = default;

    using RPCHandler = std::function<void(const std::vector<mpack_node_t>&,
                                          mpack_writer_t *writer)>;
    //string in params' order should be same with the order in function declare
    void register_method(const std::string& method_name,const std::vector<std::string>& params,
                         RPCHandler handler);

    void invoke(const std::string& method,const mpack_node_t &params,mpack_writer_t* writer);
public:
    template <typename Cls, typename RT,typename... PT>
    static RPCHandler GetHandler(RT(Cls::* method)(PT...), Cls& ins){
        std::function<RT(PT...)> fn = [&ins,method](PT&&... params)->RT{
          return (ins.*method)(std::forward<PT>(params)...);
        };
        return GetHandler(std::move(fn));
    }

    template <typename RT,typename... PT>
    static RPCHandler GetHandler(RT(* method)(PT...)){
        return std::function<RT(PT...)>(method);
    }

    template <typename RT, typename... PT>
    static RPCHandler GetHandler(std::function<RT(PT...)>&& f){
        return CreateRPCHandler(std::move(f),std::index_sequence_for<PT...>{});
    }

    template <typename RT, typename... PT,std::size_t... index>
    static RPCHandler CreateRPCHandler(std::function<RT(PT...)>&& f, std::index_sequence<index...>){
        RPCHandler handler = [f](const std::vector<mpack_node_t>& params,
                                 mpack_writer_t* writer){
            if(params.size()!=sizeof...(PT)){
                throw std::runtime_error("RPC handler: invalid params size");
            }
            auto result = f(RPCParamsUnpackHelper<std::decay_t<PT>>(params,index)...);
            mpack_write_cstr(writer,"result");
            seria::serialize(result,writer);
        };
        return handler;
    }

    template <typename Param>
    static Param RPCParamsUnpackHelper(const std::vector<mpack_node_t>& params,size_t i){
        Param param{};
        seria::deserialize(param,params[i]);
        return param;
    }


private:
    std::unordered_map<std::string,RPCHandler> rpc_methods;
    std::unordered_map<std::string,std::vector<std::string>> method_params;

};

}
VS_END
