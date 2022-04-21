//
// Created by wyz on 2021/10/8.
//
#include <Ext/iblock_volume_plugin_interface.hpp>
#include <Utils/logger.hpp>
#include <Utils/plugin_loader.hpp>
#include <VolumeSlicer/Data/cdf.hpp>
#include <VolumeSlicer/Utils/utils.hpp>
#include <cmdline.hpp>
#include <json.hpp>
#include <queue>
using namespace vs;
using nlohmann::json;
int main(int argc,char** argv){
    cmdline::parser cmd;

    cmd.add<std::string>("input_filename",'i',"input comp-volume config file path",true);
    cmd.add<std::string>("output_filename",'o',"output file name",true);
    cmd.add<int>("cdf_block_length",'l',"block length for CDF generate",false,8,cmdline::oneof<int>(8,16,32,64));
    cmd.add<int>("parallel_num",'p',"parallel thread number for CDF generate",false,1);
    cmd.add<int>("iGPU",'g',"Using GPU index",false,0);
    cmd.parse_check(argc,argv);

    int iGPU=cmd.get<int>("iGPU");
    SetCUDACtx(iGPU);
    PluginLoader::LoadPlugins("./plugins");
    auto input_filename = cmd.get<std::string>("input_filename");
    auto output_filename = cmd.get<std::string>("output_filename");
    auto comp_volume = CompVolume::Load(input_filename.c_str());
    auto lod_dim = comp_volume->GetBlockDim();

    int parallel_num = cmd.get<int>("parallel_num");
    if( parallel_num > comp_volume->GetBlockQueueMaxSize())
        parallel_num = comp_volume->GetBlockQueueMaxSize();
    int block_length = comp_volume->GetBlockLength()[0];
    int cdf_block_length = cmd.get<int>("cdf_block_length");

    json j_chebyshev_dist;
    json j_average_value;//0-255
    json j_volume_value;

    j_chebyshev_dist["cdf_block_length"]=cdf_block_length;
    j_chebyshev_dist["volume_block_length"]=block_length;
    j_average_value["cdf_block_length"] = cdf_block_length;
    j_average_value["volume_block_length"]=block_length;
    j_volume_value["volume_block_length"]=block_length;
    auto index2string=[](const std::array<uint32_t,3>& index){
        return std::to_string(index[0])+"_"+std::to_string(index[1])+"_"+std::to_string(index[2]);
    };
    auto lod2string=[](int lod){
        return "lod" + std::to_string(lod);
    };
    ThreadPool thread_pool(parallel_num);
    std::queue<std::array<uint32_t,4>> tasks;
    for(const auto& lod_item : lod_dim){
        auto lod = lod_item.first;
        auto dim = lod_item.second;
        for(uint32_t z=0;z<dim[2];z++)
            for(uint32_t y=0;y<dim[1];y++)
                for(uint32_t x=0;x<dim[0];x++)
                    tasks.push({x,y,z,lod});
    }
    LOG_INFO("Total block calculate number: {0}.",tasks.size());
    using VolumeBlock = typename CompVolume::VolumeBlock;
    using CDFRetType1 = decltype(std::declval<CDFGenerator>().GetCDFArray());
    using CDFRetType2 = decltype(std::declval<CDFGenerator>().GetCDFValArray());
    using CDFRetType  = std::pair<CDFRetType1,CDFRetType2>;
    auto order=[](const std::array<uint32_t,4>& idx1,const std::array<uint32_t,4>& idx2){
        if(idx1[3]==idx2[3]){
            if(idx1[2]==idx2[2]){
                if(idx1[1]==idx2[1]){
                    return idx1[0]<idx2[0];
                }
                else return idx1[1]<idx2[1];
            }
            else return idx1[2]<idx2[2];
        }
        else return idx1[3]<idx2[3];
    };

    std::vector<std::map<std::array<uint32_t,4>,int,decltype(order)>> volume_value_array;
    volume_value_array.emplace_back(order);
    auto AddToVolumeValue = [&volume_value_array,&order](const std::array<uint32_t,4>& index,int val){
        static std::mutex mtx;
        std::lock_guard<std::mutex> lk(mtx);
        if(volume_value_array.back().size()>volume_value_array.max_size()-10){
            volume_value_array.emplace_back(order);
        }
        volume_value_array.back()[index] = val;
    };
    auto cdf_gen = [block_length,cdf_block_length,&AddToVolumeValue](VolumeBlock volume_block)->CDFRetType {
      CDFGenerator cdf_generator;
      cdf_generator.SetVolumeBlockData(volume_block,block_length,cdf_block_length);
      AddToVolumeValue(volume_block.index,cdf_generator.GetVolumeAvgValue());
      //in future should release VolumeBlock after copy not until finish generating cdf
      //notice Release is not auto call in destruct function
      volume_block.Release();
      cdf_generator.GenerateCDF();
      return {cdf_generator.GetCDFArray(),cdf_generator.GetCDFValArray()};
    };
    while(!tasks.empty()){
        int i = 0;
        for(;i<parallel_num;i++){
            if(tasks.empty()) break;
            auto block = tasks.front();
            tasks.pop();
            comp_volume->SetRequestBlock(block);
        }

        std::vector<std::pair<std::array<uint32_t,4>,std::future<CDFRetType>>> results;
        results.reserve(parallel_num);
        while(i>0){
            auto volume_block = comp_volume->GetBlock();
            auto index = volume_block.index;
            if(!volume_block.valid) continue;
            i--;
            results.emplace_back(index,thread_pool.AppendTask(cdf_gen,std::move(volume_block)));
        }
        for(auto& ret:results){
            auto lod =ret.first[3];
            auto index = std::array<uint32_t,3>{ret.first[0],ret.first[1],ret.first[2]};
            auto cdf_ret = ret.second.get();
            j_chebyshev_dist[lod2string(lod)][index2string(index)] = cdf_ret.first;
            j_average_value[lod2string(lod)][index2string(index)] = cdf_ret.second;
        }
    }

    //chebyshev dist json
    try
    {
        std::ofstream out("chebyshev_dist_"+output_filename);
        if(!out.is_open()){
            throw std::runtime_error("Open output file for chebyshev dist failed.");
        }
        out<<j_chebyshev_dist<<std::endl;
        out.close();
        LOG_INFO("Finish writing chebyshev dist cdf config to json file.");
    }
    catch (std::exception const& err)
    {
        LOG_ERROR("Writing chebyshev dist json to file cause error: {0}.",err.what());
    }
    catch (...)
    {
        LOG_ERROR("Writing chebyshev dist result to file failed.");
    }

    //average value json
    try
    {
        std::ofstream out("average_value_"+output_filename);
        if(!out.is_open()){
            throw std::runtime_error("Open output file for average value failed.");
        }
        out<<j_average_value<<std::endl;
        out.close();
        LOG_INFO("Finish writing average value cdf config to json file.");
    }
    catch (std::exception const& err)
    {
        LOG_ERROR("Writing average value json to file cause error: {0}.",err.what());
    }
    catch (...)
    {
        LOG_ERROR("Writing average value result to file failed.");
    }

    //block of volume value json
    try{
        std::unordered_map<uint32_t,std::vector<uint32_t>> value_array;
        //volume value is sorted in the order same with flag virtual block index
        for(auto& volume_value:volume_value_array){
            for(auto& it:volume_value){
                auto lod = it.first[3];
                auto value = it.second;
                value_array[lod].push_back(value);
            }
        }
        std::ofstream out("volume_value_"+output_filename);
        if(!out.is_open()){
            throw std::runtime_error("Open output file for volume value failed.");
        }
        for(auto& it:value_array){
            LOG_INFO("lod {0} has value size {1}.",it.first,it.second.size());
            j_volume_value[lod2string(it.first)]=it.second;
        }
        out<<j_volume_value<<std::endl;
        out.close();
        LOG_INFO("Finish writing volume value config to json file.");
    }
    catch (std::exception const& err)
    {
        LOG_ERROR("Writing volume value json to file cause error: {0}.",err.what());
    }
    catch (...)
    {
        LOG_ERROR("Writing volume value result to file failed.");
    }

    return 0;
}