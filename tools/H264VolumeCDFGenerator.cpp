//
// Created by wyz on 2021/10/8.
//
#include <VolumeSlicer/cdf.hpp>
#include <cmdline.hpp>
#include <Ext/iblock_volume_plugin_interface.hpp>
#include <Utils/plugin_loader.hpp>
#include <VolumeSlicer/utils.hpp>
#include <json.hpp>
#include <queue>
#include <Utils/logger.hpp>
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

    j_chebyshev_dist["cdf_block_length"]=cdf_block_length;
    j_chebyshev_dist["volume_block_length"]=block_length;
    j_average_value["cdf_block_length"] = cdf_block_length;
    j_average_value["volume_block_length"]=block_length;
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
    auto cdf_gen = [block_length,cdf_block_length](VolumeBlock volume_block)->CDFRetType {
      CDFGenerator cdf_generator;
      cdf_generator.SetVolumeBlockData(volume_block,block_length,cdf_block_length);
      //in future should release VolumeBlock after copy not until finish generating cdf
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
            if(!volume_block.valid) continue;
            i--;
            results.emplace_back(volume_block.index,thread_pool.AppendTask(cdf_gen,std::move(volume_block)));
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
        out<<j_chebyshev_dist<<std::endl;
        out.close();
    }
    catch (std::exception const& err)
    {
        LOG_ERROR("Writing chebyshev dist json to file cause error: {0}.",err.what());
    }
    catch (...)
    {
        LOG_ERROR("Writing chebyshev dist result to file failed.");
    }
    LOG_INFO("Finish writing chebyshev dist cdf config to json file.");
    //average value json
    try
    {
        std::ofstream out("average_value_"+output_filename);
        out<<j_average_value<<std::endl;
        out.close();
    }
    catch (std::exception const& err)
    {
        LOG_ERROR("Writing average value json to file cause error: {0}.",err.what());
    }
    catch (...)
    {
        LOG_ERROR("Writing average value result to file failed.");
    }
    LOG_INFO("Finish writing average value cdf config to json file.");
}