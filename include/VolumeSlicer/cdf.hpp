//
// Created by wyz on 2021/9/22.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/define.hpp>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <Utils/hash.hpp>
#include <Utils/logger.hpp>
#include <Utils/linear_array.hpp>
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <VolumeSlicer/volume.hpp>
VS_START
FORWARD_IMPL_DECLARATION(CDFGenerator);

class CDF{
  public:
    using SizeType= std::size_t ;
    struct CDFItem{
        CDFItem()=default;
        CDFItem(int x,int y,int z):x(x),y(y),z(z),average(-1.0),
//              variance(-1.0),
              chebyshev_dist(-1){}
        int x,y,z;
        double average;
//        double variance;
        int chebyshev_dist;
    };
    auto GetCDFItems()->const std::vector<CDFItem>&{
        return this->cdf;
    }
    auto GetCDFValArray()->std::vector<uint32_t>{
        std::sort(cdf.begin(),cdf.end(),[this](CDFItem const& c1,CDFItem const& c2){
          return GetFlatID(c1)<GetFlatID(c2);
        });
        std::vector<uint32_t> res;
        res.reserve(cdf.size());
        for(const auto& it:cdf){
            res.emplace_back(it.average*255);
        }
        return res;
    }
    auto GetCDFArray()->std::vector<uint32_t> {
        std::sort(cdf.begin(),cdf.end(),[this](CDFItem const& c1,CDFItem const& c2){
            return GetFlatID(c1)<GetFlatID(c2);
        });
        std::vector<uint32_t> res;
        res.reserve(cdf.size());
        for(const auto& it:cdf){
            res.emplace_back(it.chebyshev_dist);
        }
        return res;
    }
    void SetCDFEmptyFunc(std::function<bool(CDFItem const&)>&& f){
        this->empty_func = f;
    }
    bool IsCDFItemEmpty(CDFItem const& it) const{
        if(empty_func){
            return empty_func(it);
        }
        if(it.average<0.5)
            return true;
        else
            return false;
    }
    int GetFlatID(CDFItem const& it) const{
        return it.z*dim_x*dim_y+it.y*dim_x+it.x;
    }
    void GenerateCDF(){
        std::cout<<"start gen"<<std::endl;
        //map is quick than unordered_map if unordered_map's hash function is bad
        //unordered_map is slow because hash function is bad
        std::unordered_map<std::array<int,3>,int> m;
        m.reserve(cdf.size());
//        std::cout<<"start map"<<std::endl;
        bool all_empty=true;
        for(auto& it:cdf){
            if(IsCDFItemEmpty(it)){
                m[{it.x,it.y,it.z}] = std::numeric_limits<int>::max()>>1;
            }
            else{
                m[{it.x,it.y,it.z}] = 0;
                all_empty = false;
            }
        }
        if(all_empty){
            for(auto&it :cdf){
                m[{it.x,it.y,it.z}] = (std::min)({(std::min)(it.x,dim_x-1-it.x),
                                                        (std::min)(it.y,dim_y-1-it.y),
                                                        (std::min)(it.z,dim_z-1-it.z)});
            }
        }
        bool update;
        auto AddArray=[](const auto& v1,const auto& v2){
            return decltype(v1){v1[0]+v2[0],v1[1]+v2[1],v1[2]+v2[2]};
        };
        std::cout<<"start turn"<<std::endl;
        int turn=0;
        do{
            update=false;
            for(auto& it:m){
                if(it.second==0) continue;
                for(int i=-1;i<=1;i++){
                    for(int j=-1;j<=1;j++){
                        for( int k=-1;k<=1;k++){
                            auto neighbor=AddArray(it.first,std::array<int,3>{i,j,k});
                            if(IsValidBlockIndex(neighbor[0],neighbor[1],neighbor[2])){
                                auto d=m[neighbor] + 1;
                                if(d < it.second){
                                    it.second=d;
                                    update=true;
                                }
                            }//end of valid block
                        }//end of k
                    }//end of j
                }//end of i
            }//end of a turn
            std::cout<<"turn "<<turn++<<std::endl;
        }while(update && !all_empty);
        for(auto& it:cdf){
            it.chebyshev_dist = m[{it.x,it.y,it.z}];
        }
        LOG_INFO("Finish generate CDF");
    }

    void AddCDFItem(CDFItem const& item){
        cdf.push_back(item);
    }
    void AddCDFItems(const std::vector<uint32_t>& val_array){
        assert(val_array.size()==dim_x*dim_y*dim_z);
        for(int idx = 0;idx<val_array.size();idx++){
            int z = idx/(dim_x*dim_y);
            int y = idx%(dim_x*dim_y)/dim_x;
            int x = idx%(dim_x*dim_y)%dim_x;
            CDFItem item(x,y,z);
            item.average = val_array[idx]/255.0;
            AddCDFItem(item);
        }
    }
    bool IsValidBlockIndex(int x,int y,int z) const{
        return x>=0 && x<dim_x && y>=0 && y<dim_y && z>=0 && z<dim_z;
    }
    CDF(int block_length,int len_x,int len_y,int len_z,SizeType iid=0)
    :block_length(block_length),
          dim_x(RoundUp(len_x)>>static_cast<int>(log2(block_length))),
                dim_y(RoundUp(len_y)>>static_cast<int>(log2(block_length))),dim_z(RoundUp(len_z)>>static_cast<int>(log2(block_length)))
    {
        static SizeType cls_id=0;
        cls_id++;
        if(iid) id=iid;
        else id=cls_id;
        cdf.reserve(dim_x*dim_y*dim_z);
    }
    int RoundUp(int x){
        return (x+block_length-1)& ~(block_length-1);
    }
    int GetDimX() const{return dim_x;}
    int GetDimY() const{return dim_y;}
    int GetDimZ() const{return dim_z;}
  private:
    std::function<bool(CDFItem const&)> empty_func;
    SizeType id;
    int block_length;
    int dim_x,dim_y,dim_z;//number of block for each dim
    std::vector<CDFItem> cdf;
};

class CDFGenerator{
  public:
    CDFGenerator()=default;
    ~CDFGenerator(){}

    void SetVolumeData(int len_x,int len_y,int len_z,int cdf_block_length,uint8_t* data);

    void SetVolumeData(const Linear3DArray<uint8_t>& data,int cdf_block_length);

    using VolumeBlock = typename CompVolume::VolumeBlock;
    //notice VolumeBlock's data is cuda ptr
    void SetVolumeBlockData(VolumeBlock block,int volume_block_length,int cdf_block_length);

    void GenerateCDF(){
        cdf->GenerateCDF();
    }
    auto GetCDFItems(){
        return cdf->GetCDFItems();
    }
    auto GetCDFArray(){
        return cdf->GetCDFArray();
    }
    auto GetCDFValArray(){
        return cdf->GetCDFValArray();
    }
  private:
    std::unique_ptr<CDF> cdf;
};
inline void CDFGenerator::SetVolumeData(int len_x, int len_y, int len_z, int cdf_block_length, uint8_t *data)
{
    Linear3DArray<uint8_t> array(len_x,len_y,len_z,data);
    SetVolumeData(array, cdf_block_length);
}
inline void CDFGenerator::SetVolumeData(const Linear3DArray<uint8_t> &data, int cdfblock_length)
{
    if(!cdfblock_length || (cdfblock_length &(cdfblock_length -1))){
        throw std::runtime_error("cdf_block_length is not pow of 2");
    }
    int block_size_bytes= cdfblock_length * cdfblock_length * cdfblock_length;//block is small
    std::vector<uint8_t> block_data(block_size_bytes);
    cdf=std::make_unique<CDF>(cdfblock_length,data.GetWidth(),data.GetHeight(),data.GetDepth());
    for(int z=0;z<cdf->GetDimZ();z++){
        for(int y=0;y<cdf->GetDimY();y++){
            for(int x=0;x<cdf->GetDimX();x++){
                data.ReadRegion(x* cdfblock_length,y* cdfblock_length,z* cdfblock_length, cdfblock_length,
                                cdfblock_length, cdfblock_length,block_data.data());
                CDF::CDFItem cdf_item(x,y,z);
                size_t sum=0;
                for(auto it:block_data){
                    sum += it;
                }
                cdf_item.average = static_cast<double>(sum) / static_cast<double>(block_size_bytes);
                cdf->AddCDFItem(cdf_item);
            }
        }
    }
}
inline void CDFGenerator::SetVolumeBlockData(CDFGenerator::VolumeBlock block, int volume_block_length,int cdf_block_length)
{

    Linear3DArray<uint8_t> array (volume_block_length,volume_block_length,volume_block_length);
    CUDA_RUNTIME_API_CALL(cudaMemcpy(array.RawData(),block.block_data->GetDataPtr(),block.block_data->GetSize(),cudaMemcpyDeviceToHost));
    SetVolumeData(array,cdf_block_length);
}

//class for volume block manage in runtime
//it can open the pre-computed chebyshev_dist map or pre-computed average_dist map file
//if the pre-computed average_dist map file is loaded, it can using user specified empty strategy to compute
//the chebyshev_dist map when the GetVolumeBlockCDF call
//using AddVolumeBlock to compute chebyshev_dist map which are not loaded
class CDFManager{
  public:
    CDFManager(CDFManager const&)=delete;
    CDFManager& operator=(CDFManager const&)=delete;
  public:
    explicit CDFManager(const std::string& cdf_config_file);
  public:
    void OpenValueFile(const std::string& value_file);
    bool SetComputeOnCall(bool compute, std::function<bool(CDF::CDFItem const&)>&& f = nullptr);
  public:
    CDFManager();
    //return false if open cdf_config_file successfully and read from file
    //or true that volume_block_length and cdf_block_length are same in file
    //or return true create with default construct function
    bool SetBlockLength(int volume_block_length,int cdf_block_length);

    //generate the cdf map when the function call for the VolumeBlock but not store the it
    //it will replace the data in the cdf_map
    using VolumeBlock = typename CompVolume::VolumeBlock;
    void AddVolumeBlock(VolumeBlock block);

    void AddVolumeBlock(const Linear3DArray<uint8_t>& block,const std::array<uint32_t,4>& index);
  public:
    //dim-xyz should all the same for VolumeBlock
    auto GetBlockCDFDim() const -> std::array<uint32_t,3>;

    //return false if not find the cdf map of block
    bool GetVolumeBlockCDF(const std::array<uint32_t,4>&,std::vector<uint32_t>& v);

    bool GetVolumeBlockCDF(const std::array<uint32_t,4>&,uint32_t* data,size_t length);

    bool GetVolumeBlockCDF(int lod,int x,int y,int z,std::vector<uint32_t>& v);

    bool GetVolumeBlockCDF(int lod,int x,int y,int z,uint32_t* data,size_t length);

    auto GetVolumeBlockCDF(const std::array<uint32_t,4>&)-> std::vector<uint32_t> const&;

  public:
    //same format with open, see tools/H264VolumeCDFGenerator.cpp
    bool SaveCurrentCDFMapToFile(const std::string& filename) const;
    bool SaveCurrentValueMapToFile(const std::string& filename) const;
  public:
    void clear(){
        compute = false;
        empty_fn = nullptr;
        volume_block_length = cdf_block_length = 0;
        cdf_map.clear();
        value_map.clear();
    }
  private:
    bool compute;
    std::function<bool(CDF::CDFItem const&)> empty_fn;
    int volume_block_length,cdf_block_length;
    std::unordered_map<std::array<uint32_t,4>,std::vector<uint32_t>> cdf_map;
    std::unordered_map<std::array<uint32_t,4>,std::vector<uint32_t>> value_map;
};
inline CDFManager::CDFManager()
:volume_block_length(0),cdf_block_length(0),compute(false)
{

}
inline CDFManager::CDFManager(const std::string& cdf_config_file)
:volume_block_length(0),cdf_block_length(0),compute(false)
{
    std::ifstream in(cdf_config_file);
    if(!in.is_open()){
        LOG_ERROR("CDFManager: cdf_config_file open failed");
        return;
    }
    nlohmann::json j;
    in>>j;
    in.close();

    auto get_index = [](const std::string& lod, const std::string& idx){
        uint32_t ld = std::stoul(lod.substr(3));
        auto p1 = idx.find_first_of('_');
        auto p2 = idx.find_last_of('_');
        auto x_str = idx.substr(0,p1+1);
        auto y_str = idx.substr(p1+1,p2-p1-1);
        auto z_str = idx.substr(p2+1);
        uint32_t x = std::stoul(x_str);
        uint32_t y = std::stoul(y_str);
        uint32_t z = std::stoul(z_str);
        return std::array<uint32_t,4>{x,y,z,ld};
    };
    try{
        this->volume_block_length = j["volume_block_length"];
        this->cdf_block_length    = j["cdf_block_length"];
        int min_lod=-1,max_lod;
        for(max_lod = 0; ; max_lod++){
            if(j.find("lod"+std::to_string(max_lod)) != j.end()){
                if(min_lod == -1) min_lod = max_lod;
            }
            else break;
        }
        LOG_INFO("CDF map file min_lod({0}), max_lod({1}).",min_lod,max_lod);
        for(int lod=min_lod;lod<=max_lod;lod++){
            auto lod_str = "lod"+std::to_string(lod);
            auto lod_map = j.at(lod_str);
            for(auto it=lod_map.begin();it!=lod_map.end();it++){
                std::string idx_str=it.key();
                std::vector<uint32_t> array = it.value();
                this->cdf_map[get_index(lod_str,idx_str)] = std::move(array);
            }
        }
    }
    catch (std::exception const& err)
    {
        LOG_ERROR("Load cdf map cause error: {0}.",err.what());
        this->volume_block_length = this->cdf_block_length = 0;
        this->cdf_map.clear();
    }
    catch (...)
    {
        LOG_ERROR("Load cdf map failed.");
        this->volume_block_length = this->cdf_block_length = 0;
        this->cdf_map.clear();
    }

}
inline bool CDFManager::SetBlockLength(int volume_block_length, int cdf_block_length)
{
    if(this->volume_block_length || this->cdf_block_length){
        LOG_INFO("volume_block_length or cdf_block_length is already set.");
        return false;
    }
    else{
        this->volume_block_length = volume_block_length;
        this->cdf_block_length    = cdf_block_length;
        return true;
    }
}
void CDFManager::OpenValueFile(const std::string &value_file)
{
    std::ifstream in(value_file);
    if(!in.is_open()){
        LOG_ERROR("Open value file failed.");
        return;
    }
    nlohmann::json j;
    in>>j;
    in.close();

    auto get_index = [](const std::string& lod, const std::string& idx){
      uint32_t ld = std::stoul(lod.substr(3));
      auto p1 = idx.find_first_of('_');
      auto p2 = idx.find_last_of('_');
      auto x_str = idx.substr(0,p1+1);
      auto y_str = idx.substr(p1+1,p2-p1-1);
      auto z_str = idx.substr(p2+1);
      uint32_t x = std::stoul(x_str);
      uint32_t y = std::stoul(y_str);
      uint32_t z = std::stoul(z_str);
      return std::array<uint32_t,4>{x,y,z,ld};
    };
    try{
        auto volume_block_length_ = j["volume_block_length"];
        auto cdf_block_length_    = j["cdf_block_length"];
        if(volume_block_length_!=this->volume_block_length || cdf_block_length_!=this->cdf_block_length){
            throw std::logic_error("Value file's volume_block_length or cdf_block_length not the same with cdf map's.");
        }

        int min_lod=-1,max_lod;
        for(max_lod = 0; ; max_lod++){
            if(j.find("lod"+std::to_string(max_lod)) != j.end()){
                if(min_lod == -1) min_lod = max_lod;
            }
            else break;
        }
        LOG_INFO("Value map file min_lod({0}), max_lod({1}).",min_lod,max_lod);
        for(int lod=min_lod;lod<=max_lod;lod++){
            auto lod_str = "lod"+std::to_string(lod);
            auto lod_map = j.at(lod_str);
            for(auto it=lod_map.begin();it!=lod_map.end();it++){
                std::string idx_str=it.key();
                std::vector<uint32_t> array = it.value();
                this->value_map[get_index(lod_str,idx_str)] = std::move(array);
            }
        }
    }
    catch (std::exception const& err)
    {
        LOG_ERROR("Load value map cause error: {0}.",err.what());
        this->value_map.clear();
    }
    catch (...)
    {
        LOG_ERROR("Load value map failed.");
        this->value_map.clear();
    }
}
bool CDFManager::SetComputeOnCall(bool compute, std::function<bool(const CDF::CDFItem &)>&& f)
{
    if(this->value_map.empty()){
        LOG_INFO("Pre-computed average value map is not load, so set compute(true) will not be successful");
        return false;
    }
    this->compute = compute;
    if(compute)
        this->empty_fn = f;
    return true;
}
inline void CDFManager::AddVolumeBlock(CDFManager::VolumeBlock block)
{
    CDFGenerator cdf_gen;
    cdf_gen.SetVolumeBlockData(block,volume_block_length,cdf_block_length);
    cdf_gen.GenerateCDF();
    this->cdf_map[block.index] = cdf_gen.GetCDFArray();
    this->value_map[block.index] = cdf_gen.GetCDFValArray();
}
inline void CDFManager::AddVolumeBlock(const Linear3DArray<uint8_t> &block, const std::array<uint32_t, 4> &index)
{
    CDFGenerator cdf_gen;
    cdf_gen.SetVolumeData(block,cdf_block_length);
    cdf_gen.GenerateCDF();
    this->cdf_map[index] = cdf_gen.GetCDFArray();
    this->value_map[index] = cdf_gen.GetCDFValArray();
}
inline auto CDFManager::GetBlockCDFDim() const -> std::array<uint32_t, 3>
{
    uint32_t d = volume_block_length / cdf_block_length;
    return {d,d,d};
}
auto CDFManager::GetVolumeBlockCDF(const std::array<uint32_t, 4> &index) -> std::vector<uint32_t> const &
{
    if(compute){
        if(!empty_fn){
            LOG_ERROR("Require compute cdf but empty function is nullptr.");
        }
        else{
            CDF cdf(cdf_block_length,volume_block_length,volume_block_length,volume_block_length);
            if(value_map.find(index)!=value_map.end()){
                cdf.AddCDFItems(value_map[index]);
                cdf.GenerateCDF();
                cdf_map[index] = cdf.GetCDFArray();
            }
            else{
                LOG_INFO("Not find value array in value_map, compute failed.");
            }
        }
    }
    if(cdf_map.find(index)!=cdf_map.end())
        return this->cdf_map[index];
    return {};
}
inline bool CDFManager::GetVolumeBlockCDF(const std::array<uint32_t, 4> &index, std::vector<uint32_t> &v)
{
    if(cdf_map.find(index)!=cdf_map.end()){
        v = cdf_map[index];
        return true;
    }
    return false;
}
inline bool CDFManager::GetVolumeBlockCDF(const std::array<uint32_t, 4> &index, uint32_t *data, size_t length)
{
    if(cdf_map.find(index)!=cdf_map.end()){
        ::memcpy(data,cdf_map[index].data(),length*sizeof(uint32_t));
        return true;
    }
    return false;
}
inline bool CDFManager::GetVolumeBlockCDF(int lod, int x, int y, int z, std::vector<uint32_t> &v)
{
    return GetVolumeBlockCDF({(uint32_t)x,(uint32_t)y,(uint32_t)z,(uint32_t)lod},v);
}
inline bool CDFManager::GetVolumeBlockCDF(int lod, int x, int y, int z, uint32_t *data, size_t length)
{
    return GetVolumeBlockCDF({(uint32_t)x,(uint32_t)y,(uint32_t)z,(uint32_t)lod},data,length);
}

bool CDFManager::SaveCurrentCDFMapToFile(const std::string &filename) const
{
    return false;
}
bool CDFManager::SaveCurrentValueMapToFile(const std::string &filename) const
{
    return false;
}


VS_END
