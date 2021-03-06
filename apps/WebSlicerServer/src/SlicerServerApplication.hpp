//
// Created by wyz on 2021/10/26.
//

#pragma once
#include <Poco/Util/ServerApplication.h>
#include <VolumeSlicer/Common/export.hpp>

#include <Poco/Util/Option.h>
#include <Poco/Util/OptionCallback.h>
#include <Poco/Util/OptionSet.h>

VS_START
namespace remote
{
class SlicerServerApplication : public Poco::Util::ServerApplication
{
  protected:
    void initialize(Application &self) override;

    void defineOptions(Poco::Util::OptionSet &options) override;

    void hanldle_option(const std::string &name, const std::string &value);

    int main(const std::vector<std::string> &args) override;
  public:
    static auto GetServerCap(){
        return max_server_num;
    }
  private:
    inline static int max_server_num;
    std::string m_manager_address;
    std::string m_storage_comp ;
    std::string m_storage_raw;
    uint32_t m_port = 0;
};
}

VS_END