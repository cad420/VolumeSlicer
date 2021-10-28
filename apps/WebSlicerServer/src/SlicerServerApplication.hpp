//
// Created by wyz on 2021/10/26.
//

#pragma once
#include <VolumeSlicer/export.hpp>

#include <Poco/Util/Option.h>
#include <Poco/Util/OptionCallback.h>
#include <Poco/Util/OptionSet.h>
#include <Poco/Util/ServerApplication.h>
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

  private:
};
}

VS_END