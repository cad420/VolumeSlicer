//
// Created by wyz on 2021/8/27.
//
#pragma once
#include <VolumeSlicer/export.hpp>
#include <string>
VS_START

class VS_EXPORT Library{
  public:
    Library(const std::string& name);
    void* Symbol(const std::string& name) const;
    void Close();
    ~Library();
  private:
    void* lib;
};

VS_END

