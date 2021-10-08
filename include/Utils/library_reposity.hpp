//
// Created by wyz on 2021/8/27.
//
#pragma once
#include <Utils/library.hpp>
#include <memory>
#include <unordered_map>
VS_START
class LibraryReposityImpl;
class VS_EXPORT LibraryReposity{
  public:
    ~LibraryReposity();

    static LibraryReposity* GetLibraryRepo();

    void AddLibrary(const std::string& path);

    void AddLibraries(const std::string& directory);

    void* GetSymbol(const std::string& symbol);

    void* GetSymbol(const std::string& lib,const std::string& symbol);

    bool Exists(const std::string& lib) const;

    auto GetLibrepo() const -> const std::unordered_map<std::string,std::shared_ptr<Library>>;

  private:
    LibraryReposity();
    std::unique_ptr<LibraryReposityImpl> impl;
};

std::string VS_EXPORT GetLibraryName(std::string const&);

std::string VS_EXPORT MakeValidLibraryName(std::string const&);

VS_END