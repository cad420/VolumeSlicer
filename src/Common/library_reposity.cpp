//
// Created by wyz on 2021/9/29.
//
#include <Utils/logger.hpp>
#include <Utils/library_reposity.hpp>
#include <filesystem>
#include <regex>
VS_START
class LibraryReposityImpl{
  public:
    static LibraryReposity* instance;
    std::unordered_map<std::string,std::shared_ptr<Library>> repo;
};

LibraryReposity* LibraryReposityImpl::instance = nullptr;
static LibraryReposity* GetInstance(){
    return LibraryReposityImpl::instance;
}
#define GetRepo() GetInstance()->impl->repo
//private construct
LibraryReposity::LibraryReposity()
{
    this->impl = std::make_unique<LibraryReposityImpl>();
}
LibraryReposity::~LibraryReposity()
{
    LOG_INFO("Destruct of LibraryReposity.");
}
LibraryReposity *LibraryReposity::GetLibraryRepo()
{
    if(!GetInstance())
        LibraryReposityImpl::instance = new LibraryReposity();
    return GetInstance();
}
void LibraryReposity::AddLibrary(const std::string &path)
{
    auto full_name=std::filesystem::path(path).filename().string();
    if(full_name.empty()){
        LOG_ERROR("AddLibrary pass wrong format path:{0}",path);
        return;
    }
    auto lib_name=GetLibraryName(full_name);
    if(lib_name.empty()){
        LOG_ERROR("{0} is not valid library",path);
        return;
    }
    if(GetRepo().find(lib_name)!=GetRepo().end()){
        LOG_INFO("{0} has been loaded",lib_name);
    }
    try{
        auto lib=std::make_shared<Library>(path);
        auto& repo=GetRepo();
        repo.insert({lib_name,lib});
    }
    catch (const std::exception& err)
    {
        LOG_ERROR(err.what());
    }
}
void LibraryReposity::AddLibraries(const std::string &directory)
{
    try{
        for(auto& lib:std::filesystem::directory_iterator(directory)){
            AddLibrary(lib.path().string());
        }
    }
    catch (const std::filesystem::filesystem_error& err)
    {
        LOG_ERROR("No such directory: {0}, {1}.",directory,err.what());
    }
}
void *LibraryReposity::GetSymbol(const std::string &symbol)
{
    void* sym = nullptr;
    auto& repo = GetRepo();
    for(auto it=repo.cbegin();it!=repo.cend();it++){
        sym = it->second->Symbol(symbol);
        if(sym) return sym;
    }
    return sym;
}
void *LibraryReposity::GetSymbol(const std::string &lib, const std::string &symbol)
{
    void* sym = nullptr;
    auto& repo = GetRepo();
    auto it = repo.find(lib);
    if(it != repo.end())
        sym = it->second->Symbol(symbol);
    return sym;
}
bool LibraryReposity::Exists(const std::string &lib) const
{
    auto& repo = GetRepo();
    return repo.find(lib) != repo.end();
}
auto LibraryReposity::GetLibrepo() const -> const std::unordered_map<std::string, std::shared_ptr<Library>>
{
    return GetInstance()->impl->repo;
}
std::string VS_EXPORT GetLibraryName(const std::string &full_name)
{
    std::regex reg;
    std::string lib_name=full_name.substr(0,full_name.find_last_of('.'));
#ifdef _WIN32
    reg = std::regex(R"(.+\.dll$)");
    if(std::regex_match(full_name,reg))
        return full_name;
#elif defined(__linux__)

#endif
    return "";
}
std::string VS_EXPORT MakeValidLibraryName(const std::string &name)
{
    std::string full_name;
#ifdef _WIN32
    full_name = name +".dll";
#elif defined(__linux__)

#endif
    return full_name;
}
VS_END

