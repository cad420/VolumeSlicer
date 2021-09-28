//
// Created by wyz on 2021/8/30.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <type_traits>
VS_START

template <typename T>
class LinearArrayBase{
    using SizeType=std::size_t ;
    using Self=LinearArrayBase<T>;
    T* data;
    SizeType num;

    struct UnInit{};
    LinearArrayBase(SizeType length,UnInit)
    :num(length)
    {
        data=static_cast<T*>(::operator new(sizeof(T)*num));
    }
    void ReleaseData(){
        ::operator delete(data);
        data=nullptr;
    }
  public:
    LinearArrayBase()
    :data(nullptr),num(0)
    {}

    LinearArrayBase(SizeType length,const T* copy_from)
        :LinearArrayBase(length,UnInit{})
    {
        if(copy_from!=nullptr){
            memcpy(data,copy_from,length*sizeof(T));
        }

    }

    LinearArrayBase(SizeType length,T init_value=T{})
    :LinearArrayBase(length,UnInit{})
    {
        for(SizeType i=0;i<num;i++){
            new(data+i) T(init_value);
        }
    }



    LinearArrayBase(const Self& other)
    :num(other.num)
    {
        if(other.IsAvailable()){
            data=static_cast<T*>(::operator new(sizeof(T)*num));
            if (std::is_trivially_copy_constructible_v<T>){
                ::memcpy(data,other.data,sizeof(T)*num);
            }
            else{
                for(SizeType i=0;i<num;i++){
                    new(data+i) T(other.data[i]);
                }
            }
        }
        else{
            ReleaseData();
        }
    }
    LinearArrayBase(Self&& other) noexcept
    :num(other.num),data(other.data)
    {
        other.data=nullptr;
    }
    Self& operator=(const Self& other){
        if(IsAvailable()){
            Destroy();
        }
        new(this) Self(other);
        return *this;
    }
    Self& operator=(Self&& other) noexcept{
        if(IsAvailable()){
            Destroy();
        }
        new(this) Self(std::move(other));
        return *this;
    }
    virtual ~LinearArrayBase(){
        Destroy();
    }
    Self& GetBase(){
        return *this;
    }
    const Self& GetBase() const{
        return *this;
    }

    bool IsAvailable() const{
        return data!=nullptr;
    }

    void Destroy(){
        if(IsAvailable()){
            ReleaseData();
        }
        num=0;
    }
    T& operator[](SizeType idx) noexcept{
        return data[idx];
    }
    T& operator[](SizeType idx) const noexcept{
        return data[idx];
    }
    T& At(SizeType idx) noexcept(false){
        if(idx>=num){
            throw std::out_of_range("linear array out of range");
        }
        return data[idx];
    }
    SizeType GetNum() const{
        return num;
    }

    T* RawData(){
        return data;
    }

    const T* RawData() const{
        return data;
    }
};

template <typename T>
class Linear1DArray: public LinearArrayBase<T>{
  public:

    using Self=Linear1DArray<T>;
    using SizeType=std::size_t ;
    using Base=LinearArrayBase<T>;

    Linear1DArray()=default;
    ~Linear1DArray()=default;

    Linear1DArray(SizeType length,const T* copy_from)
        :Base(length,copy_from)
    {}

    Linear1DArray(SizeType length,const T& init_value=T())
    :Base(length,init_value)
    {}
    Linear1DArray(const Self&)=default;
    Self& operator=(const Self&)=default;
    Linear1DArray(Self&& other) noexcept:Base(std::move(other.GetBase()))
    {}
    Self& operator=(Self&& other) noexcept{
        Base::GetBase()=std::move(other.GetBase());
        return *this;
    }

    T& operator()(SizeType idx){
        return Base::operator[](idx);
    }
    const T& operator()(SizeType idx) const{
        return Base::operator[](idx);
    }
    T& At(SizeType idx){
        return Base::At(idx);
    }
    const T& At(SizeType idx) const{
        return Base::At(idx);
    }
    SizeType GetLength()const{
        return Base::GetNum();
    }
    SizeType GetSizeInByte()const{
        return Base::GetNum()*sizeof(T);
    }
    void Clear(){
        Base::Destroy();
    }
};

template <typename T>
class Linear2DArray: public LinearArrayBase<T>{
  public:

    using Self=Linear2DArray<T>;
    using SizeType=std::size_t ;
    using Base=LinearArrayBase<T>;

    Linear2DArray()=default;
    ~Linear2DArray()=default;

    Linear2DArray(SizeType width,SizeType height,const T* copy_from)
    :Base(width*height,copy_from),width(width),height(height)
    {}

    Linear2DArray(SizeType width,SizeType height,const T& init_value=T())
        :Base(width*height,init_value),width(width),height(height)
    {}
    Linear2DArray(const Self&)=default;
    Self& operator=(const Self&)=default;
    Linear2DArray(Self&& other) noexcept:Base(std::move(other.GetBase()))
    {}
    Self& operator=(Self&& other) noexcept{
        Base::GetBase()=std::move(other.GetBase());
        return *this;
    }

    T& operator()(SizeType x,SizeType y){
        return Base::operator[](x+y*width);
    }
    const T& operator()(SizeType x,SizeType y) const{
        return Base::operator[](x+y*width);
    }
    T& At(SizeType x,SizeType y){
        return Base::At(x+y*width);
    }
    const T& At(SizeType x,SizeType y) const{
        return Base::At(x+y*width);
    }
    SizeType GetWidth() const{
        return width;
    }
    SizeType GetHeight() const{
        return height;
    }
    SizeType GetSizeInByte() const{
        return Base::GetNum()*sizeof(T);
    }
    void Clear(){
        Base::Destroy();
    }
  private:
    SizeType width=0,height=0;
};

template <typename T>
class Linear3DArray: public LinearArrayBase<T>{
  public:

    using Self=Linear3DArray<T>;
    using SizeType=std::size_t ;
    using Base=LinearArrayBase<T>;

    Linear3DArray()=default;
    ~Linear3DArray()=default;

    Linear3DArray(SizeType width,SizeType height,SizeType depth,const T* copy_from)
        :Base(width*height*depth,copy_from),width(width),height(height),depth(depth)
    {
    }

    Linear3DArray(SizeType width,SizeType height,SizeType depth,const T& init_value=T())
        :Base(width*height*depth,init_value),width(width),height(height),depth(depth)
    {}
    Linear3DArray(const Self&)=default;
    Self& operator=(const Self&)=default;
    Linear3DArray(Self&& other) noexcept
        :Base(std::move(other.GetBase())),width(other.width),height(other.height),depth(other.depth)
    {
    }
    Self& operator=(Self&& other) noexcept{
        Base::GetBase()=std::move(other.GetBase());
        this->width=other.width;
        this->height=other.height;
        this->depth=other.depth;
        return *this;
    }

    T& operator()(SizeType x,SizeType y,SizeType z){
        return Base::operator[](x+y*width+z*width*height);
    }
    const T& operator()(SizeType x,SizeType y,SizeType z) const{
        return Base::operator[](x+y*width+z*width*height);
    }
    T& At(SizeType x,SizeType y,SizeType z){
        return Base::At(x+y*width+z*width*height);
    }
    const T& At(SizeType x,SizeType y,SizeType z) const{
        return Base::At(x+y*width+z*width*height);
    }
    void SafeReadRegion(SizeType src_x,SizeType src_y,SizeType src_z,SizeType len_x,SizeType len_y,SizeType len_z,T* d) const{
        if(src_x+len_x >= GetWidth() || src_y+len_y >= GetHeight() || src_z+len_z >= GetDepth()){
            throw std::out_of_range("Linear3DArray::ReadRegion out of range");
        }
        assert(d);
        if(!d) return;
        //omp?
        for(int z=0;z<len_z;z++){
            for(int y=0;y<len_y;y++){
                for(int x=0;x<len_x;x++){
                    auto idx = z*len_x*len_y + y*len_x +x;
                    d[idx] = (*this)(x+src_x,y+src_y,z+src_z);
                }
            }
        }
    }
    void ReadRegion(SizeType src_x,SizeType src_y,SizeType src_z,SizeType len_x,SizeType len_y,SizeType len_z,T* d) const{
        assert(d);
        if(!d) return;
        //omp?
        for(int z=0;z<len_z;z++){
            for(int y=0;y<len_y;y++){
                for(int x=0;x<len_x;x++){
                    auto idx = z*len_x*len_y + y*len_x +x;
                    if(x<GetWidth() && y<GetHeight() && z<GetDepth())
                        d[idx] = (*this)(x+src_x,y+src_y,z+src_z);
                    else
                        d[idx] = 0;
                }
            }
        }
    }
    SizeType GetWidth() const{
        return width;
    }
    SizeType GetHeight() const{
        return height;
    }
    SizeType GetDepth() const{
        return depth;
    }
    SizeType GetSizeInByte() const{
        return Base::GetNum()*sizeof(T);
    }
    void Clear(){
        Base::Destroy();
    }
  private:
    SizeType width=0,height=0,depth=0;
};


/**
 * Pitch is count in byte
 */
template<typename T,int Pitch=512>
class PitchLinearArrayBase
{
    static constexpr int pitch=Pitch;
    static constexpr int n_ele_pitch=Pitch/sizeof(T);
    using SizeType=std::size_t ;
    T* data;
    SizeType size;
  public:
    PitchLinearArrayBase()
    :data(nullptr),size(0)
    {}
    PitchLinearArrayBase(SizeType length,T init_value=T{})
    {}

};

template <typename T,int Pitch>
class PitchLinear1DArray: public PitchLinearArrayBase<T,Pitch>{

};

template <typename T,int Pitch>
class PitchLinear2DArray: public PitchLinearArrayBase<T,Pitch>{

};

template <typename T,int Pitch>
class PitchLinear3DArray: public PitchLinearArrayBase<T,Pitch>{

};






VS_END