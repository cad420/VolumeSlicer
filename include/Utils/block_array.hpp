//
// Created by wyz on 2021/8/30.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <Utils/math.hpp>
#include <stdexcept>
VS_START

/**
 * for 3d-volume data store, quickly update sub-block-data
 * block is a cube which meanings block_length x=y=z
 * 1.pass not-blocked-volume data like raw:256*256*256 or 200*200*200,not require dim is 2^n
 * 2.
 */
template <typename T,uint32_t nLogBlockLength>
class Block3DArray{
  public:
    using Self     = Block3DArray<T,nLogBlockLength>;
    using SizeType = std::size_t ;
    using DataType = T;
    constexpr static SizeType block_length=1<<nLogBlockLength;//1<<nLogBlockLength
    constexpr static SizeType block_size=block_length*block_length*block_length;//block_length^3
  private:
    T* data;
    const SizeType nx,ny,nz;//block num for x y z
    SizeType x,y,z;//valid data dim
    const SizeType num;//nx*ny*nz*block_size
    struct UnInit{};
    Block3DArray(int nx,int ny,int nz,UnInit)
        :nx(nx),ny(ny),nz(nz),
          x(nx*block_length),y(ny*block_length),z(nz*block_length),
          num(nx*ny*nz*block_size)
    {
        data=static_cast<T*>(::operator new(sizeof(T)*num));
        if(data==nullptr){
            throw std::bad_alloc();
        }
    }
    void ReleaseData(){
        if(!std::is_trivially_destructible_v<T>){
            for(SizeType i=0;i<num;i++)
                data[i].~T();
        }
        ::free(data);
        data=nullptr;
    }
  public:
    Block3DArray(int nx,int ny,int nz)
        :Block3DArray(nx,ny,nz,UnInit{})
    {
    }
    Block3DArray(int nx,int ny,int nz,const T& init_value)
    :Block3DArray(nx,ny,nz,UnInit{})
    {
        for(SizeType i=0;i<num;i++){
            new(data+i) T(init_value);
        }
    }
    //linear_array's dim is x,y,z which need to roundup
    Block3DArray(int x,int y,int z,const T* linear_array)
    :Block3DArray(RoundUp(x)>>nLogBlockLength,RoundUp(y)>>nLogBlockLength,RoundUp(z)>>nLogBlockLength)
    {
        this->x=x;
        this->y=y;
        this->z=z;
        if(linear_array!=nullptr){
            for(int zz=0;zz<this->z;zz++)
                for(int yy=0;yy<this->y;yy++)
                    for(int xx=0;xx<this->x;xx++)
                        (*this)(xx,yy,zz)=linear_array[xx+yy*x+zz*x*y];
        }

    }
    void SetBlockData(SizeType x_block,SizeType y_block,SizeType z_block,const T* block_data){
//        ::memcpy(GetBlockData(x_block,y_block,z_block),block_data,BlockSizeInByte());
        cudaMemcpy(GetBlockData(x_block,y_block,z_block),block_data,BlockSizeInByte(),cudaMemcpyDefault);
    }
    void SetBlockData(SizeType flat_block_idx,const T* block_data){
//        ::memcpy(GetBlockData(flat_block_idx),block_data,BlockSizeInByte());
        cudaMemcpy(GetBlockData(flat_block_idx),block_data,BlockSizeInByte(),cudaMemcpyDefault);
    }
    T* GetBlockData(SizeType flat_block_idx){
        if(flat_block_idx >= BlockNum()) return nullptr;
        return data+flat_block_idx*BlockSize();
    }
    const T* GetBlockData(SizeType flat_block_idx) const{
        if(flat_block_idx >= BlockNum()) return nullptr;
        return data+flat_block_idx*BlockSize();
    }
    T* GetBlockData(SizeType x_block,SizeType y_block,SizeType z_block){
        if(x_block >= BlockNumX() || y_block >= BlockNumY() || z_block >= BlockNumZ()) return nullptr;
        auto flat_block_idx=z_block*(BlockNumX()*BlockNumY())+y_block*BlockNumX()+x_block;
        return GetBlockData(flat_block_idx);
    }
    const T* GetBlockData(SizeType x_block,SizeType y_block,SizeType z_block) const{
        if(x_block >= BlockNumX() || y_block >= BlockNumY() || z_block >= BlockNumZ()) return nullptr;
        auto flat_block_idx=z_block*(BlockNumX()*BlockNumY())+y_block*BlockNumX()+x_block;
        return GetBlockData(flat_block_idx);
    }

    constexpr static SizeType Block(int idx) {
        return idx>>nLogBlockLength;
    }
    constexpr static SizeType Offset(int idx){
        return idx & (BlockLength()-1);
    }
    T& operator()(int x,int y,int z) {
        auto x_block=Block(x),y_block=Block(y),z_block=Block(z);
        auto x_offset=Offset(x),y_offset=Offset(y),z_offset=Offset(z);
        auto idx=(z_block*nx*ny+y_block*nx+x_block)*BlockSize()+z_offset*block_length*block_length+y_offset*block_length+x_offset;
        return data[idx];
    }
    T& At(int x,int y,int z) {
        if(x>=this->x || y>= this->y || z >= this->z){
            throw std::out_of_range("Block3DArray At out of range");
        }
        auto x_block=Block(x),y_block=Block(y),z_block=Block(z);
        auto x_offset=Offset(x),y_offset=Offset(y),z_offset=Offset(z);
        auto idx=(z_block*nx*ny+y_block*nx+x_block)*BlockSize()+z_offset*block_length*block_length+y_offset*block_length+x_offset;
        return data[idx];
    }
    // sample in (0.0-1.0)
    //this is for volume which can entire store in Block3DArray
    T Sample(double u,double v,double k){
        assert(IsAvailable());
        u=Clamp(u,0.0,1.0)*(ArraySizeX()-1);
        v=Clamp(v,0.0,1.0)*(ArraySizeY()-1);
        k=Clamp(k,0.0,1.0)*(ArraySizeZ()-1);
        int u0=Clamp(static_cast<int>(u),0,static_cast<int>(ArraySizeX()-1));
        int u1=Clamp(u0+1,0,static_cast<int>(ArraySizeX()-1));
        int v0=Clamp(static_cast<int>(v),0,static_cast<int>(ArraySizeY()-1));
        int v1=Clamp(v0+1,0,static_cast<int>(ArraySizeY()-1));
        int k0=Clamp(static_cast<int>(k),0,static_cast<int>(ArraySizeZ()-1));
        int k1=Clamp(k0+1,0,static_cast<int>(ArraySizeZ()-1));
        double d_u=u-u0;
        double d_v=v-v0;
        double d_k=k-k0;
        return  (( (*this)(u0,v0,k0) * (1.0-d_u) + (*this)(u1,v0,k0) * d_u) * (1.0-d_v)
                +( (*this)(u0,v1,k0) * (1.0-d_u) + (*this)(u1,v1,k0) * d_u) *      d_v)*(1.0-d_k)
               +(( (*this)(u0,v0,k1) * (1.0-d_u) + (*this)(u1,v0,k1) * d_u) * (1.0-d_v)
                +( (*this)(u0,v1,k1) * (1.0-d_u) + (*this)(u1,v1,k1) * d_u) *      d_v)*d_k;
    }

    constexpr static SizeType IndexInBlock(SizeType idx_x,SizeType idx_y,SizeType idx_z){
        return idx_x+idx_y*BlockLength()+idx_z*BlockLength()*BlockLength();
    }
    //u v k in (0.0 - 1.0)
    T Sample(SizeType x_block,SizeType y_block,SizeType z_block,double u,double v,double k) const{
        auto d=GetBlockData(x_block,y_block,z_block);
//        if(!d){
//            throw std::out_of_range("block idx out of range, d is nullptr");
//            return T{};
//        }
        assert(d);
        u=Clamp(u,0.0,1.0)*(BlockLength()-1);
        v=Clamp(v,0.0,1.0)*(BlockLength()-1);
        k=Clamp(k,0.0,1.0)*(BlockLength()-1);
        int u0=Clamp(static_cast<int>(u),0,static_cast<int>(BlockLength()-1));
        int u1=Clamp(u0+1,0,static_cast<int>(BlockLength()-1));
        int v0=Clamp(static_cast<int>(v),0,static_cast<int>(BlockLength()-1));
        int v1=Clamp(v0+1,0,static_cast<int>(BlockLength()-1));
        int k0=Clamp(static_cast<int>(k),0,static_cast<int>(BlockLength()-1));
        int k1=Clamp(k0+1,0,static_cast<int>(BlockLength()-1));
        double d_u=u-u0;
        double d_v=v-v0;
        double d_k=k-k0;
        return (( d[IndexInBlock(u0,v0,k0)]*(1.0-d_u)+d[IndexInBlock(u1,v0,k0)]*d_u)*(1.0-d_v)
                +(d[IndexInBlock(u0,v1,k0)]*(1.0-d_u)+d[IndexInBlock(u1,v1,k0)]*d_u)*d_v)*(1.0-d_k)
               +((d[IndexInBlock(u0,v0,k1)]*(1.0-d_u)+d[IndexInBlock(u1,v0,k1)]*d_u)*(1.0-d_v)
                +(d[IndexInBlock(u0,v1,k1)]*(1.0-d_u)+d[IndexInBlock(u1,v1,k1)]*d_u)*d_v)*d_k;

    }
    T Sample(SizeType flat_block_idx,double u,double v,double k) const{
        auto d=GetBlockData(flat_block_idx);
//        if(!d){
//            throw std::out_of_range("block idx out of range, d is nullptr");
//            return T{};
//        }
        assert(d);
        u=Clamp(u,0.0,1.0)*(BlockLength()-1);
        v=Clamp(v,0.0,1.0)*(BlockLength()-1);
        k=Clamp(k,0.0,1.0)*(BlockLength()-1);
        int u0=Clamp(static_cast<int>(u),0,static_cast<int>(BlockLength()-1));
        int u1=Clamp(u0+1,0,static_cast<int>(BlockLength()-1));
        int v0=Clamp(static_cast<int>(v),0,static_cast<int>(BlockLength()-1));
        int v1=Clamp(v0+1,0,static_cast<int>(BlockLength()-1));
        int k0=Clamp(static_cast<int>(k),0,static_cast<int>(BlockLength()-1));
        int k1=Clamp(k0+1,0,static_cast<int>(BlockLength()-1));
        double d_u=u-u0;
        double d_v=v-v0;
        double d_k=k-k0;
        return (( d[IndexInBlock(u0,v0,k0)]*(1.0-d_u)+d[IndexInBlock(u1,v0,k0)]*d_u)*(1.0-d_v)
                +(d[IndexInBlock(u0,v1,k0)]*(1.0-d_u)+d[IndexInBlock(u1,v1,k0)]*d_u)*d_v)*(1.0-d_k)
               +((d[IndexInBlock(u0,v0,k1)]*(1.0-d_u)+d[IndexInBlock(u1,v0,k1)]*d_u)*(1.0-d_v)
                 +(d[IndexInBlock(u0,v1,k1)]*(1.0-d_u)+d[IndexInBlock(u1,v1,k1)]*d_u)*d_v)*d_k;

    }


    constexpr static SizeType RoundUp(int x){
        //because BlockLength must be 1<<nLogBlockLength
        return (x+BlockLength()-1) & ~(BlockLength()-1);
    }
    SizeType BlockNum() const{
        return nx*ny*nz;
    }
    SizeType BlockNumX() const{
        return nx;
    }
    SizeType BlockNumY() const{
        return ny;
    }
    SizeType BlockNumZ() const{
        return nz;
    }
    SizeType ArraySizeX() const{
        return x;
    }
    SizeType ArraySizeY() const{
        return y;
    }
    SizeType ArraySizeZ() const{
        return z;
    }

    constexpr static SizeType BlockLength() {
        return block_length;
    }
    constexpr static SizeType BlockSize(){
        return block_size;
    }
    constexpr static SizeType BlockSizeInByte() {
        return BlockSize()*sizeof(T);
    }

    bool IsAvailable() const{
        return data!=nullptr;
    }

    virtual ~Block3DArray(){
        if(IsAvailable()){
            ReleaseData();
        }
    }
  protected:
    T *RawData() noexcept
    {
        return data;
    }
    const T *RawData() const noexcept
    {
        return data;
    }
};

using BlockArray9b = Block3DArray<uint8_t,9>;
//using BlockArray8b = Block3DArray<uint8_t,8>;

VS_END