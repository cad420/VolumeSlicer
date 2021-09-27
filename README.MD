# VolumeSlicer

```
class Slicer{
function:
    SliceToOBB();
data:
    n_pixel_width,n_pixel_height;
    origin
    normal
    (in slice) up,right
    padding
    n_voxel_each_pixel_width
    n_voxel_each_pixel_height
    lod //depend by n_voxel_each_pixel_width(height)

}
//
SliceToOBB 需要根据Volume的space 和Slice的normal up right调整

class Volume{
function:


data:
    n_voxel_x,n_voxel_y,n_voxel_z (in voxel)

    space_x,space_y,space_z
   
}
```

## 单张切片和体绘制混合

    每个体都有一个代理长方体，具体的长宽高由体的三维分辨率和xyz的space相乘决定

    绘制代理长方体的线框，记录深度值和颜色

    体绘制的起始面正常计算

    体绘制的终止面 需要与切片的深度值比较，取更近的值

    在体绘制时如果光线结束的位置是切片 那么该处的颜色值应当从切片纹理采样
    or 体绘制正常结束 结束的地方就是背景色 然后将结果与切片绘制的结果 进行深度测试混合 此时切片需要额外绘制 
    
    算法：
        首先绘制体边框的光线入射和出射位置
        绘制一次切片，得到绘制的色彩和深度值，与上一步结果做深度混合，只保留体边框内的切片绘制结果
        体绘制两个pass得到可视体边界的光线入射和出射位置
        如果切片的深度值比出射位置的深度值小，那么将出射位置的深度值设为切片的深度值
        如果切片的深度值比入射位置的深度值小，那么该片段的颜色值直接从切片绘制结果采样

## 切片采样绘制

    根据切片的中心 法向量 up等 可以计算出切片每一个像素处的空间位置坐标，然后进行三维纹理采样即可

    动态调整采样纹理的lod

    采样的实现 cuda or opengl ？ cuda 但是混合绘制需要 opengl 

## 切片旋转时
    旋转时会造成相交的块集合剧烈变化，无法做到实时的切片更新
    解决方案一：在旋转的时候，用低分辨率的raw 切片采样替代，只有在停止交互的时候渲染正确分辨率的切片（比如鼠标释放） 但是可能停下来切片没有渲染完，用户又开始旋转或者其他交互动作了，

## 切片缩放时

## CUDAMemoryPool
    BlockLoader 拥有一个cuda内存池

## Volume提供体数据块
    
## RawVolume体绘制
    有一个边界框，由space与dim决定 一般dim不会变化 在 space变化的时候也相应更新
    有一个可视边框（不绘制），体绘制真正绘制的区域，默认等于边界框
    
## CompVolume采样
    主要要考虑每次采样需要一定的时间，
    如果每次采样一定要等到数据块全部加载完成，然后才可以继续操作切片，可能会卡顿，时间最大的耗费是数据解压，
    另一方面 旋转切片会造成数据块剧烈地变化
    所以在旋转时 使用低分辨率的Raw格式切片采样替代
    切片移动时，数据块的变化比较缓慢，可以不用低分辨率的Raw格式替代 这个可以用来测试
    不同于raycast，当一个像素对应采样的数据块如果没有加载好，那么就可以直接置为背景色... 本来也不需要循环    
    当volume的space发生变化时，slice的大小和空间位置应该没有变化
    
    slice的位置换算到volume空间，可以通过乘以space_xyz的最小值或者就是一个固定的值 比如0.01，这个固定的值可以是加载时才确定
```    
如果不是智能指针，还有默认函数，那么就可能忘记初始化，导致里面的值可能随机化，
所以程序有时候启动运行成功 有时候失败   
``` 
    
## Slice Render 和 Volume Slice Render
    两者之间不能共用一个slicer 因为两者的度量单位不一样
    后者采用降采样的raw格式
    主要修改slice的坐标以及 n_voxel_per_pixel
    
## Slice Zoom
    可以采样一个降采样后的raw数据 或者取lod最大的数据采样
    
## Data Format
原始的Raw格式 LOD分块压缩文件格式 
八叉树格式存储
自定义格式存储

## CPU Texture Update
二维和三维纹理的快速部分区域更新 **`可以使用多线程拷贝加速`** 
因为每次拷贝的源和目标内存区域都是独立分开的 不存在同时读或者写一个区域

## Block3DArray for Volume
因为体数据都是3维的 并且如果分块的话 块长都是固定好的 甚至于 块长都是一样的 并且都是2的幂次方
那么每次更新时不像纹理一样随机的位置和长度 它的更新都是在有规则的位置 并且长度一般都是等于块长 
即最小的更新单元是一个块  这样子每个块的数据都是线性存储的
所以这样子更新的速度相对于3D纹理就快很多(CPU)
构造:可以从一个raw数据构造 也可以构造一个空的专门用作分块数据的容器
Raw格式可以使用线性数组
- [ ] todo 如何把单个raw数据的分块加速结合到Block3DArray
## LinearArray
可以考虑 aligned size

## MappingTableManager for CPU
使用VolumeCache的接口 继承为 CPUVolumeCache 但是设置为模板类 Create的时候需要传入一个Block3DArray的shared_ptr
因为Block3DArray是可以采样的 所以将其替代为CUDA和OpenGL中的纹理使用 好处见上文
CPUVolumeCache应该还需要实现缓存的功能
不需要实现Volume数据与Texture数据的分开存储!!!
**在加载每个体数据块的时候 存储的格式就支持采样即可**

## BlockCacheManager
每一个数据块只需记录它是否缓存cached 而不需要记录是否valid
因为valid的作用是保证画一帧所需要的所有数据块都加载到纹理当中
但是对于cpu渲染来说 它画一帧本来就是一次性的事件 不需要所有块都在纹理中 并且这可能做不到 如果全部使用原始分辨率
因此将cached的数据块根据LRU策略进行缓存


## Volume Render Ambient Occlusion
### 方法一
在CPU端随机生成半球方向内的M个采样点 但是半球的法向量为Z轴正方向
所以在体绘制时 在进行光照计算的时候 根据得到的法向量N(x) 把那些随机采样方向旋转过来
但是对于三维来说 旋转起码得知道两个轴的方向(另一个轴可以随机在与N垂直的平面内得到) 
或者 一个旋转轴(可以Z轴和法向量叉乘得到)和旋转方向 这个公式比较复杂?
    
## CUDA-OffScreen-CompVolume-Render
算法与CPU的类似 但是在细节处理上需要调整
在CPU实现时 每个像素都会把VirtualSample没加载的块记录到unordered_set: missed_blocks中去 
如果CPU实现加了并行加速 那么每次insert时还需要加锁互斥
在GPU实现时 CUDA原生没有unordered_set这一数据结构 而且也不能直接拿来用
因为可以在kernel里使用的数据结构需要把它的函数加个__device__前缀 显然std的都不可以直接拿来用
第一种思路
    每一个render pass的时候 每个像素最多只会记录一个缺失块数据 那么只需要分配w*h的数组 
    数组可以默认设为-1 表示该像素不缺失数据块 或者存储该数据块的flat index
    配置比较简单
第二种思路
    使用gpu专门的unordered_set 可以使用NVIDIA/libcudacxx垃圾 or stdgpu能用但不好用
    如果要用到auto it:container 那么container需要有函数begin()
    使用起来有点麻烦
    
# 切比雪夫距离场 进行加速
对于一个体数据 对其按block_length 进行分块
首先根据传输函数得到每一个块是否全为空 可以改进 计算均值和方差 是否真的为空可以在运行时动态判断
那么在加载后也要动态改变切比雪夫距离场Map
切比雪夫距离场 对于三维中的两个点 p1 p2  定位为 max(|p1-p2|xyz) 也就是每个方向轴上的距离的绝对值
那么每一个块记录其离非空块的最小切比雪夫距离
map在数据比较大的时候 遍历速度比unordered_map要快?
因为unordered_map的hash函数自己实现的不好 导致映射到同一个bucket的很多 降低了访问速度
map是用红黑树 所以查询一个元素的时间复杂度是logN
设计好的hash函数 可以使unordered_map速度快于map
    
   
 
    
    
    
    
    
    