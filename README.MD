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
        首先绘制一次切片，得到绘制的色彩和深度值
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
    