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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    