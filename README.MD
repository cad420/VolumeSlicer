# VolumeSlicer

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
    lod

}
//
SliceToOBB 需要根据Volume的space 和Slice的normal up right调整

class Volume{
function:


data:
    n_voxel_x,n_voxel_y,n_voxel_z (in voxel)

    space_x,space_y,space_z
    


}

## 单张切片和体绘制混合

    每个体都有一个代理长方体，具体的长宽高由体的三维分辨率和xyz的space相乘决定

    绘制代理长方体的线框，记录深度值和颜色

    体绘制的起始面正常计算

    体绘制的终止面 需要与切片的深度值比较，取更近的值

    在体绘制时如果光线结束的位置是切片 那么该处的颜色值应当从切片纹理采样
    or 体绘制正常结束 结束的地方就是背景色 然后将结果与切片绘制的结果 进行深度测试混合 此时切片需要额外绘制 

## 切片采样绘制

    根据切片的中心 法向量 up等 可以计算出切片每一个像素处的空间位置坐标，然后进行三维纹理采样即可
