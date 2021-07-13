# MPIVolumeSlicer
## 多个屏幕之间如何拼接
    就是将一个大的slice按网格均分为同样大小的多个slice
    这些分割得到的slice 有一样的normal up right 都单位化 并且和原先大的slice是一样的
    不同的是每个slice的中心点origin需要计算得到
    每个slice的长宽和n_voxles_per_pixel都是一样的 
    
    每次只需要广播大的slice信息，各个节点再计算自己小的slice即可
    