# MPILargeVolumeVis
## Volume-Coord
    using 1 = 1 mm
    using 0.001= 1um
    6t mouse volume using space=0.00032 0.00032 0.001
    base space 0.00032= min3 ( space )
## Camera
    pos using space*voxel
    fps camera
    
## Calculate raycasting pos
    ray_pos += ray_direction*step*(base_space/space) 
    set: space_ration=base_space/space
    sample_pos_in_voxel=ray_pos/space
    
## Virtual sample
    input: sample_pos_in_voxel
    output: scalar
    cur_lod
    lod_block_idx = samp_pos_in_voxel / lod_no_padding_block_length
    lod_block_dim -> lod_flat_block_idx -> mapping_table -> physical_texture_pos

### screen offset to view offset
```
    |\
h/2 | \
    |_a\
      d    a=fov/2
    tan(fov/2)=h/2 / d
    d --> 1
    h/2 --> h/2/d = tan(fov/2) => d = h/2/tan(fov/2)
    x --> x'= x / d 
    =>  x'= x * 2 * tan(fov/2) / h
    => set: scale = 2 * tan(fov/2) / h
 _______
|     *x|
|   *O  |       x = (x - O)y 
|_______|
                

```
### mpi render's screen offset to view offset
```
col     0       1       2
row  _______________________
    |     *k|       |       |
 0  |   *P  |       |       |
    |_______|_______|_______|
    |       |     *m|       |
 1  |       |   *O  |       |
    |_______|_______|_______|
    |       |       |       |
 2  |       |       |       |
    |_______|_______|_______|
    world camera      pos : camera_pos
    world view center pos : O (vec3)world_view_center //in view screen
    node  view center pos : P (vec3)node_view_center
    P = O + right * x_offset - up * y_offset
    O = camera_pos + view_direction
    node  view        pos : k (vec3)node_k
    x = (node_k - P)x
    y = (node_k - P)y
    node_pixel_view_pos = P + x' * right - y' * up
                        = O + right * x_offset - up * y_offset + x' * right - y' * up
                        = (O + x' * right - y' * up) + right * x_offset - up * y_offset
                        = (O + x' * right - y' * up) + P - O
                        = m + P - O
    mx my are distance to O in x y direction 
    m = O + mx * scale * ratio * right - my * scale * up

    node_pixel_ray_direction = node_pixel_view_pos - camera_pos
                             = m + P - O -camera_pos
                             = O + mx * scale * ratio * right - my * scale * up
                                 + right * x_offset - up * y_offset
                                 - camera_pos
    
```

## MPI配置
    smpd.exe -d
    mpiexec -hosts 2 10.189.204.238 2 10.189.74.69 2 renderer.exe
    or mpiexec.exe -hosts host_num host0_ip process_num_in_host0 host1_ip process_num_in_host1 renderer.exe [arg_for_renderer]