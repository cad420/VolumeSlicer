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