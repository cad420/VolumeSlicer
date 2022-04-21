#version 430 core
out vec4 frag_color;
layout(binding = 0,rgba32f) uniform volatile image2D entryPos;
layout(binding = 1,rgba32f) uniform volatile image2D exitPos;
//layout(location = 2,rgba32f) uniform volatile image2DRect interResult;

uniform sampler1D transferFunc;
uniform sampler2D preIntTransferFunc;
uniform sampler3D cacheVolumes[20];


uniform vec3 volume_texture_shape;
uniform ivec3 volume_dim;
uniform int no_padding_block_length;
uniform int block_length;
uniform ivec3 block_dim;
uniform int padding;
uniform float lod_dist[10];
uniform uint lod_mapping_table_offset[10];
uniform float max_view_distance;
uniform int max_view_steps;
uniform float step;
uniform float voxel;
uniform vec3 camera_pos;
uniform vec3 volume_space;
uniform int max_lod;
uniform bool inside;
uniform bool render;

layout(std430,binding = 0) buffer MappingTable{
    uvec4 pageEntry[];
}mappingTable;

layout(std430,binding = 1) buffer MissedBlock{
    uint blockID[];
}missedBlock;


float CalcDistanceFromCameraToBlockCenter(in vec3 sample_pos,in vec3 view_direction){
    ivec3 virtual_block_idx = ivec3(sample_pos / (no_padding_block_length));
    vec3 virtual_block_center = (virtual_block_idx+0.5)*no_padding_block_length;
//    return abs(dot(virtual_block_center*volume_space-camera_pos,view_direction));
    return length(virtual_block_center*volume_space - camera_pos);
}
int EvaluateLod(float distance){
    for(int i = 0;i<10;i++){
        if(distance<lod_dist[i]){
            return i;
        }
    }
    return max_lod+1;
}
int PowTwo(in int y){
    return 1<<y;
}
int VirtualSample(int lod,int lod_t,in vec3 sample_pos,out float scalar,bool write){
    if(sample_pos.x < 0.f || sample_pos.y < 0.f || sample_pos.z < 0.f ||
    sample_pos.x > volume_dim.x ||
    sample_pos.y > volume_dim.y ||
    sample_pos.z > volume_dim.z){
        scalar = 0.f;
        return -1;//out of volume boundary
    }
    ivec3 virtual_block_idx = ivec3(sample_pos / (no_padding_block_length*lod_t));
    ivec3 lod_block_dim = (block_dim+lod_t-1) / lod_t;
    uint flat_virtual_block_idx = virtual_block_idx.z * lod_block_dim.x * lod_block_dim.y
                               + virtual_block_idx.y * lod_block_dim.x
                               + virtual_block_idx.x + lod_mapping_table_offset[lod];
    uvec4 physical_block_idx = mappingTable.pageEntry[flat_virtual_block_idx];//todo
    uint physical_block_flag = (physical_block_idx.w>>16) & uint(0x0000ffff);
    if(physical_block_flag==0){
        if(write && missedBlock.blockID[flat_virtual_block_idx]==0){
            atomicExchange(missedBlock.blockID[flat_virtual_block_idx],1);
        }
        scalar = 0.f;
        return 0;
    }
    uint physical_texture_idx = physical_block_idx.w & uint(0x0000ffff);
    vec3 offset_in_no_padding_block = (sample_pos - virtual_block_idx*no_padding_block_length*lod_t)/lod_t;
    vec3 physica_sample_pos = (physical_block_idx.xyz*block_length + offset_in_no_padding_block + vec3(padding))/volume_texture_shape;
    scalar = texture(cacheVolumes[physical_texture_idx],physica_sample_pos).r;
    return 1;
}
vec3 PhongShading(int lod,int lod_t,in vec3 sample_pos,in vec3 diffuse_color,in vec3 view_direction){
    vec3 N;
    float x1,x2;
    VirtualSample(lod,lod_t,sample_pos+vec3(lod_t,0.f,0.f),x1,false);
    VirtualSample(lod,lod_t,sample_pos+vec3(-lod_t,0.f,0.f),x2,false);
    N.x = x1-x2;
    VirtualSample(lod,lod_t,sample_pos+vec3(0.f,lod_t,0.f),x1,false);
    VirtualSample(lod,lod_t,sample_pos+vec3(0.f,-lod_t,0.f),x2,false);
    N.y = x1-x2;
    VirtualSample(lod,lod_t,sample_pos+vec3(0.f,0.f,lod_t),x1,false);
    VirtualSample(lod,lod_t,sample_pos+vec3(0.f,0.f,-lod_t),x2,false);
    N.z = x1-x2;
    N = -normalize(N);
    vec3 L = -view_direction;
    vec3 R = L;
    if(dot(N,L)<0.f)
        N = -N;
    vec3 ambient = 0.05f * diffuse_color;
    vec3 diffuse =  max(dot(N,L),0.f) * diffuse_color;
    vec3 specular = pow(max(dot(N,(L+R)/2.f),0.f),36.f) *vec3(1.f);
    return ambient + diffuse + specular;
}

void main() {

    vec3 ray_start_pos;
    if(inside){
      ray_start_pos = camera_pos;
    }
    else{
        ray_start_pos = imageLoad(entryPos,ivec2(gl_FragCoord)).xyz;
    }

    vec3 ray_stop_pos  = imageLoad(exitPos,ivec2(gl_FragCoord)).xyz;
    vec4 color = vec4(0.f,0.f,0.f,0.f);
//    frag_color = vec4(ray_stop_pos/10.f,1.f);
//    return ;
    vec3 start2end = ray_stop_pos- ray_start_pos;
    vec3 ray_direction = normalize(start2end);
//    frag_color = vec4(ray_direction,1.f);
//    return;
    vec3 ray_sample_pos = ray_start_pos;
    vec3 lod_sample_start_pos = ray_start_pos;
    int last_lod = 0;
    int last_lod_t = PowTwo(last_lod);
    int lod_steps = 0;
    int steps = int(dot(ray_direction,start2end) / step / last_lod_t);
    steps = min(steps,max_view_steps);
    float sample_scalar;
    float last_sample_scalar = 0.f;
    if(render){
        for(int i=0;i<steps;i++){
            float dist = CalcDistanceFromCameraToBlockCenter(ray_sample_pos/volume_space,ray_direction);
//            float dist=length(camera_pos-ray_sample_pos);
            int cur_lod = EvaluateLod(dist);
            int cur_lod_t = PowTwo(cur_lod);
            if(cur_lod>max_lod){
                break;
            }
            if(cur_lod>last_lod ){
                lod_sample_start_pos=ray_sample_pos;
                last_lod = cur_lod;
                last_lod_t = cur_lod_t;
                lod_steps = i;
            }

            int flag = VirtualSample(cur_lod,cur_lod_t,ray_sample_pos/volume_space,sample_scalar,false);
            if(flag == 0){
                //            color = vec4(0.f,1.f,0.f,1.f);
                //            sample_scalar = 0.f;
//                break;
            }
            else if(flag == -1){
                //            color = vec4(1.f,0.f,0.f,1.f);
//                break;
            }

            if(sample_scalar > 0.f){
//                vec4 sample_color = texture(transferFunc,sample_scalar);
                vec4 sample_color = texture(preIntTransferFunc,vec2(last_sample_scalar,sample_scalar));
                if(sample_color.w > 0.f){
                    sample_color.rgb = PhongShading(cur_lod,cur_lod_t,ray_sample_pos/volume_space,sample_color.rgb,ray_direction);
                    color = color + sample_color*vec4(sample_color.a,sample_color.a,sample_color.a,1.f)*(1.f-color.a);
                    if(color.a > 0.99f){
                        break;
                    }
                }
                last_sample_scalar = sample_scalar;
            }

            ray_sample_pos = lod_sample_start_pos + (i+1-lod_steps)*ray_direction*step*cur_lod_t;
        }
    }
    else{
        float l_step = step * 8;
        for(int i =0;i<steps/4;i++){
            float dist = CalcDistanceFromCameraToBlockCenter(ray_sample_pos/volume_space,ray_direction);
//            if(dist>max_view_distance){
//                break;
//            }
//            float dist=length(camera_pos-ray_sample_pos);
            int cur_lod = EvaluateLod(dist);
            int cur_lod_t = PowTwo(cur_lod);
            if(cur_lod>max_lod){
                break;
            }
            if(cur_lod>last_lod){
                lod_sample_start_pos=ray_sample_pos;
                last_lod = cur_lod;
                last_lod_t = cur_lod_t;
                lod_steps = i;
            }

            vec3 sample_pos = ray_sample_pos/volume_space;
            if(sample_pos.x < 0.f || sample_pos.y < 0.f || sample_pos.z < 0.f ||
            sample_pos.x > volume_dim.x ||
            sample_pos.y > volume_dim.y ||
            sample_pos.z > volume_dim.z){
                break;
            }
            ivec3 virtual_block_idx = ivec3(sample_pos / (no_padding_block_length*cur_lod_t));
            ivec3 lod_block_dim = (block_dim+cur_lod_t-1) / cur_lod_t;
            uint flat_virtual_block_idx = virtual_block_idx.z * lod_block_dim.x * lod_block_dim.y
            + virtual_block_idx.y * lod_block_dim.x
            + virtual_block_idx.x + lod_mapping_table_offset[cur_lod];
//            uvec4 physical_block_idx = mappingTable.pageEntry[flat_virtual_block_idx];
//            uint physical_block_flag = (physical_block_idx.w>>16) & uint(0x0000ffff);
            if(missedBlock.blockID[flat_virtual_block_idx] == 0){
                atomicExchange(missedBlock.blockID[flat_virtual_block_idx],1);
            }

            ray_sample_pos = lod_sample_start_pos + (i+1-lod_steps)*ray_direction*l_step*cur_lod_t;
        }
        discard;
    }
    color.rgb = pow(color.rgb,vec3(1/2.2f));
    color.a = 1.f;
    frag_color = color;
}
