#pragma once

namespace shader{
    inline const char* slice_render_v = R"(#version 430 core
layout(location=0) in vec3 vertex_pos;
uniform mat4 MVPMatrix;
out vec3 world_pos;
void main() {
    gl_Position=MVPMatrix*(vec4(vertex_pos,1.0));
    world_pos=vertex_pos;
}
)";

    inline const char* slice_render_f = R"(#version 430 core
in vec3 world_pos;

layout(location=0) out vec4 frag_color;
layout(location=1) out vec4 frag_pos;

layout(binding=0,rgba32f) uniform image2D entry_pos;
layout(binding=1,rgba32f) uniform image2D exit_pos;
uniform sampler3D volume_data;

uniform vec3 volume_board;

void main() {
//    vec3 start_pos=imageLoad(entry_pos,ivec2(gl_FragCoord.xy)).xyz;
//    vec3 end_pos=imageLoad(exit_pos,ivec2(gl_FragCoord.xy)).xyz;
    float start_d=imageLoad(entry_pos,ivec2(gl_FragCoord.xy)).w;
    float exit_d=imageLoad(exit_pos,ivec2(gl_FragCoord.xy)).w;
    float slice_d=gl_FragCoord.w;

    if((slice_d>=start_d && slice_d<=exit_d) || (slice_d<=start_d && slice_d>=exit_d)){
        vec3 physical_sample_pos=world_pos/volume_board;
        vec4 scalar=texture(volume_data,physical_sample_pos);
        frag_color=vec4(vec3(scalar.r),1.f);
        frag_pos=vec4(world_pos,1.f);
    }
    else{
        frag_color=vec4(1.f,1.f,1.f,1.f);
        frag_pos=vec4(0.f,0.f,0.f,0.f);
    }
}
)";

    inline const char* volume_board_render_v = R"(#version 430 core
layout(location=0) in vec3 vertex_pos;
uniform mat4 MVPMatrix;
void main() {
    gl_Position=MVPMatrix*(vec4(vertex_pos,1.0));
}
)";

    inline const char* volume_board_render_f = R"(#version 430 core
out vec4 frag_color;
void main() {
    frag_color=vec4(1.f,0.f,1.f,1.f);
}
)";

    inline const char* volume_render_pos_v = R"(#version 430 core
layout(location=0) in vec3 vertex_pos;
uniform mat4 MVPMatrix;
out vec3 world_pos;
void main() {
    gl_Position=MVPMatrix*(vec4(vertex_pos,1.0));
    world_pos=vertex_pos;
}
)";

    inline const char* volume_render_pos_f = R"(@volume_render_pos_f.glsl)";

    inline const char* multi_volume_render_v = R"(#version 430 core
layout(location=0) in vec3 vertex_pos;
uniform mat4 MVPMatrix;
void main() {
//    gl_Position=vec4(vertex_pos,0.0f,1.0f);
    gl_Position=MVPMatrix*(vec4(vertex_pos,1.0));
}
)";

    inline const char* multi_volume_render_f = R"(#version 430 core
out vec4 frag_color;

layout(binding=0,rgba32f) uniform image2D EntryPos;
layout(binding=1,rgba32f) uniform image2D ExitPos;
layout(binding=2,rgba32f) uniform image2D SliceColor;
layout(binding=3,rgba32f) uniform image2D SlicePos;
uniform sampler1D transfer_func;
uniform sampler2D preInt_transferfunc;
uniform sampler3D volume_data;

uniform float ka;
uniform float kd;
uniform float shininess;
uniform float ks;
uniform vec3 light_direction;

uniform float space_x;
uniform float space_y;
uniform float space_z;

uniform vec3 volume_board;
uniform bool slice_visible;
uniform float step;


vec3 g_ray_dir;
vec3 phongShading(vec3 samplePos,vec3 diffuseColor);
void main() {
//    frag_color=vec4(1.f,1.f,0.f,1.f);
//    return;
    vec3 start_pos=imageLoad(EntryPos,ivec2(gl_FragCoord.xy)).xyz;
    vec3 end_pos=imageLoad(ExitPos,ivec2(gl_FragCoord.xy)).xyz;
    vec4 slice_pos_=imageLoad(SlicePos,ivec2(gl_FragCoord.xy)).xyzw;
    vec3 slice_pos=slice_pos_.xyz;
    int slice_shadow=int(slice_pos_.w);
    vec3 start2slice=slice_pos-start_pos;
    vec3 start2end=end_pos-start_pos;
    vec3 ray_direction=normalize(start2end);
    g_ray_dir=ray_direction;
//    if(slice_shadow==1){
//        frag_color=vec4(1.f,0.f,0.f,1.f);
//    }
//    else if(slice_shadow==0){
//        frag_color=vec4(0.f,1.f,0.f,1.f);
//    }
//    else{
//        frag_color=vec4(0.f,0.f,1.f,1.f);
//    }
//    return;

    float distance=dot(ray_direction,start2end);
//    frag_color=vec4(distance*10,0.f,0.f,1.f);
//    return;
    if(int(distance-0.00001f)==0){
        frag_color=vec4(1.f,1.f,1.f,1.f);
        return;
    }

    float old_distance=distance;
    float dist=dot(ray_direction,start2slice);

    if(slice_visible){
        if(slice_shadow==1){
            if(dist<0.f){
                frag_color=vec4(imageLoad(SliceColor,ivec2(gl_FragCoord.xy)).xyz,1.f);

                return;
            }
            else if(dist<distance){
                distance=dist;
            }
        }
    }
    int steps=int(distance/step);
    vec4 color=vec4(0.0f);
    vec3 simple_pos=start_pos;

    for(int i=0;i<steps;i++){
        vec3 physical_simple_pos=simple_pos/volume_board;
        vec4 scalar=texture(volume_data,physical_simple_pos);
        vec4 simple_color;
        if(scalar.r>0.f){
            simple_color=texture(transfer_func,scalar.r);
            simple_color.rgb=phongShading(simple_pos,simple_color.rgb);
            color = color + simple_color * vec4(simple_color.aaa, 1.0) * (1.0 - color.a);
            if(color.a>0.99f)
            break;
        }
        simple_pos+=ray_direction*step;
    }
    if(slice_visible){
        if(color.a==0.f && dist<old_distance && slice_shadow==1){
            frag_color=vec4(imageLoad(SliceColor,ivec2(gl_FragCoord.xy)).xyz,1.f);
//            frag_color=vec4(1.f,1.f,1.f,1.f);
            return;
        }
        if(slice_shadow==0){
            color=(1-color.a)*vec4(1.0f,1.0f,1.0f,1.0f)+color*color.a;
        }

    }
    else{
        color=(1-color.a)*vec4(1.0f,1.0f,1.0f,1.0f)+color*color.a;
    }
    frag_color=color;
//    gl_FragDepth=
}


vec3 phongShading(vec3 samplePos,vec3 diffuseColor)
{
    vec3 N;
    #define CUBIC
    #ifdef CUBIC
    float value[27];
    float t1[9];
    float t2[3];
    for(int k=-1;k<2;k++){//z
        for(int j=-1;j<2;j++){//y
            for(int i=-1;i<2;i++){//x
                value[(k+1)*9+(j+1)*3+i+1]=texture(volume_data,(samplePos+vec3(space_x*i,space_y*j,space_z*k))/volume_board).r;
            }
        }
    }
    int x,y,z;
    //for x-direction
    for(z=0;z<3;z++){
        for(y=0;y<3;y++){
            t1[z*3+y]=(value[18+y*3+z]-value[y*3+z])/2;
        }
    }
    for(z=0;z<3;z++)
    t2[z]=(t1[z*3+0]+4*t1[z*3+1]+t1[z*3+2])/6;
    N.x=(t2[0]+t2[1]*4+t2[2])/6;


    //for y-direction
    for(z=0;z<3;z++){
        for(x=0;x<3;x++){
            t1[z*3+x]=(value[x*9+6+z]-value[x*9+z])/2;
        }
    }
    for(z=0;z<3;z++)
    t2[z]=(t1[z*3+0]+4*t1[z*3+1]+t1[z*3+2])/6;
    N.y=(t2[0]+t2[1]*4+t2[2])/6;

    //for z-direction
    for(y=0;y<3;y++){
        for(x=0;x<3;x++){
            t1[y*3+x]=(value[x*9+y*3+2]-value[x*9+y*3])/2;
        }
    }
    for(y=0;y<3;y++)
    t2[y]=(t1[y*3+0]+4*t1[y*3+1]+t1[y*3+2])/6;
    N.z=(t2[0]+t2[1]*4+t2[2])/6;
    #else
    //    N.x=value[14]-value[12];
    //    N.y=value[16]-value[10];
    //    N.z=value[22]-value[4];
    N.x=(texture(volume_data,samplePos+vec3(step,0,0)).r-texture(volume_data,samplePos+vec3(-step,0,0)).r);
    N.y=(texture(volume_data,samplePos+vec3(0,step,0)).r-texture(volume_data,samplePos+vec3(0,-step,0)).r);
    N.z=(texture(volume_data,samplePos+vec3(0,0,step)).r-texture(volume_data,samplePos+vec3(0,0,-step)).r);
    #endif

    N=-normalize(N);

    vec3 L=-g_ray_dir;
    vec3 R=L;//-ray_direction;
    if(dot(N,L)<0.f)
        N=-N;

    vec3 ambient=ka*diffuseColor.rgb;
    vec3 specular=ks*pow(max(dot(N,(L+R)/2.0),0.0),shininess)*vec3(1.0f);
    vec3 diffuse=kd*max(dot(N,L),0.0)*diffuseColor.rgb;
    return ambient+specular+diffuse;
})";

    inline const char* comp_render_pos_v = R"(#version 430 core
layout(location=0) in vec3 vertex_pos;
uniform mat4 MVPMatrix;
out vec3 world_pos;
void main() {
    gl_Position=MVPMatrix*(vec4(vertex_pos,1.0));
    world_pos=vertex_pos;
}
)";

    inline const char* comp_render_pos_f = R"(#version 430 core
in vec3 world_pos;
out vec4 frag_color;
void main()
{
    frag_color=vec4(world_pos,gl_FragCoord.w);
}
)";

    inline const char* comp_render_v = R"(#version 430 core
layout(location=0) in vec2 vertex_pos;
void main()
{
    gl_Position=vec4(vertex_pos,0.0f,1.0f);
}
)";

    inline const char* comp_render_f = R"(#version 430 core
out vec4 frag_color;
layout(location = 0,rgba32f) uniform volatile image2D entryPos;
layout(location = 1,rgba32f) uniform volatile image2D exitPos;
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


float CalcDistanceFromCameraToBlockCenter(in vec3 sample_pos,int lod_t){
    ivec3 virtual_block_idx = ivec3(sample_pos / (no_padding_block_length*lod_t));
    vec3 virtual_block_center = (virtual_block_idx+0.5)*no_padding_block_length*lod_t;
    return length((virtual_block_center*volume_space-camera_pos));
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
    uvec4 physical_block_idx = mappingTable.pageEntry[flat_virtual_block_idx];
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
//            float dist = CalcDistanceFromCameraToBlockCenter(ray_sample_pos/volume_space,last_lod_t);
            float dist=length(camera_pos-ray_sample_pos);
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
                break;
            }
            else if(flag == -1){
                //            color = vec4(1.f,0.f,0.f,1.f);
                break;
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
//            float dist = CalcDistanceFromCameraToBlockCenter(ray_sample_pos/volume_space,last_lod_t);
//            if(dist>max_view_distance){
//                break;
//            }
            float dist=length(camera_pos-ray_sample_pos);
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
)";

    inline const char* simple_mesh_render_v = R"(#version 430 core
layout(location = 0) in vec3 VertexPos;
layout(location = 1) in vec3 VertexNormal;
uniform mat4 MVPMatrix;
out vec3 normal;
out vec3 world_pos;
void main() {
    gl_Position=MVPMatrix*vec4(VertexPos,1.f);
    normal = VertexNormal;
    world_pos = VertexPos;
}
)";

    inline const char* simple_mesh_render_f = R"(#version 430 core
in vec3 world_pos;
in vec3 normal;
out vec4 frag_color;
uniform vec3 camera_pos;
uniform vec3 light_pos;
layout(std430,binding = 0) buffer ColorMap{
    vec4 color[];
}color_map;
uniform int surface_idx;
void main() {
//    frag_color=vec4(1.f,0.f,0.f,1.f);return;
    vec3 n = normalize(normal);
    vec3 color = vec3(color_map.color[surface_idx]);
    //ambient
    vec3 ambient = 0.05f * color;
    //diffuse
    vec3 light_dir = normalize(light_pos-world_pos);
    float diff = max(dot(light_dir,n),0.f);
    vec3 diffuse = diff * color;
    //specular
    vec3 view_dir = normalize(camera_pos - world_pos);
    vec3 halfway_dir = normalize(light_dir+view_dir);
    float spec = pow(max(dot(n,halfway_dir),0.f),32.f);
    vec3 specular = vec3(0.3f)*spec;
    frag_color = vec4(ambient+diffuse+specular,1.f);
}
)";

}
