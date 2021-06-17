#version 430 core
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
