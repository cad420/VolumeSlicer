#version 430 core
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

float step=0.003f;


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

    vec3 L=-light_direction;
    vec3 R=L;//-ray_direction;
//    if(dot(N,L)<0.f)
//        N=-N;

    vec3 ambient=ka*diffuseColor.rgb;
    vec3 specular=ks*pow(max(dot(N,(L+R)/2.0),0.0),shininess)*vec3(1.0f);
    vec3 diffuse=kd*max(dot(N,L),0.0)*diffuseColor.rgb;
    return ambient+specular+diffuse;
}