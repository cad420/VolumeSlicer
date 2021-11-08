#version 430 core
in vec3 world_pos;
in vec3 normal;
out vec4 frag_color;
uniform vec3 camera_pos;
uniform vec3 light_pos;
void main() {
//    frag_color=vec4(1.f,0.f,0.f,1.f);return;
    vec3 n = normalize(normal);
    vec3 color = vec3(0.7f,0.2f,0.1f);
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
