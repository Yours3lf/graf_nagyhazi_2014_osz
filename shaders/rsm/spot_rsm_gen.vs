
uniform mat4 mvp;
uniform mat4 normal_mat;

layout(location=0) in vec4 in_vertex;
layout(location=1) in vec2 in_texcoord;
layout(location=2) in vec3 in_normal;

#ifdef GPU_SKINNING
#include "gpu_skinning.h"
#endif

//NOTE: to reduce costs we won't do normal mapping here...
//the end result is a 32x32 VPL map anyways, so too much detail isn't worth it...

out vec2 texcoord;
out vec3 vs_normal;

void main()
{
  mat4 bone_transform = mat4(1);

#ifdef GPU_SKINNING
  bone_transform = get_bone_transform();
#endif

  texcoord = in_texcoord;
  vs_normal = (normal_mat * bone_transform * vec4(in_normal,0)).xyz;

  gl_Position = mvp * bone_transform * in_vertex;
}
