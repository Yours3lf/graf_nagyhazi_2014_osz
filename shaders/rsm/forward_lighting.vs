
uniform mat4 mvp, mv;
uniform mat4 normal_mat;

layout(location=0) in vec4 in_vertex;
layout(location=1) in vec2 in_texcoord;
layout(location=2) in vec3 in_normal;
#ifdef NORMAL_MAP
layout(location=3) in vec3 in_tangent;
#endif

#ifdef GPU_SKINNING
#include "gpu_skinning.h"
#endif

out vec2 texcoord;
out vec3 vs_normal;
out vec4 vs_pos;
#ifdef NORMAL_MAP
out vec3 vs_tangent;
out vec3 vs_bitangent;
#endif

void main()
{
  mat4 bone_transform = mat4(1);

#ifdef GPU_SKINNING
  bone_transform = get_bone_transform();
#endif

  texcoord = in_texcoord;
  vs_pos = mv * in_vertex;
  vs_normal = (normal_mat * bone_transform * vec4(in_normal,0)).xyz;
#ifdef NORMAL_MAP
  vs_tangent = (normal_mat * bone_transform * vec4(in_tangent,0)).xyz;
  vs_bitangent = cross(vs_normal, vs_tangent);
#endif

  gl_Position = mvp * bone_transform * in_vertex;
}
