layout(location=4) in ivec4 in_bone_id;
layout(location=5) in vec4 in_bone_weight;

layout(std140) uniform bone_data
{
  mat4 d[100];
} bd;

mat4 get_bone_transform()
{
  mat4 bone_transform = bd.d[in_bone_id[0]] * in_bone_weight[0];
  bone_transform     += bd.d[in_bone_id[1]] * in_bone_weight[1];
  bone_transform     += bd.d[in_bone_id[2]] * in_bone_weight[2];
  bone_transform     += bd.d[in_bone_id[3]] * in_bone_weight[3];

  return bone_transform;
}
