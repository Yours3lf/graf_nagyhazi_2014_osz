#version 430 core

layout(local_size_x = 16, local_size_y = 16) in; //local workgroup size

layout(binding=0) uniform sampler2DArray spot_shadow_tex[4];
layout(binding=4) uniform sampler2DArray spot_rsm_tex[4];

layout(binding=0, r32f) writeonly uniform image2D result_depth_tex;
layout(binding=1, rgba8) writeonly uniform image2D result_rsm_tex;

uniform int light_idx;
uniform mat4 inv_light_mvp;
uniform mat4 mv;
uniform float far;

#include "common.h"

struct spot_data
{
  vec4 diffuse_color;
  vec4 specular_color; //w is light_size
  vec4 vs_position;
  float attenuation_end;
  float attenuation_cutoff; // ]0...1], 1 (need low values for nice attenuation)
  float radius;
  float spot_exponent;
  vec4 spot_direction; //w is spot_cutoff ([0...90], 180)
  mat4 shadow_mat;
  ivec4 index;
}; //148 bytes, 9 vec4s + int

layout(std140) uniform spot_light_data
{
  spot_data d[200];
} sld;

#define WORKGROUP_SIZE (16*16)

shared float local_importances[WORKGROUP_SIZE];
shared int local_light_indices[WORKGROUP_SIZE];

void main()
{
  int rsm_tex_idx = sld.d[light_idx].index.x;
  int rsm_layer_idx = sld.d[light_idx].index.y;

	ivec2 global_id = ivec2( gl_GlobalInvocationID.xy );
	vec2 global_size = textureSize( spot_shadow_tex[rsm_tex_idx], 0 ).xy;
	ivec2 local_id = ivec2( gl_LocalInvocationID.xy );
	ivec2 local_size = ivec2( gl_WorkGroupSize.xy );
	ivec2 group_id = ivec2( gl_WorkGroupID.xy );
  ivec2 group_size = ivec2(global_size) / local_size;
	uint workgroup_index = gl_LocalInvocationIndex;
	vec2 texel = global_id / global_size;

  //reconstruct view space normals, and albedo
  vec3 albedo, normal;
  decode_rsm( spot_rsm_tex[rsm_tex_idx], rsm_layer_idx, texel, albedo, normal );

	//mat4 inv_light_mvp = inverse( sld.d[light_idx].shadow_mat);
  float depth = texture( spot_shadow_tex[rsm_tex_idx], vec3( texel, rsm_layer_idx ) ).x;
  //reconstruct view-space position
	vec4 model_space_pos = inv_light_mvp * vec4( texel * 2 - 1, depth * 2 - 1, 1 );
	model_space_pos /= model_space_pos.w;
	vec4 view_space_pos = mv * model_space_pos;

  //calculate attenuation
	vec3 light_dir = sld.d[light_idx].vs_position.xyz - view_space_pos.xyz;
	float distance = length(light_dir);
	light_dir = normalize(light_dir);
	float light_radius = sld.d[light_idx].radius;

	float attenuation = ( light_radius - distance ) * recip( light_radius );

	vec3 spot_direction = sld.d[light_idx].spot_direction.xyz;
	float spot_cos_cutoff = sld.d[light_idx].spot_direction.w;
	float spot_effect = dot( -light_dir, spot_direction );

	if( spot_effect	> spot_cos_cutoff )
  {
    float spot_exponent = sld.d[light_idx].spot_exponent;
    spot_effect = pow( spot_effect, spot_exponent );

    //if( attenuation > 0.0 )
    //{
      attenuation = spot_effect * recip( 1.0 - attenuation ) * attenuation + 1.0;
    //	attenuation = spot_effect * native_recip( attenuation ) + 1.0f;
    //	attenuation = spot_effect * attenuation + 1.0f;
    //}
  }
  attenuation -= 1.0;

  //calculate a single importance value
  float result_color_importance = length(albedo * sld.d[light_idx].diffuse_color.xyz) * (1/3.0) * attenuation;
  float rci_weight = 0.5;
  float position_importance = length( view_space_pos.xyz ) / -far * (attenuation > 0 ? 1 : 0);
  float pi_weight = 1 - rci_weight;
  float importance = (result_color_importance * rci_weight + position_importance * pi_weight) * 0.5;

  int wi = int(workgroup_index);
  
  //TODO parallelize using vectors

  //copy light properties into local memory
  local_importances[wi] = importance;
  local_light_indices[wi] = light_idx;
  barrier();

  //http://www.bealto.com/gpu-sorting_parallel-merge-local.html
  for( int width = 1; width < WORKGROUP_SIZE; width <<= 1 )
  {
    int light_index = local_light_indices[wi];
    float light_importance = local_importances[wi];

    int idx_seq0 = wi & (width - 1); //trim workgroup_index to [0...width-1]
    int seq1_start = (wi - idx_seq0) ^ width; //beginning of the sibling seq

    //we exploit the fact that both the orig and the sibling seqs
    //are sorted, therefore we can do binary search like search
    //in the sibling seq, to find out where to put this thread's importance
    //value. We already know where this value is in the orig seq.

    //dichotomic search (kinda like binary search)
    int num_val_less_than_pos = 0;
    for( int increment = width; increment > 0; increment >>= 1 )
    {
      int search_pos = seq1_start + num_val_less_than_pos + increment - 1;
      search_pos = min( search_pos, WORKGROUP_SIZE-1 );

      if( local_importances[search_pos] > light_importance )
      {
        num_val_less_than_pos += increment;
        num_val_less_than_pos = min( num_val_less_than_pos, width );
      }
    }

    //mask for dest
    int bits = 2 * width - 1;
    //dest index in merged seq
    int dest = ( ( idx_seq0 + num_val_less_than_pos ) & bits ) | ( wi & ~bits );

    //first barrier makes sure that each thread has the dest idx
    //and the original values
    barrier();
    local_light_indices[dest] = light_index;
    local_importances[dest] = light_importance;
    //the second barrier makes sure the local memory changes are coherent
    barrier();
  }

  //TODO output one (or more) light per tile
  if( wi < 1 )
	{
    //imageStore( result_rsm_tex, group_id, vec4( texel, 0, 1 ) );
		imageStore( result_rsm_tex, group_id, vec4( vec3(local_importances[0]), 1 ) );
    //imageStore( result_rsm_tex, group_id, vec4( vec3(local_importances[WORKGROUP_SIZE-1]), 1 ) );
		imageStore( result_depth_tex, group_id, vec4( 1 ) );
	}
}
