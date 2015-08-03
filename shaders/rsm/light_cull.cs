#version 430 core

layout(local_size_x = 16, local_size_y = 16) in; //local workgroup size
	
layout(binding=0) uniform sampler2D depth;
layout(binding=0) writeonly uniform uimageBuffer result;

uniform vec2 nearfar;
uniform int num_lights;
uniform vec4 far_plane0;
uniform vec2 far_plane1;
uniform mat4 proj_mat;

float proj_a, proj_b;

shared float local_far, local_near;
shared int local_lights_num;
shared vec4 local_ll, local_ur;
shared int local_lights[1024];
shared int local_num_of_lights;
shared uint local_max_depth;
shared uint local_min_depth;
shared uint local_depth_mask;

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
  int index;
}; //148 bytes, 9 vec4s + int

layout(std140) uniform spot_light_data
{
  spot_data d[200];
} sld;

void main()
{
	ivec2 global_id = ivec2( gl_GlobalInvocationID.xy );
	vec2 global_size = textureSize( depth, 0 ).xy;
	ivec2 local_id = ivec2( gl_LocalInvocationID.xy );
	ivec2 local_size = ivec2( gl_WorkGroupSize.xy );
	ivec2 group_id = ivec2( gl_WorkGroupID.xy );
  ivec2 group_size = ivec2(global_size) / local_size;
	uint workgroup_index = gl_LocalInvocationIndex;
	vec2 texel = global_id / global_size;
	vec2 pos_xy;
	
	vec4 raw_depth = texture( depth, texel );
	
	float max_depth = 0;
	float min_depth = 1;
	
	if( workgroup_index == 0 )
	{
		local_ll = vec4( far_plane0.xyz, 1.0 );
		local_ur = vec4( far_plane0.w, far_plane1.xy, 1.0 );
    local_far = nearfar.y; //-1000
    local_near = nearfar.x; //-2.5
    local_lights_num = num_lights;
		
		local_num_of_lights = 0;
		
		local_max_depth = 0;
		local_min_depth = 0x7f7fffff; // max float value
		local_depth_mask = 0;
	}
	
	barrier(); //local memory barrier
  
  float far = local_far;
  float near = local_near;
  
  //WARNING: need to linearize the depth in order to make it work...
  proj_a = -(far + near) / (far - near);
  proj_b = (-2 * far * near) / (far - near);
  float linear_depth = -proj_b / (raw_depth.x * 2 - 1 + proj_a);
  raw_depth.x = linear_depth / -far;
  
  int num_of_lights = local_lights_num;
  vec3 ll, ur;
  ll = local_ll.xyz;
  ur = local_ur.xyz;
	
	//check for skybox
	bool early_rejection = ( raw_depth.x > 0.999 || raw_depth.x < 0.001 );
  
	if( !early_rejection )
	{
		float tmp_depth = raw_depth.x;

		min_depth = min( min_depth, tmp_depth );
		max_depth = max( max_depth, tmp_depth );

		if( max_depth >= min_depth )
		{
			atomicMin( local_min_depth, floatBitsToUint( min_depth ) );
			atomicMax( local_max_depth, floatBitsToUint( max_depth ) );
		}
	}
	
	barrier(); //local memory barrier
	
	max_depth = uintBitsToFloat( local_max_depth );
	min_depth = uintBitsToFloat( local_min_depth );
	
	vec2 tile_scale = vec2( global_size.x, global_size.y ) * recip( local_size.x + local_size.y );
	vec2 tile_bias = tile_scale - vec2( group_id.x, group_id.y );
	
	float proj_11 = proj_mat[0].x;
	float proj_22 = proj_mat[1].y;
	
	vec4 c1 = vec4( proj_11 * tile_scale.x, 0.0, -tile_bias.x, 0.0 );
	vec4 c2 = vec4( 0.0, proj_22 * tile_scale.y, -tile_bias.y, 0.0 );
	vec4 c4 = vec4( 0.0, 0.0, -1.0, 0.0 );
	
	vec4 frustum_planes[6];
	
	frustum_planes[0] = c4 - c1;
	frustum_planes[1] = c4 + c1;
	frustum_planes[2] = c4 - c2;
	frustum_planes[3] = c4 + c2;
	frustum_planes[4] = vec4( 0.0, 0.0, 1.0, -min_depth * far ); //0, 0, 1, 2.5
	frustum_planes[5] = vec4( 0.0, 0.0, 1.0, -max_depth * far ); //0, 0, 1, 1000
	
	frustum_planes[0].xyz = normalize( frustum_planes[0].xyz );
	frustum_planes[1].xyz = normalize( frustum_planes[1].xyz );
	frustum_planes[2].xyz = normalize( frustum_planes[2].xyz );
	frustum_planes[3].xyz = normalize( frustum_planes[3].xyz );
  
  /*
   * Calculate per tile depth mask for 2.5D light culling
   */
   
  /**/
  float vs_min_depth = min_depth * -far;
  float vs_max_depth = max_depth * -far;
  float vs_depth = raw_depth.x * -far;
  
  float range = abs( vs_max_depth - vs_min_depth + 0.00001 ) / 32.0; //depth range in each tile
  
  vs_depth -= vs_min_depth; //so that min = 0
  float depth_slot = floor(vs_depth / range);
  
  //determine the cell for each pixel in the tile
  if( !early_rejection )
  {	
    //depth_mask = depth_mask | (1 << depth_slot)
    atomicOr( local_depth_mask, 1 << uint(depth_slot) );
  }
  
  barrier();
  /**/
	
	for( uint c = workgroup_index; c < num_of_lights; c += local_size.x * local_size.y )
	{
		bool in_frustum = true;
    int index = int(c);

    float att_end = sld.d[index].attenuation_end;
    vec3 light_pos = sld.d[index].vs_position.xyz;
		vec4 lp = vec4( light_pos, 1.0 );
    
    /**/
    //calculate per light bitmask
    uint light_bitmask = 0;
    
    float light_z_min = -(light_pos.z + att_end); //light z min [0 ... 1000]
    float light_z_max = -(light_pos.z - att_end); //light z max [0 ... 1000]
    light_z_min -= vs_min_depth; //so that min = 0
    light_z_max -= vs_min_depth; //so that min = 0
    float depth_slot_min = floor(light_z_min / range);
    float depth_slot_max = floor(light_z_max / range);
    
    if( !( ( depth_slot_max > 31.0 && 
        depth_slot_min > 31.0 ) ||
      ( depth_slot_min < 0.0 && 
       depth_slot_max < 0.0 ) ) )
    {
      if( depth_slot_max > 30.0 )
        light_bitmask = uint(~0);
      else
        light_bitmask = (1 << (uint(depth_slot_max) + 1)) - 1;
        
      if( depth_slot_min > 0.0 )
        light_bitmask -= (1 << uint(depth_slot_min)) - 1;
    }
      
    in_frustum = in_frustum && bool(local_depth_mask & light_bitmask);
    /**/

		//manual unroll
		{
			float e = dot( frustum_planes[0], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}
		{
			float e = dot( frustum_planes[1], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}
		{
			float e = dot( frustum_planes[2], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}
		{
			float e = dot( frustum_planes[3], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}
		{
			float e = dot( frustum_planes[4], lp );
			in_frustum = in_frustum && ( e <= att_end );
		}
		{
			float e = dot( frustum_planes[5], lp );
			in_frustum = in_frustum && ( e >= -att_end );
		}

		if( in_frustum )
		{
			int li = atomicAdd( local_num_of_lights, 1 );
			local_lights[li] = int(index);
		}
	}
	
	barrier(); //local memory barrier

  if( workgroup_index == 0 )
	{
    imageStore( result, int((group_id.x * group_size.y + group_id.y) * 1024), uvec4(local_num_of_lights) );
    //imageStore( result, int((group_id.x * group_size.y + group_id.y) * 1024), uvec4(abs(sld.d[0].spot_direction.z*10)) );
  }
  
  for( uint c = workgroup_index; c < local_num_of_lights; c += local_size.x * local_size.y )
  {
    imageStore( result, int((group_id.x * group_size.y + group_id.y) * 1024 + c + 1), uvec4(local_lights[c]) );
  }
}