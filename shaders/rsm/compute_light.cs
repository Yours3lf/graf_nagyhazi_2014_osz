#version 430 core

layout(local_size_x = 16, local_size_y = 16) in; //local workgroup size

layout(binding=0) uniform sampler2D depth_tex;
layout(binding=1) uniform sampler2D normals_tex;
layout(binding=2) uniform sampler2D albedo_tex;
layout(binding=3) uniform usamplerBuffer light_cull_tex;
layout(binding=4) uniform sampler2DArray spot_shadow_tex[4];
layout(binding=8) uniform sampler2D ssao_tex;
layout(binding=0) writeonly uniform image2D result_tex;

uniform vec2 nearfar;
uniform vec4 far_plane0;
uniform vec2 far_plane1;
uniform mat4 proj_mat;

float proj_a, proj_b;

shared float local_far, local_near;
shared int local_lights_num;
shared vec4 local_ll, local_ur;
shared int local_lights[1024];
shared int local_num_of_lights;

#include "common.h"
#include "lighting_common.h"

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

void main()
{
	ivec2 global_id = ivec2( gl_GlobalInvocationID.xy );
	vec2 global_size = textureSize( normals_tex, 0 ).xy;
	ivec2 local_id = ivec2( gl_LocalInvocationID.xy );
	ivec2 local_size = ivec2( gl_WorkGroupSize.xy );
	ivec2 group_id = ivec2( gl_WorkGroupID.xy );
  ivec2 group_size = ivec2(global_size) / local_size;
	uint workgroup_index = gl_LocalInvocationIndex;
	vec2 texel = global_id / global_size;
	vec2 pos_xy;

	vec4 raw_albedo = texture( albedo_tex, texel );
	vec4 raw_normal = vec4( 0 );
	vec4 raw_depth = texture( depth_tex, texel );
  float raw_ssao = texture( ssao_tex, texel ).x;

	vec4 out_color = vec4( 0 );

	if( workgroup_index == 0 )
	{
		local_ll = vec4( far_plane0.xyz, 1.0 );
		local_ur = vec4( far_plane0.w, far_plane1.xy, 1.0 );
    local_far = nearfar.y; //-1000
    local_near = nearfar.x; //-2.5

    local_num_of_lights = int(texelFetch(light_cull_tex, int((group_id.x * group_size.y + group_id.y) * 1024)).x);
	}

	barrier(); //local memory barrier

  float far = local_far;
  float near = local_near;

  //WARNING: need to linearize the depth in order to make it work...
  proj_a = -(far + near) / (far - near);
  proj_b = (-2 * far * near) / (far - near);
  float linear_depth = -proj_b / (raw_depth.x * 2 - 1 + proj_a);
  raw_depth.x = linear_depth / -far;

  vec3 ll, ur;
  ll = local_ll.xyz;
  ur = local_ur.xyz;

	//check for skybox
	bool early_rejection = ( raw_depth.x > 0.999 || raw_depth.x < 0.001 );

	for( uint c = workgroup_index; c < local_num_of_lights; c += local_size.x * local_size.y )
	{
		bool in_frustum = true;
    int index = int(c);

		local_lights[index] = int(texelFetch( light_cull_tex, int((group_id.x * group_size.y + group_id.y) * 1024 + c + 1) ).x);
	}

	barrier(); //local memory barrier

	if( !early_rejection )
	{
		pos_xy = mix( local_ll.xy, local_ur.xy, texel.xy );

		raw_depth.xyz = linear_depth_to_vs_pos( raw_depth.x, pos_xy, far );

		raw_normal = texture( normals_tex, texel );
		raw_normal.xyz = normalize(raw_normal.xyz * 2.0 - 1.0);

    //float gloss_factor = toksvig_aa( raw_normal.xyz, roughness_to_spec_power(raw_albedo.w) );
    float gloss_factor = toksvig_aa( raw_normal.xyz, raw_albedo.w*255 );
    //float gloss_factor = raw_albedo.w * 255;

		vec3 view_dir = normalize( -raw_depth.xyz );

		for( int c = 0; c < local_num_of_lights; ++c )
		{
			int index = local_lights[c];
      int shadow_tex_index = sld.d[index].index.x;
      int shadow_layer_index = sld.d[index].index.y;
      vec3 light_pos = sld.d[index].vs_position.xyz;
      float light_radius = sld.d[index].radius;
			float rcp_light_radius = recip( light_radius );

			vec3 light_dir;
			float attenuation = 0.0;

			light_dir = light_pos - raw_depth.xyz;
			float distance = length( light_dir );
			light_dir = normalize( light_dir );

      //out_color.xyz += vec3( distance > light_radius && distance < light_radius + 0.5 ? 1.0 : 0.0 ); // && dot( -light_dir, spot_direction_data[i].xyz ) > spot_cutoff_data[i]

			float coeff = 0.0;

  		attenuation = ( light_radius - distance ) * recip( light_radius );

      vec3 spot_direction = sld.d[index].spot_direction.xyz;
      float spot_cos_cutoff = sld.d[index].spot_direction.w;
      float spot_effect = dot( -light_dir, spot_direction );

      if( spot_effect > spot_cos_cutoff )
      {
        float spot_exponent = sld.d[index].spot_exponent;
        spot_effect = pow( spot_effect, spot_exponent );

        //if( attenuation > 0.0 )
        //{
          attenuation = spot_effect * recip( 1.0 - attenuation ) * attenuation + 1.0;
        //	attenuation = spot_effect * native_recip( attenuation ) + 1.0f;
        //	attenuation = spot_effect * attenuation + 1.0f;
        //}
      }

      attenuation -= 1.0;

			if( attenuation > 0.0 )
			{
        float shadow = 1;

        /**/
        int count = 0;
        int size = 5;
        float scale = 1.0;
        float k = 80;
        float bias = 0.001;

        vec4 ls_pos = sld.d[index].shadow_mat * vec4(raw_depth.xyz, 1);

        if( ls_pos.w > 0.0 )
        {
          vec4 shadow_coord = ls_pos / ls_pos.w; //transform to tex coords (0...1)

          if( bounds_check(shadow_coord.x) &&
              bounds_check(shadow_coord.y) &&
              bounds_check(shadow_coord.z) )
          {
            vec2 texcoord = shadow_coord.xy;
            shadow = 0;

            scale /= textureSize( spot_shadow_tex[shadow_tex_index], 0 ).x;
            scale *= 10;

            vec2 poisson_samples_25[25] =
            {
              vec2(-0.09956584, -0.9506168),
              vec2(0.0945536, -0.537095),
              vec2(-0.4186222, -0.8271356),
              vec2(0.2749816, -0.8264546),
              vec2(-0.3254788, -0.499952),
              vec2(-0.809692, -0.4665135),
              vec2(-0.2234173, -0.1774993),
              vec2(-0.7175227, -0.1786787),
              vec2(-0.964484, 0.03171529),
              vec2(0.0664617, -0.09118196),
              vec2(0.4066602, -0.1230906),
              vec2(0.6525381, -0.4510389),
              vec2(-0.507311, 0.1346684),
              vec2(-0.2599327, 0.4192313),
              vec2(-0.6924328, 0.3792625),
              vec2(0.1341985, 0.2107346),
              vec2(-0.3915392, 0.8384292),
              vec2(0.2936738, 0.6246353),
              vec2(-0.03307898, 0.7678965),
              vec2(0.6329493, 0.08375981),
              vec2(0.9811257, -0.1799406),
              vec2(0.4678418, 0.358797),
              vec2(0.8280757, 0.404021),
              vec2(0.700793, 0.6918929),
              vec2(0.2785904, 0.9276251)
            };

            /**/
            for( int x = 0; x < 25; ++x )
            {
              float distance_from_light = texture( spot_shadow_tex[shadow_tex_index], vec3(texcoord + scale * poisson_samples_25[x], shadow_layer_index) ).x;
              shadow += clamp( 2.0 - exp((abs(shadow_coord.z) - distance_from_light) * k), 0, 1 );
            }
            shadow = shadow / 25.0;
            /**/

            /**
            texcoord -= scale * vec2(1) * 0.5;
            for( int y = 0; y < size; ++y )
              for( int x = 0; x < size; ++x )
              {
                float distance_from_light = texture( spot_shadow_tex[shadow_tex_index], vec3(texcoord + scale * vec2(x, y), shadow_layer_index) ).x;
                shadow += clamp( 2.0 - exp((abs(shadow_coord.z) - distance_from_light) * k), 0, 1 );
                ++count;
              }
            shadow = shadow / float(count);
            /**/

            //float distance_from_light = texture( spot_shadow_tex[shadow_tex_index], vec3(texcoord, shadow_layer_index) ).x;
            //shadow = float( shadow_coord.z < distance_from_light );
            //shadow = clamp( 2.0 - exp((shadow_coord.z - distance_from_light) * k), 0, 1 );

            //out_color = vec4( shadow_tex_index * 0.25 );
            //out_color = vec4( texcoord, 0, 1 );
            //float distance_from_light = texture( spot_shadow_tex[shadow_tex_index], vec3(texcoord, shadow_layer_index) ).x;
            //shadow = clamp( 2.0 - exp((shadow_coord.z - distance_from_light) * k), 0, 1 );
            //out_color += vec4( shadow );

            //out_color = vec4( shadow_layer_index*0.5 );
          }
        }
        /**/

        /**/
        if( shadow > 0.0 )
        {
          out_color.xyz += brdf( index,
                       raw_albedo.xyz,
                       raw_normal.xyz,
                       light_dir,
                       view_dir,
                       gloss_factor,
                       attenuation,
                       sld.d[index].diffuse_color.xyz,
                       sld.d[index].specular_color.xyz )
                       * shadow * raw_ssao;
          //out_color = vec4(raw_ssao);
        }
        /**/

        //out_color.xyz += attenuation * vec3(0, 0.75, 1) * 0.01;
			}
		}

	  //out_color.xyz = max(raw_normal.xyz, 0); //view space normal
		//out_color.xyz = raw_albedo.xyz; //albedo
		//out_color.xyz = (float3)(raw_albedo.w); //specular intensity
		//out_color.xyz = vec3(raw_depth.x); //view space linear depth
		//out_color.xyz = raw_depth.xyz; //view space position

		//out_color.xyz += num_to_radar_colors( local_num_of_lights, 5 );

		//out_color.xyz = vec3(abs(normalize(vs_position_data[13].xyz)));
		//out_color = vec4(float(out_color.x > 0));
	}
  else
  {
    out_color.xyz = vec3(0, 0.75, 1);
  }

  //out_color = vec4(1);

	if( global_id.x <= global_size.x && global_id.y <= global_size.y )
	{
		//imageStore( result_tex, global_id, vec4( clamp(linear_to_gamma(tonemap(out_color.xyz)), 0.0, 1.0), 1.0 ) );
    //imageStore( result_tex, global_id, vec4( rgb_to_ycocg( clamp(tonemap(out_color.xyz), 0.0, 1.0), ivec2(global_id) ), 0, 1.0 ) );
    imageStore( result_tex, global_id, vec4( clamp(tonemap(out_color.xyz), 0.0, 1.0), 1.0 ) );

    //imageStore( result_tex, global_id, vec4( out_color.xyz, 1.0 ) );
	}
}
