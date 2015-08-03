#ifndef common_h
#define common_h

const float pi = 3.14159265;

vec3 right_vec[6] =
{
  vec3( 0, 0,  1),
  vec3( 0, 0, -1),
  vec3( 1, 0,  0),
  vec3( 1, 0,  0),
  vec3(-1, 0,  0),
  vec3( 1, 0,  0)
};

vec3 up_vec[6] =
{
  vec3( 0, 1,  0),
  vec3( 0, 1,  0),
  vec3( 0, 0,  1),
  vec3( 0, 0, -1),
  vec3( 0, 1,  0),
  vec3( 0, 1,  0)
};

const vec3 radar_colors[14] =
{
    vec3( 0, 0.9255, 0.9255 ),   // cyan
    vec3( 0, 0.62745, 0.9647 ),  // light blue
    vec3( 0, 0, 0.9647 ),        // blue
    vec3( 0, 1, 0 ),             // bright green
    vec3( 0, 0.7843, 0 ),        // green
    vec3( 0, 0.5647, 0 ),        // dark green
    vec3( 1, 1, 0 ),             // yellow
    vec3( 0.90588, 0.75294, 0 ), // yellow-orange
    vec3( 1, 0.5647, 0 ),        // orange
    vec3( 1, 0, 0 ),            // bright red
    vec3( 0.8392, 0, 0 ),        // red
    vec3( 0.75294, 0, 0 ),       // dark red
    vec3( 1, 0, 1 ),             // magenta
    vec3( 0.6, 0.3333, 0.7882 )  // purple
};

vec3 num_to_radar_colors( uint num, uint max_num )
{
    if( num == 0 )
    { // black for no lights
      return vec3( 0, 0, 0 );
    }
    else if( num == max_num )
    { // light purple for reaching the max
      return vec3( 0.847, 0.745, 0.921 );
    }
    else if ( num > max_num )
    { // white for going over the max
      return vec3(1);
    }
    else
    { // else use weather radar colors
        // use a log scale to provide more detail when the number of lights is smaller
        // want to find the base b such that the logb of max_num is 14
        // (because we have 14 radar colors)
        float log_base = exp2( 0.07142857 * log2( float(max_num) ) );

        // change of base
        // logb(x) = log2(x) / log2(b)
        uint color_index = uint( floor( log2( float(num) ) / log2( log_base ) ) );
        return radar_colors[color_index];
    }
}

bool check_sanity( float x )
{
  return isnan(x) || isinf(x);
}

bool bounds_check( float x )
{
  return x < 1.0 && x > 0.0;
}

float recip( float x )
{
  return 1.0 / x;
}

float sqr( float x )
{
	return x * x;
}

float linearize_depth( float depth, float near, float far )
{
  float proj_a = -(far + near) / (far - near);
  float proj_b = (-2 * far * near) / (far - near);
  return (-proj_b / (depth * 2 - 1 + proj_a)) / -far;
}

vec3 linear_depth_to_vs_pos( float linear_depth, vec2 position, float far )
{
  return vec3( position, far ) * linear_depth;
}

float avg( vec4 val )
{
  return (val.x + val.y + val.z + val.w) * 0.25;
}

vec2 rgb_to_ycocg(vec3 color, ivec2 frag_coord)
{
  vec3 ycocg = vec3( 0.25 * color.x + 0.5 * color.y + 0.25 * color.z,
                     0.5  * color.x - 0.5 * color.z + 0.5,
                    -0.25 * color.x + 0.5 * color.y - 0.25 * color.z + 0.5 );
  return
  ((frag_coord.x & 1) == (frag_coord.y & 1))
  //( mod( frag_coord.x, 2.0 ) == mod( frag_coord.y, 2.0 ) )
  ? ycocg.xz : ycocg.xy;
}

vec3 ycocg_to_rgb_helper(vec3 color)
{
  color.y -= 0.5;
  color.z -= 0.5;
  return vec3( color.x + color.y - color.z, color.x + color.z, color.x - color.y - color.z );
}

float ycocg_edge_filter(vec2 center, vec2 a0, vec2 a1, vec2 a2, vec2 a3)
{
  const float thresh = 30.0 / 255.0;

  vec4 lum = vec4( a0.x, a1.x, a2.x, a3.x );
  vec4 w = 1.0 - step( thresh, abs( lum - center.x ) );
  float ww = w.x + w.y + w.z + w.w;

  //Handle the special case where all the weights are zero.
  //In HDR scenes it's better to set the chrominance to zero.
  //Here we just use the chrominance of the first neighbor.
  w.x = ( ww == 0.0 ) ? 1.0 : w.x;
  ww = ( ww == 0.0 ) ? 1.0 : ww;

  return ( w.x * a0.y + w.y * a1.y + w.z * a2.y + w.w * a3.y ) / ww;
}

//tex is assumed to store ycocg in RG channels
vec3 ycocg_to_rgb(sampler2DArray tex, int index, ivec2 frag_coord, vec2 center )
{
  vec2 a0 = texelFetch(tex, ivec3(frag_coord + vec2(1, 0), index), 0).xy;
  vec2 a1 = texelFetch(tex, ivec3(frag_coord - vec2(1, 0), index), 0).xy;
  vec2 a2 = texelFetch(tex, ivec3(frag_coord + vec2(0, 1), index), 0).xy;
  vec2 a3 = texelFetch(tex, ivec3(frag_coord - vec2(0, 1), index), 0).xy;
  float chroma = ycocg_edge_filter( center, a0, a1, a2, a3 );

  vec3 col = vec3( center, chroma );
  col.xyz =
  ((frag_coord.x & 1) == (frag_coord.y & 1))
  //( mod( frag_coord.x, 2.0 ) == mod( frag_coord.y, 2.0 ) )
  ? col.xzy : col.xyz;
  return ycocg_to_rgb_helper( col );
}

//spheremap normal compression
vec2 compress_normal( vec3 n )
{
  float p = sqrt( n.z * 8 + 8 );
  return vec2( n.xy / p + 0.5 );
}

vec3 decompress_normal( vec2 enc )
{
  vec2 fenc = enc * 4 - 2;
  float f = dot( fenc, fenc );
  float g = sqrt( 1 - f / 4 );
  vec3 n;
  n.xy = fenc * g;
  n.z = 1 - f / 2;
  return n;
}

vec4 encode_rsm( vec3 albedo, vec3 normal, ivec2 frag_coord )
{
  return vec4( rgb_to_ycocg( albedo.xyz, frag_coord ), compress_normal( normal ) );
}

void decode_rsm( sampler2DArray tex, int index, vec2 texcoord, out vec3 albedo, out vec3 normal )
{
  ivec2 frag_coord = ivec2( texcoord * textureSize( tex, 0 ).xy );
  vec4 center = texelFetch( tex, ivec3( frag_coord, index ), 0 );
  albedo = ycocg_to_rgb( tex, index, frag_coord, center.xy );
  normal = decompress_normal( center.zw );
}

#endif
