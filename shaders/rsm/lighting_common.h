#ifndef lighting_common_h
#define lighting_common_h

vec3 tonemap_func(vec3 x, float a, float b, float c, float d, float e, float f)
{
  return ( ( x * ( a * x + c * b ) + d * e ) / ( x * ( a * x + b ) + d * f ) ) - e / f;
}
  
vec3 tonemap(vec3 col)
{
  //vec3 x = max( vec3(0), col - vec3(0.004));
  //return ( x * (6.2 * x + 0.5) ) / ( x * ( 6.2 * x + 1.7 ) + 0.06 );
  
  float a = 0.22; //Shoulder Strength
  float b = 0.30; //Linear Strength
  float c = 0.10; //Linear Angle
  float d = 0.20; //Toe Strength
  float e = 0.01; //Toe Numerator
  float f = 0.30; //Toe Denominator
  float linear_white = 11.2; //Linear White Point Value (11.2)
  //Note: E/F = Toe Angle
  
  return tonemap_func( col, a, b, c, d, e, f ) / tonemap_func( vec3(linear_white), a, b, c, d, e, f );
}

//NOTE: actually, just use SRGB, it's got better quality!
vec3 linear_to_gamma( vec3 col )
{
  return pow( col, vec3( 1/2.2 ) );
}

vec3 gamma_to_linear( vec3 col )
{
  return pow( col, vec3( 2.2 ) );
}

float roughness_to_spec_power(float m) 
{
  return 2.0 / (m * m) - 2.0;
}

float spec_power_to_roughness(float s) 
{
  return sqrt(2.0 / (s + 2.0));
}

float toksvig_aa(vec3 bump, float s)
{	
	//this is the alu based version
	float len = length( bump );
	float gloss = max(len / mix(s, 1.0, len), 0.01);
	
	return spec_power_to_roughness(gloss * s);
}

float fresnel_schlick( float v_dot_h, float f0 )
{
  float base = 1.0 - v_dot_h;
  float exponential = pow( base, 5.0 );
  return exponential + f0 * ( 1.0 - exponential );
}

float distribution_ggx( float n_dot_h, float alpha )
{  
  float cos_sqr = sqr(n_dot_h);
  float alpha_sqr = sqr(alpha);
  
  return alpha_sqr / ( pi * sqr( ( alpha_sqr - 1 ) * cos_sqr + 1 ) );
}

float geometric_torrance_sparrow( float n_dot_h, float n_dot_v, float v_dot_h, float n_dot_l )
{
	return min( 1.0, min( 2.0 * n_dot_h * n_dot_v / v_dot_h, 2.0 * n_dot_h * n_dot_l / v_dot_h ) );
}

float geometric_schlick_smith( float n_dot_v, float roughness )
{
  return n_dot_v / ( n_dot_v * (1 - roughness) + roughness );
}

float diffuse_lambert()
{
  return 1.0 / pi;
}

float diffuse_oren_nayar( float roughness, float n_dot_v, float n_dot_l, float v_dot_h )
{
  float v_dot_l = 2 * v_dot_h - 1;
  float m = sqr( roughness );
  float m2 = sqr( m );
  float c1 = 1 - 0.5 * m2 / ( m2 + 0.33 );
  float cos_ri = v_dot_l - n_dot_v * n_dot_l;
  float c2 = 0.45 * m2 / ( m2 + 0.09 ) * cos_ri;
  
  if( cos_ri >= 0 )
    c2 *= min( 1, n_dot_l / n_dot_v );
  else
    c2 *= n_dot_l;
  
  return diffuse_lambert() * ( n_dot_l * c1 + c2 );
}

vec3 brdf( int index, vec3 raw_albedo, vec3 raw_normal, vec3 light_dir, vec3 view_dir, float intensity, float attenuation, vec3 diffuse_color, vec3 specular_color )
{
  vec3 result = vec3( 0 );

  vec3 light_diffuse_color = diffuse_color.xyz;
  vec3 light_specular_color = specular_color.xyz;
  
  vec3 half_vector = normalize( light_dir + view_dir ); // * 0.5;

  float n_dot_l = clamp( dot( raw_normal, light_dir ), 0.0, 1.0 );
  float n_dot_h = clamp( dot( raw_normal, half_vector ), 0.0, 1.0 );
  float v_dot_h = clamp( dot( view_dir, half_vector ), 0.0, 1.0 );
  float n_dot_v = clamp( dot( raw_normal, view_dir ), 0.0, 1.0 );
  float l_dot_h = clamp( dot( light_dir, half_vector ), 0.0, 1.0 );

  float roughness = max( intensity, 0.02 );
  
  //float diffuse = diffuse_lambert() * n_dot_l;
  float diffuse = diffuse_oren_nayar( roughness, n_dot_v, n_dot_l, v_dot_h );
  
  float final_specular = 0;
  float final_diffuse = 0;
  
  if( n_dot_l > 0.0 )
  {
    /**/
    //F term
    float F = fresnel_schlick( l_dot_h, 0.028 );
    
    //D term
    //float D = distribution_ggx( n_dot_h, roughness );
    //float D = distribution_ggx( n_dot_h, pow(1 - roughness*0.7, 6.0) ); //TODO: remapped roughness?
    float D = distribution_ggx( n_dot_h, roughness );

    //G term
    //float G = geometric_torrance_sparrow( n_dot_h, n_dot_v, v_dot_h, n_dot_l );
    //float G = geometric_schlick_smith( n_dot_v, pow(0.8 + 0.5 * roughness, 2) * 0.5 );
    float G = geometric_schlick_smith( n_dot_v, roughness );

    float denom = (n_dot_l * n_dot_v) * 4.0; //TODO: do we need pi here?
    float specular = (D * G) * F 
                      * recip( denom > 0.0 ? denom : 1.0 ); //avoid div by 0 
    /**/
    
    //cheap blinn-phong specular
    //float specular = pow( n_dot_h, roughness );
  
    //result = vec3(float(check_sanity(specular)));
    final_specular = min(1.0, n_dot_l * specular);
  }

  final_diffuse = min(1 - final_specular, diffuse);
  
  result += final_specular * light_specular_color;
  result += final_diffuse * raw_albedo * light_diffuse_color;
  
  return result * attenuation;
}

#endif