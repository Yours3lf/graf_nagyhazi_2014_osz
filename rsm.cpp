//#define REDIRECT
#include "framework.h"

#include "debug_draw.h"

#include "octree.h"
#include "intersection.h"
#include "basic_types.h"

#include "browser.h"

#include <sstream>
#include <string>
#include <locale>
#include <codecvt>

using namespace prototyper;

DebugDrawManager ddman;
framework frm;

float player_health = 100;
vector<float> zombie_health;

union dcu
{
  struct{ u32 h, l; };
  struct{ u64 a; };
};

struct spot_light_data
{
  vec4 diffuse_color;
  vec4 specular_color; //w is light_size
  vec4 vs_position;
  float attenuation_end;
  float attenuation_cutoff; // ]0...1], 1 (need low values for nice attenuation)
  float radius;
  float spot_exponent;
  vec4 spot_direction; //w is spot_cutoff ([0...90], 180)
  mat4 spot_shadow_mat;
  ivec4 index;
}; //148 bytes, 9 vec4s + int

vector<spot_light_data> spot_data;

//shadow scale/bias matrix
mat4 bias_matrix( 0.5f, 0, 0, 0,
                  0, 0.5f, 0, 0,
                  0, 0, 0.5f, 0,
                  0.5f, 0.5f, 0.5f, 1 );

//pos x
mat4 posx = mat4(  0,  0, -1, 0,
                    0, -1,  0, 0,
                  -1,  0,  0, 0,
                    0,  0,  0, 1 );
//neg x
mat4 negx = mat4( 0,  0, 1, 0,
                  0, -1, 0, 0,
                  1,  0, 0, 0,
                  0,  0, 0, 1 );
//pos y
mat4 posy = mat4( 1, 0,  0, 0,
                  0, 0, -1, 0,
                  0, 1,  0, 0,
                  0, 0,  0, 1 );
//neg y
mat4 negy = mat4( 1,  0, 0, 0,
                  0,  0, 1, 0,
                  0, -1, 0, 0,
                  0,  0, 0, 1 );
//pos z
mat4 posz = mat4( 1,  0,  0, 0,
                  0, -1,  0, 0,
                  0,  0, -1, 0,
                  0,  0,  0, 1 );
//neg z
mat4 negz = mat4( -1,  0, 0, 0,
                    0, -1, 0, 0,
                    0,  0, 1, 0,
                    0,  0, 0, 1 );

const unsigned max_lights = 200;

class shader_variation_manager
{
  public:
    typedef vector<pair<string, unsigned> > base_files_type;
  private:
    unsigned counter;
    map<unsigned, map<u64, GLuint> > variations;
    map<unsigned, vector<string> > variation_strings;
    map<unsigned, base_files_type > base_files;
  public:

    void add_base_file(unsigned id, const string& s, unsigned type)
    {
      base_files[id].push_back( make_pair(s, type) );
    }

    unsigned add_variation(unsigned id, const string& s)
    {
      variation_strings[id].push_back(s);
      return variation_strings[id].size() - 1;
    }

    unsigned create_variation( const base_files_type& base )
    {
      int id = counter++;
      variations[id];
      base_files[id] = base;
      variation_strings[id];
      return id;
    }

    GLuint get_varation( unsigned id, u64 variation )
    {
      auto it = variations.find( id );
      if( it != variations.end() )
      {
        auto var_it = variations[id].find(variation);

        if( var_it != variations[id].end() )
        {
          return variations[id][variation];
        }
        else
        {
          //compile here
          for( auto& c : base_files[id] )
          {
            string var_str = "#version 430\n";

            for( int d = 0; d < sizeof(variation) * 8 && d < variation_strings[id].size(); ++d )
            {
              if( variation & (u64(1) << d) )
                var_str += variation_strings[id][d];
            }

            frm.load_shader( variations[id][variation], c.second, c.first, false, var_str );
          }

          return variations[id][variation];
        }
      }

      return 0;
    }

    shader_variation_manager() : counter(0) {}
};

class uniform_manager
{
  map<unsigned, map<string, int> > uniforms;
  public:
    int get_uniform( unsigned shader, const string& s )
    {
      auto& tmp = uniforms[shader];
      auto it = tmp.find(s);
      if( it != tmp.end() )
      {
        return it->second;
      }
      else
      {
        int val = glGetUniformLocation( shader, s.c_str() );
        tmp[s] = val;
        return val;
      }
    }

    int get_block_index( unsigned shader, const string& s )
    {
      auto& tmp = uniforms[shader];
      auto it = tmp.find(s);
      if( it != tmp.end() )
      {
        return it->second;
      }
      else
      {
        int val = glGetUniformBlockIndex( shader, s.c_str() );
        tmp[s] = val;
        return val;
      }
    }
};

const int sizes[] =
{
  256, 512, 1024, 2048
};

const int counts[] =
{
  128, 32, 8, 2
};

class shadow_map_manager
{
    enum sm_sizes
    {
      SQUATERK = 0, SHALFK, S1K, S2K, SLAST
    };

    GLuint shadow_arrays[SLAST];
    GLuint rsm_arrays[SLAST];
    vector<GLuint> fbos[SLAST];
  public:
    void init()
    {
      glGenTextures( SLAST, &shadow_arrays[0] );
      glGenTextures( SLAST, &rsm_arrays[0] ); //only for rsm

      for( int c = 0; c < SLAST; ++c )
      {
        /**/
        //only for rsm
        glBindTexture( GL_TEXTURE_2D_ARRAY, rsm_arrays[c] );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, sizes[c], sizes[c], counts[c], 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
        /**/

        glBindTexture( GL_TEXTURE_2D_ARRAY, shadow_arrays[c] );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT24, sizes[c], sizes[c], counts[c], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );

        fbos[c].resize(counts[c]);
        glGenFramebuffers( counts[c], &fbos[c][0] );

        for( int d = 0; d < counts[c]; ++d )
        {
          glBindFramebuffer( GL_FRAMEBUFFER, fbos[c][d] );
          //glDrawBuffer( GL_NONE ); //shadow tex only
          glDrawBuffer( GL_COLOR_ATTACHMENT0 ); //rsm

          glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, rsm_arrays[c], 0, d ); //only for rsm
          glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadow_arrays[c], 0, d );
          frm.check_fbo_status();
        }
      }
    }

    void assign_shadow_textures( const scene& s, vector<spot_light_data>& light_data, const mat4& mv )
    {
      static vector<pair<unsigned, float> > importances;
      importances.clear();

      vec3 vs_cam_pos = (mv * vec4( s.cam.pos, 1 )).xyz;

      for( auto& c : light_data )
      {
        float importance = c.attenuation_end / length( vs_cam_pos - c.vs_position.xyz );
        importances.push_back( make_pair(c.index.z, importance) );
      }

      sort( importances.begin(), importances.end(), [&]( pair<unsigned, float> a, pair<unsigned, float> b )
      {
        return a.second > b.second;
      } );

      int counter = 0;
      for( int c = SLAST-1; c > -1 && counter < light_data.size(); --c )
      {
        for( int d = 0; d < counts[c] && counter < light_data.size(); ++d )
        {
          light_data[importances[counter].first].index.x = c;
          light_data[importances[counter].first].index.y = d;

          counter++;
        }
      }
    }

    GLuint get_shadow_fbo( const ivec4& index )
    {
      assert( index.x < SLAST && index.y < fbos[index.x].size() );
      return fbos[index.x][index.y];
    }

    void bind_shadow_textures( unsigned index )
    {
      for( int c = 0; c < SLAST; ++c )
      {
        glActiveTexture(GL_TEXTURE0+index+c);
        glBindTexture(GL_TEXTURE_2D_ARRAY, shadow_arrays[c]);
      }
    }

    void bind_rsm_textures( unsigned index )
    {
      for( int c = 0; c < SLAST; ++c )
      {
        glActiveTexture(GL_TEXTURE0+index+c);
        glBindTexture(GL_TEXTURE_2D_ARRAY, rsm_arrays[c]);
      }
    }
};

void set_workgroup_size( vec2& gws, vec2& lws, vec2& dispatch_size, uvec2& screen )
{
  //set up work group sizes
  unsigned local_ws[2] = {16, 16};
  unsigned global_ws[2];
  unsigned gw = 0, gh = 0, count = 1;

  while( gw < screen.x )
  {
    gw = local_ws[0] * count;
    count++;
  }

  count = 1;

  while( gh < screen.y )
  {
    gh = local_ws[1] * count;
    count++;
  }

  global_ws[0] = gw;
  global_ws[1] = gh;

  gws = vec2( global_ws[0], global_ws[1] );
  lws = vec2( local_ws[0], local_ws[1] );
  dispatch_size = gws / lws;
}

spot_light_data create_spot_light( const mat4& modelview, const spot_light& l, int index, int orig_index )
{
  float spot_cos_cutoff = 0.0f;
  mm::vec3 light_spot_dir_buf;

  mm::vec4 light_pos_buf;

  spot_cos_cutoff = std::cos( mm::radians( l.spot_cutoff ) );
  light_spot_dir_buf = mm::normalize( ( modelview * mm::vec4( l.cam.view_dir.xyz, 0.0f ) ).xyz );
  light_pos_buf = modelview * mm::vec4( l.cam.pos.xyz, 1.0f );

  float att_end = 0.0f;

  if( l.att_type == attenuation_type::FULL )
  {
    att_end = l.radius / l.attenuation_coeff;
  }
  else if( l.att_type == attenuation_type::LINEAR )
  {
    att_end = l.radius;
  }

  spot_light_data cll;
  cll.diffuse_color = vec4( l.diffuse_color, 1 );
  cll.specular_color = vec4( l.specular_color, 1 );
  cll.vs_position = light_pos_buf;
  cll.attenuation_end = att_end;
  cll.attenuation_cutoff = l.attenuation_coeff;
  cll.radius = l.radius;
  cll.spot_direction = vec4( light_spot_dir_buf, 0 );
  cll.spot_exponent = l.spot_exponent;
  cll.spot_direction.w = spot_cos_cutoff;
  cll.spot_shadow_mat = l.shadow_mat;
  cll.index.z = index;
  cll.index.w = orig_index;

  return cll;
}

//if b collides w/ a, we set b's position, so that it's outside a
//an aabb is defined by it's position (center), it's half-extents, and min/max vertices
void collide_aabb_aabb( aabb& a, aabb& b )
{
  if( a.is_intersecting( &b ) && //there's an intersection, need collision response
      !b.is_inside( &a ) ) //but only if we're not inside
  {
    vec3 diff = b.get_pos() - a.get_pos();
    vec3 abs_diff = abs( diff );
    vec3 extent_diff = (a.get_extents() + b.get_extents()) - abs_diff;

    vec3 res = vec3( 0 );

    //min search
    int idx = 0;
    for( int c = 1; c < 3; ++c )
    {
      if( extent_diff[idx] > extent_diff[c] )
        idx = c;
    }

    //final "collision response"
    res[idx] = extent_diff[idx] * sign(diff[idx]);

    b = aabb( b.get_pos() + res, b.get_extents() );
  }
}

/////////////////////////////////////
//astar path finding stuff from here
/////////////////////////////////////
int astar_gridsize_y = 150; //150
int astar_gridsize_x = 200; //200
vec3 astar_offset = vec3( -200, 2, -150 ); //200 150
struct astar_node
{
  aabb bv;
  int pos;
  float g, f;
  bool walkable;
  astar_node()
  {
    g = 0; f = 0;
    pos = 0;
    walkable = true;
  }

  astar_node(const aabb& a, int b, bool c) : bv(a), pos(b), walkable(c)
  {
    g = 0; f = 0;
  }
};
vector<astar_node> astar_grid;

int xy_to_pos( int x, int y )
{
  return y * astar_gridsize_x + x;
}

void pos_to_xy( int pos, int& x, int& y )
{
  y = pos / astar_gridsize_x;
  x = pos % astar_gridsize_x;
}

void set_up_astar_grid( const scene& s, const vector<int>& ignore_list )
{
  for( int y = 0; y < astar_gridsize_y; ++y )
    for( int x = 0; x < astar_gridsize_x; ++x )
    {
      astar_grid.push_back( astar_node( aabb( vec3( x * 2, 0, y * 2 ) + astar_offset, vec3( 1 ) ), xy_to_pos(x, y), true ) );
    }

  for( auto& c : s.objects )
  {
    if( c.mesh_idx.size() > 1 ) //the zombie doesn't count...
    {
      for( auto& d : c.mesh_idx )
      {
        if( find(ignore_list.begin(), ignore_list.end(), d ) == ignore_list.end() )
        {
          for( auto& e : astar_grid )
          {
            if( s.meshes[d].trans_bv->is_intersecting( &e.bv ) )
            {
              e.walkable = false; //this cell is occupied
            }
          }
        }
      }
    }
  }
}

bool is_astar_cell_walkable( int x, int y )
{
  return xy_to_pos(x, y) < astar_grid.size() && astar_grid[xy_to_pos(x, y)].walkable;
}

int get_astar_cell_from_pos( const vec3& pos )
{
  aabb pos_aabb = aabb( vec3( pos.x, astar_offset.y, pos.z ), vec3( 0.0001 ) );

  int counter = 0;
  for( auto& c : astar_grid )
  {
    if( c.bv.is_intersecting( &pos_aabb ) )
      break;

    ++counter;
  }

  return counter;
}

void get_neighbours( int x, int y, vector<int>& n )
{
  bool s0 = false, s1 = false, s2 = false, s3 = false; //vert/hori
  bool d0, d1, d2, d3; //diag
  n.reserve(8);

  if( is_astar_cell_walkable( x, y - 1 ) )
  {
    n.push_back(xy_to_pos(x, y-1));
    s0 = true;
  }

  if( is_astar_cell_walkable( x + 1, y ) )
  {
    n.push_back(xy_to_pos(x+1, y));
    s1 = true;
  }

  if( is_astar_cell_walkable( x, y + 1 ) )
  {
    n.push_back(xy_to_pos(x, y+1));
    s2 = true;
  }

  if( is_astar_cell_walkable( x - 1, y ) )
  {
    n.push_back(xy_to_pos(x-1, y));
    s3 = true;
  }

  d0 = s3 && s0;
  d1 = s0 && s1;
  d2 = s1 && s2;
  d3 = s2 && s3;

  if( d0 && is_astar_cell_walkable( x - 1, y - 1 ) )
    n.push_back(xy_to_pos(x-1, y-1));

  if( d1 && is_astar_cell_walkable( x + 1, y - 1 ) )
    n.push_back(xy_to_pos(x+1, y-1));

  if( d2 && is_astar_cell_walkable( x + 1, y + 1 ) )
    n.push_back(xy_to_pos(x+1, y+1));

  if( d2 && is_astar_cell_walkable( x - 1, y + 1 ) )
    n.push_back(xy_to_pos(x-1, y+1));
}

int manhattan_dist( int dx, int dy )
{
  return dx + dy;
}

bool find_astar_path( int startx, int starty, int endx, int endy, vector<int>& path )
{
  if( !is_astar_cell_walkable( endx, endy ) || !is_astar_cell_walkable( startx, starty ) )
    return false;

  static vector<astar_node*> openlist;
  static vector<astar_node*> closelist;
  static map<astar_node*, astar_node*> came_from;
  openlist.clear();
  closelist.clear();
  came_from.clear();

  for( auto& c : astar_grid )
  {
    c.f = 0;
    c.g = 0;
  }

  openlist.push_back(&astar_grid[xy_to_pos(startx, starty)]);

  auto pop = [&]() -> astar_node*
  {
    int min = 0;
    for( int c = 1; c < openlist.size(); ++c )
    {
      if( openlist[c]->f < openlist[min]->f )
        min = c;
    }

    astar_node* n = openlist[min];

    auto it = openlist.begin();
    for( int c = 0; c < min; ++c )
      ++it;

    openlist.erase( it );

    return n;
  };

  auto is_in_closelist = [&]( int pos )
  {
    for( int c = 0; c < closelist.size(); ++c )
    {
      if(closelist[c]->pos == pos)
        return true;
    }

    return false;
  };

  auto is_in_openlist = [&]( int pos )
  {
    for( int c = 0; c < openlist.size(); ++c )
    {
      if(openlist[c]->pos == pos)
        return true;
    }

    return false;
  };

  while(!openlist.empty())
  {
    astar_node* current = pop();

    if( current->pos == xy_to_pos( endx, endy ) )
    {
      //reconstruct path
      int current_pos = xy_to_pos(endx, endy);

      path.push_back(current_pos);

      while( came_from.find(&astar_grid[current_pos]) != came_from.end() )
      {
        current_pos = came_from[&astar_grid[current_pos]]->pos;
        path.push_back(current_pos);
      }

      return true;
    }

    closelist.push_back(current);

    static vector<int> neighbours;
    neighbours.clear();

    int px, py;
    pos_to_xy( current->pos, px, py );
    get_neighbours( px, py, neighbours );

    for( auto& c : neighbours )
    {
      if( is_in_closelist(astar_grid[c].pos) )
        continue;

      int nx, ny;
      pos_to_xy( astar_grid[c].pos, nx, ny );

      float tentative_g_score = current->g + ((px - nx == 0 || py - ny == 0) ? 1 : std::sqrt(2.0f));

      if( !is_in_openlist(astar_grid[c].pos) || tentative_g_score < astar_grid[c].g )
      {
        came_from[&astar_grid[c]] = current;
        astar_grid[c].g = tentative_g_score;
        astar_grid[c].f = astar_grid[c].g + manhattan_dist(std::abs(nx - endx), std::abs(ny - endy));

        if( !is_in_openlist(astar_grid[c].pos) )
        {
          openlist.push_back(&astar_grid[c]);
        }
      }
    }
  }

  return false;
}

/////////////////////////////////
//end of astar stuff
/////////////////////////////////

void get_catmull_rom_advancement( vec3& pos, vec3& dir, const vector<int>& path )
{
  static vector<vec3> velocities;
  static vector<vec3> points;
  static vector<float> times;
  velocities.clear();
  points.clear();
  times.clear();

  if( path.size() < 2 )
  {
    return;
  }

  velocities.resize(path.size());
  points.resize(path.size());
  times.resize(path.size());

  points[0] = pos;

  int counter = 1;
  for( int c = path.size()-2; c > 0; --c )
  {
    points[counter] = astar_grid[path[c]].bv.get_pos();
    points[counter].y = 0;
    ++counter;
  }

  float delta = 1000;

  times[0] = 0;
  times[1] = length(points[1] - points[0]) / std::sqrt(2.0f) * delta;

  for( int c = 2; c < times.size(); ++c )
  {
    times[c] = times[c-1] + delta; //ms
  }

  velocities[0] = vec3(0);
  velocities.back() = vec3(0);

  float tau = 0;

  for( int c = 1; c < velocities.size()-1; ++c )
  {
    velocities[c] =  ( (points[c+1] - points[c]) / (times[c+1] - times[c]) + (points[c] - points[c-1]) / (times[c] - times[c-1]) ) * (1 - tau) * 0.5f;
  }


  ////////////////////////////

  int numseg = 200 / points.size();

  int i1 = 0;
  int i2 = i1+1 >= points.size()-1 ? points.size()-1 : i1+1;

  float dt = times[i2] - times[i1];

  vec3 a0 = points[i1];
  vec3 a1 = velocities[i1];
  vec3 a2 = (points[i2] - points[i1]) * 3 / (powf(dt, 2)) - (velocities[i2] + velocities[i1] * 2) / (dt);
  vec3 a3 = (points[i1] - points[i2]) * 2 / (powf(dt, 3)) + (velocities[i2] + velocities[i1]) / (powf(dt, 2));

  float t = 0;
  vec3 final_pos = a3 * powf(t, 3) + a2 * powf(t, 2) + a1 * (t) + a0;
  t += dt/(numseg);
  vec3 final_pos2 = a3 * powf(t, 3) + a2 * powf(t, 2) + a1 * (t) + a0;
  dir = normalize(final_pos2 - final_pos);

  /*for( int c = 0; c < points.size()-1; ++c )
  {
    int i1 = c;
    int i2 = i1+1 >= points.size()-1 ? points.size()-1 : i1+1;

    float dt = times[i2] - times[i1];

    vec3 a0 = points[i1];
    vec3 a1 = velocities[i1];
    vec3 a2 = (points[i2] - points[i1]) * 3 / (powf(dt, 2)) - (velocities[i2] + velocities[i1] * 2) / (dt);
    vec3 a3 = (points[i1] - points[i2]) * 2 / (powf(dt, 3)) + (velocities[i2] + velocities[i1]) / (powf(dt, 2));

    vec3 res = a3 * powf(0, 3) + a2 * powf(0, 2) + a1 * (0) + a0;

    for( float t = dt/numseg; t <= dt; t += dt / numseg )
    {
      vec3 res2 = a3 * powf(t, 3) + a2 * powf(t, 2) + a1 * (t) + a0;
      ddman.CreateLineSegment(res, res2, 0);
      res = res2;
    }
  }*/
}

void update_transformations( scene& s, octree<unsigned>* o )
{
  //update transformations
  static vector<pair<unsigned, shape*> > update_list;
  static vector<vec3> bv_vertices;
  update_list.clear();
  bv_vertices.clear();

  s.objects[0].transformation = mat4::identity;

  for( auto& c : s.objects )
  {
    mat4 inv_trans = inverse( c.transformation );

    for( auto& d : c.mesh_idx )
    {
      bv_vertices.clear();
      static_cast<aabb*>(s.meshes[d].local_bv)->get_vertices(bv_vertices);

      aabb a;

      for( auto& e : bv_vertices )
      {
        a.expand((c.transformation * vec4( e, 1 )).xyz);
      }

      //ddman.CreateAABoxPosEdges( a.pos, a.extents, 0 );

      s.meshes[d].inv_transformation = inv_trans;
      s.meshes[d].transformation = c.transformation;
      *static_cast<aabb*>(s.meshes[d].trans_bv) = a;

      update_list.push_back(make_pair(d, s.meshes[d].trans_bv));
    }
  }

  o->update(update_list);
}

namespace js
{
  static bool is_set = false;

  void bindings_complete( const browser_instance& w )
  {
    browser::get().execute_javascript( w,
    L"\
    window.open = function (open) \
    { \
      return function (url, name, features) \
      { \
        open_window(url); \
        return open.call ?  open.call(window, url, name, features):open( url, name, features); \
      }; \
    }(window.open);" );

    browser::get().execute_javascript( w,
    L"\
    var list = document.getElementsByTagName('input'); \
    for( var i = 0; i < list.length; ++i ) \
    { \
      var att = list[i].getAttribute('type'); \
      if(att && att == 'file') \
      { \
        list[i].onclick = function(){ choose_file('Open file of'); }; \
      } \
    } \
    " );

    std::wstringstream ws;
    ws << "bindings_complete();";

    browser::get().execute_javascript( w, ws.str() );

    is_set = true;
  }

  void cpp_to_js( const browser_instance& w, const std::wstring& str )
  {
    if( !is_set )
      return;

    std::wstringstream ws;
    ws << "cpp_to_js('" << str << "');";

    browser::get().execute_javascript( w, ws.str() );
  }

  void set_player_health( const browser_instance& w, int health )
  {
    if( !is_set )
      return;

    std::wstringstream ws;
    ws << "set_player_health('" << health << "');";

    browser::get().execute_javascript( w, ws.str() );
  }
}

void browser::onTitleChanged( Berkelium::Window* win,
                              Berkelium::WideString title )
{
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  std::wstring str( title.mData, title.mLength );
  frm.set_title( conv.to_bytes(str) );
}

int main( int argc, char** argv )
{
  shape::set_up_intersection();

  map<string, string> args;

  for( int c = 1; c < argc; ++c )
  {
    args[argv[c]] = c + 1 < argc ? argv[c + 1] : "";
    ++c;
  }

  cout << "Arguments: " << endl;
  for_each( args.begin(), args.end(), []( pair<string, string> p )
  {
    cout << p.first << " " << p.second << endl;
  } );

  uvec2 screen( 0 );
  bool fullscreen = false;
  bool silent = false;
  string title = "Reflective Shadow Mapping";

  /*
   * Process program arguments
   */

  stringstream ss;
  ss.str( args["--screenx"] );
  ss >> screen.x;
  ss.clear();
  ss.str( args["--screeny"] );
  ss >> screen.y;
  ss.clear();

  if( screen.x == 0 )
  {
    screen.x = 1280;
  }

  if( screen.y == 0 )
  {
    screen.y = 720;
  }

  try
  {
    args.at( "--fullscreen" );
    fullscreen = true;
  }
  catch( ... ) {}

  try
  {
    args.at( "--help" );
    cout << title << ", written by Marton Tamas." << endl <<
         "Usage: --silent      //don't display FPS info in the terminal" << endl <<
         "       --screenx num //set screen width (default:1280)" << endl <<
         "       --screeny num //set screen height (default:720)" << endl <<
         "       --fullscreen  //set fullscreen, windowed by default" << endl <<
         "       --help        //display this information" << endl;
    return 0;
  }
  catch( ... ) {}

  try
  {
    args.at( "--silent" );
    silent = true;
  }
  catch( ... ) {}

  /*
   * Initialize the OpenGL context
   */

  frm.init( screen, title, fullscreen );
  frm.set_vsync( true );

  //set opengl settings
  glEnable( GL_DEPTH_TEST );
  glDepthFunc( GL_LEQUAL );
  glFrontFace( GL_CCW );
  glEnable( GL_CULL_FACE );
  glClearColor( 0.5f, 0.5f, 0.8f, 0.0f ); //sky color
  glClearDepth( 1.0f );
  glHint(GL_GENERATE_MIPMAP_HINT, GL_NICEST);

  frm.get_opengl_error();

  /*
   * Set up mymath
   */

  glViewport( 0, 0, screen.x, screen.y );

  /*
   * Set up the scene
   */

  octree<unsigned>* o = new octree<unsigned>(aabb(vec3(0), vec3(512)));
  o->set_up_octree(&o);

  float move_amount = 10;

  string app_path = frm.get_app_path();

  scene s;
  mesh::load_into_meshes( app_path + "resources/sponza_dae/sponza.dae", s, true );
  //mesh::load_into_meshes( app_path + "resources/sponza_dae/sponza.obj", s, true );
  mesh::load_into_meshes( app_path + "resources/zombie/zombie.dae", s, true );

  s.objects[1].transformation = create_scale(vec3(10)) * create_rotation(radians(-90), vec3( 1, 0, 0 ) );

  s.cam.pos = vec3( 100, 9, 0 );
  s.cam.rotate_y( radians(90) );

  //create aabb and insert into octree
  int counter = 0;
  for( auto& c : s.meshes )
  {
    c.local_bv = new aabb( );
    c.trans_bv = new aabb();

    for( int i = 0; i < c.vertices.size( ); i += 3 )
    {
      static_cast<aabb*>( c.local_bv )->expand( vec3(
        c.vertices[i + 0],
        c.vertices[i + 1],
        c.vertices[i + 2]
        ) );
    }

    //ddman.CreateAABoxPosEdges( static_cast<aabb*>(c.bv)->pos, static_cast<aabb*>(c.bv)->extents, -1 );

    *static_cast<aabb*>( c.trans_bv ) = *static_cast<aabb*>( c.local_bv );
    o->insert( counter++, c.local_bv );
  }

  //set up cam
  s.near = 2.5;
  s.far = 1000;
  s.fov = 45;
  s.aspect = float(screen.x) / screen.y;

  s.f.set_perspective( radians( s.fov ), ( float )screen.x / ( float )screen.y, s.near, s.far );

  vector<int> collision_ignore_list;
  aabb player_aabb = aabb( s.cam.pos, vec3( 2.5 ) );

  for( int c = 0; c < s.meshes.size(); ++c )
  {
    if( player_aabb.is_inside( s.meshes[c].local_bv ) )
    {
      collision_ignore_list.push_back(c);
    }
  }

  glViewport( 0, 0, screen.x, screen.y );

  vec2 gws, lws, dispatch_size;
  set_workgroup_size( gws, lws, dispatch_size, screen );

  for( int c = -5; c < 5; ++c )
  {
    spot_light l;
    l.cam.pos.x = 10 + c * 25;
    l.cam.pos.y = 3;
    l.spot_cutoff = 45;
    l.diffuse_color = vec3(1);
    l.specular_color = l.diffuse_color;
    l.radius = 100;
    l.attenuation_coeff = 0.25;
    l.att_type = LINEAR;
    l.spot_exponent = 40;
    l.bv = new sphere( l.cam.pos, l.radius );
    //l.bv = new frustum();
    l.the_frame.set_perspective( radians( l.spot_cutoff*2 ), 1, 1, l.radius );
    //static_cast<frustum*>(l.bv)->set_up(l.cam);
    s.spot_lights.push_back(l);
  }

  for( int c = -5; c < 5; ++c )
  {
    spot_light l;
    l.cam.view_dir = vec3( 0, 0, 1 );
    l.cam.pos.x = 10 + c * 25;
    l.cam.pos.y = 3;
    l.cam.pos.z = 3;
    l.spot_cutoff = 45;
    l.diffuse_color = vec3(1);
    l.specular_color = l.diffuse_color;
    l.radius = 100;
    l.attenuation_coeff = 0.25;
    l.att_type = LINEAR;
    l.spot_exponent = 40;
    l.bv = new sphere( l.cam.pos, l.radius );
    //l.bv = new frustum();
    l.the_frame.set_perspective( radians( l.spot_cutoff*2 ), 1, 1, l.radius );
    //static_cast<frustum*>(l.bv)->set_up(l.cam);
    s.spot_lights.push_back(l);
  }

  for( auto& c : s.spot_lights )
  {
    //ddman.CreateFrustum( *c.cam.get_frame(), c.cam.pos, 1, -1 );
  }

  GLuint ss_quad = frm.create_quad( vec3(-1, -1, 0), vec3(1, -1, 0), vec3(-1, 1, 0), vec3(1, 1, 0) );
  GLuint quad = frm.create_quad( s.f.far_ll.xyz, s.f.far_lr.xyz, s.f.far_ul.xyz, s.f.far_ur.xyz );

  update_transformations( s, o );
  set_up_astar_grid( s, collision_ignore_list );

  /*
   * Set up the browser
   */

  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;

  browser::get().init( conv.from_bytes(app_path) + L"resources/berkelium/win32" );

  browser_instance b;
  browser::get().create( b, screen );
  browser::get().navigate( b, "file:///" + app_path + "resources/rsm_ui/ui.html" );

  browser::get().register_callback( L"js_to_cpp", fun( [=]( std::wstring str )
  {
    wcout << str << endl;
    js::cpp_to_js( browser::get().get_last_callback_window(), L"cpp to js function call" );
  }, std::wstring() ) );

  /*
   * Set up the shaders
   */

  shader_variation_manager svarman;
  uniform_manager uman;
  shadow_map_manager sman;
  sman.init();

  GLuint browser_shader = 0;
  frm.load_shader( browser_shader, GL_VERTEX_SHADER, app_path + "shaders/browser/browser.vs" );
  frm.load_shader( browser_shader, GL_FRAGMENT_SHADER, app_path + "shaders/browser/browser.ps" );

  GLuint debug_shader = 0;
  frm.load_shader( debug_shader, GL_VERTEX_SHADER, app_path + "shaders/rsm/debug.vs" );
  frm.load_shader( debug_shader, GL_FRAGMENT_SHADER, app_path + "shaders/rsm/debug.ps" );

  shader_variation_manager::base_files_type gbuffer_base_files;
  gbuffer_base_files.push_back(make_pair(app_path + "shaders/rsm/gbuffer.vs", GL_VERTEX_SHADER));
  gbuffer_base_files.push_back(make_pair(app_path + "shaders/rsm/gbuffer.ps", GL_FRAGMENT_SHADER));
  unsigned gbuffer_variations = svarman.create_variation(gbuffer_base_files);

  unsigned gbuffer_diffuse_var = svarman.add_variation(gbuffer_variations, "\n" );
  unsigned gbuffer_normal_map_var = svarman.add_variation(gbuffer_variations, "#define NORMAL_MAP\n" );
  unsigned gbuffer_roughness_map_var = svarman.add_variation(gbuffer_variations, "#define ROUGHNESS_MAP\n" );
  unsigned gbuffer_transparent_discard_var = svarman.add_variation(gbuffer_variations, "#define TRANSPARENT_DISCARD\n" );
  unsigned gbuffer_animated_var = svarman.add_variation(gbuffer_variations, "#define GPU_SKINNING\n" );

  shader_variation_manager::base_files_type forward_lighting_base_files;
  forward_lighting_base_files.push_back(make_pair(app_path + "shaders/rsm/forward_lighting.vs", GL_VERTEX_SHADER));
  forward_lighting_base_files.push_back(make_pair(app_path + "shaders/rsm/forward_lighting.ps", GL_FRAGMENT_SHADER));
  unsigned forward_lighting_variations = svarman.create_variation(forward_lighting_base_files);

  unsigned forward_lighting_diffuse_var = svarman.add_variation(forward_lighting_variations, "\n" );
  unsigned forward_lighting_normal_map_var = svarman.add_variation(forward_lighting_variations, "#define NORMAL_MAP\n" );
  unsigned forward_lighting_roughness_map_var = svarman.add_variation(forward_lighting_variations, "#define ROUGHNESS_MAP\n" );
  unsigned forward_animated_var = svarman.add_variation(forward_lighting_variations, "#define GPU_SKINNING\n" );

  GLuint light_cull_shader = 0;
  frm.load_shader( light_cull_shader, GL_COMPUTE_SHADER, app_path + "shaders/rsm/light_cull.cs" );

  GLuint lighting_shader = 0;
  frm.load_shader( lighting_shader, GL_COMPUTE_SHADER, app_path + "shaders/rsm/compute_light.cs" );

  GLuint display_shader = 0;
  frm.load_shader( display_shader, GL_VERTEX_SHADER, app_path + "shaders/rsm/display.vs" );
  frm.load_shader( display_shader, GL_FRAGMENT_SHADER, app_path + "shaders/rsm/display.ps" );

  shader_variation_manager::base_files_type spot_shadow_gen_base_files;
  spot_shadow_gen_base_files.push_back(make_pair(app_path + "shaders/rsm/spot_shadow_gen.vs", GL_VERTEX_SHADER));
  spot_shadow_gen_base_files.push_back(make_pair(app_path + "shaders/rsm/spot_shadow_gen.ps", GL_FRAGMENT_SHADER));
  unsigned spot_shadow_gen_variations = svarman.create_variation(spot_shadow_gen_base_files);

  unsigned spot_shadow_gen_normal_var = svarman.add_variation(spot_shadow_gen_variations, "\n" );
  unsigned spot_shadow_gen_animated_var = svarman.add_variation(spot_shadow_gen_variations, "#define GPU_SKINNING\n" );

  shader_variation_manager::base_files_type spot_rsm_gen_base_files;
  spot_rsm_gen_base_files.push_back(make_pair(app_path + "shaders/rsm/spot_rsm_gen.vs", GL_VERTEX_SHADER));
  spot_rsm_gen_base_files.push_back(make_pair(app_path + "shaders/rsm/spot_rsm_gen.ps", GL_FRAGMENT_SHADER));
  unsigned spot_rsm_gen_variations = svarman.create_variation(spot_rsm_gen_base_files);

  unsigned spot_rsm_gen_normal_var = svarman.add_variation(spot_rsm_gen_variations, "\n" );
  unsigned spot_rsm_gen_animated_var = svarman.add_variation(spot_rsm_gen_variations, "#define GPU_SKINNING\n" );

  GLuint ssao_shader = 0;
  frm.load_shader( ssao_shader, GL_VERTEX_SHADER, app_path + "shaders/rsm/ssao.vs" );
  frm.load_shader( ssao_shader, GL_FRAGMENT_SHADER, app_path + "shaders/rsm/ssao.ps" );

  GLuint ssao_blur_shader = 0;
  frm.load_shader( ssao_blur_shader, GL_VERTEX_SHADER, app_path + "shaders/rsm/ssao_blur.vs" );
  frm.load_shader( ssao_blur_shader, GL_FRAGMENT_SHADER, app_path + "shaders/rsm/ssao_blur.ps" );

  GLuint downsample_shader = 0;
  frm.load_shader( downsample_shader, GL_VERTEX_SHADER, app_path + "shaders/rsm/downsample.vs" );
  frm.load_shader( downsample_shader, GL_FRAGMENT_SHADER, app_path + "shaders/rsm/downsample.ps" );

  GLuint vpl_gen_shader = 0;
  frm.load_shader( vpl_gen_shader, GL_COMPUTE_SHADER, app_path + "shaders/rsm/vpl_gen.cs" );

  /*
   * Set up ubos
   */

  //int size = sizeof( spot_light_data );

  GLuint spot_light_data_ubo;
  glGenBuffers( 1, &spot_light_data_ubo );
  glBindBuffer( GL_UNIFORM_BUFFER, spot_light_data_ubo );
  glBufferData( GL_UNIFORM_BUFFER, max_lights * sizeof(spot_light_data), 0, GL_DYNAMIC_DRAW );

  GLuint bone_data_ubo;
  glGenBuffers( 1, &bone_data_ubo );
  glBindBuffer( GL_UNIFORM_BUFFER, bone_data_ubo );
  glBufferData( GL_UNIFORM_BUFFER, 100 * sizeof(mat4), 0, GL_DYNAMIC_DRAW );

  /*
   * Set up fbos / textures
   */

  //set up deferred shading
  GLuint depth_texture = 0;
  glGenTextures( 1, &depth_texture );

  glBindTexture( GL_TEXTURE_2D, depth_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, screen.x, screen.y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );

  //RGB8 view space normals
  //A8 reflectivity
  GLuint normal_texture = 0;
  glGenTextures( 1, &normal_texture );

  glBindTexture( GL_TEXTURE_2D, normal_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x, screen.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  //RGB8 albedo
  //A8 roughness
  GLuint albedo_texture = 0;
  glGenTextures( 1, &albedo_texture );

  glBindTexture( GL_TEXTURE_2D, albedo_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x, screen.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  //set up deferred shading
  GLuint gbuffer_fbo = 0;
  glGenFramebuffers( 1, &gbuffer_fbo );
  glBindFramebuffer( GL_FRAMEBUFFER, gbuffer_fbo );
  GLenum rts[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
  glDrawBuffers( 2, rts );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, albedo_texture, 0 );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, normal_texture, 0 );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0 );

  frm.check_fbo_status();

  //set up light cull texture
  GLuint light_cull_texture;
  glGenTextures( 1, &light_cull_texture );
  glBindTexture( GL_TEXTURE_BUFFER, light_cull_texture );

  GLuint light_cull_buffer; //TBO
  glGenBuffers( 1, &light_cull_buffer );
  glBindBuffer( GL_TEXTURE_BUFFER, light_cull_buffer );
  glBufferData( GL_TEXTURE_BUFFER, 1024 * dispatch_size.x * dispatch_size.y * sizeof(unsigned), 0, GL_DYNAMIC_DRAW );
  glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, light_cull_buffer );

  //set up light result texture
  GLuint result_texture = 0;
  glGenTextures( 1, &result_texture );
  glBindTexture( GL_TEXTURE_2D, result_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, screen.x, screen.y, 0, GL_RGBA, GL_FLOAT, 0 );

  //set up forward lighting fbo
  GLuint forward_lighting_fbo = 0;
  glGenFramebuffers( 1, &forward_lighting_fbo );
  glBindFramebuffer( GL_FRAMEBUFFER, forward_lighting_fbo );
  glDrawBuffer( GL_COLOR_ATTACHMENT0 );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, result_texture, 0 );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0 );

  frm.check_fbo_status();

  //set up random texture
  vector< vec3 > random_nums;
  random_nums.resize( 16 * 16 );
  for( int c = 0; c < random_nums.size(); ++c )
  {
    random_nums[c] = vec3( frm.get_random_num( 0, 1 ), frm.get_random_num( 0, 1 ), frm.get_random_num( 0, 1 ) );
  }

  GLuint random_texture = 0;
  glGenTextures( 1, &random_texture );
  glBindTexture( GL_TEXTURE_2D, random_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB8, 16, 16, 0, GL_RGB, GL_FLOAT, &random_nums[0][0] );

  GLuint ssao_tex = 0;
  glGenTextures( 1, &ssao_tex );
  glBindTexture( GL_TEXTURE_2D, ssao_tex );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  //glTexImage2D( GL_TEXTURE_2D, 0, GL_R8, screen.x, screen.y, 0, GL_RED, GL_UNSIGNED_BYTE, 0 );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_R8, screen.x/2, screen.y/2, 0, GL_RED, GL_UNSIGNED_BYTE, 0 );

  GLuint ssao_blur_tex = 0;
  glGenTextures( 1, &ssao_blur_tex );
  glBindTexture( GL_TEXTURE_2D, ssao_blur_tex );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  //glTexImage2D( GL_TEXTURE_2D, 0, GL_R8, screen.x, screen.y, 0, GL_RED, GL_UNSIGNED_BYTE, 0 );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_R8, screen.x/2, screen.y/2, 0, GL_RED, GL_UNSIGNED_BYTE, 0 );

  //set up ssao fbo
  GLuint ssao_fbo = 0;
  glGenFramebuffers( 1, &ssao_fbo );
  glBindFramebuffer( GL_FRAMEBUFFER, ssao_fbo );
  glDrawBuffer( GL_COLOR_ATTACHMENT0 );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssao_tex, 0 );

  frm.check_fbo_status();

  //set up ssao fbo
  GLuint ssao_blur_fbo = 0;
  glGenFramebuffers( 1, &ssao_blur_fbo );
  glBindFramebuffer( GL_FRAMEBUFFER, ssao_blur_fbo );
  glDrawBuffer( GL_COLOR_ATTACHMENT0 );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssao_blur_tex, 0 );

  frm.check_fbo_status();

  //set up downsampled depth texture
  GLuint half_depth_texture = 0;
  glGenTextures( 1, &half_depth_texture );

  glBindTexture( GL_TEXTURE_2D, half_depth_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, screen.x/2, screen.y/2, 0, GL_RED, GL_FLOAT, 0 );

  GLuint half_normal_texture = 0;
  glGenTextures( 1, &half_normal_texture );

  glBindTexture( GL_TEXTURE_2D, half_normal_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, screen.x/2, screen.y/2, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  //set up downsample fbo
  GLuint downsample_fbo = 0;
  glGenFramebuffers( 1, &downsample_fbo );
  glBindFramebuffer( GL_FRAMEBUFFER, downsample_fbo );
  //glDrawBuffer( GL_COLOR_ATTACHMENT0 );
  GLenum bufs[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
  glDrawBuffers( 2, bufs );

  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, half_depth_texture, 0 );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, half_normal_texture, 0 );

  frm.check_fbo_status();

  //set up importance downsampling textures
  GLuint importance_128_rsm_texture = 0;
  glGenTextures( 1, &importance_128_rsm_texture );

  glBindTexture( GL_TEXTURE_2D, importance_128_rsm_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, 128, 128, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  GLuint importance_128_depth_texture = 0;
  glGenTextures( 1, &importance_128_depth_texture );

  glBindTexture( GL_TEXTURE_2D, importance_128_depth_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, 128, 128, 0, GL_RED, GL_FLOAT, 0 );

  /*
   * Handle events
   */

  bool warped = false, ignore = true;
  vec2 movement_speed = vec2(0);

  auto event_handler = [&]( const sf::Event & ev )
  {
    switch( ev.type )
    {
      case sf::Event::MouseMoved:
        {
          vec2 mpos( ev.mouseMove.x / float( screen.x ), ev.mouseMove.y / float( screen.y ) );


          /**/

          if( warped )
          {
            ignore = false;
          }
          else
          {
            frm.set_mouse_pos( ivec2( screen.x / 2.0f, screen.y / 2.0f ) );
            warped = true;
            ignore = true;
          }

          if( !ignore && all( notEqual( mpos, vec2( 0.5 ) ) ) )
          {
            s.cam.rotate( radians( -180.0f * ( mpos.x - 0.5f ) ), vec3( 0.0f, 1.0f, 0.0f ) );
            s.cam.rotate_x( radians( -180.0f * ( mpos.y - 0.5f ) ) );
            frm.set_mouse_pos( ivec2( screen.x / 2.0f, screen.y / 2.0f ) );
            warped = true;
          }

          /**/

          browser::get().mouse_moved( b, mpos );

          break;
        }
      case sf::Event::KeyPressed:
        {
          /*if( ev.key.code == sf::Keyboard::A )
          {
            cam.rotate_y( radians( cam_rotation_amount ) );
          }*/
        }
      case sf::Event::TextEntered:
        {
          wchar_t txt[2];
          txt[0] = ev.text.unicode;
          txt[1] = '\0';
          browser::get().text_entered( b, txt );

          break;
        }
      case sf::Event::MouseButtonPressed:
        {
          if( ev.mouseButton.button == sf::Mouse::Left )
          {
            browser::get().mouse_button_event( b, sf::Mouse::Left, true );
          }
          else
          {
            browser::get().mouse_button_event( b, sf::Mouse::Right, true );
          }

          break;
        }
      case sf::Event::MouseButtonReleased:
        {
          if( ev.mouseButton.button == sf::Mouse::Left )
          {
            browser::get().mouse_button_event( b, sf::Mouse::Left, false );
          }
          else
          {
            browser::get().mouse_button_event( b, sf::Mouse::Right, false );
          }

          break;
        }
      case sf::Event::MouseWheelMoved:
        {
          browser::get().mouse_wheel_moved( b, ev.mouseWheel.delta * 100.0f );

          break;
        }
      default:
        break;
    }
  };

  /*
   * Render
   */

  sf::Clock timer;
  timer.restart();

  sf::Clock movement_timer;
  movement_timer.restart();

  float orig_mov_amount = move_amount;

  frm.display( [&]
  {
    frm.handle_events( event_handler );

    /**/

    float seconds = movement_timer.getElapsedTime().asMilliseconds() / 1000.0f;

    if( sf::Keyboard::isKeyPressed( sf::Keyboard::LShift ) || sf::Keyboard::isKeyPressed( sf::Keyboard::RShift ) )
    {
      move_amount = orig_mov_amount * 3.0f;
    }
    else
    {
      move_amount = orig_mov_amount;
    }

    if( seconds > 0.01667 )
    {
      //update the browser
      browser::get().update();

      //move camera
      if( sf::Keyboard::isKeyPressed( sf::Keyboard::A ) )
      {
        movement_speed.x -= move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::D ) )
      {
        movement_speed.x += move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::W ) )
      {
        movement_speed.y += move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::S ) )
      {
        movement_speed.y -= move_amount;
      }

      vec3 right = cross( s.cam.up_vector, s.cam.view_dir );
      s.cam.translate( normalize(vec3(s.cam.view_dir.x, 0, s.cam.view_dir.z)) * movement_speed.y * seconds );
      s.cam.translate( -normalize(vec3(right.x, 0, right.z)) * movement_speed.x * seconds );
      movement_speed *= 0.5;

      movement_timer.restart();
    }

    /**/

    //-----------------------------
    //update animation
    //-----------------------------

    //update animation bones
    static vector<mat4> bones;
    static bool did_init = false;
    bones.resize(100);

    if( !did_init )
    {
      for( auto& i : bones )
      {
        i = mat4::identity;
      }
      did_init = true;
    }

    if( seconds > 0.01667 )
    {
      mesh::update_animation( timer.getElapsedTime().asMilliseconds() * 0.001f, s, &bones[0] );

      if( bones.size() > 0 )
      {
        glBindBuffer( GL_UNIFORM_BUFFER, bone_data_ubo );
        glBufferSubData( GL_UNIFORM_BUFFER, 0, bones.size() * sizeof( mat4 ), &bones[0] );
      }
    }

    /**/

    //-----------------------------
    //set up matrices
    //-----------------------------

    /**/

    mat4 proj = s.f.projection_matrix;
    mat4 view_mat = s.cam.get_matrix();
    mat4 viewproj_mat = proj * view_mat;
    mat4 inv_viewproj_mat = inverse( viewproj_mat );
    mat4 inv_view_mat = inverse( view_mat );
    mat3 normal_mat = mat3( view_mat );
    mat3 inv_normal_mat = inverse( normal_mat );
    vec4 vs_eye_pos = view_mat * vec4( s.cam.pos, 1 );
    mat4 identity = mat4(1);

    frustum f;
    f.set_up( s.cam, s.f );

    //cull objects
    static vector<unsigned> culled_objs;
    culled_objs.clear();
    o->get_culled_objects(culled_objs, &f);

    /**/

    //-----------------------------
    //set up lights
    //-----------------------------

    static vector< unsigned > culled_spot_lights;
    culled_spot_lights.clear();

    for( int c = 0; c < s.spot_lights.size(); ++c )
    {
      if( f.is_intersecting( s.spot_lights[c].bv ) )
      {
        culled_spot_lights.push_back(c);
      }
    }

    spot_data.clear();

    int counter = 0;
    for( auto& c : culled_spot_lights )
    {
      spot_data.push_back(create_spot_light( view_mat, s.spot_lights[c], counter, c ));

      counter++;
    }

    sman.assign_shadow_textures( s, spot_data, view_mat );

    //-----------------------------
    //gbuffer rendering
    //-----------------------------

    /**/

    glViewport( 0, 0, screen.x, screen.y );

    glEnable( GL_DEPTH_TEST );

    glBindFramebuffer( GL_FRAMEBUFFER, gbuffer_fbo );
    //glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    static vector<std::pair< unsigned, dcu > > draw_calls;
    draw_calls.clear();

    auto get_object_depth = [&](  unsigned i, const camera<float>& cam ) -> u32
    {
      auto get_to_cam_vec = [&]() -> vec3
      {
        vec3 to_cam_vec;

        for( int c = 0; c < 3; ++c )
        {
          if( std::abs( cam.pos[c] - static_cast<aabb*>(s.meshes[i].trans_bv)->min[c] ) <
              std::abs( cam.pos[c] - static_cast<aabb*>(s.meshes[i].trans_bv)->max[c] ) )
          {
            to_cam_vec[c] = cam.pos[c] - static_cast<aabb*>(s.meshes[i].trans_bv)->min[c];
          }
          else
          {
            to_cam_vec[c] = cam.pos[c] - static_cast<aabb*>(s.meshes[i].trans_bv)->max[c];
          }
        }

        return to_cam_vec;
      };

      vec3 to_cam = get_to_cam_vec();
      float sqr_len = dot( to_cam, to_cam );
      float sqr_len_scaled = sqr_len / (s.far * s.far);
      u64 precision_scaler = std::pow(u64(2), u64(32)) - u64(1);
      u32 depth = std::floor(precision_scaler * sqr_len_scaled);

      return depth;
    };

    for( auto& c : culled_objs )
    {
      if( s.meshes[c].trans_bv->is_intersecting( &f ) )
      {
        unsigned depth_offset = 0;//32;
        unsigned material_offset = 32;//0;
        dcu adcu;
        adcu.a = (u64)0 |
        (s.materials[c].is_transparent ? (u64(1) << (gbuffer_transparent_discard_var+material_offset)) : 0) |
        (s.materials[c].normal_tex ? (u64(1) << (gbuffer_normal_map_var+material_offset)) : 0) |
        (s.materials[c].specular_tex ? (u64(1) << (gbuffer_roughness_map_var+material_offset)) : 0) |
        (s.materials[c].diffuse_tex ? (u64(1) << (gbuffer_diffuse_var+material_offset)) : 0) |
        (s.materials[c].is_animated ? (u64(1) << (gbuffer_animated_var+material_offset)) : 0) |
        ((u64)(get_object_depth(c, s.cam)) << depth_offset);

        draw_calls.push_back( make_pair(c, adcu ) );
      }
    }

    //sort draw calls front to back
    std::sort( draw_calls.begin(), draw_calls.end(), [&]( pair<unsigned, dcu> a, pair<unsigned, dcu> b ) -> bool
    {
      return a.second.a < b.second.a;
    } );

    //vector<std::pair< unsigned, dcu > >& ref = draw_calls;

    static vector<unsigned> used_shaders;
    used_shaders.clear();

    unsigned last_tex0 = 0, last_tex1 = 0, last_tex2 = 0;
    unsigned last_shader = 0;
    for( auto& c : draw_calls )
    {
      u64 variation_id = 0;

      if( s.materials[c.first].is_transparent )
      {
        //variation_id |= (1 << gbuffer_transparent_discard_var);
        continue;
      }

      if( s.materials[c.first].diffuse_tex )
      {
        if( last_tex0 != s.materials[c.first].diffuse_tex )
        {
          glActiveTexture( GL_TEXTURE0 );
          glBindTexture( GL_TEXTURE_2D, s.materials[c.first].diffuse_tex );
          last_tex0 = s.materials[c.first].diffuse_tex;
        }

        variation_id |= (1 << gbuffer_diffuse_var);
      }

      if( s.materials[c.first].normal_tex )
      {
        if( last_tex1 != s.materials[c.first].normal_tex )
        {
          glActiveTexture( GL_TEXTURE1 );
          glBindTexture( GL_TEXTURE_2D, s.materials[c.first].normal_tex );
          last_tex1 = s.materials[c.first].normal_tex;
        }

        variation_id |= (1 << gbuffer_normal_map_var);
      }

      if( s.materials[c.first].specular_tex )
      {
        if( last_tex2 != s.materials[c.first].specular_tex )
        {
          glActiveTexture( GL_TEXTURE2 );
          glBindTexture( GL_TEXTURE_2D, s.materials[c.first].specular_tex );
          last_tex2 = s.materials[c.first].specular_tex;
        }

        variation_id |= (1 << gbuffer_roughness_map_var);
      }

      if( s.materials[c.first].is_animated )
      {
        variation_id |= (1 << gbuffer_animated_var);
      }

      GLuint shader = svarman.get_varation(gbuffer_variations, variation_id);

      if( last_shader != shader )
      {
        glUseProgram( shader );

        last_shader = shader;
      }

      mat4 mv_trans = view_mat * s.meshes[c.first].transformation;
      mat4 mvp_trans = proj * mv_trans;
      glUniformMatrix4fv( uman.get_uniform( shader, "mvp" ), 1, false, &mvp_trans[0][0] );
      mat4 norm_trans = mv_trans;
      glUniformMatrix4fv( uman.get_uniform( shader, "normal_mat" ), 1, false, &norm_trans[0][0] );

      //these only need to be passed per shader
      if( std::find( used_shaders.begin(), used_shaders.end(), shader ) == used_shaders.end() )
      {
        if( s.materials[c.first].is_animated )
        {
          glBindBufferBase( GL_UNIFORM_BUFFER, uman.get_block_index( shader, "bone_data" ), bone_data_ubo );
        }
        used_shaders.push_back(shader);
      }

      s.meshes[c.first].render();
    }

    glDisable( GL_DEPTH_TEST );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    /**/

    //-----------------------------
    //render the shadows
    //-----------------------------

    /**/

    glEnable( GL_DEPTH_TEST );
    glDepthMask( GL_TRUE );

    //shadow tex only
    //glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );

    glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

    for( auto& c : spot_data )
    {
      glBindFramebuffer( GL_FRAMEBUFFER, sman.get_shadow_fbo( c.index ) );
      glViewport( 0, 0, sizes[c.index.x], sizes[c.index.x] );
      //glClear( GL_DEPTH_BUFFER_BIT );
      glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT );

      static vector<unsigned> culled_shadow_objs;
      culled_shadow_objs.clear();

      o->get_culled_objects( culled_shadow_objs, s.spot_lights[c.index.w].bv );

      static vector<pair<unsigned, u32> > shadow_draw_calls;
      shadow_draw_calls.clear();

      for( auto& d : culled_shadow_objs )
      {
        if( s.spot_lights[c.index.w].bv->is_intersecting( s.meshes[d].trans_bv ) && !s.materials[d].is_transparent )
        {
          shadow_draw_calls.push_back( make_pair( d, get_object_depth( d, s.spot_lights[c.index.w].cam ) ) );
        }
      }

      //sort draw calls front to back
      std::sort( shadow_draw_calls.begin(), shadow_draw_calls.end(), [&]( pair<unsigned, u32> a, pair<unsigned, u32> b ) -> bool
      {
        return a.second < b.second;
      } );

      for( auto& d : shadow_draw_calls )
      {
        if( last_tex0 != s.materials[d.first].diffuse_tex )
        {
          glActiveTexture( GL_TEXTURE0 );
          glBindTexture( GL_TEXTURE_2D, s.materials[d.first].diffuse_tex );
          last_tex0 = s.materials[d.first].diffuse_tex;
        }

        u64 variation_id = 0;
        if( s.materials[d.first].is_animated )
        {
          variation_id |= (1 << spot_rsm_gen_animated_var);
        }
        else
        {
          variation_id |= (1 << spot_rsm_gen_normal_var);
        }

        GLuint shader = svarman.get_varation( spot_rsm_gen_variations, variation_id );

        if( last_shader != shader )
        {
          glUseProgram( shader );

          last_shader = shader;
        }

        mat4 light_mv = s.spot_lights[c.index.w].cam.get_matrix() * s.meshes[d.first].transformation;
        mat4 light_mvp = s.spot_lights[c.index.w].the_frame.projection_matrix *  light_mv;
        mat4 light_normal_mat = light_mv;
        c.spot_shadow_mat = bias_matrix * light_mvp * inv_view_mat;

        glUniformMatrix4fv( uman.get_uniform( shader, "mvp" ), 1, false, &light_mvp[0][0] );
        glUniformMatrix4fv( uman.get_uniform( shader, "normal_mat" ), 1, false, &light_normal_mat[0][0] );

        if( s.materials[d.first].is_animated )
        {
          glBindBufferBase( GL_UNIFORM_BUFFER, uman.get_block_index( shader, "bone_data" ), bone_data_ubo );
        }

        s.meshes[d.first].render();
      }
    }

    glViewport( 0, 0, screen.x, screen.y );
    glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
    glDisable( GL_DEPTH_TEST );

    //upload spot data, now populated w/ shadow matrices
    if( spot_data.size() > 0 )
    {
      glBindBuffer( GL_UNIFORM_BUFFER, spot_light_data_ubo );
      glBufferSubData( GL_UNIFORM_BUFFER, 0, spot_data.size() * sizeof( spot_light_data ), &spot_data[0] );
    }

    /**/

    //-----------------------------
    //calculate the VPLs
    //-----------------------------

    vec4 far_plane0 = vec4( s.f.far_ll.xyz, s.f.far_ur.x );
    vec2 far_plane1 = vec2( s.f.far_ur.yz );

    /**

    glUseProgram( vpl_gen_shader );

    sman.bind_shadow_textures( 0 );
    sman.bind_rsm_textures( 4 );

    glBindImageTexture( 0, importance_128_depth_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F );
    glBindImageTexture( 1, importance_128_rsm_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8 );

    int light_idx = 0;
    mat4 inv_light_mvp = inverse(s.spot_lights[spot_data[light_idx].index.w].cam.get_frame()->projection_matrix * s.spot_lights[spot_data[light_idx].index.w].cam.get_camera_matrix( false ));

    glUniform1i( uman.get_uniform( vpl_gen_shader, "light_idx" ), light_idx );
    glUniformMatrix4fv( uman.get_uniform( vpl_gen_shader, "inv_light_mvp" ), 1, false, &inv_light_mvp[0][0] );
    glUniformMatrix4fv( uman.get_uniform( vpl_gen_shader, "mv" ), 1, false, &mv[0][0] );
    glUniform1f( uman.get_uniform( vpl_gen_shader, "far" ), -s.far );

    glBindBufferBase( GL_UNIFORM_BUFFER, uman.get_block_index( vpl_gen_shader, "spot_light_data" ), spot_light_data_ubo );

    {
      vec2 gws, lws, dispatch_size;
      uvec2 screen = uvec2( 2048 );
      set_workgroup_size( gws, lws, dispatch_size, screen );
      glDispatchCompute( dispatch_size.x, dispatch_size.y, 1 );
    }

    glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
    /**/

    //-----------------------------
    //render the ssao
    //-----------------------------

    /**/
    glDisable( GL_DEPTH_TEST );

    //downsample the depth texture
    glViewport( 0, 0, screen.x/2, screen.y/2 );
    //glViewport( 0, 0, screen.x, screen.y );

    glUseProgram( downsample_shader );

    glBindFramebuffer( GL_FRAMEBUFFER, downsample_fbo );
    //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, depth_texture );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, normal_texture );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glActiveTexture( GL_TEXTURE0 );

    glBindVertexArray( ss_quad );
    glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );

    glBindTexture( GL_TEXTURE_2D, normal_texture );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    //render ssao
    glBindFramebuffer( GL_FRAMEBUFFER, ssao_fbo );
    //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glUseProgram( ssao_shader );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, half_depth_texture );
    //glBindTexture( GL_TEXTURE_2D, depth_texture );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, half_normal_texture );
    //glBindTexture( GL_TEXTURE_2D, normal_texture );
    glActiveTexture( GL_TEXTURE2 );
    glBindTexture( GL_TEXTURE_2D, random_texture );
    glActiveTexture( GL_TEXTURE0 );

    mat4 inv_view = inv_view_mat;
    glUniformMatrix4fv( uman.get_uniform( ssao_shader, "inv_view" ), 1, GL_FALSE, &inv_view[0][0] );
    glUniformMatrix4fv( uman.get_uniform( ssao_shader, "inv_mv" ), 1, GL_FALSE, &inv_view_mat[0][0] );
    glUniformMatrix4fv( uman.get_uniform( ssao_shader, "inv_mvp" ), 1, GL_FALSE, &inv_viewproj_mat[0][0] );
    glUniform1f( uman.get_uniform( ssao_shader, "near" ), -s.near );
    glUniform1f( uman.get_uniform( ssao_shader, "far" ), -s.far );
    glUniform4fv( uman.get_uniform( ssao_shader, "far_plane0" ), 1, &far_plane0.x );
    glUniform2fv( uman.get_uniform( ssao_shader, "far_plane1" ), 1, &far_plane1.x );

    glBindVertexArray( ss_quad );
    glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );

    //blur the ssao horizontally
    glBindFramebuffer( GL_FRAMEBUFFER, ssao_blur_fbo );
    //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glUseProgram( ssao_blur_shader );

    glUniform1f( uman.get_uniform( ssao_blur_shader, "threshold" ), 0.1f );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, ssao_tex );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, half_depth_texture );
    //glBindTexture( GL_TEXTURE_2D, depth_texture );
    glActiveTexture( GL_TEXTURE0 );

    glUniform2f( uman.get_uniform( ssao_blur_shader, "direction" ), 1, 0 );

    glBindVertexArray( ss_quad );
    glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );

    //blur the ssao vertically
    glBindFramebuffer( GL_FRAMEBUFFER, ssao_fbo );
    //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, ssao_blur_tex );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, half_depth_texture );
    //glBindTexture( GL_TEXTURE_2D, depth_texture );
    glActiveTexture( GL_TEXTURE0 );

    glUniform2f( uman.get_uniform( ssao_blur_shader, "direction" ), 0, 1 );

    glBindVertexArray( ss_quad );
    glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );

    glEnable( GL_DEPTH_TEST );
    /**/

    glViewport( 0, 0, screen.x, screen.y );

    //-----------------------------
    //render the lights
    //-----------------------------

    /**/

    //cull lights
    glUseProgram( light_cull_shader );

    glBindImageTexture( 0, light_cull_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, depth_texture );

    glUniform2f( uman.get_uniform( light_cull_shader, "nearfar" ), -s.near, -s.far );
    glUniform1i( uman.get_uniform( light_cull_shader, "num_lights" ), culled_spot_lights.size() );
    vec4 tmp_far_plane0 = vec4( s.f.far_ll.xyz, s.f.far_ur.x );
    vec2 tmp_far_plane1 = vec2( s.f.far_ur.yz );
    glUniform4fv( uman.get_uniform( light_cull_shader, "far_plane0" ), 1, &tmp_far_plane0.x );
    glUniform2fv( uman.get_uniform( light_cull_shader, "far_plane1" ), 1, &tmp_far_plane1.x );
    glUniformMatrix4fv( uman.get_uniform( light_cull_shader, "proj_mat" ), 1, false, &proj[0][0] );

    glBindBufferBase( GL_UNIFORM_BUFFER, uman.get_block_index( light_cull_shader, "spot_light_data" ), spot_light_data_ubo );

    glDispatchCompute( dispatch_size.x, dispatch_size.y, 1 );

    glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
    /**/

    /**
    int size = 1024 * dispatch_size.x * dispatch_size.y;
    vector<unsigned> data(size);
    memset( &data[0], 0, size * sizeof(unsigned) );
    frm.get_opengl_error();
    glGetBufferSubData( GL_TEXTURE_BUFFER, 0, size * sizeof(unsigned), &data[0] );
    frm.get_opengl_error();

    for( auto& c : data )
    {
      if( c > 1 )
        int a = 0;
    }
    /**/

    /**/

    //ligh the opaque geometry
    glUseProgram( lighting_shader );

    glBindImageTexture( 0, result_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, depth_texture );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, normal_texture );
    glActiveTexture( GL_TEXTURE2 );
    glBindTexture( GL_TEXTURE_2D, albedo_texture );
    glActiveTexture( GL_TEXTURE3 );
    glBindTexture( GL_TEXTURE_BUFFER, light_cull_texture );
    sman.bind_shadow_textures( 4 );
    glActiveTexture( GL_TEXTURE0+8 );
    glBindTexture( GL_TEXTURE_2D, ssao_tex );
    glActiveTexture( GL_TEXTURE0 );

    glUniform2f( uman.get_uniform( lighting_shader, "nearfar" ), -s.near, -s.far );
    glUniform4fv( uman.get_uniform( lighting_shader, "far_plane0" ), 1, &tmp_far_plane0.x );
    glUniform2fv( uman.get_uniform( lighting_shader, "far_plane1" ), 1, &tmp_far_plane1.x );
    glUniformMatrix4fv( uman.get_uniform( lighting_shader, "proj_mat" ), 1, false, &proj[0][0] );

    glBindBufferBase( GL_UNIFORM_BUFFER, uman.get_block_index( lighting_shader, "spot_light_data" ), spot_light_data_ubo );

    glDispatchCompute( dispatch_size.x, dispatch_size.y, 1 );

    glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );

    /**/

    //-----------------------------
    //render forward lighting pass for translucent geometry
    //-----------------------------

    /**/

    glViewport( 0, 0, screen.x, screen.y );

    glEnable( GL_DEPTH_TEST );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glDepthMask( GL_FALSE );

    glBindFramebuffer( GL_FRAMEBUFFER, forward_lighting_fbo );

    glActiveTexture( GL_TEXTURE3 );
    glBindTexture( GL_TEXTURE_BUFFER, light_cull_texture );
    sman.bind_shadow_textures( 4 );
    glActiveTexture( GL_TEXTURE0 );

    draw_calls.clear();

    for( auto& c : culled_objs )
    {
      if( s.meshes[c].trans_bv->is_intersecting( &f ) && s.materials[c].is_transparent )
      {
        //correct depth sorting is more important for transparent objects!
        unsigned depth_offset = 32;
        unsigned material_offset = 0;
        dcu adcu;
        adcu.a = (u64)0 |
        (s.materials[c].normal_tex ? (u64(1) << (forward_lighting_normal_map_var+material_offset)) : 0) |
        (s.materials[c].specular_tex ? (u64(1) << (forward_lighting_roughness_map_var+material_offset)) : 0) |
        (s.materials[c].diffuse_tex ? (u64(1) << (forward_lighting_diffuse_var+material_offset)) : 0) |
        (s.materials[c].is_animated ? (u64(1) << (forward_animated_var + material_offset)) : 0) |
        ((u64)(get_object_depth(c, s.cam)) << depth_offset);

        draw_calls.push_back( make_pair(c, adcu ) );
      }
    }

    //sort draw calls back to front
    std::sort( draw_calls.begin(), draw_calls.end(), [&]( pair<unsigned, dcu> a, pair<unsigned, dcu> b ) -> bool
    {
      return a.second.a > b.second.a;
    } );

    for( auto& c : draw_calls )
    {
      u64 variation_id = 0;

      if( s.materials[c.first].diffuse_tex )
      {
        if( last_tex0 != s.materials[c.first].diffuse_tex )
        {
          glActiveTexture( GL_TEXTURE0 );
          glBindTexture( GL_TEXTURE_2D, s.materials[c.first].diffuse_tex );
          last_tex0 = s.materials[c.first].diffuse_tex;
        }

        variation_id |= (1 << forward_lighting_diffuse_var);
      }

      if( s.materials[c.first].normal_tex )
      {
        if( last_tex1 != s.materials[c.first].normal_tex )
        {
          glActiveTexture( GL_TEXTURE1 );
          glBindTexture( GL_TEXTURE_2D, s.materials[c.first].normal_tex );
          last_tex1 = s.materials[c.first].normal_tex;
        }

        variation_id |= (1 << forward_lighting_normal_map_var);
      }

      if( s.materials[c.first].specular_tex )
      {
        if( last_tex2 != s.materials[c.first].specular_tex )
        {
          glActiveTexture( GL_TEXTURE2 );
          glBindTexture( GL_TEXTURE_2D, s.materials[c.first].specular_tex );
          last_tex2 = s.materials[c.first].specular_tex;
        }

        variation_id |= (1 << forward_lighting_roughness_map_var);
      }

      if( s.materials[c.first].is_animated )
      {
        variation_id |= (1 << forward_animated_var);
      }

      GLuint shader = svarman.get_varation(forward_lighting_variations, variation_id);

      if( last_shader != shader )
      {
        glUseProgram( shader );

        last_shader = shader;
      }

      //these only need to be passed per shader
      if( std::find( used_shaders.begin(), used_shaders.end(), shader ) == used_shaders.end() )
      {
        glUniformMatrix4fv( uman.get_uniform( shader, "mvp" ), 1, false, &viewproj_mat[0][0] );
        glUniformMatrix4fv( uman.get_uniform( shader, "mv" ), 1, false, &view_mat[0][0] );
        glUniformMatrix4fv( uman.get_uniform( shader, "normal_mat" ), 1, false, &normal_mat[0][0] );
        glBindBufferBase( GL_UNIFORM_BUFFER, uman.get_block_index( shader, "spot_light_data" ), spot_light_data_ubo );
        used_shaders.push_back(shader);
        if( s.materials[c.first].is_animated )
        {
          glBindBufferBase( GL_UNIFORM_BUFFER, uman.get_block_index( shader, "bone_data" ), bone_data_ubo );
        }
      }

      s.meshes[c.first].render();
    }

    glDisable( GL_BLEND );
    glDisable( GL_DEPTH_TEST );
    glDepthMask( GL_TRUE );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    /**/

    //--------------------------------------
    //render the results
    //--------------------------------------

    //display the results
    /**/
    glDisable( GL_DEPTH_TEST );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glViewport( 0, 0, screen.x, screen.y );

    glUseProgram( display_shader );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, result_texture );
    //glBindTexture( GL_TEXTURE_2D, importance_128_rsm_texture );
    //glBindTexture( GL_TEXTURE_2D, ssao_tex );
    //glBindTexture( GL_TEXTURE_2D, depth_texture );
    //glBindTexture( GL_TEXTURE_2D, normal_texture );
    //glBindTexture( GL_TEXTURE_2D, albedo_texture );
    //sman.bind_shadow_textures(1);
    //sman.bind_rsm_textures(1);
    //glBindTexture( GL_TEXTURE_CUBE_MAP_ARRAY, point_shadow_translucent_texture );

    //fraps can't handle SRGB, so when taking fraps footage, switch to pow(x, 2.2) version
    glEnable( GL_FRAMEBUFFER_SRGB );

    glBindVertexArray( ss_quad );
    glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );

    glDisable( GL_FRAMEBUFFER_SRGB );
    /**/

    //-----------------------------
    //render the debug stuff
    //-----------------------------

    player_aabb = aabb( s.cam.pos, player_aabb.get_extents() );

    //ddman.CreateAABoxPosEdges( player_aabb.pos, player_aabb.extents, 0 );

    //ddman.CreateAABoxPosEdges( static_cast<aabb*>(s.meshes[s.objects[1].mesh_idx[0]].trans_bv)->pos, static_cast<aabb*>(s.meshes[s.objects[1].mesh_idx[0]].trans_bv)->extents, 0 );

    for( int c = 0; c < s.objects[0].mesh_idx.size(); ++c )
    {
      if( find(collision_ignore_list.begin(), collision_ignore_list.end(), s.objects[0].mesh_idx[c] ) == collision_ignore_list.end() )
        collide_aabb_aabb( *static_cast<aabb*>(s.meshes[s.objects[0].mesh_idx[c]].trans_bv), player_aabb );
    }

    s.cam.pos = player_aabb.get_pos();

    s.spot_lights[0].cam = s.cam;
    s.spot_lights[0].cam.pos = s.cam.pos + vec3( 1.5f, 0, 0 );
    *static_cast<sphere*>(s.spot_lights[0].bv) = sphere( s.spot_lights[0].cam.pos, static_cast<sphere*>(s.spot_lights[0].bv)->get_radius() );

    if( seconds > 0.01667 )
    {
      for( int c = 1; c < s.objects.size(); ++c )
      {
        static vector<int> path;
        path.clear();

        vec3 pos = s.objects[c].transformation[3].xyz;
        vec3 dir;

        if( length(s.cam.pos - pos) < 20 )
        {
          player_health -= 0.1f;
          continue;
        }

        int player_cell = get_astar_cell_from_pos(s.cam.pos);
        int zombie_cell = get_astar_cell_from_pos(pos);

        int px, py, zx, zy;
        pos_to_xy(player_cell, px, py);
        pos_to_xy(zombie_cell, zx, zy);

        vector<int> neigh;
        get_neighbours(zx, zy, neigh);

        /*for( auto& c : neigh )
        {
          ddman.CreateAABoxPosEdges( astar_grid[c].bv.get_pos(), astar_grid[c].bv.get_extents(), 0 );
        }*/

        if( !is_astar_cell_walkable(zx, zy) )
        {
          vector<int> neighbours;
          vector<pair< int, float > > dist;
          get_neighbours(zx, zy, neighbours);

          if( neighbours.size() > 0 )
          {
            dist.resize(neighbours.size());
            for( int c = 0; c < neighbours.size(); ++c )
            {
              dist[c] = make_pair(neighbours[c], length(astar_grid[neighbours[c]].bv.get_pos() - pos));
            }

            std::sort( dist.begin(), dist.end(), [](pair<int, float> a, pair<int, float> b)
            {
              return a.second < b.second;
            });

            zombie_cell = dist[0].first;
            pos_to_xy(zombie_cell, zx, zy);
          }
        }

        bool res = find_astar_path( zx, zy, px, py, path );

        if( res )
        {
          /*for( auto& c : path )
          {
            ddman.CreateCross( astar_grid[c].bv.get_pos(), 2, 0 );
          }*/
        }
        else
        {
          cout << "path not found" << endl;
          continue;
        }

        get_catmull_rom_advancement( pos, dir, path );
        pos += dir * 0.5;
        mat4 rot = create_rotation( get_angle( vec3(0, 0, 1), dir ) * sign(dot(dir, vec3(1, 0, 0))), vec3( 0, 1, 0 ) );
        s.objects[c].transformation = create_translation(pos) * create_scale(vec3(10)) * rot * create_rotation(radians(-90), vec3( 1, 0, 0 ) );
      }

      update_transformations( s, o );

      js::set_player_health(b, player_health);
    }

    vec3 head_pos = vec3( 0 );
    vec3 head_extent = vec3( 1 );
    head_pos = (create_translation(vec3( 0, 18, 0 )) * s.objects[1].transformation * vec4( head_pos, 1 )).xyz;

    /*ddman.CreateAABoxPosEdges( head_pos, head_extent, 0 );

    for( auto& c : astar_grid )
    {
      if( c.walkable )
      {
        ddman.CreateAABoxMinMax( c.bv.min, c.bv.max, 0 );
      }
    }*/

    /**/

    glUseProgram( debug_shader );

    glUniformMatrix4fv( 0, 1, false, &viewproj_mat[0][0] );

    ddman.DrawAndUpdate(16);

    /**/

    //-----------------------------
    //render the UI
    //-----------------------------

    /**/

    glDisable( GL_DEPTH_TEST );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glUseProgram( browser_shader );

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, b.browser_texture );

    glBindVertexArray( ss_quad );
    glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );

    glDisable( GL_BLEND );
    glEnable( GL_DEPTH_TEST );

    /**/

    static stringstream ss;
    ss.str( "" );

    ss << "Num culled lights: " << culled_spot_lights.size();

    frm.set_title(ss.str().c_str());

    frm.get_opengl_error();
  }, silent );

  return 0;
}
