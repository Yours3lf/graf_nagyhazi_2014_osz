
uniform mat4 mvp;

layout(location=0) in vec4 in_vertex;

#ifdef GPU_SKINNING
#include "gpu_skinning.h"
#endif

void main()
{
  gl_Position = mvp *
#ifdef GPU_SKINNING
  get_bone_transform() *
#endif
  in_vertex;
}
