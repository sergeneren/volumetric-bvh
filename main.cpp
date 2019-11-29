
#include "bvh_builder.h"

GPU_VDB gvdb;
std::vector<GPU_VDB *> vdbs;
BVH_Builder bvh_builder;

static int num_volumes = 10;

int main(int argc, char** argv) {


	cuInit(0);
	
	gvdb.loadVDB("dragon.vdb", "density");

	vdbs.push_back(new GPU_VDB(gvdb));
	mat4 xform = vdbs.at(0)->get_xform();

	for (int i = 1; i < num_volumes; ++i) {

		vdbs.push_back(new GPU_VDB(gvdb));
		xform.translate(make_float3(2, 0, 0));
		vdbs.at(i)->set_xform(xform);

	}
	CUcontext ctx;
	cuCtxCreate_v2(&ctx, 0, 0);

	bvh_builder.init();

	return 0;
}