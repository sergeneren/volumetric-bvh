
#include "bvh_builder.h"

GPU_VDB gvdb;
std::vector<GPU_VDB *> vdbs;
BVH_Builder bvh_builder;

static int num_volumes = 10;

int main(int argc, char** argv) {


	cuInit(0);
	
	gvdb.loadVDB("dragon.vdb", "density");

	for (int i = 0; i < num_volumes; ++i) {

		mat4 xform = gvdb.get_xform();
		xform.translate(make_float3(0, 0, 4 * i));

		vdbs.push_back(new GPU_VDB(gvdb));
		vdbs.at(i)->set_xform(xform);
	}


	AABB sceneBounds(make_float3(FLT_MIN), make_float3(FLT_MAX));

	GPU_VDB *volume_pointers = new GPU_VDB[num_volumes];
	for (int i = 0; i < num_volumes; ++i) {

		volume_pointers[i] = *vdbs.at(i);
	}

	bvh_builder.init();
	bvh_builder.build_bvh(volume_pointers, vdbs.size(), sceneBounds);


	return 0;
}