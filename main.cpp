
#include "bvh_builder.h"
#include <random>

GPU_VDB gvdb;
std::vector<GPU_VDB *> vdbs;
BVH_Builder bvh_builder;

static int num_volumes = 50;

int main(int argc, char** argv) {


	cuInit(0);
	
	gvdb.loadVDB("dragon.vdb", "density");
	
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist(0, 100);

	for (int i = 0; i < num_volumes; ++i) {

		mat4 xform = gvdb.get_xform();
		xform.translate(make_float3(dist(rng), dist(rng), dist(rng)));

		vdbs.push_back(new GPU_VDB(gvdb));
		vdbs.at(i)->set_xform(xform);
	}


	AABB sceneBounds(make_float3(.0f), make_float3(.0f));

	GPU_VDB *volume_pointers = new GPU_VDB[num_volumes];
	for (int i = 0; i < num_volumes; ++i) {

		volume_pointers[i] = *vdbs.at(i);
	}


	bvh_builder.init();
	bvh_builder.m_debug_bvh = true;
	bvh_builder.build_bvh(volume_pointers, vdbs.size(), sceneBounds);


	return 0;
}