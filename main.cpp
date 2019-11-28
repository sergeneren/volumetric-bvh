
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif


#include <sstream>
#include <iostream>
#include <cuda_runtime.h>


#ifndef M_PI
#define M_PI 3.14156265
#endif

void Timer(int obsolete) {
	glutPostRedisplay();
	glutTimerFunc(10, Timer, 0);
}


int main(int argc, char** argv) {







	return 0;
}