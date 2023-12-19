#include "Scene.h"
#include "Renderer.h"
#include "Camera.h"
#ifdef ENABLE_VISUALIZER
#include "Debug_Visualizer.h"
#endif


Scene* scene;
Renderer* renderer;
Camera* camera;

int main(){
	//Initialize camera
	camera = new Camera();
	//Generate rays
	camera->generateRays();

	//Initialize Scene
	string config = "Input data\\lucy.obj";
	scene = new Scene(config);

	//Initialize Renderer
	renderer = new Renderer();
	renderer->addScene(*scene);
	renderer->addRays(camera->rays);
	renderer->renderLoop();
	renderer->renderImage();

#ifdef ENABLE_VISUALIZER
	launch_visualizer(&(renderer->render_data));
#endif

}