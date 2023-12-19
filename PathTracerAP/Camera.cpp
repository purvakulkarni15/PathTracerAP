#include "Camera.h"

void Camera::generateRays()
{
    glm::vec3 camera_orig = glm::vec3(0, 0, 1020.0);
    rays.resize(RESOLUTION_X * RESOLUTION_Y);

    float world_x = -10.0;
    float step_x = 20.0 / RESOLUTION_X;
    float world_y = -4;
    float step_y = 16.0 / RESOLUTION_Y;
    float world_z = 1000.0;


    for (int y = 0; y < RESOLUTION_Y; ++y)
    {
        world_x = -10.0;
        for (int x = 0; x < RESOLUTION_X; ++x)
        {
            int index = y * RESOLUTION_X + x;
            glm::vec3 pix_pos = glm::vec3(world_x, world_y, world_z);

            Camera::rays[index].points_base.orig = camera_orig;
            Camera::rays[index].points_base.end = pix_pos;
            Camera::rays[index].hit_info.t = FLOAT_MAX;
            Camera::rays[index].hit_info.color = glm::vec3(200, 200, 200);
            world_x += step_x;
        }
        world_y += step_y;
    }
}

