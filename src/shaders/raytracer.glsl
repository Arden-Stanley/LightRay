#version 460 core

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (rgba32f, binding = 0) uniform image2D img;

struct Ray 
{
    vec3 origin;
    vec3 dir;
};

struct Camera
{
    vec3 center;
    float fov;
};

struct Sphere 
{
    vec3 center;
    float radius;
};

bool check_hit_sphere(Sphere sphere, Ray ray)
{
    vec3 difference = sphere.center - ray.origin;
    float a = dot(ray.dir, ray.dir);
    float b = -2.0 * dot(ray.dir, difference);
    float c = dot(difference, difference) - (sphere.radius * sphere.radius);
    float discriminant = (b * b) - (4 * a * c);
    if (discriminant >= 0)
    {
        return true;
    }
    else 
    {
        return false;
    }
}

void main() 
{
	ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img);
    float x = -(float(pixel_coords.x * 2 - size.x) / size.x);
    float y = -(float(pixel_coords.y * 2 - size.y) / size.y);

    Camera camera;
    camera.center = vec3(0.0, 0.0, 0.0);

    Ray ray;
    ray.origin = camera.center;
    ray.dir = (vec3(x, y, 0.0) - vec3(0.0, 0.0, 1.0)) - camera.center;

    Sphere sphere;
    sphere.center = vec3(0.0, 0.0, 5.0);
    sphere.radius = 1.0;

    vec4 pixel = vec4(0.0, 0.0, 0.0, 1.0);

    if (check_hit_sphere(sphere, ray))
    {
        pixel = vec4(1.0, 0.0, 0.0, 1.0);
    }
    else
    {
        pixel = vec4(0.0, 0.4, 0.8, 1.0);
    }

    imageStore(img, pixel_coords, pixel);
}
