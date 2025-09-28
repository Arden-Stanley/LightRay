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

    Camera camera;
    camera.center = vec3(0.0, 0.0, 0.0);

	ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img);
    int viewport_width = 2.0;
    int viewport_height = viewport_height * (double (size.x) / size.y);

    int viewport_u = vec3(viewport_width, 0, 0);
    int viewport_v = vec3(0, -viewport_height, 0);

    int pixel_delta_u = viewport_u / imageSize.x;
    int pixel_delta_v = viewport_v / imageSize.y;

    int viewport_upper_left = camera.center - vec3(0, 0, 1.0) - viewport_u/2 - viewport_v/2;
    int pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    int pixel_center = pixel00_loc + (pixel_coords.x * pixel_delta_u) + (pixel_coords.y * pixel_delta_v);

    Ray ray;
    ray.origin = camera.center;
    ray.dir = pixel_center - camera.center;

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
