#version 460 core

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (rgba32f, binding = 0) uniform image2D img;

uniform float time;

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
    vec4 color;
    vec3 center;
    float radius;
};

float get_random(vec2 seed)
{
    return fract(sin(dot(seed.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec4 check_hit_sphere(Sphere sphere, Ray ray)
{
    vec3 difference = sphere.center - ray.origin;
    float a = dot(ray.dir, ray.dir);
    float b = -2.0 * dot(ray.dir, difference);
    float c = dot(difference, difference) - (sphere.radius * sphere.radius);
    float discriminant = (b * b) - (4 * a * c);
    if (discriminant >= 0)
    {
        return sphere.color;
    }
    else 
    {
        return vec4(0.2, 0.2, 0.2, 1.0);
    }
}

void main() 
{

    Camera camera;
    camera.center = vec3(0.0, 0.0, tan(90.0 / 2.0));

	ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img);
    float x = -(float(pixel_coords.x * 2 - size.x) / size.x);
    float y = -(float(pixel_coords.y * 2 - size.y) / size.y);
    
    /*float viewport_height = 2.0;
    float viewport_width = viewport_height * (float(size.x) / size.y);

    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    vec3 pixel_delta_u = viewport_u / size.x;
    vec3 pixel_delta_v = viewport_v / size.y;
 
    vec3 viewport_upper_left = camera.center - vec3(0, 0, 1.0) - viewport_u/2 - viewport_v/2;
    vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    vec3 pixel_center = pixel00_loc + (pixel_coords.x * pixel_delta_u) + (pixel_coords.y * pixel_delta_v);
    */


    Sphere sphere;
    sphere.color = vec4(0.0, 1.0, 1.0, 1.0);
    sphere.center = vec3(0.0, 0.0, -1.0);
    sphere.radius = 1.0;

    int samples = 1000;

    vec4 pixel = vec4(0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < samples; i++)
    {
        vec3 offset = 
        vec3(
            get_random(time * vec2(x, y) * 30 * i) - 0.5, 
            get_random(time * vec2(x, y) * 200 * i) - 0.5, 
            0.0
        );
        vec3 sample_pixel = vec3(x, y, 0.0) + (offset * vec3(2.0 / size.x, 2.0 / size.y, 0.0));

        Ray sample_ray;
        sample_ray.origin = camera.center;
        sample_ray.dir = sample_pixel - sample_ray.origin;
        pixel = pixel + check_hit_sphere(sphere, sample_ray);
    }

    pixel = pixel / samples;

    imageStore(img, pixel_coords, pixel);
}
