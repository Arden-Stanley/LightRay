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
    float b = dot(ray.dir, difference);
    float c = dot(difference, difference) - (sphere.radius * sphere.radius);
    float discriminant = (b * b) - (a * c);
    if (discriminant < 0)
    {
        return vec4(0.2, 0.2, 0.2, 1.0);
    }
    float root = (b - sqrt(discriminant)) / a;
    if (root <= 1.0 || root >= 100000)
    {
        root = (b + sqrt(discriminant)) / a;
        if (root <= 1.0 || root >= 100000) 
        {
            return vec4(0.2, 0.2, 0.2, 1.0);
        }
    }

    return sphere.color;
}

vec4 check_hit_sphere_normal(Sphere sphere, Ray ray)
{
    vec3 difference = sphere.center - ray.origin;
    float a = dot(ray.dir, ray.dir);
    float b = dot(ray.dir, difference);
    float c = dot(difference, difference) - (sphere.radius * sphere.radius);
    float discriminant = (b * b) - (a * c);
    if (discriminant <= 0)
    {
        return vec4(0.2, 0.2, 0.2, 1.0);
    }
    float root = (b - sqrt(discriminant)) / a;
    if (root <= 0.1 || root >= 10000)
    {
        root = (b + sqrt(discriminant)) / a;
        if (root <= 0.1 || root >= 10000) 
        {
            return vec4(0.2, 0.2, 0.2, 1.0);
        }
    }

    float t = root;
    if (t == 0)
    {
	return vec4(0.2, 0.2, 0.2, 1.0);
    }
    vec3 point = root * ray.dir;
    vec3 normal = (point - sphere.center) / sphere.radius;
    return vec4(normal, 1.0);
}

void main() 
{

    Camera camera;
    camera.center = vec3(0.0, 0.0, 0.0);

	ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img);
    
    float viewport_height = 2.0;
    float viewport_width = viewport_height * (float(size.x) / size.y);

    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    vec3 pixel_delta_u = viewport_u / size.x;
    vec3 pixel_delta_v = viewport_v / size.y;
 
    vec3 viewport_upper_left = camera.center - vec3(0, 0, 1.0) - viewport_u/2 - viewport_v/2;
    vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    //vec3 pixel_center = pixel00_loc + (pixel_coords.x * pixel_delta_u) + (pixel_coords.y * pixel_delta_v);
    


    Sphere sphere;
    sphere.color = vec4(0.0, 1.0, 1.0, 1.0);
    sphere.center = vec3(0.0, 0.0, -1.0);
    sphere.radius = 0.5;

    Sphere ground;
    ground.color = vec4(0.0, 1.0, 0.0, 1.0);
    ground.center = vec3(0.0, 100.5, -1.0);
    ground.radius = 100;

    int samples = 100;

    vec4 pixel = vec4(0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < samples; i++)
    {
        vec3 offset = 
        vec3(
            get_random(time * pixel_coords * 300 * i) - 0.5, 
            get_random(time * pixel_coords * 200 * i) - 0.5, 
            0.0
        );
        vec3 sample_pixel = pixel00_loc + ((pixel_coords.x + offset.x) * pixel_delta_u)
                                        + ((pixel_coords.y + offset.y) * pixel_delta_v);

        Ray sample_ray;
        sample_ray.origin = camera.center;
        sample_ray.dir = sample_pixel - sample_ray.origin;

        //pixel = pixel + check_hit_sphere(ground, sample_ray);

        vec4 normal = check_hit_sphere_normal(sphere, sample_ray);
        pixel = pixel + 0.5 * vec4(normal.x + 1, normal.y + 1, normal.z + 1, 2.0);
        //pixel = pixel + check_hit_sphere();
    }

    pixel = pixel / samples;

    imageStore(img, pixel_coords, pixel);
}
