#version 460 core

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (rgba32f, binding = 0) uniform image2D img;

uniform float time;

struct Ray {
    vec3 origin, dir;
};

struct Camera {
    vec3 center;
    float fov;
};

struct Viewport {
    float width, height;
    vec3 u, v, du, dv, upper_left, center;
};

struct Sphere {
    vec4 color;
    vec3 center;
    float radius;
};

float get_random(vec2 seed) {
    return fract(sin(dot(seed.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

Ray get_sample_ray(int seed, Viewport vp, ivec2 pixel_coords) {
    vec3 offset =  
        vec3(
            get_random(time * pixel_coords * seed) - 0.5, 
            get_random(time * pixel_coords * seed) - 0.5, 
            0.0
        );
        vec3 sample_pixel = vp.center + ((pixel_coords.x + offset.x) * vp.dv)
                                        + ((pixel_coords.y + offset.y) * vp.dv);

        Ray sample_ray;
        sample_ray.origin = vp.center;
        sample_ray.dir = sample_pixel - sample_ray.origin;

        return sample_ray;
}

//returns t value (intersection scalar)
float get_sphere_hit(Ray ray, Sphere sphere) {
    vec3 offset = sphere.center - ray.origin;
    float a = dot(ray.dir, ray.dir);
    float b = 2.0 * dot(ray.dir, offset);
    float c = dot(offset, offset) - (sphere.radius * sphere.radius);
    float discriminant = (b * b) - (4 * a * c);

    if (discriminant < 0) {
        return -1;
    }
    else if (discriminant == 0) {

    }
}

void main() {

    Camera camera;
    camera.center = vec3(0.0, 0.0, 0.0);

    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img);

    Viewport vp;
    
    vp.height = 2.0;
    vp.width = vp.height * (float(size.x) / size.y);

    vp.u = vec3(vp.width, 0, 0);
    vp.v = vec3(0, -vp.height, 0);

    vp.du = vp.u / size.x;
    vp.dv = vp.v / size.y;
    vp.upper_left = camera.center - vec3(0, 0, 1.0) - vp.u/2 - vp.v/2;
    vp.center = vp.upper_left + 0.5 * (vp.du + vp.dv);    

    Sphere sphere;
    sphere.color = vec4(0.0, 1.0, 1.0, 1.0);
    sphere.center = vec3(0.0, 0.0, -1.0);
    sphere.radius = 0.5;

    Sphere ground;
    ground.color = vec4(0.0, 1.0, 0.0, 1.0);
    ground.center = vec3(0.0, 100.5, -1.0);
    ground.radius = 100;

    int samples = 100;

    vec4 pixel_color = vec4(0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < samples; i++) {
        Ray sample_ray = get_sample_ray(i * 100, vp, pixel_coords);
    }

    pixel_color = pixel_color / samples;

    imageStore(img, pixel_coords, pixel_color);
}
