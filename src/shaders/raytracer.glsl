#version 460 core

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (rgba32f, binding = 0) uniform image2D img;

uniform float time;

ivec2 PIXEL_LOC = ivec2(gl_GlobalInvocationID.xy);
ivec2 SCREEN_DIMS = imageSize(img);

struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 hit;
    float t;
    vec3 normal;
};

struct Sphere {
    vec3 center;
    float radius;
};

struct Viewport {
    vec3 center;
    float focal_length;
    float width;
    float height;
    vec3 u;
    vec3 v;
    vec3 du;
    vec3 dv;
    vec3 first_pixel;
};

float get_random_f(vec2 seed) {
    return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

vec3 get_random_unit(float j) {
    vec3 p;
    float length_squared;
    float i = 10;
    while (true) {
        p = vec3(
            (get_random_f(PIXEL_LOC + vec2(41.0 + i + j, 102.0 + i + j))), 
            (get_random_f(PIXEL_LOC + vec2(67.0 + i + j, 127.0 + i + j))), 
            (get_random_f(PIXEL_LOC + vec2(420.0 + i + j, 273.0 + i + j)))
        );
        i += 10;
        length_squared = p.x * p.x + p.y * p.y + p.z * p.z;
        if (length_squared <= 1) {
            return normalize(p);
        }
    };
}

vec3 get_unit_on_hemi(Ray ray, float j) {
    vec3 on_unit_sphere = get_random_unit(j);
    if (dot(on_unit_sphere, ray.normal) > 0.0)
        return on_unit_sphere;
    else 
        return -on_unit_sphere;
}

bool get_sphere_hit(inout Ray ray, in Sphere sphere) {
    vec3 oc = sphere.center - ray.origin;
    float a = dot(ray.direction, ray.direction);
    float h = dot(ray.direction, oc);
    float c = dot(oc, oc) - (sphere.radius * sphere.radius);

    float d = h * h - a * c;

    if (d < 0) {
        return false;
    }
    float sqrtd = sqrt(d);
    float root = (h - sqrtd) / a;
    if (root <= 0.0) {
        root = (h + sqrtd) / a;
        if (root <= 0.0) {
            return false;
        }
    }
    ray.t = d;
    ray.hit = ray.origin + d * ray.direction;
    ray.normal = (ray.hit - sphere.center) / sphere.radius;
    return true;    
}

void main() {
    Viewport viewport;
    viewport.center = vec3(0, 0, 0);
    viewport.focal_length = 1.0;
    viewport.height = 2.0;
    viewport.width = viewport.height * (float(SCREEN_DIMS.x) / SCREEN_DIMS.y);
    viewport.u = vec3(viewport.width, 0, 0);
    viewport.v = vec3(0, -viewport.height, 0);
    viewport.du = viewport.u / SCREEN_DIMS.x;
    viewport.dv = viewport.v / SCREEN_DIMS.y; 
    viewport.first_pixel = viewport.center - vec3(0, 0, viewport.focal_length)
         - (viewport.u / 2) - (viewport.v / 2)
         + 0.5 * (viewport.du + viewport.dv);


    Sphere sphere1;
    sphere1.center = vec3(0.0, 0.0, -3.0);
    sphere1.radius = 1.0;

    Sphere sphere2;
    sphere2.center = vec3(0.0, 100.0, -3.0);
    sphere2.radius = 100.0;


    vec3 target_pixel =
        viewport.first_pixel +
        (PIXEL_LOC.x * viewport.du) +
        (PIXEL_LOC.y * viewport.dv);
    
    Ray ray;
    ray.origin = viewport.center;
    ray.direction = normalize(target_pixel - viewport.center);
    
    vec3 unit_direction = normalize(ray.direction);
    float a = 0.5 * (unit_direction.y + 1.0);
    vec3 gradient = (1.0-a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
    vec4 pixel_color = vec4(gradient, 1.0);
    

    //vec4 pixel_color = vec4(get_random_unit(), 1.0);
    for (int i = 0; i < 10; i++) {
        if (get_sphere_hit(ray, sphere1)) {
            pixel_color *= 0.5;
        }
        else if (get_sphere_hit(ray, sphere2)) {
            pixel_color *= 0.5;
        }
        else {
            break;
        }
        ray.direction = get_unit_on_hemi(ray, i * 132);
        ray.origin = ray.hit;
    }

    imageStore(img, PIXEL_LOC, pixel_color);
}