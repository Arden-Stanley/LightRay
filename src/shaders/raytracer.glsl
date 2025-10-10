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
    vec3 u, v, du, dv, upperLeft, firstPixel;
};

struct Sphere {
    vec4 color;
    vec3 center;
    float radius;
};

float getRandom(vec2 seed) {
    return fract(sin(dot(seed.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

//returns t value (intersection scalar)
float getSphereHit(Ray ray, Sphere sphere) {
    vec3 offset = ray.origin - sphere.center;
    float a = dot(ray.dir, ray.dir);
    float b = 2.0 * dot(offset, ray.dir);
    float c = dot(offset, offset) - (sphere.radius * sphere.radius);
    float discriminant = (b * b) - (4.0 * a * c);

    if (discriminant < 0.0) {
        return -1;
    }
    else {
        float t0 = (-b - sqrt(discriminant)) / (2.0 * a);
        float t1 = (-b + sqrt(discriminant)) / (2.0 * a);

        if (t0 > 0.001) {
            return t0;
        }
        else if (t1 > 0.001) {
            return t1;
        }
        else {
            return -1.0;
        }
    }
}

vec3 getSurfaceNormal(Ray ray, Sphere sphere) {
    float t = getSphereHit(ray, sphere);
    if (t != -1) {
        vec3 p = ray.origin + t * ray.dir;
        vec3 normal = (p - sphere.center) / sphere.radius;
        return normal;
    }
    else {
        return vec3(0, 0, 0);
    }
}



void main() {

    Camera camera;
    camera.center = vec3(0.0, 0.0, 0.0);

    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img);

    Viewport vp;
    
    vp.height = 2.0;
    vp.width = vp.height * (float(size.x) / size.y);

    vp.u = vec3(vp.width, 0, 0);
    vp.v = vec3(0, -vp.height, 0);

    vp.du = vp.u / float(size.x);
    vp.dv = vp.v / float(size.y);
    vp.upperLeft = camera.center - vec3(0, 0, 1.0) - vp.u/2 - vp.v/2;
    vp.firstPixel = vp.upperLeft + 0.5 * (vp.du + vp.dv);    

    Sphere sphere;
    sphere.color = vec4(0.0, 1.0, 1.0, 1.0);
    sphere.center = vec3(0.0, 0.0, -2.0);
    sphere.radius = 1.0;

    Sphere ground;
    ground.color = vec4(0.0, 1.0, 0.0, 1.0);
    ground.center = vec3(0.0, 100.5, -1.0);
    ground.radius = 100;

    int samples = 4;


    vec4 pixelColor = vec4(0.0, 0.0, 1 - float(pixelCoords.y) / float(size.y), 1.0);

    vec3 targetPixel = vp.firstPixel + (pixelCoords.x * vp.du) + (pixelCoords.y * vp.dv);
    /*
    for (int i = 0; i < samples; i++) {
        Ray sample_ray = get_sample_ray(i * 100, vp, pixel_coords);
        if (get_sphere_hit(sample_ray, sphere) != -1) {
            pixel_color += sphere.color;
        }
    }

    pixel_color = pixel_color / samples;
    */

    Ray ray;
    ray.origin = camera.center;
    ray.dir = targetPixel - camera.center;

    /*
    //RENDER SURFACE NORMALS AS COLOR
    vec3 normal = getSurfaceNormal(ray, sphere);
    if (normal != vec3(0, 0, 0)) {
        pixelColor = vec4(normal, 1.0);
    }
    */ 

    imageStore(img, pixelCoords, pixelColor);
}
