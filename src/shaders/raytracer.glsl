#version 460 core

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (rgba32f, binding = 0) uniform image2D img;

uniform float time;

struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 hit;
    float t;
};

struct Sphere {
    vec3 center;
    float radius;
};

struct Camera {
    vec3 center;
    float focalLength;
};

bool getSphereHit(inout Ray ray, in Sphere sphere) {
    vec3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float s = dot(ray.direction, oc);
    float c = dot(oc, oc) - (sphere.radius * sphere.radius);

    float delta = s * s - (c);
    float d = -s - sqrt(delta);

    if (delta < 0.0) {
        return false;
    }
    else {
        return true;
    }
}

void main() {
    ivec2 pixelLoc = ivec2(gl_GlobalInvocationID.xy);
    ivec2 screenDims = imageSize(img);

    vec4 pixelColor = vec4(0.2, 0.5, 0.8, 1.0);

    Camera camera;
    camera.center = vec3(0.0, 0.0, 1.0);

    Sphere sphere;
    sphere.center = vec3(0.0, 0.0, -5.0);
    sphere.radius = 1.0;

    Ray ray;
    ray.origin = vec3(0, 0, 0);
    ray.direction = vec3(float(pixelLoc.x), float(pixelLoc.y), 1.0) - camera.center;

    if (getSphereHit(ray, sphere)) {
        pixelColor = vec4(0.5, 0.0, 0.2, 1.0);
    }


    imageStore(img, pixelLoc, pixelColor);
}