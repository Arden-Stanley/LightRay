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
    vec3 u, v, du, dv, upperLeft, center;
};

struct Sphere {
    vec4 color;
    vec3 center;
    float radius;
};

float getRandom(vec2 seed) {
    return fract(sin(dot(seed.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

Ray getSampleRay(int seed, Viewport vp, ivec2 pixelCoords) {
    vec3 offset =  
        vec3(
            getRandom(time * pixelCoords * seed) - 0.5, 
            getRandom(time * pixelCoords * seed) - 0.5, 
            0.0
        );
        vec3 samplePixel = vp.center + ((pixelCoords.x + offset.x) * vp.dv)
                                        + ((pixelCoords.y + offset.y) * vp.dv);

        Ray sampleRay;
        sampleRay.origin = vp.center;
        sampleRay.dir = samplePixel - sampleRay.origin;

        return sampleRay;
}

float getSphereHit() {
    
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

    vp.du = vp.u / size.x;
    vp.dv = vp.v / size.y;
    vp.upperLeft = camera.center - vec3(0, 0, 1.0) - vp.u/2 - vp.v/2;
    vp.center = vp.upperLeft + 0.5 * (vp.du + vp.dv);    

    Sphere sphere;
    sphere.color = vec4(0.0, 1.0, 1.0, 1.0);
    sphere.center = vec3(0.0, 0.0, -1.0);
    sphere.radius = 0.5;

    Sphere ground;
    ground.color = vec4(0.0, 1.0, 0.0, 1.0);
    ground.center = vec3(0.0, 100.5, -1.0);
    ground.radius = 100;

    int samples = 100;

    vec4 pixelColor = vec4(0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < samples; i++) {
        Ray sampleRay = getSampleRay(i * 100, vp, pixelCoords);
    }

    pixelColor = pixelColor / samples;

    imageStore(img, pixelCoords, pixelColor);
}
