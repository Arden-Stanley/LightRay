#version 460 core

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (rgba32f, binding = 0) uniform image2D img;

uniform float time;

vec2 pixelCoords = vec2(float(gl_GlobalInvocationID.x), float(gl_GlobalInvocationID.y));


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

vec3 getUnitSphere() {
    int i = 100;
    while (true) {
        vec3 p = vec3(
            (getRandom(pixelCoords.xy * 102 * i * time) * 2.0) - 1.0, 
            (getRandom(pixelCoords.xx * 300 * i * time) * 2.0) - 1.0, 
            (getRandom(pixelCoords.yy * 827 * i * time) * 2.0) - 1.0
        );
        i += 1;
        float lengthSqrd = dot(p, p);
        if (lengthSqrd <= 1.0) {
            return (p / sqrt(lengthSqrd));
        }
    }
}

vec3 getRandomOnHemi(vec3 normal) {
    vec3 randOnSphere = getUnitSphere();
    if (dot(randOnSphere, normal) > 0.0) {
        return randOnSphere;
    }
    else {
        return -randOnSphere;
    }
}

//returns t value (intersection scalar)
float getSphereHit(Ray ray, Sphere sphere, float tMin, float tMax) {
    vec3 offset = ray.origin - sphere.center;
    float a = dot(ray.dir, ray.dir);
    float b = dot(offset, ray.dir);
    float c = dot(offset, offset) - (sphere.radius * sphere.radius);
    float discriminant = (b * b) - (a * c);

    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < tMax && temp > tMin) {
            return temp;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < tMax && temp > tMin) {
            return temp;
        }
    }
    
    return -1;
}

vec3 getSurfaceNormal(Ray ray, Sphere sphere, float t) {
    vec3 p = ray.origin + t * ray.dir;
    vec3 normal = normalize(p - sphere.center); // sphere.radius;
    return normal;
}

vec4 getPixelColor(Ray ray, Sphere sphere, Sphere ground, int rayDepth, vec4 background) {
    Ray currentRay = ray;
    vec3 normal;
    float scalar = 1.0;

    for (int i = 0; i < rayDepth; i++) {
        float tSphere = getSphereHit(currentRay, sphere, 0, 10000);
        float tGround = getSphereHit(currentRay, ground, 0, 10000);
        if (tSphere != -1) {
            scalar = scalar * 0.5;
            normal = getSurfaceNormal(currentRay, sphere, tSphere);
            currentRay.origin = currentRay.origin + (currentRay.dir * tSphere);
            currentRay.dir = getRandomOnHemi(normal);
        }
        else if (tGround != -1) {
            scalar = scalar * 0.2;
            normal = getSurfaceNormal(currentRay, ground, tGround);
            currentRay.origin = currentRay.origin + (currentRay.dir * tGround);
            currentRay.dir = getRandomOnHemi(normal);
        }
        else {
            return (background * vec4(scalar, scalar, scalar, 1.0));
        }
    }
    return (background * vec4(scalar, scalar, scalar, 1.0));
}

void main() {

    Camera camera;
    camera.center = vec3(0.0, 0.0, 0.0);

    ivec2 size = imageSize(img);

    Viewport vp;
    
    vp.height = 2.0;
    vp.width = vp.height * (float(size.x) / size.y);

    vp.u = vec3(vp.width, 0, 0);
    vp.v = vec3(0, vp.height, 0);

    vp.du = vp.u / float(size.x);
    vp.dv = vp.v / float(size.y);
    vp.upperLeft = camera.center - vec3(0, 0, 1.0) - vp.u/2 - vp.v/2;
    vp.firstPixel = vp.upperLeft + 0.5 * (vp.du + vp.dv);    

    Sphere sphere = Sphere(vec4(0.0, 1.0, 1.0, 1.0), vec3(0.0, 0.0, -2.0), 0.5);
    Sphere ground = Sphere(vec4(0.0, 1.0, 0.0, 1.0), vec3(0.0, -100.5, -2.0), 100.0);



    vec3 targetPixel = vp.firstPixel + (pixelCoords.x * vp.du) + (pixelCoords.y * vp.dv);

    Ray ray = Ray(camera.center, targetPixel - camera.center);  
    
    vec4 pixelColor = getPixelColor(ray, sphere, ground, 4, vec4(0.2, 0.4, 0.9, 1.0));

    
    ivec2 pixelLoc = ivec2(gl_GlobalInvocationID.xy);
     

    imageStore(img, pixelLoc, pixelColor);
}
