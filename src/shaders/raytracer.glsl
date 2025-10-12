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

vec3 getRandomOnHemi(vec3 normal) {
    vec3 randOnSphere;
    while (true) {
        vec3 p = vec3(
            (getRandom(pixelCoords.xy * 102 * time) * 2.0) - 1.0, 
            (getRandom(pixelCoords.xy * 232 * time) * 2.0) - 1.0, 
            (getRandom(pixelCoords.xy * 827 * time) * 2.0) - 1.0
        );
        float lengthSqrd = dot(p, p);
        if (1e-160 < lengthSqrd && lengthSqrd <= 1)
            p = normalize(p);
            break;
    }
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
        return -1.0;
    }
    else {
        return -1.0;
    }
    

}

vec3 getSurfaceNormal(Ray ray, Sphere sphere, float t) {
    if (t > 0) {
        vec3 p = ray.origin + t * ray.dir;
        vec3 normal = normalize((p - sphere.center)); // sphere.radius;
        return normal;
    }
    else {
        return vec3(0.0, 0.0, 0.0);
    }
}

void main() {

    Camera camera;
    camera.center = vec3(0.0, 0.0, 0.0);

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

    Sphere sphere = Sphere(vec4(0.0, 1.0, 1.0, 1.0), vec3(0.0, 0.0, -2.0), 0.5);
    Sphere ground = Sphere(vec4(0.0, 1.0, 0.0, 1.0), vec3(0.0, 100.5, -2.0), 100.0);



    vec3 targetPixel = vp.firstPixel + (pixelCoords.x * vp.du) + (pixelCoords.y * vp.dv);

    Ray ray = Ray(camera.center, targetPixel - camera.center);  
    vec4 pixelColor = vec4(0.2, 0.5, 0.8, 1.0);
    int rayDepth = 20;
    float scalar = 1.0;
    Ray currentRay = ray;
    for (int i = 0; i < rayDepth; i++) {
        float tSphere = getSphereHit(ray, sphere, 0, 100000);
        float tGround = getSphereHit(ray, ground, 0, 100000);
        float t;
        vec3 normal;
        if (tSphere != -1) {
            scalar = scalar * 0.5;
            t = tSphere;
            normal = getSurfaceNormal(ray, sphere, tSphere);
        }
        else if (tGround != -1) {
            scalar = scalar * 0.5;
            t = tGround;
            normal = getSurfaceNormal(ray, ground, tGround);
        }
        
        else {
            break;
        }
        currentRay.origin = currentRay.origin + currentRay.dir * t;
        currentRay.dir = getRandomOnHemi(normal);

    }
    //vec4(scalar, scalar, scalar, 1.0);
    pixelColor = pixelColor * scalar; 

    
    ivec2 pixelLoc = ivec2(gl_GlobalInvocationID.xy);
     

    imageStore(img, pixelLoc, pixelColor);
}
