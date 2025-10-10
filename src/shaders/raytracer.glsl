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

vec3 getUnitVec3(vec3 v) {
    vec3 unitVec = v / (sqrt(v.x * v.x + v.y * v.y + v.z + v.z));
    return unitVec;
}

vec3 getRandomOnHemi(vec3 normal) {
    vec3 randOnSphere;
    while (true) {
        vec3 p = vec3(
            (getRandom(gl_GlobalInvocationID.xy * 102) + 1.0) / 2.0, 
            (getRandom(gl_GlobalInvocationID.xy * 232) + 1.0) / 2.0, 
            (getRandom(gl_GlobalInvocationID.xy * 827) + 1.0) / 2.0
        );
        float lengthSqrd = p.x * p.x + p.y * p.y + p.z * p.z;
        if (1e-160 < lengthSqrd && lengthSqrd <= 1)
            randOnSphere =  p / sqrt(lengthSqrd);
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

vec3 getSurfaceNormal(Ray ray, Sphere sphere, float t) {
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

    int rayDepth = 4;

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

    Sphere sphere = Sphere(vec4(0.0, 1.0, 1.0, 1.0), vec3(0.0, 0.0, -2.0), 1.0);
    Sphere ground = Sphere(vec4(0.0, 1.0, 0.0, 1.0), vec3(0.0, 100.5, -1.0), 100.0);


    vec4 pixelColor = vec4(0.0, 0.0, 0.0, 1.0);

    vec3 targetPixel = vp.firstPixel + (pixelCoords.x * vp.du) + (pixelCoords.y * vp.dv);

    Ray ray = Ray(camera.center, targetPixel - camera.center);  

    
    //RENDER SURFACE NORMALS AS COLOR
    /*
    vec3 normal = getSurfaceNormal(ray, sphere, getSphereHit(ray, sphere));
    if (normal != vec3(0, 0, 0)) {
        pixelColor = vec4(normal, 2.0);
    }
    */
    float t;
    vec3 unitVec = getUnitVec3(ray.dir);

    for (int i = 0; i < rayDepth; i++) {
        t = getSphereHit(ray, sphere);
        if (t != -1) {
            vec3 direction = getRandomOnHemi(getSurfaceNormal(ray, sphere, t));
            ray.origin = t * ray.dir;
            ray.dir = direction;
            pixelColor += vec4(0.0, 1.0, 0.0, 1.0);
        }
        else {
            float a = 0.5 * (unitVec.y + 1.0);
            vec4 color = (1.0 - a) * vec4(1.0, 1.0, 1.0, 1.0) + a * vec4(0.0, 0.0, 1.0, 1.0);

            if (i != 0) {
                pixelColor += color;
                pixelColor = pixelColor / rayDepth;
            }
            else {
                pixelColor = color;
            }
            break;
        }
    }


    //pixelColor = getRayColor(ray, sphere);
    
    

    imageStore(img, pixelCoords, pixelColor);
}
