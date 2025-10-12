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

vec3 getRandomOnHemi(vec3 normal) {
    vec3 randOnSphere;
    while (true) {
        vec3 p = vec3(
            (getRandom(gl_GlobalInvocationID.yy * 102) * 2.0) - 1.0, 
            (getRandom(gl_GlobalInvocationID.yy * 232) * 2.0) - 1.0, 
            (getRandom(gl_GlobalInvocationID.yy * 827) * 2.0) - 1.0
        );
        float lengthSqrd = p.x * p.x + p.y * p.y + p.z * p.z;
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
float getSphereHit(Ray ray, Sphere sphere) {
    vec3 offset = sphere.center - ray.origin;
    float a = dot(ray.dir, ray.dir);
    float h = dot(ray.dir, offset);
    float c = dot(offset, offset) - (sphere.radius * sphere.radius);
    float discriminant = (h * h) - (a * c);

    if (discriminant < 0) {
        return -1;
    }
    else {
        float sqrtd = sqrt(discriminant);
        float root = (h - sqrtd) / a;
        if (root <= 0.0) {
            root = (h + sqrtd) / a;
            if (root <= 0.0) {
                return -1;
            }
        }
        return root;
    }
}

vec3 getSurfaceNormal(Ray ray, Sphere sphere, float t) {
    if (t != -1) {
        vec3 p = ray.origin + t * ray.dir;
        vec3 normal = (p - sphere.center) / sphere.radius;
        normal = normalize(normal);
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
    
    Sphere sphere = Sphere(vec4(0.0, 1.0, 1.0, 1.0), vec3(0.0, 0.0, -2.0), 1.0);
    Sphere ground = Sphere(vec4(0.0, 1.0, 0.0, 1.0), vec3(0.0, 0.0, -5.0), 3.0);


    vec3 targetPixel = vp.firstPixel + (pixelCoords.x * vp.du) + (pixelCoords.y * vp.dv);

    Ray ray = Ray(camera.center, targetPixel - camera.center);  

    
    //RENDER SURFACE NORMALS AS COLOR
    /*
    vec3 normal = getSurfaceNormal(ray, sphere, getSphereHit(ray, sphere));
    if (normal != vec3(0, 0, 0)) {
        pixelColor = vec4(normal, 2.0);
    }
    */
    
    Ray currentRay = ray;
    float scalar = 1;
    vec4 pixelColor;

    int rayDepth = 4;

    for (int i = 0; i < rayDepth; i++) {
        //float tSphere = getSphereHit(currentRay, sphere);
        float tGround = getSphereHit(currentRay, ground);
        float t;

        /*if (tSphere != -1 && tSphere <= tGround) {
            t = tSphere;
            scalar *= 0.3;
        }*/
        
        if (tGround != -1) {
            t = tGround;
            scalar *= 0.5;
        }
        
        else {
            pixelColor = vec4(0.2, 0.8, 1.0, 1.0) * scalar;
            break;
        }
        
        vec3 normal = getSurfaceNormal(currentRay, sphere, t);
        currentRay.origin = t * currentRay.dir;
        currentRay.dir = getRandomOnHemi(normal);
    }

    
    
    

    imageStore(img, pixelCoords, pixelColor);
}
