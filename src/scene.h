#ifndef SCENE_H
#define SCENE_H

#include "Common.h"

namespace LR {
    class Sphere {
        public:
            Sphere(const Vec3 &position, float radius);
            ~Sphere();
        private:
            Vec3 m_position;
            float m_radius;
    };

    class Scene {
        public:
            Scene();
            ~Scene();
            void AddObject(const Sphere &sphere);
            void Render() const;
        private:
            std::vector<Sphere> _objects;
    };
}

#endif