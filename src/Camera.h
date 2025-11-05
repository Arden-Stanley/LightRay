#ifndef CAMERA_H
#define CAMERA_H

namespace LR {
    typedef struct {
        float x, y, z;
    } Position;

    class Camera {
        public:
            Camera(const Position &pos);
            ~Camera();
            void Move(const Position &offset);
            void MoveTo(const Position &pos);
        private:
            Position m_pos;
    };
}

#endif