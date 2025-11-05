#include "Camera.h"

namespace LR {
    Camera::Camera(const Position &pos) : m_pos(pos) {}

    Camera::~Camera() {}

    void Camera::Move(const Position &offset) {
        m_pos = offset;
    }

    void Camera::MoveTo(const Position &pos) {
        m_pos = pos;
    }
}