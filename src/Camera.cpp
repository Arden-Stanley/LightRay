#include "Camera.h"

namespace LR {
    Camera::Camera(const Position &pos) : m_pos(pos) {}

    Camera::~Camera() {}

    void Camera::Move(const Position &offset) {
        m_pos.x += offset.x;
        m_pos.y += offset.y;
        m_pos.z += offset.z;
    }

    void Camera::MoveTo(const Position &pos) {
        m_pos = pos;
    }

    Position Camera::GetPos() const {
        return m_pos;
    }
}