#pragma once
#include "imgui.h"

class UI
{
public:
    UI() = default;
    ~UI() = default;

    void BeginFrame();   // Start ImGui frame
    void EndFrame();     // Render ImGui frame

    void ShowDemoWindow(bool* p_open = nullptr);  // Example function
};
