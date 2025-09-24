#include "UI.h"
#include "imgui.h"

void UI::BeginFrame()
{
    ImGui::NewFrame();
}

void UI::EndFrame()
{
    ImGui::Render();
    // Normally you would call ImGui_ImplXXXX_RenderDrawData here for your backend
}

void UI::ShowDemoWindow(bool* p_open)
{
    ImGui::ShowDemoWindow(p_open);
}
