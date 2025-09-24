#version 460 core

out vec4 fragColor;
in vec2 oTextureCoordinates;

uniform sampler2D tex;

void main() {
	vec3 textureColor = texture(tex, oTextureCoordinates).rgb;
	fragColor = vec4(textureColor, 1.0);
}
