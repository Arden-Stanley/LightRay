#version 460 core

out vec4 fragColor;
in vec2 oTextureCoordinates;

uniform sampler2D texture;

void main() {
	vec3 textureColor = texture(texture, oTextureCoordinates).rgb
	fragColor = vec4(textureColor, 1.0);
}
