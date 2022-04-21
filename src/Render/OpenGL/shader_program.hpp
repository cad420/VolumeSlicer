//
// Created by wyz on 2021/4/7.
//

#pragma once

#include <VolumeSlicer/Common/export.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <glad/glad.h>
#include <glm/glm.hpp>

VS_START

class Shader
{
  private:
    unsigned int ID;

  public:
    Shader() = default;
    void setShader(const char *vertexShader, const char *fragmentShader, const char *geometryShader = nullptr);

    Shader(const char *vertexPath, const char *fragmentPath, const char *geometryPath = nullptr);

    void use();

    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;

    void setVec2(const std::string &name, const glm::vec2 &value) const;
    void setVec2(const std::string &name, float x, float y) const;
    void setIVec2(const std::string &name, const glm::ivec2 &value) const;
    void setVec3(const std::string &name, const glm::vec3 &value) const;
    void setVec3(const std::string &name, float x, float y, float z) const;
    void setIVec3(const std::string &name, const glm::ivec3 &value) const;
    void setIVec3(const std::string &name, int x, int y, int z) const;
    void setVec4(const std::string &name, const glm::vec4 &value) const;
    void setVec4(const std::string &name, float x, float y, float z, float w) const;

    void setMat2(const std::string &name, const glm::mat2 &mat) const;
    void setMat3(const std::string &name, const glm::mat3 &mat) const;
    void setMat4(const std::string &name, const glm::mat4 &mat) const;

    void setUIArray(const std::string &name, uint32_t *data, int count);
    void setFloatArray(const std::string &name, float *data, int count);
    void setIntArray(const std::string &name, int *data, int count);

    void checkCompileErrors(GLuint id, std::string type);
};

inline Shader::Shader(const char *vertexPath, const char *fragmentPath, const char *geometryPath)
{
    // 1.load glsl file
    std::string vertexCode;
    std::string fragmentCode;
    std::string geometryCode;
    std::ifstream vShaderFile, fShaderFile, gShaderFile;

    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;

        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();

        vShaderFile.close();
        fShaderFile.close();

        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();

        if (geometryPath != nullptr)
        {
            gShaderFile.open(geometryPath);
            std::stringstream gShaderStream;
            gShaderStream << gShaderFile.rdbuf();
            gShaderFile.close();
            fragmentCode = gShaderStream.str();
        }
    }
    catch (std::ifstream::failure &e)
    {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }
    // 2.compile shaders
    const char *vShaderCode = vertexCode.c_str();
    const char *fShaderCode = fragmentCode.c_str();

    unsigned int vShader, fShader, gShader;
    vShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vShader, 1, &vShaderCode, NULL);
    glCompileShader(vShader);
    checkCompileErrors(vShader, "VERTEX");

    fShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fShader, 1, &fShaderCode, NULL);
    glCompileShader(fShader);
    checkCompileErrors(fShader, "FRAGMENT");

    if (geometryPath != nullptr)
    {
        const char *gShaderCode = geometryCode.c_str();

        gShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(gShader, 1, &gShaderCode, NULL);
        glCompileShader(gShader);
        checkCompileErrors(gShader, "GEOMETRY");
    }

    ID = glCreateProgram();
    glAttachShader(ID, vShader);
    glAttachShader(ID, fShader);
    if (geometryPath != nullptr)
        glAttachShader(ID, gShader);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");

    glDeleteShader(vShader);
    glDeleteShader(fShader);
    if (geometryPath != nullptr)
        glDeleteShader(gShader);
}

inline void Shader::use()
{
    glUseProgram(ID);
}

inline void Shader::setBool(const std::string &name, bool value) const
{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}

inline void Shader::setInt(const std::string &name, int value) const
{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

inline void Shader::setFloat(const std::string &name, float value) const
{
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

inline void Shader::setVec2(const std::string &name, const glm::vec2 &value) const
{
    glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

inline void Shader::setVec2(const std::string &name, float x, float y) const
{
    glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
}

inline void Shader::setIVec2(const std::string &name, const glm::ivec2 &value) const
{
    glUniform2iv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

inline void Shader::setVec3(const std::string &name, const glm::vec3 &value) const
{
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

inline void Shader::setVec3(const std::string &name, float x, float y, float z) const
{
    glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

inline void Shader::setIVec3(const std::string &name, int x, int y, int z) const
{
    glUniform3i(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

inline void Shader::setIVec3(const std::string &name, const glm::ivec3 &value) const
{
    glUniform3iv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

inline void Shader::setVec4(const std::string &name, const glm::vec4 &value) const
{
    glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

inline void Shader::setVec4(const std::string &name, float x, float y, float z, float w) const
{
    glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

inline void Shader::setMat2(const std::string &name, const glm::mat2 &mat) const
{
    glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

inline void Shader::setMat3(const std::string &name, const glm::mat3 &mat) const
{
    glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

inline void Shader::setMat4(const std::string &name, const glm::mat4 &mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

inline void Shader::checkCompileErrors(GLuint id, std::string type)
{
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM")
    {
        glGetShaderiv(id, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(id, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                      << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
    else
    {
        glGetProgramiv(id, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(id, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                      << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}

inline void Shader::setShader(const char *vertexShader, const char *fragmentShader, const char *geometryShader)
{
    // 1.load glsl file
    std::string vertexCode;
    std::string fragmentCode;
    std::string geometryCode;
    std::ifstream vShaderFile, fShaderFile, gShaderFile;

    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    // 2.compile shaders
    const char *vShaderCode = vertexShader;
    const char *fShaderCode = fragmentShader;

    unsigned int vShader, fShader, gShader;

    vShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vShader, 1, &vShaderCode, nullptr);
    glCompileShader(vShader);
    checkCompileErrors(vShader, "VERTEX");

    fShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fShader, 1, &fShaderCode, nullptr);
    glCompileShader(fShader);
    checkCompileErrors(fShader, "FRAGMENT");

    if (geometryShader != nullptr)
    {
        const char *gShaderCode = geometryCode.c_str();

        gShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(gShader, 1, &gShaderCode, nullptr);
        glCompileShader(gShader);
        checkCompileErrors(gShader, "GEOMETRY");
    }

    ID = glCreateProgram();
    glAttachShader(ID, vShader);
    glAttachShader(ID, fShader);
    if (geometryShader != nullptr)
        glAttachShader(ID, gShader);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");

    glDeleteShader(vShader);
    glDeleteShader(fShader);
    if (geometryShader != nullptr)
        glDeleteShader(gShader);
}

inline void Shader::setUIArray(const std::string &name, uint32_t *data, int count)
{
    glUniform1uiv(glGetUniformLocation(ID, name.c_str()), count, data);
}

inline void Shader::setFloatArray(const std::string &name, float *data, int count)
{
    glUniform1fv(glGetUniformLocation(ID, name.c_str()), count, data);
}

inline void Shader::setIntArray(const std::string &name, int *data, int count)
{
    glUniform1iv(glGetUniformLocation(ID, name.c_str()), count, data);
}

VS_END
