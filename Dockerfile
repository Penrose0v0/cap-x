FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# base utils
RUN apt-get update && apt-get install -y \
    wget git zsh curl ca-certificates bzip2 \
    htop vim ffmpeg \
    cmake build-essential \
    libegl1 libopengl0 libglvnd0 libgles2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Point libglvnd EGL loader at NVIDIA's vendor implementation.
# Driver provides libEGL_nvidia.so.0 at runtime when container is run with --gpus.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' \
        '{' \
        '    "file_format_version" : "1.0.0",' \
        '    "ICD" : {' \
        '        "library_path" : "libEGL_nvidia.so.0"' \
        '    }' \
        '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# zsh
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN sed -i "s/# zstyle ':omz:update' mode disabled/zstyle ':omz:update' mode disabled/g" ~/.zshrc

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv

# utils
RUN apt-get update && apt-get install -y htop vim ffmpeg

# Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root
CMD ["zsh"]