# Reflex environment: Isaac Sim 5.1.0 + Isaac Lab v2.3.1 + rsl-rl 3.0.1 + this repository.
# build:  docker build -f docker/Dockerfile -t reflex .
# run:    docker/run.sh            (mounts ./data and ./outputs, GPU, headless)
FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y DEBIAN_FRONTEND=noninteractive TERM=xterm \
    ISAACSIM_PATH=/isaac-sim ISAACLAB_PATH=/opt/IsaacLab ISAACLAB_PY=/isaac-sim/python.sh
RUN apt-get update && apt-get install -y --no-install-recommends git curl cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isaac Lab v2.3.1 (the version the paper results were produced with)
RUN git clone --depth 1 --branch v2.3.1 https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} \
    && ln -sf ${ISAACSIM_PATH} ${ISAACLAB_PATH}/_isaac_sim \
    && ${ISAACLAB_PY} -m pip install --no-cache-dir --no-build-isolation flatdict==4.0.1 \
    && cd ${ISAACLAB_PATH} && ./isaaclab.sh --install rsl_rl
# flatdict (an Isaac Lab dependency) only builds with a setuptools that still has pkg_resources; building it
# against Isaac Sim's bundled setuptools avoids a silent failure of the isaaclab core install.

# pinned learning stack + small extras used by the scripts
RUN ${ISAACLAB_PY} -m pip install --no-cache-dir rsl-rl-lib==3.0.1 tensordict==0.12.0 numpy==1.26.4 \
    hydra-core==1.3.2 onnx==1.21.0 toml joblib scipy matplotlib "imageio[ffmpeg]" onnxscript psutil wandb

# this repository (installed in editable mode so ./data and ./outputs can be mounted)
COPY . /reflex
WORKDIR /reflex
RUN ${ISAACLAB_PY} -m pip install --no-cache-dir --no-deps -e source/whole_body_tracking -e source/reflex \
    && bash scripts/setup_assets.sh \
    && ${ISAACLAB_PY} -m pip show isaaclab isaaclab_rl isaaclab_tasks rsl-rl-lib whole_body_tracking reflex > /dev/null
# (the last line fails the build if any core package is missing -- isaaclab.sh does not propagate install errors)

ENTRYPOINT ["/bin/bash"]
