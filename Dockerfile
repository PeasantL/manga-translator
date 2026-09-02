# The torch stack is the reason this base image is used rather than a slim
# Python one: building it from the PyTorch index adds gigabytes to the build and
# has to be kept in step with the host's CUDA driver by hand.
#
# 2.5.1 rather than something older because transformers reaches for
# torch.utils._pytree.register_pytree_node, which does not exist before 2.2.
# cu121 rather than a newer CUDA because its wheels still carry sm_61, which is
# what a GTX 1070 is; the cu128 builds dropped Pascal.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# curl backs fetch_models.sh and the healthcheck. libglib2.0-0 is opencv's one
# remaining system dependency -- the headless build drops the rest.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch and torchvision are already in the base image and satisfy the pins, so
# pip leaves them alone instead of pulling the CPU-only wheels over them.
#
# The compiler is installed and removed within the one layer because pyhyphen
# publishes no wheels and builds from source. Leaving build-essential behind
# would add ~200 MB to an image that never compiles anything again.
#
# ultralytics depends on the full opencv-python, which lands alongside the
# headless build this project asks for and wins the import -- and then wants
# libGL, which a headless container has no reason to carry. Both are removed
# and headless reinstalled rather than just dropping the full build: the two
# unpack into the same cv2 directory, so uninstalling either takes the module
# with it. numpy is named again on that reinstall so pip picks an opencv built
# against 1.x -- left to itself it takes the newest, which requires numpy 2 and
# breaks every other package here against the pin in requirements.txt.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir "opencv-python-headless>=4.8.1.78,<5.0" "numpy>=1.26.2,<2.0" && \
    apt-get purge -y --auto-remove build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Weights are not in the image. All four models -- detection, segmentation,
# cleaning and OCR -- are fetched from the Hugging Face hub the first time they
# are used, into these caches. Volumes in the compose file, so a rebuild does
# not re-download five gigabytes.
# YOLO_CONFIG_DIR is set because ultralytics writes a settings file on import
# and warns on every run when it cannot: /app belongs to the image, and the
# container does not necessarily run as a user who can write there.
ENV HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    JOB_DIR=/jobs \
    PORT=1007 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

COPY . .

EXPOSE 1007

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

# Weights are fetched at boot rather than baked in, so an image rebuild does
# not re-download them and a running container always has them.
CMD ["sh", "-c", "./fetch_models.sh && exec uvicorn service:app --host 0.0.0.0 --port ${PORT}"]
