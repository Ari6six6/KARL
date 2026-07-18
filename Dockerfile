# KARL — run the whole harness in a container.
#
# This is the clean way to let the crew use the shell freely: the container is
# the sandbox, so `--shell host` inside it can only touch this disposable box.
# Mount a workspace and pass the engine endpoint at run time:
#
#   docker build -t karl .
#   docker run --rm -it \
#     -e KARL_BASE_URL=http://your-gpu-box:8080/v1 \
#     -e KARL_MODEL=your-model \
#     -e KARL_SHELL=host \
#     -v "$PWD":/work -e KARL_WORKSPACE=/work \
#     -v karl-state:/root/.karl \
#     karl
#
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY karl ./karl
RUN pip install --no-cache-dir .

# state lives here; mount a volume to persist projects/config across runs
ENV KARL_HOME=/root/.karl
ENTRYPOINT ["karl"]
