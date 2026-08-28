# cifsQuant application container — Stages 2/3 (gating + spatial) and the GUI.
# Stage 1 (Nextflow segmentation) launches its own Docker containers and must
# run on the host — see nextflow.config.
#
# Build:  docker compose build
# GUI:    docker compose up gui          → http://localhost:8501
# Shell:  docker compose run --rm cli

FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/environment.yaml
RUN micromamba install -y -n base -f /tmp/environment.yaml && \
    micromamba clean --all --yes

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# micromamba's entrypoint activates the env before CMD
ARG MAMBA_DOCKERFILE_ACTIVATE=1

EXPOSE 8501
CMD ["streamlit", "run", "gui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
