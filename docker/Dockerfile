# syntax=docker/dockerfile:1
# Two-stage image: compile external solvers, then Python runtime with DRISPI.

FROM ubuntu:22.04 AS solvers
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake gcc g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY vendor ./vendor
# TODO: when submodules are present: cmake build for vendor/filo, vendor/filo2, vendor/ails2
RUN mkdir -p /opt/solvers/bin \
    && printf '%s\n' "# placeholder — replace with compiled solver binaries" > /opt/solvers/bin/README.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=solvers /opt/solvers /opt/solvers
ENV PATH="/opt/solvers/bin:${PATH}"
COPY pyproject.toml README.md ./
COPY drispi ./drispi
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir .
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "drispi.pipeline.runner", "--help"]
