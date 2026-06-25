FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tzdata \
    default-jre-headless \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./
COPY drispi/ ./drispi/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/bks/ ./data/bks/
COPY ext/ ./ext/
COPY README.md ./

RUN uv sync --frozen --no-dev
RUN uv pip install setuptools

RUN cd ext/ails2 && ./build.sh 2>/dev/null || mvn package -q 2>/dev/null || true
RUN cd ext/filo && make -s 2>/dev/null || true
RUN cd ext/filo2 && make -s 2>/dev/null || true

ENV GRB_LICENSE_FILE=/root/gurobi.lic
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

ENTRYPOINT ["uv", "run", "python", "-m", "drispi.pipeline.runner"]
