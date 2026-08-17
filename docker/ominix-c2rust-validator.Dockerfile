ARG RUST_IMAGE=rust@sha256:8e8cf8f7fd54a2d23d5a743b3a03f56e26b6c774276c33fa0595111704ebb15c
FROM ${RUST_IMAGE}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -o 65534 -g 65534 /work

COPY scripts/ominix/validate_c2rust_rust_outputs.py /opt/ominix/validate.py

USER 65534:65534
WORKDIR /work
ENTRYPOINT ["python3", "/opt/ominix/validate.py", "--execute-generated-code"]
