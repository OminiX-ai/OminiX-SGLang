ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:b688781f3ef66522365ec570885a064b6734750ee22f0d32b3ed49aad87fbf90
FROM ${SGLANG_IMAGE}

# Required by Transformers' device_map loader during offline FP8 conversion.
RUN python3 -m pip install --no-cache-dir accelerate==1.14.0
