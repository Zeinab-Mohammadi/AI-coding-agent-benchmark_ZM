FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone MNE-Python source at v1.7.0 so tests are available
RUN git clone --depth=1 --branch v1.7.0 \
    https://github.com/mne-tools/mne-python.git /mne-python

# Install MNE in editable mode plus test dependencies
# Pin scipy<1.15 because sph_harm was removed in 1.15 and MNE 1.7.0 needs it
RUN pip install --no-cache-dir "scipy<1.15" \
    && pip install --no-cache-dir -e /mne-python[test] \
    && pip install --no-cache-dir pytest pytest-timeout

WORKDIR /workspace
