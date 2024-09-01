FROM amazonlinux:2023
RUN mkdir -p /app && chown -R root:root /app
WORKDIR /app
COPY requirements.txt ./
RUN dnf -y update && \
    dnf -y install python3 python3-pip && \
    pip3 install jupyter && \
    pip3 install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
