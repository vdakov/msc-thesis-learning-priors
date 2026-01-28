FROM python:3.12.12-slim
LABEL org.opencontainers.image.source="https://github.com/vdakov/msc-thesis-learning-priors"
LABEL org.opencontainers.image.description="ML Training Image for Prior Learning using Prior-Fitted Networks"
LABEL maintainer="v.dakov02@gmail.com"

WORKDIR /src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/


ENTRYPOINT ["python", "src/training_script.py"]