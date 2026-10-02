FROM python:3.13-slim
RUN useradd -m -u 1000 app
WORKDIR /app
COPY outputs/kalshi-helper/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=app:app outputs/kalshi-helper /app
COPY --chown=app:app REVISION /app/REVISION
ENV PYTHONUNBUFFERED=1 PORT=7860 BETCHECK_PUBLIC_ORIGIN=https://wuhp-kp.hf.space BETCHECK_DATA=/tmp/betcheck
USER app
EXPOSE 7860
CMD ["python", "hosted.py"]
