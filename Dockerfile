FROM python:3.9-buster

COPY ./app /app

RUN apt-get update -y && apt-get upgrade -y \
    && pip install -r /app/requirements.txt \
    && mkdir /apps

RUN cd /apps \
    && git clone https://github.com/facebookresearch/LASER.git \
    && export LASER=/apps/LASER \
    && ./LASER/install_models.sh \
    && ./LASER/install_external_tools.sh

RUN cd /apps \
    && git clone https://github.com/thompsonb/vecalign.git 

WORKDIR /app

RUN useradd -ms /bin/bash user \
    && chown -R user /app \
    && chown -R user /apps
USER user

EXPOSE 80

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]