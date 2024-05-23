FROM apache/airflow:2.9.0-python3.10

USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         poppler-utils \
         heimdal-dev \
         libgl1-mesa-glx \
         libgtk2.0-dev \
         libgl1 \
         libglib2.0-0 \
         libnss3 \
         libasound2 \
         libgbm-dev \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN python3 -m pip install playwright
RUN python3 -m playwright install --with-deps

RUN su airflow -c 'pip install --default-timeout=2000 -r requirements.txt'

RUN su airflow -c 'pip install git+https://github.com/facebookresearch/detectron2.git'
# RUN \
#     pip install --upgrade --no-cache-dir --no-user pip && \
#     pip install --no-cache-dir --no-user -r requirements.txt
RUN su airflow -c 'pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_lg-3.7.0/fr_core_news_lg-3.7.0-py3-none-any.whl'

RUN su airflow -c 'playwright install'