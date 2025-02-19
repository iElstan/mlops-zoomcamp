FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

LABEL maintainer="Roman Peklov"

COPY [ "./requirements.txt", "./" ]

RUN pip install -U pip && pip install -r requirements.txt

RUN mkdir outputs

COPY [ "./starter.py", "./" ]

VOLUME ./outputs

ENTRYPOINT [ "python3", "starter.py" ]

CMD [ "2023", "5" ]