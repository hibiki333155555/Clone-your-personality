FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install -r requirements.txt -t .

COPY app.py ./
# 追加
COPY model01 /opt/ml/model   
ENV TRANSFORMERS_CACHE /opt/cache/transformers             

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
