FROM python:3.12.0-slim

WORKDIR /miniproject-docker

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . ./miniproject-docker

EXPOSE 5000

CMD [ "python", "-m" , "flask", "run", "0.0.0.5000"]