FROM debian:latest

RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

RUN pip install flask==1.0.2
RUN pip install requests==2.21.0
RUN pip install pillow==5.4.1
RUN pip install numpy==1.15.1

RUN mkdir static
ADD app.py .
ADD static/* static/
ENV FLASK_ENV development
ENTRYPOINT ["flask", "run", "--host=0.0.0.0"]
