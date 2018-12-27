FROM centos:7

LABEL maintainer="Aagam Shah <aagam@redhat.com>"

RUN yum install -y epel-release &&\
    yum install -y gcc-c++ git python34-pip python34-requests httpd httpd-devel python34-devel &&\
    yum clean all
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install Cython==0.29.1 && pip3 install hpfrec==0.2.2.9

COPY ./entrypoint.sh /bin/entrypoint.sh

COPY ./src /src

RUN chmod +x /bin/entrypoint.sh


# Don't put this in requirements.txt. It will start failing.

ENTRYPOINT ["/bin/entrypoint.sh"]
