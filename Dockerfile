FROM centos:7

LABEL maintainer="Aagam Shah <aagam@redhat.com>"

COPY ./src /src
COPY ./requirements.txt /requirements.txt
COPY ./entrypoint.sh /bin/entrypoint.sh

RUN yum install -y epel-release &&\
    yum install -y gcc-c++ git python34-pip python34-requests httpd httpd-devel python34-devel &&\
    yum clean all

RUN chmod 0777 /bin/entrypoint.sh

RUN pip3 install -r requirements.txt

# Don't put this in requirements.txt. It will start failing.
RUN pip3 install Cython==0.29.1 && pip3 install hpfrec==0.2.2.9

ENTRYPOINT ["/bin/entrypoint.sh"]
