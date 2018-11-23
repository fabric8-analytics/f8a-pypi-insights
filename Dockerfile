FROM centos:7

LABEL maintainer="Aagam Shah <aagam@redhat.com>"

COPY ./src /src
COPY ./requirements.txt /requirements.txt
COPY ./entrypoint.sh /bin/entrypoint.sh

RUN yum install -y epel-release &&\
    yum install -y gcc git python34-pip python34-requests httpd httpd-devel python34-devel &&\
    yum clean all

RUN chmod 0777 /bin/entrypoint.sh

RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/entrypoint.sh"]
