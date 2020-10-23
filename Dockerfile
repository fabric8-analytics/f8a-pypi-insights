FROM centos:7

RUN yum install -y epel-release &&\
    yum install -y gcc-c++ git python36-pip python36-requests httpd httpd-devel python36-devel &&\
    yum clean all

COPY ./requirements.txt /requirements.txt

RUN pip3 install --upgrade pip

RUN pip3 install git+https://github.com/fabric8-analytics/fabric8-analytics-rudra#egg=rudra

RUN pip3 install -r requirements.txt

RUN pip3 install Cython==0.29.1 && pip3 install hpfrec==0.2.2.9

COPY ./entrypoint.sh /bin/entrypoint.sh

COPY ./src /src

RUN chmod +x /bin/entrypoint.sh

ENTRYPOINT ["/bin/entrypoint.sh"]
