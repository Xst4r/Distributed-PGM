FROM ubuntu:18.04

#8074:9001

RUN groupadd --gid 9001 s876clal                                && \
    useradd -ms /bin/bash --uid 8030 --8074 9001  heinrich      && \
    echo "root:root" | chpasswd                                 && \
    echo "heinrich:root" | chpasswd

RUN apt-get update && \
    apt-get install -y python3 \
                       python3-pip

RUN python3 -m pip install pxpy