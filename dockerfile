FROM python:3.9

LABEL mantainer="Josip Janzic <josip@jjanzic.com>"

WORKDIR /opt/build

ENV OPENCV_VERSION="4.5.1"

RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libopenjp2-7-dev \
        libavformat-dev \
        libpq-dev \
    && pip install numpy \
    && wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip \
    && unzip -qq opencv.zip -d /opt \
    && rm -rf opencv.zip \
    && cmake \
        -D BUILD_TIFF=ON \
        -D BUILD_opencv_java=OFF \
        -D WITH_CUDA=OFF \
        -D WITH_OPENGL=ON \
        -D WITH_OPENCL=ON \
        -D WITH_IPP=ON \
        -D WITH_TBB=ON \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=$(python3.9 -c "import sys; print(sys.prefix)") \
        -D PYTHON_EXECUTABLE=$(which python3.9) \
        -D PYTHON_INCLUDE_DIR=$(python3.9 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON_PACKAGES_PATH=$(python3.9 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        /opt/opencv-${OPENCV_VERSION} \
    && make -j$(nproc) \
    && make install \
    && rm -rf /opt/build/* \
    && rm -rf /opt/opencv-${OPENCV_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq autoremove \
    && apt-get -qq clean

    RUN pip install torch torchvision torchaudio flask PyYAML
# install sql
# Install tools required for the ODBC driver installation and pyodbc compilation
RUN apt-get update \
    && apt-get install -y gnupg2 curl apt-transport-https unixodbc-dev gcc g++ build-essential \
    && apt-get clean

# Add the Microsoft repository for the ODBC Driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Install the ODBC Driver
RUN apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean




WORKDIR /app
# ADD ./camera/main.py ./camera/main.py
# ADD ./camera/alerts.py ./camera/alerts.py
# ADD ./camera/camera.py ./camera/camera.py
# ADD ./camera/filemanager.py ./camera/filemanager.py
# ADD ./camera/MotionDetector.py ./camera/MotionDetector.py
# ADD ./camera/ImageRecognitionSender.py ./camera/ImageRecognitionSender.py
# ADD ./flask/app.py ./flask/app.py



RUN pip install kafka-python avro-python3 pandas numpy pyodbc sqlalchemy


# add camera folder
ADD ./camera ./camera
ADD ./api ./api
ADD ./run.py ./run.py
ADD ./log_config.py ./log_config.py
ADD ./logging.yaml ./logging.yaml
ADD ./avro_schemas ./avro_schemas
# ADD ./flask/config.yaml ./flask/config.yaml

EXPOSE 5000
CMD ["python", "./run.py", "--host=0.0.0.0"]