FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone UnMICST repo (not pip-installable)
#RUN git clone https://github.com/labsyspharm/unmicst.git /opt/unmicst
# Create a proper Python package
#RUN mkdir /opt/unmicst/unmicst && \
#    mv /opt/unmicst/UnMicst.py /opt/unmicst/unmicst/__init__.py
    
#ENV PYTHONPATH="/opt/unmicst"

