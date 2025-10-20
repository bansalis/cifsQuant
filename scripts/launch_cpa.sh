#!/bin/bash
echo "🚀 Launching CellProfiler Analyst..."

# Launch CPA with Docker
docker run --rm -it \
    -v "$(pwd)/cellprofiler_analyst:/data" \
    -p 6080:6080 \
    -e DISPLAY=:0 \
    cellprofiler/cellprofiler:4.2.1 \
    cellprofiler-analyst

echo "CPA should be accessible via web interface"
