#!/bin/bash
# Docker build script for RAG System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building RAG System Docker image...${NC}"

# Build the Docker image
echo -e "${YELLOW}Step 1: Building production image...${NC}"
docker build -t rag:latest .

# Verify the build
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
    
    # Display image information
    echo -e "${YELLOW}Image information:${NC}"
    docker images rag:latest
    
    echo -e "${YELLOW}To run the container:${NC}"
    echo "docker run -d -p 8000:8000 --name rag rag:latest"
    echo ""
    echo -e "${YELLOW}Or use docker-compose:${NC}"
    echo "docker-compose up -d"
    
else
    echo -e "${RED}✗ Docker build failed!${NC}"
    exit 1
fi