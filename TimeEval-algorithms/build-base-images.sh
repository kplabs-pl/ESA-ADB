#!/usr/bin/env bash
# Build the base Docker images locally so algorithms can be built without
# access to the private GitLab registry at registry.gitlab.hpi.de/akita/i/.
#
# Usage: ./build-base-images.sh [--all]
#   Without --all: builds only the base images (required before any algorithm).
#   With --all:    builds base images, then every algorithm in this directory.
#
# Images are tagged with the same registry.gitlab.hpi.de/akita/i/ prefix so
# that algorithm Dockerfiles work without modification.
#
# See https://github.com/kplabs-pl/ESA-ADB/issues/36

set -euo pipefail
cd "$(dirname "$0")"

REGISTRY="registry.gitlab.hpi.de/akita/i"

# Base images in dependency order (later ones depend on earlier ones).
BASE_IMAGES=(
  python3-base
  python36-base
  python2-base
  java-base
  r-base
  r4-base
  rust-base
  pyod
  pyod-1.1.2
  python3-torch
  tsmp
  timeeval-test-algorithm
)

echo "Building base images..."
for img in "${BASE_IMAGES[@]}"; do
  dir="0-base-images/$img"
  if [ ! -d "$dir" ]; then
    echo "  SKIP $img (directory not found)"
    continue
  fi
  echo "  BUILD $REGISTRY/$img"
  docker build -t "$REGISTRY/$img" "$dir" || {
    echo "  FAILED $img (continuing)"
    continue
  }
done

if [ "${1:-}" = "--all" ]; then
  echo ""
  echo "Building algorithm images..."
  for dir in */; do
    [ "$dir" = "0-base-images/" ] && continue
    [ ! -f "$dir/Dockerfile" ] && continue
    name="${dir%/}"
    echo "  BUILD $REGISTRY/$name"
    docker build -t "$REGISTRY/$name" "$dir" || {
      echo "  FAILED $name (continuing)"
    }
  done
fi

echo ""
echo "Done. Built images:"
docker images --filter "reference=$REGISTRY/*" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}" | sort
