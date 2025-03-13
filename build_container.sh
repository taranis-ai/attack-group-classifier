#!/bin/bash

if [ ! -e "uv.lock" ]
then
    echo "uv.lock does not exist. Please run 'uv sync' or 'uv lock' to create it"
    echo "Error: Cannot build container without uv.lock -> Abort"
    exit
fi

set -eou pipefail

build_tags=""
build_args=""

INCLUDED_MODEL=${INCLUDED_MODEL:-"mpnet"}

# if not a git repo, build locally
if [ -d .git ]; then
    cd $(git rev-parse --show-toplevel)

    repo_name=$(grep 'url =' .git/config | sed -E 's/.*[:\/]([^\/]+)\.git/\1/')
    # if remote url for repo not given -> abort
    if [ -z "$repo_name" ]; then
        echo "Error: Could not get remote repository name. Please check your .git/config"
        exit 1
    fi

    GITHUB_REPOSITORY_OWNER=${GITHUB_REPOSITORY_OWNER:-"ghcr.io/taranis-ai"}
    build_args+="--build-arg GITHUB_REPOSITORY_OWNER='${GITHUB_REPOSITORY_OWNER}'"
    build_tags+="--tag ${GITHUB_REPOSITORY_OWNER}/${REPO_NAME}>:latest "
    build_tags+="--tag '${GITHUB_REPOSITORY_OWNER}/${REPO_NAME}:${INCLUDED_MODEL}'"

    # check if there are any commits yet
    if git rev-parse --quiet --verify HEAD >/dev/null; then
        CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9_.-]/_/g')
        echo "Building image for branch ${CURRENT_BRANCH} and model ${INCLUDED_MODEL} on ${GITHUB_REPOSITORY_OWNER}"
        build_tags+="--tag ${GITHUB_REPOSITORY_OWNER}/${REPO_NAME}:${CURRENT_BRANCH} "
    else
        echo "The repository has no commits yet."
    fi

else
    build_tags+="--tag attack_group_classifier:${INCLUDED_MODEL} "
    build_tags+="--tag attack_group_classifier:latest"
    echo "Current directory is not a git repo. Building image for model ${INCLUDED_MODEL} locally"

fi;

build_args+="--build-arg MODEL='${INCLUDED_MODEL}'"

docker buildx build --file Containerfile \
  $build_args \
  $build_tags \
  --load .
