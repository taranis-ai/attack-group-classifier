# Taranis AI attack-group-classifier Bot

Add more verbose description of your Bot here


## Pre-requisites

- uv - https://docs.astral.sh/uv/getting-started/installation/
- docker (for building container) - https://docs.docker.com/engine/

Create a python venv and install the necessary packages for the bot to run.

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras --dev
```

## Usage

You can run your bot locally with

```bash
flask run
# or
granian run
```


## Docker

You can also create a Docker image out of this bot. For this, you first need to build the image with the build_container.sh

You can specify which model the image should be built with the INCLUDED_MODEL environment variable. If you omit it, the image will be built with the default model.

```bash
INCLUDED_MODEL=model_name ./build_container.sh
```

then you can run it with:

```bash
docker run -p 5500:8000 <image-name>:<tag>
```

If you encounter errors, make sure that port 5500 is not in use by another application.


## Test the bot

Once the bot is running, you can send test data to it on which it runs its inference method:

```bash
curl -X POST http://127.0.0.1:5500 -H "Content-Type: application/json" -d '{"key": "some data"}'
```

## License

EUROPEAN UNION PUBLIC LICENCE v. 1.2
