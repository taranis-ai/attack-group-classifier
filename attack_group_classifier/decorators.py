from flask import request
from functools import wraps

from attack_group_classifier.config import Config
from attack_group_classifier.log import logger


def debug_request(debug=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not debug or request is None:
                return func(*args, **kwargs)
            log_str = f"Method: {request.method}, Endpoint: {request.path}, "
            payload = request.get_json(silent=True)
            if payload is not None:
                log_str += f"Payload: {payload}"
            logger.debug(log_str)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def api_key_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not Config.API_KEY:
            return fn(*args, **kwargs)

        error = ({"error": "not authorized"}, 401)
        auth_header = request.headers.get("Authorization", None)

        if not auth_header or not auth_header.startswith("Bearer"):
            logger.warning("Missing Authorization Bearer")
            return error

        api_key = auth_header.replace("Bearer ", "")

        if Config.API_KEY != api_key:
            logger.warning("Incorrect api key")
            return error

        # allow
        return fn(*args, **kwargs)

    return wrapper
