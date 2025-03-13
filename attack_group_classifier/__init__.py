from flask import Flask
from attack_group_classifier import router


def create_app():
    app = Flask(__name__)
    app.config.from_object("attack_group_classifier.config.Config")

    with app.app_context():
        init(app)

    return app


def init(app: Flask):
    router.init(app)


if __name__ == "__main__":
    create_app().run()
