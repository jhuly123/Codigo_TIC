import os
import logging
import logging.config

logging.config.dictConfig({
  "version": 1,
  "formatters": {
    "standard": {
      "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "DEBUG",
      "stream": "ext://sys.stdout",
      "formatter": "standard"
    }
  },
  "root": {
    "level": "ERROR",
    "handlers": ["console"],
    "propagate": True
  }
})

from label_studio_ml.api import init_app
from ml_backend import SignLanguageBackend


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Label studio ML backend')
    parser.add_argument('-p', '--port', dest='port', type=int, default=9090)
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.add_argument('--log-level', dest='log_level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None)
    parser.add_argument('--model-dir', dest='model_dir', default=os.path.dirname(__file__))
    args = parser.parse_args()

    if args.log_level:
        logging.root.setLevel(args.log_level)

    app = init_app(
        model_class=SignLanguageBackend,
        model_dir=os.environ.get('MODEL_DIR', args.model_dir),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379),
    )
    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    app = init_app(
        model_class=SignLanguageBackend,
        model_dir=os.environ.get('MODEL_DIR', os.path.dirname(__file__)),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379),
    )
