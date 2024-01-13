import argparse
from api.app import app
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Flask app')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the Flask app on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on')
    args = parser.parse_args()

    # hack to get logging to work as I can't make the logger before running flask
    sys.stderr.write(f'Running Flask app on {args.host}:{args.port}')
    sys.stderr.flush()
    print(f'Running Flask app on {args.host}:{args.port}')
    app.run(debug = True, host='0.0.0.0', port=7007)