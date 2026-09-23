"""Container health probe.

The app is designed to start even when the artifacts are missing or mismatched;
/health then answers 503 with a user-safe reason. So any HTTP answer means the
process is alive and worth keeping. Only a connection failure is unhealthy, and
the readiness detail stays in /health's own body.
"""
import sys
import urllib.error
import urllib.request

URL = 'http://127.0.0.1:8000/health'

try:
    urllib.request.urlopen(URL, timeout=5)
except urllib.error.HTTPError:
    pass
except Exception as error:  # connection refused, timeout, DNS, ...
    print(f'{type(error).__name__}: {error}', file=sys.stderr)
    sys.exit(1)
