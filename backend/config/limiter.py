"""Shared slowapi limiter instance.

Lives in config/ rather than the router or main.py because BOTH need to
import it: routers apply the @limiter.limit decorator, main.py wires the
RateLimitExceeded handler and stashes the limiter on app.state. A shared
location avoids circular imports.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# key_func=get_remote_address rate-limits per client IP. For dev/test that's
# fine; in prod behind a load balancer, ensure the upstream sets X-Forwarded-For
# or the limiter will treat all traffic as coming from the LB's IP.
limiter = Limiter(key_func=get_remote_address)
