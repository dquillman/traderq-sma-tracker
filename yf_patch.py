# yf_patch.py - force a proper session and (optionally) pdr override
import requests
import yfinance as yf
import sys

# 1. Try to enable pdr override
try:
    from pandas_datareader import data as pdr  # noqa: F401
    yf.pdr_override()
except Exception:
    pass

# 2. Create a shared session with realistic headers
try:
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    })
except Exception as e:
    print(f"Error creating session in yf_patch: {e}", file=sys.stderr)
    sess = requests.Session() # Fallback to default session

# 3. Patch yfinance session
try:
    import yfinance.shared as yfs
    yfs._base._requests = sess
    yfs._requests = sess
    # Also try to set it in the base module
    if hasattr(yfs, '_base'):
        if hasattr(yfs._base, '_requests'):
            yfs._base._requests = sess
except Exception:
    pass

# 4. Monkey-patch yfinance.download
try:
    import yfinance as yf_module
    _original_download = yf_module.download
    
    def _patched_download(*args, **kwargs):
        # Always inject our session if not provided
        if 'session' not in kwargs:
            kwargs['session'] = sess
        # Disable threads by default to avoid session issues
        if 'threads' not in kwargs:
            kwargs['threads'] = False
        return _original_download(*args, **kwargs)
    
    yf_module.download = _patched_download
except Exception:
    pass

# 5. Convenience wrapper
def download(*args, **kwargs):
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)
    return yf.download(*args, **kwargs, session=sess)
