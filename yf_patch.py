# yf_patch.py ??? force a proper session and (optionally) pdr override
import requests
import yfinance as yf
try:
    from pandas_datareader import data as pdr  # noqa: F401
    yf.pdr_override()
except Exception:
    pass

# Shared session with realistic headers fixes 403/HTML responses
sess = requests.Session()
sess.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
})

# yfinance uses shared._base._requests.Session under the hood; set the default session
try:
    import yfinance.shared as yfs
    yfs._base._requests = sess
    yfs._requests = sess
except Exception:
    pass

# Convenience wrapper if you want to call via yf_patch.download(...)
def download(*args, **kwargs):
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)
    return yf.download(*args, **kwargs, session=sess)
