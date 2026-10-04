"""Local, read-only BRTI history capture. Raw payloads are not model features.

Requires optional cryptography and entitled Kalshi API credentials. Never logs
credentials or signs arbitrary endpoints. No order API or hosted-app integration.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .core import stamp

HOST = 'https://external-api.kalshi.com'
PATH = '/trade-api/v2/cfbenchmarks/history/values'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Refusing redirect for authenticated data request')


def headers(key, key_id, timestamp_ms):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa, ed25519
    message = (str(timestamp_ms)+'GET'+PATH).encode()
    if isinstance(key, ed25519.Ed25519PrivateKey):
        signature = key.sign(message)
    elif isinstance(key, rsa.RSAPrivateKey):
        signature = key.sign(message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                             salt_length=hashes.SHA256().digest_size), hashes.SHA256())
    else:
        raise ValueError('Supported keys: RSA or Ed25519')
    return {'KALSHI-ACCESS-KEY': key_id, 'KALSHI-ACCESS-TIMESTAMP': str(timestamp_ms),
            'KALSHI-ACCESS-SIGNATURE': base64.b64encode(signature).decode(),
            'Accept': 'application/json'}


def hour_url(hour):
    dt = stamp(hour).astimezone(timezone.utc)
    if dt.minute or dt.second or dt.microsecond:
        raise ValueError('History timestamp must be aligned to a UTC hour')
    return HOST+PATH+'?'+urlencode(dict(id='BRTI', timespan='HOUR',
        timestamp=dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')))


def load_credentials():
    key_id = os.environ.get('KALSHI_API_KEY_ID')
    key_path = os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    if not key_id or not key_path:
        raise ValueError('Configure KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH locally; do not paste secrets into chat')
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    try:
        key = load_pem_private_key(Path(key_path).read_bytes(), password=None)
    except (OSError, ValueError, TypeError):
        raise ValueError('Unable to load the configured local PEM key') from None
    return key_id, key


def optional_capture(hour, output, require_brti=False):
    """Missing optional credentials never block public/offline research.

    Configured credentials that fail are still reported as failures, not silently
    replaced with another data source. Direct capture() remains explicitly strict.
    """
    hour_url(hour)  # Validate input even when capture is deferred.
    configured = [bool(os.environ.get(name, '').strip()) for name in
                  ('KALSHI_API_KEY_ID', 'KALSHI_PRIVATE_KEY_PATH')]
    if not all(configured) and not require_brti:
        return dict(status='optional_source_not_configured', index='BRTI',
            captured=False, usable_for_training=False, api_key_required_for_app=False,
            credential_state='partial' if any(configured) else 'absent',
            available_research_source='Public Kalshi outcomes/quotes + Coinbase BTC-USD proxy',
            message='Continue without a key. BRTI capture is optional; Coinbase is not the settlement index.',
            offline_research_command='python -m helper.btc_calibration',
            setup='BRTI-SETUP.md')
    return capture(hour, output)


def capture(hour, output, credentials=None, opener=None):
    url = hour_url(hour)
    key_id, key = credentials if credentials is not None else load_credentials()
    directory = Path(output); directory.mkdir(parents=True, exist_ok=True)
    # Never overwrite a response: later revisions must be independently auditable.
    filename = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    started = datetime.now(timezone.utc).isoformat()
    request = Request(url, headers=headers(key, key_id, int(time.time()*1000)), method='GET')
    client = opener or build_opener(NoRedirect())
    try:
        with client.open(request, timeout=40) as response:
            raw = response.read(32*1024*1024+1)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError(f'BRTI access denied (HTTP {exc.code}); verify credentials and account entitlement') from None
        raise ValueError(f'BRTI request failed (HTTP {exc.code}); no substitute data used') from None
    except (URLError, TimeoutError):
        raise ValueError('BRTI network request failed; no substitute data used') from None
    if len(raw) > 32*1024*1024: raise ValueError('BRTI response exceeds capture limit')
    received = datetime.now(timezone.utc).isoformat()
    try: payload = json.loads(raw)
    except (ValueError, UnicodeError): raise ValueError('BRTI response is not JSON') from None
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), dict):
        raise ValueError('Missing documented Kalshi data envelope; payload not accepted')
    raw_path = directory/(filename+'.raw.json')
    raw_path.write_bytes(raw)
    record = dict(url=url, requested_hour=hour, request_started_at=started,
        response_received_at=received, sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw), raw_file=raw_path.name, index='BRTI',
        status='captured_unvalidated', usable_for_training=False,
        note='Raw authenticated historical response; tick schema/coverage unverified. Retrieval time is not historical availability. History may be delayed by up to 15 minutes.')
    (directory/(filename+'.meta.json')).write_text(json.dumps(record, indent=2), encoding='utf8')
    return record


def verify_capture(metadata):
    path = Path(metadata)
    record = json.loads(path.read_text(encoding='utf8'))
    filename = record['raw_file']
    if Path(filename).name != filename or '/' in filename or '\\' in filename:
        raise ValueError('Invalid raw capture filename')
    raw = (path.parent/filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['sha256'] or len(raw) != record['bytes']:
        raise ValueError('BRTI capture integrity failure')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hour', help='UTC hour, e.g. 2026-09-20T12:00:00Z')
    parser.add_argument('--output', default='data/brti-private')
    parser.add_argument('--verify', help='Verify an existing .meta.json offline')
    parser.add_argument('--require-brti', action='store_true',
                        help='Fail if optional BRTI credentials are absent (default: skip BRTI)')
    args = parser.parse_args()
    try:
        if args.verify: result = verify_capture(args.verify)
        elif args.hour: result = optional_capture(args.hour, args.output, args.require_brti)
        else: parser.error('Specify --hour or --verify')
        print(json.dumps(result, indent=2))
    except (ValueError, ImportError) as exc:
        parser.exit(2, str(exc)+'\n')
