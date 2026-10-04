import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from helper.brti_capture import PATH, NoRedirect, capture, headers, hour_url, load_credentials, optional_capture, verify_capture

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa, padding
    from cryptography.hazmat.primitives import hashes
except ImportError:
    ed25519 = None


class BRTICaptureTests(unittest.TestCase):
    def test_optional_key_skips_network_and_file_creation(self):
        for env in [{}, {'KALSHI_API_KEY_ID': 'partial'}]:
            with patch.dict(os.environ, env, clear=True), patch('helper.brti_capture.capture') as fetch:
                with tempfile.TemporaryDirectory() as d:
                    result = optional_capture('2026-09-20T12:00:00Z', d)
                    self.assertEqual(result['status'], 'optional_source_not_configured')
                    self.assertFalse(result['captured'])
                    self.assertFalse(result['usable_for_training'])
                    self.assertEqual(list(Path(d).iterdir()), [])
                    fetch.assert_not_called()

    def test_explicit_required_mode_and_configured_failure_are_not_hidden(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Configure'):
                optional_capture('2026-09-20T12:00:00Z', 'unused', require_brti=True)
        with patch.dict(os.environ, {'KALSHI_API_KEY_ID':'id', 'KALSHI_PRIVATE_KEY_PATH':'key'}, clear=True):
            with patch('helper.brti_capture.capture', side_effect=ValueError('access denied')) as fetch:
                with self.assertRaisesRegex(ValueError, 'access denied'):
                    optional_capture('2026-09-20T12:00:00Z', 'unused')
                fetch.assert_called_once()

    def test_hour_alignment_and_naive_timestamp_rejection(self):
        self.assertIn('id=BRTI', hour_url('2026-09-20T12:00:00Z'))
        self.assertEqual(hour_url('2026-09-20T08:00:00-04:00'), hour_url('2026-09-20T12:00:00Z'))
        for value in ['2026-09-20T12:01:00Z', '2026-09-20T12:00:00']:
            with self.assertRaises(ValueError): hour_url(value)

    def test_missing_credentials_and_redirect_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Configure'): load_credentials()
        with self.assertRaisesRegex(ValueError, 'redirect'):
            NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.test')

    def test_verifier_rejects_escaping_raw_paths(self):
        with tempfile.TemporaryDirectory() as d:
            meta = Path(d, 'capture.meta.json')
            meta.write_text(json.dumps({'raw_file': '../outside.json'}))
            with self.assertRaisesRegex(ValueError, 'filename'): verify_capture(meta)

    @unittest.skipIf(ed25519 is None, 'Optional cryptography not installed')
    def test_invalid_response_never_saved_as_accepted_capture(self):
        for raw in [b'not-json', b'{"data":[]}', b'{"error":"failed"}']:
            class Opener:
                def open(self, request, timeout): return io.BytesIO(raw)
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    capture('2026-09-20T12:00:00Z', d, ('test',ed25519.Ed25519PrivateKey.generate()), Opener())
                self.assertEqual(list(Path(d).iterdir()), [])

    @unittest.skipIf(ed25519 is None, 'Optional cryptography not installed')
    def test_real_rsa_and_ed25519_signature_verification(self):
        for key in [ed25519.Ed25519PrivateKey.generate(), rsa.generate_private_key(public_exponent=65537, key_size=2048)]:
            h = headers(key, 'test-key-id', 1234)
            signature = base64.b64decode(h['KALSHI-ACCESS-SIGNATURE'])
            message = ('1234GET'+PATH).encode()
            if isinstance(key, ed25519.Ed25519PrivateKey): key.public_key().verify(signature, message)
            else: key.public_key().verify(signature, message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())

    @unittest.skipIf(ed25519 is None, 'Optional cryptography not installed')
    def test_capture_integrity_no_credentials_persisted_and_no_training_claim(self):
        key = ed25519.Ed25519PrivateKey.generate()
        class Opener:
            def open(self, request, timeout):
                assert request.get_method() == 'GET'
                assert request.full_url.startswith('https://external-api.kalshi.com'+PATH+'?')
                return io.BytesIO(b'{"data":{"payload":{}}}')
        with tempfile.TemporaryDirectory() as d:
            result = capture('2026-09-20T12:00:00Z', d, ('secret-id',key), Opener())
            self.assertFalse(result['usable_for_training'])
            meta = next(Path(d).glob('*.meta.json'))
            self.assertEqual(verify_capture(meta), result)
            self.assertNotIn('secret-id', meta.read_text())
            Path(d,result['raw_file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'integrity'): verify_capture(meta)

    @unittest.skipIf(ed25519 is None, 'Optional cryptography not installed')
    def test_access_denial_does_not_save_or_fallback(self):
        class Opener:
            def open(self, request, timeout):
                raise HTTPError(request.full_url,403,'sensitive response',{},None)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'entitlement'):
                capture('2026-09-20T12:00:00Z',d,('test',ed25519.Ed25519PrivateKey.generate()),Opener())
            self.assertEqual(list(Path(d).iterdir()), [])
