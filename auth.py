"""Moved auth helper into src package."""
# ...existing code...
import os
import sys
from getpass import getpass
import uuid
from typing import Optional

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None

try:
    import bcrypt
except Exception:
    bcrypt = None


def _get_supabase_client():
    # Hardcoded Supabase URL / Key (edit these values below)
    # WARNING: Hardcoding keys in source is insecure. Keep this file private.
    SUPABASE_URL = 'https://blmejkvzmxtrfeckqqod.supabase.co'  # <-- replace with your Supabase URL
    SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJsbWVqa3Z6bXh0cmZlY2txcW9kIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTY1NjY5NTgsImV4cCI6MjA3MjE0Mjk1OH0.BeeKo6Uz4KpBhzsMDdpjQDhGy8JMK41cKSDLXyNczfA'                 # <-- replace with your Supabase anon/service key

    url = SUPABASE_URL
    key = SUPABASE_KEY
    if not create_client:
        raise RuntimeError('supabase package is not installed')
    return create_client(url, key)


def sign_in(email: str, password: str) -> Optional[dict]:
    # Note: bcrypt is optional if your users table stores plaintext `password` values
    # (insecure). Only require bcrypt when verifying bcrypt hashes stored in
    # `password_hash`.
    client = _get_supabase_client()
    try:
        res = client.table('users').select('*').eq('email', email).limit(1).execute()
    except Exception as e:
        raise RuntimeError(f"Supabase query failed: {e}")

    # Robust error checking across different client versions / response shapes
    err = None
    if hasattr(res, 'error') and res.error:
        err = res.error
    elif isinstance(res, dict) and res.get('error'):
        err = res.get('error')
    elif hasattr(res, 'status_code') and getattr(res, 'status_code', 200) >= 400:
        err = getattr(res, 'status_text', f"HTTP {getattr(res, 'status_code')}")

    if err:
        raise RuntimeError(f"Supabase error: {err}")

    # Extract data rows from different response formats
    data = None
    if hasattr(res, 'data'):
        data = res.data
    elif isinstance(res, (list, tuple)) and len(res) > 0:
        # some clients return (data, count)
        data = res[0]
    elif isinstance(res, dict) and 'data' in res:
        data = res.get('data')
    else:
        data = res

    if not data:
        return None

    # Normalize rows to a list
    if isinstance(data, dict) and ('data' in data or 'rows' in data):
        rows = data.get('data') or data.get('rows') or []
    else:
        rows = data

    if not rows:
        return None

    # rows may already be a single dict or a list
    if isinstance(rows, (list, tuple)):
        user = rows[0]
    elif isinstance(rows, dict):
        user = rows
    else:
        return None

    # Enforce that the user account is approved. The `approved` column in
    # Supabase may be a boolean, integer, or string. Treat common truthy
    # representations as approval; otherwise deny sign-in.
    try:
        _approved_val = user.get('approved')
    except Exception:
        _approved_val = None

    def _is_approved(v) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        return s in ('1', 'true', 't', 'yes', 'y')

    if not _is_approved(_approved_val):
        # Not approved: deny sign-in
        return None

    # Enforce optional account expiration. If `end_date` is set and in the past,
    # deny sign-in by returning None.
    try:
        end_date_val = user.get('end_date') or user.get('expires_at') or None
    except Exception:
        end_date_val = None

    if end_date_val:
        from datetime import datetime, timezone
        parsed = None
        try:
            # handle common types: datetime, string, numeric timestamp
            if hasattr(end_date_val, 'tzinfo') and getattr(end_date_val, 'tzinfo', None) is not None:
                parsed = end_date_val
            elif isinstance(end_date_val, (int, float)):
                parsed = datetime.fromtimestamp(float(end_date_val), tz=timezone.utc)
            elif isinstance(end_date_val, str):
                # Prefer dateutil if available; otherwise fall back to fromisoformat
                try:
                    from dateutil import parser as _parser  # type: ignore
                    parsed = _parser.isoparse(end_date_val)
                except Exception:
                    try:
                        parsed = datetime.fromisoformat(end_date_val.replace('Z', '+00:00'))
                    except Exception:
                        parsed = None
        except Exception:
            parsed = None

        if parsed is not None:
            # ensure timezone-aware in UTC for comparison
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if parsed <= now:
                # expired
                return None

    # Collect local MACs from interfaces (wifi/ethernet) and normalize
    try:
        import psutil
    except Exception:
        psutil = None

    def _get_local_macs() -> set:
        """Return a set of normalized local MACs (lowercase aa:bb:cc:dd:ee:ff).
        Try psutil.net_if_addrs(), fallback to uuid.getnode()."""
        result = set()
        try:
            if psutil is not None:
                for ifname, addrs in psutil.net_if_addrs().items():
                    for addr in addrs:
                        s = getattr(addr, 'address', None)
                        if not s:
                            continue
                        # Extract hex digits and normalize
                        import re
                        hexchars = ''.join(re.findall(r'[0-9a-fA-F]', str(s)))
                        if len(hexchars) >= 12:
                            hex12 = hexchars[-12:]
                            mac = ':'.join(hex12[i:i+2] for i in range(0, 12, 2)).lower()
                            result.add(mac)
            if not result:
                node = uuid.getnode()
                mac_hex = f"{node:012x}"
                mac = ':'.join(mac_hex[i:i+2] for i in range(0, 12, 2)).lower()
                result.add(mac)
        except Exception:
            try:
                node = uuid.getnode()
                mac_hex = f"{node:012x}"
                mac = ':'.join(mac_hex[i:i+2] for i in range(0, 12, 2)).lower()
                result.add(mac)
            except Exception:
                pass
        return result

    def _normalize_db_mac(val) -> str:
        """Normalize DB-stored mac to lowercase colon-separated format if possible."""
        if not val:
            return ''
        try:
            s = str(val)
        except Exception:
            return ''
        import re
        hexchars = ''.join(re.findall(r'[0-9a-fA-F]', s))
        if len(hexchars) < 12:
            return ''
        hex12 = hexchars[-12:]
        mac = ':'.join(hex12[i:i+2] for i in range(0, 12, 2))
        return mac.lower()

    # If the DB record contains a mac_address, allow login if it matches any local MAC.
    db_mac_raw = user.get('mac_address') or user.get('mac')
    db_mac_norm = _normalize_db_mac(db_mac_raw)
    if db_mac_norm:
        local_macs = _get_local_macs()
        try:
            if os.getenv('AUTH_DEBUG') == '1':
                print(f"[AUTH_DEBUG] local_macs={sorted(local_macs)} db_mac_norm={db_mac_norm}")
        except Exception:
            pass
        if db_mac_norm in local_macs:
            user.pop('password', None)
            user.pop('password_hash', None)
            user.pop('mac_address', None)
            user.pop('mac', None)
            return user
        else:
            return None

    # First, support plaintext `password` column if present (convenience/dev only)
    plain = user.get('password')
    if plain is not None:
        # Normalize stored plaintext to str
        if isinstance(plain, (bytes, bytearray)):
            try:
                plain = plain.decode('utf-8')
            except Exception:
                plain = plain.decode('latin1', errors='ignore')

        # Normalize provided password to str
        pw_str = password.decode('utf-8') if isinstance(password, (bytes, bytearray)) else str(password)

        if pw_str == plain:
            # Remove sensitive field before returning
            user.pop('password', None)
            user.pop('password_hash', None)
            return user
        return None

    # Fallback: verify bcrypt password_hash if present
    hashed = user.get('password_hash')
    if not hashed:
        return None

    if bcrypt is None:
        raise RuntimeError('bcrypt is required for password verification when using password_hash')

    if isinstance(hashed, str):
        hashed = hashed.encode('utf-8')
    if isinstance(password, str):
        password = password.encode('utf-8')

    try:
        ok = bcrypt.checkpw(password, hashed)
    except Exception:
        return None

    if ok:
        user.pop('password_hash', None)
        return user
    return None


def require_signin(interactive: bool = True) -> dict:
    if create_client is None or bcrypt is None:
        print('Authentication dependencies missing. Please install requirements (supabase, bcrypt).')
        if interactive:
            sys.exit(1)
        raise RuntimeError('Missing auth dependencies')

    print('\n=== Authentication Required ===')
    email = input('Email: ').strip()
    password = getpass('Password: ')

    try:
        user = sign_in(email, password)
    except Exception as e:
        print(f'Auth error: {e}')
        if interactive:
            sys.exit(1)
        raise

    if not user:
        print('Invalid credentials.')
        if interactive:
            sys.exit(1)
        raise RuntimeError('Invalid credentials')

    print(f"Signed in as: {user.get('email')}")
    return user
