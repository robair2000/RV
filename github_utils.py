"""Gedeelde GitHub-sync helpers.

- github_pull: haalt een bestand op met ETag (304-antwoorden tellen niet mee voor de
  rate limit) en een minimum interval, zodat 60s-loops de API niet leegtrekken.
- github_put: schrijft een bestand met retry bij SHA-conflicten (409/422) en logt
  elke fout in plaats van hem stil te negeren.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request

GITHUB_REPO = os.environ.get('GITHUB_REPO', 'robair2000/RV')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

_ETAGS = {}
_LAST_PULL = {}
_WARNED_404 = set()


def _auth_headers(accept, agent):
    headers = {'Accept': accept, 'User-Agent': agent, 'X-GitHub-Api-Version': '2022-11-28'}
    token = os.environ.get('GITHUB_TOKEN', '')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def github_pull(filename, min_interval=0.0, timeout=15, agent='rv-bot'):
    """True als er na afloop een (actuele) lokale kopie is, anders False (fout wordt gelogd)."""
    now = time.time()
    if min_interval > 0 and os.path.exists(filename) and now - _LAST_PULL.get(filename, 0.0) < min_interval:
        return True

    url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}?ref={GITHUB_BRANCH}'
    headers = _auth_headers('application/vnd.github.raw+json', agent)
    etag = _ETAGS.get(filename)
    if etag and os.path.exists(filename):
        headers['If-None-Match'] = etag
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            new_etag = resp.headers.get('ETag')
        with open(filename + '.tmp', 'wb') as f:
            f.write(data)
        os.replace(filename + '.tmp', filename)
        if new_etag:
            _ETAGS[filename] = new_etag
        _LAST_PULL[filename] = now
        return True
    except urllib.error.HTTPError as e:
        if e.code == 304:
            _LAST_PULL[filename] = now
            return True
        if e.code == 404:
            if filename not in _WARNED_404:
                _WARNED_404.add(filename)
                print(f"ℹ️ GitHub: {filename} bestaat (nog) niet op {GITHUB_BRANCH}.")
            return False
        print(f"⚠️ GitHub pull {filename} mislukt: HTTP {e.code}")
        return False
    except Exception as e:
        print(f"⚠️ GitHub pull {filename} mislukt: {e}")
        return False


def github_put(filename, content_bytes, message=None, agent='rv-bot', retries=3):
    """Schrijft `content_bytes` naar GitHub. Retry met verse SHA bij conflicten. True bij succes."""
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        return False
    url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}'
    headers = _auth_headers('application/vnd.github+json', agent)
    message = message or f'Auto-update {filename} [skip ci]'

    for attempt in range(1, retries + 1):
        sha = None
        try:
            req = urllib.request.Request(url + f'?ref={GITHUB_BRANCH}', headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                sha = json.loads(resp.read().decode('utf-8')).get('sha')
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"⚠️ GitHub sha-lookup {filename}: HTTP {e.code} (poging {attempt}/{retries})")
        except Exception as e:
            print(f"⚠️ GitHub sha-lookup {filename}: {e} (poging {attempt}/{retries})")

        payload = {
            'message': message,
            'content': base64.b64encode(content_bytes).decode('ascii'),
            'branch': GITHUB_BRANCH,
        }
        if sha:
            payload['sha'] = sha
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='PUT')
            with urllib.request.urlopen(req, timeout=15):
                pass
            _ETAGS.pop(filename, None)  # eigen schrijfactie -> volgende pull haalt vers op
            return True
        except urllib.error.HTTPError as e:
            if e.code in (409, 422) and attempt < retries:
                print(f"🔁 GitHub push {filename}: SHA-conflict (HTTP {e.code}), nieuwe poging {attempt + 1}/{retries}")
                time.sleep(1.5 * attempt)
                continue
            print(f"❌ GitHub push {filename} mislukt: HTTP {e.code}")
            return False
        except Exception as e:
            print(f"❌ GitHub push {filename} mislukt: {e}")
            if attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            return False
    return False
