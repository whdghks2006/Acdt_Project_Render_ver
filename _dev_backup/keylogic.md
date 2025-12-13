# Key Logic: Google OAuth on Hugging Face Spaces
# 핵심 로직: Hugging Face Spaces에서의 구글 OAuth 인증

This document explains the specific implementation details required to make Google OAuth work on Hugging Face Spaces, which sits behind a reverse proxy.
이 문서는 리버스 프록시 뒤에 있는 Hugging Face Spaces 환경에서 Google OAuth를 정상적으로 작동시키기 위해 필요한 구체적인 구현 내용을 설명합니다.

---

## 1. Session Middleware Configuration (세션 미들웨어 설정)

**File:** `main.py`

```python
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=True,      # Critical for HF Spaces (HF Spaces는 HTTPS를 강제함)
    same_site='none',     # Required for cross-site cookies in iframe/proxy (iframe/프록시 환경에서 크로스 사이트 쿠키 허용)
    path='/',
    max_age=3600
)
```

- **English**: `https_only=True` and `same_site='none'` are essential because HF Spaces serves content via HTTPS and often embeds the app in an iframe or behind a proxy. Without these, the session cookie will be dropped, leading to "404 Not Found" or "Unauthorized" errors after login.
- **Korean**: `https_only=True`와 `same_site='none'` 설정은 필수입니다. HF Spaces는 HTTPS를 통해 서비스를 제공하며, 종종 앱을 iframe이나 프록시 뒤에서 실행하기 때문입니다. 이 설정이 없으면 세션 쿠키가 유지되지 않아 로그인 후 "404 Not Found" 또는 "Unauthorized" 오류가 발생합니다.

---

## 2. Login Endpoint (`/login`)

**File:** `main.py`

```python
@app.get('/login')
async def login(request: Request):
    # Hardcode the redirect URI to match the HF Space URL exactly.
    # HF Space의 URL과 정확히 일치하도록 리다이렉트 URI를 하드코딩합니다.
    fixed_redirect_uri = "https://snowmang-ai-scheduler-g14.hf.space/auth/callback"
    return await oauth.google.authorize_redirect(request, fixed_redirect_uri)
```

- **English**: We explicitly define `fixed_redirect_uri` to ensure the redirect goes to the correct public HTTPS URL of the Space, not the internal container IP (`http://10.x.x.x`). `request.url_for` might return the internal HTTP URL, causing a mismatch.
- **Korean**: `fixed_redirect_uri`를 명시적으로 정의하여 리다이렉트가 내부 컨테이너 IP(`http://10.x.x.x`)가 아닌 Space의 정확한 공용 HTTPS URL로 이동하도록 보장합니다. `request.url_for`를 사용하면 내부 HTTP URL을 반환하여 불일치 오류가 발생할 수 있습니다.

---

## 3. Auth Callback Endpoint (`/auth/callback`)

**File:** `main.py`

```python
@app.get('/auth/callback')
async def auth(request: Request):
    try:
        # DO NOT pass redirect_uri here.
        # 여기서는 redirect_uri를 전달하지 마십시오.
        token = await oauth.google.authorize_access_token(request)
        ...
```

- **English**: In the `auth` function, we call `authorize_access_token(request)` **WITHOUT** the `redirect_uri` argument. Passing the URI here often causes a `MismatchingStateError` or `400 Bad Request` in the HF environment because the library tries to validate the request URL against the passed URI and fails due to proxy headers. Letting the library handle it automatically (or relying on the state) works reliably here.
- **Korean**: `auth` 함수에서는 `redirect_uri` 인자 **없이** `authorize_access_token(request)`를 호출합니다. 여기서 URI를 전달하면 라이브러리가 요청 URL과 전달된 URI를 대조하는 과정에서 프록시 헤더 문제로 인해 `MismatchingStateError` 또는 `400 Bad Request` 오류가 자주 발생합니다. 라이브러리가 자동으로 처리하도록 두는 것(또는 state에 의존하는 것)이 이 환경에서는 안정적으로 작동합니다.

---

## Summary (요약)

1.  **Middleware**: `https_only=True`, `same_site='none'`
2.  **Login**: Use explicit `fixed_redirect_uri` (HTTPS). (로그인: 명시적인 HTTPS URI 사용)
3.  **Callback**: Do NOT pass `redirect_uri` to `authorize_access_token`. (콜백: 토큰 요청 시 URI 전달 금지)
