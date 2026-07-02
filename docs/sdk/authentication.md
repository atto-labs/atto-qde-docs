# Authentication

The SDK authenticates to the console with an **org-scoped API key** plus
the org's ID. Both are configured through environment variables and read
once when `build_default_runtime()` runs (which `AttoEngine.__init__`
calls by default).

## Environment variables

| Variable                | Required for hosted mode | Default                        | Notes                                                        |
| ----------------------- | ------------------------ | ------------------------------ | ------------------------------------------------------------ |
| `ATTO_API_KEY`          | Yes                      | —                              | Org-scoped API key. Must include the `core` product scope.   |
| `ATTO_ORG_ID`           | Yes                      | —                              | Identifier of the calling organisation.                      |
| `ATTO_CONSOLE_BASE_URL` | No                       | `https://console.atto-qde.com` | Override for staging or self-hosted console deployments.     |
| `ATTO_API_BASE`         | No                       | (mirrored from base URL)       | Lower-level override read by the HTTP client. Rarely needed. |

If either `ATTO_API_KEY` or `ATTO_ORG_ID` is missing, the SDK falls back to
**offline mode** and logs once at `INFO`:

```
atto-qde running offline (free-tier cap: 100 decisions/month)
```

In offline mode, a local free-tier token is used. The compiled binary
enforces a 100-decision-per-month cap locally. Once the cap is reached the
SDK raises `PlanUpgradeRequired` and requires a console round-trip to
continue.

## Signed licence tokens

In hosted mode, the console issues a **signed JWT** (Ed25519) when the SDK
calls `POST /api/v1/licence/{org_id}/token`. The token:

- Is cached in-process for 24 hours (the token's TTL).
- Contains the org ID, product scope, decision cap, and expiry.
- Is verified offline against a public key embedded in the compiled binary.
- Enables a configurable **grace period** (default 72 h) during which the
  SDK continues to operate if the console is unreachable, as long as the
  cached token has not expired.

After the grace period elapses without a successful token refresh, the SDK
raises `LicenceError`.

## Issuing a key

1. Sign in to [console.atto-qde.com](https://console.atto-qde.com).
2. Go to **API keys** → **Create key**.
3. In the **Products** multi-select, tick **`core`**. (Adaptive-only
   keys cannot make Core SDK calls — the licence endpoint will return
   `403 forbidden_product`.)
4. Copy the key. The plaintext is shown once.

A single key can be scoped to one or both products. Rotate by creating
a new key, deploying it, and revoking the old one from the same screen.

## How requests are signed

The SDK adds two headers to every console request:

```
X-Atto-Api-Key:  atto_live_...
X-Atto-Org-Id:   org_...
```

There is no per-request signing; transport security is TLS to
`console.atto-qde.com`.

## Energy-industry signups

Self-service signup from a known energy-industry email domain is held
for manual review. The signup form returns:

> Thanks — we manually review signups associated with the energy
> industry. Your account will be activated within one business day.

You will receive an email when the review completes. After approval,
sign in and create an API key as above.

This affects only signup; existing keys continue to work.
