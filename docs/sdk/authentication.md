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

If either `ATTO_API_KEY` or `ATTO_ORG_ID` is missing, the SDK silently
falls back to **offline mode** and logs once at `INFO`:

```
atto-qde running offline; no licence or usage emission
```

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
