/**
 * Credential proxy for container isolation.
 * Containers connect here instead of directly to the upstream model API.
 * The proxy injects real credentials so containers never see them.
 *
 * Two auth modes:
 *   API key:  Proxy injects x-api-key on every request.
 *   OAuth:    Container CLI exchanges its placeholder token for a temp
 *             API key via /api/oauth/claude_cli/create_api_key.
 *             Proxy injects real OAuth token on that exchange request;
 *             subsequent requests carry the temp key which is valid as-is.
 */
import { createServer, Server } from 'http';
import { request as httpsRequest } from 'https';
import { request as httpRequest, RequestOptions } from 'http';

import { logger } from './logger.js';
import { handleLlamaCppAnthropicRequest } from './llama-cpp-shim.js';
import {
  detectContainerAuthMode,
  readModelBackendEnv,
  resolveModelBackendSettings,
} from './model-backend.js';

export type AuthMode = 'api-key' | 'oauth';

export interface ProxyConfig {
  authMode: AuthMode;
}

export function startCredentialProxy(
  port: number,
  host = '127.0.0.1',
): Promise<Server> {
  const secrets = readModelBackendEnv();
  const { provider, upstreamUrl, authScheme, modelName, apiKey, bearerToken } =
    resolveModelBackendSettings(secrets);
  const authMode: AuthMode = detectContainerAuthMode(secrets);
  const isHttps = upstreamUrl.protocol === 'https:';
  const makeRequest = isHttps ? httpsRequest : httpRequest;

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', () => {
        const body = Buffer.concat(chunks);

        if (provider === 'llama.cpp') {
          void handleLlamaCppAnthropicRequest(req, res, body, {
            provider,
            upstreamUrl,
            authScheme,
            modelName,
            apiKey,
            bearerToken,
          }).catch((err) => {
            logger.error(
              { err, url: req.url, provider },
              'Credential proxy llama.cpp translation error',
            );
            if (!res.headersSent) {
              res.writeHead(502);
              res.end('Bad Gateway');
            }
          });
          return;
        }

        const headers: Record<string, string | number | string[] | undefined> =
          {
            ...(req.headers as Record<string, string>),
            host: upstreamUrl.host,
            'content-length': body.length,
          };

        // Strip hop-by-hop headers that must not be forwarded by proxies
        delete headers['connection'];
        delete headers['keep-alive'];
        delete headers['transfer-encoding'];

        if (authMode === 'api-key') {
          // API key mode: inject the upstream credential on every request.
          delete headers['x-api-key'];
          delete headers['authorization'];

          if (authScheme === 'bearer') {
            const credential = bearerToken || apiKey;
            if (credential) {
              headers['authorization'] = `Bearer ${credential}`;
            }
          } else if (authScheme === 'api-key') {
            const credential = apiKey || bearerToken;
            if (credential) {
              headers['x-api-key'] = credential;
            }
          }
        } else {
          // OAuth mode: replace placeholder Bearer token with the real one
          // only when the container actually sends an Authorization header
          // (exchange request + auth probes). Post-exchange requests use
          // x-api-key only, so they pass through without token injection.
          if (authScheme === 'none') {
            delete headers['authorization'];
            delete headers['x-api-key'];
          } else if (headers['authorization']) {
            delete headers['authorization'];
            if (bearerToken) {
              headers['authorization'] = `Bearer ${bearerToken}`;
            }
          }
        }

        const upstream = makeRequest(
          {
            hostname: upstreamUrl.hostname,
            port: upstreamUrl.port || (isHttps ? 443 : 80),
            path: req.url,
            method: req.method,
            headers,
          } as RequestOptions,
          (upRes) => {
            res.writeHead(upRes.statusCode!, upRes.headers);
            upRes.pipe(res);
          },
        );

        upstream.on('error', (err) => {
          logger.error(
            { err, url: req.url },
            'Credential proxy upstream error',
          );
          if (!res.headersSent) {
            res.writeHead(502);
            res.end('Bad Gateway');
          }
        });

        upstream.write(body);
        upstream.end();
      });
    });

    server.listen(port, host, () => {
      logger.info({ port, host, authMode }, 'Credential proxy started');
      resolve(server);
    });

    server.on('error', reject);
  });
}

/** Detect which auth mode the host is configured for. */
export function detectAuthMode(): AuthMode {
  const secrets = readModelBackendEnv();
  return detectContainerAuthMode(secrets);
}
