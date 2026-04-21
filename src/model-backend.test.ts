import { describe, expect, it } from 'vitest';

import {
  detectContainerAuthMode,
  resolveModelBackendSettings,
} from './model-backend.js';

describe('model-backend', () => {
  it('defaults to anthropic settings and api-key mode', () => {
    const settings = resolveModelBackendSettings({});

    expect(settings.provider).toBe('anthropic');
    expect(settings.modelName).toBeUndefined();
    expect(detectContainerAuthMode({})).toBe('api-key');
  });

  it('recognizes llama.cpp provider and model name', () => {
    const settings = resolveModelBackendSettings({
      MODEL_PROVIDER: 'llama.cpp',
      MODEL_BASE_URL: 'http://127.0.0.1:8080',
      MODEL_NAME: 'llama3.1',
    });

    expect(settings.provider).toBe('llama.cpp');
    expect(settings.modelName).toBe('llama3.1');
    expect(settings.upstreamUrl.toString()).toBe('http://127.0.0.1:8080/');
    expect(
      detectContainerAuthMode({
        MODEL_PROVIDER: 'llama.cpp',
        MODEL_BASE_URL: 'http://127.0.0.1:8080',
        MODEL_NAME: 'llama3.1',
      }),
    ).toBe('api-key');
  });

  it('falls back to oauth mode when only Claude OAuth credentials are present', () => {
    expect(
      detectContainerAuthMode({
        CLAUDE_CODE_OAUTH_TOKEN: 'sk-ant-oat01-example',
      }),
    ).toBe('oauth');
  });
});
