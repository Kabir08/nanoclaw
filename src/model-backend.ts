import { readEnvFile } from './env.js';

export type ModelAuthScheme = 'api-key' | 'bearer' | 'none';
export type ModelProvider = 'anthropic' | 'llama.cpp';

export interface ModelBackendSettings {
  provider: ModelProvider;
  upstreamUrl: URL;
  authScheme: ModelAuthScheme;
  modelName?: string;
  apiKey?: string;
  bearerToken?: string;
}

const MODEL_ENV_KEYS = [
  'MODEL_PROVIDER',
  'LLM_PROVIDER',
  'ANTHROPIC_PROVIDER',
  'ANTHROPIC_BASE_URL',
  'MODEL_BASE_URL',
  'LLM_BASE_URL',
  'MODEL_NAME',
  'LLM_MODEL',
  'ANTHROPIC_MODEL',
  'ANTHROPIC_API_KEY',
  'MODEL_API_KEY',
  'LLM_API_KEY',
  'OPENAI_API_KEY',
  'GROQ_API_KEY',
  'CLAUDE_CODE_OAUTH_TOKEN',
  'ANTHROPIC_AUTH_TOKEN',
  'MODEL_AUTH_TOKEN',
  'LLM_AUTH_TOKEN',
  'ANTHROPIC_AUTH_SCHEME',
  'MODEL_AUTH_SCHEME',
  'LLM_AUTH_SCHEME',
];

export function readModelBackendEnv(): Record<string, string> {
  return readEnvFile(MODEL_ENV_KEYS);
}

function firstDefined(
  ...values: Array<string | undefined>
): string | undefined {
  for (const value of values) {
    if (value) return value;
  }
  return undefined;
}

function normalizeProvider(value?: string): ModelProvider {
  switch (value?.trim().toLowerCase()) {
    case 'llama.cpp':
    case 'llamacpp':
    case 'llama-cpp':
    case 'llama':
    case 'local':
      return 'llama.cpp';
    default:
      return 'anthropic';
  }
}

function normalizeAuthScheme(value?: string): ModelAuthScheme {
  switch (value?.trim().toLowerCase()) {
    case 'bearer':
      return 'bearer';
    case 'none':
      return 'none';
    default:
      return 'api-key';
  }
}

export function resolveModelBackendSettings(
  env: Record<string, string> = readModelBackendEnv(),
): ModelBackendSettings {
  const provider = normalizeProvider(
    firstDefined(env.MODEL_PROVIDER, env.LLM_PROVIDER, env.ANTHROPIC_PROVIDER),
  );

  const upstreamUrl = new URL(
    firstDefined(
      env.ANTHROPIC_BASE_URL,
      env.MODEL_BASE_URL,
      env.LLM_BASE_URL,
    ) || 'https://api.anthropic.com',
  );

  const authScheme = normalizeAuthScheme(
    firstDefined(
      env.MODEL_AUTH_SCHEME,
      env.LLM_AUTH_SCHEME,
      env.ANTHROPIC_AUTH_SCHEME,
    ),
  );

  const modelName = firstDefined(
    env.MODEL_NAME,
    env.LLM_MODEL,
    env.ANTHROPIC_MODEL,
  );

  const apiKey = firstDefined(
    env.ANTHROPIC_API_KEY,
    env.MODEL_API_KEY,
    env.LLM_API_KEY,
    env.OPENAI_API_KEY,
    env.GROQ_API_KEY,
  );

  const bearerToken = firstDefined(
    env.CLAUDE_CODE_OAUTH_TOKEN,
    env.ANTHROPIC_AUTH_TOKEN,
    env.MODEL_AUTH_TOKEN,
    env.LLM_AUTH_TOKEN,
  );

  return {
    provider,
    upstreamUrl,
    authScheme,
    modelName,
    apiKey,
    bearerToken,
  };
}

export function detectContainerAuthMode(
  env: Record<string, string> = readModelBackendEnv(),
): 'api-key' | 'oauth' {
  const settings = resolveModelBackendSettings(env);

  if (settings.provider === 'llama.cpp') {
    return 'api-key';
  }

  if (settings.apiKey) {
    return 'api-key';
  }

  if (settings.authScheme === 'none') {
    return 'api-key';
  }

  if (settings.authScheme === 'bearer') {
    return 'api-key';
  }

  return settings.bearerToken ? 'oauth' : 'api-key';
}

export function hasConfiguredModelBackend(envContent: string): boolean {
  return /^(CLAUDE_CODE_OAUTH_TOKEN|ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN|MODEL_API_KEY|MODEL_AUTH_TOKEN|LLM_API_KEY|LLM_AUTH_TOKEN|OPENAI_API_KEY|GROQ_API_KEY|ANTHROPIC_BASE_URL|MODEL_BASE_URL|LLM_BASE_URL|MODEL_AUTH_SCHEME|LLM_AUTH_SCHEME|ANTHROPIC_AUTH_SCHEME|MODEL_NAME|LLM_MODEL|ANTHROPIC_MODEL)=/m.test(
    envContent,
  );
}
