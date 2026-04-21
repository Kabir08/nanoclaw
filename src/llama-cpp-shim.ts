import type { IncomingMessage, ServerResponse } from 'http';

import { logger } from './logger.js';
import type { ModelBackendSettings } from './model-backend.js';

interface AnthropicRequestBody {
  model?: string;
  max_tokens?: number;
  system?: unknown;
  messages?: Array<Record<string, unknown>>;
  tools?: Array<Record<string, unknown>>;
  tool_choice?: unknown;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  stream?: boolean;
}

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
}

interface OpenAIResponse {
  id?: string;
  model?: string;
  choices?: Array<{
    finish_reason?: string | null;
    message?: {
      role?: string;
      content?: string | null;
      tool_calls?: Array<{
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function toText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value
      .map((entry) => toText(entry))
      .filter(Boolean)
      .join('\n');
  }
  if (isRecord(value)) {
    if (typeof value.text === 'string') return value.text;
    if (typeof value.content === 'string') return value.content;
    return JSON.stringify(value, null, 2);
  }
  return '';
}

function normalizeSystemPrompt(system: unknown): string | undefined {
  const text = toText(system).trim();
  return text || undefined;
}

function normalizeToolChoice(choice: unknown): unknown {
  if (choice == null) return undefined;
  if (typeof choice === 'string') {
    if (choice === 'auto' || choice === 'none' || choice === 'required') {
      return choice;
    }
    if (choice === 'any') {
      return 'required';
    }
  }
  if (
    isRecord(choice) &&
    choice.type === 'tool' &&
    typeof choice.name === 'string'
  ) {
    return {
      type: 'function',
      function: { name: choice.name },
    };
  }
  return undefined;
}

function translateTools(
  tools: AnthropicRequestBody['tools'],
): unknown[] | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;

  return tools
    .map((tool) => {
      const name = typeof tool.name === 'string' ? tool.name : undefined;
      if (!name) return undefined;
      return {
        type: 'function',
        function: {
          name,
          description:
            typeof tool.description === 'string' ? tool.description : undefined,
          parameters:
            isRecord(tool.input_schema) &&
            Object.keys(tool.input_schema).length > 0
              ? tool.input_schema
              : { type: 'object', properties: {} },
        },
      };
    })
    .filter(Boolean);
}

function translateMessage(message: Record<string, unknown>): OpenAIMessage[] {
  const role = message.role;
  const content = message.content;

  if (role === 'system') {
    const text = toText(content).trim();
    return text ? [{ role: 'system', content: text }] : [];
  }

  if (role === 'assistant') {
    const assistantParts = Array.isArray(content) ? content : [content];
    const textParts: string[] = [];
    const toolCalls: OpenAIMessage['tool_calls'] = [];

    for (const part of assistantParts) {
      if (!isRecord(part)) {
        const text = toText(part).trim();
        if (text) textParts.push(text);
        continue;
      }

      if (part.type === 'tool_use' && typeof part.name === 'string') {
        toolCalls.push({
          id:
            typeof part.id === 'string'
              ? part.id
              : `tool_${toolCalls.length + 1}`,
          type: 'function',
          function: {
            name: part.name,
            arguments: JSON.stringify(part.input ?? {}),
          },
        });
        continue;
      }

      const text = toText(part).trim();
      if (text) textParts.push(text);
    }

    const messageBlock: OpenAIMessage = {
      role: 'assistant',
      content: textParts.join('\n\n') || null,
    };

    if (toolCalls.length > 0) {
      messageBlock.tool_calls = toolCalls;
    }

    return [messageBlock];
  }

  if (role === 'user') {
    const userParts = Array.isArray(content) ? content : [content];
    const output: OpenAIMessage[] = [];

    let pendingText: string[] = [];
    const flushText = () => {
      const text = pendingText.join('\n\n').trim();
      pendingText = [];
      if (text) {
        output.push({ role: 'user', content: text });
      }
    };

    for (const part of userParts) {
      if (isRecord(part) && part.type === 'tool_result') {
        flushText();
        output.push({
          role: 'tool',
          tool_call_id:
            typeof part.tool_use_id === 'string'
              ? part.tool_use_id
              : typeof part.id === 'string'
                ? part.id
                : 'tool_result',
          content: toText(part.content).trim() || toText(part).trim(),
          name: typeof part.name === 'string' ? part.name : undefined,
        });
        continue;
      }

      const text = toText(part).trim();
      if (text) pendingText.push(text);
    }

    flushText();
    return output;
  }

  const text = toText(content).trim();
  return text ? [{ role: 'user', content: text }] : [];
}

function translateAnthropicToOpenAI(
  body: AnthropicRequestBody,
  settings: ModelBackendSettings,
): Record<string, unknown> {
  const messages: OpenAIMessage[] = [];

  const systemPrompt = normalizeSystemPrompt(body.system);
  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }

  for (const message of body.messages ?? []) {
    if (!isRecord(message)) continue;
    messages.push(...translateMessage(message));
  }

  return {
    model: settings.modelName || body.model || 'llama.cpp',
    messages,
    stream: false,
    temperature: body.temperature,
    top_p: body.top_p,
    max_tokens: body.max_tokens,
    stop: body.stop_sequences,
    tools: translateTools(body.tools),
    tool_choice: normalizeToolChoice(body.tool_choice),
  };
}

function mapFinishReason(reason?: string | null): string {
  switch (reason) {
    case 'tool_calls':
      return 'tool_use';
    case 'length':
      return 'max_tokens';
    default:
      return 'end_turn';
  }
}

function serializeAnthropicSseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function synthesizeAnthropicResponse(
  response: OpenAIResponse,
  body: AnthropicRequestBody,
  settings: ModelBackendSettings,
): string {
  const choice = response.choices?.[0];
  const message = choice?.message;
  const model =
    response.model || settings.modelName || body.model || 'llama.cpp';
  const messageId = response.id || `msg_${Date.now()}`;
  const promptTokens = response.usage?.prompt_tokens || 0;
  const completionTokens = response.usage?.completion_tokens || 0;

  const contentBlocks: Array<Record<string, unknown>> = [];

  const text =
    typeof message?.content === 'string' ? message.content : undefined;
  if (text) {
    contentBlocks.push({ type: 'text', text });
  }

  for (const toolCall of message?.tool_calls ?? []) {
    if (!toolCall || typeof toolCall !== 'object') continue;
    const functionName = toolCall.function?.name;
    if (typeof functionName !== 'string') continue;
    const argumentsJson = toolCall.function?.arguments || '{}';
    let input: unknown = {};
    try {
      input = JSON.parse(argumentsJson);
    } catch {
      input = argumentsJson;
    }

    contentBlocks.push({
      type: 'tool_use',
      id: toolCall.id || `tool_${contentBlocks.length + 1}`,
      name: functionName,
      input,
    });
  }

  const stopReason = mapFinishReason(choice?.finish_reason ?? undefined);

  let sse = '';
  sse += serializeAnthropicSseEvent('message_start', {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: promptTokens,
        output_tokens: 0,
      },
    },
  });

  contentBlocks.forEach((contentBlock, index) => {
    sse += serializeAnthropicSseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: contentBlock,
    });

    if (contentBlock.type === 'tool_use') {
      sse += serializeAnthropicSseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: {
          type: 'input_json_delta',
          partial_json: JSON.stringify(contentBlock.input ?? {}),
        },
      });
    } else if (contentBlock.type === 'text') {
      sse += serializeAnthropicSseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: {
          type: 'text_delta',
          text: contentBlock.text ?? '',
        },
      });
    }

    sse += serializeAnthropicSseEvent('content_block_stop', {
      type: 'content_block_stop',
      index,
    });
  });

  sse += serializeAnthropicSseEvent('message_delta', {
    type: 'message_delta',
    delta: {
      stop_reason: stopReason,
      stop_sequence: null,
    },
    usage: {
      output_tokens: completionTokens,
    },
  });

  sse += serializeAnthropicSseEvent('message_stop', {
    type: 'message_stop',
  });

  return sse;
}

export async function handleLlamaCppAnthropicRequest(
  req: IncomingMessage,
  res: ServerResponse,
  body: Buffer,
  settings: ModelBackendSettings,
): Promise<boolean> {
  if (!req.url?.startsWith('/v1/messages')) {
    return false;
  }

  let parsedBody: AnthropicRequestBody;
  try {
    parsedBody = JSON.parse(body.toString('utf-8')) as AnthropicRequestBody;
  } catch (err) {
    logger.warn(
      { err, url: req.url },
      'Failed to parse llama.cpp request body',
    );
    res.writeHead(400, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'Invalid JSON body' }));
    return true;
  }

  const upstreamUrl = new URL('/v1/chat/completions', settings.upstreamUrl);
  const upstreamHeaders: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
  };

  const credential = settings.bearerToken || settings.apiKey;
  if (credential) {
    upstreamHeaders.authorization = `Bearer ${credential}`;
  }

  const upstreamResponse = await fetch(upstreamUrl, {
    method: 'POST',
    headers: upstreamHeaders,
    body: JSON.stringify(translateAnthropicToOpenAI(parsedBody, settings)),
  });

  if (!upstreamResponse.ok) {
    const errorText = await upstreamResponse.text();
    logger.warn(
      {
        url: req.url,
        status: upstreamResponse.status,
        errorText: errorText.slice(0, 500),
      },
      'llama.cpp upstream returned an error',
    );

    res.writeHead(upstreamResponse.status, {
      'content-type':
        upstreamResponse.headers.get('content-type') || 'text/plain',
    });
    res.end(errorText);
    return true;
  }

  const responseJson = (await upstreamResponse.json()) as OpenAIResponse;
  const sse = synthesizeAnthropicResponse(responseJson, parsedBody, settings);

  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
  });
  res.end(sse);
  return true;
}
