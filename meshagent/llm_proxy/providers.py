OPENAI_ALLOWED_EXACT_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/responses/compact",
        "/v1/responses/input_tokens",
        "/v1/embeddings",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/models",
        "/v1/realtime",
    }
)
OPENAI_ALLOWED_PREFIXES = (
    "/v1/models/",
    "/v1/images/",
    "/v1/realtime/",
)
OPENAI_ALLOWED_WEBSOCKET_PATHS = frozenset({"/v1/realtime", "/v1/responses"})

ANTHROPIC_ALLOWED_EXACT_PATHS = frozenset(
    {
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/complete",
        "/v1/models",
    }
)
ANTHROPIC_ALLOWED_PREFIXES = (
    "/v1/models/",
    "/v1/messages/batches",
)


def is_openai_path_allowed(api_path: str) -> bool:
    return api_path in OPENAI_ALLOWED_EXACT_PATHS or api_path.startswith(
        OPENAI_ALLOWED_PREFIXES
    )


def is_anthropic_path_allowed(api_path: str) -> bool:
    return api_path in ANTHROPIC_ALLOWED_EXACT_PATHS or api_path.startswith(
        ANTHROPIC_ALLOWED_PREFIXES
    )


def is_openai_websocket_path_allowed(api_path: str) -> bool:
    return api_path in OPENAI_ALLOWED_WEBSOCKET_PATHS
