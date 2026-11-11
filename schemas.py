"""Tool schemas the model sees."""

PRACTICE = {
    "name": "saywhat_practice",
    "description": (
        "Pronunciation practice: the user says a sentence and saywhat compares what they meant with "
        "what speech recognition heard, word by word. action 'start' sets the sentence to practise "
        "(and plays a spoken model of it); the user then records themselves with voice mode, and each "
        "recording is a practice attempt. action 'score' scores the latest recording (or audio_path) "
        "against the sentence: per-word verdicts (clear / unclear / misheard / missed), concrete tips, "
        "a 0-100 clarity score and the change since the last attempt, and it plays a slow model of the "
        "problem words. action 'stop' ends practice and summarises progress. Use when the user wants to "
        "practise pronunciation or check how clearly a phrase comes across."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "score", "stop"]},
            "text": {"type": "string", "description": "The sentence the user means to say (required for start)."},
            "audio_path": {"type": "string",
                           "description": "Optional local audio file to score instead of the latest voice recording."},
        },
        "required": ["action"],
    },
}

PROGRESS = {
    "name": "saywhat_progress",
    "description": (
        "The user's speaking profile from saywhat: practised phrases with score trends, the words they "
        "most often aren't understood on, the vocabulary saywhat has learned, and mishearings it now "
        "corrects automatically. Use when the user asks how their speaking or dictation is improving."),
    "parameters": {"type": "object", "properties": {}},
}

DICTIONARY = {
    "name": "saywhat_dictionary",
    "description": (
        "Manage the user's personal voice dictionary, which biases speech recognition toward their "
        "vocabulary. action 'list' shows terms and learned mishearings; 'add' adds a term (e.g. a "
        "product or library name); 'remove' forgets a term; 'teach' tells saywhat that what it hears "
        "as heard_as means term (offered first from then on, and automatic once the user confirms it)."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "add", "remove", "teach"]},
            "term": {"type": "string", "description": "The word or phrase as it should be written."},
            "heard_as": {"type": "string", "description": "For teach: what recognition produces instead."},
        },
        "required": ["action"],
    },
}
