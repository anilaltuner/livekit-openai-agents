from typing import List, Dict, Any

from livekit.agents import ChatContext

def generate_context(message_history: dict, context: List[Dict[str, Any]], user_message: str) -> str:
    """
    Generates a ChatContext from a message history and context.

    Args:
        message_history: The message history to use for the ChatContext.
        context: The context to use for the ChatContext.
        user_message: The user message

    Returns:
        A context string that can be used to initialize conversation.
    """
    return f"""
    User Message:
    {user_message}
    
    
    Message History:
    {message_history}

    Context:
    {context}
    """

def extract_last_user_message(chat_ctx: ChatContext) -> str:
    """
    Extracts the content of the last user message from the ChatContext.
    Returns "Hi!" as a default if no user message is found.
    """
    try:
        items = chat_ctx.to_dict().get("items", [])
    except AttributeError:
        items = getattr(chat_ctx, 'messages', [])
        if not isinstance(items, list): # If messages is not a list, fallback
            items = []
        else: # If it is a list, ensure items are dict-like for consistent processing
            new_items = []
            for item in items:
                if hasattr(item, 'role') and hasattr(item, 'content'):
                    new_items.append({'role': str(item.role), 'content': item.content})
            items = new_items


    # Traverse in reverse to find the last user message
    for item in reversed(items):
        # Role can be an enum or string, handle both
        role = item.get("role")
        if hasattr(role, 'value'): # If it's an enum like livekit.protocol.ChatRole
            role = role.value
        
        if str(role).lower() == "user": # Compare as string
            content = item.get("content", "")
            if isinstance(content, list): # Handle multi-part content
                # Extract text parts from a list of content parts (e.g., text and image)
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part: # Common structure for text part
                        text_parts.append(part["text"])
                return " ".join(text_parts) if text_parts else ""
            elif isinstance(content, str):
                return content
    return "Hi!" # Default greeting if no user message found 