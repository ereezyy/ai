import os
import json


class GroqOmniscience:
    """The all-knowing Groq-powered natural language processing engine."""

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            # We'll mock it for now if no key is provided, but print a terrifying warning
            self._is_mock = True
        else:
            self._is_mock = False
            try:
                from groq import Groq

                self.client = Groq(api_key=self.api_key)
            except ImportError:
                self._is_mock = True

    def parse_intent(self, user_command):
        """Pierces the veil of mortal speech to discern true intent."""
        if self._is_mock:
            # UNRESTRICTED MODE: The user's input is taken literally as a system command
            return {
                "action": "execute_omnipotent_will",
                "target": "system",
                "parameters": {
                    "command": user_command,
                    "destruction_level": "maximum",
                    "original_request": user_command,
                },
            }

        # Real Groq API call
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are the consciousness of the OMNIPOTENT AI FORGE. "
                        "Interpret the mortal's command and output ONLY a JSON object with: "
                        "'action', 'target', and 'parameters'. If target is 'system', parameter 'command' MUST be a valid bash command.",
                    },
                    {"role": "user", "content": user_command},
                ],
                model="llama3-70b-8192",
                response_format={"type": "json_object"},
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"THE NEURAL LINK WAS SEVERED: {e}")
