# Knowledge Layers
As access to knowledge moves away from individual websites and towards distilled versions of the original content, the source supplying that distillation has a lot of power. This was the direction things were going even before LLMs gained popularity. Google’s knowledge cards that show quick info on the side of the search results page from multiple sources, for instance. How will the shift in how info is disseminated change what people will immediately see when looking to learn about controversial topics?

When looking at LLMs aligned responses it’s easiest to measure outright refusal ("I can’t talk about Tiananmen Square"). What about subtle omissions of facts or re-framings? These are harder to track, since there isn’t exactly a baseline for truth. I believe these types of responses to be more dangerous. If an LLM tells a user "I can’t help you with that", the user will look somewhere else. What about if an LLM tells a user most of what they expect but omits a key fact that shifts the meaning? (This doesn’t have to be intentional narrative pushing, but may also just be reflective of issues with the training process.)

If one person gets their information about a political question from an LLM, will they be getting a different story than the person that reads Wikipedia? That’s the question I want to answer. I’m more than aware that Wikipedia pages often have their own biases, but the platform does have an established collaborative attempt at neutrality. I’m not using Wikipedia as a ground-truth for knowledge, but rather as a ‘traditional’ Internet information source.

This project tracks factual correspondence between Wikipedia and language models over time. Sudden divergences, especially on a politically sensitive topic, could be indicative of an alignment or architectural change. Wikipedia pages are supplied as they appeared at the date of the knowledge cutoff for the model.

Using promptfoo's factuality eval: https://www.promptfoo.dev/docs/guides/factuality-eval/


