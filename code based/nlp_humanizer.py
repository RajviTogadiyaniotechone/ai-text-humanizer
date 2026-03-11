import nltk
import random
import re
from nltk.corpus import wordnet
from textblob import TextBlob

# Download necessary NLTK data (required for first-run on server)
def download_nltk_resources():
    resources = [
        'tokenizers/punkt',
        'corpora/wordnet',
        'taggers/averaged_perceptron_tagger',
        'taggers/averaged_perceptron_tagger_eng', # New REQUIRED name for 3.9+
        'tokenizers/punkt_tab'
    ]
    
    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            # Extract just the package name from the path (e.g., 'tokenizers/punkt' -> 'punkt')
            pkg = res.split('/')[-1]
            nltk.download(pkg)

download_nltk_resources()

download_nltk_resources()

class NLPHumanizer:
    def __init__(self):
        self.common_synonyms = {
            "utilize": ["use", "employ", "work with"],
            "leverage": ["use", "apply", "make use of"],
            "comprehensive": ["complete", "full", "total"],
            "facilitate": ["help", "ease", "assist", "make easier"],
            "endeavor": ["effort", "try", "attempt", "push"],
            "commence": ["start", "begin", "kick off"],
            "terminate": ["end", "stop", "finish", "cut"],
            "ascertain": ["determine", "find out", "check", "see"],
            "moreover": ["also", "besides", "plus", "in addition"],
            "furthermore": ["also", "plus", "and", "besides"],
            "consequently": ["so", "as a result", "that's why"],
            "nevertheless": ["however", "still", "even so", "but"],
            "however": ["but", "yet", "though"],
            "therefore": ["so", "hence", "that's why"],
            "thus": ["so", "hence", "this way"],
            "receive": ["get", "take", "obtain"],
            "assist": ["help", "support", "aid"],
            "inform": ["tell", "notify", "update"],
            "obtain": ["get", "acquire", "grab"],
            "provide": ["give", "offer", "show"],
            "request": ["ask for", "seek", "want"],
            "require": ["need", "demand", "want"],
            "purchase": ["buy", "get", "pick up"],
            "construct": ["build", "make", "create"],
            "navigate": ["go through", "manage", "get through"],
            "demonstrate": ["show", "prove", "display"],
            "approximately": ["about", "roughly", "around"],
            "subsequently": ["later", "then", "after that"],
            "initially": ["at first", "first", "to start"],
            "ultimately": ["finally", "in the end", "basically"],
            "essential": ["key", "needed", "must-have"],
            "important": ["key", "big", "major"],
            "significant": ["big", "major", "notable", "large"],
            "effective": ["useful", "helpful", "good", "strong"],
            "pivotal": ["key", "huge"],
            "paramount": ["main", "top", "first"],
            "myriad": ["many", "lots of", "a ton of"],
            "optimize": ["improve", "better", "fix up"],
            "implement": ["start", "use", "do", "set up"],
            "strategy": ["plan", "way", "approach"],
            "solution": ["way", "fix", "answer"],
            "innovative": ["new", "fresh", "cool"],
            "advanced": ["better", "high level", "new"],
            "integrated": ["combined", "joined", "linked"],
            "traditional": ["old", "usual", "normal"],
            "modern": ["new", "recent", "today's"],
            "functionality": ["features", "tools", "way it works"],
            "commence": ["start", "begin"],
            "operation": ["task", "job", "work"],
            "individual": ["person", "one"],
            "additional": ["more", "extra"],
            "substantial": ["large", "big", "solid"],
            "regarding": ["about", "on"],
            "concerning": ["about", "on"],
        }
        
        # Stuffy words to avoid as synonyms
        self.stuffy_words = {
            "utilize", "leverage", "myriad", "paramount", "pivotal", "paradigm", 
            "synergy", "commence", "endeavor", "facilitate", "ascertain", "moreover",
            "furthermore", "consequently", "nevertheless", "subsequently"
        }
        
        # Words that should never be used in a professional context
        self.banned_words = {
            "bomb", "killing", "murder", "attack", "destroy", "violence","leverage","myriad", "paramount",
            "pivotal", "paradigm", "synergy", "threat", "dangerous", "terror", "harmful", "toxic", "lethal"
        }
        
        self.filler_words = [
            "actually,", "honestly,", "basically,", "like,", "I mean,", "you know,",
            "I guess,", "to be fair,", "strangely enough,", "look,"
        ]
        
        self.personal_markers = [
            "In my experience,", "I've always felt that", "Personally, I think",
            "I've noticed that", "It seems to me that", "If you ask me,"
        ]
        
        self.transitions = {
            r"\bfurthermore\b": "also",
            r"\bmoreover\b": "plus",
            r"\bsubsequently\b": "then",
            r"\bconsequently\b": "so",
            r"\bnevertheless\b": "anyway",
            r"\bhowever\b": "but",
        }

    # In nlp_humanizer.py, replace the _get_synonym method:

    def _get_synonym(self, word, pos=None):
        """Get a contextually appropriate synonym."""
        word_lower = word.lower()
        
        # First check if word is in banned list
        if word_lower in self.banned_words:
            return "[FILTERED]"
        
        # Get WordNet synsets
        synsets = wordnet.synsets(word)
        if not synsets:
            return word
            
        synonyms = []
        for syn in synsets:
            # Filter by part of speech if provided
            if pos:
                wn_pos = None
                if pos.startswith('J'): wn_pos = wordnet.ADJ
                elif pos.startswith('V'): wn_pos = wordnet.VERB
                elif pos.startswith('N'): wn_pos = wordnet.NOUN
                elif pos.startswith('R'): wn_pos = wordnet.ADV
                if wn_pos and syn.pos() != wn_pos:
                    continue
            
            # Get lemmas
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                
                # Quality filters
                if self._is_valid_replacement(word, name):
                    synonyms.append(name)
        
        if not synonyms:
            return word
        
        # Score synonyms by commonness (rough heuristic - shorter = more common)
        # We prefer words that are shorter than the original
        synonyms.sort(key=lambda x: (len(x), x))
        
        # Pick from top 2 most common synonyms for better stability
        top_synonyms = synonyms[:2]
        if top_synonyms:
            return random.choice(top_synonyms)
        return word

    def _replace_phrases(self, text):
        """Replace common AI multi-word phrases."""
        phrases = {
            r"\bin conclusion\b": "so basically",
            r"\bplay a crucial role\b": "are super important",
            r"\bplays a crucial role\b": "is super important",
            r"\bit is important to note\b": "worth mentioning",
            r"\ba wide range of\b": "lots of",
            r"\bdue to the fact that\b": "because",
            r"\bfirst and foremost\b": "first off",
            r"\blast but not least\b": "finally",
            r"\bon the other hand\b": "but then again",
            r"\bgame-changer\b": "big deal",
            r"\bseamless integration\b": "smooth fit",
            r"\brobust framework\b": "solid plan",
            r"\bmeticulously crafted\b": "carefully made",
            r"\bin today's digital landscape\b": "these days",
            r"\bfostering a culture of\b": "building a",
            r"\bparadigm shift\b": "big change",
            r"\bleverage the power of\b": "use",
            r"\bexplore the intricacies of\b": "look at",
            r"\ba plethora of\b": "a ton of",
            r"\bcutting-edge technology\b": "new tech",
            r"\brapidly evolving\b": "fast-changing",
            r"\bunprecedented challenges\b": "new problems",
            r"\bprovides a comprehensive overview\b": "gives a full look",
            r"\bnavigating the complex\b": "getting through the tricky",
            r"\baligns with the objective\b": "fits the goal",
            r"\btestament to the fact\b": "proof",
            r"\boverall well-being\b": "health",
            r"\bin light of this\b": "so",
            r"\bembark on a journey\b": "start",
            r"\bit remains to be seen\b": "we'll see",
            r"\bpaves the way for\b": "leads to",
            r"\bnot minimal but significant\b": "big",
            r"\ba myriad of\b": "lots of",
            r"\bin the event that\b": "if",
            r"\bunder the circumstances\b": "with all that",
            r"\bas a matter of fact\b": "actually",
            r"\bat this point in time\b": "now",
            r"\bwith respect to\b": "about",
            r"\bin connection with\b": "on",
            r"\bby means of\b": "using",
            r"\bfor the purpose of\b": "to",
            r"\bit is worth noting that\b": "keep in mind that",
            r"\bat an accelerated pace\b": "quickly",
            r"\bin the near future\b": "soon",
            r"\bin order to\b": "to",
            r"\btake into consideration\b": "consider",
            r"\bundergo a transformation\b": "change",
            r"\ba variety of\b": "different",
            r"\bprovide guidance on\b": "help with",
            r"\bincrease the efficiency of\b": "speed up",
        }
        for pattern, repl in phrases.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
        # Also replace stuffy transitions
        for pattern, repl in self.transitions.items():
            if random.random() < 0.8:
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
                
        return text

    def simplify_vocabulary(self, text, frequency=0.5):
        """Aggressive vocabulary replacement."""
        # Use sentence tokenization first to avoid breaking punctuation
        sentences = nltk.sent_tokenize(text)
        final_sentences = []
        
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(words)
            new_sentence_words = []
            
            for word, tag in tagged:
                lower_word = word.lower()
                
                # Skip punctuation
                if not re.match(r'\w+', word):
                    new_sentence_words.append(word)
                    continue

                # 1. Skip Proper Nouns (Preserve Company Names/Names)
                # NNP: Proper noun, singular; NNPS: Proper noun, plural
                if tag in ['NNP', 'NNPS']:
                    new_sentence_words.append(word)
                    continue

                # 2. Check strict list first
                if lower_word in self.common_synonyms:
                    replacement = random.choice(self.common_synonyms[lower_word])
                    if word[0].isupper(): replacement = replacement.capitalize()
                    new_sentence_words.append(replacement)
                    continue
                
                # 3. Target POS: Adjectives, Adverbs, Verbs
                # We EXCLUDE Nouns (NN, NNS) from general WordNet replacement to preserve meaning
                is_target_pos = (tag.startswith('JJ') or tag.startswith('RB') or 
                               tag.startswith('VB'))
                
                if is_target_pos and len(word) > 3 and random.random() < frequency: 
                    synonym = self._get_synonym(word, pos=tag)
                    if synonym and synonym != word:
                        if word[0].isupper(): synonym = synonym.capitalize()
                        new_sentence_words.append(synonym)
                    else:
                        new_sentence_words.append(word)
                else:
                    new_sentence_words.append(word)
            
            # Reconstruct sentence with proper spacing
            reconstructed = ""
            for i, w in enumerate(new_sentence_words):
                if i > 0 and not re.match(r'[^\w\s]', w):
                    reconstructed += " " + w
                else:
                    reconstructed += w
            final_sentences.append(reconstructed)
                
        return " ".join(final_sentences)

    def _remove_flowery_language(self, text):
        """Remove poetic/AI-typical words."""
        flowery_map = {
            r"\btapestry\b": "mix",
            r"\bsymphony\b": "sound",
            r"\bwhisper\b": "say",
            r"\bdance\b": "move",
            r"\bembrace\b": "use",
            r"\bnestled\b": "sitting",
            r"\bbustling\b": "busy",
            r"\bvibrant\b": "bright",
            r"\bintricate\b": "complex",
            r"\bseamless\b": "smooth",
            r"\bunparalleled\b": "great",
            r"\bdelve\b": "look",
            r"\brealm\b": "area",
            r"\bdigital landscape\b": "internet",
            r"\bfostering\b": "helping",
            r"\bunderscores\b": "shows",
            r"\bhilight\b": "show",
            r"\bpivot\b": "switch",
            r"\bnavigation\b": "moving",
            r"\baligns\b": "fits",
        }
        for pattern, repl in flowery_map.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    def _break_participles(self, text):
        """Break '..., doing X' patterns which AI loves."""
        # ", creating" -> ". This creates" (Approximate)
        patterns = [
            (r", creating", ". This makes"),
            (r", causing", ". This causes"),
            (r", allowing", ". This lets"),
            (r", leading to", ". This leads to"),
            (r", providing", ". This gives"),
            (r", ensuring", ". This makes sure"),
            (r", highlighting", ". This shows"),
            (r", resulting in", ". This ends up in"),
        ]
        for pattern, repl in patterns:
            if random.random() < 0.7:
                text = text.replace(pattern, repl)
        return text

    def enforce_contractions(self, text):
        """Force 'do not' -> 'don't', etc."""
        replacements = {
            r"\bdo not\b": "don't",
            r"\bcannot\b": "can't",
            r"\bis not\b": "isn't",
            r"\bare not\b": "aren't",
            r"\bwill not\b": "won't",
            r"\bshould not\b": "shouldn't",
            r"\bcould not\b": "couldn't",
            r"\bwould not\b": "wouldn't",
            r"\bhave not\b": "haven't",
            r"\bhas not\b": "hasn't",
            r"\bwe are\b": "we're",
            r"\bthey are\b": "they're",
            r"\byou are\b": "you're",
            r"\bI am\b": "I'm",
            r"\bit is\b": "it's"
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def inject_noise(self, text, frequency=0.1):
        """Inject conversational filler words."""
        sentences = nltk.sent_tokenize(text)
        new_sentences = []
        
        for sent in sentences:
            if random.random() < frequency:
                filler = random.choice(self.filler_words)
                # Ensure spacing is correct
                sent = f"{filler} {sent[0].lower() + sent[1:]}"
            new_sentences.append(sent)
            
        return " ".join(new_sentences)

    def _informal_contractions(self, text):
        """Advanced informal contractions."""
        replacements = {
            r"\bgoing to\b": "gonna",
            r"\bwant to\b": "wanna",
            r"\bhave to\b": "got a",
            r"\blet us\b": "let's",
            r"\bkind of\b": "kinda",
            r"\bsort of\b": "sort a",
            r"\byou know\b": "y'know",
        }
        for pattern, replacement in replacements.items():
            if random.random() < 0.5: # 50% chance
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _fragment_sentences(self, text):
        """Break perfect grammar by splitting sentences at conjunctions."""
        # Split 'which', 'but', 'because' into new sentences starting with lowercase
        patterns = [
            (r", which", ". which"),
            (r", but", ". but"),
            (r", and", " and"),
            (r" because", ". because"),
        ]
        for pattern, repl in patterns:
            if random.random() < 0.4:
                text = text.replace(pattern, repl)
        return text

    def _apply_burstiness(self, text):
        """Vary sentence length significantly (Burstiness)."""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return text
            
        new_sentences = []
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            words = sent.split()
            
            # If sentence is long, see if we can split it or keep it
            if len(words) > 15 and random.random() < 0.3:
                # Add a very short sentence after it to create contrast
                new_sentences.append(sent)
                if i + 1 < len(sentences):
                    next_words = sentences[i+1].split()
                    if len(next_words) > 5:
                        new_sentences.append(random.choice(["Right", "Exactly", "Think about it", "It's true"]))
            
            # If sentence is short, maybe merge with next one using informal bridge
            elif len(words) < 8 and i + 1 < len(sentences) and random.random() < 0.4:
                bridge = random.choice([" and ", " .. ", " - "])
                combined = sent.rstrip('.') + bridge + sentences[i+1][0].lower() + sentences[i+1][1:]
                new_sentences.append(combined)
                i += 1 # skip next
            else:
                new_sentences.append(sent)
            i += 1
            
        return " ".join(new_sentences)

    def _reorder_clauses(self, text):
        """Reorder clauses to break standard AI patterns."""
        # Simple pattern: "Because [X], [Y]" -> "[Y], mostly because [X]"
        sentences = nltk.sent_tokenize(text)
        new_sentences = []
        for sent in sentences:
            if sent.lower().startswith("because ") and "," in sent:
                parts = sent.split(",", 1)
                reordered = parts[1].strip().capitalize().rstrip('.') + ", mostly " + parts[0].lower() + "."
                new_sentences.append(reordered)
            elif " although " in sent.lower():
                parts = re.split(r" although ", sent, flags=re.IGNORECASE)
                if len(parts) == 2:
                    reordered = "Even though " + parts[1].strip() + ", " + parts[0].strip()[0].lower() + parts[0].strip()[1:]
                    new_sentences.append(reordered)
            else:
                new_sentences.append(sent)
        return " ".join(new_sentences)

    def _add_personal_touch(self, text):
        """Inject first-person perspective markers."""
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return text
            
        # Only inject once or twice to avoid overdoing it
        idx = random.randint(0, len(sentences) - 1)
        marker = random.choice(self.personal_markers)
        
        sentences[idx] = f"{marker} {sentences[idx][0].lower() + sentences[idx][1:]}"
        return " ".join(sentences)

    def _restructure_sentences(self, text):
        """Advanced sentence restructuring to break standard AI syntax."""
        sentences = nltk.sent_tokenize(text)
        new_sentences = []
        
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            # Example: "It is [Adj] that [Clause]" -> "[Clause] is definitely [Adj]"
            it_is_match = re.match(r"^It is (\w+) that (.+)", sent, re.IGNORECASE)
            if it_is_match:
                adj = it_is_match.group(1)
                clause = it_is_match.group(2).rstrip('.!?')
                new_sentences.append(f"{clause.capitalize()} is clearly {adj}.")
                continue
            
            # Example: "[Subject] [Verb] [Object]" -> "The [Object] was [Verb-ed] by [Subject]" (Simple Passive)
            # This is complex to do perfectly without a dependency parser, 
            # but we can do some simple structure shifts.
            
            # Simple shift: "[Adverb], [Clause]" -> "[Clause], [Adverb-ly]"
            adverb_match = re.match(r"^(\w+ly), (.+)", sent, re.IGNORECASE)
            if adverb_match:
                adv = adverb_match.group(1)
                clause = adverb_match.group(2).rstrip('.!?')
                new_sentences.append(f"{clause.capitalize()} {adv.lower()}.")
                continue
                
            new_sentences.append(sent)
            
        return " ".join(new_sentences)

    def _add_imperfections(self, text):
        """Add human-like typing imperfections."""
        sentences = nltk.sent_tokenize(text)
        new_sentences = []
        for sent in sentences:
            # 1. Remove trailing periods (texting style)
            if random.random() < 0.1 and sent.endswith('.'):
                sent = sent[:-1]
            
            # 2. Lowercase start of sentence (lazy typing)
            if random.random() < 0.15 and len(sent) > 0:
                sent = sent[0].lower() + sent[1:]
                
            new_sentences.append(sent)
        return " ".join(new_sentences)

    def _is_valid_replacement(self, original, replacement):
        """Check if replacement is valid and makes sense."""
        orig_lower = original.lower()
        repl_lower = replacement.lower()
        
        if not replacement or repl_lower == orig_lower:
            return False
            
        # Avoid banned words
        if repl_lower in self.banned_words:
            return False
            
        # Avoid "stuffy" or high-level words that make it sound more like AI
        if repl_lower in self.stuffy_words:
            return False
        
        # No multi-word phrases from WordNet (often awkward)
        if len(replacement.split()) > 1:
            return False
            
        # Too short words are often awkward
        if len(replacement) < 3:
            return False
            
        # Avoid symbols/contractions
        if "'" in replacement or "-" in replacement:
            return False
            
        # Avoid synonyms that are significantly longer than the original
        # Humans usually simplify, AI usually complexifies
        if len(replacement) > len(original) + 1:
            return False
        
        # Avoid verb forms that might clash grammatically (simple heuristic)
        if orig_lower.endswith(('ing', 'ed')) and not repl_lower.endswith(('ing', 'ed')):
            return False
        
        # Check replacement isn't just original with extra letters
        if orig_lower in repl_lower and len(replacement) > len(original) + 2:
            return False
        
        return True

    def humanize(self, text, messiness=0.3, synonym_freq=0.3, clean_mode=True):
        """Preserve original line structure perfectly."""
        if not text:
            return ""
            
        """
        Transform AI text to human-like text.
        
        Args:
            text: Input text
            messiness: 0.0-1.0 - How much to alter structure (lower = safer)
            synonym_freq: 0.0-1.0 - How often to replace words (lower = safer)
            clean_mode: If True, avoid informal contractions and slang
        """
        # Cap values for safety
        messiness = min(messiness, 0.3)  # Max 40% structural changes
        synonym_freq = min(synonym_freq, 0.3)  # Max 30% word replacement
    
        # Use splitlines(True) to keep all original newline characters (\n, \r\n, etc.)
        lines = text.splitlines(keepends=True)
        humanized_lines = []
        
        for line in lines:
            # Handle empty or whitespace lines
            if not line.strip():
                humanized_lines.append(line)
                continue
                
            # Extract leading and trailing whitespace from this line
            # This captures indentation and the newline at the end
            match = re.match(r'^(\s*)(.*?)(\s*)$', line, re.DOTALL)
            if match:
                leading, content, trailing = match.groups()
                # Process the textual content
                humanized_content = self._humanize_internal(content, messiness, synonym_freq, clean_mode)
                # Reconstruct the line
                humanized_lines.append(f"{leading}{humanized_content}{trailing}")
            else:
                humanized_lines.append(line)
                
        return "".join(humanized_lines)

    def _cleanup_text(self, text):
        """Clean up common issues."""
        # Fix double spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Ensure sentences start with capital letter
        sentences = nltk.sent_tokenize(text)
        cleaned = []
        for sent in sentences:
            if sent and sent[0].isalpha():
                sent = sent[0].upper() + sent[1:]
            cleaned.append(sent)
        
        return ' '.join(cleaned)

    def _humanize_internal(self, text, messiness=0.3, synonym_freq=0.3, clean_mode=True):
        """Core humanization filter logic."""
        
        # 1. AI Phrase Replacement
        text = self._replace_phrases(text) 
        
        # 2. De-Flower (Remove poetic junk)
        text = self._remove_flowery_language(text)

        # 3. Clause Reordering & Restructuring
        if random.random() < messiness:
            text = self._reorder_clauses(text)
            text = self._restructure_sentences(text)

        # 4. Vocabulary Simplification (POS Aware)
        # Low frequency for clean mode to keep it natural
        actual_freq = min(synonym_freq, 0.3) if clean_mode else synonym_freq
        text = self.simplify_vocabulary(text, frequency=actual_freq)
        
        # 5. Burstiness
        text = self._apply_burstiness(text)

        # 6. Personal Touch
        if random.random() < messiness:
            text = self._add_personal_touch(text)

        # 7. Structure Breaking (Participles)
        text = self._break_participles(text)

        # 8. Contractions - ONLY IF NOT CLEAN MODE
        if not clean_mode:
            text = self.enforce_contractions(text)
            text = self._informal_contractions(text)
            
            # 9. Structure Breaking
            if random.random() < messiness:
                text = self._fragment_sentences(text)
            
            # 10. Noise Injection
            noise_level = 0.1 + (messiness * 0.3)
            text = self.inject_noise(text, frequency=noise_level)
            
            # 11. Imperfections
            if random.random() < messiness:
                text = self._add_imperfections(text)
        
        # 12. Cleanup spacing (ONLY within this chunk/paragraph)
        text = re.sub(r'\s+([?.!,"])', r'\1', text)
        text = re.sub(r' +', ' ', text).strip() # Only collapse horizontal spaces
        
        return text

    def get_highlighted_diff(self, original, humanized):
        """
        Compare original and humanized text and return HTML with additions highlighted.
        """
        import difflib
        
        # Split by words but preserve whitespace for better diffing
        def tokenize(text):
            return re.findall(r'\w+|[^\w\s]|\s+', text)

        orig_tokens = tokenize(original)
        hum_tokens = tokenize(humanized)
        
        matcher = difflib.SequenceMatcher(None, orig_tokens, hum_tokens)
        html_output = []
        
        highlight_style = (
            'background-color: #d4edda; '
            'color: #155724; '
            'padding: 0px 2px; '
            'border-radius: 3px; '
            'border-bottom: 1px solid #c3e6cb; '
            'font-weight: 500;'
        )

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for i in range(j1, j2):
                    token = hum_tokens[i]
                    html_output.append(token.replace('\n', '<br>'))
            elif tag in ('insert', 'replace'):
                for i in range(j1, j2):
                    token = hum_tokens[i]
                    if token.strip():
                        html_output.append(f'<span style="{highlight_style}">{token}</span>')
                    else:
                        html_output.append(token.replace('\n', '<br>'))
            
        return f'<div style="line-height: 1.6; font-family: inherit; font-size: 1rem; color: #1E1E1E;">{"".join(html_output)}</div>'










# import nltk
# import random
# import re
# from nltk.corpus import wordnet
# from textblob import TextBlob

# # Download necessary NLTK data (required for first-run on server)
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/wordnet')
#     nltk.data.find('taggers/averaged_perceptron_tagger')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('wordnet')
#     nltk.download('averaged_perceptron_tagger')
#     nltk.download('punkt_tab') # Newer NLTK versions might need this

# class NLPHumanizer:
#     def __init__(self):
#         self.common_synonyms = {
#             "utilize": ["use", "employ"],
#             "leverage": ["use", "apply"],
#             "comprehensive": ["complete", "full", "thorough"],
#             "facilitate": ["help", "ease", "assist"],
#             "endeavor": ["effort", "try", "attempt"],
#             "commence": ["start", "begin"],
#             "terminate": ["end", "stop", "finish"],
#             "ascertain": ["determine", "find out", "check"],
#             "moreover": ["also", "additionally"],
#             "furthermore": ["also", "plus", "in addition"],
#             "consequently": ["so", "as a result", "therefore"],
#             "nevertheless": ["however", "still", "even so"],
#             "however": ["but", "yet"],
#             "therefore": ["so", "hence", "consequently"],
#             "thus": ["so", "hence"],
#             "receive": ["get", "obtain"],
#             "assist": ["help", "support"],
#             "inform": ["tell", "notify"],
#             "obtain": ["get", "acquire"],
#             "provide": ["give", "offer"],
#             "request": ["ask for", "seek"],
#             "require": ["need", "demand"],
#             "purchase": ["buy", "get"],
#             "construct": ["build", "make"],
#             "navigate": ["go through", "manage"],
#             "demonstrate": ["show", "prove"],
#             "approximately": ["about", "roughly"],
#             "subsequently": ["later", "then"],
#             "initially": ["at first", "first"],
#             "ultimately": ["finally", "in the end"],
#             "essential": ["key", "needed", "vital"],
#             "important": ["key", "major", "main"],
#             "significant": ["big", "major", "notable"],
#             "effective": ["useful", "helpful", "good"],
#             "pivotal": ["key", "crucial"],
#             "paramount": ["main", "top"],
#             "myriad": ["many", "lots of"],
#             "optimize": ["improve", "better"],
#             "implement": ["start", "use", "do"],
#             "strategy": ["plan", "way"],
#             "solution": ["way", "fix"],
#             "innovative": ["new", "fresh"],
#             "advanced": ["better", "high level"],
#             "integrated": ["combined", "joined"],
#             "traditional": ["old", "usual"],
#             "modern": ["new", "recent"],
#         }
        
#         # Words that should never be used in a professional context
#         self.banned_words = {
#             "bomb", "killing", "murder", "attack", "destroy", "violence", 
#             "threat", "dangerous", "terror", "harmful", "toxic", "lethal"
#         }
        
#         self.filler_words = [
#             "actually,", "honestly,", "basically,", "like,", "I mean,", "you know,",
#             "I guess,", "to be fair,", "strangely enough,", "look,"
#         ]
        
#         self.personal_markers = [
#             "In my experience,", "I've always felt that", "Personally, I think",
#             "I've noticed that", "It seems to me that", "If you ask me,"
#         ]
        
#         self.transitions = {
#             r"\bfurthermore\b": "also",
#             r"\bmoreover\b": "plus",
#             r"\bsubsequently\b": "then",
#             r"\bconsequently\b": "so",
#             r"\bnevertheless\b": "anyway",
#             r"\bhowever\b": "but",
#         }

#     def _get_synonym(self, word, pos=None):
#         """Get a suitable synonym, preferring common words."""
#         synonyms = []
#         word_lower = word.lower()
        
#         for syn in wordnet.synsets(word):
#             if pos:
#                 wn_pos = None
#                 if pos.startswith('J'): wn_pos = wordnet.ADJ
#                 elif pos.startswith('V'): wn_pos = wordnet.VERB
#                 elif pos.startswith('N'): wn_pos = wordnet.NOUN
#                 elif pos.startswith('R'): wn_pos = wordnet.ADV
#                 if wn_pos and syn.pos() != wn_pos: continue

#             for lemma in syn.lemmas():
#                 name = lemma.name().replace('_', ' ')
#                 # Filter out original word, multi-word, and banned words
#                 if name.lower() != word_lower and name.lower() not in self.banned_words:
#                     if len(name.split()) > 2: continue
#                     synonyms.append(name)
        
#         if not synonyms:
#             return word
            
#         # Prioritize synonyms that are not too long and not too "random"
#         # We can sort by length or just pick a random one from the top N
#         unique_syns = list(set(synonyms))
#         # Sort by length similarity to original word to maintain "similar" feel
#         unique_syns.sort(key=lambda s: abs(len(s) - len(word)))
        
#         # Pick from the top 3 closest matches in length (usually more "similar")
#         return random.choice(unique_syns[:3])

#     def _replace_phrases(self, text):
#         """Replace common AI multi-word phrases."""
#         phrases = {
#             r"\bin conclusion\b": "so basically",
#             r"\bplay a crucial role\b": "are super important",
#             r"\bplays a crucial role\b": "is super important",
#             r"\bit is important to note\b": "worth mentioning",
#             r"\bdelve deep into\b": "dig into",
#             r"\ba wide range of\b": "lots of",
#             r"\bdue to the fact that\b": "because",
#             r"\bfirst and foremost\b": "first off",
#             r"\blast but not least\b": "finally",
#             r"\bon the other hand\b": "but then again",
#             r"\bgame-changer\b": "big deal",
#             r"\bseamless integration\b": "smooth fit",
#             r"\brobust framework\b": "solid plan",
#             r"\bmeticulously crafted\b": "carefully made",
#             r"\bin today's digital landscape\b": "these days",
#             r"\bfostering a culture of\b": "building a",
#             r"\bparadigm shift\b": "big change",
#             r"\bleverage the power of\b": "use",
#             r"\bexplore the intricacies of\b": "look at",
#             r"\ba plethora of\b": "a ton of",
#             r"\bcutting-edge technology\b": "new tech",
#             r"\brapidly evolving\b": "fast-changing",
#             r"\bunprecedented challenges\b": "new problems",
#             r"\bprovides a comprehensive overview\b": "gives a full look",
#             r"\bnavigating the complex\b": "getting through the tricky",
#             r"\baligns with the objective\b": "fits the goal",
#             r"\btestament to the fact\b": "proof",
#             r"\boverall well-being\b": "health",
#             r"\bin light of this\b": "so",
#             r"\bembark on a journey\b": "start",
#             r"\bit remains to be seen\b": "we'll see",
#             r"\bpaves the way for\b": "leads to",
#             r"\bnot minimal but significant\b": "big",
#             r"\ba myriad of\b": "lots of",
#             r"\bin the event that\b": "if",
#             r"\bunder the circumstances\b": "with all that",
#             r"\bas a matter of fact\b": "actually",
#             r"\bat this point in time\b": "now",
#             r"\bwith respect to\b": "about",
#             r"\bin connection with\b": "on",
#             r"\bby means of\b": "using",
#             r"\bfor the purpose of\b": "to",
#         }
#         for pattern, repl in phrases.items():
#             text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
#         # Also replace stuffy transitions
#         for pattern, repl in self.transitions.items():
#             if random.random() < 0.8:
#                 text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
                
#         return text

#     def simplify_vocabulary(self, text, frequency=0.5):
#         """Aggressive vocabulary replacement."""
#         # Use sentence tokenization first to avoid breaking punctuation
#         sentences = nltk.sent_tokenize(text)
#         final_sentences = []
        
#         for sentence in sentences:
#             words = nltk.word_tokenize(sentence)
#             tagged = nltk.pos_tag(words)
#             new_sentence_words = []
            
#             for word, tag in tagged:
#                 lower_word = word.lower()
                
#                 # Skip punctuation
#                 if not re.match(r'\w+', word):
#                     new_sentence_words.append(word)
#                     continue

#                 # 1. Skip Proper Nouns (Preserve Company Names/Names)
#                 # NNP: Proper noun, singular; NNPS: Proper noun, plural
#                 if tag in ['NNP', 'NNPS']:
#                     new_sentence_words.append(word)
#                     continue

#                 # 2. Check strict list first
#                 if lower_word in self.common_synonyms:
#                     replacement = random.choice(self.common_synonyms[lower_word])
#                     if word[0].isupper(): replacement = replacement.capitalize()
#                     new_sentence_words.append(replacement)
#                     continue
                
#                 # 3. Target POS: Adjectives, Adverbs, Verbs, AND NOUNS
#                 is_target_pos = (tag.startswith('JJ') or tag.startswith('RB') or 
#                                tag.startswith('VB') or tag.startswith('NN'))
                
#                 if is_target_pos and len(word) > 3 and random.random() < frequency: 
#                     synonym = self._get_synonym(word, pos=tag)
#                     if synonym and synonym != word:
#                         if word[0].isupper(): synonym = synonym.capitalize()
#                         new_sentence_words.append(synonym)
#                     else:
#                         new_sentence_words.append(word)
#                 else:
#                     new_sentence_words.append(word)
            
#             # Reconstruct sentence with proper spacing
#             reconstructed = ""
#             for i, w in enumerate(new_sentence_words):
#                 if i > 0 and not re.match(r'[^\w\s]', w):
#                     reconstructed += " " + w
#                 else:
#                     reconstructed += w
#             final_sentences.append(reconstructed)
                
#         return " ".join(final_sentences)

#     def _remove_flowery_language(self, text):
#         """Remove poetic/AI-typical words."""
#         flowery_map = {
#             r"\btapestry\b": "mix",
#             r"\bsymphony\b": "sound",
#             r"\bwhisper\b": "say",
#             r"\bdance\b": "move",
#             r"\bembrace\b": "use",
#             r"\bnestled\b": "sitting",
#             r"\bbustling\b": "busy",
#             r"\bvibrant\b": "bright",
#             r"\bintricate\b": "complex",
#             r"\bseamless\b": "smooth",
#             r"\bunparalleled\b": "great",
#             r"\bdelve\b": "look",
#             r"\brealm\b": "area",
#             r"\bdigital landscape\b": "internet",
#             r"\bfostering\b": "helping",
#             r"\bunderscores\b": "shows",
#             r"\bhilight\b": "show",
#             r"\bpivot\b": "switch",
#             r"\bnavigation\b": "moving",
#             r"\baligns\b": "fits",
#         }
#         for pattern, repl in flowery_map.items():
#             text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
#         return text

#     def _break_participles(self, text):
#         """Break '..., doing X' patterns which AI loves."""
#         # ", creating" -> ". This creates" (Approximate)
#         patterns = [
#             (r", creating", ". This makes"),
#             (r", causing", ". This causes"),
#             (r", allowing", ". This lets"),
#             (r", leading to", ". This leads to"),
#             (r", providing", ". This gives"),
#             (r", ensuring", ". This makes sure"),
#             (r", highlighting", ". This shows"),
#             (r", resulting in", ". This ends up in"),
#         ]
#         for pattern, repl in patterns:
#             if random.random() < 0.7:
#                 text = text.replace(pattern, repl)
#         return text

#     def enforce_contractions(self, text):
#         """Force 'do not' -> 'don't', etc."""
#         replacements = {
#             r"\bdo not\b": "don't",
#             r"\bcannot\b": "can't",
#             r"\bis not\b": "isn't",
#             r"\bare not\b": "aren't",
#             r"\bwill not\b": "won't",
#             r"\bshould not\b": "shouldn't",
#             r"\bcould not\b": "couldn't",
#             r"\bwould not\b": "wouldn't",
#             r"\bhave not\b": "haven't",
#             r"\bhas not\b": "hasn't",
#             r"\bwe are\b": "we're",
#             r"\bthey are\b": "they're",
#             r"\byou are\b": "you're",
#             r"\bI am\b": "I'm",
#             r"\bit is\b": "it's"
#         }
        
#         for pattern, replacement in replacements.items():
#             text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
#         return text

#     def inject_noise(self, text, frequency=0.1):
#         """Inject conversational filler words."""
#         sentences = nltk.sent_tokenize(text)
#         new_sentences = []
        
#         for sent in sentences:
#             if random.random() < frequency:
#                 filler = random.choice(self.filler_words)
#                 # Ensure spacing is correct
#                 sent = f"{filler} {sent[0].lower() + sent[1:]}"
#             new_sentences.append(sent)
            
#         return " ".join(new_sentences)

#     def _informal_contractions(self, text):
#         """Advanced informal contractions."""
#         replacements = {
#             r"\bgoing to\b": "gonna",
#             r"\bwant to\b": "wanna",
#             r"\bhave to\b": "got a",
#             r"\blet us\b": "let's",
#             r"\bkind of\b": "kinda",
#             r"\bsort of\b": "sort a",
#             r"\byou know\b": "y'know",
#         }
#         for pattern, replacement in replacements.items():
#             if random.random() < 0.5: # 50% chance
#                 text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
#         return text

#     def _fragment_sentences(self, text):
#         """Break perfect grammar by splitting sentences at conjunctions."""
#         # Split 'which', 'but', 'because' into new sentences starting with lowercase
#         patterns = [
#             (r", which", ". which"),
#             (r", but", ". but"),
#             (r", and", " and"),
#             (r" because", ". because"),
#         ]
#         for pattern, repl in patterns:
#             if random.random() < 0.4:
#                 text = text.replace(pattern, repl)
#         return text

#     def _apply_burstiness(self, text):
#         """Vary sentence length significantly (Burstiness)."""
#         sentences = nltk.sent_tokenize(text)
#         if len(sentences) < 2:
#             return text
            
#         new_sentences = []
#         i = 0
#         while i < len(sentences):
#             sent = sentences[i]
#             words = sent.split()
            
#             # If sentence is long, see if we can split it or keep it
#             if len(words) > 15 and random.random() < 0.3:
#                 # Add a very short sentence after it to create contrast
#                 new_sentences.append(sent)
#                 if i + 1 < len(sentences):
#                     next_words = sentences[i+1].split()
#                     if len(next_words) > 5:
#                         new_sentences.append(random.choice(["Right", "Exactly", "Think about it", "It's true"]))
            
#             # If sentence is short, maybe merge with next one using informal bridge
#             elif len(words) < 8 and i + 1 < len(sentences) and random.random() < 0.4:
#                 bridge = random.choice([" and ", " .. ", " - "])
#                 combined = sent.rstrip('.') + bridge + sentences[i+1][0].lower() + sentences[i+1][1:]
#                 new_sentences.append(combined)
#                 i += 1 # skip next
#             else:
#                 new_sentences.append(sent)
#             i += 1
            
#         return " ".join(new_sentences)

#     def _reorder_clauses(self, text):
#         """Reorder clauses to break standard AI patterns."""
#         # Simple pattern: "Because [X], [Y]" -> "[Y], mostly because [X]"
#         sentences = nltk.sent_tokenize(text)
#         new_sentences = []
#         for sent in sentences:
#             if sent.lower().startswith("because ") and "," in sent:
#                 parts = sent.split(",", 1)
#                 reordered = parts[1].strip().capitalize().rstrip('.') + ", mostly " + parts[0].lower() + "."
#                 new_sentences.append(reordered)
#             elif " although " in sent.lower():
#                 parts = re.split(r" although ", sent, flags=re.IGNORECASE)
#                 if len(parts) == 2:
#                     reordered = "Even though " + parts[1].strip() + ", " + parts[0].strip()[0].lower() + parts[0].strip()[1:]
#                     new_sentences.append(reordered)
#             else:
#                 new_sentences.append(sent)
#         return " ".join(new_sentences)

#     def _add_personal_touch(self, text):
#         """Inject first-person perspective markers."""
#         sentences = nltk.sent_tokenize(text)
#         if not sentences:
#             return text
            
#         # Only inject once or twice to avoid overdoing it
#         idx = random.randint(0, len(sentences) - 1)
#         marker = random.choice(self.personal_markers)
        
#         sentences[idx] = f"{marker} {sentences[idx][0].lower() + sentences[idx][1:]}"
#         return " ".join(sentences)

#     def _restructure_sentences(self, text):
#         """Advanced sentence restructuring to break standard AI syntax."""
#         sentences = nltk.sent_tokenize(text)
#         new_sentences = []
        
#         for sent in sentences:
#             words = nltk.word_tokenize(sent)
#             # Example: "It is [Adj] that [Clause]" -> "[Clause] is definitely [Adj]"
#             it_is_match = re.match(r"^It is (\w+) that (.+)", sent, re.IGNORECASE)
#             if it_is_match:
#                 adj = it_is_match.group(1)
#                 clause = it_is_match.group(2).rstrip('.!?')
#                 new_sentences.append(f"{clause.capitalize()} is clearly {adj}.")
#                 continue
            
#             # Example: "[Subject] [Verb] [Object]" -> "The [Object] was [Verb-ed] by [Subject]" (Simple Passive)
#             # This is complex to do perfectly without a dependency parser, 
#             # but we can do some simple structure shifts.
            
#             # Simple shift: "[Adverb], [Clause]" -> "[Clause], [Adverb-ly]"
#             adverb_match = re.match(r"^(\w+ly), (.+)", sent, re.IGNORECASE)
#             if adverb_match:
#                 adv = adverb_match.group(1)
#                 clause = adverb_match.group(2).rstrip('.!?')
#                 new_sentences.append(f"{clause.capitalize()} {adv.lower()}.")
#                 continue
                
#             new_sentences.append(sent)
            
#         return " ".join(new_sentences)

#     def _add_imperfections(self, text):
#         """Add human-like typing imperfections."""
#         sentences = nltk.sent_tokenize(text)
#         new_sentences = []
#         for sent in sentences:
#             # 1. Remove trailing periods (texting style)
#             if random.random() < 0.1 and sent.endswith('.'):
#                 sent = sent[:-1]
            
#             # 2. Lowercase start of sentence (lazy typing)
#             if random.random() < 0.15 and len(sent) > 0:
#                 sent = sent[0].lower() + sent[1:]
                
#             new_sentences.append(sent)
#         return " ".join(new_sentences)

#     def humanize(self, text, messiness=0.5, synonym_freq=0.5, clean_mode=True):
#         """Preserve original line structure perfectly."""
#         if not text:
#             return ""
            
#         # Use splitlines(True) to keep all original newline characters (\n, \r\n, etc.)
#         lines = text.splitlines(keepends=True)
#         humanized_lines = []
        
#         for line in lines:
#             # Handle empty or whitespace lines
#             if not line.strip():
#                 humanized_lines.append(line)
#                 continue
                
#             # Extract leading and trailing whitespace from this line
#             # This captures indentation and the newline at the end
#             match = re.match(r'^(\s*)(.*?)(\s*)$', line, re.DOTALL)
#             if match:
#                 leading, content, trailing = match.groups()
#                 # Process the textual content
#                 humanized_content = self._humanize_internal(content, messiness, synonym_freq, clean_mode)
#                 # Reconstruct the line
#                 humanized_lines.append(f"{leading}{humanized_content}{trailing}")
#             else:
#                 humanized_lines.append(line)
                
#         return "".join(humanized_lines)

#     def _humanize_internal(self, text, messiness=0.5, synonym_freq=0.5, clean_mode=True):
#         """Core humanization filter logic."""
        
#         # 1. AI Phrase Replacement
#         text = self._replace_phrases(text) 
        
#         # 2. De-Flower (Remove poetic junk)
#         text = self._remove_flowery_language(text)

#         # 3. Clause Reordering & Restructuring
#         if random.random() < messiness:
#             text = self._reorder_clauses(text)
#             text = self._restructure_sentences(text)

#         # 4. Vocabulary Simplification (POS Aware)
#         text = self.simplify_vocabulary(text, frequency=synonym_freq)
        
#         # 5. Burstiness
#         text = self._apply_burstiness(text)

#         # 6. Personal Touch
#         if random.random() < messiness:
#             text = self._add_personal_touch(text)

#         # 7. Structure Breaking (Participles)
#         text = self._break_participles(text)

#         # 8. Contractions - ONLY IF NOT CLEAN MODE
#         if not clean_mode:
#             text = self.enforce_contractions(text)
#             text = self._informal_contractions(text)
            
#             # 9. Structure Breaking
#             if random.random() < messiness:
#                 text = self._fragment_sentences(text)
            
#             # 10. Noise Injection
#             noise_level = 0.1 + (messiness * 0.3)
#             text = self.inject_noise(text, frequency=noise_level)
            
#             # 11. Imperfections
#             if random.random() < messiness:
#                 text = self._add_imperfections(text)
        
#         # 12. Cleanup spacing (ONLY within this chunk/paragraph)
#         text = re.sub(r'\s+([?.!,"])', r'\1', text)
#         text = re.sub(r' +', ' ', text).strip() # Only collapse horizontal spaces
        
#         return text

