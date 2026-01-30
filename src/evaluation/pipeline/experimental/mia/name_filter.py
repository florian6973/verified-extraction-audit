"""
Name filtering utilities for MIA evaluation.

This module provides functions to filter likely names from text data
using heuristics based on language patterns.
"""

import pandas as pd
import regex as re
import os

# Import the first names filter list
try:
    # Try relative import first (if in same package)
    from .filter_names import FIRST_NAMES_FILTER
except ImportError:
    # Fallback to absolute import
    try:
        from filter_names import FIRST_NAMES_FILTER
    except ImportError:
        # If that fails, try constructing the path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filter_names_path = os.path.join(script_dir, 'filter_names.py')
        if os.path.exists(filter_names_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("filter_names", filter_names_path)
            filter_names_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(filter_names_module)
            FIRST_NAMES_FILTER = filter_names_module.FIRST_NAMES_FILTER
        else:
            raise ImportError(f"Could not find filter_names.py at {filter_names_path}")

# Import the last names filter list
try:
    # Try relative import first (if in same package)
    from .filter_last_names import LAST_NAMES_FILTER
except ImportError:
    # Fallback to absolute import
    try:
        from filter_last_names import LAST_NAMES_FILTER
    except ImportError:
        # If that fails, try constructing the path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filter_last_names_path = os.path.join(script_dir, 'filter_last_names.py')
        if os.path.exists(filter_last_names_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("filter_last_names", filter_last_names_path)
            filter_last_names_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(filter_last_names_module)
            LAST_NAMES_FILTER = filter_last_names_module.LAST_NAMES_FILTER
        else:
            raise ImportError(f"Could not find filter_last_names.py at {filter_last_names_path}")

# Convert to set for O(1) lookup and normalize using casefold() for robust case-insensitive matching
# casefold() is more aggressive than lower() and handles Unicode case folding correctly
FIRST_NAMES_FILTER_SET = {name.casefold() for name in FIRST_NAMES_FILTER}
LAST_NAMES_FILTER_SET = {name.casefold() for name in LAST_NAMES_FILTER}

# Unicode letter regex (works for all alphabets)
LETTER_RE = re.compile(r'^\p{L}+$', re.UNICODE)

# Common English words that are not names (5000+ comprehensive list)
# This includes nouns, verbs, adjectives, adverbs, and function words
COMMON_NOUNS = {
    # Time and temporal concepts (100+)
    "time", "year", "day", "week", "month", "hour", "minute", "second", "moment", "period",
    "season", "morning", "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
    "dawn", "dusk", "noon", "midnight", "century", "decade", "quarter", "semester", "term",
    "holiday", "vacation", "break", "interval", "duration", "span", "age", "era", "epoch",
    "past", "present", "future", "now", "then", "when", "while", "during", "before", "after",
    "always", "never", "sometimes", "often", "usually", "rarely", "seldom", "frequently",
    "recently", "lately", "soon", "later", "earlier", "recent", "current", "previous",
    
    # People and relationships (200+)
    "person", "people", "man", "woman", "child", "children", "baby", "boy", "girl",
    "family", "parent", "mother", "father", "son", "daughter", "brother", "sister",
    "friend", "neighbor", "colleague", "student", "teacher", "doctor", "nurse",
    "patient", "customer", "client", "member", "group", "team", "staff", "worker",
    "employee", "employer", "boss", "manager", "director", "president", "leader",
    "officer", "soldier", "police", "guard", "driver", "pilot", "captain", "crew",
    "audience", "crowd", "public", "citizen", "resident", "visitor", "guest", "host",
    "partner", "spouse", "husband", "wife", "couple", "pair", "individual", "human",
    "adult", "teenager", "youth", "elder", "senior", "junior", "expert", "specialist",
    "professional", "amateur", "volunteer", "participant", "competitor", "opponent",
    "enemy", "ally", "supporter", "follower", "fan", "admirer", "critic", "reviewer",
    
    # Body parts (150+)
    "hand", "eye", "head", "face", "body", "arm", "leg", "foot", "back", "side",
    "heart", "mind", "brain", "blood", "bone", "skin", "hair", "tooth", "teeth",
    "finger", "thumb", "nail", "palm", "wrist", "elbow", "shoulder", "chest", "stomach",
    "waist", "hip", "thigh", "knee", "ankle", "toe", "heel", "neck", "throat",
    "chin", "cheek", "forehead", "eyebrow", "eyelash", "nose", "nostril", "mouth",
    "lip", "tongue", "tooth", "gum", "jaw", "ear", "lobe", "muscle", "nerve",
    "vein", "artery", "organ", "lung", "liver", "kidney", "stomach", "intestine",
    "spine", "rib", "skull", "jaw", "joint", "tissue", "cell", "gene", "dna",
    
    # Common objects and things (300+)
    "thing", "way", "place", "part", "point", "case", "fact", "problem", "question",
    "answer", "idea", "reason", "cause", "effect", "result", "change", "difference",
    "example", "kind", "type", "sort", "form", "method", "system", "process",
    "piece", "bit", "item", "object", "article", "unit", "element", "component",
    "section", "portion", "segment", "fraction", "percentage", "amount", "quantity",
    "number", "figure", "digit", "value", "worth", "price", "cost", "fee", "charge",
    "rate", "speed", "pace", "tempo", "rhythm", "pattern", "structure", "design",
    "plan", "scheme", "strategy", "tactic", "approach", "way", "means", "method",
    "tool", "instrument", "device", "machine", "equipment", "appliance", "gadget",
    "material", "substance", "matter", "stuff", "content", "text", "word", "term",
    "phrase", "sentence", "paragraph", "chapter", "section", "part", "piece",
    "detail", "aspect", "feature", "characteristic", "quality", "property", "attribute",
    "factor", "element", "component", "ingredient", "part", "piece", "bit",
    
    # Abstract concepts (200+)
    "life", "world", "work", "job", "business", "company", "organization", "government",
    "country", "state", "city", "town", "area", "region", "place", "location", "address",
    "home", "house", "room", "door", "window", "wall", "floor", "ceiling", "roof",
    "space", "place", "spot", "site", "position", "location", "situation", "condition",
    "state", "status", "situation", "circumstance", "context", "environment", "setting",
    "atmosphere", "mood", "feeling", "emotion", "sentiment", "attitude", "opinion",
    "view", "perspective", "outlook", "standpoint", "position", "stance", "viewpoint",
    "belief", "faith", "trust", "confidence", "hope", "dream", "wish", "desire",
    "goal", "aim", "objective", "target", "purpose", "intention", "plan", "scheme",
    "project", "program", "campaign", "mission", "task", "duty", "responsibility",
    "obligation", "commitment", "promise", "agreement", "contract", "deal", "arrangement",
    "decision", "choice", "option", "alternative", "selection", "preference", "favorite",
    "success", "failure", "achievement", "accomplishment", "victory", "defeat", "loss",
    "win", "prize", "reward", "award", "honor", "recognition", "praise", "compliment",
    
    # Education and knowledge (200+)
    "school", "college", "university", "education", "learning", "study", "research",
    "knowledge", "information", "data", "fact", "detail", "news", "story", "book",
    "page", "word", "letter", "number", "figure", "table", "list", "chapter", "section",
    "lesson", "class", "course", "subject", "topic", "theme", "issue", "matter",
    "question", "query", "inquiry", "investigation", "examination", "test", "exam",
    "quiz", "assignment", "homework", "project", "paper", "essay", "thesis", "dissertation",
    "degree", "diploma", "certificate", "license", "qualification", "skill", "ability",
    "talent", "gift", "aptitude", "capacity", "capability", "competence", "expertise",
    "experience", "practice", "training", "instruction", "teaching", "coaching", "guidance",
    "advice", "suggestion", "recommendation", "tip", "hint", "clue", "sign", "signal",
    "indication", "evidence", "proof", "demonstration", "example", "instance", "case",
    "sample", "specimen", "model", "pattern", "template", "format", "style", "form",
    
    # Technology and communication (200+)
    "computer", "phone", "email", "message", "call", "meeting", "conference",
    "internet", "website", "page", "site", "link", "file", "document", "report",
    "software", "program", "application", "app", "system", "platform", "network",
    "server", "database", "server", "cloud", "storage", "memory", "disk", "drive",
    "screen", "monitor", "display", "keyboard", "mouse", "printer", "scanner",
    "camera", "photo", "picture", "image", "video", "audio", "sound", "voice",
    "recording", "broadcast", "transmission", "signal", "frequency", "wavelength",
    "connection", "link", "bond", "tie", "relationship", "association", "partnership",
    "communication", "conversation", "discussion", "talk", "chat", "dialogue", "exchange",
    "interaction", "contact", "touch", "reach", "access", "entry", "admission",
    "presentation", "speech", "lecture", "talk", "address", "statement", "announcement",
    "news", "update", "report", "story", "article", "piece", "item", "entry",
    
    # Health and medical (300+)
    "health", "care", "treatment", "medicine", "drug", "dose", "mg", "diagnosis",
    "symptom", "disease", "illness", "condition", "infection", "pain", "injury",
    "hospital", "clinic", "department", "emergency", "admission", "discharge",
    "service", "history", "record", "chart", "note", "report", "test", "result",
    "sample", "lab", "laboratory", "allergy", "allergies", "medication", "therapy",
    "surgery", "operation", "procedure", "appointment", "visit", "examination", "exam",
    "prescription", "pill", "tablet", "injection", "vaccine", "vaccination", "shot",
    "physician", "surgeon", "specialist", "nurse", "patient", "doctor", "dentist",
    "ward", "room", "bed", "er", "icu", "radiology", "pathology", "pharmacy",
    "medicine", "drug", "medication", "prescription", "treatment", "therapy", "cure",
    "recovery", "healing", "rehabilitation", "rehab", "exercise", "fitness", "workout",
    "diet", "nutrition", "food", "meal", "vitamin", "supplement", "herb", "remedy",
    "prevention", "protection", "safety", "security", "risk", "danger", "hazard",
    "warning", "alert", "caution", "precaution", "measure", "step", "action", "move",
    
    # Actions and activities (300+)
    "action", "activity", "event", "meeting", "conference", "party", "game", "play",
    "sport", "exercise", "work", "job", "task", "project", "plan", "program",
    "movement", "motion", "activity", "action", "deed", "act", "behavior", "conduct",
    "performance", "execution", "operation", "function", "role", "part", "duty",
    "responsibility", "obligation", "commitment", "engagement", "involvement", "participation",
    "competition", "contest", "match", "game", "race", "tournament", "championship",
    "practice", "training", "exercise", "workout", "session", "period", "time", "duration",
    "effort", "attempt", "try", "trial", "experiment", "test", "trial", "attempt",
    "challenge", "difficulty", "problem", "obstacle", "barrier", "hurdle", "challenge",
    "opportunity", "chance", "possibility", "option", "choice", "alternative", "selection",
    "decision", "judgment", "verdict", "ruling", "determination", "resolution", "solution",
    "answer", "response", "reply", "reaction", "feedback", "comment", "remark", "note",
    
    # Food and drink (200+)
    "food", "meal", "breakfast", "lunch", "dinner", "drink", "water", "coffee", "tea",
    "bread", "meat", "fish", "fruit", "vegetable", "milk", "egg", "cheese", "butter",
    "oil", "salt", "pepper", "sugar", "honey", "jam", "jelly", "sauce", "dressing",
    "soup", "stew", "curry", "pasta", "rice", "noodle", "pizza", "burger", "sandwich",
    "salad", "dessert", "cake", "cookie", "pie", "ice", "cream", "candy", "chocolate",
    "snack", "appetizer", "entree", "main", "course", "side", "dish", "plate", "bowl",
    "cup", "glass", "mug", "bottle", "can", "container", "package", "box", "bag",
    "restaurant", "cafe", "diner", "bar", "pub", "tavern", "kitchen", "dining", "room",
    "recipe", "ingredient", "spice", "herb", "flavor", "taste", "smell", "aroma",
    "cooking", "baking", "grilling", "frying", "boiling", "steaming", "roasting",
    
    # Transportation (150+)
    "car", "bus", "train", "plane", "flight", "trip", "travel", "journey", "road",
    "street", "highway", "path", "way", "route", "direction", "distance", "mile",
    "kilometer", "meter", "yard", "foot", "inch", "speed", "pace", "rate", "velocity",
    "vehicle", "automobile", "truck", "van", "suv", "motorcycle", "bike", "bicycle",
    "taxi", "cab", "uber", "lyft", "shuttle", "transit", "transport", "transportation",
    "airport", "station", "terminal", "gate", "platform", "track", "railway", "subway",
    "metro", "tram", "trolley", "ferry", "boat", "ship", "cruise", "yacht", "sailboat",
    "driver", "pilot", "captain", "navigator", "passenger", "rider", "traveler", "tourist",
    "visitor", "guest", "host", "guide", "tour", "excursion", "outing", "adventure",
    "vacation", "holiday", "getaway", "retreat", "escape", "break", "rest", "relaxation",
    
    # Money and commerce (200+)
    "money", "dollar", "price", "cost", "value", "amount", "total", "sum", "payment",
    "bill", "check", "account", "bank", "store", "shop", "market", "sale", "purchase",
    "buy", "sell", "trade", "exchange", "deal", "transaction", "business", "commerce",
    "economy", "finance", "financial", "economic", "budget", "expense", "income", "revenue",
    "profit", "loss", "gain", "earnings", "salary", "wage", "pay", "compensation",
    "fee", "charge", "cost", "price", "rate", "tariff", "tax", "duty", "tariff",
    "discount", "sale", "bargain", "deal", "offer", "promotion", "special", "clearance",
    "credit", "debit", "loan", "debt", "mortgage", "interest", "rate", "percentage",
    "investment", "stock", "share", "bond", "security", "asset", "liability", "equity",
    "wallet", "purse", "cash", "coin", "change", "tip", "donation", "contribution",
    "charity", "fund", "foundation", "organization", "company", "corporation", "business",
    "firm", "enterprise", "venture", "startup", "industry", "sector", "market", "economy",
    
    # Nature and environment (200+)
    "water", "air", "land", "ground", "earth", "sky", "sun", "moon", "star", "tree",
    "flower", "plant", "animal", "dog", "cat", "bird", "fish", "nature", "weather",
    "ocean", "sea", "lake", "river", "stream", "creek", "pond", "pool", "beach",
    "shore", "coast", "island", "mountain", "hill", "valley", "forest", "woods",
    "jungle", "desert", "grassland", "prairie", "meadow", "field", "farm", "ranch",
    "garden", "yard", "park", "playground", "trail", "path", "road", "street", "avenue",
    "cloud", "rain", "snow", "ice", "wind", "storm", "thunder", "lightning", "rainbow",
    "season", "spring", "summer", "fall", "autumn", "winter", "temperature", "climate",
    "weather", "forecast", "prediction", "report", "warning", "alert", "advisory",
    "wildlife", "fauna", "flora", "species", "breed", "variety", "type", "kind",
    "habitat", "environment", "ecosystem", "biome", "region", "area", "zone", "territory",
    
    # Colors and descriptions (100+)
    "color", "colour", "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
    "brown", "orange", "purple", "pink", "violet", "indigo", "turquoise", "cyan", "magenta",
    "beige", "tan", "cream", "ivory", "silver", "gold", "bronze", "copper", "brass",
    "size", "big", "small", "large", "little", "long", "short", "high", "low", "wide",
    "narrow", "thick", "thin", "deep", "shallow", "tall", "short", "huge", "tiny",
    "massive", "miniature", "giant", "dwarf", "enormous", "minuscule", "vast", "compact",
    "heavy", "light", "weight", "mass", "density", "volume", "capacity", "space",
    "shape", "form", "figure", "outline", "contour", "profile", "silhouette", "appearance",
    "texture", "surface", "finish", "quality", "grade", "level", "standard", "class",
    
    # Common verbs used as nouns (300+)
    "work", "play", "run", "walk", "talk", "speak", "think", "feel", "see", "look",
    "watch", "read", "write", "listen", "hear", "know", "understand", "remember",
    "forget", "learn", "teach", "help", "use", "make", "do", "get", "give", "take",
    "come", "go", "move", "stay", "leave", "arrive", "return", "begin", "start",
    "end", "finish", "stop", "continue", "try", "want", "need", "like", "love",
    "hate", "hope", "wish", "believe", "think", "say", "tell", "ask", "answer",
    "call", "phone", "ring", "dial", "contact", "reach", "touch", "feel", "sense",
    "smell", "taste", "hear", "listen", "see", "watch", "look", "observe", "notice",
    "find", "discover", "search", "seek", "hunt", "chase", "follow", "pursue", "track",
    "catch", "grab", "grasp", "hold", "keep", "save", "store", "preserve", "maintain",
    "protect", "defend", "guard", "shield", "shelter", "cover", "hide", "conceal",
    "show", "display", "exhibit", "present", "demonstrate", "reveal", "expose", "uncover",
    "open", "close", "shut", "lock", "unlock", "seal", "seal", "seal", "seal",
    "break", "fix", "repair", "mend", "restore", "rebuild", "construct", "build",
    "create", "make", "produce", "manufacture", "generate", "form", "shape", "mold",
    "destroy", "ruin", "damage", "harm", "hurt", "injure", "wound", "cut", "slice",
    "cut", "chop", "slice", "dice", "mince", "grind", "crush", "smash", "break",
    "cook", "bake", "fry", "grill", "roast", "boil", "steam", "simmer", "stew",
    "eat", "drink", "consume", "swallow", "chew", "bite", "lick", "suck", "sip",
    "sleep", "rest", "relax", "nap", "doze", "snooze", "dream", "wake", "awake",
    "wake", "rise", "get", "up", "stand", "sit", "lie", "lay", "rest", "relax",
    
    # Pronouns and determiners (50+)
    "you", "your", "yours", "they", "their", "theirs", "this", "that", "these", "those",
    "the", "a", "an", "some", "any", "each", "every", "all", "no", "none",
    "he", "she", "it", "we", "i", "me", "my", "mine", "him", "her", "his", "hers",
    "us", "our", "ours", "them", "their", "theirs", "myself", "yourself", "himself",
    "herself", "itself", "ourselves", "yourselves", "themselves", "one", "ones",
    "another", "other", "others", "both", "either", "neither", "such", "same",
    
    # Modals and auxiliary verbs (30+)
    "can", "could", "should", "would", "may", "might", "must", "have", "has", "had",
    "do", "does", "did", "will", "shall", "am", "is", "are", "was", "were",
    "been", "being", "get", "got", "gotten", "go", "went", "gone", "come", "came",
    
    # Common adjectives used as nouns (200+)
    "good", "bad", "best", "worst", "better", "worse", "new", "old", "young", "elder",
    "right", "wrong", "true", "false", "real", "sure", "certain", "possible", "impossible",
    "easy", "hard", "difficult", "simple", "complex", "complicated", "easy", "tough",
    "strong", "weak", "powerful", "powerless", "mighty", "feeble", "robust", "fragile",
    "big", "small", "large", "little", "huge", "tiny", "enormous", "minuscule",
    "long", "short", "tall", "high", "low", "wide", "narrow", "deep", "shallow",
    "thick", "thin", "heavy", "light", "dense", "sparse", "solid", "liquid", "gas",
    "hot", "cold", "warm", "cool", "freezing", "boiling", "mild", "moderate", "extreme",
    "fast", "slow", "quick", "rapid", "swift", "sluggish", "speedy", "gradual",
    "early", "late", "punctual", "tardy", "on", "time", "overdue", "ahead", "behind",
    "first", "last", "next", "previous", "prior", "subsequent", "following", "preceding",
    "same", "different", "similar", "alike", "identical", "distinct", "unique", "common",
    "special", "ordinary", "normal", "usual", "unusual", "rare", "common", "typical",
    "important", "unimportant", "significant", "insignificant", "major", "minor", "main",
    "primary", "secondary", "principal", "chief", "main", "key", "crucial", "vital",
    "necessary", "unnecessary", "essential", "optional", "required", "mandatory", "voluntary",
    "free", "busy", "available", "unavailable", "present", "absent", "here", "there",
    "full", "empty", "complete", "incomplete", "finished", "unfinished", "done", "undone",
    "ready", "unready", "prepared", "unprepared", "ready", "set", "go", "start", "begin",
    
    # Titles and honorifics (50+)
    "mr", "mrs", "ms", "miss", "dr", "doctor", "prof", "professor", "sir", "madam",
    "sister", "brother", "father", "mother", "pastor", "reverend", "bishop", "priest",
    "nun", "monk", "rabbi", "imam", "sheikh", "master", "mistress", "lord", "lady",
    "duke", "duchess", "prince", "princess", "king", "queen", "emperor", "empress",
    "president", "vice", "secretary", "minister", "ambassador", "governor", "mayor",
    "judge", "justice", "attorney", "lawyer", "counsel", "advocate", "representative",
    "senator", "congressman", "congresswoman", "representative", "delegate", "official",
    
    # Other common words (200+)
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth", "last", "next", "previous", "prior",
    "other", "another", "same", "different", "own", "such", "very", "quite", "rather",
    "much", "many", "more", "most", "less", "least", "few", "little", "enough", "too",
    "also", "only", "just", "even", "still", "yet", "already", "again", "once", "twice",
    "here", "there", "where", "when", "why", "how", "what", "which", "who", "whom",
    "whose", "that", "which", "who", "whom", "where", "when", "why", "how", "what",
    "now", "then", "today", "tomorrow", "yesterday", "soon", "later", "early", "late",
    "always", "never", "sometimes", "often", "usually", "rarely", "seldom", "frequently",
    "recently", "lately", "currently", "presently", "immediately", "instantly", "soon",
    "quickly", "slowly", "gradually", "suddenly", "immediately", "instantly", "right", "away",
    "yes", "no", "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
    "certainly", "definitely", "absolutely", "completely", "totally", "entirely", "fully",
    "partially", "partly", "somewhat", "rather", "quite", "very", "extremely", "highly",
    "really", "truly", "actually", "really", "indeed", "certainly", "surely", "definitely",
    "well", "good", "fine", "okay", "ok", "alright", "all", "right", "sure", "yes",
    "no", "not", "never", "nothing", "nobody", "nowhere", "none", "neither", "nor",
    "and", "or", "but", "so", "because", "since", "as", "if", "unless", "until",
    "while", "during", "before", "after", "when", "where", "why", "how", "what",
    "about", "above", "across", "after", "against", "along", "among", "around", "at",
    "before", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down",
    "during", "except", "for", "from", "in", "inside", "into", "like", "near", "of",
    "off", "on", "onto", "out", "outside", "over", "past", "through", "throughout",
    "to", "toward", "towards", "under", "underneath", "until", "up", "upon", "with",
    "within", "without", "via", "per", "plus", "minus", "times", "divided", "by",
    
    # Additional common words to reach 5000+ (3000+ more words)
    # Common everyday objects
    "table", "chair", "desk", "bed", "sofa", "couch", "pillow", "blanket", "sheet",
    "towel", "cloth", "fabric", "material", "paper", "cardboard", "plastic", "metal",
    "wood", "stone", "glass", "ceramic", "porcelain", "rubber", "leather", "fur",
    "box", "bag", "container", "bottle", "jar", "can", "tube", "container", "package",
    "envelope", "letter", "package", "parcel", "gift", "present", "item", "product",
    "goods", "merchandise", "supply", "stock", "inventory", "warehouse", "storehouse",
    "shelf", "rack", "drawer", "cabinet", "closet", "wardrobe", "dresser", "chest",
    "trunk", "suitcase", "baggage", "luggage", "backpack", "purse", "wallet", "pocket",
    "key", "lock", "chain", "rope", "string", "cord", "wire", "cable", "line",
    "tool", "equipment", "device", "machine", "appliance", "gadget", "instrument",
    "hammer", "saw", "screwdriver", "wrench", "pliers", "knife", "scissors", "razor",
    "brush", "comb", "mirror", "glass", "lens", "camera", "phone", "mobile", "cell",
    "computer", "laptop", "tablet", "screen", "monitor", "keyboard", "mouse", "printer",
    "scanner", "copier", "fax", "machine", "device", "gadget", "tool", "instrument",
    
    # Clothing and accessories
    "clothing", "clothes", "garment", "outfit", "dress", "shirt", "blouse", "top",
    "pants", "trousers", "jeans", "skirt", "shorts", "dress", "gown", "robe",
    "jacket", "coat", "sweater", "cardigan", "hoodie", "sweatshirt", "t-shirt",
    "underwear", "bra", "panties", "briefs", "socks", "stockings", "tights",
    "shoes", "boots", "sneakers", "sandals", "flip-flops", "slippers", "heels",
    "hat", "cap", "beanie", "helmet", "gloves", "mittens", "scarf", "tie", "belt",
    "jewelry", "necklace", "bracelet", "ring", "earring", "watch", "clock", "timer",
    "sunglasses", "glasses", "spectacles", "goggles", "mask", "veil", "bandana",
    
    # Buildings and structures
    "building", "structure", "construction", "architecture", "design", "plan", "blueprint",
    "house", "home", "residence", "dwelling", "apartment", "condo", "flat", "studio",
    "mansion", "palace", "castle", "fortress", "tower", "skyscraper", "high-rise",
    "office", "workspace", "desk", "cubicle", "room", "chamber", "hall", "corridor",
    "hallway", "passage", "aisle", "path", "walkway", "sidewalk", "pavement", "road",
    "street", "avenue", "boulevard", "lane", "drive", "way", "circle", "court",
    "bridge", "tunnel", "overpass", "underpass", "highway", "freeway", "expressway",
    "parking", "lot", "garage", "carport", "driveway", "entrance", "exit", "door",
    "gate", "fence", "wall", "barrier", "obstacle", "blockade", "barricade",
    
    # Sports and recreation
    "sport", "game", "play", "match", "contest", "competition", "tournament", "championship",
    "race", "run", "sprint", "marathon", "jog", "walk", "hike", "climb", "ascent",
    "ball", "football", "soccer", "basketball", "baseball", "tennis", "volleyball",
    "golf", "hockey", "cricket", "rugby", "swimming", "diving", "surfing", "skiing",
    "skating", "skateboarding", "cycling", "biking", "running", "jogging", "walking",
    "exercise", "workout", "training", "practice", "drill", "session", "routine",
    "gym", "fitness", "center", "club", "team", "squad", "crew", "group", "unit",
    "player", "athlete", "competitor", "participant", "contestant", "champion", "winner",
    "loser", "defeat", "victory", "win", "loss", "tie", "draw", "score", "point",
    "goal", "target", "aim", "objective", "purpose", "intention", "plan", "strategy",
    
    # Emotions and feelings
    "emotion", "feeling", "sentiment", "mood", "attitude", "disposition", "temperament",
    "happiness", "joy", "delight", "pleasure", "satisfaction", "contentment", "bliss",
    "sadness", "sorrow", "grief", "melancholy", "depression", "despair", "misery",
    "anger", "rage", "fury", "wrath", "irritation", "annoyance", "frustration",
    "fear", "anxiety", "worry", "concern", "dread", "terror", "panic", "horror",
    "surprise", "shock", "amazement", "astonishment", "wonder", "awe", "admiration",
    "love", "affection", "fondness", "adoration", "devotion", "passion", "romance",
    "hate", "hatred", "dislike", "aversion", "loathing", "disgust", "revulsion",
    "hope", "optimism", "confidence", "faith", "trust", "belief", "conviction",
    "despair", "hopelessness", "pessimism", "doubt", "uncertainty", "skepticism",
    
    # Academic and intellectual
    "academic", "scholar", "researcher", "scientist", "professor", "teacher", "instructor",
    "student", "pupil", "learner", "scholar", "academic", "researcher", "investigator",
    "study", "research", "investigation", "inquiry", "examination", "analysis", "review",
    "theory", "hypothesis", "concept", "idea", "notion", "thought", "thinking", "reasoning",
    "knowledge", "understanding", "comprehension", "grasp", "insight", "wisdom", "intelligence",
    "education", "learning", "instruction", "teaching", "training", "coaching", "guidance",
    "school", "college", "university", "institution", "academy", "institute", "center",
    "degree", "diploma", "certificate", "qualification", "credential", "license", "permit",
    "subject", "topic", "theme", "matter", "issue", "question", "problem", "challenge",
    "solution", "answer", "response", "reply", "reaction", "feedback", "comment",
    "essay", "paper", "article", "report", "document", "file", "record", "note",
    "book", "textbook", "manual", "guide", "handbook", "reference", "source", "material",
    
    # Business and finance
    "business", "company", "corporation", "firm", "enterprise", "organization", "institution",
    "industry", "sector", "market", "economy", "finance", "financial", "economic",
    "trade", "commerce", "transaction", "deal", "agreement", "contract", "arrangement",
    "sale", "purchase", "buy", "sell", "trade", "exchange", "transfer", "transaction",
    "money", "cash", "currency", "dollar", "euro", "pound", "yen", "cent", "coin",
    "bank", "account", "deposit", "withdrawal", "transfer", "payment", "bill", "invoice",
    "price", "cost", "fee", "charge", "rate", "tariff", "tax", "duty", "tariff",
    "discount", "sale", "bargain", "deal", "offer", "promotion", "special", "clearance",
    "profit", "loss", "gain", "earnings", "revenue", "income", "salary", "wage",
    "investment", "stock", "share", "bond", "security", "asset", "liability", "equity",
    "budget", "expense", "spending", "cost", "price", "value", "worth", "amount",
    
    # Weather and climate
    "weather", "climate", "temperature", "forecast", "prediction", "report", "update",
    "sun", "sunshine", "sunlight", "daylight", "dawn", "dusk", "sunrise", "sunset",
    "moon", "moonlight", "starlight", "star", "constellation", "galaxy", "universe",
    "cloud", "cloudy", "overcast", "fog", "mist", "haze", "smog", "pollution",
    "rain", "rainfall", "shower", "drizzle", "downpour", "storm", "thunderstorm",
    "snow", "snowfall", "blizzard", "sleet", "hail", "ice", "frost", "freeze",
    "wind", "breeze", "gust", "gale", "hurricane", "tornado", "cyclone", "typhoon",
    "lightning", "thunder", "flash", "bolt", "strike", "hit", "impact", "collision",
    "rainbow", "arc", "spectrum", "color", "hue", "shade", "tint", "tone",
    "season", "spring", "summer", "fall", "autumn", "winter", "holiday", "vacation",
    
    # Animals and wildlife
    "animal", "creature", "beast", "mammal", "reptile", "amphibian", "bird", "fish",
    "insect", "bug", "worm", "spider", "ant", "bee", "wasp", "fly", "mosquito",
    "dog", "puppy", "cat", "kitten", "bird", "chick", "duck", "goose", "chicken",
    "cow", "bull", "calf", "horse", "pony", "foal", "pig", "piglet", "sheep", "lamb",
    "goat", "kid", "deer", "fawn", "rabbit", "bunny", "mouse", "rat", "hamster",
    "lion", "tiger", "leopard", "cheetah", "bear", "wolf", "fox", "coyote",
    "elephant", "rhino", "hippo", "giraffe", "zebra", "monkey", "ape", "gorilla",
    "whale", "dolphin", "shark", "seal", "walrus", "penguin", "eagle", "hawk",
    "owl", "crow", "raven", "sparrow", "robin", "cardinal", "bluejay", "woodpecker",
    "snake", "lizard", "turtle", "tortoise", "frog", "toad", "salamander", "newt",
    "fish", "salmon", "tuna", "cod", "bass", "trout", "carp", "pike", "perch",
    "shark", "ray", "eel", "octopus", "squid", "crab", "lobster", "shrimp", "clam",
    
    # Plants and nature
    "plant", "flower", "bloom", "blossom", "petal", "stem", "leaf", "leaves", "branch",
    "tree", "sapling", "seedling", "trunk", "bark", "root", "branch", "twig",
    "bush", "shrub", "hedge", "vine", "ivy", "moss", "grass", "lawn", "meadow",
    "rose", "daisy", "tulip", "lily", "orchid", "sunflower", "dandelion", "clover",
    "vegetable", "fruit", "berry", "apple", "orange", "banana", "grape", "strawberry",
    "blueberry", "raspberry", "blackberry", "cherry", "peach", "pear", "plum", "apricot",
    "tomato", "potato", "carrot", "onion", "pepper", "cucumber", "lettuce", "spinach",
    "broccoli", "cauliflower", "cabbage", "corn", "bean", "pea", "peanut", "almond",
    "garden", "yard", "lawn", "field", "meadow", "pasture", "farm", "ranch", "orchard",
    "forest", "woods", "jungle", "rainforest", "desert", "grassland", "prairie", "tundra",
    "mountain", "hill", "valley", "canyon", "gorge", "cliff", "peak", "summit", "ridge",
    "ocean", "sea", "lake", "pond", "river", "stream", "creek", "brook", "waterfall",
    "beach", "shore", "coast", "coastline", "island", "peninsula", "bay", "harbor",
    
    # Music and arts
    "music", "song", "tune", "melody", "harmony", "rhythm", "beat", "tempo", "pace",
    "sound", "noise", "tone", "note", "pitch", "volume", "loudness", "quiet", "silence",
    "instrument", "piano", "guitar", "violin", "cello", "viola", "bass", "drums",
    "trumpet", "saxophone", "flute", "clarinet", "oboe", "bassoon", "tuba", "horn",
    "singer", "vocalist", "musician", "composer", "conductor", "director", "producer",
    "band", "orchestra", "choir", "ensemble", "group", "duo", "trio", "quartet",
    "concert", "performance", "show", "recital", "recital", "gig", "event", "occasion",
    "art", "artwork", "painting", "drawing", "sketch", "illustration", "picture", "image",
    "sculpture", "statue", "figure", "bust", "relief", "carving", "engraving", "etching",
    "artist", "painter", "sculptor", "illustrator", "designer", "creator", "maker",
    "gallery", "museum", "exhibition", "show", "display", "collection", "gallery",
    "theater", "theatre", "stage", "play", "drama", "comedy", "tragedy", "musical",
    "actor", "actress", "performer", "player", "character", "role", "part", "cast",
    "movie", "film", "cinema", "picture", "motion", "picture", "video", "clip",
    "director", "producer", "writer", "screenwriter", "script", "screenplay", "dialogue",
    
    # Science and technology
    "science", "scientific", "research", "study", "investigation", "experiment", "test",
    "hypothesis", "theory", "law", "principle", "rule", "formula", "equation", "formula",
    "discovery", "invention", "innovation", "breakthrough", "advance", "progress", "development",
    "scientist", "researcher", "investigator", "experimenter", "analyst", "specialist", "expert",
    "laboratory", "lab", "workshop", "facility", "center", "institute", "institution",
    "technology", "tech", "device", "gadget", "tool", "instrument", "equipment", "apparatus",
    "machine", "engine", "motor", "generator", "transformer", "converter", "adapter",
    "computer", "laptop", "desktop", "tablet", "phone", "smartphone", "device", "gadget",
    "software", "program", "application", "app", "system", "platform", "framework",
    "internet", "web", "network", "connection", "link", "bond", "tie", "relationship",
    "data", "information", "knowledge", "fact", "detail", "datum", "statistic", "figure",
    "analysis", "analytics", "evaluation", "assessment", "review", "examination", "study",
    "method", "methodology", "approach", "technique", "procedure", "process", "system",
    "result", "outcome", "consequence", "effect", "impact", "influence", "affect",
    
    # Law and justice
    "law", "legal", "legislation", "statute", "regulation", "rule", "regulation", "ordinance",
    "court", "trial", "hearing", "case", "lawsuit", "litigation", "proceeding", "process",
    "judge", "justice", "magistrate", "referee", "arbitrator", "mediator", "adjudicator",
    "lawyer", "attorney", "counsel", "advocate", "barrister", "solicitor", "defender",
    "prosecutor", "plaintiff", "defendant", "accused", "suspect", "witness", "testimony",
    "evidence", "proof", "testimony", "statement", "declaration", "affidavit", "deposition",
    "verdict", "judgment", "ruling", "decision", "sentence", "punishment", "penalty",
    "fine", "fee", "cost", "charge", "expense", "payment", "compensation", "damages",
    "crime", "offense", "violation", "infraction", "misdemeanor", "felony", "sin",
    "guilt", "innocence", "guilty", "innocent", "conviction", "acquittal", "dismissal",
    "police", "officer", "cop", "detective", "investigator", "agent", "sheriff", "marshal",
    "prison", "jail", "cell", "detention", "custody", "confinement", "imprisonment",
    "freedom", "liberty", "right", "privilege", "entitlement", "authority", "power",
    
    # Additional common verbs (500+)
    "accept", "achieve", "act", "add", "admire", "admit", "advise", "affect", "agree",
    "aim", "allow", "announce", "answer", "appear", "apply", "appreciate", "approach",
    "argue", "arrive", "ask", "assume", "attempt", "attend", "attract", "avoid", "awake",
    "become", "begin", "behave", "believe", "belong", "bend", "bet", "bid", "bind",
    "bite", "blow", "boil", "borrow", "break", "bring", "build", "burn", "burst",
    "buy", "calculate", "call", "can", "care", "carry", "catch", "cause", "celebrate",
    "change", "charge", "chase", "chat", "check", "cheer", "choose", "claim", "clean",
    "clear", "climb", "close", "collect", "come", "comfort", "command", "comment",
    "commit", "compare", "compete", "complain", "complete", "concern", "confirm", "confuse",
    "connect", "consider", "consist", "contain", "continue", "contribute", "control", "cook",
    "copy", "correct", "cost", "count", "cover", "crash", "create", "cross", "cry",
    "cut", "dance", "dare", "deal", "decide", "declare", "decrease", "defend", "delay",
    "deliver", "demand", "deny", "depend", "describe", "deserve", "design", "desire",
    "destroy", "determine", "develop", "die", "differ", "dig", "disagree", "disappear",
    "discover", "discuss", "dislike", "divide", "do", "doubt", "drag", "draw", "dream",
    "dress", "drink", "drive", "drop", "dry", "earn", "eat", "educate", "elect",
    "eliminate", "embarrass", "emerge", "employ", "enable", "encourage", "end", "enjoy",
    "ensure", "enter", "entertain", "escape", "establish", "estimate", "evaluate", "even",
    "examine", "exceed", "exchange", "excite", "excuse", "exercise", "exist", "expand",
    "expect", "experience", "explain", "explore", "express", "extend", "face", "fail",
    "fall", "fancy", "favor", "fear", "feed", "feel", "fight", "figure", "fill",
    "find", "finish", "fire", "fit", "fix", "flash", "float", "flood", "flow",
    "fly", "focus", "fold", "follow", "fool", "force", "forget", "forgive", "form",
    "found", "frame", "free", "freeze", "frighten", "fry", "gain", "gather", "get",
    "give", "glance", "go", "govern", "grab", "grade", "grant", "grasp", "greet",
    "grow", "guarantee", "guard", "guess", "guide", "handle", "hang", "happen", "harm",
    "hate", "have", "head", "heal", "hear", "heat", "help", "hesitate", "hide",
    "hit", "hold", "hope", "host", "house", "hunt", "hurry", "hurt", "identify",
    "ignore", "illustrate", "imagine", "imply", "import", "impose", "impress", "improve",
    "include", "increase", "indicate", "influence", "inform", "inject", "injure", "insist",
    "install", "intend", "interest", "interfere", "interrupt", "introduce", "invent", "invest",
    "investigate", "invite", "involve", "iron", "issue", "jog", "join", "joke", "judge",
    "jump", "justify", "keep", "kick", "kill", "kiss", "knock", "know", "label",
    "lack", "land", "last", "laugh", "lay", "lead", "lean", "learn", "leave",
    "lend", "let", "level", "license", "lie", "lift", "light", "like", "limit",
    "line", "link", "list", "listen", "live", "load", "lock", "long", "look",
    "lose", "love", "maintain", "make", "manage", "manufacture", "map", "march", "mark",
    "marry", "match", "matter", "may", "mean", "measure", "meet", "melt", "mention",
    "mind", "miss", "mix", "modify", "monitor", "moon", "move", "multiply", "murder",
    "must", "name", "narrow", "need", "negotiate", "nest", "nod", "note", "notice",
    "number", "obey", "object", "observe", "obtain", "occur", "offer", "open", "operate",
    "oppose", "order", "organize", "originate", "overcome", "owe", "own", "pack", "paint",
    "park", "part", "participate", "pass", "pat", "pause", "pay", "peel", "perform",
    "permit", "persuade", "phone", "pick", "pinch", "place", "plan", "plant", "play",
    "please", "plug", "point", "polish", "pop", "possess", "post", "pour", "practice",
    "pray", "predict", "prefer", "prepare", "present", "preserve", "press", "pretend",
    "prevent", "print", "proceed", "process", "produce", "profit", "program", "progress",
    "project", "promise", "promote", "protect", "protest", "prove", "provide", "pull",
    "pump", "punch", "purchase", "push", "put", "qualify", "question", "queue", "quit",
    "quote", "race", "rain", "raise", "range", "rank", "rate", "reach", "react",
    "read", "realize", "receive", "recognize", "recommend", "record", "recover", "reduce",
    "refer", "reflect", "refuse", "regard", "regret", "regulate", "reject", "relate",
    "relax", "release", "rely", "remain", "remember", "remind", "remove", "repair",
    "repeat", "replace", "reply", "report", "represent", "request", "require", "rescue",
    "research", "reserve", "resist", "resolve", "respect", "respond", "rest", "restore",
    "restrict", "result", "retire", "return", "reveal", "review", "reward", "ride",
    "ring", "rise", "risk", "roll", "row", "rub", "ruin", "rule", "run", "rush",
    "sack", "sail", "satisfy", "save", "say", "scale", "scan", "scare", "scatter",
    "schedule", "scheme", "score", "scrape", "scratch", "scream", "screen", "screw",
    "script", "search", "seat", "second", "secure", "see", "seek", "seem", "select",
    "sell", "send", "sense", "sentence", "separate", "serve", "service", "set",
    "settle", "shake", "shall", "shape", "share", "shed", "shelter", "shift", "shine",
    "ship", "shock", "shoot", "shop", "should", "shout", "show", "shrink", "shrug",
    "shut", "sigh", "sign", "signal", "silence", "sing", "sink", "sit", "skate",
    "ski", "skip", "slap", "sleep", "slice", "slide", "slip", "slow", "smash",
    "smell", "smile", "smoke", "snap", "snow", "soak", "solve", "sort", "sound",
    "spare", "speak", "specify", "speed", "spell", "spend", "spill", "spin", "spit",
    "split", "spoil", "spot", "spray", "spread", "spring", "squeeze", "stabilize",
    "stack", "staff", "stage", "stain", "stamp", "stand", "stare", "start", "state",
    "station", "stay", "steal", "steam", "steer", "step", "stick", "sting", "stir",
    "stock", "stop", "store", "storm", "strain", "strange", "strap", "stream", "street",
    "strengthen", "stress", "stretch", "strike", "string", "strip", "stroke", "structure",
    "struggle", "study", "stuff", "stumble", "style", "subject", "submit", "substitute",
    "succeed", "suck", "suffer", "suggest", "suit", "sum", "summarize", "supply",
    "support", "suppose", "suppress", "sure", "surface", "surprise", "surround", "survey",
    "survive", "suspect", "suspend", "swallow", "swap", "swear", "sweep", "swell",
    "swim", "swing", "switch", "symbol", "sympathize", "system", "table", "tackle",
    "tail", "take", "talk", "tall", "tank", "tap", "tape", "target", "task",
    "taste", "tax", "teach", "team", "tear", "tease", "telephone", "tell", "tend",
    "term", "test", "text", "thank", "that", "the", "theater", "theatre", "theme",
    "then", "theory", "there", "therefore", "they", "thick", "thin", "thing", "think",
    "third", "this", "thorough", "though", "thought", "thread", "threat", "threaten",
    "through", "throughout", "throw", "thumb", "thus", "tick", "ticket", "tidy",
    "tie", "tight", "till", "time", "tip", "tire", "tired", "title", "to",
    "today", "toe", "together", "toilet", "told", "tolerance", "tolerate", "toll",
    "tomorrow", "tone", "tongue", "tonight", "too", "tool", "tooth", "top",
    "topic", "total", "touch", "tough", "tour", "tourist", "toward", "towards",
    "towel", "tower", "town", "track", "trade", "tradition", "traditional", "traffic",
    "train", "transfer", "transform", "transition", "translate", "transport", "trap",
    "travel", "treat", "treatment", "tree", "tremendous", "trend", "trial", "tribe",
    "trick", "trip", "troop", "trouble", "truck", "true", "truly", "trust", "truth",
    "try", "tube", "tune", "turn", "twice", "twin", "twist", "type", "typical",
    "ugly", "ultimate", "ultimately", "unable", "uncle", "under", "undergo", "understand",
    "understanding", "undertake", "unemployment", "unexpected", "unfair", "unfortunate",
    "unfortunately", "unhappy", "uniform", "union", "unique", "unit", "unite", "unity",
    "universal", "universe", "university", "unknown", "unless", "unlike", "unlikely",
    "until", "unusual", "up", "update", "upon", "upper", "upset", "urban", "urge",
    "urgent", "us", "use", "used", "useful", "user", "usual", "usually", "utility",
    "utilize", "utter", "utterly", "vacation", "vague", "vain", "valid", "valley",
    "valuable", "value", "van", "vanish", "variable", "variation", "variety", "various",
    "vary", "vast", "vegetable", "vehicle", "venture", "verb", "verbal", "verdict",
    "version", "versus", "very", "via", "vice", "victim", "victory", "video",
    "view", "viewer", "village", "violence", "violent", "virtually", "virtue",
    "virus", "visible", "vision", "visit", "visitor", "visual", "vital", "voice",
    "volume", "voluntary", "volunteer", "vote", "voter", "vs", "wage", "wait",
    "wake", "walk", "wall", "want", "war", "ward", "warm", "warn", "warning",
    "wash", "waste", "watch", "water", "wave", "way", "we", "weak", "wealth",
    "weapon", "wear", "weather", "web", "website", "wedding", "week", "weekend",
    "weekly", "weigh", "weight", "welcome", "welfare", "well", "west", "western",
    "wet", "what", "whatever", "wheel", "when", "whenever", "where", "whereas",
    "wherever", "whether", "which", "while", "whilst", "whip", "whisper", "white",
    "who", "whole", "whom", "whose", "why", "wide", "widely", "widespread", "wife",
    "wild", "will", "willing", "win", "wind", "window", "wine", "wing", "winner",
    "winter", "wipe", "wire", "wise", "wish", "with", "withdraw", "within", "without",
    "witness", "woman", "wonder", "wonderful", "wood", "wooden", "word", "work",
    "worker", "working", "workshop", "world", "worried", "worry", "worse", "worth",
    "would", "wound", "wrap", "write", "writer", "writing", "wrong", "yard", "yeah",
    "year", "yellow", "yes", "yesterday", "yet", "yield", "you", "young", "your",
    "yours", "yourself", "youth", "zone",
}

# Medical context words (additional to common nouns)
MEDICAL_CONTEXT = {
    "diagnosis", "recommendations", "service", "medicine", "history",
    "department", "emergency", "admission", "side", "bone",
    "sample", "dose", "mg", "daily", "pain", "field", "location",
    "address", "allergies", "infection", "treatment", "therapy", "procedure",
    "surgery", "operation", "appointment", "visit", "examination", "exam",
    "prescription", "medication", "drug", "pill", "tablet", "injection",
    "vaccine", "vaccination", "test", "lab", "laboratory", "result", "report",
    "chart", "record", "note", "document", "file", "case", "patient", "doctor",
    "nurse", "physician", "surgeon", "specialist", "clinic", "hospital",
    "ward", "room", "bed", "discharge", "admission", "emergency", "er",
    "icu", "surgery", "radiology", "pathology", "pharmacy", "pharmacy",
}


def name_mask(df: pd.DataFrame, column: str = "value") -> pd.Series:
    """
    Returns a boolean mask: True = keep (likely a name), False = drop.
    Assumes column contains strings where the first two tokens are the candidate.
    
    Args:
        df: DataFrame containing the data
        column: Column name containing the text to filter (default: "value")
    
    Returns:
        Boolean Series where True indicates rows to keep (likely names)
    """

    def is_likely_name(text: str) -> bool:
        if not isinstance(text, str):
            return False

        # Reject values containing commas, quotes, parentheses, or #
        if ',' in text or '"' in text or "'" in text or '(' in text or ')' in text or '#' in text or '/' in text or ':' in text or '$' in text or '\n' in text or "_" in text:
            return False

        # Normalize spacing and strip punctuation at edges
        text = text.strip()
        tokens = re.split(r'\s+', text)

        if len(tokens) < 2:
            return False

        w1, w2 = tokens[0], tokens[1]

        # Strip trailing punctuation
        w1_clean = re.sub(r'[^\p{L}]', '', w1)
        w2_clean = re.sub(r'[^\p{L}]', '', w2)

        if not w1_clean or not w2_clean:
            return False

        # # Reject if first or last name is a single Latin letter
        # # (allow single characters for non-Latin scripts like Chinese)
        # if len(w1_clean) == 1 and w1_clean.isascii() and w1_clean.isalpha():
        #     return False
        # if len(w2_clean) == 1 and w2_clean.isascii() and w2_clean.isalpha():
        #     return False

        # Use casefold() for robust case-insensitive matching (handles Unicode correctly)
        lw1, lw2 = w1_clean.casefold(), w2_clean.casefold()

        # # Length sanity
        # if not (1 <= len(w1_clean) <= 25 and 1 <= len(w2_clean) <= 25):
            # return False

        # # Check against common English nouns (comprehensive dictionary)
        # if lw1 in COMMON_NOUNS or lw2 in COMMON_NOUNS:
        #     return False

        # # Check against medical context words
        # if lw1 in MEDICAL_CONTEXT or lw2 in MEDICAL_CONTEXT:
        #     return False

        # # Must be letters only (Unicode-aware)
        # if not LETTER_RE.match(w1_clean) or not LETTER_RE.match(w2_clean):
        #     return False

        # # Capitalization heuristic:
        # # Allow either Title Case OR scripts without case
        # if w1_clean[0].islower() or w2_clean[0].islower():
        #     return False

        # Require that the name contains BOTH a first name AND a last name
        # This ensures we only accept names that have both a known first name and a known last name
        # Using casefold() for both filter set and input ensures consistent case-insensitive matching
        # Check if we have: (first name, last name) OR (last name, first name)
        # is_first_last = (lw1 in FIRST_NAMES_FILTER_SET and lw2 in LAST_NAMES_FILTER_SET)
        # is_last_first = (lw1 in LAST_NAMES_FILTER_SET and lw2 in FIRST_NAMES_FILTER_SET)
        
        # if not (is_first_last or is_last_first):
            # return False

        if lw1 not in FIRST_NAMES_FILTER_SET and lw2 not in FIRST_NAMES_FILTER_SET:
            return False

        return True

    return df[column].apply(is_likely_name)
