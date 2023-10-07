import random

# Sample word lists
nouns = ["cat", "dog", "apple", "banana", "car", "tree", "book", "chair", "sun", "moon"]
verbs = ["runs", "jumps", "eats", "drives", "sleeps", "reads", "talks", "swims", "plays", "writes"]
adjectives = ["happy", "sad", "big", "small", "red", "green", "bright", "dark", "loud", "quiet"]
adverbs = ["quickly", "slowly", "loudly", "quietly", "carefully", "eagerly", "gently", "well", "badly"]
conjunctions = ["and", "but", "or", "so"]
prepositions = ["on", "in", "under", "over", "between", "among", "beside", "near"]
articles = ["the", "a", "an"]
punctuation = [".", "!", "?"]

# New Features
emotions = ["happy", "sad", "excited", "angry", "calm", "surprised", "confused"]
settings = ["forest", "beach", "city", "mountain", "desert", "space"]
tenses = ["past", "present", "future"]
styles = ["poetically", "scientifically", "mysteriously"]
dialogue_tags = ["asked", "replied", "mused", "exclaimed"]
time_periods = ["morning", "afternoon", "evening", "night"]
weather_conditions = ["sunny", "cloudy", "rainy", "stormy"]
transitions = ["Meanwhile", "Nevertheless", "In addition", "As a result"]
genres = ["fantasy", "mystery", "sci-fi", "romance", "historical"]
pronouns = ["he", "she", "they", "it"]
character_names = ["Alice", "Bob", "Eva", "David", "Sophia", "Leo"]

# Lists of animals, plants, and non-living things
animals = ["cat", "dog"]
plants = ["apple", "banana", "tree"]
non_living_things = ["car", "book", "chair", "sun", "moon"]

# Additional Features
feeling_verbs = ["feels", "experiences", "senses", "perceives", "understands"]
time_units = ["minutes", "hours", "days", "weeks", "months"]
distances = ["nearby", "far away", "within sight", "beyond the horizon"]
relationships = ["friend", "enemy", "stranger", "ally"]
colors = ["blue", "yellow", "purple", "orange", "pink"]
sounds = ["whisper", "roar", "chirp", "howl", "murmur"]
actions = ["dances", "paints", "dreams", "whistles", "naps", "meditates", "laughs"]
vehicles = ["bicycle", "bus", "train", "airplane", "boat", "helicopter", "submarine"]
feelings = ["joy", "sadness", "fear", "anger", "surprise", "disgust"]
temperatures = ["hot", "cold", "warm"]
numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

# 2 New Features
locations = ["cave", "castle", "ocean", "valley", "countryside", "cottage"]
textures = ["smooth", "rough", "soft", "hard", "velvety"]

# Generate sentences
def generate_sentence():
    subject = random.choice(nouns)
    verb = random.choice(verbs)
    adjective = random.choice(adjectives)
    adverb = random.choice(adverbs)
    conjunction = random.choice(conjunctions)
    preposition = random.choice(prepositions)
    article = random.choice(articles)
    sentence_punctuation = random.choice(punctuation)
    emotion = random.choice(emotions)
    setting = random.choice(settings)
    tense = random.choice(tenses)
    style = random.choice(styles)
    dialogue_tag = random.choice(dialogue_tags)
    time_period = random.choice(time_periods)
    weather = random.choice(weather_conditions)
    transition = random.choice(transitions)
    genre = random.choice(genres)
    pronoun = random.choice(pronouns)
    character_name = random.choice(character_names)
    
    # Actions based on nouns
    if subject in animals:
        if verb in ["reads", "talks"]:
            verb = random.choice(["runs", "jumps", "swims", "plays"])
    elif subject in plants:
        if verb in ["runs", "jumps", "eats", "swims", "plays"]:
            verb = random.choice(["grows", "blooms", "photosynthesizes"])
    elif subject in non_living_things:
        if verb in ["eats", "swims"]:
            verb = random.choice(["drives", "talks", "writes"])
    
    # Additional Features
    feeling_verb = random.choice(feeling_verbs)
    time_unit = random.choice(time_units)
    distance = random.choice(distances)
    relationship = random.choice(relationships)
    color = random.choice(colors)
    sound = random.choice(sounds)
    action = random.choice(actions)
    vehicle = random.choice(vehicles)
    feeling = random.choice(feelings)
    temperature = random.choice(temperatures)
    number = random.choice(numbers)
    
    # 2 New Features
    location = random.choice(locations)
    texture = random.choice(textures)
    
    sentence = f"{article.capitalize()} {adjective} {subject} {verb} {adverb}{sentence_punctuation}"
    
    if random.random() > 0.5:
        object_ = random.choice(nouns)
        sentence += f" {preposition} {article} {object_}"
    
    if random.random() > 0.7:
        sentence += f" {conjunction} {generate_sentence()}"
    
    if random.random() > 0.6:
        sentence = f"In a {mood} mood, {sentence}"
    
    if random.random() > 0.6:
        sentence = f"The {subject} {talking_verb} {adverb}."
    
    if random.random() > 0.6:
        sentence = f"The {quantity} {subject} were {color} and {texture}."
    
    if random.random() > 0.6:
        sentence = f"The air was filled with the scent of {scent}. {sentence}"
    
    if random.random() > 0.6:
        sentence = f"The {location} was {adjective}, {sentence}"
    
    return sentence

# Generate and print paragraphs
def generate_paragraph():
    paragraph = ""
    num_sentences = random.randint(2, 5)  # Random number of sentences per paragraph
    for _ in range(num_sentences):
        paragraph += generate_sentence() + " "
    return paragraph.strip()

def generate_story():
    story = ""
    num_paragraphs = random.randint(3, 6)  # Random number of paragraphs per story
    for _ in range(num_paragraphs):
        story += generate_paragraph() + "\n\n"
    return story.strip()

print(generate_story())
