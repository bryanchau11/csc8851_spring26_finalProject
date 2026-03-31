import os
import json
import urllib.parse
import urllib.request

# Config
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
USDA_API_KEY = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_CACHE_FILE = os.path.join(MODELS_DIR, 'usda_cache.json')

# food_name -> (kcal/g, fat/g, protein/g, carb/g, serving_g)
_usda_cache: dict = {}

# Typical served weights (g) used when USDA serving sizes look too small.
_RESTAURANT_PORTIONS = {
    'pasta': 350, 'spaghetti': 350, 'carbonara': 350, 'linguine': 350,
    'fettuccine': 350, 'lasagna': 300, 'gnocchi': 250, 'ravioli': 280,
    'ramen': 450, 'pho': 450, 'noodle': 350, 'pad thai': 300,
    'pizza': 250, 'burger': 250, 'hamburger': 250, 'hot dog': 150,
    'steak': 220, 'pork chop': 200, 'chicken': 220, 'salmon': 180,
    'rice': 300, 'fried rice': 300, 'bibimbap': 350, 'paella': 300,
    'curry': 350, 'soup': 350, 'stew': 350, 'chowder': 300,
    'sandwich': 220, 'club sandwich': 220, 'grilled cheese': 200, 'burrito': 250,
    'sushi': 250, 'sashimi': 150, 'tacos': 220, 'gyoza': 180,
    'pancakes': 180, 'waffles': 180, 'french toast': 180,
    'omelette': 200, 'eggs benedict': 220, 'huevos': 250,
    'cake': 120, 'cheesecake': 120, 'tiramisu': 150, 'ice cream': 150,
    'salad': 250, 'caesar salad': 250, 'greek salad': 200,
    'nachos': 200, 'chicken wings': 250, 'spring rolls': 150,
    'dumplings': 200, 'peking duck': 250,
    'apple': 182, 'banana': 118, 'orange': 131, 'pear': 178,
    'peach': 150, 'mango': 200, 'pineapple': 165, 'grape': 150,
    'strawberry': 150, 'watermelon': 280, 'grapes': 150,
}


def _food_name_to_key(name: str) -> str:
    return name.lower().replace('_', ' ').replace('-', ' ').strip()


def _restaurant_serving_g(food_name: str) -> float:
    lower = food_name.lower()
    for kw, g in _RESTAURANT_PORTIONS.items():
        if kw in lower:
            return float(g)
    return 280.0


def _load_usda_cache():
    global _usda_cache
    try:
        if os.path.isfile(USDA_CACHE_FILE):
            with open(USDA_CACHE_FILE) as f:
                raw = json.load(f)
            _usda_cache = {k: tuple(v) for k, v in raw.items()}
            print(f'✓ USDA cache loaded ({len(_usda_cache)} entries) from {USDA_CACHE_FILE}')
    except Exception:
        _usda_cache = {}


def _save_usda_cache():
    os.makedirs(os.path.dirname(USDA_CACHE_FILE), exist_ok=True)
    with open(USDA_CACHE_FILE, 'w') as f:
        json.dump({k: list(v) for k, v in _usda_cache.items()}, f, indent=2)


def _lookup_usda(food_name: str):
    """Return (kcal/g, fat/g, protein/g, carb/g, serving_g) or None."""

    cache_key = _food_name_to_key(food_name)

    _PROTEIN_TOKENS = {
        'pork', 'beef', 'steak', 'chicken', 'turkey', 'duck', 'lamb', 'mutton',
        'goat', 'ham', 'bacon', 'sausage',
        'salmon', 'tuna', 'fish', 'shrimp', 'prawn', 'crab', 'lobster',
        'calamari', 'squid', 'octopus', 'mussel', 'mussels', 'oyster', 'oysters',
        'clam', 'clams', 'scallop', 'scallops',
        'egg', 'eggs',
    }
    _STARCH_TOKENS = {
        'rice', 'noodle', 'noodles', 'pasta', 'spaghetti', 'macaroni', 'bread',
        'potato', 'potatoes', 'fries', 'french fries', 'ramen',
    }

    def _looks_like_protein(name: str) -> bool:
        n = f' {name} '
        return any(f' {t} ' in n for t in _PROTEIN_TOKENS) or any(t in name for t in _PROTEIN_TOKENS)

    def _looks_like_starch(name: str) -> bool:
        n = f' {name} '
        return any(f' {t} ' in n for t in _STARCH_TOKENS) or any(t in name for t in _STARCH_TOKENS)

    def _cache_suspicious(name: str, kcal_g: float) -> bool:
        try:
            if _looks_like_protein(name) and float(kcal_g) < 1.6:
                return True
            if _looks_like_starch(name) and float(kcal_g) > 2.6:
                return True
            return False
        except Exception:
            return False

    if cache_key in _usda_cache:
        entry = _usda_cache[cache_key]
        try:
            if isinstance(entry, (tuple, list)) and len(entry) >= 1 and _cache_suspicious(cache_key, float(entry[0])):
                pass
            else:
                return entry
        except Exception:
            return entry

    if not USDA_API_KEY:
        return None

    base_words = cache_key.split()
    is_short_query = len(base_words) <= 2
    prefer_as_eaten = is_short_query and (_looks_like_protein(cache_key) or _looks_like_starch(cache_key))

    words = [w for w in base_words if len(w) >= 4]
    words.sort(key=len, reverse=True)

    queries_to_try = []
    if prefer_as_eaten:
        queries_to_try.extend([f'{cache_key} cooked', f'{cache_key} prepared'])
    queries_to_try.append(cache_key)
    queries_to_try.extend([w for w in words if w != cache_key])

    def _fetch_foods(q: str):
        params = urllib.parse.urlencode({'query': q, 'pageSize': 8, 'api_key': USDA_API_KEY})
        url = f'https://api.nal.usda.gov/fdc/v1/foods/search?{params}'
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=7) as resp:
            return json.loads(resp.read()).get('foods', [])

    def _parse_food(food):
        nmap = {n['nutrientName']: float(n.get('value') or 0) for n in food.get('foodNutrients', [])}
        fat_100 = nmap.get('Total lipid (fat)', 0.0)
        prot_100 = nmap.get('Protein', 0.0)
        carb_100 = nmap.get('Carbohydrate, by difference', 0.0)
        kcal_100 = prot_100 * 4.0 + fat_100 * 9.0 + carb_100 * 4.0
        kcal_g = kcal_100 / 100.0
        fat_g = fat_100 / 100.0
        prot_g = prot_100 / 100.0
        carb_g = carb_100 / 100.0
        srv = 0.0
        for m in food.get('finalFoodInputFoods', []):
            gw = m.get('gramWeight')
            if gw and float(gw) > 20:
                srv = float(gw)
                break
        return kcal_g, fat_g, prot_g, carb_g, srv

    KCAL_MIN, KCAL_MAX = 0.5, 4.5

    def _median_candidate(candidates):
        candidates.sort(key=lambda t: t[0])
        return candidates[len(candidates) // 2]

    try:
        full_name_candidates = []
        try:
            for food in _fetch_foods(queries_to_try[0]):
                parsed = _parse_food(food)
                if KCAL_MIN <= parsed[0] <= KCAL_MAX:
                    full_name_candidates.append(parsed)
        except Exception:
            pass

        if full_name_candidates:
            best = _median_candidate(full_name_candidates)
        else:
            kw_candidates = []
            for q in queries_to_try[1:]:
                try:
                    for food in _fetch_foods(q):
                        parsed = _parse_food(food)
                        if KCAL_MIN <= parsed[0] <= KCAL_MAX:
                            kw_candidates.append(parsed)
                except Exception:
                    continue
            if not kw_candidates:
                return None
            best = max(kw_candidates, key=lambda t: t[0])

        kcal_g, fat_g, prot_g, carb_g, srv = best
        if srv < 200:
            srv = _restaurant_serving_g(cache_key)

        entry = (kcal_g, fat_g, prot_g, carb_g, srv)
        _usda_cache[cache_key] = entry
        _save_usda_cache()
        return entry
    except Exception as e:
        print(f'USDA lookup failed for "{cache_key}": {e}')
        return None
