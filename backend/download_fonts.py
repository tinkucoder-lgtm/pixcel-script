import requests
import os

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

FONTS = {
    "dancing-script": "https://github.com/google/fonts/raw/refs/heads/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf",
    "pacifico": "https://github.com/google/fonts/raw/refs/heads/main/ofl/pacifico/Pacifico-Regular.ttf",
    "lobster": "https://github.com/google/fonts/raw/refs/heads/main/ofl/lobster/Lobster-Regular.ttf",
    "caveat": "https://github.com/google/fonts/raw/refs/heads/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "playfair-display": "https://github.com/google/fonts/raw/refs/heads/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "lora": "https://github.com/google/fonts/raw/refs/heads/main/ofl/lora/Lora%5Bwght%5D.ttf",
    "abril-fatface": "https://github.com/google/fonts/raw/refs/heads/main/ofl/abrilfatface/AbrilFatface-Regular.ttf",
    "bebas-neue": "https://github.com/google/fonts/raw/refs/heads/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "oswald": "https://github.com/google/fonts/raw/refs/heads/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "raleway": "https://github.com/google/fonts/raw/refs/heads/main/ofl/raleway/Raleway%5Bwght%5D.ttf",
    "satisfy": "https://github.com/google/fonts/raw/refs/heads/main/apache/satisfy/Satisfy-Regular.ttf",
    "kalam": "https://github.com/google/fonts/raw/refs/heads/main/ofl/kalam/Kalam-Regular.ttf",
    "patrick-hand": "https://github.com/google/fonts/raw/refs/heads/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    "indie-flower": "https://github.com/google/fonts/raw/refs/heads/main/ofl/indieflower/IndieFlower-Regular.ttf",
    "shadows-into-light": "https://github.com/google/fonts/raw/refs/heads/main/ofl/shadowsintolight/ShadowsIntoLight.ttf",
    "sacramento": "https://github.com/google/fonts/raw/refs/heads/main/ofl/sacramento/Sacramento-Regular.ttf",
    "great-vibes": "https://github.com/google/fonts/raw/refs/heads/main/ofl/greatvibes/GreatVibes-Regular.ttf",
    "pinyon-script": "https://github.com/google/fonts/raw/refs/heads/main/ofl/pinyonscript/PinyonScript-Regular.ttf",
    "righteous": "https://github.com/google/fonts/raw/refs/heads/main/ofl/righteous/Righteous-Regular.ttf",
    "alfa-slab-one": "https://github.com/google/fonts/raw/refs/heads/main/ofl/alfaslabone/AlfaSlabOne-Regular.ttf",
    "fjalla-one": "https://github.com/google/fonts/raw/refs/heads/main/ofl/fjallaone/FjallaOne-Regular.ttf",
    "yanone-kaffeesatz": "https://github.com/google/fonts/raw/refs/heads/main/ofl/yanonekaffeesatz/YanoneKaffeesatz%5Bwght%5D.ttf",
    "josefin-sans": "https://github.com/google/fonts/raw/refs/heads/main/ofl/josefinsans/JosefinSans%5Bwght%5D.ttf",
    "nunito": "https://github.com/google/fonts/raw/refs/heads/main/ofl/nunito/Nunito%5Bwght%5D.ttf",
    "quicksand": "https://github.com/google/fonts/raw/refs/heads/main/ofl/quicksand/Quicksand%5Bwght%5D.ttf",
    "comfortaa": "https://github.com/google/fonts/raw/refs/heads/main/ofl/comfortaa/Comfortaa%5Bwght%5D.ttf",
    "philosopher": "https://github.com/google/fonts/raw/refs/heads/main/ofl/philosopher/Philosopher-Regular.ttf",
    "cormorant-garamond": "https://github.com/google/fonts/raw/refs/heads/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
    "libre-baskerville": "https://github.com/google/fonts/raw/refs/heads/main/ofl/librebaskerville/LibreBaskerville%5Bwght%5D.ttf",
    "crimson-text": "https://github.com/google/fonts/raw/refs/heads/main/ofl/crimsontext/CrimsonText-Regular.ttf",
    "special-elite": "https://github.com/google/fonts/raw/refs/heads/main/apache/specialelite/SpecialElite-Regular.ttf",
    "courier-prime": "https://github.com/google/fonts/raw/refs/heads/main/ofl/courierprime/CourierPrime-Regular.ttf",
    "permanent-marker": "https://github.com/google/fonts/raw/refs/heads/main/apache/permanentmarker/PermanentMarker-Regular.ttf",
    "rock-salt": "https://github.com/google/fonts/raw/refs/heads/main/apache/rocksalt/RockSalt-Regular.ttf",
}

for name, url in FONTS.items():
    path = os.path.join(FONTS_DIR, f"{name}.ttf")
    if os.path.exists(path):
        print(f"Already exists: {name}")
        continue
    print(f"Downloading {name}...")
    try:
        r = requests.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Done: {name}")
    except Exception as e:
        print(f"Failed {name}: {e}")

print("All fonts downloaded.")
