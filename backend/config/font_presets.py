"""Font presets + the fixed anti-AI-design block used by every generation call.

Each preset has a `display_name` (for UI labels), a `headline_font` description
written in prompt-language (e.g. "a high-contrast editorial serif, Didot/Bodoni
style"), and a `body_font` description in the same style. The descriptions are
substituted directly into the generation prompt — they're sentence fragments,
not class names — so the model interprets them.

The ANTI_AI_DESIGN constant is the fixed art-direction block appended to every
generation. Encodes what makes generic AI-generated design look generic, and
explicitly forbids those patterns. Validated by gemini_generate_test.py;
modifying it changes output for every endpoint call.
"""

FONT_PRESETS = {
    "editorial-elegant": {
        "display_name": "Editorial Elegant",
        "headline_font": "a high-contrast editorial serif, Didot/Bodoni style",
        "body_font": "a clean humanist sans-serif",
    },
    "modern-minimal": {
        "display_name": "Modern Minimal",
        "headline_font": "a bold clean geometric sans-serif",
        "body_font": "a light geometric sans-serif",
    },
    "warm-handcrafted": {
        "display_name": "Warm Handcrafted",
        "headline_font": "a friendly hand-lettered brush script",
        "body_font": "a rounded humanist sans-serif",
    },
    "classic-serif": {
        "display_name": "Classic Serif",
        "headline_font": "a traditional old-style serif, Garamond/Caslon style",
        "body_font": "a complementary old-style serif at text weight",
    },
    "bold-impact": {
        "display_name": "Bold Impact",
        "headline_font": "a heavy condensed sans-serif with strong presence",
        "body_font": "a clean neutral grotesque sans-serif",
    },
    "vintage-retro": {
        "display_name": "Vintage Retro",
        "headline_font": "a retro display slab-serif",
        "body_font": "a vintage-style sans-serif",
    },
    "luxury-refined": {
        "display_name": "Luxury Refined",
        "headline_font": "an elegant thin serif with generous letter-spacing",
        "body_font": "a refined sans-serif, small caps where appropriate",
    },
    "playful-friendly": {
        "display_name": "Playful Friendly",
        "headline_font": "a rounded playful display font",
        "body_font": "a friendly rounded sans-serif",
    },
}


ANTI_AI_DESIGN = (
    "ART DIRECTION — this must look like the work of a senior human designer, "
    "NOT AI-generated. Follow these rules strictly: intentional asymmetric "
    "composition with a single clear focal point (never centered symmetry); "
    "restrained purposeful decoration with NO gratuitous glows, sparkles, or "
    "filler ornaments; a cohesive type system, not a grab-bag of fonts; "
    "generous intentional whitespace and clear visual hierarchy; sophisticated "
    "restrained color, avoid oversaturation and heavy vignettes; clean "
    "real-looking photography or intentional illustration, never AI-merged "
    "hybrid objects; overall premium editorial deliberately-designed feel. "
    "Render ALL text crisply, legibly, and with correct spelling. "
    "Produce the flat design itself, full-bleed filling the entire frame — "
    "NOT a mockup, NOT framed, NOT photographed in a setting, no outer "
    "borders or margins."
)
