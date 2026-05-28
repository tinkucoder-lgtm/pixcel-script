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
        "headline_font": (
            "an ultra-thin high-contrast serif with dramatic hairline strokes "
            "and pronounced thick-to-thin transitions, like a Vogue or "
            "Harper's Bazaar magazine cover"
        ),
        "body_font": (
            "refined thin-serif lettering with high contrast strokes, like "
            "Vogue magazine captions"
        ),
    },
    "modern-minimal": {
        "display_name": "Modern Minimal",
        "headline_font": (
            "a bold geometric sans-serif with perfectly even stroke widths, "
            "circular o's, and confident proportions, like a contemporary "
            "museum identity or an Apple product launch"
        ),
        "body_font": (
            "clean geometric sans-serif with even stroke width and generous "
            "letter-spacing, like Apple product pages"
        ),
    },
    "warm-handcrafted": {
        "display_name": "Warm Handcrafted",
        "headline_font": (
            "soft brush-lettered script with organic imperfect curves, "
            "varying stroke widths, and gentle ink-like character, like "
            "artisan craft packaging or a bakery wordmark"
        ),
        "body_font": (
            "rounded casual hand-drawn lettering with soft edges and slightly "
            "uneven baselines, looks hand-lettered not typed"
        ),
    },
    "classic-serif": {
        "display_name": "Classic Serif",
        "headline_font": (
            "a stately old-style serif with bracketed serifs, modest stroke "
            "contrast, and traditional Roman proportions, like a hardcover "
            "book title or a newspaper masthead"
        ),
        "body_font": (
            "traditional Roman serif letterforms with moderate contrast, like "
            "a newspaper or book"
        ),
    },
    "bold-impact": {
        "display_name": "Bold Impact",
        "headline_font": (
            "an extra-heavy condensed sans-serif with chunky uppercase "
            "letters, minimal letter-spacing, and assertive black weight, "
            "like a movie poster or sports ad"
        ),
        "body_font": (
            "thick condensed sans-serif with tight spacing and heavy weight, "
            "like a sports poster"
        ),
    },
    "vintage-retro": {
        "display_name": "Vintage Retro",
        "headline_font": (
            "a chunky slab-serif with squared-off blocky serifs and a "
            "slightly distressed letterpress texture, like a 1950s diner "
            "sign or vintage travel poster"
        ),
        "body_font": (
            "weathered slab-serif with a slightly worn texture, like old "
            "letterpress printing"
        ),
    },
    "luxury-refined": {
        "display_name": "Luxury Refined",
        "headline_font": (
            "an elongated thin serif with extreme letter-spacing, hairline "
            "strokes, and elegant tall proportions, like a luxury fashion "
            "house wordmark"
        ),
        "body_font": (
            "elegant thin sans-serif with wide letter-spacing and small caps "
            "feel, like a high-end fashion brand"
        ),
    },
    "playful-friendly": {
        "display_name": "Playful Friendly",
        "headline_font": (
            "a rounded display font with bubbly bouncy letters, oversized "
            "lowercase, soft friendly curves, and a slightly irregular "
            "baseline, like a children's brand or modern playful startup"
        ),
        "body_font": (
            "bubbly rounded sans-serif with oversized lowercase and bouncy "
            "baseline, like a children's brand"
        ),
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
